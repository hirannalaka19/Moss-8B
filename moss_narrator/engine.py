"""MOSS-TTS-v1.5 (8B) long-form narration engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_utils import (
    apply_edge_fades,
    cut_at_quiet,
    denoise,
    load_audio_mono,
    normalize_loudness,
    soft_limit,
    stitch,
    trim_silence,
)
from .planning import (
    MODEL_ID,
    REFERENCE_MAX_S,
    SAMPLE_RATE,
    NarrationSettings,
    check_gpu,
    duration_score,
    is_duration_plausible,
    language_tag,
    max_new_tokens_for,
)
from .text_chunker import Chunk, chunk_narration

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class NoAudioError(RuntimeError):
    """The model returned no decodable audio for a chunk."""


@dataclass(frozen=True)
class NarrationResult:
    audio: np.ndarray
    sample_rate: int
    chunk_count: int
    warnings: tuple[str, ...]
    complete: bool = True

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def _configure_torch_backends(torch) -> None:
    # Upstream README: the cuDNN SDPA backend is broken for this model.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


class MossNarrator:
    def __init__(self, model, processor, sample_rate: int) -> None:
        self._model = model
        self._processor = processor
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, model_id: str = MODEL_ID) -> "MossNarrator":
        import torch
        from transformers import AutoModel, AutoProcessor

        cuda = torch.cuda.is_available()
        free_bytes, total_bytes = torch.cuda.mem_get_info() if cuda else (0, 0)
        check_gpu(cuda, total_bytes / 1024**3, free_bytes / 1024**3)
        _configure_torch_backends(torch)

        logger.info("Loading processor + audio codec from %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")

        logger.info("Loading %s in bfloat16", model_id)
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        ).to("cuda")
        model.eval()
        sample_rate = int(getattr(processor.model_config, "sampling_rate", SAMPLE_RATE))
        return cls(model, processor, sample_rate)

    def encode_reference(self, wav: np.ndarray):
        """Encode a mono waveform once into audio codes reusable for every chunk."""
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        return self._processor.encode_audios_from_wav([tensor], self.sample_rate)[0]

    def synthesize(self, text: str, reference_codes, language: str | None, settings: NarrationSettings) -> np.ndarray | None:
        """Generate one chunk. Returns None when the model produced no audio."""
        import torch

        reference = [reference_codes] if reference_codes is not None else None
        message = self._processor.build_user_message(text=text, reference=reference, language=language)
        batch = self._processor([[message]], mode="generation")
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_new_tokens=max_new_tokens_for(text),
                audio_temperature=settings.temperature,
                audio_top_p=settings.top_p,
                audio_top_k=settings.top_k,
                audio_repetition_penalty=settings.repetition_penalty,
            )
        messages = self._processor.decode(outputs)
        if not messages or messages[0] is None or not messages[0].audio_codes_list:
            return None
        audio = messages[0].audio_codes_list[0]
        return audio.detach().float().cpu().numpy().reshape(-1)

    def _clean(self, wav: np.ndarray) -> np.ndarray:
        trimmed = trim_silence(wav, self.sample_rate)
        return apply_edge_fades(normalize_loudness(trimmed), self.sample_rate)

    def synthesize_checked(
        self, text: str, reference_codes, language: str | None, settings: NarrationSettings
    ) -> tuple[np.ndarray, str | None]:
        """Retry truncated/runaway generations; keep the closest attempt if all fail."""
        best: np.ndarray | None = None
        best_score = float("inf")
        for attempt in range(settings.max_retries + 1):
            raw = self.synthesize(text, reference_codes, language, settings)
            if raw is None or raw.size == 0:
                logger.warning("Attempt %d produced no audio", attempt + 1)
                continue
            cleaned = self._clean(raw)
            seconds = len(cleaned) / self.sample_rate
            if is_duration_plausible(seconds, text):
                return cleaned, None
            score = duration_score(seconds, text)
            logger.warning("Attempt %d: implausible duration %.1fs", attempt + 1, seconds)
            if score < best_score:
                best, best_score = cleaned, score
        if best is None:
            raise NoAudioError(f"MOSS-TTS returned no audio for: {text[:80]!r}")
        return best, f"Kept best of {settings.max_retries + 1} attempts (timing looked off) for: {text[:60]!r}"

    def _load_reference(self, path: str | Path) -> tuple[object, str | None]:
        wav = load_audio_mono(path, self.sample_rate)
        clipped = cut_at_quiet(wav, self.sample_rate, REFERENCE_MAX_S)
        note = None
        if len(clipped) < len(wav):
            note = f"Voice sample is {len(wav) / self.sample_rate:.0f}s; using the first {len(clipped) / self.sample_rate:.0f}s."
        return self.encode_reference(clipped), note

    def narrate(
        self,
        text: str,
        reference_audio: str | Path | None = None,
        language: str | None = "Auto",
        settings: NarrationSettings = NarrationSettings(),
        on_progress: ProgressCallback | None = None,
    ) -> NarrationResult:
        """Turn a full narration script into one continuous waveform.

        If a chunk crashes mid-job, the chunks finished so far are still returned
        (complete=False) so long A100 runs are never thrown away.
        """
        chunks = chunk_narration(text, settings.max_chunk_weight)
        if not chunks:
            raise ValueError("The narration text is empty.")
        if settings.seed is not None:
            import torch

            torch.manual_seed(settings.seed)
        lang = language_tag(language)
        reference_codes, note = self._load_reference(reference_audio) if reference_audio else (None, None)
        warnings = [note] if note else []

        rendered, complete = self._render_chunks(chunks, reference_codes, lang, settings, on_progress, warnings)
        gaps = [_gap_after(chunk, settings) for chunk, _ in rendered[:-1]]
        audio = stitch([wav for _, wav in rendered], gaps, self.sample_rate)
        if settings.denoise:
            audio = denoise(audio, self.sample_rate)
        audio = soft_limit(audio)
        return NarrationResult(audio, self.sample_rate, len(chunks), tuple(warnings), complete)

    def _render_chunks(
        self,
        chunks: list[Chunk],
        reference_codes,
        lang: str | None,
        settings: NarrationSettings,
        on_progress: ProgressCallback | None,
        warnings: list[str],
    ) -> tuple[list[tuple[Chunk, np.ndarray]], bool]:
        rendered: list[tuple[Chunk, np.ndarray]] = []
        lock_voice = reference_codes is None and settings.voice_lock
        for index, chunk in enumerate(chunks):
            if on_progress:
                on_progress(index, len(chunks), chunk.text)
            try:
                audio, warning = self.synthesize_checked(chunk.text, reference_codes, lang, settings)
            except NoAudioError as error:
                warnings.append(f"Skipped chunk {index + 1}: {error}")
                continue
            except (Exception, KeyboardInterrupt) as error:
                if not rendered:
                    raise
                logger.exception("Chunk %d failed", index + 1)
                warnings.append(f"Stopped at chunk {index + 1}/{len(chunks)} ({error!r}). Saved the {len(rendered)} chunks finished before it.")
                return rendered, False
            rendered.append((chunk, audio))
            if warning:
                warnings.append(warning)
            if lock_voice and warning is None and index < len(chunks) - 1:
                # No voice sample given: reuse the first clean chunk's voice for the rest of the narration.
                reference_codes = self.encode_reference(cut_at_quiet(audio, self.sample_rate, settings.voice_lock_max_s))
                lock_voice = False
        if not rendered:
            raise NoAudioError("MOSS-TTS produced no audio for any chunk.")
        if on_progress:
            on_progress(len(chunks), len(chunks), "")
        return rendered, True


def _gap_after(chunk: Chunk, settings: NarrationSettings) -> float:
    if chunk.ends_paragraph:
        return settings.paragraph_gap_s
    if chunk.ends_sentence:
        return settings.sentence_gap_s
    return settings.mid_sentence_gap_s
