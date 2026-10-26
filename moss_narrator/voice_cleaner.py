"""Remove background sound (music, noise, hum, room) from a voice-clone reference with Demucs.

MOSS-TTS copies the acoustics of the reference along with the voice, so any background
in the sample ends up in every generated chunk. Demucs' fine-tuned hybrid transformer
(htdemucs_ft) isolates the vocal stem. Measured on speech mixed with backgrounds at 5 dB SNR:
music -> 17.5 dB, hum+hiss -> 19.3 dB, crowd/traffic noise -> 10.3 dB; clean speech passes
through almost untouched (31 dB, level within 0.1 dB).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_HF_REPO = "adefossez/HTDemucs-ft"


class ReferenceCleaner:
    """Lazily loads Demucs on first use and keeps it resident (~170M params, small next to the 8B TTS)."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._separator = None

    def _get_separator(self):
        if self._separator is None:
            from demucs.api import Separator

            logger.info("Loading Demucs %s for reference cleanup", DEMUCS_MODEL)
            self._separator = Separator(DEMUCS_MODEL, device=self._device)
        return self._separator

    def clean(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the mono vocal stem of `wav` at the same sample rate and length."""
        import torch
        from demucs.audio import convert_audio

        separator = self._get_separator()
        tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None]
        with torch.no_grad():
            _, stems = separator.separate_tensor(tensor, sample_rate)
        vocals = convert_audio(stems["vocals"].cpu(), separator.samplerate, sample_rate, 1)[0]
        out = vocals.numpy().astype(np.float32)[: len(wav)]
        return np.pad(out, (0, len(wav) - len(out)))
