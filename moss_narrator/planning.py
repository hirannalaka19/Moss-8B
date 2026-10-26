"""Torch-free generation planning: settings, token budgets, and sanity checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .text_chunker import CJK_RE

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
SAMPLE_RATE = 24000
AUDIO_TOKENS_PER_SECOND = 12.5
# Measured averages from the MOSS-TTS reference app (clis/moss_tts_app.py).
CJK_TOKENS_PER_CHAR = 3.098
LATIN_TOKENS_PER_CHAR = 0.867
MIN_VRAM_GB = 30.0
# bf16 model (~17 GB) + fp32 codec (~7 GB) + headroom for activations and KV cache.
MIN_FREE_VRAM_GB = 26.0
MAX_NEW_TOKENS_CAP = 4096
# Longer voice samples only slow every chunk's prompt down without improving similarity.
REFERENCE_MAX_S = 30.0

# Model + codec on one GPU needs ~25 GB before KV cache.
GPU_HELP = (
    "MOSS-TTS-v1.5 (8B) + its audio codec need about 25 GB of GPU memory. "
    "In Colab choose Runtime > Change runtime type > A100 GPU, then Runtime > Run all."
)

LANGUAGES = (
    "Auto", "English", "Chinese", "Cantonese", "Arabic", "Czech", "Danish", "Dutch",
    "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Italian",
    "Japanese", "Korean", "Macedonian", "Malay", "Persian", "Polish", "Portuguese",
    "Romanian", "Russian", "Spanish", "Swahili", "Swedish", "Tagalog", "Thai",
    "Turkish", "Vietnamese",
)


@dataclass(frozen=True)
class NarrationSettings:
    max_chunk_weight: int = 500
    sentence_gap_s: float = 0.30
    mid_sentence_gap_s: float = 0.08
    paragraph_gap_s: float = 0.80
    temperature: float = 1.7
    top_p: float = 0.8
    top_k: int = 25
    repetition_penalty: float = 1.0
    max_retries: int = 2
    voice_lock: bool = True
    voice_lock_max_s: float = 20.0
    seed: int | None = None
    denoise: bool = True


def estimate_audio_tokens(text: str) -> int:
    cjk = len(CJK_RE.findall(text))
    latin = len(text) - cjk
    return max(1, round(cjk * CJK_TOKENS_PER_CHAR + latin * LATIN_TOKENS_PER_CHAR))


def expected_seconds(text: str) -> float:
    return estimate_audio_tokens(text) / AUDIO_TOKENS_PER_SECOND


def max_new_tokens_for(text: str) -> int:
    """Generous ceiling (3x expected + delay-pattern headroom) so slow readers never get cut off."""
    budget = estimate_audio_tokens(text) * 3 + 200
    return int(min(MAX_NEW_TOKENS_CAP, max(400, budget)))


def duration_score(actual_s: float, text: str) -> float:
    """0 means duration matches the estimate; larger means further off (log-ratio)."""
    if actual_s <= 0:
        return float("inf")
    return abs(math.log(actual_s / expected_seconds(text)))


def is_duration_plausible(actual_s: float, text: str, low: float = 0.4, high: float = 2.5) -> bool:
    """Catch truncated or runaway generations by comparing against the text-length estimate."""
    expected = expected_seconds(text)
    # Very short chunks have unreliable estimates; only reject clear failures.
    if expected < 2.0:
        return 0.15 <= actual_s <= expected * high + 3.0
    return expected * low <= actual_s <= expected * high


def language_tag(choice: str | None) -> str | None:
    if not choice or choice == "Auto":
        return None
    if choice not in LANGUAGES:
        raise ValueError(f"Unsupported language: {choice}")
    return choice


def check_gpu(cuda_available: bool, vram_gb: float, free_gb: float | None = None) -> None:
    if not cuda_available:
        raise RuntimeError("No GPU detected. " + GPU_HELP)
    if vram_gb < MIN_VRAM_GB:
        raise RuntimeError(f"This GPU has {vram_gb:.0f} GB, not enough. " + GPU_HELP)
    if free_gb is not None and free_gb < MIN_FREE_VRAM_GB:
        raise RuntimeError(
            f"Only {free_gb:.0f} of {vram_gb:.0f} GB GPU memory is free; another process (often an earlier "
            "run of the app) is using the rest. In Colab: Runtime > Restart session, then run the cells again."
        )
