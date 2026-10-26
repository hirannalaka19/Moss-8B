"""ReferenceCleaner against a fake Demucs separator (real torch + demucs.audio, no model download)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("demucs")

from moss_narrator.voice_cleaner import ReferenceCleaner  # noqa: E402

SR = 24000


class FakeSeparator:
    """Mimics demucs.api.Separator: resamples to 44.1 kHz stereo, 'vocals' = half the input."""

    samplerate = 44100

    def __init__(self):
        self.calls = 0

    def separate_tensor(self, wav, sr):
        from demucs.audio import convert_audio

        self.calls += 1
        resampled = convert_audio(wav, sr, self.samplerate, 2)
        return resampled, {"vocals": resampled * 0.5, "drums": resampled * 0.0}


def make_cleaner():
    cleaner = ReferenceCleaner(device="cpu")
    cleaner._separator = FakeSeparator()
    return cleaner


def test_clean_returns_mono_vocals_at_input_rate_and_length():
    t = np.arange(SR) / SR
    wav = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = make_cleaner().clean(wav, SR)
    assert out.dtype == np.float32 and out.shape == wav.shape
    core = slice(SR // 10, -SR // 10)  # ignore resampler edge effects
    assert np.allclose(out[core], wav[core] * 0.5, atol=0.02)


def test_separator_is_loaded_once_and_reused():
    cleaner = make_cleaner()
    wav = np.zeros(SR // 2, dtype=np.float32)
    cleaner.clean(wav, SR)
    cleaner.clean(wav, SR)
    assert cleaner._separator.calls == 2
