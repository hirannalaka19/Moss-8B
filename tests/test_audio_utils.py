import shutil
import wave

import numpy as np
import pytest

from moss_narrator.audio_utils import (
    PEAK_CEILING,
    apply_edge_fades,
    cut_at_quiet,
    denoise,
    quiet_passages,
    load_audio_mono,
    normalize_loudness,
    save_audio,
    soft_limit,
    stitch,
    trim_silence,
)

SR = 24000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_removes_padding_but_keeps_margin():
    wav = np.concatenate([np.zeros(SR), tone(1.0), np.zeros(SR)]).astype(np.float32)
    trimmed = trim_silence(wav, SR, keep_ms=80)
    assert 1.0 <= len(trimmed) / SR <= 1.0 + 2 * 0.08 + 0.05


def test_trim_silence_leaves_all_silent_and_tiny_input_unchanged():
    silent = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(trim_silence(silent, SR), silent)
    tiny = np.ones(10, dtype=np.float32)
    assert np.array_equal(trim_silence(tiny, SR), tiny)


def test_trim_silence_does_not_mutate_input():
    wav = np.concatenate([np.zeros(SR), tone(0.5)]).astype(np.float32)
    original = wav.copy()
    trim_silence(wav, SR)
    assert np.array_equal(wav, original)


def test_normalize_loudness_hits_target_rms():
    out = normalize_loudness(tone(1.0, amplitude=0.05), target_dbfs=-20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out**2)))
    assert rms_db == pytest.approx(-20.0, abs=0.1)


def test_normalize_loudness_caps_gain_for_near_silent_audio():
    out = normalize_loudness(tone(1.0, amplitude=1e-4), target_dbfs=-20.0, max_gain_db=20.0)
    assert np.max(np.abs(out)) == pytest.approx(1e-3, rel=0.01)


def test_soft_limit_passes_quiet_samples_and_bounds_loud_ones():
    wav = np.array([0.0, 0.5, -0.79, 0.9, -1.5, 4.0], dtype=np.float32)
    out = soft_limit(wav)
    assert np.array_equal(out[:3], wav[:3])
    assert np.all(np.abs(out) <= PEAK_CEILING)
    assert out[3] > 0.8 and out[4] < -0.8
    assert out[5] > out[3]  # monotonic


def test_cut_at_quiet_keeps_short_audio_and_cuts_long_audio_in_a_gap():
    short = tone(1.0)
    assert np.array_equal(cut_at_quiet(short, SR, 5.0), short)
    gap_at = 3.5
    wav = np.concatenate([tone(gap_at), np.zeros(int(0.2 * SR), np.float32), tone(3.0)])
    cut = cut_at_quiet(wav, SR, 5.0)
    assert gap_at <= len(cut) / SR <= gap_at + 0.2


def test_normalize_loudness_handles_empty_and_silent():
    assert normalize_loudness(np.zeros(0, dtype=np.float32)).size == 0
    assert np.array_equal(normalize_loudness(np.zeros(5, dtype=np.float32)), np.zeros(5))


def test_edge_fades_zero_the_endpoints_only():
    out = apply_edge_fades(np.ones(SR, dtype=np.float32), SR)
    assert out[0] == 0.0 and out[-1] == 0.0
    assert out[SR // 2] == 1.0


def test_stitch_inserts_requested_gaps():
    a, b, c = np.ones(100, np.float32), np.ones(200, np.float32), np.ones(50, np.float32)
    out = stitch([a, b, c], [0.01, 0.5], SR)
    assert len(out) == 100 + 240 + 200 + 12000 + 50
    assert np.all(out[100:340] == 0)


def test_stitch_validates_arguments():
    with pytest.raises(ValueError):
        stitch([], [], SR)
    with pytest.raises(ValueError):
        stitch([np.ones(3, np.float32)] * 2, [], SR)


def test_save_wav_roundtrip(tmp_path):
    path = save_audio(tone(0.5), SR, tmp_path / "out" / "a.wav")
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == SR
        assert handle.getnchannels() == 1
        assert handle.getnframes() == SR // 2


def test_save_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        save_audio(tone(0.1), SR, tmp_path / "a.ogg")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio_mono(tmp_path / "nope.wav", SR)


@needs_ffmpeg
def test_mp3_export_and_load_resamples(tmp_path):
    path = save_audio(tone(1.0), SR, tmp_path / "a.mp3")
    loaded = load_audio_mono(path, 16000)
    assert loaded.dtype == np.float32
    assert abs(len(loaded) / 16000 - 1.0) < 0.1


def narration_with_hiss(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Alternating 'speech' bursts and pauses, all over a constant hiss floor; returns (audio, pause mask)."""
    rng = np.random.default_rng(seed)
    pattern = [(1.2, True), (0.4, False)] * 6
    parts, mask = [], []
    for seconds, voiced in pattern:
        n = int(seconds * SR)
        t = np.arange(n) / SR
        voice = 0.3 * np.sin(2 * np.pi * 180 * t) * np.sin(np.pi * t / seconds) if voiced else np.zeros(n)
        parts.append(voice + 0.002 * rng.standard_normal(n))
        mask.append(np.full(n, not voiced))
    return np.concatenate(parts).astype(np.float32), np.concatenate(mask)


def level_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def test_quiet_passages_finds_model_pauses_but_not_digital_silence():
    wav, _ = narration_with_hiss()
    wav = np.concatenate([wav, np.zeros(SR, np.float32)])
    runs = quiet_passages(wav, SR)
    assert len(runs) >= 5
    assert all(b <= len(wav) - SR for _, b in runs)  # the appended digital silence is never used


def test_denoise_lowers_pause_noise_and_keeps_speech():
    pytest.importorskip("noisereduce")
    wav, pauses = narration_with_hiss()
    out = denoise(wav, SR)
    assert out.dtype == np.float32 and len(out) == len(wav)
    assert level_db(out[pauses]) < level_db(wav[pauses]) - 10
    assert abs(level_db(out[~pauses]) - level_db(wav[~pauses])) < 1.5


def test_denoise_without_pauses_only_high_passes():
    pytest.importorskip("scipy")
    t = np.arange(2 * SR) / SR
    wav = (0.3 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
    out = denoise(wav, SR)
    spectrum = np.abs(np.fft.rfft(out[SR // 2: SR // 2 + SR]))
    assert spectrum[20] < spectrum[200] * 0.2  # 20 Hz rumble removed, 200 Hz tone kept
