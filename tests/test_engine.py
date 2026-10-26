"""Engine orchestration tests with a fake synthesizer (no torch / GPU needed)."""

import numpy as np
import pytest

from moss_narrator.engine import MossNarrator, NoAudioError
from moss_narrator.planning import NarrationSettings, expected_seconds

SR = 24000


@pytest.fixture(autouse=True)
def no_real_denoise(monkeypatch):
    """Engine tests cover orchestration; the real denoiser is tested in test_audio_utils."""
    calls = []
    monkeypatch.setattr("moss_narrator.engine.denoise", lambda wav, sr: calls.append(len(wav)) or wav * 0.5)
    return calls


def speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


class FakeCleaner:
    def __init__(self):
        self.calls = 0

    def clean(self, wav, sample_rate):
        self.calls += 1
        return wav * 0.5


class FakeNarrator(MossNarrator):
    """Plays back scripted durations (multiples of the expected length) per call."""

    def __init__(self, length_factors=None):
        self.cleaner = FakeCleaner()
        super().__init__(model=None, processor=None, sample_rate=SR, reference_cleaner=self.cleaner)
        self.length_factors = list(length_factors or [])
        self.calls = []
        self.encoded = []

    def synthesize(self, text, reference_codes, language, settings):
        self.calls.append((text, reference_codes, language))
        factor = self.length_factors.pop(0) if self.length_factors else 1.0
        if factor is None:
            return None
        return speech(expected_seconds(text) * factor)

    def encode_reference(self, wav):
        self.encoded.append(len(wav))
        return f"codes#{len(self.encoded)}"


TEXT = "First sentence is here and fine.\n\nSecond paragraph sentence that is also fine."


def test_narrate_stitches_chunks_with_paragraph_gap():
    narrator = FakeNarrator()
    settings = NarrationSettings(paragraph_gap_s=1.0, voice_lock=False)
    result = narrator.narrate(TEXT, settings=settings)
    assert result.chunk_count == 2
    assert result.warnings == ()
    pieces = sum(expected_seconds(t) for t, _, _ in narrator.calls)
    assert result.duration_s == pytest.approx(pieces + 1.0, abs=0.3)


def test_voice_lock_reuses_first_chunk_voice():
    narrator = FakeNarrator()
    narrator.narrate(TEXT, settings=NarrationSettings(voice_lock_max_s=1.0))
    assert narrator.calls[0][1] is None
    assert narrator.calls[1][1] == "codes#1"
    assert len(narrator.encoded) == 1
    assert narrator.encoded[0] <= SR  # capped at voice_lock_max_s


def test_reference_audio_is_encoded_once_and_used_everywhere(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(3.0))
    narrator = FakeNarrator()
    result = narrator.narrate(TEXT, reference_audio="voice.wav", language="English")
    assert narrator.encoded == [3 * SR]
    assert result.warnings == ()
    assert [ref for _, ref, _ in narrator.calls] == ["codes#1", "codes#1"]
    assert {lang for _, _, lang in narrator.calls} == {"English"}


def test_retries_implausible_duration_then_accepts():
    narrator = FakeNarrator(length_factors=[5.0, None, 1.0])
    audio, warning = narrator.synthesize_checked(TEXT.split("\n")[0], None, None, NarrationSettings())
    assert warning is None
    assert len(narrator.calls) == 3
    assert len(audio) / SR == pytest.approx(expected_seconds(TEXT.split("\n")[0]), abs=0.3)


def test_keeps_closest_attempt_when_all_retries_fail():
    narrator = FakeNarrator(length_factors=[6.0, 0.1, 3.0])
    audio, warning = narrator.synthesize_checked("A sentence of reasonable length for testing.", None, None, NarrationSettings())
    assert warning is not None and "best of 3" in warning
    expected = expected_seconds("A sentence of reasonable length for testing.")
    assert len(audio) / SR == pytest.approx(expected * 3.0, rel=0.1)


def test_raises_when_model_never_returns_audio():
    narrator = FakeNarrator(length_factors=[None, None, None])
    with pytest.raises(NoAudioError, match="no audio"):
        narrator.synthesize_checked("Hello there.", None, None, NarrationSettings())


def test_progress_callback_reports_every_chunk():
    events = []
    FakeNarrator().narrate(TEXT, on_progress=lambda done, total, text: events.append((done, total)))
    assert events == [(0, 2), (1, 2), (2, 2)]


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        FakeNarrator().narrate("   ")


THREE = "\n\n".join(["One sentence that is fine.", "Two sentence that is fine.", "Three sentence that is fine."])


class CrashingNarrator(FakeNarrator):
    def __init__(self, crash_on_call):
        super().__init__()
        self.crash_on_call = crash_on_call

    def synthesize(self, text, reference_codes, language, settings):
        if len(self.calls) + 1 == self.crash_on_call:
            self.calls.append((text, reference_codes, language))
            raise RuntimeError("CUDA out of memory")
        return super().synthesize(text, reference_codes, language, settings)


def test_crash_mid_job_returns_finished_chunks_as_partial():
    result = CrashingNarrator(crash_on_call=3).narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert result.complete is False
    assert any("Stopped at chunk 3/3" in w for w in result.warnings)
    assert result.duration_s > 0


def test_crash_on_first_chunk_propagates():
    with pytest.raises(RuntimeError, match="out of memory"):
        CrashingNarrator(crash_on_call=1).narrate(THREE)


def test_chunk_with_no_audio_is_skipped_with_warning():
    narrator = FakeNarrator(length_factors=[1.0, None, 1.0])
    result = narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert result.complete is True
    assert any("Skipped chunk 2" in w for w in result.warnings)


def test_voice_lock_waits_for_a_clean_chunk():
    narrator = FakeNarrator(length_factors=[6.0, 1.0, 1.0])
    narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert [ref for _, ref, _ in narrator.calls] == [None, None, "codes#1"]


def test_long_reference_is_capped(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(45.0))
    narrator = FakeNarrator()
    result = narrator.narrate(TEXT, reference_audio="long.wav")
    assert narrator.encoded[0] <= 30 * SR
    assert any("using the first" in w for w in result.warnings)


def test_mid_sentence_chunks_use_short_gap():
    from moss_narrator.engine import _gap_after
    from moss_narrator.text_chunker import Chunk

    settings = NarrationSettings()
    assert _gap_after(Chunk("a", ends_paragraph=False, ends_sentence=False), settings) == settings.mid_sentence_gap_s
    assert _gap_after(Chunk("a", ends_paragraph=False), settings) == settings.sentence_gap_s
    assert _gap_after(Chunk("a", ends_paragraph=True), settings) == settings.paragraph_gap_s


def test_denoise_runs_once_on_the_stitched_narration(no_real_denoise):
    FakeNarrator().narrate(TEXT)
    assert len(no_real_denoise) == 1


def test_denoise_can_be_switched_off(no_real_denoise):
    FakeNarrator().narrate(TEXT, settings=NarrationSettings(denoise=False))
    assert no_real_denoise == []


def test_reference_background_is_removed_by_default(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(3.0))
    narrator = FakeNarrator()
    narrator.narrate(TEXT, reference_audio="voice.wav")
    assert narrator.cleaner.calls == 1


def test_reference_cleanup_can_be_switched_off(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(3.0))
    narrator = FakeNarrator()
    narrator.narrate(TEXT, reference_audio="voice.wav", settings=NarrationSettings(clean_reference=False))
    assert narrator.cleaner.calls == 0


def test_previewed_reference_is_reused_by_generate(monkeypatch):
    loads = []
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: loads.append(path) or speech(3.0))
    narrator = FakeNarrator()
    preview, _ = narrator.prepare_reference("voice.wav", clean=True)
    narrator.narrate(TEXT, reference_audio="voice.wav")
    assert loads == ["voice.wav"] and narrator.cleaner.calls == 1
    assert np.allclose(preview, speech(3.0) * 0.5)
    narrator.prepare_reference("other.wav", clean=True)
    assert loads == ["voice.wav", "other.wav"]
