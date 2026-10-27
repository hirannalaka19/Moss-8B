"""Engine orchestration tests with a fake synthesizer (no torch / GPU needed)."""

import numpy as np
import pytest

from moss_narrator.engine import MossNarrator, NoAudioError
from moss_narrator.planning import NarrationSettings, expected_seconds

SR = 24000


def speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


class FakeNarrator(MossNarrator):
    """Plays back scripted durations (multiples of the expected length) per call."""

    def __init__(self, length_factors=None):
        super().__init__(model=None, processor=None, sample_rate=SR)
        self.length_factors = list(length_factors or [])
        self.calls = []
        self.prefixes = []
        self.encoded = []

    def synthesize(self, text, reference_codes, language, settings, prefix=None):
        self.calls.append((text, reference_codes, language))
        self.prefixes.append(prefix)
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


def test_warm_up_take_guides_every_chunk_including_the_first():
    narrator = FakeNarrator()
    narrator.narrate(TEXT, settings=NarrationSettings(voice_lock_max_s=1.0))
    first_chunk = TEXT.splitlines()[0]
    assert narrator.calls[0] == (first_chunk, None, None)  # warm-up: the only unguided take
    assert [ref for _, ref, _ in narrator.calls[1:]] == ["codes#1", "codes#1"]
    assert len(narrator.encoded) == 2  # voice reference + continuation prefix
    assert narrator.encoded[0] <= SR  # reference capped at voice_lock_max_s


def test_warm_up_skips_the_opening_seconds_where_intro_music_sits():
    narrator = FakeNarrator()
    narrator.narrate(TEXT, settings=NarrationSettings(voice_lock_max_s=60.0))
    warm_up_len = int(expected_seconds(TEXT.splitlines()[0]) * SR)
    skipped = warm_up_len - narrator.encoded[0]
    assert skipped >= min(2 * SR, warm_up_len // 3) - SR // 5  # minus silence-trim slack


def test_no_warm_up_without_voice_lock():
    narrator = FakeNarrator()
    narrator.narrate(TEXT, settings=NarrationSettings(voice_lock=False))
    assert len(narrator.calls) == 2 and narrator.encoded == []


def test_reference_audio_is_encoded_once_and_used_everywhere(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(3.0))
    narrator = FakeNarrator()
    result = narrator.narrate(TEXT, reference_audio="voice.wav", language="English")
    assert narrator.encoded[0] == 3 * SR and len(narrator.encoded) == 2  # sample + warm-up prefix
    assert result.warnings == ()
    assert [ref for _, ref, _ in narrator.calls] == ["codes#1", "codes#1", "codes#1"]  # warm-up + 2 chunks
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
    assert events == [(0, 2), (0, 2), (1, 2), (2, 2)]  # warm-up, chunk 1, chunk 2, done


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        FakeNarrator().narrate("   ")


THREE = "\n\n".join(["One sentence that is fine.", "Two sentence that is fine.", "Three sentence that is fine."])


class CrashingNarrator(FakeNarrator):
    def __init__(self, crash_on_call):
        super().__init__()
        self.crash_on_call = crash_on_call

    def synthesize(self, text, reference_codes, language, settings, prefix=None):
        if len(self.calls) + 1 == self.crash_on_call:
            self.calls.append((text, reference_codes, language))
            raise RuntimeError("CUDA out of memory")
        return super().synthesize(text, reference_codes, language, settings, prefix)


def test_crash_mid_job_returns_finished_chunks_as_partial():
    result = CrashingNarrator(crash_on_call=4).narrate(THREE, settings=NarrationSettings(max_retries=0))  # call 1 = warm-up
    assert result.complete is False
    assert any("Stopped at chunk 3/3" in w for w in result.warnings)
    assert result.duration_s > 0


def test_crash_on_first_chunk_propagates():
    with pytest.raises(RuntimeError, match="out of memory"):
        CrashingNarrator(crash_on_call=1).narrate(THREE)


def test_chunk_with_no_audio_is_skipped_with_warning():
    narrator = FakeNarrator(length_factors=[1.0, 1.0, None, 1.0])  # warm-up, chunk 1, chunk 2 (fails), chunk 3
    result = narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert result.complete is True
    assert any("Skipped chunk 2" in w for w in result.warnings)


def test_bad_warm_up_falls_back_to_locking_on_first_clean_chunk():
    narrator = FakeNarrator(length_factors=[6.0, 1.0, 1.0, 1.0])  # warm-up timing is off -> discarded
    narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert [ref for _, ref, _ in narrator.calls] == [None, None, "codes#1", "codes#1"]


def test_warm_up_with_no_audio_falls_back_too():
    narrator = FakeNarrator(length_factors=[None, 1.0, 1.0, 1.0])
    result = narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert result.complete and [ref for _, ref, _ in narrator.calls] == [None, None, "codes#1", "codes#1"]


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


def test_warm_up_reads_only_the_opening_sentences_of_a_long_first_chunk():
    long_paragraph = " ".join(f"Sentence number {i} keeps the story moving forward." for i in range(15))
    narrator = FakeNarrator()
    narrator.narrate(long_paragraph)
    warm_up_text = narrator.calls[0][0]
    assert len(warm_up_text) <= 250 and long_paragraph.startswith(warm_up_text)
    assert narrator.calls[1][0] != warm_up_text  # the real chunk 1 is the full first chunk


def test_chunk_one_continues_from_the_warm_up_take():
    narrator = FakeNarrator()
    narrator.narrate(TEXT)
    warm_up_text = narrator.calls[0][0]
    warm_up, chunk1, chunk2 = narrator.prefixes
    assert warm_up is None and chunk2 is None  # only chunk 1 is a continuation
    assert chunk1.text == warm_up_text and chunk1.codes == "codes#2"


def test_voice_clone_also_gets_the_warm_up_continuation(monkeypatch):
    monkeypatch.setattr("moss_narrator.engine.load_audio_mono", lambda path, sr: speech(3.0))
    narrator = FakeNarrator()
    narrator.narrate(TEXT, reference_audio="voice.wav", settings=NarrationSettings(voice_lock=False))
    assert narrator.prefixes[0] is None and narrator.prefixes[1] is not None


def test_no_continuation_when_the_warm_up_take_is_unusable():
    narrator = FakeNarrator(length_factors=[6.0, 1.0, 1.0, 1.0])
    narrator.narrate(THREE, settings=NarrationSettings(max_retries=0))
    assert narrator.prefixes == [None, None, None, None]
