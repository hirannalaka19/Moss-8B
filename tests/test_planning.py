import pytest

from moss_narrator.planning import (
    MAX_NEW_TOKENS_CAP,
    LANGUAGES,
    NarrationSettings,
    check_gpu,
    duration_score,
    estimate_audio_tokens,
    expected_seconds,
    is_duration_plausible,
    language_tag,
    max_new_tokens_for,
)

ENGLISH = "The quick brown fox jumps over the lazy dog. " * 4  # 180 chars, ~12.5 s


def test_estimates_scale_with_script():
    assert estimate_audio_tokens("你好") > estimate_audio_tokens("hi") * 2
    assert expected_seconds(ENGLISH) == pytest.approx(180 * 0.867 / 12.5, rel=0.01)


def test_max_new_tokens_has_floor_and_cap():
    assert max_new_tokens_for("Hi.") == 400
    assert max_new_tokens_for("word " * 5000) == MAX_NEW_TOKENS_CAP
    assert max_new_tokens_for(ENGLISH) > estimate_audio_tokens(ENGLISH) * 2


def test_duration_plausibility_window():
    expected = expected_seconds(ENGLISH)
    assert is_duration_plausible(expected, ENGLISH)
    assert not is_duration_plausible(expected * 0.2, ENGLISH)
    assert not is_duration_plausible(expected * 4, ENGLISH)


def test_short_text_plausibility_is_lenient():
    assert is_duration_plausible(2.5, "Yes.")
    assert not is_duration_plausible(0.05, "Yes.")
    assert not is_duration_plausible(30.0, "Yes.")


def test_duration_score_prefers_closer_attempts():
    expected = expected_seconds(ENGLISH)
    assert duration_score(expected, ENGLISH) == pytest.approx(0.0, abs=1e-9)
    assert duration_score(expected * 1.5, ENGLISH) < duration_score(expected * 3, ENGLISH)
    assert duration_score(0, ENGLISH) == float("inf")


def test_language_tag_mapping():
    assert language_tag("Auto") is None
    assert language_tag(None) is None
    assert language_tag("French") == "French"
    assert len(LANGUAGES) == 32  # Auto + 31 supported languages
    with pytest.raises(ValueError):
        language_tag("Klingon")


def test_check_gpu_requires_a100_class_memory():
    check_gpu(True, 39.6)
    with pytest.raises(RuntimeError, match="No GPU"):
        check_gpu(False, 0.0)
    with pytest.raises(RuntimeError, match="A100"):
        check_gpu(True, 15.0)


def test_settings_are_immutable():
    settings = NarrationSettings()
    with pytest.raises(AttributeError):
        settings.temperature = 1.0  # type: ignore[misc]
