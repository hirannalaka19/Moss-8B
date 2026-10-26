import pytest

from moss_narrator.text_chunker import Chunk, chunk_narration, split_sentences, text_weight


def test_split_sentences_keeps_terminal_punctuation_and_quotes():
    text = 'He said "Stop!" Then he left. Did she follow?'
    assert split_sentences(text) == ['He said "Stop!"', "Then he left.", "Did she follow?"]


def test_split_sentences_does_not_break_pause_markers_or_decimals():
    text = "The answer is [pause 3.2s] forty-two. Pi is 3.14 roughly."
    assert split_sentences(text) == ["The answer is [pause 3.2s] forty-two.", "Pi is 3.14 roughly."]


def test_split_sentences_handles_cjk_without_spaces():
    assert split_sentences("你好。今天天气很好！我们走吧？") == ["你好。", "今天天气很好！", "我们走吧？"]


def test_text_weight_counts_cjk_heavier():
    assert text_weight("abc") == 3
    assert text_weight("你好") == 6


def test_short_text_is_single_chunk():
    assert chunk_narration("Hello world. This is short.") == [Chunk("Hello world. This is short.", True)]


def test_empty_text_returns_no_chunks():
    assert chunk_narration("   \n\n  ") == []
    assert chunk_narration(None) == []


def test_paragraphs_are_never_merged_and_flag_paragraph_ends():
    chunks = chunk_narration("First paragraph.\n\nSecond paragraph.\r\n\r\nThird.")
    assert [c.text for c in chunks] == ["First paragraph.", "Second paragraph.", "Third."]
    assert all(c.ends_paragraph for c in chunks)


def test_sentences_packed_up_to_budget_without_losing_text():
    sentence = "This sentence has exactly forty chars.."  # 39 chars + period
    text = " ".join([sentence] * 10)
    chunks = chunk_narration(text, max_chunk_weight=100)
    assert all(text_weight(c.text) <= 100 for c in chunks)
    assert " ".join(c.text for c in chunks) == text
    assert [c.ends_paragraph for c in chunks] == [False] * (len(chunks) - 1) + [True]


def test_overlong_sentence_splits_on_clauses_then_words():
    clause = "one two three four five six seven eight"
    text = ", ".join([clause] * 6) + "."
    chunks = chunk_narration(text, max_chunk_weight=60)
    assert len(chunks) > 1
    assert all(text_weight(c.text) <= 60 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_word_without_spaces_is_hard_split():
    chunks = chunk_narration("x" * 130, max_chunk_weight=50)
    assert [len(c.text) for c in chunks] == [50, 50, 30]


def test_cjk_long_paragraph_respects_budget_and_joins_without_spaces():
    text = "我今天学习了一首中国的古诗，它的名字是静夜思。" * 12
    chunks = chunk_narration(text, max_chunk_weight=120)
    assert all(text_weight(c.text) <= 120 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_single_newlines_are_kept_inside_paragraph():
    chunks = chunk_narration("Line one\nLine two\nLine three")
    assert chunks == [Chunk("Line one Line two Line three", True)]


def test_rejects_tiny_budget():
    with pytest.raises(ValueError):
        chunk_narration("hello", max_chunk_weight=5)


def test_mid_sentence_cuts_are_flagged():
    text = ", ".join(["alpha beta gamma delta"] * 8) + ". Short end."
    chunks = chunk_narration(text, max_chunk_weight=60)
    assert chunks[0].ends_sentence is False
    assert chunks[-1].ends_sentence is True


def test_split_sentences_handles_hindi_and_arabic_terminators():
    assert split_sentences("यह पहला वाक्य है। यह दूसरा है।") == ["यह पहला वाक्य है।", "यह दूसरा है।"]
    assert split_sentences("كيف حالك؟ أنا بخير.") == ["كيف حالك؟", "أنا بخير."]


def test_long_word_pieces_are_not_rejoined_with_spaces():
    word = "x" * 120
    texts = [c.text for c in chunk_narration(f"start {word} end", max_chunk_weight=50)]
    assert texts == ["start", "x" * 50, "x" * 50, "x" * 20 + " end"]
