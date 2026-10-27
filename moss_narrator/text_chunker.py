"""Split long narration text into chunks MOSS-TTS can synthesize reliably.

Paragraphs are never merged, so paragraph breaks become longer pauses.
Inside a paragraph, sentences are packed into chunks up to a size budget.
Oversized sentences fall back to clause, word, and finally character splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# CJK characters carry ~3.5x more speech per character than Latin text
# (MOSS-TTS reference app: 3.10 vs 0.87 audio tokens per character).
CJK_CHAR_WEIGHT = 3
# ~900 Latin chars ≈ 60 s of speech: fewer joins, and MOSS-TTS stays stable at this length.
DEFAULT_MAX_CHUNK_WEIGHT = 900

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff66-\uff9f]")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Latin terminators only count when followed by whitespace, so "3.2s" in
# "[pause 3.2s]" or "e.g." mid-word never splits. CJK terminators always split.
# Also covers Devanagari (। ॥), Arabic/Persian (؟) and Urdu (۔) full stops.
_SENTENCE_END_RE = re.compile(r"[.!?…؟।॥۔]+[\"'”’)\]]*(?=\s)|[。！？]+[”’」』）]*")
_CLAUSE_END_RE = re.compile(r"[,;:—،؛]+(?=\s)|[，；：、]+")


@dataclass(frozen=True)
class Chunk:
    text: str
    ends_paragraph: bool
    # False when a long sentence had to be cut here, so the join needs only a tiny pause.
    ends_sentence: bool = True


def text_weight(text: str) -> int:
    cjk = len(CJK_RE.findall(text))
    return (len(text) - cjk) + cjk * CJK_CHAR_WEIGHT


def _cut_after(text: str, pattern: re.Pattern) -> list[str]:
    pieces, start = [], 0
    for match in pattern.finditer(text):
        piece = text[start : match.end()].strip()
        if piece:
            pieces.append(piece)
        start = match.end()
    tail = text[start:].strip()
    return pieces + [tail] if tail else pieces


def split_sentences(text: str) -> list[str]:
    return _cut_after(text, _SENTENCE_END_RE)


def _hard_split(token: str, budget: int) -> list[str]:
    pieces, current = [], ""
    for char in token:
        if current and text_weight(current + char) > budget:
            pieces.append(current)
            current = ""
        current += char
    return pieces + [current] if current else pieces


def _pack_flagged(units: list[tuple[str, bool]], budget: int, joiner: str) -> list[tuple[str, bool]]:
    """Greedily join (text, ends_sentence) units into groups within budget; a group takes its last unit's flag."""
    groups: list[tuple[str, bool]] = []
    current, current_ends = "", True
    for unit, ends in units:
        candidate = f"{current}{joiner}{unit}" if current else unit
        if current and text_weight(candidate) > budget:
            groups.append((current, current_ends))
            candidate = unit
        current, current_ends = candidate, ends
    return groups + [(current, current_ends)] if current else groups


def _pack(units: list[str], budget: int, joiner: str) -> list[str]:
    return [text for text, _ in _pack_flagged([(unit, True) for unit in units], budget, joiner)]


def _joiner_for(text: str) -> str:
    return "" if CJK_RE.search(text[-1:] or "") else " "


def _fit(sentence: str, budget: int) -> list[str]:
    """Break one sentence into pieces that each fit the budget."""
    if text_weight(sentence) <= budget:
        return [sentence]
    clauses = _cut_after(sentence, _CLAUSE_END_RE)
    if len(clauses) > 1:
        packed = _pack(clauses, budget, _joiner_for(sentence))
        return [piece for clause in packed for piece in _fit(clause, budget)]
    words = sentence.split()
    if len(words) > 1:
        # An oversized single word ends up alone in its group and is hard-split there,
        # so its pieces are never re-joined with spaces.
        groups = _pack(words, budget, " ")
        return [piece for group in groups for piece in _fit(group, budget)]
    return _hard_split(sentence, budget)


def _paragraph_units(paragraph: str, budget: int) -> list[tuple[str, bool]]:
    units: list[tuple[str, bool]] = []
    for line in paragraph.splitlines():
        for sentence in split_sentences(line.strip()):
            pieces = _fit(sentence, budget)
            units.extend((piece, i == len(pieces) - 1) for i, piece in enumerate(pieces))
    return units


def chunk_narration(text: str, max_chunk_weight: int = DEFAULT_MAX_CHUNK_WEIGHT) -> list[Chunk]:
    """Split narration into ordered chunks, flagging the last chunk of each paragraph."""
    if max_chunk_weight < 20:
        raise ValueError("max_chunk_weight must be at least 20")
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks: list[Chunk] = []
    for paragraph in _PARAGRAPH_RE.split(normalized):
        units = _paragraph_units(paragraph, max_chunk_weight)
        if not units:
            continue
        joiner = _joiner_for(units[0][0])
        groups = _pack_flagged(units, max_chunk_weight, joiner)
        chunks.extend(
            Chunk(text=group, ends_paragraph=(i == len(groups) - 1), ends_sentence=ends)
            for i, (group, ends) in enumerate(groups)
        )
    return chunks
