"""One relevance function, one threshold, used everywhere a retrieved chunk is
checked against a gold span (M01-RETRIEVAL.md §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int


def overlap_fraction(chunk: Span, gold: Span) -> float:
    """Fraction of `gold`'s length covered by its overlap with `chunk`.
    Zero if they're in different documents — char offsets are only comparable
    within the same doc_id."""
    if chunk.doc_id != gold.doc_id:
        return 0.0
    overlap = max(0, min(chunk.end, gold.end) - max(chunk.start, gold.start))
    gold_len = gold.end - gold.start
    if gold_len <= 0:
        return 0.0
    return overlap / gold_len


def is_relevant(chunk: Span, gold: Span, thresh: float = DEFAULT_THRESHOLD) -> bool:
    """A retrieved chunk counts as relevant to a gold span if it overlaps
    that span by >= thresh (default 50%) of the gold span's length."""
    return overlap_fraction(chunk, gold) >= thresh
