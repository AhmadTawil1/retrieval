"""Hand-built cases with known answers. An off-by-one in recall@5 should
surface here on day 2, not in a 96-cell sweep on day 6."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import context_precision, mrr, recall_at_k, score
from relevance import Span, is_relevant, overlap_fraction

GOLD = Span("doc1", 100, 200)  # length 100


def chunk(start: int, end: int, doc_id: str = "doc1") -> Span:
    return Span(doc_id, start, end)


# --- overlap_fraction / is_relevant -----------------------------------------


def test_overlap_fraction_full_containment_is_one():
    assert overlap_fraction(chunk(50, 250), GOLD) == 1.0


def test_overlap_fraction_no_overlap_is_zero():
    assert overlap_fraction(chunk(300, 400), GOLD) == 0.0


def test_overlap_fraction_different_doc_is_zero_even_with_matching_offsets():
    assert overlap_fraction(chunk(100, 200, doc_id="doc2"), GOLD) == 0.0


def test_overlap_fraction_exactly_half():
    # overlaps [150,200) = 50 of the gold span's 100 chars
    assert overlap_fraction(chunk(150, 300), GOLD) == 0.5


def test_is_relevant_at_exactly_the_threshold_counts_as_relevant():
    half_overlap = chunk(150, 300)  # exactly 0.5
    assert is_relevant(half_overlap, GOLD, thresh=0.5) is True


def test_is_relevant_just_under_threshold_does_not_count():
    just_under = chunk(151, 300)  # 49/100 = 0.49
    assert is_relevant(just_under, GOLD, thresh=0.5) is False


# --- recall_at_k: off-by-one on the k boundary ------------------------------


def test_recall_at_k_hit_exactly_at_position_k_counts():
    retrieved = [chunk(0, 1), chunk(1, 2), chunk(100, 200)]  # gold hit is 3rd
    assert recall_at_k(retrieved, [GOLD], k=3) == 1.0


def test_recall_at_k_hit_just_past_k_does_not_count():
    retrieved = [chunk(0, 1), chunk(1, 2), chunk(100, 200)]  # gold hit is 3rd
    assert recall_at_k(retrieved, [GOLD], k=2) == 0.0


def test_recall_at_k_no_hit_anywhere_is_zero():
    retrieved = [chunk(0, 1), chunk(1, 2)]
    assert recall_at_k(retrieved, [GOLD], k=10) == 0.0


def test_recall_at_k_matches_any_of_multiple_gold_spans():
    other_gold = Span("doc1", 500, 600)
    retrieved = [chunk(0, 1), chunk(500, 600)]
    assert recall_at_k(retrieved, [GOLD, other_gold], k=2) == 1.0


# --- mrr ---------------------------------------------------------------------


def test_mrr_hit_at_rank_1():
    retrieved = [chunk(100, 200), chunk(0, 1)]
    assert mrr(retrieved, [GOLD]) == 1.0


def test_mrr_hit_at_rank_3():
    retrieved = [chunk(0, 1), chunk(1, 2), chunk(100, 200)]
    assert mrr(retrieved, [GOLD]) == 1.0 / 3


def test_mrr_no_hit_is_zero():
    retrieved = [chunk(0, 1), chunk(1, 2)]
    assert mrr(retrieved, [GOLD]) == 0.0


# --- context_precision --------------------------------------------------------


def test_context_precision_half_of_top_k_relevant():
    retrieved = [chunk(100, 200), chunk(0, 1), chunk(1, 2), chunk(2, 3)]
    assert context_precision(retrieved, [GOLD], k=4) == 0.25


def test_context_precision_empty_top_k_is_zero():
    assert context_precision([], [GOLD], k=5) == 0.0


def test_context_precision_ignores_chunks_past_k():
    # only the first 2 are considered; the 3rd (relevant) chunk is past k
    retrieved = [chunk(0, 1), chunk(1, 2), chunk(100, 200)]
    assert context_precision(retrieved, [GOLD], k=2) == 0.0


# --- score: end-to-end aggregate over a tiny fake gold set -------------------


def test_score_aggregates_across_records():
    gold_records = [
        {"qid": "q1", "gold_spans": [{"doc_id": "doc1", "char_start": 100, "char_end": 200}]},
        {"qid": "q2", "gold_spans": [{"doc_id": "doc1", "char_start": 100, "char_end": 200}]},
    ]
    retrieved_by_qid = {
        "q1": [chunk(100, 200)],  # hit at rank 1
        "q2": [chunk(0, 1), chunk(1, 2)],  # no hit
    }
    result = score(retrieved_by_qid, gold_records, k5=5, k10=10)
    assert result["recall@5"] == 0.5
    assert result["mrr"] == 0.5
    assert result["ctx_prec"] == 0.5  # (1/1 + 0/2) / 2
