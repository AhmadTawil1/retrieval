"""Fast unit tests for analysis.py's pure-logic helpers — every number in
PAPER.md §4 traces back to these, so they get the same scrutiny pareto.py's
frontier function did (Day 6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from analysis import frontier_delta_table, knob_breakdown, latency_kendall_tau, reranker_attribution


def _record(cid, chunk_size=256, overlap=0.0, embed_model="small", top_k=5, reranker="off",
            recall5=0.8, p95=1000.0, stages=None):
    return {
        "config_id": cid,
        "config": {"chunk_size": chunk_size, "overlap": overlap, "embed_model": embed_model,
                    "top_k": top_k, "reranker": reranker},
        "run": {"status": "ok"},
        "quality": {"recall@5": recall5},
        "cost": {"p95_ms": p95, "stages_p50_ms": stages or {"embed": 10.0, "search": 1.0, "rerank": 0.0, "generate": 900.0}},
    }


# --- frontier_delta_table ------------------------------------------------------


def test_frontier_delta_table_labels_exclusive_members_on_each_side():
    # On A100: "winner" (0.9, 100) dominates "runner_up" (0.5, 200) -> A100 frontier = {winner}
    # On L4: "runner_up" (0.9, 100) ties winner's recall and beats its p95 (0.9, 300) -> L4 frontier = {runner_up}
    a = [_record("winner", recall5=0.9, p95=100), _record("runner_up", recall5=0.5, p95=200)]
    b = [_record("winner", recall5=0.9, p95=300), _record("runner_up", recall5=0.9, p95=100)]
    rows = frontier_delta_table(a, b)
    by_id = {r["config_id"]: r for r in rows}
    assert by_id["winner"]["delta"] == "A100 only"
    assert by_id["runner_up"]["delta"] == "L4 only"


def test_frontier_delta_table_labels_shared_member_as_both():
    a = [_record("only", recall5=0.8, p95=100)]
    b = [_record("only", recall5=0.8, p95=100)]
    rows = frontier_delta_table(a, b)
    assert rows[0]["delta"] == "both"


def test_frontier_delta_table_only_includes_frontier_relevant_configs():
    a = [_record("a", recall5=0.9, p95=100), _record("dominated_a", recall5=0.5, p95=500)]
    b = [_record("a", recall5=0.9, p95=100), _record("dominated_a", recall5=0.5, p95=500)]
    rows = frontier_delta_table(a, b)
    ids = {r["config_id"] for r in rows}
    assert ids == {"a"}  # dominated_a is off both frontiers, excluded


# --- knob_breakdown -------------------------------------------------------------


def test_knob_breakdown_detects_unchanged_values():
    a = [_record("a", chunk_size=256, recall5=0.9, p95=100)]
    b = [_record("a", chunk_size=256, recall5=0.9, p95=100)]
    breakdown = knob_breakdown(a, b)
    assert breakdown["chunk_size"] == {"a100": [256], "l4": [256], "changed": False}


def test_knob_breakdown_detects_changed_values():
    # a100 frontier uses chunk_size 256; l4 frontier (different winning config) uses 1024
    a = [_record("small_chunk", chunk_size=256, recall5=0.9, p95=100), _record("big_chunk", chunk_size=1024, recall5=0.5, p95=500)]
    b = [_record("small_chunk", chunk_size=256, recall5=0.5, p95=500), _record("big_chunk", chunk_size=1024, recall5=0.9, p95=100)]
    breakdown = knob_breakdown(a, b)
    assert breakdown["chunk_size"]["a100"] == [256]
    assert breakdown["chunk_size"]["l4"] == [1024]
    assert breakdown["chunk_size"]["changed"] is True


# --- latency_kendall_tau ---------------------------------------------------------


def test_latency_kendall_tau_perfect_agreement_is_one():
    a = [_record("x", p95=100), _record("y", p95=200), _record("z", p95=300)]
    b = [_record("x", p95=110), _record("y", p95=220), _record("z", p95=330)]
    assert latency_kendall_tau(a, b) == 1.0


def test_latency_kendall_tau_reversed_order_is_minus_one():
    a = [_record("x", p95=100), _record("y", p95=200), _record("z", p95=300)]
    b = [_record("x", p95=330), _record("y", p95=220), _record("z", p95=110)]
    assert latency_kendall_tau(a, b) == -1.0


# --- reranker_attribution ---------------------------------------------------------


def test_reranker_attribution_computes_paired_recall_gain():
    a = [
        _record("off1", reranker="off", top_k=5, recall5=0.7, p95=100, stages={"embed": 1, "search": 1, "rerank": 0, "generate": 98}),
        _record("on1", reranker="cross-encoder", top_k=5, recall5=0.9, p95=150, stages={"embed": 1, "search": 1, "rerank": 40, "generate": 108}),
    ]
    b = [
        _record("off1", reranker="off", top_k=5, recall5=0.7, p95=110, stages={"embed": 1, "search": 1, "rerank": 0, "generate": 108}),
        _record("on1", reranker="cross-encoder", top_k=5, recall5=0.9, p95=200, stages={"embed": 1, "search": 1, "rerank": 80, "generate": 118}),
    ]
    attribution = reranker_attribution(a, b)
    assert attribution["mean_recall_gain_a100"] == pytest.approx(0.2)
    assert attribution["mean_recall_gain_l4"] == pytest.approx(0.2)
    assert attribution["rerank_stage_ratio_l4_over_a100"] == pytest.approx(2.0)  # 80/40
