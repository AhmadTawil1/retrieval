"""Fast unit tests for verify_pair.py's pure-logic helpers — no model
loading, no GPU, no real result files (M01-RETRIEVAL.md Day 5: 'if this
fails, the comparison is void and nothing downstream matters')."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from verify_pair import check_pair, oom_records, render_refusals_md


def _record(cid, corpus_sha="c1", gold_sha="g1", status="ok", error=None,
            chunk_size=512, overlap=0.15, embed_model="small", top_k=5, reranker="off"):
    run = {"status": status}
    if status == "oom":
        run["error"] = error or "CUDA out of memory"
    return {
        "config": {"chunk_size": chunk_size, "overlap": overlap, "embed_model": embed_model,
                    "top_k": top_k, "reranker": reranker},
        "config_id": cid,
        "run": run,
        "prov": {"corpus_sha": corpus_sha, "gold_sha": gold_sha},
    }


# --- check_pair ---------------------------------------------------------------


def test_check_pair_passes_when_ids_and_shas_match():
    a = [_record("a"), _record("b")]
    b = [_record("a"), _record("b")]
    assert check_pair(a, b, "A", "B") == []


def test_check_pair_flags_ids_only_in_a():
    a = [_record("a"), _record("b")]
    b = [_record("a")]
    errors = check_pair(a, b, "A", "B")
    assert any("b" in e and "A but not B" in e for e in errors)


def test_check_pair_flags_ids_only_in_b():
    a = [_record("a")]
    b = [_record("a"), _record("z")]
    errors = check_pair(a, b, "A", "B")
    assert any("z" in e and "B but not A" in e for e in errors)


def test_check_pair_flags_corpus_sha_inconsistent_within_one_file():
    a = [_record("a", corpus_sha="c1"), _record("b", corpus_sha="c2")]
    b = [_record("a", corpus_sha="c1"), _record("b", corpus_sha="c1")]
    errors = check_pair(a, b, "A", "B")
    assert any("not even consistent within itself" in e and "corpus_sha" in e for e in errors)


def test_check_pair_flags_corpus_sha_differing_between_cards():
    a = [_record("a", corpus_sha="c1")]
    b = [_record("a", corpus_sha="c2")]
    errors = check_pair(a, b, "A", "B")
    assert any("corpus_sha differs between cards" in e for e in errors)


def test_check_pair_flags_gold_sha_differing_between_cards():
    a = [_record("a", gold_sha="g1")]
    b = [_record("a", gold_sha="g2")]
    errors = check_pair(a, b, "A", "B")
    assert any("gold_sha differs between cards" in e for e in errors)


# --- oom_records / render_refusals_md -----------------------------------------


def test_oom_records_filters_to_oom_status_only():
    records = [_record("a", status="ok"), _record("b", status="oom"), _record("c", status="ok")]
    assert [r["config_id"] for r in oom_records(records)] == ["b"]


def test_render_refusals_md_reports_no_refusals_when_none_present():
    md = render_refusals_md([("A", [_record("a", status="ok")])])
    assert "No refusals recorded." in md


def test_render_refusals_md_includes_config_and_error():
    records = [_record("a", status="oom", error="CUDA out of memory", chunk_size=1024, top_k=20, reranker="cross-encoder")]
    md = render_refusals_md([("L4", records)])
    assert "## L4" in md
    assert "1024" in md and "20" in md and "cross-encoder" in md
    assert "CUDA out of memory" in md


def test_render_refusals_md_skips_cards_with_no_oom():
    records_ok = [("A", [_record("a", status="ok")])]
    records_mixed = records_ok + [("B", [_record("b", status="oom")])]
    md = render_refusals_md(records_mixed)
    assert "## A" not in md
    assert "## B" in md
