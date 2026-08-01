"""Fast unit tests for sweep.py/verify_grid.py's pure-logic helpers — no
model loading, no GPU, no real grid/gold files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sweep import group_cells_by_index, load_completed_config_ids
from verify_grid import check_completeness

# --- sweep.group_cells_by_index ---------------------------------------------


def _cell(cid, chunk_size, overlap, embed_model, top_k=5, reranker="off"):
    return {
        "config_id": cid,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embed_model": embed_model,
        "top_k": top_k,
        "reranker": reranker,
    }


def test_group_cells_by_index_groups_matching_index_keys_together():
    cells = [
        _cell("c1", 256, 0.0, "small"),
        _cell("c2", 512, 0.0, "small"),
        _cell("c3", 256, 0.0, "small"),  # same index as c1
    ]
    groups = group_cells_by_index(cells)
    keys = [k for k, _ in groups]
    assert keys == [(256, 0.0, "small"), (512, 0.0, "small")]
    assert [c["config_id"] for c in dict(groups)[(256, 0.0, "small")]] == ["c1", "c3"]


def test_group_cells_by_index_preserves_first_seen_order():
    cells = [_cell("a", 1024, 0.15, "base"), _cell("b", 256, 0.0, "small"), _cell("c", 1024, 0.15, "base")]
    groups = group_cells_by_index(cells)
    keys = [k for k, _ in groups]
    assert keys == [(1024, 0.15, "base"), (256, 0.0, "small")]


# --- sweep.load_completed_config_ids ----------------------------------------


def test_load_completed_config_ids_empty_when_file_absent(tmp_path):
    assert load_completed_config_ids(tmp_path / "nope.jsonl") == set()


def test_load_completed_config_ids_reads_valid_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"config_id": "a"}\n{"config_id": "b"}\n')
    assert load_completed_config_ids(p) == {"a", "b"}


def test_load_completed_config_ids_ignores_truncated_last_line(tmp_path):
    """Simulates the process being killed mid-write — the incomplete last
    line must not count as a completed cell, so it gets safely redone."""
    p = tmp_path / "out.jsonl"
    p.write_text('{"config_id": "a"}\n{"config_id": "b", "trunc')
    assert load_completed_config_ids(p) == {"a"}


def test_load_completed_config_ids_ignores_blank_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"config_id": "a"}\n\n{"config_id": "b"}\n')
    assert load_completed_config_ids(p) == {"a", "b"}


# --- verify_grid.check_completeness -----------------------------------------


def _ok_record(cid, status="ok"):
    if status == "ok":
        return {
            "config": {"chunk_size": 512, "overlap": 0.15, "embed_model": "small", "top_k": 5, "reranker": "off"},
            "config_id": cid,
            "quality": {"recall@5": 0.8, "recall@10": 0.9, "mrr": 0.7, "ctx_prec": 0.3},
            "cost": {
                "p50_ms": 100.0, "p95_ms": 150.0,
                "stages_p50_ms": {"embed": 5.0, "search": 2.0, "rerank": 0.0, "generate": 90.0},
                "peak_vram_mb": 0.0, "tok_per_s": 20.0, "index_build_s": 3.0,
            },
            "run": {"n_queries": 30, "repeats": 3, "warmup": 3, "seed": 1337, "status": "ok"},
            "prov": {"gpu": "A100", "driver": "x", "cuda": "x", "torch": "x", "sbert": "x", "faiss": "x",
                      "git_sha": "abc", "corpus_sha": "abc", "gold_sha": "abc"},
        }
    run = {"n_queries": 30, "repeats": 3, "warmup": 3, "seed": 1337, "status": status}
    if status == "oom":
        run["error"] = "CUDA out of memory"
    return {
        "config": {"chunk_size": 512, "overlap": 0.15, "embed_model": "small", "top_k": 5, "reranker": "off"},
        "config_id": cid,
        "run": run,
        "prov": {"gpu": "A100"},
    }


def test_check_completeness_passes_when_all_expected_ids_present_once():
    expected = {"a", "b", "c"}
    records = [_ok_record("a"), _ok_record("b"), _ok_record("c", status="oom")]
    assert check_completeness(expected, records) == []


def test_check_completeness_flags_missing_ids():
    expected = {"a", "b", "c"}
    records = [_ok_record("a"), _ok_record("b")]
    errors = check_completeness(expected, records)
    assert any("missing" in e for e in errors)


def test_check_completeness_flags_duplicates():
    expected = {"a", "b"}
    records = [_ok_record("a"), _ok_record("a"), _ok_record("b")]
    errors = check_completeness(expected, records)
    assert any("duplicate" in e for e in errors)


def test_check_completeness_flags_bad_status():
    expected = {"a"}
    records = [_ok_record("a", status="error")]
    errors = check_completeness(expected, records)
    assert any("status" in e for e in errors)


def test_check_completeness_flags_extra_unexpected_ids():
    expected = {"a"}
    records = [_ok_record("a"), _ok_record("z")]
    errors = check_completeness(expected, records)
    assert any("not in grid.yaml" in e for e in errors)
