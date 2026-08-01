"""Fast unit tests for bench.py and run_cell.py's record schema — no model
loading, no GPU. An off-by-one in a percentile or a missing schema key should
surface here in seconds, not after a 90-minute sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from retrieval.bench import p50, p95, percentile
from run_cell import validate_record

# --- bench.py: percentile / p50 / p95 ---------------------------------------


def test_percentile_p50_of_odd_count_is_the_middle_value():
    assert p50([10, 30, 20]) == 20  # sorted: 10,20,30 -> middle


def test_percentile_p95_of_ten_values_is_nearest_rank():
    values = list(range(1, 11))  # 1..10
    # nearest-rank: index = round(0.95 * 9) = round(8.55) = 9 -> value 10
    assert p95(values) == 10


def test_percentile_single_value_returns_that_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_empty_raises():
    try:
        percentile([], 50)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_p50_and_p95_agree_on_uniform_values():
    values = [7.0] * 20
    assert p50(values) == 7.0
    assert p95(values) == 7.0


# --- run_cell.py: validate_record --------------------------------------------


def _ok_record() -> dict:
    return {
        "config": {"chunk_size": 512, "overlap": 0.15, "embed_model": "small", "top_k": 5, "reranker": "off"},
        "config_id": "sha1:deadbeef0000",
        "quality": {"recall@5": 0.8, "recall@10": 0.9, "mrr": 0.7, "ctx_prec": 0.3},
        "cost": {
            "p50_ms": 100.0,
            "p95_ms": 150.0,
            "stages_p50_ms": {"embed": 5.0, "search": 2.0, "rerank": 0.0, "generate": 90.0},
            "peak_vram_mb": 0.0,
            "tok_per_s": 20.0,
            "index_build_s": 3.0,
        },
        "run": {"n_queries": 30, "repeats": 3, "warmup": 3, "seed": 1337, "status": "ok"},
        "prov": {"gpu": "none (CPU)", "driver": None, "cuda": None, "torch": "x", "sbert": "x", "faiss": "x",
                  "git_sha": "abc", "corpus_sha": "abc", "gold_sha": "abc"},
    }


def test_validate_record_accepts_well_formed_ok_record():
    assert validate_record(_ok_record()) == []


def _oom_record(error="CUDA out of memory") -> dict:
    return {
        "config": {"chunk_size": 1024, "overlap": 0.15, "embed_model": "base", "top_k": 20, "reranker": "cross-encoder"},
        "config_id": "sha1:oom000000000",
        "run": {"n_queries": 30, "repeats": 3, "warmup": 3, "seed": 1337, "status": "oom", "error": error},
        "prov": {"gpu": "L4"},
    }


def test_validate_record_accepts_oom_record_without_quality_or_cost():
    assert validate_record(_oom_record()) == []


def test_validate_record_flags_oom_record_missing_error():
    record = _oom_record(error="")
    del record["run"]["error"]
    errors = validate_record(record)
    assert any("error" in e for e in errors)


def test_validate_record_flags_oom_record_with_empty_error():
    errors = validate_record(_oom_record(error=""))
    assert any("error" in e for e in errors)


def test_validate_record_flags_missing_top_level_key():
    record = _ok_record()
    del record["prov"]
    errors = validate_record(record)
    assert any("prov" in e for e in errors)


def test_validate_record_flags_missing_quality_metric():
    record = _ok_record()
    del record["quality"]["mrr"]
    errors = validate_record(record)
    assert any("mrr" in e for e in errors)


def test_validate_record_flags_missing_stage():
    record = _ok_record()
    del record["cost"]["stages_p50_ms"]["rerank"]
    errors = validate_record(record)
    assert any("rerank" in e for e in errors)


def test_validate_record_flags_unknown_status():
    record = _ok_record()
    record["run"]["status"] = "weird"
    errors = validate_record(record)
    assert any("status" in e for e in errors)
