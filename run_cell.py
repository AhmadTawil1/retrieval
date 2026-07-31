"""One config in, one JSON-line record out — quality, cost, and provenance in
a single invocation, matching the schema in M01-RETRIEVAL.md §3 exactly.

Usage:
  uv run python run_cell.py <config_id> [--n-queries 30] [--warmup 3] [--repeats 3]

`--n-queries/--warmup/--repeats` default to the real protocol (30/3/3, 90
timed measurements per cell). Override with smaller values for a fast local
smoke test on a CPU dev box — the real numbers are what days 4-5 use on the
rented cards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml

import bench
import chunker
import embed as embed_module
import eval as eval_module
import pipeline
import provenance
from pipeline import Config
from relevance import Span
from store import Store

INDEX_CACHE_DIR = Path(".cache/index")
STAGES = ("embed", "search", "rerank", "generate")


def _index_key(cfg: Config) -> str:
    return f"{cfg.chunk_size}_{cfg.overlap}_{cfg.embed_model}"


def get_or_build_index(cfg: Config, corpus_dir: Path = Path("corpus")) -> tuple[Store, float]:
    """Cached on (chunk_size, overlap, embed_model) — the three knobs that
    actually change the index (M01-RETRIEVAL.md §4.2). Returns (store,
    build_seconds); build_seconds is 0.0 on a cache hit."""
    cache_path = INDEX_CACHE_DIR / _index_key(cfg)
    if (cache_path / "index.faiss").exists():
        return Store.load(cache_path), 0.0

    t0 = time.perf_counter()
    chunks = chunker.chunk_corpus(corpus_dir, size=cfg.chunk_size, overlap=cfg.overlap)
    vecs = embed_module.embed([c.text for c in chunks], cfg.embed_model)
    store = Store.build(chunks, vecs)
    build_s = time.perf_counter() - t0
    store.save(cache_path)
    return store, build_s


def load_config(config_id: str, grid_path: Path = Path("configs/grid.yaml")) -> Config:
    grid = yaml.safe_load(grid_path.read_text())
    for cell in grid["cells"]:
        if cell["config_id"] == config_id:
            return Config(
                chunk_size=cell["chunk_size"],
                overlap=cell["overlap"],
                embed_model=cell["embed_model"],
                top_k=cell["top_k"],
                reranker=cell["reranker"],
            )
    raise ValueError(f"no cell with config_id={config_id!r} in {grid_path}")


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_cell(
    cfg: Config,
    gold_records: list[dict],
    n_queries: int = 30,
    warmup: int = 3,
    repeats: int = 3,
    corpus_dir: Path = Path("corpus"),
    gold_path: Path = Path("data/gold.jsonl"),
) -> dict:
    cid = pipeline.config_id(cfg)
    queries = gold_records[:n_queries]
    config_dict = dataclasses.asdict(cfg)

    try:
        store, build_s = get_or_build_index(cfg, corpus_dir)
        bench.reset_vram()

        for rec in queries[:warmup]:
            hits = pipeline.retrieve(rec["question"], store, cfg)
            pipeline.generate(rec["question"], hits)

        stage_ms: dict[str, list[float]] = {s: [] for s in STAGES}
        generate_tokens: list[int] = []
        retrieved_by_qid: dict[str, list[Span]] = {}

        for rec in queries:
            hits: list = []
            for _ in range(repeats):
                bench.sync()
                t0 = time.perf_counter()
                q_vec = pipeline.embed_query(rec["question"], cfg)
                bench.sync()
                stage_ms["embed"].append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                hits = pipeline.search(q_vec, store, cfg)
                bench.sync()
                stage_ms["search"].append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                hits = pipeline.rerank(rec["question"], hits, cfg)[: cfg.top_k]
                bench.sync()
                stage_ms["rerank"].append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                _, n_tokens = pipeline.generate(rec["question"], hits)
                bench.sync()
                stage_ms["generate"].append((time.perf_counter() - t0) * 1000)
                generate_tokens.append(n_tokens)

            retrieved_by_qid[rec["qid"]] = [Span(h.doc_id, h.char_start, h.char_end) for h in hits]

        end_to_end = [sum(vals) for vals in zip(*(stage_ms[s] for s in STAGES))]
        quality = eval_module.score(retrieved_by_qid, queries)
        total_generate_s = sum(stage_ms["generate"]) / 1000
        tok_per_s = sum(generate_tokens) / total_generate_s if total_generate_s > 0 else 0.0

        return {
            "config": config_dict,
            "config_id": cid,
            "quality": quality,
            "cost": {
                "p50_ms": bench.p50(end_to_end),
                "p95_ms": bench.p95(end_to_end),
                "stages_p50_ms": {s: bench.p50(stage_ms[s]) for s in STAGES},
                "peak_vram_mb": bench.peak_vram_mb(),
                "tok_per_s": tok_per_s,
                "index_build_s": build_s,
            },
            "run": {
                "n_queries": len(queries),
                "repeats": repeats,
                "warmup": warmup,
                "seed": pipeline.SEED,
                "status": "ok",
            },
            "prov": provenance.stamp(corpus_dir, gold_path),
        }
    except Exception as exc:
        if not _is_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "config": config_dict,
            "config_id": cid,
            "run": {"n_queries": len(queries), "repeats": repeats, "warmup": warmup, "seed": pipeline.SEED, "status": "oom"},
            "prov": provenance.stamp(corpus_dir, gold_path),
        }


REQUIRED_TOP_KEYS = ("config", "config_id", "run", "prov")
REQUIRED_OK_KEYS = ("quality", "cost")
REQUIRED_QUALITY_KEYS = ("recall@5", "recall@10", "mrr", "ctx_prec")
REQUIRED_COST_KEYS = ("p50_ms", "p95_ms", "stages_p50_ms", "peak_vram_mb", "tok_per_s", "index_build_s")


def validate_record(record: dict) -> list[str]:
    """Structural check against the §3 schema — not a correctness check on
    the numbers themselves. Returns a list of problems; empty means well-formed."""
    errors = []
    for key in REQUIRED_TOP_KEYS:
        if key not in record:
            errors.append(f"missing top-level key {key!r}")

    status = record.get("run", {}).get("status")
    if status == "ok":
        for key in REQUIRED_OK_KEYS:
            if key not in record:
                errors.append(f"status=ok but missing {key!r}")
        for key in REQUIRED_QUALITY_KEYS:
            if key not in record.get("quality", {}):
                errors.append(f"quality missing {key!r}")
        for key in REQUIRED_COST_KEYS:
            if key not in record.get("cost", {}):
                errors.append(f"cost missing {key!r}")
        for stage in STAGES:
            if stage not in record.get("cost", {}).get("stages_p50_ms", {}):
                errors.append(f"cost.stages_p50_ms missing stage {stage!r}")
    elif status == "oom":
        pass  # quality/cost are legitimately absent — a refusal is a result, not a gap
    else:
        errors.append(f"run.status is {status!r}, expected 'ok' or 'oom'")

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_id")
    parser.add_argument("--gold", type=Path, default=Path("data/gold.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config_id)
    gold_records = eval_module.load_gold(args.gold)
    record = run_cell(cfg, gold_records, args.n_queries, args.warmup, args.repeats, args.corpus, args.gold)
    print(json.dumps(record))
