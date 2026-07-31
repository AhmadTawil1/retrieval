"""Drives the full 96-cell grid through run_cell.py for one card, writing one
JSON line per cell to `--output` (e.g. results/a100.jsonl or results/l4.jsonl).

Resumable: on (re)start, any config_id already present in `--output` is
skipped, so killing the process and rerunning the same command picks up
where it left off rather than redoing completed cells (M01-RETRIEVAL.md
Day 4 — "resumability is load-bearing"). Cells are grouped and iterated by
index key (chunk_size, overlap, embed_model) so the 12 distinct indices are
each built once, not up to 96 times (§4.2) — this also happens to match
`configs/grid.yaml`'s natural ordering, but is enforced explicitly here
rather than relied on by accident.

Usage:
  uv run python scripts/sweep.py --output results/a100.jsonl
  uv run python scripts/sweep.py --output results/l4.jsonl

For a fast local dry run (this dev box has no GPU — real timings only come
from the rented card):
  uv run python scripts/sweep.py --output /tmp/dry.jsonl --n-queries 2 --warmup 1 --repeats 1 --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from retrieval.pipeline import Config
from retrieval import eval as eval_module
from run_cell import run_cell


def load_grid(grid_path: Path) -> list[dict]:
    return yaml.safe_load(grid_path.read_text())["cells"]


def group_cells_by_index(cells: list[dict]) -> list[tuple[tuple, list[dict]]]:
    """Groups cells by (chunk_size, overlap, embed_model), preserving the
    order each index key first appears in — explicit, not assumed from
    `grid.yaml`'s generation order."""
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        key = (cell["chunk_size"], cell["overlap"], cell["embed_model"])
        groups.setdefault(key, []).append(cell)
    return list(groups.items())


def load_completed_config_ids(output_path: Path) -> set[str]:
    """Config IDs already present in `output_path`. Tolerant of a truncated
    final line (the process was killed mid-write) — that line is simply not
    counted as done, so its cell gets safely redone rather than silently
    treated as complete."""
    if not output_path.exists():
        return set()
    completed = set()
    for line in output_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = record.get("config_id")
        if cid:
            completed.add(cid)
    return completed


def append_record(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run_sweep(
    output_path: Path,
    grid_path: Path = Path("configs/grid.yaml"),
    gold_path: Path = Path("data/gold.jsonl"),
    corpus_dir: Path = Path("corpus"),
    n_queries: int = 30,
    warmup: int = 3,
    repeats: int = 3,
    limit: int | None = None,
) -> None:
    cells = load_grid(grid_path)
    grouped = group_cells_by_index(cells)
    gold_records = eval_module.load_gold(gold_path)
    completed = load_completed_config_ids(output_path)

    print(f"{len(cells)} cells across {len(grouped)} indices; {len(completed)} already done in {output_path}")

    run_count = 0
    for index_key, index_cells in grouped:
        for cell in index_cells:
            if limit is not None and run_count >= limit:
                print(f"--limit {limit} reached, stopping")
                return
            cfg = Config(
                chunk_size=cell["chunk_size"],
                overlap=cell["overlap"],
                embed_model=cell["embed_model"],
                top_k=cell["top_k"],
                reranker=cell["reranker"],
            )
            cid = cell["config_id"]
            if cid in completed:
                continue

            print(f"running {cid} index={index_key} ...", flush=True)
            t0 = time.perf_counter()
            record = run_cell(cfg, gold_records, n_queries, warmup, repeats, corpus_dir, gold_path)
            elapsed = time.perf_counter() - t0
            append_record(output_path, record)
            completed.add(cid)
            run_count += 1

            status = record["run"]["status"]
            print(f"[{len(completed)}/{len(cells)} total] {cid} status={status} elapsed={elapsed:.1f}s")

    print(f"sweep done: {run_count} cells run this session, {len(completed)}/{len(cells)} total complete in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="e.g. results/a100.jsonl or results/l4.jsonl")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid.yaml"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="stop after N newly-run cells (dry runs)")
    args = parser.parse_args()

    run_sweep(
        args.output, args.grid, args.gold, args.corpus,
        args.n_queries, args.warmup, args.repeats, args.limit,
    )
