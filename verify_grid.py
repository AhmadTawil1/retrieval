"""Run before closing the machine (M01-RETRIEVAL.md Day 4): asserts a card's
result file has all 96 distinct config_ids from configs/grid.yaml, no
duplicates, and no status outside {ok, oom} — a status of "error" or anything
unexpected means a cell partially failed and needs attention before the
machine is released.

Usage:
  uv run python verify_grid.py results/a100.jsonl
  uv run python verify_grid.py results/a100.jsonl --sanity   # print the day-4 sanity summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from run_cell import validate_record

ALLOWED_STATUSES = {"ok", "oom"}


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def expected_config_ids(grid_path: Path) -> set[str]:
    grid = yaml.safe_load(grid_path.read_text())
    return {cell["config_id"] for cell in grid["cells"]}


def check_completeness(expected: set[str], records: list[dict]) -> list[str]:
    """Returns a list of problems; empty means the file is complete and clean."""
    errors: list[str] = []

    present_ids = [r.get("config_id") for r in records]
    present_set = set(present_ids)

    missing = expected - present_set
    if missing:
        errors.append(f"{len(missing)} config_id(s) missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    extra = present_set - expected
    if extra:
        errors.append(f"{len(extra)} config_id(s) present but not in grid.yaml: {sorted(extra)[:5]}")

    seen = set()
    duplicates = set()
    for cid in present_ids:
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate config_id(s): {sorted(duplicates)[:5]}")

    for i, r in enumerate(records):
        status = r.get("run", {}).get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"record {i} (config_id={r.get('config_id')}): unexpected status {status!r}")
        schema_errors = validate_record(r)
        for e in schema_errors:
            errors.append(f"record {i} (config_id={r.get('config_id')}): {e}")

    return errors


def print_sanity_summary(records: list[dict]) -> None:
    """Non-blocking eyeball check (M01-RETRIEVAL.md Day 4): recall should
    rise with top-k, context precision should fall, rerank latency should be
    non-trivial at k=20. This does not fail verify_grid — whether a pattern
    looks inverted enough to stop and fix is a human judgment call on the
    real data, not an automated assertion here."""
    ok_records = [r for r in records if r.get("run", {}).get("status") == "ok"]
    if not ok_records:
        print("no status=ok records to summarize")
        return

    print("\n--- sanity summary (status=ok records only) ---")
    by_top_k: dict[int, list[dict]] = {}
    for r in ok_records:
        by_top_k.setdefault(r["config"]["top_k"], []).append(r)

    print("recall@5 / ctx_prec by top_k (should rise / should fall):")
    for k in sorted(by_top_k):
        rs = by_top_k[k]
        recall = sum(r["quality"]["recall@5"] for r in rs) / len(rs)
        prec = sum(r["quality"]["ctx_prec"] for r in rs) / len(rs)
        print(f"  top_k={k:>3}: recall@5={recall:.3f}  ctx_prec={prec:.3f}  (n={len(rs)})")

    rerank_on = [r for r in ok_records if r["config"]["reranker"] == "cross-encoder" and r["config"]["top_k"] == 20]
    if rerank_on:
        mean_rerank_ms = sum(r["cost"]["stages_p50_ms"]["rerank"] for r in rerank_on) / len(rerank_on)
        print(f"\nmean rerank stage p50 at top_k=20, reranker=on: {mean_rerank_ms:.1f} ms (should be non-trivial)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="e.g. results/a100.jsonl")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid.yaml"))
    parser.add_argument("--sanity", action="store_true", help="also print the non-blocking sanity summary")
    args = parser.parse_args()

    records = load_records(args.results)
    expected = expected_config_ids(args.grid)
    errors = check_completeness(expected, records)

    if errors:
        print(f"{len(errors)} problem(s) in {args.results}:")
        for e in errors:
            print(" -", e)
    else:
        print(f"{args.results}: {len(records)} records, all {len(expected)} config_ids present, no duplicates, no bad status. GREEN.")

    if args.sanity:
        print_sanity_summary(records)

    raise SystemExit(1 if errors else 0)
