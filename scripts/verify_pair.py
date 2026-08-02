"""Asserts two cards' result files describe the identical measurement: the
same set of config_ids, and a single agreed corpus_sha and gold_sha shared by
both. If this fails, the two cards were not run on the same ground and the
frontier comparison Measurement 01 rests on is void (M01-RETRIEVAL.md Day 5:
"If this fails, the comparison is void and nothing downstream matters.").

Also renders a refusals table from each file's `status:"oom"` records — every
cell a card declined, with its config and error, never hand-typed.

Usage:
  uv run python scripts/verify_pair.py results/a100.jsonl results/l4.jsonl
  uv run python scripts/verify_pair.py results/a100.jsonl results/l4.jsonl --refusals results/refusals.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verify_grid import load_records

PROV_FIELDS_MUST_MATCH = ("corpus_sha", "gold_sha")


def config_id_set(records: list[dict]) -> set[str]:
    return {r["config_id"] for r in records}


def prov_field_values(records: list[dict], field: str) -> set[str]:
    return {r["prov"][field] for r in records}


def check_pair(a_records: list[dict], b_records: list[dict], a_label: str, b_label: str) -> list[str]:
    """Returns a list of problems; empty means the two cards measured the
    identical thing and a frontier comparison between them is valid."""
    errors: list[str] = []

    a_ids, b_ids = config_id_set(a_records), config_id_set(b_records)
    only_a, only_b = a_ids - b_ids, b_ids - a_ids
    if only_a:
        errors.append(f"{len(only_a)} config_id(s) in {a_label} but not {b_label}: {sorted(only_a)[:5]}")
    if only_b:
        errors.append(f"{len(only_b)} config_id(s) in {b_label} but not {a_label}: {sorted(only_b)[:5]}")

    for label, records in ((a_label, a_records), (b_label, b_records)):
        for field in PROV_FIELDS_MUST_MATCH:
            values = prov_field_values(records, field)
            if len(values) > 1:
                errors.append(f"{label}: {field} is not even consistent within itself: {sorted(values)}")

    for field in PROV_FIELDS_MUST_MATCH:
        a_vals, b_vals = prov_field_values(a_records, field), prov_field_values(b_records, field)
        if a_vals and b_vals and a_vals != b_vals:
            errors.append(f"{field} differs between cards: {a_label}={sorted(a_vals)} {b_label}={sorted(b_vals)}")

    return errors


def oom_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("run", {}).get("status") == "oom"]


def render_refusals_md(cards: list[tuple[str, list[dict]]]) -> str:
    """cards: [(label, records), ...]. One table per card that has any OOM cell."""
    lines = ["# Refusals\n"]
    any_oom = False
    for label, records in cards:
        ooms = oom_records(records)
        if not ooms:
            continue
        any_oom = True
        lines.append(f"## {label}\n")
        lines.append("| config_id | chunk_size | overlap | embed_model | top_k | reranker | error |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in ooms:
            cfg = r["config"]
            error = r.get("run", {}).get("error", "").replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {r['config_id']} | {cfg['chunk_size']} | {cfg['overlap']} | "
                f"{cfg['embed_model']} | {cfg['top_k']} | {cfg['reranker']} | {error} |"
            )
        lines.append("")
    if not any_oom:
        lines.append("No refusals recorded.\n")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="e.g. results/a100.jsonl")
    parser.add_argument("b", type=Path, help="e.g. results/l4.jsonl")
    parser.add_argument("--refusals", type=Path, help="write the OOM/refusal table here, e.g. results/refusals.md")
    args = parser.parse_args()

    a_records = load_records(args.a)
    b_records = load_records(args.b)
    errors = check_pair(a_records, b_records, str(args.a), str(args.b))

    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
    else:
        print(f"{args.a} and {args.b}: identical config_id sets, identical corpus_sha, identical gold_sha. GREEN.")

    if args.refusals:
        args.refusals.parent.mkdir(parents=True, exist_ok=True)
        args.refusals.write_text(
            render_refusals_md([(str(args.a), a_records), (str(args.b), b_records)]), encoding="utf-8"
        )
        print(f"refusals written to {args.refusals}")

    raise SystemExit(1 if errors else 0)
