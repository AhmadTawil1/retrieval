"""recall@k, MRR, context precision — pure functions over retrieved chunks plus
a gold record, no pipeline dependency, so they're unit-testable on their own
(tests/test_eval.py) independent of any actual retrieval run.

Also owns loading, validating, and hashing the gold set (data/gold.jsonl),
since scoring and the gold set are tightly coupled and Day 2's file list
doesn't call for a separate module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .relevance import DEFAULT_THRESHOLD, Span, is_relevant


def recall_at_k(retrieved: list[Span], gold_spans: list[Span], k: int, thresh: float = DEFAULT_THRESHOLD) -> float:
    """1.0 if any of the top-k retrieved chunks is relevant to any gold span, else 0.0."""
    top_k = retrieved[:k]
    hit = any(is_relevant(r, g, thresh) for r in top_k for g in gold_spans)
    return 1.0 if hit else 0.0


def mrr(retrieved: list[Span], gold_spans: list[Span], thresh: float = DEFAULT_THRESHOLD) -> float:
    """Reciprocal rank (1-indexed) of the first retrieved chunk relevant to any gold span."""
    for rank, r in enumerate(retrieved, start=1):
        if any(is_relevant(r, g, thresh) for g in gold_spans):
            return 1.0 / rank
    return 0.0


def context_precision(retrieved: list[Span], gold_spans: list[Span], k: int, thresh: float = DEFAULT_THRESHOLD) -> float:
    """Fraction of the top-k retrieved chunks relevant to some gold span."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if any(is_relevant(r, g, thresh) for g in gold_spans))
    return relevant / len(top_k)


def score(
    retrieved_by_qid: dict[str, list[Span]],
    gold_records: list[dict],
    k5: int = 5,
    k10: int = 10,
    thresh: float = DEFAULT_THRESHOLD,
) -> dict:
    """The `quality` block of the per-cell record schema (M01-RETRIEVAL.md §3),
    averaged over every gold record. `retrieved_by_qid[qid]` must already be
    rank-ordered (best first)."""
    if not gold_records:
        raise ValueError("gold_records is empty")

    recalls5, recalls10, mrrs, precisions = [], [], [], []
    for rec in gold_records:
        gold_spans = [Span(g["doc_id"], g["char_start"], g["char_end"]) for g in rec["gold_spans"]]
        retrieved = retrieved_by_qid[rec["qid"]]
        recalls5.append(recall_at_k(retrieved, gold_spans, k5, thresh))
        recalls10.append(recall_at_k(retrieved, gold_spans, k10, thresh))
        mrrs.append(mrr(retrieved, gold_spans, thresh))
        precisions.append(context_precision(retrieved, gold_spans, k5, thresh))

    n = len(gold_records)
    return {
        "recall@5": sum(recalls5) / n,
        "recall@10": sum(recalls10) / n,
        "mrr": sum(mrrs) / n,
        "ctx_prec": sum(precisions) / n,
    }


# --- Gold set loading / validation / hashing --------------------------------

REQUIRED_QUESTION_COUNT = 30


def load_gold(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def validate_gold(records: list[dict], corpus_dir: Path) -> list[str]:
    """Returns a list of problems; empty means the gold set is well-formed.
    Does not judge question quality — only structural validity."""
    errors: list[str] = []
    doc_ids = {p.stem for p in corpus_dir.glob("*.pdf")}
    qids_seen: set[str] = set()

    if len(records) != REQUIRED_QUESTION_COUNT:
        errors.append(f"expected {REQUIRED_QUESTION_COUNT} records, found {len(records)}")

    for i, rec in enumerate(records):
        where = f"record {i} (qid={rec.get('qid', '?')})"
        for field in ("qid", "question", "gold_spans"):
            if field not in rec:
                errors.append(f"{where}: missing field {field!r}")
        qid = rec.get("qid")
        if qid in qids_seen:
            errors.append(f"{where}: duplicate qid")
        qids_seen.add(qid)

        spans = rec.get("gold_spans", [])
        if not spans:
            errors.append(f"{where}: gold_spans is empty")
        for j, g in enumerate(spans):
            span_where = f"{where} gold_spans[{j}]"
            for field in ("doc_id", "char_start", "char_end"):
                if field not in g:
                    errors.append(f"{span_where}: missing field {field!r}")
                    continue
            doc_id = g.get("doc_id")
            if doc_id is not None and doc_id not in doc_ids:
                errors.append(f"{span_where}: doc_id {doc_id!r} not found in {corpus_dir}")
            start, end = g.get("char_start"), g.get("char_end")
            if start is not None and end is not None and not (0 <= start < end):
                errors.append(f"{span_where}: char_start={start} char_end={end} is not a valid non-empty span")

    return errors


def gold_sha(path: Path) -> str:
    """SHA1 over the raw file bytes. Any later edit invalidates both cards."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    import sys

    gold_path = Path("data/gold.jsonl")
    records = load_gold(gold_path)
    errors = validate_gold(records, Path("corpus"))
    if errors:
        print(f"{len(errors)} problem(s) in {gold_path}:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"{len(records)} gold records OK. gold_sha={gold_sha(gold_path)}")
