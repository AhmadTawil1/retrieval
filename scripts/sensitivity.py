"""Robustness of the frontier-delta result, computed from the existing sweep
records — no new measurement.

Two questions the Day-6 analysis did not answer, both raised in review:

1. Does the frontier delta depend on the latency statistic? The headline uses
   p95. If the same delta appears under p50, it is not an artifact of the tail.
2. How much host noise would it take to erase the delta? Recomputing the
   frontier under epsilon-dominance --- A dominates B only if it is faster by
   more than epsilon --- answers this directly: the epsilon at which the two
   cards' frontiers become identical is the noise level the result tolerates.

Usage:
    uv run python scripts/sensitivity.py data/a100.jsonl data/l4.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Latency margins (ms) at which to recompute frontiers. 0 reproduces the
# headline result; the upper end is far beyond any plausible host jitter on a
# ~5,500 ms cell.
EPSILONS = (0, 10, 25, 50, 75, 100, 150, 200, 300)

# The pair whose swap is reported as the mechanism in the Results section.
SWAP_PAIR = ("sha1:93ce410a3c36", "sha1:0ca6aa10c184")


def load(path: Path) -> dict[str, dict]:
    return {
        r["config_id"]: r
        for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if r["run"]["status"] == "ok"
    }


def frontier(records: dict[str, dict], latency_key: str, eps: float = 0.0) -> set[str]:
    """Pareto frontier maximising recall@5, minimising latency.

    `eps` widens what counts as domination: B is only dominated if some A is at
    least as accurate and faster by more than `eps` ms. At eps=0 this is the
    ordinary frontier used in the Results section.
    """
    pts = {
        cid: (r["quality"]["recall@5"], r["cost"][latency_key])
        for cid, r in records.items()
    }
    members = set()
    for cid, (rec, lat) in pts.items():
        dominated = any(
            rec2 >= rec and lat2 <= lat - eps and (rec2 > rec or lat2 < lat - eps)
            for cid2, (rec2, lat2) in pts.items()
            if cid2 != cid
        )
        if not dominated:
            members.add(cid)
    return members


def statistic_agreement(a: dict, b: dict) -> list[str]:
    lines = ["## Frontier delta under p50 vs p95", ""]
    lines.append("| statistic | A100 | L4 | shared | frontiers identical? | reported swap present? |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("p95_ms", "p50_ms"):
        fa, fb = frontier(a, key), frontier(b, key)
        x, y = SWAP_PAIR
        swap = (x in fa and x not in fb) and (y in fb and y not in fa)
        lines.append(
            f"| {key} | {len(fa)} | {len(fb)} | {len(fa & fb)} | "
            f"{'yes' if fa == fb else 'no'} | {'yes' if swap else 'no'} |"
        )
    return lines


def epsilon_sweep(a: dict, b: dict, key: str = "p95_ms") -> list[str]:
    lines = ["", "## Frontier delta under epsilon-dominance (p95)", ""]
    lines.append("| epsilon (ms) | A100 | L4 | shared | frontiers identical? |")
    lines.append("|---|---|---|---|---|")
    for eps in EPSILONS:
        fa, fb = frontier(a, key, eps), frontier(b, key, eps)
        lines.append(
            f"| {eps} | {len(fa)} | {len(fb)} | {len(fa & fb)} | {'yes' if fa == fb else 'no'} |"
        )
    return lines


def movers_under(a: dict, b: dict, key: str) -> list[str]:
    fa, fb = frontier(a, key), frontier(b, key)
    lines = [f"", f"## Configs on exactly one card's frontier ({key})", ""]
    lines.append("| config_id | chunk | overlap | embed | top_k | reranker | recall@5 | card |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cid in sorted(fa ^ fb):
        c = a[cid]["config"]
        lines.append(
            f"| {cid} | {c['chunk_size']} | {c['overlap']} | {c['embed_model']} | "
            f"{c['top_k']} | {c['reranker']} | {a[cid]['quality']['recall@5']:.3f} | "
            f"{'A100 only' if cid in fa else 'L4 only'} |"
        )
    return lines


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(a_path), load(b_path)
    shared = set(a) & set(b)
    a = {k: v for k, v in a.items() if k in shared}
    b = {k: v for k, v in b.items() if k in shared}

    out = [f"# Sensitivity analysis — {a_path.name} vs {b_path.name}", ""]
    out += [f"{len(shared)} configurations present on both cards.", ""]
    out += statistic_agreement(a, b)
    out += epsilon_sweep(a, b)
    out += movers_under(a, b, "p95_ms")
    out += movers_under(a, b, "p50_ms")

    text = "\n".join(out) + "\n"
    Path("results/sensitivity.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
