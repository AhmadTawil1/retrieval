"""Pareto frontier membership over the two axes Hypothesis 1 is about:
maximise recall@5, minimise p95 latency (M01-RETRIEVAL.md Day 6).

Only `status:"ok"` cells are eligible — an OOM'd cell has no recall/p95 pair
to plot, so it cannot be a frontier member by construction (§3.8), not by a
judgment call made after seeing the data.
"""

from __future__ import annotations


def pareto_frontier(records: list[dict]) -> set[str]:
    """config_ids of the non-dominated cells: no other eligible cell has
    both recall@5 >= this one's and p95_ms <= this one's, with at least one
    of those strict. Ties (identical recall@5 and p95_ms) are all frontier
    members — neither dominates the other."""
    eligible = [r for r in records if r.get("run", {}).get("status") == "ok"]

    def dominates(a: dict, b: dict) -> bool:
        a_recall, a_p95 = a["quality"]["recall@5"], a["cost"]["p95_ms"]
        b_recall, b_p95 = b["quality"]["recall@5"], b["cost"]["p95_ms"]
        at_least_as_good = a_recall >= b_recall and a_p95 <= b_p95
        strictly_better = a_recall > b_recall or a_p95 < b_p95
        return at_least_as_good and strictly_better

    frontier = set()
    for r in eligible:
        if not any(dominates(other, r) for other in eligible if other["config_id"] != r["config_id"]):
            frontier.add(r["config_id"])
    return frontier


def kendall_tau_b(pairs: list[tuple[float, float]]) -> float:
    """Kendall's tau-b over paired (a_value, b_value) observations — how much
    two cards' latency-based rankings of the same configs agree. +1 fully
    concordant, -1 fully reversed, 0 no relationship. Ties on one side only
    are excluded from the concordant/discordant count and folded into the
    tau-b denominator (the standard tie-corrected form); a pair tied on both
    sides is dropped entirely, since it is neither concordant nor discordant."""
    n = len(pairs)
    n0 = n * (n - 1) // 2
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        xi, yi = pairs[i]
        for j in range(i + 1, n):
            xj, yj = pairs[j]
            dx, dy = xi - xj, yi - yj
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((n0 - tied_x) * (n0 - tied_y)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom
