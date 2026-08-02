"""Day 6: computes both cards' Pareto frontiers, the frontier delta, a
knob-by-knob breakdown, Kendall tau over the latency ordering, and the
reranker's per-stage cost/benefit attribution — then renders Figure 1 and a
markdown report. Every number in PAPER.md §4 traces back to this script; none
are hand-typed (M01-RETRIEVAL.md Day 6, REFERENCE.md ground rule 5).

`plot.py` was assumed to exist from Block 0 (skipped — LOG.md Day 1/3); its
job is folded in here rather than split into a separate file with nothing
else to share it with.

Usage:
  uv run python scripts/analysis.py data/a100.jsonl data/l4.jsonl
  uv run python scripts/analysis.py data/a100.jsonl data/l4.jsonl --figure figs/fig1.pdf --report results/analysis.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from retrieval.pareto import kendall_tau_b, pareto_frontier

KNOBS = ("chunk_size", "overlap", "embed_model", "top_k", "reranker")

COLOR_A = "#2a78d6"  # categorical slot 1 (blue)
COLOR_B = "#eb6834"  # categorical slot 2 (orange)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_config_id(records: list[dict]) -> dict[str, dict]:
    return {r["config_id"]: r for r in records}


def frontier_delta_table(a_records: list[dict], b_records: list[dict]) -> list[dict]:
    """One row per config that is on EITHER card's frontier — the informative
    subset of 'every config, on-frontier per card, and the delta' (the
    remaining configs are, by construction, off both and add no information;
    the full 96-row table is reproducible from this same function against
    the union of every config_id if ever needed)."""
    a_by_id, b_by_id = by_config_id(a_records), by_config_id(b_records)
    a_frontier, b_frontier = pareto_frontier(a_records), pareto_frontier(b_records)
    relevant = sorted(a_frontier | b_frontier)

    rows = []
    for cid in relevant:
        on_a, on_b = cid in a_frontier, cid in b_frontier
        if on_a and on_b:
            delta = "both"
        elif on_a:
            delta = "A100 only"
        else:
            delta = "L4 only"
        rows.append({
            "config_id": cid,
            "config": a_by_id[cid]["config"],
            "recall@5": a_by_id[cid]["quality"]["recall@5"],
            "a100_p95_ms": a_by_id[cid]["cost"]["p95_ms"],
            "l4_p95_ms": b_by_id[cid]["cost"]["p95_ms"],
            "on_a100_frontier": on_a,
            "on_l4_frontier": on_b,
            "delta": delta,
        })
    return rows


def knob_breakdown(a_records: list[dict], b_records: list[dict]) -> dict[str, dict[str, list]]:
    """For each knob, the distinct values appearing among each card's
    frontier configs — turns 'the frontier changed' into 'which knob moved'."""
    a_by_id, b_by_id = by_config_id(a_records), by_config_id(b_records)
    a_frontier, b_frontier = pareto_frontier(a_records), pareto_frontier(b_records)

    breakdown = {}
    for knob in KNOBS:
        a_values = sorted({a_by_id[cid]["config"][knob] for cid in a_frontier}, key=str)
        b_values = sorted({b_by_id[cid]["config"][knob] for cid in b_frontier}, key=str)
        breakdown[knob] = {"a100": a_values, "l4": b_values, "changed": a_values != b_values}
    return breakdown


def latency_kendall_tau(a_records: list[dict], b_records: list[dict]) -> float:
    a_by_id, b_by_id = by_config_id(a_records), by_config_id(b_records)
    shared = sorted(set(a_by_id) & set(b_by_id))
    pairs = [(a_by_id[cid]["cost"]["p95_ms"], b_by_id[cid]["cost"]["p95_ms"]) for cid in shared]
    return kendall_tau_b(pairs)


def reranker_attribution(a_records: list[dict], b_records: list[dict]) -> dict:
    """Paired on/off recall gain from the reranker, and its stage cost on
    each card relative to the other stages — is the reranker's slowdown
    disproportionate, or in line with everything else getting slower?"""

    def paired_recall_gain(records: list[dict]) -> float:
        by_cfg = {tuple(sorted(r["config"].items())): r for r in records}
        gains = []
        for r in records:
            if r["config"]["reranker"] != "cross-encoder":
                continue
            off_cfg = dict(r["config"], reranker="off")
            off = by_cfg.get(tuple(sorted(off_cfg.items())))
            if off:
                gains.append(r["quality"]["recall@5"] - off["quality"]["recall@5"])
        return sum(gains) / len(gains)

    a_by_id, b_by_id = by_config_id(a_records), by_config_id(b_records)
    shared = sorted(set(a_by_id) & set(b_by_id))

    def stage_ratio(stage: str, ids: list[str]) -> float:
        a_vals = [a_by_id[cid]["cost"]["stages_p50_ms"][stage] for cid in ids]
        b_vals = [b_by_id[cid]["cost"]["stages_p50_ms"][stage] for cid in ids]
        return (sum(b_vals) / len(b_vals)) / (sum(a_vals) / len(a_vals))

    def overall_p95_ratio(ids: list[str]) -> float:
        a_vals = [a_by_id[cid]["cost"]["p95_ms"] for cid in ids]
        b_vals = [b_by_id[cid]["cost"]["p95_ms"] for cid in ids]
        return (sum(b_vals) / len(b_vals)) / (sum(a_vals) / len(a_vals))

    reranked_ids = [cid for cid in shared if a_by_id[cid]["config"]["reranker"] == "cross-encoder"]

    return {
        "mean_recall_gain_a100": paired_recall_gain(a_records),
        "mean_recall_gain_l4": paired_recall_gain(b_records),
        "rerank_stage_ratio_l4_over_a100": stage_ratio("rerank", reranked_ids),
        "overall_p95_ratio_l4_over_a100": overall_p95_ratio(shared),
        "embed_stage_ratio": stage_ratio("embed", shared),
        "search_stage_ratio": stage_ratio("search", shared),
        "generate_stage_ratio": stage_ratio("generate", shared),
    }


def render_figure(a_records: list[dict], b_records: list[dict], output_path: Path) -> None:
    a_by_id, b_by_id = by_config_id(a_records), by_config_id(b_records)
    a_frontier, b_frontier = pareto_frontier(a_records), pareto_frontier(b_records)
    shared_ids = sorted(set(a_by_id) & set(b_by_id))
    relevant = a_frontier | b_frontier

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for cid in shared_ids:
        if cid not in relevant:
            continue
        ax.plot(
            [a_by_id[cid]["cost"]["p95_ms"], b_by_id[cid]["cost"]["p95_ms"]],
            [a_by_id[cid]["quality"]["recall@5"], b_by_id[cid]["quality"]["recall@5"]],
            color=BASELINE, linewidth=0.9, zorder=1,
        )

    for records, frontier, color, label in ((a_records, a_frontier, COLOR_A, "A100"), (b_records, b_frontier, COLOR_B, "L4")):
        on_x = [r["cost"]["p95_ms"] for r in records if r["config_id"] in frontier]
        on_y = [r["quality"]["recall@5"] for r in records if r["config_id"] in frontier]
        off_x = [r["cost"]["p95_ms"] for r in records if r["config_id"] not in frontier]
        off_y = [r["quality"]["recall@5"] for r in records if r["config_id"] not in frontier]

        ax.scatter(off_x, off_y, s=18, color=color, alpha=0.35, edgecolors="none", zorder=2)
        ax.scatter(
            on_x, on_y, s=60, color=color, alpha=0.95,
            edgecolors=INK_PRIMARY, linewidths=0.8, zorder=3, label=label,
        )

    for cid in sorted(relevant):
        r = a_by_id[cid]
        short = cid.split(":")[1][:6]
        x = a_by_id[cid]["cost"]["p95_ms"] if cid in a_frontier else b_by_id[cid]["cost"]["p95_ms"]
        y = a_by_id[cid]["quality"]["recall@5"]
        ax.annotate(
            short, (x, y), textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color=INK_SECONDARY,
        )

    ax.set_xlabel("p95 latency (ms)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("recall@5", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Figure 1 — recall@5 vs p95 latency, A100 vs L4\n"
        "filled = on that card's Pareto frontier; thin lines trace one config across cards",
        color=INK_PRIMARY, fontsize=11, loc="left",
    )
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.grid(True, color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def render_report_md(
    frontier_rows: list[dict], breakdown: dict, tau: float, attribution: dict,
    a_path: Path, b_path: Path, n_shared: int,
) -> str:
    lines = [f"# Day 6 analysis — {a_path} vs {b_path}\n"]

    lines.append("## Frontier delta (configs on either card's frontier)\n")
    lines.append("| config_id | chunk_size | overlap | embed_model | top_k | reranker | recall@5 | A100 p95 | L4 p95 | on A100 | on L4 | delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in frontier_rows:
        c = row["config"]
        lines.append(
            f"| {row['config_id']} | {c['chunk_size']} | {c['overlap']} | {c['embed_model']} | "
            f"{c['top_k']} | {c['reranker']} | {row['recall@5']:.3f} | {row['a100_p95_ms']:.0f}ms | "
            f"{row['l4_p95_ms']:.0f}ms | {row['on_a100_frontier']} | {row['on_l4_frontier']} | {row['delta']} |"
        )

    lines.append("\n## Knob-by-knob frontier composition\n")
    lines.append("| knob | A100 frontier values | L4 frontier values | changed? |")
    lines.append("|---|---|---|---|")
    for knob, vals in breakdown.items():
        lines.append(f"| {knob} | {vals['a100']} | {vals['l4']} | {vals['changed']} |")

    lines.append(f"\n## Kendall tau (latency ordering, A100 vs L4, n={n_shared})\n")
    lines.append(f"tau_b = {tau:.4f}")

    lines.append("\n## Reranker attribution\n")
    lines.append(f"- mean recall@5 gain from reranker, A100: {attribution['mean_recall_gain_a100']:.4f}")
    lines.append(f"- mean recall@5 gain from reranker, L4: {attribution['mean_recall_gain_l4']:.4f}")
    lines.append(f"- rerank-stage p50 ratio (L4/A100): {attribution['rerank_stage_ratio_l4_over_a100']:.3f}")
    lines.append(f"- overall p95 ratio (L4/A100), all shared configs: {attribution['overall_p95_ratio_l4_over_a100']:.3f}")
    lines.append(f"- embed-stage p50 ratio (L4/A100): {attribution['embed_stage_ratio']:.3f}")
    lines.append(f"- search-stage p50 ratio (L4/A100): {attribution['search_stage_ratio']:.3f}")
    lines.append(f"- generate-stage p50 ratio (L4/A100): {attribution['generate_stage_ratio']:.3f}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="e.g. data/a100.jsonl")
    parser.add_argument("b", type=Path, help="e.g. data/l4.jsonl")
    parser.add_argument("--figure", type=Path, default=Path("figs/fig1.pdf"))
    parser.add_argument("--report", type=Path, default=Path("results/analysis.md"))
    args = parser.parse_args()

    a_records = load_records(args.a)
    b_records = load_records(args.b)

    frontier_rows = frontier_delta_table(a_records, b_records)
    breakdown = knob_breakdown(a_records, b_records)
    tau = latency_kendall_tau(a_records, b_records)
    attribution = reranker_attribution(a_records, b_records)

    n_shared = len(set(by_config_id(a_records)) & set(by_config_id(b_records)))
    render_figure(a_records, b_records, args.figure)
    report = render_report_md(frontier_rows, breakdown, tau, attribution, args.a, args.b, n_shared)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nfigure written to {args.figure} (+ .png)")
    print(f"report written to {args.report}")
