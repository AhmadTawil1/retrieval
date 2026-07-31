"""Generic timing/percentile/VRAM harness — the shared instrument REFERENCE.md
Rule 3 calls for ("one harness, three studies"), even though only Measurement
01 uses it today. `bench.py` and `provenance.py` were meant to already exist
from Block 0; Block 0 was skipped, so both are built fresh here on Day 3.
"""

from __future__ import annotations

import time

import torch


def sync() -> None:
    """Must bracket every timed GPU call, or the timing measures kernel
    launch, not kernel completion — the numbers would be fiction."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation, no extra dependency."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def p50(values: list[float]) -> float:
    return percentile(values, 50)


def p95(values: list[float]) -> float:
    return percentile(values, 95)


def reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def timed_repeats(fn, warmup: int = 3, repeats: int = 3) -> list[float]:
    """Runs fn() `warmup` times (discarded) then `repeats` times, returning
    wall-clock durations in milliseconds for the kept runs, each bracketed
    by sync(). Convenience wrapper for a single-stage callable; run_cell.py
    times embed/search/rerank/generate individually instead of using this."""
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        durations.append((time.perf_counter() - t0) * 1000)
    return durations
