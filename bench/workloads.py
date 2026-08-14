"""
CPU-bound pure computation kernels shared by experiments 2 and 3.

Monte Carlo representation: workers return the raw *inside-circle count* for their chunk.  
Counts are integers, so (a) their sum is an exact monoid
with no floating-point associativity caveat, and (b) for fixed seeds
the correct total is deterministic — which lets Experiment 2 measure
CONCURRENCY loss for the unsynchronised condition exactly, separated
from ordinary Monte Carlo sampling error.
"""

# Libraries
import math
import random


def compute_item(i: int) -> float:
    """Pure CPU-bound kernel: sqrt(i * log(i + 1))."""
    return math.sqrt(i * math.log(i + 1.0))


def mc_inside_count(n_samples: int, seed: int) -> int:
    """
    Count random points inside the unit circle (pure, deterministic
    for a given (n_samples, seed)).
    """
    rng = random.Random(seed)
    return sum(
        1 for _ in range(n_samples)
        if rng.random() ** 2 + rng.random() ** 2 <= 1.0
    )


def partial_sum(start: int, end: int) -> float:
    """
    Pure function: explicit-loop partial sum of compute_item over
    [start, end). Shared by ALL value-returning and lock-based
    conditions in Experiment 2 (synthetic), so conditions differ ONLY
    in how partials are communicated and combined.
    """
    local = 0.0
    for i in range(start, end):
        local += compute_item(i)
    return local


def compute_chunk_stats(start: int, end: int) -> dict:
    """Summary statistics of compute_item over [start, end). Pure."""
    if end <= start:
        return {"sum": 0.0, "min": 0.0, "max": 0.0,
                "count": 0, "mean": 0.0}
    values = [compute_item(i) for i in range(start, end)]
    total = sum(values)
    return {
        "sum": total,
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "mean": total / len(values),
    }


def chunk_bounds(tid: int, n_threads: int,
                 n_items: int) -> tuple[int, int]:
    """Identical chunking for every condition (per-thread work
    equivalence): ceil-sized chunks in thread-id order."""
    chunk = math.ceil(n_items / n_threads)
    start = tid * chunk
    end = min(start + chunk, n_items)
    return start, end
