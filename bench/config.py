"""
Shared configuration for benchmark suite.

Hardware-agnostic: the thread ladder is derived from the visible CPU
count at runtime, so the same package runs unmodified on any machine
(GCP t2d-standard-16, c3-highcpu-44 with SMT off, or anything else).
The exact environment is captured separately by envinfo.py and stored
next to the results, so the paper reports facts recorded at runtime.

Protocol constants:
  R_MEASURED = 30  measured repetitions per (condition, thread-count)
                   cell. 
  R_WARMUP = 3     warm-up repetitions per cell, executed before the
                   measured block and excluded from analysis.
  DEFAULT_SEED = 42 seeds the shuffled execution order so the exact
                    interleaving is reproducible.
"""

#Library
import os

# Constants
R_MEASURED: int = 30
R_WARMUP: int = 3
DEFAULT_SEED: int = 42

# Work sizes 
ITERATIONS_EXP1: int = 100_000 # accumulation operations per thread
N_ITEMS: int = 500_000 # total work items, experiments 2 and 3

_LADDER = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 28, 32, 48, 64]


def visible_cpus() -> int:
    """Number of CPUs available to this process (affinity-aware)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux fallback
        return os.cpu_count() or 1


def thread_counts(max_threads: int | None = None) -> list[int]:
    """
    Thread ladder for the sweep: standard steps up to the CPU count,
    always including the CPU count itself as the top rung.

    Args:
        max_threads: Override the ceiling (defaults to visible CPUs).

    Returns:
        Sorted list of thread counts, e.g. 16 CPUs ->
        [1, 2, 4, 6, 8, 10, 12, 14, 16].
    """
    ceiling = max_threads or visible_cpus()
    counts = [n for n in _LADDER if n <= ceiling]
    if ceiling not in counts:
        counts.append(ceiling)
    return sorted(counts)
