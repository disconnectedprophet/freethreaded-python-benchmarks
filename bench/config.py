"""
Shared configuration for benchmark suite and its analysis.

Experiment procedures' constants:
  R_MEASURED = 30    measured repetitions per condition and thread-count
                     (e.g. coarse lock with 3 threads). 
  R_WARMUP = 3       warm-up repetitions per condition and thread-count, 
                     executed before the measured block and excluded from analysis.
  DEFAULT_SEED = 42  seeds the shuffled execution order so the exact
                     interleaving is reproducible.

Experiment work sizes' constants:
  ITERATIONS = 100_000   the number of accumulation operations (e.g. additions)
                         per thread. Important for the experiment 1
                         (correctness -> race freedom).
  N_ITEMS = 500_000      the total number of work items per experiment (each 
                         thread takes {N_ITEMS / the number of threads} of work items).
                         Important for experiment 2 (coordination and scaling -> 
                         coordination.cost) and experiment 3 (structured accumulation 
                         -> Writer monad).
  BATCH = 1_000          the number of work items a thread accumulates locally 
                         before one lock flush, in the batched_lock condition of 
                         experiment 2. Each thread therefore ceil (its item count / BATCH)
                         lock acquisitions. 

The thread-count ceiling is derived from the visible CPU count at
runtime (see thread_counts), so the package runs unmodified across
machines. The ladder's step pattern, however, is not hardware-agnostic:
_LADDER samples densely (step of 2) up to 16 and sparsely above it,
because this study was run on 16-physical-core machines (GCP
n2-standard-32 and t2d-standard-16, both with SMT disabled), where the
scaling behaviour of interest lies at or below the core count and the
region above it only needs coarse confirmation. On machines with a
substantially different core count, _LADDER should be adjusted so the
dense region tracks that machine's core count.

The exact environment is captured separately by envinfo.py and stored
next to the results (`results/<amd/intel>/expN_*.env.json`), so the paper reports facts recorded at runtime.

Statistican analysis constants (prespecified, i.e. fixed before the results were examined):
  ALPHA = 0.05          significance level for all tests. A result is
                        reported as significant only below this value,
                        after correction for multiple testing.
  MARGIN_FRAC = 0.05    equivalence margin for TOST, as a fraction of
                        the reference condition's mean wall time at a
                        given thread count. A difference smaller than
                        this is treated as practically irrelevant.
                        Important for experiment 2 and experiment 3.

#Library
import os

# Procedure constants
R_MEASURED: int = 30
R_WARMUP: int = 3
DEFAULT_SEED: int = 42

# Work size constants 
ITERATIONS: int = 100_000
N_ITEMS: int = 500_000
BATCH: int = 1_000

# Statistical analysis constants
ALPHA = 0.05
MARGIN_FRAC = 0.05

# Threads ladder
_LADDER = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 28, 32, 48, 64]


def visible_cpus() -> int:
    """
    Return the number of logical CPUs this process may run on.

    Counts logical CPUs, i.e. SMT/hyper-threading siblings count separately.

    On platforms providing ``os.sched_getaffinity`` (Linux, some BSDs) the
    result reflects the process's CPU affinity mask, so restrictions imposed
    by ``taskset``, ``sched_setaffinity`` or a cgroup *cpuset* are taken into
    account. Elsewhere it falls back to the system-wide CPU count, which
    ignores any such restriction.

    Returns at least 1.
    """
    try: # Linux
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux fallback
        return os.cpu_count() or 1


def thread_counts(max_threads: int | None = None) -> list[int]:
    """
    Thread ladder for the sweep: the _LADDER steps up to the logical CPU count,
    always including the CPU count itself as the top rung.

    Arguments:
        max_threads: Override the ceiling (defaults to visible CPUs).

    Returns:
        Sorted list of thread counts, e.g. 16 CPUs ->
        [1, 2, 4, 6, 8, 10, 12, 14, 16].
    """
    ceiling = max_threads or visible_cpus()
    counts = [_ for _ in _LADDER if _ <= ceiling]
    if ceiling not in counts:
        counts.append(ceiling)
    return sorted(counts)
