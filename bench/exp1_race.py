"""
Experiment 1: correctness of accumulation strategies under
free-threading - with matched algorithms.

Conditions:

  imp_racy  shared int, unsynchronised `+= 1` per item.
            The known-broken baseline. Loss is the measurand.
  imp_locked  shared int, one lock acquisition per item.
              Correct but serialised textbook fix (reference point).
  imp_threadlocal  local accumulator, partial value returned. Main
                   thread combines with an imperative loop.  Safe,
                   imperative, no locks.
  fp_fold  per-thread pure fold (functools.reduce over the
           item range). Partials combined by a second reduce.
           Safe, functional, no locks.
  disjoint_slot local accumulation, then a single write into a
                pre-allocated per-thread slot of a shared list.

Note on per-item cost: instruction-level equivalence across paradigms
is impossible in Python (a fold's lambda call is not a `+=` statement),
hence Experiment 1 draws correctness conclusions only. Runtime is
recorded for transparency but scheduling/performance claims belong to
Experiment 2.

Run (from the directory containing bench/):
    python -m bench.exp1_race --out results/
"""

# Libraries
import argparse
import functools
import operator
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, envinfo
from .runner import Runner

# Iterations variable
ITER = config.ITERATIONS

# Shared state for the racy / locked conditions (reset per run)
_counter: int = 0
_lock = threading.Lock()


# Workers
def _w_racy(_tid: int) -> None:
    """ITER unsynchronised increments of the shared counter."""
    global _counter
    for _ in range(ITER):
        _counter += 1


def _w_locked(_tid: int) -> None:
    """ITER increments, each guarded by the shared lock."""
    global _counter
    for _ in range(ITER):
        with _lock:
            _counter += 1


def _w_threadlocal(_tid: int) -> int:
    """ITER increments of a local variable; partial value returned."""
    local = 0
    for _ in range(ITER):
        local += 1
    return local


def _w_fold(_tid: int) -> int:
    """Pure fold over the item range; no assignment statements."""
    return functools.reduce(lambda acc, _i: acc + 1, range(ITER), 0)


# Conditions
def _dispatch(worker, n_threads: int) -> list:
    """Uniform dispatch: executor.map over thread ids."""
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(worker, range(n_threads)))


def cond_imp_racy(n_threads: int) -> dict:
    global _counter
    _counter = 0
    _dispatch(_w_racy, n_threads)
    return {"result": _counter, "expected": n_threads * ITER}


def cond_imp_locked(n_threads: int) -> dict:
    global _counter
    _counter = 0
    _dispatch(_w_locked, n_threads)
    return {"result": _counter, "expected": n_threads * ITER,
            "lock_acquisitions": n_threads * ITER}


def cond_imp_threadlocal(n_threads: int) -> dict:
    partials = _dispatch(_w_threadlocal, n_threads)
    total = 0
    for p in partials: # imperative combination
        total += p
    return {"result": total, "expected": n_threads * ITER}


def cond_fp_fold(n_threads: int) -> dict:
    partials = _dispatch(_w_fold, n_threads)
    total = functools.reduce(operator.add, partials, 0)
    return {"result": total, "expected": n_threads * ITER}


def cond_disjoint_slot(n_threads: int) -> dict:
    slots = [0] * n_threads

    def worker(tid: int) -> None:
        local = 0
        for _ in range(ITER):
            local += 1
        slots[tid] = local # single disjoint write to shared list

    _dispatch(worker, n_threads)
    total = functools.reduce(operator.add, slots, 0)
    return {"result": total, "expected": n_threads * ITER}


CONDITIONS = {
    "imp_racy": cond_imp_racy,
    "imp_locked": cond_imp_locked,
    "imp_threadlocal": cond_imp_threadlocal,
    "fp_fold": cond_fp_fold,
    "disjoint_slot": cond_disjoint_slot,
}


# Main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 1: race-condition detection with "
                    "matched algorithms")
    ap.add_argument("--out", default="results",
                    help="output directory for CSV + env.json")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--runs", type=int, default=config.R_MEASURED)
    ap.add_argument("--warmup", type=int, default=config.R_WARMUP)
    ap.add_argument("--max-threads", type=int, default=None)
    args = ap.parse_args()

    envinfo.assert_free_threading()

    runner = Runner("exp1_race", args.out, workload="counter",
                    seed=args.seed, r_warmup=args.warmup,
                    r_measured=args.runs)
    runner.sweep(CONDITIONS, config.thread_counts(args.max_threads))


# Main guard
if __name__ == "__main__":
    main()
