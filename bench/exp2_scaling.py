"""
Experiment 2: coordination cost and scaling under a thread-count
sweep.

Run (from the directory containing bench/):
    python -m bench.exp2_scaling --out results/
    python -m bench.exp2_scaling --out results/ --workload montecarlo
"""

# Libraries
import argparse
import functools
import operator
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, envinfo, workloads
from .workloads import (chunk_bounds, compute_item, mc_inside_count,
                        partial_sum)
from .runner import Runner, with_memory


# Configuration variables
N_ITEMS = config.N_ITEMS
BATCH = config.BATCH


class CountingLock:
    """
    threading.Lock wrapper counting acquisitions (verify pass only;
    never used in timed runs — the count increment itself would
    perturb the measurement).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0

    def __enter__(self):
        self._lock.acquire()
        self.count += 1  # safe: incremented while holding the lock
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _dispatch(worker, n_threads: int) -> list:
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(worker, range(n_threads)))


# --------------------------------------------------------------------
# Synthetic workload conditions. Each returns {"result", "reference",
# "user_lock_acquisitions", ...}.  counting=True builds the verify
# variant with CountingLock and empirical counts.
# --------------------------------------------------------------------
_ref_cache: dict = {}


def _reference_synthetic() -> float:
    if "syn" not in _ref_cache:
        _ref_cache["syn"] = sum(
            compute_item(i) for i in range(N_ITEMS))
    return _ref_cache["syn"]


def _reference_mc(n_threads: int) -> int:
    """Deterministic correct total for the given chunking and seeds."""
    key = ("mc", n_threads)
    if key not in _ref_cache:
        total = 0
        for tid in range(n_threads):
            start, end = chunk_bounds(tid, n_threads, N_ITEMS)
            total += mc_inside_count(end - start, seed=tid)
        _ref_cache[key] = total
    return _ref_cache[key]


def make_synthetic_conditions(counting: bool = False) -> dict:
    Lock = CountingLock if counting else threading.Lock

    def fp_reduce(n: int) -> dict:
        """Matched-kernel FP condition: pure partial_sum + reduce.
        Differs from imp_threadlocal ONLY in the combination step
        (functools.reduce vs an imperative loop)."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            return partial_sum(start, end)
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return {"result": result, "reference": _reference_synthetic(),
                "user_lock_acquisitions": 0}

    def fp_sum_genexpr(n: int) -> dict:
        """Idiomatic-FP variant: sum() over a generator expression.
        Kept as a separate condition to quantify the kernel-style
        effect under free-threading, cleanly attributed by contrast
        with the matched-kernel fp_reduce."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            return sum(compute_item(i)
                       for i in range(start, end))
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return {"result": result, "reference": _reference_synthetic(),
                "user_lock_acquisitions": 0}

    def imp_threadlocal(n: int) -> dict:
        """Imperative thread-local baseline. The worker is IDENTICAL
        to fp_reduce's worker (same partial_sum call); the conditions
        differ only in the combination step (imperative loop here,
        functools.reduce there)."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            return partial_sum(start, end)
        partials = _dispatch(worker, n)
        total = 0.0
        for p in partials:
            total += p
        return {"result": total, "reference": _reference_synthetic(),
                "user_lock_acquisitions": 0}

    def coarse_lock(n: int) -> dict:
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            partial = partial_sum(start, end)
            with lock:
                shared[0] += partial
        _dispatch(worker, n)
        out = {"result": shared[0],
               "reference": _reference_synthetic(),
               "user_lock_acquisitions": n}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def batched_lock(n: int) -> dict:
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            batch = 0.0
            pending = 0
            for i in range(start, end):
                batch += compute_item(i)
                pending += 1
                if pending == BATCH:
                    with lock:
                        shared[0] += batch
                    batch, pending = 0.0, 0
            if pending:
                with lock:
                    shared[0] += batch
        _dispatch(worker, n)
        acq = sum(
            -(-((chunk_bounds(t, n, N_ITEMS)[1]
                 - chunk_bounds(t, n, N_ITEMS)[0])) // BATCH)
            for t in range(n))
        out = {"result": shared[0],
               "reference": _reference_synthetic(),
               "user_lock_acquisitions": acq}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def fine_lock(n: int) -> dict:
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            for i in range(start, end):
                val = compute_item(i)   # outside the lock
                with lock:
                    shared[0] += val              # guarded add only
        _dispatch(worker, n)
        out = {"result": shared[0],
               "reference": _reference_synthetic(),
               "user_lock_acquisitions": N_ITEMS}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def queue_agg(n: int) -> dict:
        q: queue.SimpleQueue = queue.SimpleQueue()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            local = 0.0
            for i in range(start, end):
                local += compute_item(i)
            q.put(local)
        _dispatch(worker, n)
        total = 0.0
        for _ in range(n):
            total += q.get()
        return {"result": total, "reference": _reference_synthetic(),
                "user_lock_acquisitions": 0}

    def no_lock(n: int) -> dict:
        shared = [0.0]

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            for i in range(start, end):
                val = compute_item(i)
                shared[0] += val                  # non-atomic RMW
        _dispatch(worker, n)
        return {"result": shared[0],
                "reference": _reference_synthetic(),
                "user_lock_acquisitions": 0}

    return {
        "fp_reduce": fp_reduce,
        "fp_sum_genexpr": fp_sum_genexpr,
        "imp_threadlocal": imp_threadlocal,
        "coarse_lock": coarse_lock,
        "batched_lock": batched_lock,
        "fine_lock": fine_lock,
        "queue_agg": queue_agg,
        "no_lock": no_lock,
    }


# ---------------------------------------------------------------------
# Monte Carlo workload conditions (integer inside-counts; deterministic
# reference per thread count).
# ---------------------------------------------------------------------
def make_mc_conditions(counting: bool = False) -> dict:
    Lock = CountingLock if counting else threading.Lock

    def fp_reduce(n: int) -> dict:
        def worker(tid: int) -> int:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            return mc_inside_count(end - start, seed=tid)
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0)
        return {"result": result, "expected": _reference_mc(n),
                "user_lock_acquisitions": 0}

    def imp_threadlocal(n: int) -> dict:
        def worker(tid: int) -> int:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            return mc_inside_count(end - start, seed=tid)
        partials = _dispatch(worker, n)
        total = 0
        for p in partials:
            total += p
        return {"result": total, "expected": _reference_mc(n),
                "user_lock_acquisitions": 0}

    def coarse_lock(n: int) -> dict:
        shared = [0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            inside = mc_inside_count(end - start, seed=tid)
            with lock:
                shared[0] += inside
        _dispatch(worker, n)
        out = {"result": shared[0], "expected": _reference_mc(n),
               "user_lock_acquisitions": n}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def fine_lock(n: int) -> dict:
        import random as _random
        shared = [0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            rng = _random.Random(tid)
            for _ in range(end - start):
                inside = int(rng.random() ** 2
                             + rng.random() ** 2 <= 1.0)
                with lock:
                    shared[0] += inside
        _dispatch(worker, n)
        out = {"result": shared[0], "expected": _reference_mc(n),
               "user_lock_acquisitions": N_ITEMS}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def no_lock(n: int) -> dict:
        import random as _random
        shared = [0]

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, N_ITEMS)
            rng = _random.Random(tid)
            for _ in range(end - start):
                inside = int(rng.random() ** 2
                             + rng.random() ** 2 <= 1.0)
                shared[0] += inside
        _dispatch(worker, n)
        return {"result": shared[0], "expected": _reference_mc(n),
                "user_lock_acquisitions": 0}

    return {
        "fp_reduce": fp_reduce,
        "imp_threadlocal": imp_threadlocal,
        "coarse_lock": coarse_lock,
        "fine_lock": fine_lock,
        "no_lock": no_lock,
    }


# Main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 2: scaling with fair baselines")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workload", choices=["synthetic", "montecarlo"],
                    default="synthetic")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--runs", type=int, default=config.R_MEASURED)
    ap.add_argument("--warmup", type=int, default=config.R_WARMUP)
    ap.add_argument("--max-threads", type=int, default=None)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the counting-lock verification pass")
    ap.add_argument("--skip-memory", action="store_true",
                    help="skip the tracemalloc memory pass")
    args = ap.parse_args()

    envinfo.assert_free_threading()

    maker = (make_mc_conditions if args.workload == "montecarlo"
             else make_synthetic_conditions)
    tc = config.thread_counts(args.max_threads)

    runner = Runner("exp2_scaling", args.out, workload=args.workload,
                    seed=args.seed, r_warmup=args.warmup,
                    r_measured=args.runs)
    runner.sweep(maker(counting=False), tc)
    if not args.skip_verify:
        runner.single_pass(maker(counting=True), tc, phase="verify")
    if not args.skip_memory:
        runner.single_pass(with_memory(maker(counting=False)), tc,
                           phase="memory")

# Main guard
if __name__ == "__main__":
    main()
