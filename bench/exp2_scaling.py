"""
Experiment 2: coordination cost and scaling under a thread-count
sweep.

Kernel-style conditions (synthetic workload). The idiomatic
sum(genexpr) differs from the explicit-loop partial_sum in more than
style: built-in sum() applies Neumaier compensated summation to floats
in a C double accumulator and consumes the generator from C, while
partial_sum is naive addition of Python floats driven by a bytecode
for loop.

Numerical axis (exactly controlled). The conditions fall into two
bit-identity groups (asserted in tests/test_conditions.py):
  naive        fp_reduce (= imp_threadlocal's worker), fp_reduce_genexpr
  compensated  imp_loop_compensated, fp_sum_genexpr, fp_sum_map,
               fp_sum_listcomp
Within a group every per-chunk partial is bit-for-bit identical, so
fp_reduce_genexpr and imp_loop_compensated are controls for the
NUMERICAL axis only.

Runtime. Only these pairs differ in exactly one runtime factor:
  fp_sum_genexpr  vs fp_sum_map            generator-frame overhead
  fp_sum_map      vs fp_sum_listcomp       list materialisation
  fp_reduce       vs imp_threadlocal       combination step
  imp_threadlocal vs imp_loop_compensated  Neumaier compensation
                                           written in Python bytecode
These are multi-factor contrasts, reported as such and not attributed
to one cause:
  fp_reduce         vs fp_sum_genexpr   iteration driver, generator
                                        frame, summation algorithm,
                                        accumulator representation
  fp_reduce         vs fp_reduce_genexpr  iteration driver, generator
                                        frame, per-item Python-level
                                        operator.add call
  fp_reduce_genexpr vs fp_sum_genexpr   summation algorithm and
                                        accumulator representation
                                        (not separable with the
                                        standard library)
  fp_reduce         vs fp_sum_map       iteration driver, summation
                                        algorithm, accumulator
                                        representation

Numerical error. The synthetic reference is the exactly-rounded sum
(math.fsum over the full range). sum() is unsuitable as a reference
because it is itself compensated, which biases the reported error of
the naive conditions downward relative to the compensated ones;
error_pct is therefore comparable only across datasets measured
against the same reference. Every synthetic condition also reports
abs_err_vs_fsum and rel_err_vs_fsum in the aux column.

Work size. --n-items (default config.N_ITEMS) sets the total work;
it is threaded through the condition factories, and both reference
caches are keyed on the value used. It is recorded in env.json.

Timing scope: each condition creates its ThreadPoolExecutor inside the
condition function (_dispatch), so pool creation and thread spawn are
inside the timed region. This is deliberate and applies identically to
every condition (see runner.py); at small N it is a visible share of
wall time, which the larger-N run (--n-items 5000000) addresses.

Run (from the directory containing bench/):
    python -m bench.exp2_scaling --out results/
    python -m bench.exp2_scaling --out results/ --n-items 5000000 --tag n5m
    python -m bench.exp2_scaling --out results/ --workload montecarlo
    PYTHON_GIL=1 python -m bench.exp2_scaling --out results/ \
        --allow-gil --tag gil
"""

# Libraries
import argparse
import functools
import operator
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config, envinfo
from .workloads import (chunk_bounds, compute_item, exact_sum,
                        mc_inside_count, partial_sum,
                        partial_sum_compensated)
from .runner import Runner, with_memory


# Configuration variables. The work size is NOT a module constant: it is
# a parameter of the condition factories (n_items, CLI --n-items), and
# every reference cache is keyed on the value actually used.
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


def _reference_synthetic(n_items: int) -> float:
    """Exactly-rounded reference (math.fsum over the full range),
    cached per n_items."""
    key = ("syn", n_items)
    if key not in _ref_cache:
        _ref_cache[key] = exact_sum(0, n_items)
    return _ref_cache[key]


def _syn_out(result: float, lock_acq: int, n_items: int) -> dict:
    """Common synthetic result dict: runtime row fields plus the
    numerical error against the exactly-rounded reference."""
    ref = _reference_synthetic(n_items)
    err = abs(result - ref)
    return {"result": result, "reference": ref,
            "user_lock_acquisitions": lock_acq,
            "abs_err_vs_fsum": err,
            "rel_err_vs_fsum": err / abs(ref)}


def _reference_mc(n_threads: int, n_items: int) -> int:
    """Deterministic correct total for the given chunking and seeds,
    cached per (n_items, n_threads)."""
    key = ("mc", n_items, n_threads)
    if key not in _ref_cache:
        total = 0
        for tid in range(n_threads):
            start, end = chunk_bounds(tid, n_threads, n_items)
            total += mc_inside_count(end - start, seed=tid)
        _ref_cache[key] = total
    return _ref_cache[key]


def make_synthetic_conditions(counting: bool = False,
                              n_items: int | None = None) -> dict:
    """Synthetic conditions over n_items work items (default
    config.N_ITEMS, read at call time)."""
    n_items = config.N_ITEMS if n_items is None else n_items
    Lock = CountingLock if counting else threading.Lock

    def fp_reduce(n: int) -> dict:
        """Matched-kernel FP condition: pure partial_sum + reduce.
        Differs from imp_threadlocal ONLY in the combination step
        (functools.reduce vs an imperative loop)."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return partial_sum(start, end)
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return _syn_out(result, 0, n_items)

    def fp_sum_genexpr(n: int) -> dict:
        """Idiomatic FP: built-in sum() over a generator expression.
        sum() consumes the generator from C and applies Neumaier
        compensation in a C double accumulator. Versus fp_reduce it
        differs in several runtime factors at once (iteration driver,
        generator frame, summation algorithm, accumulator
        representation). Paired with fp_sum_map it isolates
        generator-frame overhead."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return sum(compute_item(i)
                       for i in range(start, end))
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return _syn_out(result, 0, n_items)

    def fp_sum_map(n: int) -> dict:
        """C-driven iteration with the same compensated accumulation as
        fp_sum_genexpr, but without a Python generator frame: map() is a
        C-level iterator. Paired with fp_sum_genexpr this isolates
        generator-frame overhead, holding the summation algorithm, the
        accumulator representation and the per-item compute_item call
        identical. Verified bit-identical in value to fp_sum_genexpr."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return sum(map(compute_item, range(start, end)))
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return _syn_out(result, 0, n_items)

    def fp_sum_listcomp(n: int) -> dict:
        """sum() over a list built by a list comprehension. Same
        compensated C accumulation as fp_sum_map; the values are first
        materialised in a list. Paired with fp_sum_map it isolates list
        materialisation. Bit-identical in value to fp_sum_genexpr and
        fp_sum_map."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return sum([compute_item(i) for i in range(start, end)])
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return _syn_out(result, 0, n_items)

    def fp_reduce_genexpr(n: int) -> dict:
        """Control for the NUMERICAL axis only:
        functools.reduce(operator.add, genexpr, 0.0) is bit-for-bit
        identical to partial_sum (naive left-to-right float addition)
        while iterating C-driven over a generator. Its runtime measures
        the cost of functools.reduce with a Python-level callable
        (a call and a new float object per item), which is its own
        result; it is NOT the cost of naive summation and must not be
        read as such."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return functools.reduce(operator.add,
                                    (compute_item(i)
                                     for i in range(start, end)),
                                    0.0)
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0.0)
        return _syn_out(result, 0, n_items)

    def imp_threadlocal(n: int) -> dict:
        """Imperative thread-local baseline. The worker is IDENTICAL
        to fp_reduce's worker (same partial_sum call); the conditions
        differ only in the combination step (imperative loop here,
        functools.reduce there)."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return partial_sum(start, end)
        partials = _dispatch(worker, n)
        total = 0.0
        for p in partials:
            total += p
        return _syn_out(total, 0, n_items)

    def imp_loop_compensated(n: int) -> dict:
        """Control for the NUMERICAL axis only: the worker calls
        partial_sum_compensated, whose per-chunk partials are
        bit-for-bit equal to sum()'s. Partials are combined with an
        imperative loop, matching imp_threadlocal. Its runtime measures
        the cost of writing Neumaier compensation in Python bytecode,
        NOT the cost of compensation as such: CPython's sum() performs
        the same algorithm in C, and fp_sum_map provides that C-level
        comparison."""
        def worker(tid: int) -> float:
            start, end = chunk_bounds(tid, n, n_items)
            return partial_sum_compensated(start, end)
        partials = _dispatch(worker, n)
        total = 0.0
        for p in partials:
            total += p
        return _syn_out(total, 0, n_items)

    def coarse_lock(n: int) -> dict:
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            partial = partial_sum(start, end)
            with lock:
                shared[0] += partial
        _dispatch(worker, n)
        out = _syn_out(shared[0], n, n_items)
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def _batched_acq(n: int) -> int:
        """Analytic lock count: ceil((end - start) / BATCH) per thread."""
        total = 0
        for t in range(n):
            start, end = chunk_bounds(t, n, n_items)
            total += -(-(end - start) // BATCH)
        return total

    def batched_lock(n: int) -> dict:
        """Matched-kernel batched locking. Each batch is accumulated by
        partial_sum itself, so the per-item inner loop is byte-identical
        to the one every other matched condition runs; the only extra
        work is one partial_sum call and one lock acquisition per
        BATCH items. Acquisitions per thread: ceil((end-start)/BATCH),
        the same as batched_lock_unmatched."""
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            for b0 in range(start, end, BATCH):
                b1 = min(b0 + BATCH, end)
                batch = partial_sum(b0, b1)   # identical inner loop
                with lock:
                    shared[0] += batch
        _dispatch(worker, n)
        out = _syn_out(shared[0], _batched_acq(n), n_items)
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def batched_lock_unmatched(n: int) -> dict:
        """Unmatched batched variant. Its per-item loop additionally
        executes `pending += 1` and `if pending == BATCH` on every item,
        which is work partial_sum does not perform, so the condition
        carries a kernel confound on top of its locking. It is included
        under an explicit name so that the size of that confound is
        measured against batched_lock rather than assumed. Lock
        acquisitions are identical to batched_lock."""
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
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
        out = _syn_out(shared[0], _batched_acq(n), n_items)
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def fine_lock(n: int) -> dict:
        """Per-item locked write to shared state. Deliberately NOT built
        on partial_sum: writing every item to the shared accumulator is
        the property under test, so its loop legitimately differs."""
        shared = [0.0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            for i in range(start, end):
                val = compute_item(i)   # outside the lock
                with lock:
                    shared[0] += val              # guarded add only
        _dispatch(worker, n)
        out = _syn_out(shared[0], n_items, n_items)
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def queue_agg(n: int) -> dict:
        """Partials are communicated through a SimpleQueue. The worker
        calls partial_sum, so its kernel is byte-identical to the other
        matched conditions; only the communication step differs."""
        q: queue.SimpleQueue = queue.SimpleQueue()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            q.put(partial_sum(start, end))
        _dispatch(worker, n)
        total = 0.0
        for _ in range(n):
            total += q.get()
        return _syn_out(total, 0, n_items)

    def no_lock(n: int) -> dict:
        """Per-item unsynchronised write to shared state (the known-unsafe
        baseline). Like fine_lock, its loop legitimately differs from
        partial_sum because the per-item shared write is the measurand."""
        shared = [0.0]

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            for i in range(start, end):
                val = compute_item(i)
                shared[0] += val    # non-atomic RMW
        _dispatch(worker, n)
        return _syn_out(shared[0], 0, n_items)

    return {
        "fp_reduce": fp_reduce,
        "fp_sum_genexpr": fp_sum_genexpr,
        "fp_sum_map": fp_sum_map,
        "fp_sum_listcomp": fp_sum_listcomp,
        "fp_reduce_genexpr": fp_reduce_genexpr,
        "imp_threadlocal": imp_threadlocal,
        "imp_loop_compensated": imp_loop_compensated,
        "coarse_lock": coarse_lock,
        "batched_lock": batched_lock,
        "batched_lock_unmatched": batched_lock_unmatched,
        "fine_lock": fine_lock,
        "queue_agg": queue_agg,
        "no_lock": no_lock,
    }


# ---------------------------------------------------------------------
# Monte Carlo workload conditions (integer inside-counts; deterministic
# reference per thread count).
# ---------------------------------------------------------------------
def make_mc_conditions(counting: bool = False,
                       n_items: int | None = None) -> dict:
    """Monte Carlo conditions over n_items samples (default
    config.N_ITEMS, read at call time)."""
    n_items = config.N_ITEMS if n_items is None else n_items
    Lock = CountingLock if counting else threading.Lock

    def fp_reduce(n: int) -> dict:
        def worker(tid: int) -> int:
            start, end = chunk_bounds(tid, n, n_items)
            return mc_inside_count(end - start, seed=tid)
        partials = _dispatch(worker, n)
        result = functools.reduce(operator.add, partials, 0)
        return {"result": result, "expected": _reference_mc(n, n_items),
                "user_lock_acquisitions": 0}

    def imp_threadlocal(n: int) -> dict:
        def worker(tid: int) -> int:
            start, end = chunk_bounds(tid, n, n_items)
            return mc_inside_count(end - start, seed=tid)
        partials = _dispatch(worker, n)
        total = 0
        for p in partials:
            total += p
        return {"result": total, "expected": _reference_mc(n, n_items),
                "user_lock_acquisitions": 0}

    def coarse_lock(n: int) -> dict:
        shared = [0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            inside = mc_inside_count(end - start, seed=tid)
            with lock:
                shared[0] += inside
        _dispatch(worker, n)
        out = {"result": shared[0], "expected": _reference_mc(n, n_items),
               "user_lock_acquisitions": n}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def fine_lock(n: int) -> dict:
        import random as _random
        shared = [0]
        lock = Lock()

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            rng = _random.Random(tid)
            for _ in range(end - start):
                inside = int(rng.random() ** 2
                             + rng.random() ** 2 <= 1.0)
                with lock:
                    shared[0] += inside
        _dispatch(worker, n)
        out = {"result": shared[0], "expected": _reference_mc(n, n_items),
               "user_lock_acquisitions": n_items}
        if counting:
            out["measured_lock_acquisitions"] = lock.count
        return out

    def no_lock(n: int) -> dict:
        import random as _random
        shared = [0]

        def worker(tid: int) -> None:
            start, end = chunk_bounds(tid, n, n_items)
            rng = _random.Random(tid)
            for _ in range(end - start):
                inside = int(rng.random() ** 2
                             + rng.random() ** 2 <= 1.0)
                shared[0] += inside
        _dispatch(worker, n)
        return {"result": shared[0], "expected": _reference_mc(n, n_items),
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
    ap.add_argument("--n-items", type=int, default=config.N_ITEMS,
                    help="total work items (default config.N_ITEMS = "
                         f"{config.N_ITEMS:,}); recorded in env.json")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the counting-lock verification pass")
    ap.add_argument("--skip-memory", action="store_true",
                    help="skip the tracemalloc memory pass")
    ap.add_argument("--allow-gil", action="store_true",
                    help="permit a GIL-enabled baseline run (free-threaded "
                         "build with PYTHON_GIL=1); prints a warning")
    ap.add_argument("--tag", default=None,
                    help="label appended to output file names and recorded "
                         "in env.json (use 'gil' for GIL-enabled runs)")
    args = ap.parse_args()

    envinfo.assert_free_threading(allow_gil=args.allow_gil)

    if args.n_items < 1:
        ap.error("--n-items must be >= 1")
    maker = functools.partial(
        make_mc_conditions if args.workload == "montecarlo"
        else make_synthetic_conditions, n_items=args.n_items)
    tc = config.thread_counts(args.max_threads)

    # Compute the reference values BEFORE any timed run, so the
    # sequential reference never falls inside a timed call.
    if args.workload == "montecarlo":
        for n in tc:
            _reference_mc(n, args.n_items)
    else:
        _reference_synthetic(args.n_items)

    runner = Runner("exp2_scaling", args.out, workload=args.workload,
                    seed=args.seed, r_warmup=args.warmup,
                    r_measured=args.runs, tag=args.tag,
                    extra_env={"n_items": args.n_items,
                               "batch": BATCH})
    runner.sweep(maker(counting=False), tc)
    if not args.skip_verify:
        runner.single_pass(maker(counting=True), tc, phase="verify")
    if not args.skip_memory:
        runner.single_pass(with_memory(maker(counting=False)), tc,
                           phase="memory")


# Main guard
if __name__ == "__main__":
    main()
