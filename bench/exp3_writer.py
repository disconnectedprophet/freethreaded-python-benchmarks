"""
Experiment 3: structured multi-field accumulation — Writer monad
versus lock, indexed, and plain-record baselines, over a pipeline-depth
sweep k (number of stages).

Pipeline. Stage 1 computes per-chunk summary statistics
(workloads.compute_chunk_stats). Stage 2 is the pure _stage2 transform
(range and coefficient of variation from the stage-1 statistics). Every
stage j >= 3 applies the pure transform _stage_next(stats, j), so stage
k is well defined for any k.
Each stage emits exactly one log entry per worker, so the final log
holds k * n entries. All four designs run the SAME stage functions
and the same entry formatting; they differ only in how per-stage
state and log entries are threaded and combined.

Depth k = 1 and k = 2 keep the workload tags "single" and "chained"
(including the k = 1 log-entry format), so earlier data remain
comparable; deeper runs are tagged "depth<k>".

Conditions (at depth k):
  lock      shared dict + shared log list + one lock; one acquisition
            per worker. Workers append their entry block in
            lock-acquisition order (nondeterministic), so the log is
            reordered into contract order afterwards, inside the timed
            region.
  indexed   exactly two pre-allocated per-thread slot arrays (final
            stats, logs), no lock, at every depth. Each worker
            accumulates its entries locally and writes once per slot.
  record    ChunkRecord(stats, log-tuple) returned per worker, manual
            combination.
  writer    Writer(stats, [entry]) for stage 1, then one .bind() per
            further stage (k - 1 binds per worker), one merge_writers
            call. bind copies the accumulated log, so per-worker log
            copying grows as O(k^2) (see writer_monad cost model).

Output contract (all designs, every depth; validated per run against a
cached sequential reference, see _validate):
  stats  list of per-worker final-stage stats dicts, ascending worker id
  log    flat list, worker-major and stage-minor

Structural proxy, reported in aux for every run:
  aux_structures      auxiliary coordination structures the design
                      allocates (shared dicts, shared lists,
                      pre-allocated arrays, locks), counted from the
                      structures the implementation actually creates
  bookkeeping_steps   explicit user-code steps that thread stage state
                      and log entries to the combined output:
                        lock     n(k+2) + 2   (entry list create + k-1
                                 appends; dict write + log append under
                                 the lock; sort + flatten)
                        indexed  n(k+2) + 1   (entry list create + k-1
                                 appends; two slot writes; flatten)
                        record   n(k+1) + 2   (entry list create + k-1
                                 appends; record construction; stats
                                 and log comprehensions)
                        writer   n*k + 1      (Writer construction +
                                 k-1 binds; merge_writers)
  bind_calls          writer only: n(k - 1)
Timed runs report these from the formulas above; the verify pass
(phase=verify) runs instrumented variants that count every site and
every lock acquisition, recording measured_* fields so the analysis
checks measured == analytic.

Timing scope: like every experiment, each condition creates its
ThreadPoolExecutor inside the condition function, so pool creation and
thread spawn are inside the timed region. This is deliberate (see
runner.py).

Run (from the directory containing bench/):
    python -m bench.exp3_writer --out results/
    python -m bench.exp3_writer --out results/ --depths 1,2,4,8,16
"""

# Libraries
import argparse
import functools
import operator
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from . import config, envinfo, workloads
from .writer_monad import Writer, merge_writers
from .runner import Runner, with_memory

# Configuration variables
N_ITEMS = config.N_ITEMS
DEFAULT_DEPTHS = "1,2,4,8,16"
DESIGNS = ("lock", "indexed", "record", "writer")


class ChunkRecord(NamedTuple):
    """
    Plain immutable record baseline: same payload as Writer,
    no monadic interface.

    Immutability is SHALLOW. The NamedTuple's fields cannot be rebound
    and its log is a tuple, but `stats` is an ordinary, mutable dict:
    code holding a ChunkRecord could still change the dict's contents.
    No code in this suite does so. The Writer is shallowly immutable in
    exactly the same sense (it does not copy its value; see
    writer_monad), so record and writer carry the same guarantee.
    """
    stats: dict
    log: tuple[str, ...]


class CountingLock:
    """threading.Lock wrapper counting acquisitions (verify pass only)."""

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


_ref_cache: dict = {}
_by_tid = operator.itemgetter(0)


def depth_tag(k: int) -> str:
    """Workload tag for depth k: 'single' (1), 'chained' (2), 'depth<k>'."""
    return {1: "single", 2: "chained"}.get(k, f"depth{k}")


def _reference_sum() -> float:
    if "sum" not in _ref_cache:
        _ref_cache["sum"] = workloads.compute_chunk_stats(
            0, N_ITEMS)["sum"]
    return _ref_cache["sum"]


# ---------------------------------------------------------------------
# Pure stage chain and log entries (shared by every design)
# ---------------------------------------------------------------------
def _stage2(stats: dict) -> dict:
    """Pure Stage-2 transform."""
    mean = stats["mean"]
    spread = stats["max"] - stats["min"]
    return {"range": spread,
            "cv": spread / mean if mean != 0.0 else 0.0,
            "sum": stats["sum"], "count": stats["count"]}


def _stage_next(stats: dict, j: int) -> dict:
    """Pure transform for stage j >= 3 (deterministic, same shape as
    _stage2's output, keeps the chunk sum so the aggregate is defined
    at every depth)."""
    return {"range": stats["range"] / j,
            "cv": stats["cv"] * (j - 1) / j,
            "sum": stats["sum"], "count": stats["count"]}


def _stage(j: int, stats: dict) -> dict:
    """Stage j >= 2 of the chain."""
    return _stage2(stats) if j == 2 else _stage_next(stats, j)


def _entry1(tid: int, start: int, end: int, s1: dict, k: int) -> str:
    """Stage-1 log entry. k = 1 keeps the original single-stage format."""
    if k == 1:
        return (f"thread={tid} slice=[{start},{end}) "
                f"sum={s1['sum']:.4f} count={s1['count']}")
    return f"s1 thread={tid} sum={s1['sum']:.4f} mean={s1['mean']:.4f}"


def _entry(tid: int, j: int, s: dict) -> str:
    """Log entry for stage j >= 2."""
    return f"s{j} thread={tid} cv={s['cv']:.4f} range={s['range']:.4f}"


def _run_stages(tid: int, start: int, end: int, k: int,
                tick=None) -> tuple[dict, list[str]]:
    """Run all k stages for one worker with plain local threading of
    state and entries (used by lock, indexed and record). `tick`, when
    given (verify pass only), is called once per bookkeeping site."""
    s = workloads.compute_chunk_stats(start, end)
    entries = [_entry1(tid, start, end, s, k)]
    if tick:
        tick()
    for j in range(2, k + 1):
        s = _stage(j, s)
        entries.append(_entry(tid, j, s))
        if tick:
            tick()
    return s, entries


def _ticker(ticks: list, tid: int):
    """Verify pass only: a callable that counts one bookkeeping step
    in worker tid's own slot (no sharing between workers)."""
    def tick() -> None:
        ticks[tid] += 1
    return tick


def _writer_stage(j: int, tid: int, stats: dict) -> Writer:
    """Stage j >= 2 as a Writer-returning function for bind."""
    s = _stage(j, stats)
    return Writer(s, [_entry(tid, j, s)])


# ---------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------
class ContractViolation(AssertionError):
    """A design's output does not satisfy the common output contract."""


def _expected(k: int, n: int) -> tuple:
    """Canonical expected output at depth k with n workers, from a
    sequential reference implementation; cached per (k, n)."""
    key = ("expected", k, n)
    if key not in _ref_cache:
        stats_list, log = [], []
        for tid in range(n):
            start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
            s, entries = _run_stages(tid, start, end, k)
            stats_list.append(s)
            log.extend(entries)
        agg = sum(s["sum"] for s in stats_list)
        _ref_cache[key] = (stats_list, log, agg)
    return _ref_cache[key]


def _validate(stats_list: list, log: list, agg: float,
              expected: tuple) -> None:
    """
    Check one design's output against the canonical expected output.
    Raises ContractViolation on the first mismatch.

    Contract:
      stats  list of per-worker final-stage stats dicts, ascending
             worker id; every dict equal to the reference.
      log    flat list, worker-major and stage-minor (all of worker 0's
             entries in stage order, then worker 1's, ...), equal to
             the reference element for element (content AND order).
      agg    the aggregate sum, equal to the reference aggregate
             computed in the same (worker-id) order.
    """
    exp_stats, exp_log, exp_agg = expected
    if len(stats_list) != len(exp_stats):
        raise ContractViolation(
            f"stats: {len(stats_list)} results, expected {len(exp_stats)}")
    for tid, (got, want) in enumerate(zip(stats_list, exp_stats)):
        if got != want:
            raise ContractViolation(f"stats[{tid}]: {got!r} != {want!r}")
    if len(log) != len(exp_log):
        raise ContractViolation(
            f"log: {len(log)} entries, expected {len(exp_log)}")
    for i, (got, want) in enumerate(zip(log, exp_log)):
        if got != want:
            raise ContractViolation(f"log[{i}]: {got!r} != {want!r}")
    if agg != exp_agg:
        raise ContractViolation(f"aggregate sum {agg!r} != {exp_agg!r}")


def _finalise(stats_list: list[dict], log: list[str], *, k: int, n: int,
              lock_acq: int, aux_structures: int, bookkeeping: int,
              bind_calls: int | None = None) -> dict:
    """
    Common result assembly: aggregate sum, non-associativity diagnostic,
    log length, structural proxies, plus a deferred validator (see
    runner._run_one) that checks every result field and the log content
    and order against the canonical expected output. The validator runs
    outside the timed region; a mismatch raises instead of recording a
    row.
    """
    agg = sum(s["sum"] for s in stats_list)
    ref = _reference_sum()
    expected = _expected(k, n)
    out = {
        "result": agg,
        "depth": k,
        "sum_rel_diff": abs(agg - ref) / ref,
        "log_entries": len(log),
        "n_results": len(stats_list),
        "user_lock_acquisitions": lock_acq,
        "aux_structures": aux_structures,
        "bookkeeping_steps": bookkeeping,
        "_validate": lambda: _validate(stats_list, log, agg, expected),
    }
    if bind_calls is not None:
        out["bind_calls"] = bind_calls
    return out


# ---------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------
def make_conditions(k: int, counting: bool = False) -> dict:
    """
    The four designs at pipeline depth k. counting=True builds the
    verify variants: CountingLock for the lock design and a per-worker
    tick at every bookkeeping site (never used in timed runs).
    """
    if k < 1:
        raise ValueError("depth must be >= 1")
    Lock = CountingLock if counting else threading.Lock

    def lock(n: int) -> dict:
        """One lock acquisition per worker; log reordered into contract
        order after the workers finish. The reordering is a real cost
        of this design and is deliberately inside the timed region."""
        structures = shared, shared_log, lk = ({}, [], Lock())
        ticks = [0] * n if counting else None

        def worker(tid: int) -> None:
            start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
            tick = _ticker(ticks, tid) if counting else None
            s, entries = _run_stages(tid, start, end, k, tick)
            with lk:
                shared[tid] = s
                shared_log.append((tid, entries))
            if counting:
                ticks[tid] += 2

        _dispatch(worker, n)
        stats_list = [shared[t] for t in range(n)]
        shared_log.sort(key=_by_tid)                    # contract order
        log = [e for _, block in shared_log for e in block]
        out = _finalise(stats_list, log, k=k, n=n, lock_acq=n,
                        aux_structures=len(structures),
                        bookkeeping=n * (k + 2) + 2)
        if counting:
            out["measured_lock_acquisitions"] = lk.count
            out["measured_bookkeeping_steps"] = sum(ticks) + 2
        return out

    def indexed(n: int) -> dict:
        """Exactly two pre-allocated arrays (final stats, logs) at every
        depth; each worker writes once to each of its slots."""
        structures = stats_slots, log_slots = ([None] * n, [None] * n)
        ticks = [0] * n if counting else None

        def worker(tid: int) -> None:
            start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
            tick = _ticker(ticks, tid) if counting else None
            s, entries = _run_stages(tid, start, end, k, tick)
            stats_slots[tid] = s
            log_slots[tid] = entries
            if counting:
                ticks[tid] += 2

        _dispatch(worker, n)
        log = [e for entries in log_slots for e in entries]
        out = _finalise(list(stats_slots), log, k=k, n=n, lock_acq=0,
                        aux_structures=len(structures),
                        bookkeeping=n * (k + 2) + 1)
        if counting:
            out["measured_bookkeeping_steps"] = sum(ticks) + 1
        return out

    def record(n: int) -> dict:
        """Immutable record per worker, manual combination."""
        structures = ()
        ticks = [0] * n if counting else None

        def worker(tid: int) -> ChunkRecord:
            start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
            tick = _ticker(ticks, tid) if counting else None
            s, entries = _run_stages(tid, start, end, k, tick)
            if counting:
                ticks[tid] += 1
            return ChunkRecord(s, tuple(entries))

        records = _dispatch(worker, n)
        stats_list = [r.stats for r in records]          # manual
        log = [e for r in records for e in r.log]        # combination
        out = _finalise(stats_list, log, k=k, n=n, lock_acq=0,
                        aux_structures=len(structures),
                        bookkeeping=n * (k + 1) + 2)
        if counting:
            out["measured_bookkeeping_steps"] = sum(ticks) + 2
        return out

    def writer(n: int) -> dict:
        """Writer for stage 1, one bind per further stage, one merge."""
        structures = ()
        ticks = [0] * n if counting else None
        binds = [0] * n if counting else None

        def worker(tid: int) -> Writer:
            start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
            s1 = workloads.compute_chunk_stats(start, end)
            w = Writer(s1, [_entry1(tid, start, end, s1, k)])
            if counting:
                ticks[tid] += 1
            for j in range(2, k + 1):
                w = w.bind(functools.partial(_writer_stage, j, tid))
                if counting:
                    ticks[tid] += 1
                    binds[tid] += 1
            return w

        writers = _dispatch(worker, n)
        stats_list, log = merge_writers(writers)
        out = _finalise(stats_list, log, k=k, n=n, lock_acq=0,
                        aux_structures=len(structures),
                        bookkeeping=n * k + 1, bind_calls=n * (k - 1))
        if counting:
            out["measured_bookkeeping_steps"] = sum(ticks) + 1
            out["measured_bind_calls"] = sum(binds)
        return out

    return {"lock": lock, "indexed": indexed,
            "record": record, "writer": writer}


# Legacy names (k = 1 and k = 2), kept for callers and tests
SINGLE = make_conditions(1)
CHAINED = make_conditions(2)


def parse_depths(text: str) -> list[int]:
    depths = [int(x) for x in text.split(",") if x.strip()]
    if not depths or any(d < 1 for d in depths):
        raise argparse.ArgumentTypeError(
            "--depths must be a comma-separated list of integers >= 1")
    return depths


# Main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 3: Writer monad with plain-record "
                    "baseline, pipeline-depth sweep")
    ap.add_argument("--out", default="results")
    ap.add_argument("--depths", type=parse_depths, default=None,
                    help="comma-separated pipeline depths "
                         f"(default {DEFAULT_DEPTHS}); 1 and 2 are "
                         "tagged 'single' and 'chained'")
    ap.add_argument("--stage", choices=["single", "chained", "both"],
                    default=None,
                    help="legacy alias: single = --depths 1, chained = "
                         "--depths 2, both = --depths 1,2")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--runs", type=int, default=config.R_MEASURED)
    ap.add_argument("--warmup", type=int, default=config.R_WARMUP)
    ap.add_argument("--max-threads", type=int, default=None)
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the instrumented counting pass")
    ap.add_argument("--allow-gil", action="store_true",
                    help="permit a GIL-enabled baseline run (free-threaded "
                         "build with PYTHON_GIL=1); prints a warning")
    ap.add_argument("--tag", default=None,
                    help="label appended to output file names and recorded "
                         "in env.json (use 'gil' for GIL-enabled runs)")
    args = ap.parse_args()

    if args.stage is not None and args.depths is not None:
        ap.error("use either --depths or the legacy --stage, not both")
    if args.stage is not None:
        depths = {"single": [1], "chained": [2], "both": [1, 2]}[args.stage]
    else:
        depths = args.depths or parse_depths(DEFAULT_DEPTHS)

    envinfo.assert_free_threading(allow_gil=args.allow_gil)
    tc = config.thread_counts(args.max_threads)

    # Build the canonical expected outputs BEFORE any timed run, so the
    # sequential reference computation never falls inside a timed
    # region (conditions only look the cached value up).
    _reference_sum()
    for k in depths:
        for n in tc:
            _expected(k, n)

    for k in depths:
        runner = Runner("exp3_writer", args.out, workload=depth_tag(k),
                        seed=args.seed, r_warmup=args.warmup,
                        r_measured=args.runs, tag=args.tag,
                        extra_env={"depth": k})
        runner.sweep(make_conditions(k), tc)
        if not args.skip_verify:
            runner.single_pass(make_conditions(k, counting=True), tc,
                               phase="verify")
        if not args.skip_memory:
            runner.single_pass(with_memory(make_conditions(k)), tc,
                               phase="memory")


# Main guard
if __name__ == "__main__":
    main()
