"""
Experiment 3: structured multi-field accumulation — Writer monad
versus lock, indexed, and PLAIN-RECORD baselines.

Conditions (single-stage / two-stage "chained"):
  lock      shared dict + shared log list, one lock acquisition per
            worker / new shared dict + same pattern per stage
  indexed   pre-allocated per-thread slots, no lock / two further
            pre-allocated arrays per stage
  record    NamedTuple(stats, log) returned per worker, manual
            combination / manual log concatenation across stages
  writer    Writer(stats, [entry]) returned per worker, one
            merge_writers call / one .bind() call per extra stage

Run (from the directory containing bench/):
    python -m bench.exp3_writer --out results/
"""

# Libraries
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from . import config, envinfo, workloads
from .monads import Writer, merge_writers
from .runner import Runner, with_memory

# Configuration variable
N_ITEMS = config.N_ITEMS


class ChunkRecord(NamedTuple):
    """Plain immutable record baseline: same payload as Writer,
    no monadic interface."""
    stats: dict
    log: tuple[str, ...]


def _dispatch(worker, n_threads: int) -> list:
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(worker, range(n_threads)))


_ref_cache: dict = {}


def _reference_sum() -> float:
    if "sum" not in _ref_cache:
        _ref_cache["sum"] = workloads.compute_chunk_stats(
            0, N_ITEMS)["sum"]
    return _ref_cache["sum"]


def _entry(tid: int, start: int, end: int, stats: dict) -> str:
    return (f"thread={tid} slice=[{start},{end}) "
            f"sum={stats['sum']:.4f} count={stats['count']}")


def _finalise(stats_list: list[dict], log: list[str],
              n_threads: int, lock_acq: int) -> dict:
    """Common result assembly: aggregate sum, correctness fields,
    non-associativity diagnostic, log length."""
    agg = sum(s["sum"] for s in stats_list)
    ref = _reference_sum()
    return {
        "result": agg,
        "sum_rel_diff": abs(agg - ref) / ref,
        "log_entries": len(log),
        "n_results": len(stats_list),
        "user_lock_acquisitions": lock_acq,
    }


def _stage2(stats: dict) -> dict:
    """Pure Stage-2 transform."""
    mean = stats["mean"]
    spread = stats["max"] - stats["min"]
    return {"range": spread,
            "cv": spread / mean if mean != 0.0 else 0.0,
            "sum": stats["sum"], "count": stats["count"]}


# Single-stage
def cond_lock(n: int) -> dict:
    shared: dict[int, dict] = {}
    shared_log: list[str] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        stats = workloads.compute_chunk_stats(start, end)
        entry = _entry(tid, start, end, stats)
        with lock:
            shared[tid] = stats
            shared_log.append(entry)

    _dispatch(worker, n)
    stats_list = [shared[t] for t in range(n)]
    return _finalise(stats_list, shared_log, n, lock_acq=n)


def cond_indexed(n: int) -> dict:
    results: list = [None] * n
    logs: list = [None] * n

    def worker(tid: int) -> None:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        stats = workloads.compute_chunk_stats(start, end)
        results[tid] = stats
        logs[tid] = _entry(tid, start, end, stats)

    _dispatch(worker, n)
    return _finalise(list(results), list(logs), n, lock_acq=0)


def cond_record(n: int) -> dict:
    def worker(tid: int) -> ChunkRecord:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        stats = workloads.compute_chunk_stats(start, end)
        return ChunkRecord(stats, (_entry(tid, start, end, stats),))

    records = _dispatch(worker, n)
    stats_list = [r.stats for r in records]           # manual
    log = [e for r in records for e in r.log]         # combination
    return _finalise(stats_list, log, n, lock_acq=0)


def cond_writer(n: int) -> dict:
    def worker(tid: int) -> Writer:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        stats = workloads.compute_chunk_stats(start, end)
        return Writer(stats, [_entry(tid, start, end, stats)])

    writers = _dispatch(worker, n)
    stats_list, log = merge_writers(writers)
    return _finalise(stats_list, log, n, lock_acq=0)


# Two-stage
def _entries2(tid: int, s1: dict, s2: dict) -> tuple[str, str]:
    return (f"s1 thread={tid} sum={s1['sum']:.4f} mean={s1['mean']:.4f}",
            f"s2 thread={tid} cv={s2['cv']:.4f} range={s2['range']:.4f}")


def cond_lock_chained(n: int) -> dict:
    shared: dict[int, dict] = {}
    shared_log: list[str] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        s1 = workloads.compute_chunk_stats(start, end)
        s2 = _stage2(s1)
        e1, e2 = _entries2(tid, s1, s2)
        with lock:
            shared[tid] = s2
            shared_log.append(e1)
            shared_log.append(e2)

    _dispatch(worker, n)
    stats_list = [shared[t] for t in range(n)]
    return _finalise(stats_list, shared_log, n, lock_acq=n)


def cond_indexed_chained(n: int) -> dict:
    s1_res: list = [None] * n
    s2_res: list = [None] * n
    s1_logs: list = [None] * n
    s2_logs: list = [None] * n

    def worker(tid: int) -> None:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        s1 = workloads.compute_chunk_stats(start, end)
        s2 = _stage2(s1)
        e1, e2 = _entries2(tid, s1, s2)
        s1_res[tid], s2_res[tid] = s1, s2
        s1_logs[tid], s2_logs[tid] = e1, e2

    _dispatch(worker, n)
    log = list(s1_logs) + list(s2_logs)
    return _finalise(list(s2_res), log, n, lock_acq=0)


def cond_record_chained(n: int) -> dict:
    def worker(tid: int) -> ChunkRecord:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        s1 = workloads.compute_chunk_stats(start, end)
        s2 = _stage2(s1)
        e1, e2 = _entries2(tid, s1, s2)
        return ChunkRecord(s2, (e1, e2))

    records = _dispatch(worker, n)
    stats_list = [r.stats for r in records]
    log = [e for r in records for e in r.log]
    return _finalise(stats_list, log, n, lock_acq=0)


def cond_writer_chained(n: int) -> dict:
    def worker(tid: int) -> Writer:
        start, end = workloads.chunk_bounds(tid, n, N_ITEMS)
        s1 = workloads.compute_chunk_stats(start, end)
        w1 = Writer(s1, [f"s1 thread={tid} sum={s1['sum']:.4f} "
                         f"mean={s1['mean']:.4f}"])

        def stage2_fn(stats: dict) -> Writer:
            s2 = _stage2(stats)
            return Writer(s2, [f"s2 thread={tid} cv={s2['cv']:.4f} "
                               f"range={s2['range']:.4f}"])

        return w1.bind(stage2_fn)

    writers = _dispatch(worker, n)
    stats_list, log = merge_writers(writers)
    return _finalise(stats_list, log, n, lock_acq=0)


SINGLE = {"lock": cond_lock, "indexed": cond_indexed,
          "record": cond_record, "writer": cond_writer}
CHAINED = {"lock": cond_lock_chained, "indexed": cond_indexed_chained,
           "record": cond_record_chained, "writer": cond_writer_chained}


# Main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 3: Writer monad with plain-record "
                    "baseline")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stage", choices=["single", "chained", "both"],
                    default="both")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--runs", type=int, default=config.R_MEASURED)
    ap.add_argument("--warmup", type=int, default=config.R_WARMUP)
    ap.add_argument("--max-threads", type=int, default=None)
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args()

    envinfo.assert_free_threading()
    tc = config.thread_counts(args.max_threads)

    stages = {"single": SINGLE, "chained": CHAINED}
    wanted = (["single", "chained"] if args.stage == "both"
              else [args.stage])
    for name in wanted:
        runner = Runner("exp3_writer", args.out, workload=name,
                        seed=args.seed, r_warmup=args.warmup,
                        r_measured=args.runs)
        runner.sweep(stages[name], tc)
        if not args.skip_memory:
            runner.single_pass(with_memory(stages[name]), tc,
                               phase="memory")

# Main guard
if __name__ == "__main__":
    main()
