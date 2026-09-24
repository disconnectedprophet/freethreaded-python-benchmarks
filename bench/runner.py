"""
Benchmark harness shared by all experiments.

Design goals:

  Randomised interleaved execution order to combat the effects of the 
  potential confound variable (i.e. environment changes over time: 
  cache memory filled with data from the previous experiment, 
  thermal throttling, etc.). Within every thread-count cell, the condition-repetition pairs 
  (e.g. 3 threads - 2 warmup repetitions - fp_reduce vs. 3 threads - 
  3 warmup repetitions - coarse lock, etc.) are shuffled together with a seeded RNG, 
  so no condition systematically runs first/last or hot/cold.

  Warm-up: R_WARMUP repetitions per condition are executed before the
  measured block (also interleaved) and written to the CSV with the tag -
  phase=warmup, hence the analysis can inspect but exclude them.

  Raw-data persistence: every individual run is appended to a CSV the
  moment it finishes. Tables and statistics are later derived from
  this single file by the analysis script.

  Environment capture: env.json is written next to the CSV before the
  first run via envinfo.write().

A condition is a callable fn(n_threads) -> dict executing exactly one
full run and returning at least {"result": <value>}. Optionally, it returns
"expected" when the exact target is known (e.g. increments of int numbers) as 
exact target — loss/error. It can also return "reference" when the target is 
approximated (i.e. imprecision stemming from the aggregation of floating point 
values, where the order of summation depends on thread processing). Please note 
that the CSV has a single "expected" column that holds both kinds of target. 
Which kind a row carries follows from its "experiment" and "workload"
columns: exp1 and the exp2 montecarlo workload report an exact target,
so any non-zero error_pct is real loss. Exp2 synthetic workload
reports an approximate serial reference, where error_pct at the 1e-16
level is floating-point summation-order noise rather than loss. Exp3
has no target and leaves both the expected and error_pct columns empty.
"""

# Libraries
import csv
import json
import os
import random
import time
from typing import Callable

from . import envinfo

# Output table schema
CSV_FIELDS = [
    "experiment", "workload", "condition", "n_threads",
    "phase", "rep", "order_idx", "wall_s",
    "result", "expected", "error_pct", "aux", "timestamp_utc",
]

Condition = Callable[[int], dict]


class Runner:
    """Executes a sweep and streams every run to a CSV file."""

    def __init__(self, experiment: str, out_dir: str, *,
                 workload: str = "default",
                 seed: int, r_warmup: int, r_measured: int) -> None:
        self.experiment = experiment
        self.workload = workload
        self.seed = seed
        self.r_warmup = r_warmup
        self.r_measured = r_measured

        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        base = f"{experiment}_{workload}_{stamp}"
        self.csv_path = os.path.join(out_dir, base + ".csv")
        self.env_path = os.path.join(out_dir, base + ".env.json")

        envinfo.write(self.env_path, seed=seed, extra={
            "experiment": experiment,
            "workload": workload,
            "r_warmup": r_warmup,
            "r_measured": r_measured,
        })
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    
    def _write_row(self, row: dict) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

    def _schedule(self, conditions: dict[str, Condition],
                  n_threads: int, phase: str,
                  reps: int) -> list[tuple[str, int]]:
        """Interleaved, seeded shuffle of (condition, rep) pairs."""
        pairs = [(name, r) for name in conditions for r in range(reps)]
        rng = random.Random(
            f"{self.seed}|{self.experiment}|{self.workload}"
            f"|{n_threads}|{phase}"
        )
        rng.shuffle(pairs)
        return pairs

    def _run_one(self, conditions: dict[str, Condition], name: str,
                 n_threads: int, phase: str, rep: int,
                 order_idx: int) -> None:
        t0 = time.perf_counter()
        out = conditions[name](n_threads)
        wall = time.perf_counter() - t0

        result = out.pop("result", None)
        expected = out.pop("expected", out.pop("reference", None))
        error_pct = None
        if (expected is not None and result is not None
                and expected != 0):
            error_pct = abs(expected - result) / abs(expected) * 100.0

        self._write_row({
            "experiment": self.experiment,
            "workload": self.workload,
            "condition": name,
            "n_threads": n_threads,
            "phase": phase,
            "rep": rep,
            "order_idx": order_idx,
            "wall_s": f"{wall:.6f}",
            "result": result,
            "expected": expected,
            "error_pct": (f"{error_pct:.6f}"
                          if error_pct is not None else ""),
            "aux": json.dumps(out) if out else "",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
        })

    def sweep(self, conditions: dict[str, Condition],
              thread_counts: list[int]) -> None:
        """Run all conditions over thread_counts; stream rows to CSV."""
        print(f"[{self.experiment}/{self.workload}] "
              f"conditions={list(conditions)} "
              f"threads={thread_counts} "
              f"warmup={self.r_warmup} measured={self.r_measured} "
              f"seed={self.seed}")
        print(f"  csv: {self.csv_path}")

        for n in thread_counts:
            t_cell = time.perf_counter()
            order_idx = 0
            for phase, reps in (("warmup", self.r_warmup),
                                ("measured", self.r_measured)):
                for name, rep in self._schedule(conditions, n,
                                                phase, reps):
                    self._run_one(conditions, name, n, phase, rep,
                                  order_idx)
                    order_idx += 1
            print(f"  n={n:>3}: {order_idx} runs "
                  f"in {time.perf_counter() - t_cell:.1f} s")
        print(f"[{self.experiment}/{self.workload}] done -> "
              f"{self.csv_path}")

    def single_pass(self, conditions: dict[str, Condition],
                    thread_counts: list[int], phase: str) -> None:
        """
        One untimed-analysis repetition per (condition, thread count),
        written with the given phase tag. Used for the memory and
        lock-verification passes, which are kept OUT of the measured
        timing block because their instrumentation (tracemalloc,
        counting locks) perturbs wall time.
        """
        print(f"[{self.experiment}/{self.workload}] "
              f"single pass phase={phase}")
        for n in thread_counts:
            for idx, name in enumerate(conditions):
                self._run_one(conditions, name, n, phase, 0, idx)
        print(f"  phase={phase} done")


def with_memory(conditions: dict[str, Condition]
                ) -> dict[str, Condition]:
    """
    Wrap each condition to record Python-level peak allocation via
    tracemalloc. The peak is stored in the aux column as tracemalloc_peak_bytes.
    """
    import tracemalloc

    def wrap(fn: Condition) -> Condition:
        def wrapped(n_threads: int) -> dict:
            tracemalloc.start()
            try:
                out = fn(n_threads)
                _cur, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            out["tracemalloc_peak_bytes"] = peak
            return out
        return wrapped

    return {name: wrap(fn) for name, fn in conditions.items()}
