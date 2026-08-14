"""
Smoke tests for the three experiment CLIs: confirm each one runs
end-to-end and produces well-formed output before committing to a
full multi-hour sweep (RUNNING.md, step 3).

Every experiment calls bench.envinfo.assert_free_threading() and
exits non-zero under a GIL-enabled interpreter, so these only
exercise the real CLIs under a free-threaded build; run under a
regular interpreter they are skipped with a one-line notice rather
than failing.

Run:
    uv run --python 3.14t python -m tests.test_smoke
"""

#Libraries
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile

from bench.exp1_race import CONDITIONS as EXP1_CONDITIONS
from bench.exp2_scaling import make_synthetic_conditions
from bench.exp3_writer import SINGLE, CHAINED
from bench.runner import CSV_FIELDS

# Configuration constants
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_ARGS = ["--runs", "1", "--warmup", "1", "--max-threads", "2"]


class _Skipped(Exception):
    pass


def _run_module(module: str, out_dir: str) -> None:
    """Invoke a bench CLI with minimal --runs/--warmup; raise on failure."""
    proc = subprocess.run(
        [sys.executable, "-m", module, "--out", out_dir, *SMOKE_ARGS],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 and "GIL is enabled" in (proc.stderr or ""):
        raise _Skipped(proc.stderr.strip().splitlines()[-1])
    assert proc.returncode == 0, (
        f"{module} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _check_csv(csv_path: str, expected_conditions: set) -> None:
    """Header matches the runner schema; every condition produced
    exactly one measured-phase row (--runs 1)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS, csv_path
        rows = list(reader)
    measured = {r["condition"] for r in rows if r["phase"] == "measured"}
    assert measured == expected_conditions, (
        f"{csv_path}: expected {expected_conditions}, got {measured}"
    )


def _check_env(csv_path: str) -> None:
    env_path = csv_path[: -len(".csv")] + ".env.json"
    with open(env_path) as f:
        env = json.load(f)
    assert env["free_threading_build"] is True, env_path
    assert env["gil_enabled"] is False, env_path


def _check_run(out_dir: str, expected: dict[str, set]) -> None:
    """expected: {csv-filename-stem-prefix: expected condition names}."""
    for prefix, conditions in expected.items():
        matches = glob.glob(os.path.join(out_dir, f"{prefix}_*.csv"))
        assert len(matches) == 1, (prefix, matches)
        _check_csv(matches[0], conditions)
        _check_env(matches[0])


def test_exp1_race_smoke() -> None:
    with tempfile.TemporaryDirectory() as out:
        _run_module("bench.exp1_race", out)
        _check_run(out, {"exp1_race_counter": set(EXP1_CONDITIONS)})


def test_exp2_scaling_smoke() -> None:
    with tempfile.TemporaryDirectory() as out:
        _run_module("bench.exp2_scaling", out)
        _check_run(out, {
            "exp2_scaling_synthetic": set(make_synthetic_conditions()),
        })


def test_exp3_writer_smoke() -> None:
    with tempfile.TemporaryDirectory() as out:
        _run_module("bench.exp3_writer", out)
        _check_run(out, {
            "exp3_writer_single": set(SINGLE),
            "exp3_writer_chained": set(CHAINED),
        })


# Main guard
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except _Skipped as exc:
                print(f"skip {name}: {exc}")
                continue
            print(f"ok  {name}")
    print("smoke tests done")
