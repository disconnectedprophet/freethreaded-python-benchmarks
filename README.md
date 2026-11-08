# freethreaded-python-benchmarks

Matched-design benchmark suite for functional programming patterns in free-threaded Python (3.13+): race-freedom, coordination cost, and the Writer monad. It comprises the raw (per run) measurements and the scripts that can be used to regenerate all reported measurements.

## Study design

Free-threaded CPython (PEP 703/PEP 779) removes the Global Interpreter Lock (GIL), so for the first time thread safety in the reference interpreter depends entirely on how user code accesses shared state. This suite compares accumulation strategies drawn from functional and imperative practice under a matched design. Each headline pair (`fp_reduce` against `imp_threadlocal`, `coarse_lock`, `batched_lock` and `queue_agg`, all of which run the same `partial_sum` kernel) differs in exactly one property, so runtime differences between them can be attributed to the communication topology rather than to unrelated implementation details. The kernel-style conditions do not all meet that standard and are reported accordingly: as single-factor contrasts where a pair differs in one factor, and as multi-factor contrasts, with every differing factor named, where it does not.

Three experiments:

- **Experiment 1 - correctness.** Five conditions performing identical accumulation work: an unsynchronised shared counter (the known-unsafe baseline), a lock-protected counter (fine-grained), an imperative thread-local accumulator, a pure functional fold, and ownership-based disjoint slot writes.
- **Experiment 2 - coordination and scaling.** Synthetic conditions: functional reduce, imperative thread-local, coarse / batched locking and queue aggregation share a byte-identical worker kernel (`partial_sum`); fine-grained locking and the unsynchronised condition deliberately do not, because the per-item write to shared state is what they measure. Plus a count-based Monte Carlo workload with a deterministic reference, plus lock-count verification and memory passes. Additional conditions:
  - kernel-style conditions: `fp_sum_genexpr`, `fp_sum_map` (`sum(map(...))`) and `fp_sum_listcomp` (`sum([...])`), which give single-factor runtime contrasts (generator-frame overhead; list materialisation), plus `fp_reduce_genexpr` and `imp_loop_compensated`, which are controls for the numerical axis only (bit-identical to the naive loop and to `sum()` respectively). Every synthetic condition reports its numerical error against the exactly rounded `math.fsum` reference;
  - `batched_lock` (matched kernel: each batch summed by `partial_sum`) and `batched_lock_unmatched` (the same locking with an inline per-item batch counter in the worker loop, kept to measure that kernel confound).
- **Experiment 3 - structured accumulation.** The Writer monad against lock-based, index-isolated, and plain immutable record baselines over a pipeline-depth sweep (default k = 1, 2, 4, 8, 16). All designs satisfy one validated output contract (per-worker stats in worker order; log worker-major, stage-minor) and report structural proxies (auxiliary structures, bookkeeping steps, bind calls).

Every experiment can also run as a **GIL-enabled baseline** on the same free-threaded interpreter (`PYTHON_GIL=1` plus `--allow-gil`), so that effects specific to free-threading can be separated from effects that are not.

All statistical claims use one-sample tests for correctness, Welch tests with Holm correction for superiority, and two one-sided tests (TOST) with a prespecified ±5% margin for equivalence. All numbers in the paper are produced by `analysis/analyze.py` from the raw CSVs in this repository.

## Repository layout

```
bench/                 core suite (standard-library only)
  config.py            protocol constants, analysis constants, runtime-derived thread ladder
  envinfo.py           environment capture; refuses to run with the GIL enabled unless --allow-gil
  runner.py            harness: seeded interleaved order, warm-up, per-run CSV streaming
  workloads.py         pure CPU-bound kernels; count-based Monte Carlo
  writer_monad.py      Writer monad (formal definition in the docstring; matches the manuscript listing)
  exp1_race.py         Experiment 1
  exp2_scaling.py      Experiment 2
  exp3_writer.py       Experiment 3 (pipeline-depth sweep)
analysis/
  analyze.py           reads the raw CSVs, regenerates every table and statistical test
tests/
  test_writer_monad.py  monad- and functor-law verification
  test_conditions.py    condition-level properties: bit-identity groups, lock counts, Exp3 output contract
  test_smoke.py         CLI smoke tests: each experiment with --runs 1, incl. --depths, --n-items, --allow-gil
  test_analysis.py      analysis consistency: one verdict per pair, contrast tables carry no verdict (needs the analysis dependencies; skips without them)
results/               data from this release (written by the full session, see RUNNING.md, step 4)
  amd/                 raw per-run CSVs + env.json, AMD platform (20 files)
  intel/               raw per-run CSVs + env.json, Intel platform (20 files)
RUNNING.md             detailed GCP run instructions
setup_gcp.sh           VM setup: installs uv + free-threaded CPython 3.14
requirements.txt       analysis dependencies
CITATION.cff           citation metadata for GitHub "Cite this repository"
LICENSE                MIT
```

## Requirements

**Running the benchmarks** requires only a free-threaded CPython 3.14 build (`3.14t`) and the standard library - no third-party packages on the benchmark machine. `setup_gcp.sh` installs the interpreter via [uv](https://docs.astral.sh/uv/).

**Running the analysis** requires Python 3.11+ with the packages in `requirements.txt` (NumPy, pandas, SciPy, tabulate). The analysis runs anywhere; it does not need a free-threaded build.

## Reproducing the measurements

The measurements in the paper were produced on two Google Cloud compute instances, both configured with one thread per core (no simultaneous multithreading), so a software thread maps to one physical core:

- **Intel** - `n2-standard-32` with `--min-cpu-platform="Intel Ice Lake"` and `--threads-per-core=1`, which exposes 16 physical cores (Ice Lake, family 6 model 106).
- **AMD** - `t2d-standard-16` (EPYC 7B13, Milan), 16 physical cores, no SMT by design.

The `--min-cpu-platform` pin is required on the n2 family, which otherwise may schedule on Cascade Lake. See **RUNNING.md** for the full step-by-step (instance creation, setup, smoke test, full session, result collection). In brief, on each VM:

```bash
bash setup_gcp.sh    # installs uv + CPython 3.14t; prints "GIL enabled: False"

R=results/amd     # or results/intel
uv run --python 3.14t python -m bench.exp1_race    --out $R
uv run --python 3.14t python -m bench.exp2_scaling --out $R --workload synthetic
uv run --python 3.14t python -m bench.exp2_scaling --out $R --workload synthetic --n-items 5000000 --tag n5m
uv run --python 3.14t python -m bench.exp2_scaling --out $R --workload montecarlo
uv run --python 3.14t python -m bench.exp3_writer  --out $R --depths 1,2,4,8,16
# GIL-enabled baseline: SAME free-threaded interpreter, GIL switched on
PYTHON_GIL=1 uv run --python 3.14t python -m bench.exp2_scaling --out $R --workload synthetic --allow-gil --tag gil
```

Command-line flags:

| Flag | Experiments | Meaning |
|---|---|---|
| `--out DIR` | all | output directory for the CSV and its env.json (default `results`) |
| `--workload NAME` | exp2 | `synthetic` (default) or `montecarlo` |
| `--seed N` | all | seed for the interleaved execution order (default 42); recorded in env.json |
| `--runs N` | all | measured repetitions per (condition × thread count) (default 30) |
| `--warmup N` | all | warm-up repetitions per (condition × thread count) (default 3); written to the CSV as `phase=warmup` |
| `--max-threads N` | all | cap the thread ladder; by default it is derived from the visible CPU count |
| `--n-items N` | exp2 | total work items (default 500,000); recorded in env.json |
| `--depths LIST` | exp3 | comma-separated pipeline depths (default `1,2,4,8,16`); k = 1 and 2 keep the workload tags `single` and `chained`, deeper runs are tagged `depth<k>` |
| `--stage NAME` | exp3 | alias for a single depth: `single` = `--depths 1`, `chained` = `--depths 2`, `both` = `--depths 1,2`; cannot be combined with `--depths` |
| `--skip-verify` | exp2, exp3 | skip the instrumented counting pass |
| `--skip-memory` | exp2, exp3 | skip the tracemalloc memory pass |
| `--allow-gil` | all | permit a GIL-enabled baseline run; requires the free-threaded build with `PYTHON_GIL=1` (a default build is rejected) and prints a warning |
| `--tag TAG` | all | appended to output file names (`{experiment}_{workload}_{stamp}_{TAG}`) and recorded in env.json; use `gil` for GIL-enabled runs |

Timing scope: every condition creates its `ThreadPoolExecutor` inside the timed call, so pool creation and thread spawn are part of each measured wall time (deliberate; see `bench/runner.py`).

Defaults: 30 measured + 3 warm-up repetitions per (condition × thread count), seeded randomized interleaved execution order, plus the counting-lock verification and tracemalloc memory passes. A full session takes roughly 1.5-3h per machine.

Every run streams to a CSV as it completes, and an `env.json` capturing the interpreter build, GIL state, CPU topology, affinity, and instance metadata (plus `n_items`, `depth` and `tag` where applicable) is written alongside it. `results/intel/` and `results/amd/` hold the data from this release.

## Regenerating the results

From a machine with the analysis dependencies installed:

```bash
pip install -r requirements.txt

python -m analysis.analyze --data results/intel \
  --out report/intel --platform "Intel Xeon Ice Lake (n2-standard-32, 16 physical cores, SMT off)"
python -m analysis.analyze --data results/amd \
  --out report/amd   --platform "AMD EPYC 7B13 (t2d-standard-16, 16 physical cores, no SMT)"
```

Each run writes a `report.md` plus one CSV per table. Each CSV is identified by its env.json (GIL state, `n_items`, depth), so free-threaded, GIL-enabled and larger-N runs can share one directory.

## Verifying the monad laws and the condition properties

The monad and functor laws hold independently of the interpreter build, so either interpreter will do:

```bash
uv run --python 3.14t python -m tests.test_writer_monad   # free-threaded
python -m tests.test_writer_monad   # any Python
```

Condition-level properties the matched design relies on (bit-identity groups, lock counts of both batched variants, the Experiment 3 output contract at depths 1, 2, 4):

```bash
uv run --python 3.14t python -m tests.test_conditions
```

Analysis consistency (every tested pair has one verdict; contrast tables report effect sizes only and agree with the pairwise tables). Needs the packages in `requirements.txt`; without them each test is skipped with a notice:

```bash
python -m tests.test_analysis
```

## Smoke-testing the suite

Before committing to a full multi-hour sweep, confirm all three experiments run end-to-end (`--runs 1 --warmup 1`) and produce well-formed CSV + env.json output:

```bash
uv run --python 3.14t python -m tests.test_smoke
```

Run under a GIL-enabled interpreter, each test is skipped with a one-line notice rather than failing, since the experiments themselves refuse to run without free-threading. The smoke tests also exercise the `--allow-gil` path by starting the free-threaded interpreter with `PYTHON_GIL=1`.

## Citing

If you use this artifact, please cite the paper (full reference and DOI to be added on publication). A `CITATION.cff` is included so GitHub's "Cite this repository" produces a machine-readable citation.

## License

MIT - see `LICENSE`.
