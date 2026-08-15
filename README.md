# freethreaded-python-benchmarks

Matched-design benchmark suite for functional programming patterns in free-threaded Python: race-freedom, coordination cost, and the Writer monad.

This repository is the reproducibility artifact for the paper *"Functional Programming Patterns in Free-Threaded Python: A Matched-Design Empirical Study of Race-Freedom, Coordination Cost, and the Writer Monad"* (IEEE Access, submitted). It contains the complete benchmark suite, the raw per-run measurements reported in the paper, and the analysis scripts that regenerate every table from those measurements.

## Study design

Free-threaded CPython (PEP 703 / PEP 779) removes the Global Interpreter Lock, so for the first time thread safety in the reference interpreter depends entirely on how user code accesses shared state. This suite compares accumulation strategies drawn from functional and imperative practice under a **matched design**: the headline conditions differ in exactly one property at a time, so runtime differences can be attributed to the communication topology rather than to unrelated implementation details.

Three experiments:

- **Experiment 1 - correctness.** Five conditions performing identical accumulation work: an unsynchronised shared counter (the known-unsafe baseline), a lock-protected counter, an imperative thread-local accumulator, a pure functional fold, and ownership-based disjoint slot writes.
- **Experiment 2 - coordination and scaling.** Eight synthetic conditions with byte-identical worker kernels (functional reduce vs. imperative thread-local vs. coarse / batched / fine locking vs. queue aggregation vs. unsynchronised), plus a count-based Monte Carlo workload with a deterministic reference, plus lock-count verification and memory passes.
- **Experiment 3 - structured accumulation.** The Writer monad against lock-based, index-isolated, and plain immutable record baselines, single- and two-stage.

All statistical claims use one-sample tests for correctness, Welch tests with Holm correction for superiority, and two one-sided tests (TOST) with a prespecified ±5% margin for equivalence. All numbers in the paper are produced by `analysis/analyze.py` from the raw CSVs in this repository.

## Repository layout

```
bench/                 core suite (standard-library only)
  config.py            protocol constants, hardware-agnostic thread ladder
  envinfo.py           environment capture; refuses to run with the GIL enabled
  runner.py            harness: seeded interleaved order, warm-up, per-run CSV streaming
  workloads.py         pure CPU-bound kernels; count-based Monte Carlo
  monads.py            Writer monad (formal definition in the docstring; matches Listing 1)
  exp1_race.py         Experiment 1
  exp2_scaling.py      Experiment 2
  exp3_writer.py       Experiment 3
analysis/
  analyze.py           reads the raw CSVs, regenerates every table and statistical test
tests/
  test_monads.py       monad-law verification cited by the paper
  test_smoke.py         CLI smoke tests: each experiment with --runs 1, confirms the suite runs before a full sweep
results/
  amd/               raw per-run CSVs + env.json, AMD platform (10 files)
  intel/                 raw per-run CSVs + env.json, Intel platform (10 files)
RUNNING.md             detailed GCP run instructions
setup_gcp.sh           VM setup: installs uv + free-threaded CPython 3.14
requirements.txt       analysis dependencies
CITATION.cff            citation metadata for GitHub "Cite this repository"
LICENSE                MIT
```

## Requirements

**Running the benchmarks** requires only a free-threaded CPython 3.14 build (`3.14t`) and the standard library - no third-party packages on the benchmark machine. `setup_gcp.sh` installs the interpreter via [uv](https://docs.astral.sh/uv/).

**Running the analysis** requires Python 3.11+ with the packages in `requirements.txt` (NumPy, pandas, SciPy). The analysis runs anywhere; it does not need a free-threaded build.

## Reproducing the measurements

The measurements in the paper were produced on two Google Cloud instances, both configured with one thread per core (no SMT), so a software thread maps to one physical core:

- **Intel** - `n2-standard-32` with `--min-cpu-platform="Intel Ice Lake"` and `--threads-per-core=1`, which exposes 16 physical cores (Ice Lake, family 6 model 106).
- **AMD** - `t2d-standard-16` (EPYC 7B13, Milan), 16 physical cores, no SMT by design.

The `--min-cpu-platform` pin is required on the n2 family, which otherwise may schedule on Cascade Lake. See **RUNNING.md** for the full step-by-step (instance creation, setup, smoke test, full session, result collection). In brief, on each VM:

```bash
bash setup_gcp.sh    # installs uv + CPython 3.14t; prints "GIL enabled: False"

uv run --python 3.14t python -m bench.exp1_race    --out results/
uv run --python 3.14t python -m bench.exp2_scaling --out results/ --workload synthetic
uv run --python 3.14t python -m bench.exp2_scaling --out results/ --workload montecarlo
uv run --python 3.14t python -m bench.exp3_writer  --out results/
```

Defaults: 30 measured + 3 warm-up repetitions per (condition × thread count), seeded randomized interleaved execution order, plus the counting-lock verification and tracemalloc memory passes. A full session takes roughly 1.5–3 h per machine.

Every run streams to a CSV as it completes, and an `env.json` capturing the interpreter build, CPU topology, affinity, and instance metadata is written alongside it. The `results/intel/` and `results/amd/` directories contain the exact files used in the paper.

## Regenerating the results

From a machine with the analysis dependencies installed:

```bash
pip install -r requirements.txt

python -m analysis.analyze --data results/intel \
  --out report/intel --platform "Intel Xeon Ice Lake (n2-standard-32, 16 physical cores, SMT off)"
python -m analysis.analyze --data results/amd \
  --out report/amd   --platform "AMD EPYC 7B13 (t2d-standard-16, 16 physical cores, no SMT)"
```

Each run writes a `report.md` plus one CSV per table. These reproduce the manuscript's main-text tables (Intel) and appendix tables (AMD) from the raw data.

## Verifying the monad laws

```bash
uv run --python 3.14t python -m tests.test_monads
python -m tests.test_monads
```

## Smoke-testing the suite

Before committing to a full multi-hour sweep, confirm all three experiments run end-to-end (`--runs 1 --warmup 1`) and produce well-formed CSV + env.json output:

```bash
uv run --python 3.14t python -m tests.test_smoke
```

Run under a GIL-enabled interpreter, each test is skipped with a one-line notice rather than failing, since the experiments themselves refuse to run without free-threading.

## Citing

If you use this artifact, please cite the paper (full reference and DOI to be added on publication). A `CITATION.cff` is included so GitHub's "Cite this repository" produces a machine-readable citation.

## License

MIT - see `LICENSE`.
