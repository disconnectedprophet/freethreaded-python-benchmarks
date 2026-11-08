# Benchmark Suite - Running Instructions (GCP)

## 1. Create the instances

**AMD - EPYC 7B13, Milan, 16 physical cores, no SMT:**
```bash
gcloud compute instances create bench-amd \
  --machine-type=t2d-standard-16 \
  --zone=europe-west4-b \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB
```

**Intel - Ice Lake, SMT disabled, 16 physical cores:**
```bash
gcloud compute instances create bench-intel \
  --machine-type=n2-standard-32 \
  --min-cpu-platform="Intel Ice Lake" \
  --threads-per-core=1 \
  --zone=europe-west4-b \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB
```

## 2. Setup (on each VM)

```bash
# from your machine:
gcloud compute scp --recurse freethreaded-python-benchmarks/ bench-amd:~/
gcloud compute ssh bench-amd
cd ~/freethreaded-python-benchmarks && bash setup_gcp.sh
```

## 3. Smoke test (do this first, report any problem)

```bash
cd ~/freethreaded-python-benchmarks
S="--runs 1 --warmup 1 --max-threads 4"
uv run --python 3.14t python -m bench.exp1_race    --out smoke/ $S
uv run --python 3.14t python -m bench.exp2_scaling --out smoke/ $S --workload synthetic
uv run --python 3.14t python -m bench.exp2_scaling --out smoke/ $S --workload synthetic --n-items 5000000 --tag n5m
uv run --python 3.14t python -m bench.exp2_scaling --out smoke/ $S --workload montecarlo
uv run --python 3.14t python -m bench.exp3_writer  --out smoke/ $S --depths 1,2,4,8,16
PYTHON_GIL=1 uv run --python 3.14t python -m bench.exp2_scaling --out smoke/ $S --workload synthetic --allow-gil --tag gil
uv run --python 3.14t python -m tests.test_writer_monad
uv run --python 3.14t python -m tests.test_conditions

# or, all CLIs in one shot (--runs 1, asserts CSV/env.json shape,
# including the --depths, --n-items and --allow-gil paths):
uv run --python 3.14t python -m tests.test_smoke
```

## 4. Full session (run on BOTH machines, same order)

Each VM writes directly to `results/<platform>/`, so the tree that comes
off the VM already has the published layout. Set `P` to `amd` or `intel`.

```bash
cd ~/freethreaded-python-benchmarks
P=amd   # or intel
nohup bash -c "
  set -e
  R=results/$P
  # 1. Experiment 1 (re-run so the whole dataset comes from one session)
  uv run --python 3.14t python -m bench.exp1_race    --out \$R
  # 2. Experiment 2, synthetic, free-threaded, default N
  uv run --python 3.14t python -m bench.exp2_scaling --out \$R --workload synthetic
  # 3. Experiment 2, synthetic, larger-N check
  uv run --python 3.14t python -m bench.exp2_scaling --out \$R --workload synthetic --n-items 5000000 --tag n5m
  # 4. Experiment 2, Monte Carlo, free-threaded
  uv run --python 3.14t python -m bench.exp2_scaling --out \$R --workload montecarlo
  # 5. Experiment 3, depth sweep, free-threaded
  uv run --python 3.14t python -m bench.exp3_writer  --out \$R --depths 1,2,4,8,16
  # 6. GIL-enabled baseline, Experiment 2 synthetic (same 3.14t interpreter)
  PYTHON_GIL=1 uv run --python 3.14t python -m bench.exp2_scaling --out \$R --workload synthetic --allow-gil --tag gil
" > full_run.log 2>&1 &
tail -f full_run.log
```

Defaults: 30 measured + 3 warm-up repetitions per (condition × thread
count), seeded randomized interleaved execution order, plus verify
(counting-lock / instrumented proxy counts) and memory (tracemalloc)
passes. The CSVs grow continuously - you can `wc -l results/*/*.csv`
to watch progress.

**GIL-enabled baseline.** Step 6 uses the same free-threaded
3.14t interpreter as every other step, with `PYTHON_GIL=1`, so the GIL
is the only difference. Without `--allow-gil` the experiments refuse to
run with the GIL enabled; with it they print a warning and continue. A
default (non-free-threaded) build is rejected even with `--allow-gil`.
`--tag gil` appends `_gil` to the output file names, and each env.json
records `gil_enabled` (true for step 6, false otherwise).

**Result layout.** Per platform, `results/<platform>/` holds one CSV and
one env.json per run, 10 runs and 20 files:

```
exp1_race_counter_<stamp>.csv
exp2_scaling_synthetic_<stamp>.csv
exp2_scaling_synthetic_<stamp>_n5m.csv
exp2_scaling_montecarlo_<stamp>.csv
exp3_writer_{single,chained,depth4,depth8,depth16}_<stamp>.csv
exp2_scaling_synthetic_<stamp>_gil.csv
```

File names carry no platform suffix: the directory name and env.json's
`cpu_model` identify the platform. The analysis tells the runs apart by
their env.json (GIL state, `n_items`, depth), not by file name.

## 5. Collect results

```bash
# from your machine:
mkdir -p results/amd results/intel
gcloud compute scp 'bench-amd:~/freethreaded-python-benchmarks/results/amd/*'     ./results/amd/
gcloud compute scp 'bench-intel:~/freethreaded-python-benchmarks/results/intel/*' ./results/intel/
gcloud compute instances delete bench-amd bench-intel  # stop billing
```

To analyse the collected data, see "Regenerating the results" in README.md.

## Contents

| Path | Purpose |
|---|---|
| `bench/config.py` | protocol constants, runtime-derived thread ladder |
| `bench/envinfo.py` | environment capture (build, GIL state, CPU, governor, GCP machine type); refuses to run with GIL on unless `--allow-gil` |
| `bench/runner.py` | harness: randomized interleaved order, warm-up, per-run CSV streaming, memory wrapper |
| `bench/workloads.py` | pure kernels; count-based Monte Carlo (deterministic reference) |
| `bench/writer_monad.py` | Writer monad (formal definition in docstring, matches manuscript listing) |
| `bench/exp1_race.py` | Exp 1: matched algorithms - racy / locked / imperative thread-local / pure fold / disjoint slot |
| `bench/exp2_scaling.py` | Exp 2: fp_reduce / fp_reduce_genexpr / fp_sum_genexpr / fp_sum_map / fp_sum_listcomp / imp_threadlocal / imp_loop_compensated / coarse / batched / batched_unmatched / fine / queue / no_lock; `--n-items`; verify + memory passes |
| `bench/exp3_writer.py` | Exp 3: lock / indexed / plain record / writer over `--depths`; output-contract validation; verify + memory passes |
| `tests/test_writer_monad.py` | monad- and functor-law verification cited by the manuscript |
| `tests/test_conditions.py` | bit-identity groups, batched lock counts, Exp 3 output contract |
| `tests/test_analysis.py` | analysis consistency: one verdict per pair, no verdict column in contrast tables; needs the analysis dependencies (`requirements.txt`) and skips without them, e.g. on the VMs |
| `tests/test_smoke.py` | CLI smoke tests: each experiment with `--runs 1`, incl. `--depths`, `--n-items`, `--allow-gil`, before a full sweep |