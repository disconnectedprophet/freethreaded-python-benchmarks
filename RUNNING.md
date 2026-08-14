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
--zone=europe-west4-b --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud --boot-disk-size=20GB
```

## 2. Setup (on each VM)

```bash
# from your machine:
gcloud compute scp --recurse v2/ bench-amd:~/
gcloud compute ssh bench-amd
cd ~/v2 && bash setup_gcp.sh  # must print "GIL enabled: False"
```

## 3. Smoke test (~2 minutes; do this first, report any problem)

```bash
cd ~/v2
uv run --python 3.14t python -m bench.exp1_race    --out smoke/ --runs 2 --warmup 1 --max-threads 4
uv run --python 3.14t python -m bench.exp2_scaling --out smoke/ --runs 2 --warmup 1 --max-threads 4
uv run --python 3.14t python -m bench.exp3_writer  --out smoke/ --runs 2 --warmup 1 --max-threads 4
uv run --python 3.14t python -m tests.test_monads

# or, all three CLIs in one shot (--runs 1, asserts CSV/env.json shape):
uv run --python 3.14t python -m tests.test_smoke
```

## 4. Full session (run on BOTH machines, same order)

```bash
cd ~/v2
nohup bash -c '
  set -e
  uv run --python 3.14t python -m bench.exp1_race    --out results/
  uv run --python 3.14t python -m bench.exp2_scaling --out results/ --workload synthetic
  uv run --python 3.14t python -m bench.exp2_scaling --out results/ --workload montecarlo
  uv run --python 3.14t python -m bench.exp3_writer  --out results/
' > full_run.log 2>&1 &
tail -f full_run.log
```

Defaults: 30 measured + 3 warm-up repetitions per (condition × thread
count), seeded randomized interleaved execution order, plus verify
(counting-lock) and memory (tracemalloc) passes. Expected duration:
roughly 1.5–3 h per machine (exp1's `imp_locked` and exp2's
`fine_lock` dominate). The CSV grows continuously - you can `wc -l
results/*.csv` to watch progress.

## 5. Collect results

```bash
# from your machine:
gcloud compute scp 'bench-amd:~/v2/results/*'   ./results-amd/
gcloud compute scp 'bench-intel:~/v2/results/*' ./results-intel/
gcloud compute instances delete bench-amd bench-intel  # stop billing
```

## Contents

| Path | Purpose |
|---|---|
| `bench/config.py` | protocol constants, hardware-agnostic thread ladder |
| `bench/envinfo.py` | environment capture (build, CPU, governor, GCP machine type); refuses to run with GIL on |
| `bench/runner.py` | harness: randomized interleaved order, warm-up, per-run CSV streaming, memory wrapper |
| `bench/workloads.py` | pure kernels; count-based Monte Carlo (deterministic reference) |
| `bench/monads.py` | Writer monad (formal definition in docstring, matches manuscript listing) |
| `bench/exp1_race.py` | Exp 1: matched algorithms - racy / locked / imperative thread-local / pure fold / disjoint slot |
| `bench/exp2_scaling.py` | Exp 2: fp_reduce / imp_threadlocal / coarse / batched / fine / queue / no_lock; verify + memory passes |
| `bench/exp3_writer.py` | Exp 3: lock / indexed / plain record / writer, single- and two-stage; memory pass |
| `tests/test_monads.py` | monad-law verification cited by the manuscript |
| `tests/test_smoke.py` | CLI smoke tests: each experiment with `--runs 1`, before a full sweep |
```
