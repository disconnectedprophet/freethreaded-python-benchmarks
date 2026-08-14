#!/usr/bin/env bash
# Setup for a fresh GCP Ubuntu VM: installs uv and a free-threaded
# CPython 3.14 build. Benchmarks are stdlib-only, so no packages are
# installed on the VM; analysis (scipy etc.) runs elsewhere.
set -euo pipefail

# 1. uv (Python toolchain manager)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Free-threaded CPython 3.14 ("3.14t")
uv python install 3.14t

# 3. Verify: must print False (GIL disabled)
uv run --python 3.14t python - <<'PY'
import sys, sysconfig
print("Python:", sys.version)
print("Free-threading build:", bool(sysconfig.get_config_var("Py_GIL_DISABLED")))
print("GIL enabled:", sys._is_gil_enabled())
assert not sys._is_gil_enabled(), "GIL is enabled - aborting"
PY

echo
echo "OK. Run experiments from the repo root, e.g.:"
echo "  uv run --python 3.14t python -m bench.exp1_race --out results/"
