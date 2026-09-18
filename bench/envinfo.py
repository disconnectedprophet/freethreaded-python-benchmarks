"""
Environment capture for reproducibility reporting (CPython build and configuration, 
OS details, CPU model and topology, frequency policy, processor affinity).

capture() collects everything into a JSON-serialisable dict. The
runner.py writes it as env.json next to the results CSV (`results/<amd/intel>/expN_*.env.json`), 
so every result carries a complete record of the machine and interpreter that
produced it.

Standard library-only (by design): the benchmark VMs need no third-party packages.
"""

# Libraries
import json
import os
import platform
import subprocess
import sys
import sysconfig
import time


def _read(path: str) -> str | None:
    """
    Return the contents of ``path`` as text, or ``None`` if unreadable.

    Leading and trailing whitespace is stripped. The whole file is read into
    memory at once, hence it is meant for small files.

    ``None`` is returned for any filesystem-level failure and for content
    that cannot be decoded. The two cases are not distinguished and the underlying
    exception is discarded.

    The file is decoded using the platform's default encoding, so the same
    file may read successfully on one system and fail on another.
    """
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _cmd(args: list[str]) -> str | None:
    """
    Run ``args`` and return its stdout as text, or ``None`` on failure.

    ``None`` is returned if the executable is missing, if the process
    cannot be started, or if it does not finish within 10 seconds.

    Output is decoded using the platform's default encoding.
    """
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str | None:
    """
    Return the CPU model string, or ``None`` if it cannot be determined.

    Reads the first ``model name`` field from ``/proc/cpuinfo``. Linux and
    x86 specific.
    """
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def capture(seed: int | None = None,
            extra: dict | None = None) -> dict:
    """
    Collect the full execution environment.

    Parameters:
        seed: Execution-order seed used for this session (recorded).
        extra: Experiment-specific fields added to the returned dict at top
        level; keys already present are overwritten.
    Returns:
        dict ready for json.dump.
    """
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None

    env = {
        "captured_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "seed": seed,
        # Interpreter
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "gil_enabled": gil_enabled,
        "free_threading_build": bool(
            sysconfig.get_config_var("Py_GIL_DISABLED")
        ),
        "config_args": sysconfig.get_config_var("CONFIG_ARGS"),
        "cc": sysconfig.get_config_var("CC"),
        # OS
        "platform": platform.platform(),
        "kernel": platform.release(),
        "os_release": _read("/etc/os-release"),
        # CPU
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_affinity": affinity,
        "lscpu": _cmd(["lscpu"]),
        # cpufreq: bare metal only; None under a hypervisor, which does not
        # expose frequency control to the guest (all three are None on GCP)
        "scaling_governor": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpuinfo_max_freq_khz": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        ),
        "cpuinfo_min_freq_khz": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
        ),
        # Virtualisation
        "hypervisor": _cmd(["systemd-detect-virt"]),
        "gcp_machine_type": _cmd([
            "curl", "-s", "-m", "2",
            "-H", "Metadata-Flavor: Google",
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/machine-type",
        ]),
    }
    if extra:
        env.update(extra)
    return env


def write(path: str, seed: int | None = None,
          extra: dict | None = None) -> dict:
    """Capture the environment and write it to path as JSON."""
    env = capture(seed=seed, extra=extra)
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    return env


def assert_free_threading() -> None:
    """
    Abort with a clear message if the GIL is active.

    Every experiment calls this first, so a mis-configured interpreter
    can never silently produce GIL-masked results.
    """
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if gil_enabled:
        sys.exit(
            "ERROR: the GIL is enabled in this interpreter.\n"
            "Run under a free-threading CPython build (e.g. `uv run "
            "--python 3.14t ...`) with PYTHON_GIL unset or =0."
        )
