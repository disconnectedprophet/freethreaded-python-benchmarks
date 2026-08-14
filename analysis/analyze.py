"""
Analysis for the benchmark suite: every number in the manuscript's
tables is produced by this module from the raw per-run CSVs.

Statistical design (prespecified; matrix items C1, C2, C3):

  Experiment 1 (correctness). The random variable is the per-run
  loss percentage of each condition (error_pct column). For the racy
  condition we report mean loss with a 95% t-interval and a one-sample
  t-test of H0: mean loss = 0 (one-sided greater).
  Safe conditions are reported as max observed loss across all runs
  (exactly 0 expected).

  Superiority tests (Exp 2, FP vs. fine lock etc.). Welch's t-test on
  wall times, Cohen's d, with Holm-Bonferroni correction applied
  within each comparison family (a family = one condition pair across
  all thread counts on one platform/workload).

  Equivalence tests (Exp 2 FP vs. thread-local/coarse; Exp 3 pairwise).
  TOST (two one-sided Welch tests) with a prespecified equivalence
  margin of MARGIN_FRAC = 5% of the reference condition's mean wall
  time in that cell. Equivalence is claimed only when the Holm-
  adjusted TOST p-value < 0.05; we additionally report the 90% CI of
  the mean difference (equivalently: TOST at alpha=.05). Failure to
  reject difference is never reported as equivalence.

Usage:
    python -m analysis.analyze --data results-amd/ --out report-amd/ \
        --platform "AMD EPYC 7B13 (GCP t2d-standard-16)"
"""

# Libraries
import argparse
import glob
import json
import math
import os
import numpy as np
import pandas as pd
from scipy import stats

# Configuration variables
MARGIN_FRAC = 0.05 # prespecified equivalence margin: ±5% of ref mean
ALPHA = 0.05


# Helper functions
def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def welch(a: np.ndarray, b: np.ndarray):
    """Welch t-test, two-sided, plus Cohen's d (pooled SD)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.std(ddof=1) ** 2
                    + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    d = abs(a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return t, p, d


def tost(a: np.ndarray, b: np.ndarray, margin: float):
    """
    TOST equivalence via two one-sided Welch tests.
    H0: |mean(a)-mean(b)| >= margin.  Returns (p, diff, lo90, hi90).
    """
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    diff = a.mean() - b.mean()
    if se == 0:
        p = 0.0 if abs(diff) < margin else 1.0
        return p, diff, diff, diff
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p_upper = stats.t.cdf((diff - margin) / se, df)   # mean diff <  m
    p_lower = stats.t.sf((diff + margin) / se, df)    # mean diff > -m
    p = max(p_upper, p_lower)
    tcrit = stats.t.ppf(1 - ALPHA, df)                # 90% CI
    return p, diff, diff - tcrit * se, diff + tcrit * se


def ci95(x: np.ndarray) -> float:
    """Half-width of the 95% t-interval for the mean."""
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))


def load(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all CSVs, keyed by experiment/workload."""
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "exp*.csv"))):
        df = pd.read_csv(path)
        key = f"{df.experiment.iloc[0]}/{df.workload.iloc[0]}"
        out[key] = df
        df.attrs["path"] = path
    return out


def aux_col(df: pd.DataFrame, field: str) -> pd.Series:
    """Extract a field from the aux JSON column."""
    return df["aux"].map(
        lambda s: json.loads(s).get(field) if isinstance(s, str) and s
        else None)


class Report:
    """Accumulates markdown sections and companion CSVs."""

    def __init__(self, out_dir: str, platform: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.lines = [f"# v2 Analysis Report — {platform}", ""]

    def section(self, title: str) -> None:
        self.lines += [f"## {title}", ""]

    def text(self, s: str) -> None:
        self.lines += [s, ""]

    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(os.path.join(self.out_dir, name + ".csv"), index=False)
        self.lines += [df.to_markdown(index=False), ""]

    def save(self) -> str:
        path = os.path.join(self.out_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(self.lines))
        return path


# Experiment 1
def analyze_exp1(df: pd.DataFrame, rep: Report) -> None:
    rep.section("Experiment 1 — correctness (matched algorithms)")
    m = df[df.phase == "measured"].copy()
    m["error_pct"] = pd.to_numeric(m["error_pct"])

    rows = []
    for n, g in m[m.condition == "imp_racy"].groupby("n_threads"):
        loss = g.error_pct.to_numpy()
        if n == 1 or loss.std(ddof=1) == 0:
            t = p = float("nan")
        else:
            t, p = stats.ttest_1samp(loss, 0.0, alternative="greater")
        rows.append({
            "n_threads": n, "runs": len(loss),
            "mean_loss_pct": round(loss.mean(), 2),
            "ci95_half": round(ci95(loss), 2),
            "min_loss_pct": round(loss.min(), 2),
            "max_loss_pct": round(loss.max(), 2),
            "t_1samp": None if math.isnan(t) else round(t, 1),
            "p_greater0": ("—" if math.isnan(p)
                           else "<.0001" if p < 1e-4 else round(p, 4)),
        })
    rep.text("Racy condition (`imp_racy`): per-run loss%, one-sample "
             "t-test of H0: mean loss = 0 (one-sided greater). "
             "Min/max are observed per-run extremes; the mean always "
             "lies inside [min, max] by construction.")
    rep.table(pd.DataFrame(rows), "exp1_racy_loss")

    safe = (m[m.condition != "imp_racy"]
            .groupby("condition").error_pct.agg(["count", "max"])
            .reset_index()
            .rename(columns={"count": "runs",
                             "max": "max_loss_pct_any_run"}))
    rep.text("Safe conditions: maximum observed loss across ALL runs "
             "and thread counts (expected exactly 0).")
    rep.table(safe, "exp1_safe_conditions")


# Experiment 2
def _scaling_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t1 = {c: g[g.n_threads == 1].wall_s.mean()
          for c, g in m.groupby("condition")}
    for (n, c), g in m.groupby(["n_threads", "condition"]):
        w = g.wall_s.to_numpy()
        rows.append({
            "n_threads": n, "condition": c,
            "mean_s": round(w.mean(), 4),
            "ci95": round(ci95(w), 4),
            "speedup": round(t1[c] / w.mean(), 2),
            "efficiency": round(t1[c] / (n * w.mean()), 2),
        })
    return pd.DataFrame(rows).sort_values(["n_threads", "condition"])


def _pairwise(m: pd.DataFrame, pairs: list[tuple[str, str, str]],
              rep: Report, prefix: str) -> None:
    """
    Run the prespecified comparison plan.
    pairs: (cond_a, cond_b, kind) with kind in {'superiority',
    'equivalence'}.  Holm correction within each (pair, kind) family
    across thread counts.
    """
    for a_name, b_name, kind in pairs:
        rows, pvals = [], []
        for n in sorted(m.n_threads.unique()):
            a = m[(m.condition == a_name)
                  & (m.n_threads == n)].wall_s.to_numpy()
            b = m[(m.condition == b_name)
                  & (m.n_threads == n)].wall_s.to_numpy()
            if kind == "superiority":
                t, p, d = welch(a, b)
                rows.append({"n_threads": n,
                             f"{a_name}_s": round(a.mean(), 4),
                             f"{b_name}_s": round(b.mean(), 4),
                             "welch_t": round(t, 1), "p_raw": p,
                             "cohen_d": round(d, 2)})
            else:
                margin = MARGIN_FRAC * a.mean()
                p, diff, lo, hi = tost(a, b, margin)
                rows.append({"n_threads": n,
                             f"{a_name}_s": round(a.mean(), 4),
                             f"{b_name}_s": round(b.mean(), 4),
                             "margin_s": round(margin, 5),
                             "diff_s": round(diff, 5),
                             "ci90": f"[{lo:.5f}, {hi:.5f}]",
                             "p_raw": p})
            pvals.append(rows[-1]["p_raw"])
        adj = holm(pvals)
        for r, pa in zip(rows, adj):
            r["p_holm"] = "<.0001" if pa < 1e-4 else round(pa, 4)
            verdict = pa < ALPHA
            r["verdict"] = (
                ("different" if verdict else "n.s.")
                if kind == "superiority"
                else ("EQUIVALENT (±5%)" if verdict
                      else "not shown equivalent"))
            r["p_raw"] = ("<.0001" if r["p_raw"] < 1e-4
                          else round(r["p_raw"], 4))
        rep.text(f"**{a_name} vs {b_name}** — "
                 f"{'Welch two-sided (superiority family)' if kind == 'superiority' else 'TOST equivalence, margin ±5% of ' + a_name + ' mean'}, "
                 f"Holm-adjusted across thread counts.")
        rep.table(pd.DataFrame(rows),
                  f"{prefix}_{a_name}_vs_{b_name}_{kind}")


def analyze_exp2(df: pd.DataFrame, rep: Report, workload: str) -> None:
    rep.section(f"Experiment 2 — scaling ({workload})")
    m = df[df.phase == "measured"].copy()
    m["wall_s"] = pd.to_numeric(m["wall_s"])
    m["error_pct"] = pd.to_numeric(m["error_pct"])

    rep.text("Wall time (mean ± 95% CI), speedup and efficiency; each "
             "condition's efficiency uses its OWN single-thread mean "
             "(fixes v1's mixed baseline).")
    rep.table(_scaling_table(m), f"exp2_{workload}_scaling")

    err = (m.groupby("condition").error_pct
           .agg(["max"]).reset_index()
           .rename(columns={"max": "max_error_pct"}))
    rep.text("Correctness: maximum error vs. reference across all runs "
             "(no_lock is expected to be wrong; float conditions may "
             "show ~1e-12 reordering noise, quantified separately).")
    rep.table(err, f"exp2_{workload}_correctness")

    pairs = [
        ("fp_reduce", "imp_threadlocal", "equivalence"),
        ("fp_reduce", "coarse_lock", "equivalence"),
        ("fp_reduce", "fine_lock", "superiority"),
    ]
    if "batched_lock" in m.condition.unique():
        pairs.insert(2, ("fp_reduce", "batched_lock", "equivalence"))
        pairs.append(("fp_reduce", "queue_agg", "equivalence"))
    if "fp_sum_genexpr" in m.condition.unique():
        pairs.append(("fp_reduce", "fp_sum_genexpr", "superiority"))
    _pairwise(m, pairs, rep, f"exp2_{workload}")

    v = df[df.phase == "verify"].copy()
    if len(v):
        v["analytic"] = aux_col(v, "user_lock_acquisitions")
        v["measured"] = aux_col(v, "measured_lock_acquisitions")
        v = v.dropna(subset=["measured"])
        v["match"] = v.analytic == v.measured
        rep.text(f"Lock-count verification pass: "
                 f"{int(v.match.sum())}/{len(v)} cells match the "
                 f"analytic count exactly.")

    mem = df[df.phase == "memory"].copy()
    if len(mem):
        mem["peak_kib"] = (pd.to_numeric(aux_col(mem,
                           "tracemalloc_peak_bytes")) / 1024).round(1)
        piv = mem.pivot_table(index="n_threads", columns="condition",
                              values="peak_kib").reset_index()
        rep.text("Python-level peak allocation (tracemalloc, KiB) — "
                 "one instrumented pass per cell.")
        rep.table(piv, f"exp2_{workload}_memory")


# Experiment 3
def analyze_exp3(df: pd.DataFrame, rep: Report, stage: str) -> None:
    rep.section(f"Experiment 3 — structured accumulation ({stage})")
    m = df[df.phase == "measured"].copy()
    m["wall_s"] = pd.to_numeric(m["wall_s"])

    rep.table(_scaling_table(m), f"exp3_{stage}_scaling")

    m["rel_diff"] = pd.to_numeric(aux_col(m, "sum_rel_diff"))
    rep.text(f"Correctness & float non-associativity: max |relative "
             f"difference| of the parallel aggregate vs the sequential "
             f"reference across all runs = "
             f"{m.rel_diff.max():.3e} (matrix item B4).")

    conds = ["lock", "indexed", "record", "writer"]
    pairs = [(a, b, "equivalence")
             for i, a in enumerate(conds) for b in conds[i + 1:]]
    _pairwise(m, pairs, rep, f"exp3_{stage}")

    mem = df[df.phase == "memory"].copy()
    if len(mem):
        mem["peak_kib"] = (pd.to_numeric(aux_col(mem,
                           "tracemalloc_peak_bytes")) / 1024).round(1)
        piv = mem.pivot_table(index="n_threads", columns="condition",
                              values="peak_kib").reset_index()
        rep.text("Peak allocation (tracemalloc, KiB): Writer's object "
                 "cost vs record vs shared structures.")
        rep.table(piv, f"exp3_{stage}_memory")


# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", default="unknown platform")
    args = ap.parse_args()

    data = load(args.data)
    rep = Report(args.out, args.platform)
    rep.text(f"Prespecified: alpha={ALPHA}, equivalence margin = "
             f"±{MARGIN_FRAC:.0%} of reference mean, Holm correction "
             f"within each comparison family across thread counts. "
             f"Source files: "
             f"{[os.path.basename(d.attrs['path']) for d in data.values()]}")

    if "exp1_race/counter" in data:
        analyze_exp1(data["exp1_race/counter"], rep)
    for wl in ("synthetic", "montecarlo"):
        key = f"exp2_scaling/{wl}"
        if key in data:
            analyze_exp2(data[key], rep, wl)
    for stage in ("single", "chained"):
        key = f"exp3_writer/{stage}"
        if key in data:
            analyze_exp3(data[key], rep, stage)

    path = rep.save()
    print("report:", path)

# Maing guard
if __name__ == "__main__":
    main()
