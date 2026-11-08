"""
Analysis for the benchmark suite: every number in the manuscript's
tables is produced by this module from the raw per-run CSVs (and the
env.json written next to each CSV, which identifies the run variant:
GIL state, work size, pipeline depth).

Statistical design (prespecified):

  Experiment 1 (correctness). For each condition, the random variable
  is the loss percentage recorded in a single run:
    error_pct = abs(expected - result)/abs(expected)*100;
  The loss percentage is suitable for comparability between different
  number of threads.

  The suite runs each condition 30 times at each thread count,
  so every condition and thread count has 30 such values in the raw CSV.
  How those 30 values are summarised depends on whether the condition
  can lose increments at all.

  (a) The racy condition does lose them, and how many varies from run to run,
  so it gets a distribution. We report the mean loss with a 95%
  t-interval, and test H0: mean loss = 0 against the one-sided
  alternative that it is greater.

  (b) The safe conditions lose nothing, so all 30 values are exactly 0 and
  there is no variation to summarise. A test would be meaningless here,
  so we report the maximum loss observed across all runs instead: the
  strongest statement the data supports is that not a single increment
  was lost in any run.

  Experiments 2 and 3 (timing). Here the question is not correctness but
  coordination cost, and it comes in two opposite forms: whether one condition is
  faster than another, and whether two conditions perform the same.

  For both, the random variable is the wall time of a single run
  (wall_s column), again 30 values per condition per thread count.

  (a) Is one faster? We compare the two conditions with Welch's t-test,
  which does not assume equal variance, and report Cohen's d and the
  RATIO OF MEANS (b / a). With run-to-run variance this small, d can
  exceed 200 and stops being informative about practical size; the
  ratio of means is the practically interpretable effect size. Each
  pair is tested at every thread count, and testing repeatedly inflates
  the chance of a false positive, so we apply the Holm-Bonferroni
  correction across all tests for one condition pair on one platform
  and workload.

  (b) Are they the same? An ordinary t-test cannot answer this, because
  failing to find a difference is not evidence that none exists. We use
  TOST instead, which tests whether the difference is small enough to be
  practically irrelevant. What counts as small enough is fixed before
  looking at the data: MARGIN_FRAC = 5% of the reference condition's
  mean wall time at that thread count. Equivalence is claimed only when
  the Holm-adjusted TOST p-value is below 0.05. Failing to reject a
  difference is never reported as equivalence.

  Robustness check (ratio-based TOST). The absolute margin above is
  itself estimated from the reference mean. As a robustness check we
  also run TOST on log wall times with the margin +/- ln(1 + MARGIN_FRAC),
  i.e. equivalence of the geometric-mean RATIO within [1/1.05, 1.05],
  which does not depend on an estimated absolute bound. Both TOSTs are
  reported, each Holm-adjusted within its own family.

  Confidence intervals vs decisions. The superiority and equivalence
  DECISIONS (verdict columns) use Holm-adjusted p-values. The confidence
  intervals reported alongside (ci95_unadj, ci90_unadj, ...) are
  per-comparison intervals and are NOT adjusted for multiplicity. Every
  interval column is labelled "_unadj" to make this explicit.

Usage:
    python -m analysis.analyze --data results/amd --out report/amd \\
        --platform "AMD EPYC 7B13 (GCP t2d-standard-16)"
"""

# Libraries
import argparse
import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from bench import config

# Configuration variables
MARGIN_FRAC = config.MARGIN_FRAC
ALPHA = config.ALPHA
LOG_MARGIN = math.log(1.0 + MARGIN_FRAC)
DEFAULT_N_ITEMS = config.N_ITEMS
DESIGNS = ["lock", "indexed", "record", "writer"]


# Helper functions
def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def fmt_p(p: float) -> str | float:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "<.0001" if p < 1e-4 else round(p, 4)


def welch(a: np.ndarray, b: np.ndarray):
    """Welch t-test, two-sided, plus Cohen's d (pooled SD)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.std(ddof=1) ** 2
                    + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    d = abs(a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return t, p, d


def _welch_se_df(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, float("inf")
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return se, df


def tost(a: np.ndarray, b: np.ndarray, margin: float):
    """
    TOST equivalence via two one-sided Welch tests.
    H0: |mean(a)-mean(b)| >= margin.  Returns (p, diff, lo90, hi90);
    the 90% interval is per-comparison (unadjusted).
    """
    se, df = _welch_se_df(a, b)
    diff = a.mean() - b.mean()
    if se == 0:
        p = 0.0 if abs(diff) < margin else 1.0
        return p, diff, diff, diff
    p_upper = stats.t.cdf((diff - margin) / se, df)   # mean diff <  m
    p_lower = stats.t.sf((diff + margin) / se, df)    # mean diff > -m
    p = max(p_upper, p_lower)
    tcrit = stats.t.ppf(1 - ALPHA, df)  # 90% CI
    return p, diff, diff - tcrit * se, diff + tcrit * se


def log_ratio_ci(a: np.ndarray, b: np.ndarray,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Geometric-mean ratio b/a with an unadjusted Welch CI on the log
    scale. Returns (ratio, lo, hi)."""
    la, lb = np.log(a), np.log(b)
    se, df = _welch_se_df(lb, la)
    d = lb.mean() - la.mean()
    if se == 0:
        return math.exp(d), math.exp(d), math.exp(d)
    t = stats.t.ppf(0.5 + level / 2, df)
    return math.exp(d), math.exp(d - t * se), math.exp(d + t * se)


def ci95(x: np.ndarray) -> float:
    """Half-width of the (unadjusted) 95% t-interval for the mean."""
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))


def aux_col(df: pd.DataFrame, field: str) -> pd.Series:
    """Extract a field from the aux JSON column."""
    return df["aux"].map(
        lambda s: json.loads(s).get(field) if isinstance(s, str) and s
        else None)


def measured(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df.phase == "measured"].copy()
    m["wall_s"] = pd.to_numeric(m["wall_s"])
    return m


# ---------------------------------------------------------------------
# Loading: one Dataset per CSV, identified by its env.json
# ---------------------------------------------------------------------
@dataclass
class Dataset:
    experiment: str
    workload: str
    gil: bool
    n_items: int | None
    depth: int | None
    tag: str | None
    path: str
    df: pd.DataFrame

    @property
    def key(self) -> tuple:
        return (self.experiment, self.workload, self.n_items, self.gil)

    @property
    def slug(self) -> str:
        """Table-name fragment, e.g. 'synthetic', 'synthetic_n5000000',
        'synthetic_gil'. Default-N free-threaded runs keep the original
        names so earlier tables remain comparable."""
        s = self.workload
        if (self.experiment == "exp2_scaling" and self.n_items
                and self.n_items != DEFAULT_N_ITEMS):
            s += f"_n{self.n_items}"
        if self.gil:
            s += "_gil"
        return s

    @property
    def label(self) -> str:
        s = self.workload
        if (self.experiment == "exp2_scaling" and self.n_items
                and self.n_items != DEFAULT_N_ITEMS):
            s += f", N = {self.n_items:,}"
        if self.gil:
            s += ", GIL ENABLED baseline"
        return s


def _depth_from_workload(workload: str) -> int | None:
    if workload == "single":
        return 1
    if workload == "chained":
        return 2
    if workload.startswith("depth"):
        return int(workload[len("depth"):])
    return None


def load(data_dir: str) -> list[Dataset]:
    """Load every CSV with its env.json. Two CSVs describing the same
    (experiment, workload, n_items, GIL state) are an error rather than
    a silent choice."""
    out: list[Dataset] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "exp*.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            print(f"warning: {path} has no rows; skipped")
            continue
        env_path = path[: -len(".csv")] + ".env.json"
        env = {}
        if os.path.exists(env_path):
            with open(env_path) as f:
                env = json.load(f)
        exp = df.experiment.iloc[0]
        wl = df.workload.iloc[0]
        n_items = env.get("n_items")
        if exp == "exp2_scaling" and n_items is None:
            n_items = DEFAULT_N_ITEMS     # no n_items in env.json: default N
        depth = env.get("depth")
        if exp == "exp3_writer" and depth is None:
            depth = _depth_from_workload(wl)
        df.attrs["path"] = path
        out.append(Dataset(exp, wl, bool(env.get("gil_enabled", False)),
                           n_items, depth, env.get("tag"), path, df))
    seen: dict = {}
    for ds in out:
        if ds.key in seen:
            raise SystemExit(
                f"ERROR: two runs for the same variant {ds.key}:\n"
                f"  {seen[ds.key]}\n  {ds.path}\n"
                "Move one of them out of the data directory.")
        seen[ds.key] = ds.path
    return out


class Report:
    """Accumulates markdown sections and companion CSVs."""

    def __init__(self, out_dir: str, platform: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.lines = [f"# Analysis Report — {platform}", ""]
        self.tables: list[str] = []

    def section(self, title: str) -> None:
        self.lines += [f"## {title}", ""]

    def subsection(self, title: str) -> None:
        self.lines += [f"### {title}", ""]

    def text(self, s: str) -> None:
        self.lines += [s, ""]

    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(os.path.join(self.out_dir, name + ".csv"), index=False)
        self.tables.append(name)
        self.lines += [df.to_markdown(index=False), ""]

    def save(self) -> str:
        path = os.path.join(self.out_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(self.lines))
        return path


# ---------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------
def analyze_exp1(ds: Dataset, rep: Report) -> None:
    rep.section(f"Experiment 1 — correctness (matched algorithms)"
                f"{' — GIL ENABLED baseline' if ds.gil else ''}")
    sfx = "_gil" if ds.gil else ""
    m = ds.df[ds.df.phase == "measured"].copy()
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
            "ci95_half_unadj": round(ci95(loss), 2),
            "min_loss_pct": round(loss.min(), 2),
            "max_loss_pct": round(loss.max(), 2),
            "t_1samp": None if math.isnan(t) else round(t, 1),
            "p_greater0": fmt_p(p),
        })
    rep.text("Racy condition (`imp_racy`): per-run loss%, one-sample "
             "t-test of H0: mean loss = 0 (one-sided greater). "
             "Min/max are observed per-run extremes; the mean always "
             "lies inside [min, max] by construction.")
    rep.table(pd.DataFrame(rows), f"exp1_racy_loss{sfx}")

    safe = (m[m.condition != "imp_racy"]
            .groupby("condition").error_pct.agg(["count", "max"])
            .reset_index()
            .rename(columns={"count": "runs",
                             "max": "max_loss_pct_any_run"}))
    rep.text("Safe conditions: maximum observed loss across ALL runs "
             "and thread counts (expected exactly 0).")
    rep.table(safe, f"exp1_safe_conditions{sfx}")


# ---------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------
def _scaling_table(m: pd.DataFrame, ref: str | None = None) -> pd.DataFrame:
    """Mean wall time (+ unadjusted 95% CI), speedup and efficiency per
    condition and n; with ref, also the ratio of each condition's mean
    to the reference condition's mean at the same n (slowdown factor)."""
    rows = []
    t1 = {c: g[g.n_threads == 1].wall_s.mean()
          for c, g in m.groupby("condition")}
    ref_mean = ({n: g[g.condition == ref].wall_s.mean()
                 for n, g in m.groupby("n_threads")}
                if ref and ref in set(m.condition) else {})
    for (n, c), g in m.groupby(["n_threads", "condition"]):
        w = g.wall_s.to_numpy()
        row = {
            "n_threads": n, "condition": c,
            "mean_s": round(w.mean(), 4),
            "ci95_unadj": round(ci95(w), 4),
            "speedup": round(t1[c] / w.mean(), 2),
            "efficiency": round(t1[c] / (n * w.mean()), 2),
        }
        if ref_mean:
            row[f"ratio_vs_{ref}"] = round(w.mean() / ref_mean[n], 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_threads", "condition"])


def _pairwise(m: pd.DataFrame, pairs: list[tuple[str, str, str]],
              rep: Report, prefix: str) -> None:
    """
    Run the prespecified comparison plan.
    pairs: (cond_a, cond_b, kind) with kind in {'superiority',
    'equivalence'}. Holm correction within each (pair, kind) family
    across thread counts; for equivalence, the absolute-margin and the
    ratio-based TOST are separate families, each Holm-adjusted.
    """
    present = set(m.condition)
    for a_name, b_name, kind in pairs:
        if a_name not in present or b_name not in present:
            continue
        rows, pvals, pvals_r = [], [], []
        for n in sorted(m.n_threads.unique()):
            a = m[(m.condition == a_name)
                  & (m.n_threads == n)].wall_s.to_numpy()
            b = m[(m.condition == b_name)
                  & (m.n_threads == n)].wall_s.to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            ratio = b.mean() / a.mean()
            if kind == "superiority":
                t, p, d = welch(a, b)
                gr, lo, hi = log_ratio_ci(a, b)
                rows.append({"n_threads": n,
                             f"{a_name}_s": round(a.mean(), 4),
                             f"{b_name}_s": round(b.mean(), 4),
                             "ratio_of_means_b_over_a": round(ratio, 4),
                             "geo_ratio_b_over_a": round(gr, 4),
                             "geo_ratio_ci95_unadj": f"[{lo:.4f}, {hi:.4f}]",
                             "welch_t": round(t, 1), "p_raw": p,
                             "cohen_d": round(d, 2)})
            else:
                margin = MARGIN_FRAC * a.mean()
                p, diff, lo, hi = tost(a, b, margin)
                pr, dlog, llo, lhi = tost(np.log(a), np.log(b), LOG_MARGIN)
                rows.append({"n_threads": n,
                             f"{a_name}_s": round(a.mean(), 4),
                             f"{b_name}_s": round(b.mean(), 4),
                             "ratio_of_means_b_over_a": round(ratio, 4),
                             "margin_s": round(margin, 5),
                             "diff_s": round(diff, 5),
                             "diff_ci90_unadj": f"[{lo:.5f}, {hi:.5f}]",
                             "p_raw": p,
                             "geo_ratio_b_over_a": round(math.exp(-dlog), 4),
                             "geo_ratio_ci90_unadj":
                                 f"[{math.exp(-lhi):.4f}, "
                                 f"{math.exp(-llo):.4f}]",
                             "p_ratio_raw": pr})
                pvals_r.append(pr)
            pvals.append(rows[-1]["p_raw"])
        if not rows:
            continue
        adj = holm(pvals)
        adj_r = holm(pvals_r) if pvals_r else [None] * len(rows)
        for r, pa, par in zip(rows, adj, adj_r):
            r["p_holm"] = fmt_p(pa)
            verdict = pa < ALPHA
            if kind == "superiority":
                r["verdict"] = "different" if verdict else "n.s."
            else:
                r["verdict_abs"] = ("EQUIVALENT (±5%)" if verdict
                                    else "not shown equivalent")
                r["p_ratio_holm"] = fmt_p(par)
                r["verdict_ratio"] = ("EQUIVALENT (ratio ±5%)"
                                      if par < ALPHA
                                      else "not shown equivalent")
                r["p_ratio_raw"] = fmt_p(r["p_ratio_raw"])
            r["p_raw"] = fmt_p(r["p_raw"])
        if kind == "superiority":
            desc = ("Welch two-sided (superiority family); effect size is "
                    "the ratio of arithmetic means b/a; the interval is a "
                    "per-comparison (unadjusted) 95% interval for the "
                    "ratio of geometric means (geo_ratio_b_over_a); "
                    "verdict Holm-adjusted across thread counts.")
        else:
            desc = (f"TOST equivalence — absolute margin ±5% of "
                    f"{a_name} mean AND ratio-based (log-scale) margin "
                    f"±ln(1.05); each verdict Holm-adjusted across thread "
                    f"counts; CIs unadjusted.")
        rep.text(f"**{a_name} vs {b_name}** — {desc}")
        rep.table(pd.DataFrame(rows),
                  f"{prefix}_{a_name}_vs_{b_name}_{kind}")
        xref = _contrast_table_of(a_name, b_name)
        if xref:
            rep.text(f"_This pair also appears in `{prefix}_{xref}`, which "
                     f"reports effect sizes only (means, ratio of means, "
                     f"geo_ratio_ci95_unadj) and no verdict; this table "
                     f"holds the pair's one verdict, which the manuscript "
                     f"cites._")


EXP2_PAIRS = [
    ("fp_reduce", "imp_threadlocal", "equivalence"),
    ("fp_reduce", "coarse_lock", "equivalence"),
    ("fp_reduce", "batched_lock", "equivalence"),
    ("fp_reduce", "fine_lock", "superiority"),
    ("fp_reduce", "queue_agg", "equivalence"),
    ("fp_reduce", "fp_sum_genexpr", "superiority"),
    # Kernel-style contrast pairs: their one verdict lives here; the
    # contrast tables report effect sizes only (see analyze_kernel_style)
    ("fp_sum_genexpr", "fp_sum_map", "superiority"),
    ("fp_sum_map", "fp_sum_listcomp", "superiority"),
    ("imp_threadlocal", "imp_loop_compensated", "superiority"),
    ("fp_reduce_genexpr", "fp_sum_genexpr", "superiority"),
    ("fp_reduce", "fp_sum_map", "superiority"),
    ("fp_reduce", "fp_reduce_genexpr", "superiority"),
    # Kernel confound carried by the unmatched batched variant
    ("batched_lock", "batched_lock_unmatched", "superiority"),
    ("fp_reduce", "batched_lock_unmatched", "superiority"),
    ("fp_reduce", "no_lock", "superiority"),
]


NAIVE_GROUP = ["fp_reduce", "fp_reduce_genexpr"]
COMPENSATED_GROUP = ["imp_loop_compensated", "fp_sum_genexpr",
                     "fp_sum_map", "fp_sum_listcomp"]
SINGLE_FACTOR = [
    ("fp_sum_genexpr", "fp_sum_map", "generator-frame overhead"),
    ("fp_sum_map", "fp_sum_listcomp", "list materialisation"),
    ("fp_reduce", "imp_threadlocal", "combination step"),
    ("imp_threadlocal", "imp_loop_compensated",
     "Neumaier compensation written in Python bytecode"),
]
MULTI_FACTOR = [
    ("fp_reduce", "fp_sum_genexpr",
     "iteration driver (bytecode for loop vs C-consumed generator); "
     "generator frame; summation algorithm (naive vs Neumaier); "
     "accumulator representation (Python float objects vs C double)"),
    ("fp_reduce", "fp_reduce_genexpr",
     "iteration driver (bytecode for loop vs C-consumed generator); "
     "generator frame; per-item Python-level operator.add call via "
     "functools.reduce (vs bytecode +=)"),
    ("fp_reduce_genexpr", "fp_sum_genexpr",
     "summation algorithm (naive vs Neumaier); accumulator "
     "representation (per-item Python-level operator.add call plus a new "
     "float object, vs a C double accumulator) - not separable with the "
     "standard library; generator frame held identical"),
    ("fp_reduce", "fp_sum_map",
     "iteration driver (bytecode for loop vs C-level iterator); "
     "summation algorithm (naive vs Neumaier); accumulator representation "
     "(Python float objects vs C double)"),
]


def _contrast_table_of(a: str, b: str) -> str | None:
    """Name of the contrast table (single- or multi-factor) that lists
    the pair, or None."""
    if any((a, b) == (x, y) for x, y, _ in SINGLE_FACTOR):
        return "contrasts_single_factor"
    if any((a, b) == (x, y) for x, y, _ in MULTI_FACTOR):
        return "contrasts_multi_factor"
    return None


def _verdict_table_of(a: str, b: str) -> str:
    """Name suffix of the pairwise table holding the pair's one verdict.
    Raises unless the pair has exactly one EXP2_PAIRS entry: zero entries
    would leave it without a verdict, and two (e.g. differing only in
    kind) would give it two verdicts in two tables."""
    kinds = [k for x, y, k in EXP2_PAIRS if (x, y) == (a, b)]
    if len(kinds) != 1:
        raise ValueError(
            f"contrast pair {a} vs {b} has {len(kinds)} EXP2_PAIRS entries "
            f"{kinds}: every SINGLE_FACTOR/MULTI_FACTOR pair requires "
            f"exactly one EXP2_PAIRS entry")
    return f"{a}_vs_{b}_{kinds[0]}"


def _contrast_xrefs(pairs: list, present: set, prefix: str) -> list[str]:
    """One cross-reference line per contrast-table pair, naming the
    table where that pair's verdict lives."""
    out = []
    for a, b, _ in pairs:
        if not {a, b} <= present:
            continue
        v = _verdict_table_of(a, b)
        out.append(f"_`{a}` vs `{b}`: verdict in `{prefix}_{v}` "
                   f"(its one prespecified test, Holm within this "
                   f"pair's thread counts); the manuscript cites that "
                   f"table for the verdict._")
    return out


def _contrast_table(m: pd.DataFrame, pairs: list[tuple[str, str, str]],
                    factor_col: str) -> pd.DataFrame:
    """Effect sizes only, per thread count: the two means, the ratio of
    arithmetic means b/a, the ratio of geometric means with its
    unadjusted 95% CI, and the factor(s) differing.
    No test and no verdict: each pair's one verdict comes from its
    EXP2_PAIRS entry. Means are rounded as in the pairwise tables so
    the same number reads identically in both."""
    rows = []
    present = set(m.condition)
    for a_name, b_name, factor in pairs:
        if a_name not in present or b_name not in present:
            continue
        for n in sorted(m.n_threads.unique()):
            a = m[(m.condition == a_name) & (m.n_threads == n)].wall_s
            b = m[(m.condition == b_name) & (m.n_threads == n)].wall_s
            if len(a) < 2 or len(b) < 2:
                continue
            a, b = a.to_numpy(), b.to_numpy()
            gr, lo, hi = log_ratio_ci(a, b)
            rows.append({"a": a_name, "b": b_name, factor_col: factor,
                         "n_threads": n,
                         "a_s": round(a.mean(), 4), "b_s": round(b.mean(), 4),
                         "ratio_of_means_b_over_a": round(b.mean() / a.mean(),
                                                          4),
                         "geo_ratio_b_over_a": round(gr, 4),
                         "geo_ratio_ci95_unadj": f"[{lo:.4f}, {hi:.4f}]"})
    return pd.DataFrame(rows)


def analyze_kernel_style(m: pd.DataFrame, rep: Report, prefix: str) -> None:
    """Kernel-style conditions: the numerical axis (exactly controlled
    by bit-identity groups) and runtime contrasts, separated into
    single-factor and multi-factor pairs."""
    if not {"fp_reduce", "fp_sum_genexpr"} <= set(m.condition):
        return
    rep.subsection("Kernel style: numerical axis and runtime contrasts")

    if any("rel_err_vs_fsum" in str(x) for x in m.aux.dropna()):
        e = m.copy()
        e["abs_err"] = pd.to_numeric(aux_col(e, "abs_err_vs_fsum"))
        e["rel_err"] = pd.to_numeric(aux_col(e, "rel_err_vs_fsum"))
        e = e.dropna(subset=["rel_err"])
        err = (e.groupby(["condition", "n_threads"])
               .agg(runs=("rel_err", "size"),
                    max_abs_err=("abs_err", "max"),
                    max_rel_err=("rel_err", "max"),
                    mean_rel_err=("rel_err", "mean"))
               .reset_index())
        err.insert(1, "bit_identity_group", err.condition.map(
            lambda c: "naive" if c in NAIVE_GROUP
            else "compensated" if c in COMPENSATED_GROUP else "—"))
        for col in ("max_abs_err", "max_rel_err", "mean_rel_err"):
            err[col] = err[col].map(lambda v: f"{v:.3e}")
        rep.text(
            "**Numerical axis, exactly controlled.** Relative error vs the "
            "exactly rounded math.fsum reference for every condition, per "
            "run, from the aux column. Bit-identity groups (per-chunk "
            "partials bit-for-bit identical, asserted in "
            "tests/test_conditions.py): naive = "
            + ", ".join(f"`{c}`" for c in NAIVE_GROUP)
            + "; compensated = "
            + ", ".join(f"`{c}`" for c in COMPENSATED_GROUP) + ".")
        rep.table(err, f"{prefix}_numerical_error")

    single = _contrast_table(m, SINGLE_FACTOR, "factor")
    if len(single):
        rep.text(
            "**Runtime, single-factor contrasts only.** Each pair differs "
            "in exactly the named factor. Effect sizes only: means; "
            "effect size is the ratio of arithmetic means b/a; the "
            "interval is a per-comparison (unadjusted) 95% interval for "
            "the ratio of geometric means (geo_ratio_b_over_a). No "
            "verdict here; each pair's one verdict is in the table named "
            "below it.")
        rep.table(single, f"{prefix}_contrasts_single_factor")
        for line in _contrast_xrefs(SINGLE_FACTOR, set(m.condition), prefix):
            rep.text(line)

    multi = _contrast_table(m, MULTI_FACTOR, "factors_differing")
    if len(multi):
        rep.text(
            "**Runtime, multi-factor contrasts.** These pairs differ in "
            "several factors at once and are NOT attributed to one cause. "
            + " ".join(f"`{a}` vs `{b}` differ in: {f}."
                       for a, b, f in MULTI_FACTOR)
            + " Effect sizes only: means; effect size is the ratio of "
              "arithmetic means b/a; the interval is a per-comparison "
              "(unadjusted) 95% interval for the ratio of geometric means "
              "(geo_ratio_b_over_a). No verdict here; each pair's one "
              "verdict is in the table named below it.")
        rep.table(multi, f"{prefix}_contrasts_multi_factor")
        for line in _contrast_xrefs(MULTI_FACTOR, set(m.condition), prefix):
            rep.text(line)


def _verify_table(df: pd.DataFrame, fields: list[tuple[str, str]],
                  rep: Report, name: str, what: str) -> None:
    v = df[df.phase == "verify"].copy()
    if not len(v):
        return
    rows = []
    for analytic, meas in fields:
        v["a"] = aux_col(v, analytic)
        v["m"] = aux_col(v, meas)
        w = v.dropna(subset=["m"])
        for c, g in w.groupby("condition"):
            rows.append({"quantity": analytic, "condition": c,
                         "cells": len(g),
                         "matches": int((g.a == g.m).sum())})
    if not rows:
        return
    t = pd.DataFrame(rows)
    rep.text(f"{what}: {int(t.matches.sum())}/{int(t.cells.sum())} "
             f"instrumented cells match the analytic count exactly.")
    rep.table(t, name)


def analyze_exp2(ds: Dataset, rep: Report) -> None:
    rep.section(f"Experiment 2 — scaling ({ds.label})")
    prefix = f"exp2_{ds.slug}"
    m = measured(ds.df)
    m["error_pct"] = pd.to_numeric(m["error_pct"])

    rep.text("Wall time (mean ± unadjusted 95% CI), speedup and "
             "efficiency; each condition's efficiency uses its OWN "
             "single-thread mean. ratio_vs_fp_reduce = condition mean / "
             "fp_reduce mean at the same n (every condition, including "
             "no_lock and fp_sum_genexpr).")
    rep.table(_scaling_table(m, ref="fp_reduce"), f"{prefix}_scaling")

    agg = {"max_error_pct": ("error_pct", "max")}
    has_err = any("rel_err_vs_fsum" in (s or "") for s in
                  m.aux.dropna().astype(str))
    if has_err:
        m["rel_err_vs_fsum"] = pd.to_numeric(aux_col(m, "rel_err_vs_fsum"))
        agg["max_rel_err_vs_fsum"] = ("rel_err_vs_fsum", "max")
    err = m.groupby("condition").agg(**agg).reset_index()
    if has_err:
        err["max_rel_err_vs_fsum"] = err.max_rel_err_vs_fsum.map(
            lambda v: f"{v:.3e}")
    rep.text("Correctness: maximum error vs. reference across all runs "
             "(no_lock is expected to be wrong; float conditions show "
             "rounding/reordering error, quantified against the exactly "
             "rounded math.fsum reference where available).")
    rep.table(err, f"{prefix}_correctness")

    _pairwise(m, EXP2_PAIRS, rep, prefix)
    analyze_kernel_style(m, rep, prefix)

    _verify_table(ds.df, [("user_lock_acquisitions",
                           "measured_lock_acquisitions")],
                  rep, f"{prefix}_lock_verify",
                  "Lock-count verification pass")

    mem = ds.df[ds.df.phase == "memory"].copy()
    if len(mem):
        mem["peak_kib"] = (pd.to_numeric(aux_col(mem,
                           "tracemalloc_peak_bytes")) / 1024).round(1)
        piv = mem.pivot_table(index="n_threads", columns="condition",
                              values="peak_kib").reset_index()
        rep.text("Python-level peak allocation (tracemalloc, KiB) — "
                 "one instrumented pass per cell.")
        rep.table(piv, f"{prefix}_memory")


# ---------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------
def analyze_exp3(ds: Dataset, rep: Report) -> None:
    rep.section(f"Experiment 3 — structured accumulation "
                f"({ds.workload}, depth k = {ds.depth}"
                f"{', GIL ENABLED baseline' if ds.gil else ''})")
    prefix = f"exp3_{ds.slug}"
    m = measured(ds.df)

    rep.table(_scaling_table(m), f"{prefix}_scaling")

    m["rel_diff"] = pd.to_numeric(aux_col(m, "sum_rel_diff"))
    rep.text(f"Correctness & float non-associativity: max |relative "
             f"difference| of the parallel aggregate vs the sequential "
             f"reference across all runs = {m.rel_diff.max():.3e}. "
             f"Every recorded row passed the output-contract validation "
             f"(stats, log content and order, aggregate) — a failing run "
             f"raises before its row is written.")

    pairs = [(a, b, "equivalence")
             for i, a in enumerate(DESIGNS) for b in DESIGNS[i + 1:]]
    _pairwise(m, pairs, rep, prefix)

    _verify_table(ds.df, [
        ("user_lock_acquisitions", "measured_lock_acquisitions"),
        ("bookkeeping_steps", "measured_bookkeeping_steps"),
        ("bind_calls", "measured_bind_calls"),
    ], rep, f"{prefix}_proxy_verify",
        "Structural-proxy verification pass")

    mem = ds.df[ds.df.phase == "memory"].copy()
    if len(mem):
        mem["peak_kib"] = (pd.to_numeric(aux_col(mem,
                           "tracemalloc_peak_bytes")) / 1024).round(1)
        piv = mem.pivot_table(index="n_threads", columns="condition",
                              values="peak_kib").reset_index()
        rep.text("Peak allocation (tracemalloc, KiB): Writer's object "
                 "cost vs record vs shared structures.")
        rep.table(piv, f"{prefix}_memory")


def writer_log_copies(k: int) -> int:
    """Per-worker log elements copied by the k - 1 binds (writer_monad
    cost model): bind j concatenates (j-1) + 1 elements and the
    constructor copies the j-element result, i.e. 2j per bind:
    sum_{j=2..k} 2j = k^2 + k - 2. The O(k^2) prediction."""
    return k * k + k - 2


def analyze_depth_sweep(dsets: list[Dataset], rep: Report,
                        sfx: str) -> None:
    """Runtime and structural proxies by depth; the O(k^2)
    Writer prediction against the measured writer - record excess."""
    dsets = sorted((d for d in dsets if d.depth is not None),
                   key=lambda d: d.depth)
    if len(dsets) < 2:
        return
    rep.section(f"Experiment 3 — pipeline-depth sweep"
                f"{' (GIL ENABLED baseline)' if sfx else ''}")
    rows = []
    for ds in dsets:
        m = measured(ds.df)
        for (n, c), g in m.groupby(["n_threads", "condition"]):
            w = g.wall_s.to_numpy()
            a0 = json.loads(g.aux.iloc[0]) if isinstance(
                g.aux.iloc[0], str) else {}
            rows.append({
                "depth": ds.depth, "design": c, "n_threads": n,
                "mean_s": round(w.mean(), 5),
                "ci95_unadj": round(ci95(w), 5),
                "aux_structures": a0.get("aux_structures"),
                "bookkeeping_steps": a0.get("bookkeeping_steps"),
                "bind_calls": a0.get("bind_calls"),
                "log_entries": a0.get("log_entries"),
            })
    sweep = pd.DataFrame(rows).sort_values(["n_threads", "design", "depth"])
    rep.text("Runtime and structural proxies by depth. aux_structures, "
             "bookkeeping_steps and bind_calls are emitted by the "
             "implementation (aux column) and checked against "
             "instrumented counts in each depth's verify pass.")
    rep.table(sweep, f"exp3_depth_sweep{sfx}")

    # O(k^2) prediction: writer - record excess vs k^2 + k - 2
    fit_rows, pred_rows = [], []
    for n in sorted(sweep.n_threads.unique()):
        ks, ex, lo, hi = [], [], [], []
        for ds in dsets:
            m = measured(ds.df)
            w = m[(m.condition == "writer") & (m.n_threads == n)].wall_s
            r = m[(m.condition == "record") & (m.n_threads == n)].wall_s
            if len(w) < 2 or len(r) < 2:
                continue
            w, r = w.to_numpy(), r.to_numpy()
            se, df = _welch_se_df(w, r)
            t = stats.t.ppf(0.975, df) if se else 0.0
            d = w.mean() - r.mean()
            ks.append(ds.depth)
            ex.append(d)
            lo.append(d - t * se)
            hi.append(d + t * se)
        if len(ks) < 3:
            continue
        ks_a, ex_a = np.array(ks, float), np.array(ex)
        x = np.array([writer_log_copies(int(k)) for k in ks], float)
        quad = stats.linregress(x, ex_a)
        lin = stats.linregress(ks_a, ex_a)
        fit_rows.append({
            "n_threads": n, "depths": ",".join(map(str, ks)),
            "slope_s_per_copy": f"{quad.slope:.3e}",
            "slope_se": f"{quad.stderr:.3e}",
            "slope_p_raw": quad.pvalue,
            "r2_quadratic_model": round(quad.rvalue ** 2, 3),
            "r2_linear_in_k": round(lin.rvalue ** 2, 3),
        })
        for k, e, l, h, xc in zip(ks, ex, lo, hi, x):
            pred_rows.append({
                "n_threads": n, "depth": k,
                "writer_log_copies_per_worker": int(xc),
                "measured_excess_s": f"{e:.3e}",
                "excess_ci95_unadj": f"[{l:.3e}, {h:.3e}]",
                "predicted_excess_s":
                    f"{quad.intercept + quad.slope * xc:.3e}",
            })
    if not fit_rows:
        return
    slope_adj = holm([r["slope_p_raw"] for r in fit_rows])
    for r, pa in zip(fit_rows, slope_adj):
        r["slope_p_raw"] = fmt_p(r["slope_p_raw"])
        r["slope_p_holm"] = fmt_p(pa)
    rep.text("O(k²) prediction. Writer's bind copies the accumulated log, "
             "so the per-worker copying is k² + k − 2 elements. The "
             "measured writer − record mean wall-time difference is "
             "regressed on that count (quadratic model) and, for "
             "comparison, on k (linear model). Uncertainty columns: "
             "slope_se is the regression standard error (standard errors "
             "are not adjusted for multiplicity); slope_p_holm is "
             "Holm-adjusted within this fit's family across thread counts, "
             "and slope_p_raw is not adjusted; excess_ci95_unadj is a "
             "per-comparison interval, not adjusted for multiplicity.")
    rep.table(pd.DataFrame(fit_rows), f"exp3_depth_ok2_fit{sfx}")
    rep.table(pd.DataFrame(pred_rows), f"exp3_depth_ok2_pred{sfx}")


# ---------------------------------------------------------------------
# GIL-enabled vs free-threaded
# ---------------------------------------------------------------------
def analyze_gil(ft: Dataset, gil: Dataset, rep: Report) -> None:
    """Every condition, every n present in both runs: free-threaded vs
    GIL-enabled mean wall time on the SAME interpreter build. Holm per
    condition across thread counts. For exp2, each run's ratio to its
    own fp_reduce shows whether a kernel-style effect depends on the
    GIL."""
    rep.subsection(f"{ft.experiment} / {ft.workload}"
                   + (f" (N = {ft.n_items:,})" if ft.n_items else ""))
    mf, mg = measured(ft.df), measured(gil.df)
    refs = {}
    if ft.experiment == "exp2_scaling":
        for tag, m in (("ft", mf), ("gil", mg)):
            refs[tag] = {n: g[g.condition == "fp_reduce"].wall_s.mean()
                         for n, g in m.groupby("n_threads")}
    rows = []
    for c in sorted(set(mf.condition) & set(mg.condition)):
        cand, pv = [], []
        for n in sorted(set(mf.n_threads) & set(mg.n_threads)):
            a = mf[(mf.condition == c) & (mf.n_threads == n)].wall_s
            b = mg[(mg.condition == c) & (mg.n_threads == n)].wall_s
            if len(a) < 2 or len(b) < 2:
                continue
            a, b = a.to_numpy(), b.to_numpy()
            _, p, _ = welch(a, b)
            r = {"condition": c, "n_threads": n,
                 "free_threaded_s": round(a.mean(), 5),
                 "gil_s": round(b.mean(), 5),
                 "gil_over_ft": round(b.mean() / a.mean(), 3)}
            if refs:
                r["ft_ratio_vs_fp_reduce"] = round(
                    a.mean() / refs["ft"][n], 4)
                r["gil_ratio_vs_fp_reduce"] = round(
                    b.mean() / refs["gil"][n], 4)
            cand.append(r)
            pv.append(p)
        for r, p, pa in zip(cand, pv, holm(pv)):
            r["p_raw"] = fmt_p(p)
            r["p_holm"] = fmt_p(pa)
            r["verdict"] = "different" if pa < ALPHA else "n.s."
            rows.append(r)
    if rows:
        rep.table(pd.DataFrame(rows), f"gil_vs_ft_{ft.experiment}_"
                  f"{ft.slug}")


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
             f"±{MARGIN_FRAC:.0%} of reference mean (absolute TOST) and "
             f"±ln(1+{MARGIN_FRAC}) on log wall time (ratio TOST, "
             f"robustness check); Holm correction within each comparison "
             f"family across thread counts.")
    rep.text("**Multiplicity families.** A family is one condition pair "
             "(and one test kind) on one platform, workload and run "
             "variant, across its thread counts (nine on a 16-core "
             "machine). Holm adjusts only within that family. Adding a "
             "condition therefore adds new families, not members of an "
             "existing family, so it imposes no multiplicity penalty on "
             "the existing pairs.")
    rep.text("**Intervals vs decisions.** All superiority and equivalence "
             "DECISIONS (verdict columns) are Holm-adjusted. Confidence "
             "intervals (columns ending in `_unadj`) are per-comparison "
             "intervals and are NOT adjusted for multiplicity.")
    rep.text("Source files: "
             + ", ".join(f"`{os.path.basename(d.path)}`" for d in data))

    by = {(d.experiment, d.workload, d.gil, d.n_items): d for d in data}

    for d in data:
        if d.experiment == "exp1_race":
            analyze_exp1(d, rep)
    exp2 = sorted((d for d in data if d.experiment == "exp2_scaling"),
                  key=lambda d: (d.gil, d.workload != "synthetic",
                                 d.n_items or 0))
    for d in exp2:
        analyze_exp2(d, rep)
    exp3 = sorted((d for d in data if d.experiment == "exp3_writer"),
                  key=lambda d: (d.gil, d.depth or 0))
    for d in exp3:
        analyze_exp3(d, rep)
    analyze_depth_sweep([d for d in exp3 if not d.gil], rep, "")
    analyze_depth_sweep([d for d in exp3 if d.gil], rep, "_gil")

    pairs = [(by[(e, w, False, n)], d) for (e, w, g, n), d in by.items()
             if g and (e, w, False, n) in by]
    if pairs:
        rep.section("GIL-enabled baseline vs free-threaded (same "
                    "free-threaded build; PYTHON_GIL=1 vs unset)")
        rep.text("gil_over_ft = GIL-enabled mean / free-threaded mean. "
                 "For Experiment 2, *_ratio_vs_fp_reduce is each run's "
                 "condition mean over its OWN fp_reduce mean at the same "
                 "n: if a kernel-style effect (e.g. fp_sum_genexpr) has "
                 "the same ratio under both, it is not a free-threading "
                 "effect. Welch two-sided, Holm per condition across "
                 "thread counts.")
        for ft, gil in sorted(pairs, key=lambda p: p[0].key):
            analyze_gil(ft, gil, rep)

    path = rep.save()
    print("report:", path)
    print(f"tables: {len(rep.tables)} CSVs in {args.out}")


# Main guard
if __name__ == "__main__":
    main()
