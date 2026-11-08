"""
Analysis-consistency tests.

One verdict per pair: every tested pair has exactly one EXP2_PAIRS
entry, which produces its only verdict. The single-/multi-factor
contrast tables report effect sizes only (means, ratio of means,
unadjusted ratio CI) and carry no verdict column; where a pair appears
in both a pairwise table and a contrast table, the shared numbers must
agree exactly.

Needs the analysis dependencies (requirements.txt). Where they are not
installed (e.g. on the benchmark VMs, which are standard-library only),
each test is skipped with a one-line notice rather than failing.

Run:
    python -m tests.test_analysis
"""

# Libraries
import os
import random
import tempfile


class _Skipped(Exception):
    pass


def _analysis():
    try:
        import pandas as pd
        from analysis import analyze
    except ImportError as exc:
        raise _Skipped(f"analysis dependencies not installed ({exc})")
    return pd, analyze


def _synthetic_measured(pd, conditions: list[str]):
    """Deterministic fake measured rows: 30 runs x 9 thread counts per
    condition, with condition-specific effects of different sizes (some
    detectable, some not), so both verdicts occur."""
    rng = random.Random(1234)
    effect = {c: 1.0 + 0.004 * i for i, c in enumerate(sorted(conditions))}
    rows = []
    for c in conditions:
        for n in (1, 2, 4, 6, 8, 10, 12, 14, 16):
            base = 0.07 / n
            for rep in range(30):
                rows.append({"condition": c, "n_threads": n,
                             "phase": "measured", "aux": "",
                             "wall_s": base * effect[c]
                             * (1.0 + rng.gauss(0.0, 0.01))})
    return pd.DataFrame(rows)


def _run_tables(pd, analyze, out: str):
    contrast = analyze.SINGLE_FACTOR + analyze.MULTI_FACTOR
    conds = sorted({c for a, b, _ in contrast for c in (a, b)}
                   | {c for a, b, _ in analyze.EXP2_PAIRS for c in (a, b)})
    m = _synthetic_measured(pd, conds)
    rep = analyze.Report(out, "test")
    analyze._pairwise(m, analyze.EXP2_PAIRS, rep, "t")
    analyze.analyze_kernel_style(m, rep, "t")
    ctab = pd.concat([pd.read_csv(os.path.join(out, f"t_contrasts_{k}.csv"),
                                  dtype=str)
                      for k in ("single_factor", "multi_factor")])
    return rep, ctab


def test_duplicated_pairs_agree() -> None:
    """For every pair in both a pairwise table and a contrast table, the
    means, the ratio of arithmetic means and the ratio of geometric
    means (and, for superiority pairs, its CI) agree exactly at every
    thread count; both tables carry both ratios; and the report
    cross-references the verdict table."""
    pd, analyze = _analysis()
    with tempfile.TemporaryDirectory() as out:
        rep, ctab = _run_tables(pd, analyze, out)
        report = "\n".join(rep.lines)
        compared = 0
        for a, b, kind in analyze.EXP2_PAIRS:
            c = ctab[(ctab.a == a) & (ctab.b == b)]
            if not len(c):
                continue
            pw = pd.read_csv(os.path.join(out, f"t_{a}_vs_{b}_{kind}.csv"),
                             dtype=str)
            assert len(pw) == len(c) == 9, (a, b, len(pw), len(c))
            for tab in (pw, c):
                assert {"ratio_of_means_b_over_a",
                        "geo_ratio_b_over_a"} <= set(tab.columns), (a, b)
            for (_, x), (_, y) in zip(pw.iterrows(), c.iterrows()):
                assert x.n_threads == y.n_threads, (a, b)
                assert x[f"{a}_s"] == y.a_s, (a, b, x.n_threads)
                assert x[f"{b}_s"] == y.b_s, (a, b, x.n_threads)
                assert (x.ratio_of_means_b_over_a
                        == y.ratio_of_means_b_over_a), (a, b, x.n_threads)
                assert (x.geo_ratio_b_over_a
                        == y.geo_ratio_b_over_a), (a, b, x.n_threads)
                if kind == "superiority":
                    assert (x.geo_ratio_ci95_unadj
                            == y.geo_ratio_ci95_unadj), (a, b)
                compared += 1
            assert f"verdict in `t_{a}_vs_{b}_{kind}`" in report, (a, b)
        assert compared, "no rows compared"


def test_no_verdict_column_in_contrast_tables() -> None:
    pd, analyze = _analysis()
    with tempfile.TemporaryDirectory() as out:
        _, ctab = _run_tables(pd, analyze, out)
        assert len(ctab), "contrast tables empty"
        bad = [c for c in ctab.columns
               if "verdict" in c or c.startswith("p_")]
        assert not bad, bad


def test_superiority_tables_carry_ratio_of_means() -> None:
    pd, analyze = _analysis()
    with tempfile.TemporaryDirectory() as out:
        _run_tables(pd, analyze, out)
        for a, b, kind in analyze.EXP2_PAIRS:
            if kind != "superiority":
                continue
            t = pd.read_csv(os.path.join(out, f"t_{a}_vs_{b}_{kind}.csv"))
            assert "ratio_of_means_b_over_a" in t.columns, (a, b)


def test_every_contrast_pair_has_exactly_one_verdict_entry() -> None:
    """Every pair in SINGLE_FACTOR or MULTI_FACTOR has exactly one
    EXP2_PAIRS entry, which is its one verdict. No exemptions."""
    _, analyze = _analysis()
    for a, b, _ in analyze.SINGLE_FACTOR + analyze.MULTI_FACTOR:
        entries = [k for x, y, k in analyze.EXP2_PAIRS if (x, y) == (a, b)]
        assert len(entries) == 1, (a, b, entries)


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
    print("analysis tests done")
