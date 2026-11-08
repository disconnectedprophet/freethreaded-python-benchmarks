"""
Condition-level tests for Experiments 2 and 3: the properties the
matched design depends on, checked directly on the condition
implementations rather than through the CLIs.

Run:
    python -m tests.test_conditions
"""

# Libraries
import functools
import math
import operator

from bench import config
from bench import exp3_writer
from bench.exp2_scaling import make_synthetic_conditions
from bench.workloads import (chunk_bounds, compute_item, exact_sum,
                             partial_sum, partial_sum_compensated)

N = config.N_ITEMS

# Values verified on CPython 3.14t (naive and compensated bit-identity groups).
NAIVE_N500K = 831537362.185985
COMPENSATED_N500K = 831537362.1859643


def _reduce_genexpr(start: int, end: int) -> float:
    return functools.reduce(operator.add,
                            (compute_item(i) for i in range(start, end)),
                            0.0)


def _sum_genexpr(start: int, end: int) -> float:
    return sum(compute_item(i) for i in range(start, end))


# ---------------------------------------------------------------------
# Bit-identity groups
# ---------------------------------------------------------------------
def test_bit_identity_full_range() -> None:
    """At N = 500,000 on the real kernel, each summation algorithm
    gives the SAME float under both iteration drivers. If this breaks,
    the decomposition is no longer clean."""
    loop_naive = partial_sum(0, N)
    gen_naive = _reduce_genexpr(0, N)
    loop_comp = partial_sum_compensated(0, N)
    gen_comp = _sum_genexpr(0, N)
    assert loop_naive == gen_naive == NAIVE_N500K, (loop_naive, gen_naive)
    assert loop_comp == gen_comp == COMPENSATED_N500K, (loop_comp, gen_comp)
    assert exact_sum(0, N) == COMPENSATED_N500K


def test_bit_identity_per_chunk() -> None:
    """Bit identity also holds for every per-thread chunk the sweep
    uses, so the pairwise contrasts are clean at every thread count."""
    for n in config.thread_counts(16):
        for tid in range(n):
            s, e = chunk_bounds(tid, n, N)
            assert partial_sum(s, e) == _reduce_genexpr(s, e), (n, tid)
            assert partial_sum_compensated(s, e) == _sum_genexpr(s, e), \
                (n, tid)


def test_bit_identity_conditions_pairwise() -> None:
    """The four condition functions themselves: the naive pair and the
    compensated pair return identical final results at each n."""
    conds = make_synthetic_conditions()
    for n in (1, 2, 4):
        r = {name: conds[name](n)["result"]
             for name in ("fp_reduce", "fp_reduce_genexpr",
                          "imp_loop_compensated", "fp_sum_genexpr")}
        assert r["fp_reduce"] == r["fp_reduce_genexpr"], (n, r)
        assert r["imp_loop_compensated"] == r["fp_sum_genexpr"], (n, r)


def test_compensated_pair_beats_naive_on_ill_conditioned_input() -> None:
    """On an ill-conditioned input the compensated algorithm agrees
    with math.fsum far more closely than naive addition, under both
    iteration drivers."""
    data = [1.0, 1e100, 1.0, -1e100] * 1000
    exact = math.fsum(data)            # 2000.0

    naive_loop = 0.0
    for x in data:
        naive_loop += x
    naive_gen = functools.reduce(operator.add, (x for x in data), 0.0)

    s = c = 0.0                         # same algorithm as
    for x in data:                      # partial_sum_compensated
        t = s + x
        c += (s - t + x) if abs(s) >= abs(x) else (x - t + s)
        s = t
    comp_loop = s + c
    comp_gen = sum(x for x in data)

    assert naive_loop == naive_gen
    assert comp_loop == comp_gen == exact
    assert abs(naive_loop - exact) >= 1000 * max(abs(comp_loop - exact), 1.0)


def test_synthetic_conditions_report_error_vs_fsum() -> None:
    conds = make_synthetic_conditions()
    ref = exact_sum(0, N)
    for name, fn in conds.items():
        out = fn(2)
        assert out["reference"] == ref, name
        assert out["abs_err_vs_fsum"] == abs(out["result"] - ref), name
        assert out["rel_err_vs_fsum"] == abs(out["result"] - ref) / abs(ref)


def test_compensated_group_bit_identical() -> None:
    """fp_sum_genexpr, fp_sum_map and fp_sum_listcomp produce a
    bit-identical value (full range, every per-thread chunk, and the
    condition functions themselves); the two naive conditions produce a
    bit-identical value to each other."""
    assert (_sum_genexpr(0, N) == sum(map(compute_item, range(N)))
            == sum([compute_item(i) for i in range(N)]))
    for n in config.thread_counts(16):
        for tid in range(n):
            s, e = chunk_bounds(tid, n, N)
            g = _sum_genexpr(s, e)
            assert g == sum(map(compute_item, range(s, e))), (n, tid)
            assert g == sum([compute_item(i) for i in range(s, e)]), (n, tid)
    conds = make_synthetic_conditions()
    for n in (1, 2, 4):
        comp = {c: conds[c](n)["result"]
                for c in ("fp_sum_genexpr", "fp_sum_map", "fp_sum_listcomp",
                          "imp_loop_compensated")}
        assert len(set(comp.values())) == 1, (n, comp)
        naive = {c: conds[c](n)["result"]
                 for c in ("fp_reduce", "fp_reduce_genexpr")}
        assert len(set(naive.values())) == 1, (n, naive)


# ---------------------------------------------------------------------
# batched_lock: matched kernel, same lock count as the unmatched variant
# ---------------------------------------------------------------------
EXPECTED_BATCHED_ACQ = {1: 500, 2: 500, 4: 500, 6: 504, 8: 504,
                        10: 500, 12: 504, 14: 504, 16: 512}


def test_batched_variants_same_lock_acquisitions() -> None:
    conds = make_synthetic_conditions(counting=True)
    for n, expected in EXPECTED_BATCHED_ACQ.items():
        for name in ("batched_lock", "batched_lock_unmatched"):
            out = conds[name](n)
            assert out["user_lock_acquisitions"] == expected, (name, n)
            assert out["measured_lock_acquisitions"] == expected, (name, n)


def test_all_lock_conditions_measured_equals_analytic() -> None:
    conds = make_synthetic_conditions(counting=True)
    for n in (1, 3, 4):
        for name, fn in conds.items():
            out = fn(n)
            if "measured_lock_acquisitions" in out:
                assert (out["measured_lock_acquisitions"]
                        == out["user_lock_acquisitions"]), (name, n)


# ---------------------------------------------------------------------
# Experiment 3 output contract
# ---------------------------------------------------------------------
def test_exp3_output_contract_depths_1_2_4() -> None:
    """Every design returns identical stats and an identical log
    (content and order) at depths 1, 2 and 4, matching the sequential
    reference; the runner-side validator raises on any mismatch."""
    for k in (1, 2, 4):
        for n in (1, 2, 3, 4):
            stats_ref, log_ref, agg_ref = exp3_writer._expected(k, n)
            assert len(log_ref) == k * n
            for name, fn in exp3_writer.make_conditions(k).items():
                out = fn(n)
                out.pop("_validate")()            # raises on mismatch
                assert out["result"] == agg_ref, (k, n, name)
                assert out["log_entries"] == k * n, (k, n, name)
                assert out["n_results"] == n, (k, n, name)


def test_exp3_validator_rejects_wrong_order() -> None:
    stats_ref, log_ref, agg_ref = exp3_writer._expected(2, 3)
    stage_major = log_ref[0::2] + log_ref[1::2]
    try:
        exp3_writer._validate(stats_ref, stage_major, agg_ref,
                              (stats_ref, log_ref, agg_ref))
    except exp3_writer.ContractViolation:
        return
    raise AssertionError("stage-major log was not rejected")


def test_exp3_indexed_two_arrays_and_proxies_counted() -> None:
    for k in (1, 2, 4):
        for name, fn in exp3_writer.make_conditions(k, counting=True).items():
            out = fn(3)
            out.pop("_validate")()
            if name == "indexed":
                assert out["aux_structures"] == 2, k
            for a, m in (("user_lock_acquisitions",
                          "measured_lock_acquisitions"),
                         ("bookkeeping_steps", "measured_bookkeeping_steps"),
                         ("bind_calls", "measured_bind_calls")):
                if m in out:
                    assert out[a] == out[m], (k, name, a)


# Main guard
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all condition tests passed")
