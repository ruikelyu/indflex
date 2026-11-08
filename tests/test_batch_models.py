"""RTN / cRTN ports are checked against the original MATLAB code of
https://github.com/ruikelyu/Efficient-Scheduling-of-Discrete-Industrial-Processes

Reference values in tests/fixtures/batch were produced by
tools/matlab_reference/make_reference_crtn.m (YALMIP + Gurobi, MIP gap 1e-6),
with a linear energy-cost objective and the price vectors stored next to them.
"""

from pathlib import Path

import numpy as np
import pytest

import indflex as fx

FIX = Path(__file__).parent / "fixtures" / "batch"

# (case, plant, dt [h], heats)
CASES = [
    ("small_s1", "eaf_small_case", 1, 2),
    ("small_s2", "eaf_small_case", 1, 2),
    ("steel_30min_h1_d26", "steelmaking_zhang2017", 0.5, 1),
    ("steel_30min_h2_d26", "steelmaking_zhang2017", 0.5, 2),
    ("steel_30min_h3_d26", "steelmaking_zhang2017", 0.5, 3),
    ("steel_15min_h2_d26", "steelmaking_zhang2017", 0.25, 2),
]
BUILD = {"crtn": fx.build_crtn, "rtn": fx.build_rtn}


@pytest.mark.parametrize("model", ["crtn", "rtn"])
@pytest.mark.parametrize("case,plant,dt,heats", CASES)
def test_matches_matlab(case, plant, dt, heats, model):
    price = np.loadtxt(FIX / f"{case}_price.csv", delimiter=",")
    ref_cost, _, ref_problem = np.loadtxt(FIX / f"{case}_{model}_cost.csv", delimiter=",")
    r = fx.solve(BUILD[model](fx.load_batch_plant(plant), price, heats, dt=dt), mip_gap=1e-6)
    assert r.status == "optimal"
    if case == "steel_30min_h1_d26" and model == "rtn":
        # Known issue in the MATLAB RTN: with one heat, `sum(S_HT(:,:)) <= 1` sums the
        # 1xT row over time (not over heats), so MATLAB reports the case infeasible.
        assert ref_problem != 0
        return
    assert r.cost == pytest.approx(ref_cost, rel=1e-6)
    ref_energy = np.loadtxt(FIX / f"{case}_{model}_energy.csv", delimiter=",")
    assert r.load.sum() == pytest.approx(ref_energy.sum(), rel=1e-6)


def test_crtn_avoids_rtn_rounding_error():
    """Small case, scenario 2: whole-slot RTN tasks cost more than the cRTN schedule."""
    plant = fx.load_batch_plant("eaf_small_case")
    price = np.loadtxt(FIX / "small_s2_price.csv", delimiter=",")
    rtn = fx.solve(fx.build_rtn(plant, price, 2))
    crtn = fx.solve(fx.build_crtn(plant, price, 2))
    assert crtn.cost < rtn.cost - 1


def test_crtn_binaries_do_not_grow_with_heats():
    plant = fx.load_batch_plant("steelmaking_zhang2017")
    price = np.repeat(fx.daily_price("2022-07", 26), 2)
    n_bin = {}
    for heats in (1, 2, 3):
        for name, build in BUILD.items():
            n_bin[name, heats], _ = fx.count_vars(build(plant, price, heats, dt=0.5))
    assert n_bin["crtn", 1] == n_bin["crtn", 3]
    assert n_bin["rtn", 1] < n_bin["rtn", 2] < n_bin["rtn", 3]


def test_batch_and_continuous_loaders_are_separate():
    with pytest.raises(ValueError):
        fx.load_plant("steelmaking_zhang2017")
    with pytest.raises(ValueError):
        fx.load_batch_plant("steel_powder_lu2021")


def test_heat_schedule_small_case():
    """Thesis Ch. 2 illustrative case, scenario 2: cRTN melts heats in 0-1.5 h and 1.5-3 h."""
    from indflex.models import crtn_progress, heat_schedule

    plant = fx.load_batch_plant("eaf_small_case")
    price = np.loadtxt(FIX / "small_s2_price.csv", delimiter=",")
    m = fx.build_crtn(plant, price, 2)
    fx.solve(m)
    sched = heat_schedule(m).sort_values("start")
    np.testing.assert_allclose(sched[["start", "end"]].to_numpy(), [[0, 1.5], [1.5, 3.0]], atol=1e-3)
    t, progress = crtn_progress(m)
    assert progress.max() == pytest.approx(1.0) and progress[-1] == pytest.approx(0.0)

    m = fx.build_rtn(plant, price, 2)
    fx.solve(m)
    ends = sorted(heat_schedule(m)["end"])
    assert ends == [2.0, 4.0]  # whole slots: the second heat finishes in an expensive hour


def test_rtn_tie_break_keeps_energy_cost():
    plant = fx.load_batch_plant("eaf_small_case")
    price = np.r_[np.loadtxt(FIX / "small_s2_price.csv", delimiter=","), 200, 200]
    plain = fx.solve(fx.build_rtn(plant, price, 2), mip_gap=0)
    tied = fx.solve(fx.build_rtn(plant, price, 2, tie_break=1e-2), mip_gap=0)
    assert tied.cost == pytest.approx(plain.cost, abs=1e-6)  # reported cost excludes the tie-breaker
