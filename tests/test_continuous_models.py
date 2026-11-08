"""STN / LSTN ports are checked against the original MATLAB/YALMIP code.

Reference values in tests/fixtures were produced by
tools/matlab_reference/make_reference.m (YALMIP + Gurobi). Optimal schedules can
be non-unique, so we compare the objective and the daily energy, which are unique.
"""

from pathlib import Path

import numpy as np
import pytest

import indflex as fx

FIX = Path(__file__).parent / "fixtures"


def ref(case):
    cost = float(np.loadtxt(FIX / f"{case}_cost.csv", delimiter=","))
    energy = np.loadtxt(FIX / f"{case}_energy.csv", delimiter=",")
    return cost, energy


# (fixture, plant, month, day, relax, cost scale, energy scale)
# The legacy cement script works in MWh and $/kWh, so its cost is in k$ and energy in MWh.
CASES = [
    ("stn_steelpowder_jul23", "steel_powder_lu2021", "2022-07", 23, False, 1, 1),
    ("lstn_steelpowder_jul23", "steel_powder_lu2021", "2022-07", 23, True, 1, 1),
    ("stn_cement_jul01", "cement_golmohamadi2020", "2022-07", 1, False, 1e3, 1e3),
    ("stn_cement_jul02", "cement_golmohamadi2020", "2022-07", 2, False, 1e3, 1e3),
    ("stn_cement_jul03", "cement_golmohamadi2020", "2022-07", 3, False, 1e3, 1e3),
    ("lstn_model_aug01", "steel_powder_lu2021_lstn", "2022-08", 1, True, 1, 1),
]


@pytest.mark.parametrize("case,plant,month,day,relax,cscale,escale", CASES)
def test_matches_matlab(case, plant, month, day, relax, cscale, escale):
    cost_ref, energy_ref = ref(case)
    r = fx.solve(fx.build_stn(fx.load_plant(plant), fx.daily_price(month, day), relax=relax))
    assert r.status == "optimal"
    assert r.cost == pytest.approx(cost_ref * cscale, rel=1e-6)
    assert r.load.sum() == pytest.approx(energy_ref.sum() * escale, rel=1e-6)


def test_fleet_matches_matlab():
    cost_ref, energy_ref = ref("lstn_fleet5_aug05")
    r = fx.solve(fx.build_fleet(fx.load_fleet(5), fx.daily_price("2022-08", 5)))
    assert r.cost == pytest.approx(cost_ref, rel=1e-6)
    assert r.load.sum() == pytest.approx(energy_ref.sum(), rel=1e-6)


def test_lstn_is_lower_bound_of_stn():
    plant, price = fx.load_plant("steel_powder_lu2021"), fx.daily_price("2022-07", 10)
    stn = fx.solve(fx.build_stn(plant, price))
    lstn = fx.solve(fx.build_lstn(plant, price))
    assert lstn.n_binary == 0 and stn.n_binary > 0
    assert lstn.cost <= stn.cost + 1e-6
    assert (stn.cost - lstn.cost) / stn.cost < 0.02  # LSTN is a tight approximation here


def test_targets_and_capacity_respected():
    plant = fx.load_plant("steel_powder_lu2021")
    r = fx.solve(fx.build_lstn(plant, fx.daily_price("2022-07", 5)))
    for k, s in enumerate(plant.stages):
        lvl = r.storage.iloc[k].to_numpy()
        assert lvl.min() >= -1e-6 and lvl.max() <= s.s_max + 1e-6
        assert lvl[-1] >= s.s_init + s.s_target_change - 1e-6


def test_prices_shape():
    p = fx.load_prices("2022-07")
    assert p.shape == (31, 24)
