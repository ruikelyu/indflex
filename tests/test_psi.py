"""PSI (parameter identification): data generation, priors, and a small identification run.

The cement fixtures are the meter data shipped with the paper repository
https://github.com/ruikelyu/Production-Scheduling-Identification (dataset_cement.mat).
Identification needs Gurobi with a full license (free for academics); those tests are
skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

import indflex as fx
from indflex.identify import MSTNParams, make_meter_dataset, nrmse, simulate, train_psi
from indflex.identify.psi import target_from_energy

FIX = Path(__file__).parent / "fixtures" / "psi"
G_CEMENT = np.array([1000 / 2200, 250 / 11000, 300 / 11550, 350 / 11220])


def _gurobi_licensed() -> bool:
    try:
        import gurobipy as gp

        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x = m.addVars(2100)
            m.addConstr(x[0] * x[1] <= 1)
            m.optimize()
        return True
    except Exception:
        return False


needs_gurobi = pytest.mark.skipif(not _gurobi_licensed(), reason="needs Gurobi with a full license")


@pytest.fixture(scope="module")
def cement():
    return make_meter_dataset(fx.load_plant("cement_golmohamadi2020"))


def test_stn_reproduces_paper_dataset(cement):
    np.testing.assert_allclose(cement.load_train, np.loadtxt(FIX / "dataset_cement_train_load.csv", delimiter=","), atol=1e-5)
    np.testing.assert_allclose(cement.load_test, np.loadtxt(FIX / "dataset_cement_test_load.csv", delimiter=","), atol=1e-5)


def test_target_from_energy(cement):
    # true daily product gain of the cement plant: 250 t/h x 23 h = 5.75 kt
    assert target_from_energy(G_CEMENT, cement.load_train) == pytest.approx(5.75, rel=0.05)


TOY = MSTNParams(g=np.array([0.5, 0.25]), p_max=np.array([4.0, 8.0]), s_init=np.array([200, 10, 0.0]),
                 s_target=np.array([-130, 0, 30.0]), s_max=np.array([400, 20, 100.0]))


def test_mstn_forward_meets_target():
    prices = fx.load_prices("2022-07").to_numpy()[:2] / 1000
    loads = simulate(TOY, prices)
    assert loads.shape == (2, 24)
    # producing 30 units through both machines takes 30 * (1/0.5 + 1/0.25) = 180 energy units
    np.testing.assert_allclose(loads.sum(axis=1), 180, rtol=1e-6)


@needs_gurobi
def test_identifies_toy_plant():
    jul = fx.load_prices("2022-07").to_numpy() / 1000
    aug = fx.load_prices("2022-08").to_numpy()[:5] / 1000
    theta, hist = train_psi(TOY.g, jul[:10], simulate(TOY, jul[:10]), n_iter=22)
    np.testing.assert_allclose(theta.p_max, TOY.p_max, rtol=1e-2)
    assert theta.s_target[-1] == pytest.approx(30, rel=1e-2)
    assert nrmse(simulate(theta, aug), simulate(TOY, aug)) < 0.01
