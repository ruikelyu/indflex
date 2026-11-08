"""D3R (dimension reduction): ALF forward model vs. the paper's saved results, and fitting.

Fixtures come from https://github.com/ruikelyu/Data-Driven-Dimension-Reduction
(results/data_<plant><I>ALs.mat parameters and data_rc_<plant><I>ALs.mat responses).
"""

from pathlib import Path

import numpy as np
import pytest

import indflex as fx
from indflex.identify import make_meter_dataset
from indflex.reduce import ALFParams, fit_alf, nrmse_paper, simulate_alf

FIX = Path(__file__).parent / "fixtures" / "d3r"


@pytest.mark.parametrize("plant", ["steelpowder", "steelmaking"])
@pytest.mark.parametrize("n_loads", [1, 2])
def test_alf_matches_paper_responses(plant, n_loads):
    params = np.atleast_2d(np.loadtxt(FIX / f"{plant}_{n_loads}loads_params.csv", delimiter=",")).reshape(4, n_loads)
    alf = ALFParams(*params)
    prices = np.loadtxt(FIX / f"{plant}_cv_price.csv", delimiter=",")
    expected = np.loadtxt(FIX / f"{plant}_{n_loads}loads_cv_response.csv", delimiter=",")
    np.testing.assert_allclose(simulate_alf(alf, prices), expected, atol=1e-6)


def test_fit_one_load_with_free_solver():
    ds = make_meter_dataset(fx.load_plant("cement_golmohamadi2020"))
    alf, info = fit_alf(ds.price_train[:3], ds.load_train[:3], n_loads=1, solver="highs")
    assert info["status"] == "optimal"
    # the plant's daily energy is fixed by its production target; the ALF learns a tight energy band near it
    assert alf.e_min[0] == pytest.approx(ds.load_train[0].sum(), rel=0.02)
    assert alf.e_max[0] == pytest.approx(alf.e_min[0], rel=0.02)
    assert nrmse_paper(simulate_alf(alf, ds.price_cv), ds.load_cv) < 0.12


def test_integer_meter_data():
    prices = np.array([[1.0, 2, 3, 1]])
    loads = np.array([[5, 1, 1, 5]], dtype=np.uint8)  # e.g. MATLAB uint8 data
    alf, _ = fit_alf(prices, loads, n_loads=1, solver="highs")
    np.testing.assert_allclose(simulate_alf(alf, prices)[0], [5, 1, 1, 5], atol=1e-6)
