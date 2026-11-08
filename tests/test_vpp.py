"""VPP bidding and fast disaggregation vs. the paper code.

Fixtures were produced by the MATLAB code of
https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants
(day 21: maxProfit_1, then maxProfit_t at the first step and the first 2 minutes of
fastControl_prepare / fastControl_implement).
"""

from pathlib import Path

import numpy as np
import pytest

from indflex.coordinate import allocate, allocate_lp, bid, load_vpp_case, merit_order, simulate_hour, to_resources

FIX = Path(__file__).parent / "fixtures" / "vpp"


def ref(name):
    return np.loadtxt(FIX / name, delimiter=",")


@pytest.fixture(scope="module")
def case():
    return load_vpp_case("day21")


@pytest.fixture(scope="module")
def day_ahead(case):
    return bid(case)


@pytest.fixture(scope="module")
def rebid(case, day_ahead):
    return bid(case, cur_slot=0, E_cur=case.energy_init, bid_cur=(day_ahead.bid_p[0], day_ahead.bid_r[0]), rest=1.0)


def test_day_ahead_bidding(day_ahead):
    assert day_ahead.profit == pytest.approx(float(ref("dayahead_profit.csv")), rel=1e-7)
    bids = ref("dayahead_bids.csv")
    np.testing.assert_allclose(day_ahead.bid_p, bids[:, 0], atol=1e-6)
    np.testing.assert_allclose(day_ahead.bid_r, bids[:, 1], atol=1e-6)


def test_rebid_and_shadow_prices(rebid):
    assert rebid.profit == pytest.approx(float(ref("rebid_profit.csv")), rel=1e-7)
    # the paper code stores the negated duals; ours are d(profit)/d(stored energy)
    np.testing.assert_allclose(rebid.lam, -ref("rebid_lambda_paper.csv"), atol=1e-6)
    np.testing.assert_allclose(rebid.p_none_reg, np.atleast_1d(ref("rebid_p_none_reg.csv")), atol=1e-6)


def test_fast_control_reproduces_paper(case, day_ahead, rebid):
    h = simulate_hour(case, 0, day_ahead.bid_p[0], day_ahead.bid_r[0], rebid.lam, case.energy_init,
                      rebid.p_none_reg, eta_convention="paper", n_steps=60)
    np.testing.assert_allclose(h.p_dis - h.p_ch, ref("first2min_p_alloc.csv"), atol=1e-5)


def test_merit_order_equals_lp(case, day_ahead, rebid):
    seg = merit_order(case, rebid.lam, case.energy_init, 0, rebid.p_none_reg)
    for delta in case.signal[:600:150]:
        req = day_ahead.bid_p[0] + day_ahead.bid_r[0] * delta
        fast, lp = allocate(seg, req), allocate_lp(seg, req)
        assert seg.cost @ fast == pytest.approx(seg.cost @ lp, abs=1e-6)
        assert fast.sum() == pytest.approx(req, abs=1e-9)


def test_transposed_eta_is_never_worse(case, day_ahead, rebid):
    """True cost = usage cost - value of the change in stored energy (with E' = η p)."""
    lam = rebid.lam

    def true_cost(d, ch):
        return case.pr_dis @ d + case.pr_ch @ ch - lam @ (case.eta_ch @ ch - case.eta_dis @ d)

    for delta in case.signal[:900:90]:
        req = day_ahead.bid_p[0] + day_ahead.bid_r[0] * delta
        costs = {}
        for conv in ("transpose", "paper"):
            seg = merit_order(case, lam, case.energy_init, 0, rebid.p_none_reg, conv)
            costs[conv] = true_cost(*to_resources(seg, allocate(seg, req), case.n))
        assert costs["transpose"] <= costs["paper"] + 1e-6
