"""Precompute the data behind the interactive demos on the project web page.

    python tools/export_web_demos.py OUTPUT.js

Writes a JavaScript file that sets ``window.INDFLEX_DEMOS`` (a few tens of KB). The page
draws the five explorers from it in the browser, so no server is needed. Requires Gurobi
for the PSI demo (everything else uses HiGHS).
"""

import json
import sys
from pathlib import Path

import numpy as np

import indflex as fx
from indflex.coordinate import bid, load_vpp_case, merit_order, simulate_hour
from indflex.identify import MSTNParams, build_inverse_day, make_meter_dataset, simulate
from indflex.identify.psi import theta_of
from indflex.models import crtn_progress, heat_schedule
from indflex.reduce import fit_alf, nrmse_paper, simulate_alf


def r(x, nd=3):
    """Round numbers (and nested lists) to keep the file small."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def lstn_demo():
    """STN vs LSTN: one steel powder plant, every day of July 2022."""
    plant = fx.load_plant("steel_powder_lu2021")
    prices = fx.load_prices("2022-07").to_numpy()
    days = []
    for d, price in enumerate(prices, start=1):
        stn, lstn = fx.solve(fx.build_stn(plant, price)), fx.solve(fx.build_lstn(plant, price))
        days.append({"day": d, "price": r(price, 1), "stn": r(stn.load, 1), "lstn": r(lstn.load, 1),
                     "stn_cost": round(stn.cost, 2), "lstn_cost": round(lstn.cost, 2)})
    n_bin, _ = fx.count_vars(fx.build_stn(plant, prices[0]))
    return {"plant": plant.description, "stn_binaries": n_bin, "days": days}


def crtn_demo():
    """The one-furnace example of the cRTN paper: 2 heats, two price scenarios."""
    plant = fx.load_batch_plant("eaf_small_case")
    scenarios = {"1": [100, 100, 100, 100, 200, 200], "2": [100, 100, 100, 200, 200, 100]}
    out = {}
    for key, base in scenarios.items():
        price = np.r_[base, 200, 200]  # two expensive hours so every delivery is visible
        m_rtn, m_crtn = fx.build_rtn(plant, price, 2, tie_break=1e-2), fx.build_crtn(plant, price, 2)
        rtn, crtn = fx.solve(m_rtn, mip_gap=0), fx.solve(m_crtn)
        status = [[round(m_rtn.S[h, t]()) for t in range(len(price))] for h in range(2)]
        t, prog = crtn_progress(m_crtn)
        out[key] = {
            "price": r(price, 0),
            "rtn": {"cost": round(rtn.cost, 2), "status": status, "binaries": rtn.n_binary,
                    "heats": heat_schedule(m_rtn).round(3).to_dict("records")},
            "crtn": {"cost": round(crtn.cost, 2), "t": r(t), "progress": r(prog), "binaries": crtn.n_binary,
                     "heats": heat_schedule(m_crtn).round(3).to_dict("records")},
        }
    steel = fx.load_batch_plant("steelmaking_zhang2017")
    price5 = np.repeat(fx.daily_price("2022-07", 26), 12)
    size = [{"heats": h, "rtn": fx.count_vars(fx.build_rtn(steel, price5, h, dt=1 / 12))[0],
             "crtn": fx.count_vars(fx.build_crtn(steel, price5, h, dt=1 / 12))[0]} for h in range(1, 11)]
    return {"scenarios": out, "size": size}


def psi_demo():
    """Identify a 2-machine plant from one day of meter data, then predict a demand-response event."""
    true = MSTNParams(g=np.array([0.5, 0.25]), p_max=np.array([6.0, 8.0]), s_init=np.array([200.0, 10, 0]),
                      s_target=np.array([-130.0, 0, 30]), s_max=np.array([400.0, 20, 100]))
    prices = fx.load_prices("2022-07").to_numpy()[:10] / 1000
    meter = simulate(true, prices)
    from pyomo.contrib.appsi.solvers import Gurobi

    m = build_inverse_day(true.g, 24, 1.0, float(meter.max()))
    for t in range(24):
        m.price[t], m.E_obs[t] = prices[0, t], meter[0, t]
    opt = Gurobi()
    opt.gurobi_options["NonConvex"] = 2
    opt.config.time_limit = 60
    opt.solve(m)
    theta = theta_of(m)
    base = fx.load_prices("2022-08").to_numpy()[0] / 1000
    events = []
    for start in range(0, 20):
        ev = base.copy()
        ev[start:start + 5] = 0.3
        events.append({"start": start, "identified": r(simulate(theta, ev[None])[0], 2),
                       "true": r(simulate(true, ev[None])[0], 2)})
    return {
        "meter": r(meter[0], 2), "meter_price": r(prices[0] * 1000, 1),
        "params": {"true_p_max": r(true.p_max, 2), "identified_p_max": r(theta.p_max, 2),
                   "true_target": float(true.s_target[-1]), "identified_target": round(float(theta.s_target[-1]), 2)},
        "base_price": r(base * 1000, 1), "baseline": r(simulate(theta, base[None])[0], 2), "events": events,
    }


def d3r_demo():
    """A 1-load ALF fitted to the cement plant (3 days), checked on 10 unseen days."""
    plant = fx.load_plant("cement_golmohamadi2020")
    ds = make_meter_dataset(plant)
    alf, info = fit_alf(ds.price_train[:3], ds.load_train[:3], n_loads=1, solver="highs")
    pred = simulate_alf(alf, ds.price_cv)
    n_bin, n_cont = fx.count_vars(fx.build_stn(plant, ds.price_train[0] * 1000))
    days = [{"day": 22 + d, "price": r(ds.price_cv[d] * 1000, 1), "stn": r(ds.load_cv[d], 2), "alf": r(pred[d], 2),
             "nrmse": round(100 * nrmse_paper(pred[d], ds.load_cv[d]), 1)} for d in range(len(pred))]
    return {"alf": {"p_min": round(float(alf.p_min[0]), 2), "p_max": round(float(alf.p_max[0]), 2),
                    "e_min": round(float(alf.e_min[0]), 1), "e_max": round(float(alf.e_max[0]), 1)},
            "stn_binaries": n_bin, "stn_continuous": n_cont, "overall_nrmse": round(100 * nrmse_paper(pred, ds.load_cv), 1),
            "fit_seconds": round(info["seconds"], 1), "days": days}


def vpp_demo(step=2):
    """One hour of the VPP following the RegD signal, and its merit order."""
    case = load_vpp_case("day21")
    da = bid(case)
    rb = bid(case, cur_slot=0, E_cur=case.energy_init, bid_cur=(da.bid_p[0], da.bid_r[0]), rest=1.0)
    hour = simulate_hour(case, 0, da.bid_p[0], da.bid_r[0], rb.lam, case.energy_init, rb.p_none_reg)
    kinds = ["IPP", "ES", "TCL", "EV", "PV"]
    kind = np.array(case.kind)
    net = hour.p_dis - hour.p_ch
    seg = merit_order(case, rb.lam, case.energy_init, 0, rb.p_none_reg)
    keep = seg.p_hi - seg.p_lo > 1e-9
    return {
        "hours": list(range(case.T)), "bid_p": r(da.bid_p), "bid_r": r(da.bid_r),
        "price_e": r(case.price_e, 1), "price_reg": r(case.price_reg[:, 0], 1),
        "minutes": r(np.arange(0, len(hour.p_req), step) * 2 / 60, 3),
        "p_req": r(hour.p_req[::step]), "by_kind": {k: r(net[kind == k].sum(axis=0)[::step]) for k in kinds},
        "merit": {"base": round(float(seg.p_lo.sum()), 3), "kind": [case.kind[i] for i in seg.der[keep]],
                  "cost": r(seg.cost[keep], 2), "width": r((seg.p_hi - seg.p_lo)[keep])},
        "counts": {k: int((kind == k).sum()) for k in kinds}, "profit": round(da.profit, 2),
    }


def _plain(o):
    """JSON fallback for NumPy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def main(out: str):
    data = {"lstn": lstn_demo(), "crtn": crtn_demo(), "psi": psi_demo(), "d3r": d3r_demo(), "vpp": vpp_demo()}
    text = ("/* Generated by indflex tools/export_web_demos.py (https://github.com/ruikelyu/indflex). */\n"
            "window.INDFLEX_DEMOS = " + json.dumps(data, separators=(",", ":"), default=_plain) + ";\n")
    Path(out).write_text(text)
    print(f"wrote {out} ({len(text) / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
