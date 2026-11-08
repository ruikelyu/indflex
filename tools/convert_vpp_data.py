"""Convert the VPP case of the Applied Energy paper to plain text.

Source: https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants
(data_prepare/param_day_21.mat). Run:

    python tools/convert_vpp_data.py /path/to/Real-time-operation-strategy-of-virtual-power-plants

Writes data/vpp/vpp_day21.json (resources, prices, signal statistics) and
data/vpp/regd_day21.csv (PJM RegD signal, one value per 2 s).
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def main(repo: str):
    d = sio.loadmat(Path(repo) / "data_prepare" / "param_day_21.mat", squeeze_me=True, struct_as_record=False)
    p, s = d["param"], d["param_std"]
    ranges = np.atleast_2d(p.resource_range) - 1  # 0-based, inclusive
    kinds = ["PV", "ES", "EV", "TCL", "IPP"]  # data_prepare_std.m: 1 PV, 2 ES, EVs, TCLs (air conditioning), IPP stages
    kind = [""] * int(d["NOFDER"])
    for k, (a, b) in zip(kinds, ranges):
        for i in range(a, b + 1):
            kind[i] = k

    def sparse(A):
        return [[int(i), int(j), float(A[i, j])] for i, j in zip(*np.nonzero(A))]

    case = {
        "description": "VPP of the Applied Energy paper (Chen, Lyu et al., 2024), day 21: "
                       "1 PV, 1 storage, 120 EVs, 3 air-conditioning loads (TCL), 10 industrial stages (IPP)",
        "source": "Real-time-operation-strategy-of-virtual-power-plants/data_prepare/param_day_21.mat",
        "kind": kind,
        "delta_t": float(d["delta_t"]), "delta_t_req": float(d["delta_t_req"]), "penalty": float(d["M"]),
        "price_e": np.asarray(p.price_e, float).tolist(),
        "price_reg": np.asarray(p.price_reg, float).tolist(),  # columns: capacity, mileage
        "s_perf": float(p.s_perf),
        "hourly_mileage": np.asarray(p.hourly_Mileage, float).tolist(),
        "hourly_distribution": np.asarray(p.hourly_Distribution, float).tolist(),  # slots x scenarios
        "d_s": np.asarray(p.d_s, float).tolist(),  # regulation-signal scenarios
        "index_none_reg": (np.atleast_1d(p.index_none_reg) - 1).tolist(),  # do not provide regulation
        "energy_init": np.asarray(s.energy_init, float).tolist(),
        "energy_upper": np.asarray(s.energy_upper_limit, float).tolist(),
        "energy_lower": np.asarray(s.energy_lower_limit, float).tolist(),
        "p_dis_upper": np.asarray(s.power_dis_upper_limit, float).tolist(),
        "p_dis_lower": np.asarray(s.power_dis_lower_limit, float).tolist(),
        "p_ch_upper": np.asarray(s.power_ch_upper_limit, float).tolist(),
        "p_ch_lower": np.asarray(s.power_ch_lower_limit, float).tolist(),
        "theta": np.asarray(s.theta, float).tolist(),
        "eta_dis": sparse(np.asarray(s.eta_dis, float)),
        "eta_ch": sparse(np.asarray(s.eta_ch, float)),
        "pr_dis": np.asarray(s.pr_dis, float).tolist(),
        "pr_ch": np.asarray(s.pr_ch, float).tolist(),
        "w": np.asarray(s.wOmiga, float).tolist(),  # external energy drift (e.g. heat gains)
    }
    out = ROOT / "data" / "vpp"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vpp_day21.json").write_text(json.dumps(case))
    np.savetxt(out / "regd_day21.csv", np.asarray(d["Signal_day"], float), fmt="%.6g", header="RegD signal, 2-s steps")
    print("wrote data/vpp/vpp_day21.json and data/vpp/regd_day21.csv")


if __name__ == "__main__":
    main(sys.argv[1])
