"""Extract the PSI and D3R test fixtures from the paper repositories' saved data.

    python tools/extract_paper_fixtures.py /path/to/Production-Scheduling-Identification \
                                           /path/to/Data-Driven-Dimension-Reduction
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def psi(repo: Path):
    out = ROOT / "tests" / "fixtures" / "psi"
    out.mkdir(parents=True, exist_ok=True)
    d = sio.loadmat(repo / "data_prepare" / "dataset_cement.mat")
    np.savetxt(out / "dataset_cement_train_load.csv", d["E_primal_days_train"].T, delimiter=",", fmt="%.6f")
    np.savetxt(out / "dataset_cement_test_load.csv", d["E_primal_days_test_baseline"].T, delimiter=",", fmt="%.6f")


def d3r(repo: Path):
    out = ROOT / "tests" / "fixtures" / "d3r"
    out.mkdir(parents=True, exist_ok=True)
    for plant in ["steelpowder", "steelmaking"]:
        d = sio.loadmat(repo / "data_set" / f"dataset_{plant}.mat")
        np.savetxt(out / f"{plant}_cv_price.csv", d["Price_days_cv"].T, delimiter=",")
        for n in [1, 2]:
            r = sio.loadmat(repo / "results" / f"data_{plant}{n}ALs.mat", squeeze_me=True, struct_as_record=False)["result"]
            params = np.array([np.asarray(getattr(r, k)).reshape(-1, n)[-1] for k in ["P_min_i", "P_max_i", "E_min_i", "E_max_i"]])
            np.savetxt(out / f"{plant}_{n}loads_params.csv", params, delimiter=",",
                       header="rows: P_min, P_max, E_min, E_max; columns: loads")
            response = sio.loadmat(repo / "results" / f"data_rc_{plant}{n}ALs.mat")["E_reduced_constraints"].T
            np.savetxt(out / f"{plant}_{n}loads_cv_response.csv", response, delimiter=",")


if __name__ == "__main__":
    psi(Path(sys.argv[1]))
    d3r(Path(sys.argv[2]))
    print("fixtures written to tests/fixtures/psi and tests/fixtures/d3r")
