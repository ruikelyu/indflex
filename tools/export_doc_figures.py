"""Copy key figures from the executed notebooks into docs/figures (for the README and docs).

    python tools/export_doc_figures.py

Figures are picked by their order of appearance in each notebook; update FIGURES if the
notebooks change.
"""

import base64
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "01_stn_vs_lstn": {1: "lstn_schedule", 3: "lstn_scaling"},
    "02_rtn_vs_crtn": {2: "crtn_rounding_error", 3: "crtn_structure", 4: "crtn_gantt", 5: "crtn_binaries"},
    "03_psi": {3: "psi_machines", 4: "psi_dr_event"},
    "04_d3r": {2: "d3r_alf_band", 3: "d3r_heldout"},
    "05_vpp": {2: "vpp_bids", 3: "vpp_merit_order", 4: "vpp_signal"},
}


def main():
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, wanted in FIGURES.items():
        nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        images = [o.data["image/png"] for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        for idx, fig in wanted.items():
            (out / f"{fig}.png").write_bytes(base64.b64decode(images[idx - 1]))
            print(f"docs/figures/{fig}.png  <- {name} figure {idx}")


if __name__ == "__main__":
    main()
