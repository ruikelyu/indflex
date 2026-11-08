# Documentation

One page per step of the chain. Each page explains the idea, states what the model can and cannot claim, shows the main figures, lists the results checked against the original paper code, and links the paper, code and video.

1. [Continuous processes: STN and LSTN](01_stn_lstn.md)
2. [Batch processes: RTN and cRTN](02_rtn_crtn.md)
3. [Parameter identification: PSI](03_psi.md)
4. [Dimension reduction: D3R](04_d3r.md)
5. [Virtual power plant: bidding and fast disaggregation](05_vpp.md)

[Reproducibility notes on the original paper code](notes_on_paper_code.md) lists the differences and pitfalls found while porting.

## Regenerating notebooks and figures

```bash
python tools/build_notebooks.py            # all notebooks (or e.g. "03" for one)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
python tools/export_doc_figures.py         # copy figures into docs/figures
```

Notebook sources live in `tools/build_notebooks.py`, so changes are reviewable as plain Python.

## Regenerating the MATLAB references

The tests compare the Python models with outputs of the original MATLAB code (YALMIP + Gurobi), stored in `tests/fixtures/`. The scripts that produced them are in `tools/matlab_reference/`.
