# Reproducibility notes on the original paper code

While porting, every Python model was checked against the original MATLAB/YALMIP code (`tests/` and `tools/matlab_reference/`). These are the differences and pitfalls found along the way. Each is either handled in the toolkit or documented in the relevant test.

## Process models (STN, LSTN, RTN, cRTN)

* **Two versions of the steel powder parameters.** `STN_Implementation/load_parameters_Lu_milp.xlsx` gives the Separator 10 t/h; `LSTN_matlab/load_parameters_Lu_milp.xlsx` gives 15 t/h. The PSI and D3R datasets use the 10 t/h version (`data/plants/steel_powder_lu2021.yaml`); the LSTN paper code uses 15 t/h (`steel_powder_lu2021_lstn.yaml`).
* **Missing script.** `LSTN_matlab/data_generate_parameters.m` calls `load_primal_problem_milp`, which is not in this repository (it is in the PSI repository's `data_prepare/`).
* **RTN with one heat.** In the RTN of the cRTN paper code, `sum(S_HT(:,:)) <= 1` is meant to allow one heat melting per slot. With one heat, `S_HT` is a 1×T row and MATLAB sums it over time, so only one melting slot is allowed in total and the case is reported infeasible. The toolkit applies the limit per slot (`tests/test_batch_models.py`).
* **YALMIP + Gurobi options.** Setting any `gurobi.*` option through `sdpsettings` can pass `TuneTimeLimit = -1`, which recent Gurobi versions reject (error 10008). The paper code sets `ops.gurobi.TuneTimeLimit` explicitly; scripts that don't will fail.

## PSI

* **Time limit.** The saved paper results used `TimeLimit = 120` s per iteration for about 212 iterations. The repository's current default is 1 s, at which both the original code and the port often converge to a trivial model with a zero production target (on the cement plant, 85% test error with the original code on Gurobi 12.0.1).
* **Remedy.** Bounding the production target by its energy-derived estimate (*daily energy* / $\sum_i 1/g_i$, `target_from_energy`) avoids the trivial solution: 8–14% test error on cement at 1 s per iteration.

## D3R

* **Integer meter data.** `dataset_steelmaking.mat` stores the meter data as `uint8`. Squared errors then overflow in NumPy; the toolkit converts inputs to floats.
* **Which result files.** `test_reduced_constraints_ALF.m` writes `data_rc_<plant><I>ALs.mat`; the matching parameters are in the un-suffixed `data_<plant><I>ALs.mat`, not the `_<k>batch` files.
* **Cement responses.** The saved cement responses do not match the saved cement parameters (for two loads, the saved load exceeds the sum of the power limits), so they were produced with an earlier parameter set. Steel powder and steelmaking match exactly.

## VPP (Applied Energy 2024)

* **Efficiency matrix in the merit order.** The energy update is `E' = eta * p`, so the value of running resource `j` is `(eta' * lambda)(j)`. `fastControl_prepare.m` uses `eta(j, :) * lambda`, i.e. `eta * lambda`. They differ only for coupled resources (industrial stages). In the toolkit's tests the transposed version is never more expensive, and on average cheaper. Default: `eta_convention="transpose"`; `"paper"` reproduces the original exactly.
* **Charging-duration constraint.** In `maxProfit_1.m` and `maxProfit_t.m`, the constraint that the VPP can sustain full charging for `delta_t_req` subtracts the charging energy (`- delta_t_req * eta_ch * P_ch`), where adding it seems intended. Ported as written, so results match.
* **Current-slot revenue.** `maxProfit_t.m` subtracts the current slot's expected energy revenue from regulation. It is a constant in that problem and does not change the decisions.
