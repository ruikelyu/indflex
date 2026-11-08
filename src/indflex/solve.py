"""Solve a model and collect results in plain pandas/numpy objects."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo


@dataclass
class Result:
    status: str
    cost: float  # $
    load: np.ndarray  # total energy per interval, kWh
    energy: pd.DataFrame | None  # energy per stage (rows) and interval (columns), single-plant models
    storage: pd.DataFrame | None  # buffer levels per stage, intervals 0..T
    solve_seconds: float
    n_binary: int
    n_continuous: int

    def summary(self) -> str:
        return (
            f"{self.status}: cost ${self.cost:,.2f}, total {self.load.sum():,.1f} kWh, "
            f"{self.n_binary} binary / {self.n_continuous} continuous vars, {self.solve_seconds:.2f} s"
        )


def _count_vars(m: pyo.ConcreteModel) -> tuple[int, int]:
    nb = nc = 0
    for v in m.component_data_objects(pyo.Var, active=True):
        if v.is_binary() or v.is_integer():
            nb += 1
        else:
            nc += 1
    return nb, nc


def solve(m: pyo.ConcreteModel, solver: str = "highs", time_limit: float | None = None,
          mip_gap: float | None = None, tee: bool = False) -> Result:
    """Solve with HiGHS (default, free) or Gurobi (``solver="gurobi"``, needs a license).

    If a time limit stops a MIP early, the best solution found so far is returned
    (status ``"time limit"``); if none was found, ``cost`` is NaN.
    """
    t0 = time.perf_counter()
    has_solution = True
    if solver == "highs":
        from pyomo.contrib.appsi.solvers import Highs

        opt = Highs()
        opt.config.load_solution = False
        opt.config.stream_solver = tee
        if time_limit is not None:
            opt.config.time_limit = time_limit
        if mip_gap is not None:
            opt.config.mip_gap = mip_gap
        res = opt.solve(m)
        tc = res.termination_condition.name
        status = {"optimal": "optimal", "maxTimeLimit": "time limit"}.get(tc, tc)
        if res.best_feasible_objective is not None:
            res.solution_loader.load_vars()
        else:
            has_solution = False
    else:
        opt = pyo.SolverFactory(solver)
        if time_limit is not None:
            opt.options["TimeLimit"] = time_limit
        if mip_gap is not None:
            opt.options["MIPGap"] = mip_gap
        res = opt.solve(m, tee=tee, load_solutions=False)
        status = str(res.solver.termination_condition)
        status = {"maxTimeLimit": "time limit"}.get(status, status)
        if len(res.solution) > 0:
            m.solutions.load_from(res)
        else:
            has_solution = False
    seconds = time.perf_counter() - t0
    nb, nc = _count_vars(m)
    T = list(m.T)
    if not has_solution:
        return Result(status, float("nan"), np.full(len(T), np.nan), None, None, seconds, nb, nc)

    load = np.array([pyo.value(m.net_load[t]) for t in T])
    energy = storage = None
    if not m.plant.is_indexed():  # single plant: expose per-stage detail
        plant = m._meta["plants"][0]
        names = [f"{k + 1}. {s.name}" for k, s in enumerate(plant.stages)]  # unique even if names repeat
        N = range(plant.n_stages)
        energy = pd.DataFrame([[pyo.value(m.plant.E[n, t]) for t in T] for n in N], index=names, columns=T)
        storage = pd.DataFrame(
            [[pyo.value(m.plant.S[n, t]) for t in range(len(T) + 1)] for n in N], index=names
        )
    return Result(status, float(pyo.value(m.cost)), load, energy, storage, seconds, nb, nc)
