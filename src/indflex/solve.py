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
    load: np.ndarray  # total energy per interval (kWh for STN/LSTN plants, MWh for batch plants)
    energy: pd.DataFrame | None  # energy per stage (rows) and interval (columns), single-plant models
    storage: pd.DataFrame | None  # buffer levels per stage, intervals 0..T
    solve_seconds: float
    n_binary: int
    n_continuous: int
    energy_unit: str = "kWh"

    def summary(self) -> str:
        return (
            f"{self.status}: cost ${self.cost:,.2f}, total {self.load.sum():,.1f} {self.energy_unit}, "
            f"{self.n_binary} binary / {self.n_continuous} continuous vars, {self.solve_seconds:.2f} s"
        )


def count_vars(m: pyo.ConcreteModel) -> tuple[int, int]:
    """(binary/integer, continuous) decision variables, ignoring fixed ones."""
    nb = nc = 0
    for v in m.component_data_objects(pyo.Var, active=True):
        if v.fixed:
            continue
        if v.is_binary() or v.is_integer():
            nb += 1
        else:
            nc += 1
    return nb, nc


def solve(m: pyo.ConcreteModel, solver: str = "highs", time_limit: float | None = None,
          mip_gap: float | None = None, tee: bool = False) -> Result:
    """Solve with HiGHS (default, free) or Gurobi (``solver="gurobi"``: ``pip install gurobipy`` plus a license).

    If a time limit stops a MIP early, the best solution found so far is returned
    (status ``"time limit"``); if none was found, ``cost`` is NaN.
    """
    from pyomo.contrib.appsi.solvers import Gurobi, Highs

    solvers = {"highs": Highs, "gurobi": Gurobi}
    if solver not in solvers:
        raise ValueError(f"solver must be one of {list(solvers)}")
    opt = solvers[solver]()
    opt.config.load_solution = False
    opt.config.stream_solver = tee
    if time_limit is not None:
        opt.config.time_limit = time_limit
    if mip_gap is not None:
        opt.config.mip_gap = mip_gap
    t0 = time.perf_counter()
    res = opt.solve(m)
    tc = res.termination_condition.name
    status = {"optimal": "optimal", "maxTimeLimit": "time limit"}.get(tc, tc)
    has_solution = res.best_feasible_objective is not None
    if has_solution:
        res.solution_loader.load_vars()
    seconds = time.perf_counter() - t0
    nb, nc = count_vars(m)
    unit = m._meta.get("energy_unit", "kWh")
    T = list(m.T)
    if not has_solution:
        return Result(status, float("nan"), np.full(len(T), np.nan), None, None, seconds, nb, nc, unit)

    load = np.array([pyo.value(m.net_load[t]) for t in T])
    detail = m._meta.get("detail")
    energy, storage = detail(m) if detail else (None, None)
    cost = m.energy_cost if hasattr(m, "energy_cost") else m.cost  # excludes tie-breaking terms
    return Result(status, float(pyo.value(cost)), load, energy, storage, seconds, nb, nc, unit)
