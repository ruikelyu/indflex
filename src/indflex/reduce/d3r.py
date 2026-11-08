"""Data-driven dimension reduction (D3R): learn a small linear model of a plant.

Setting
-------
A grid or planning model cannot carry the full process model (STN/RTN, with
thousands of binaries) of every factory. D3R replaces it with **reduced constraints**
whose optimal response to prices mimics the full model's.

Reduced form: an adjustable load fleet (ALF)
--------------------------------------------
``I`` parallel adjustable loads; load ``i`` draws ``p[t, i]`` with

    P_min[i] <= p[t, i] <= P_max[i]            (power limits, every hour)
    E_min[i] <= sum_t p[t, i] * dt <= E_max[i]  (daily energy limits)

and the plant's load is ``sum_i p[t, i]``. That is ``4 * I`` continuous parameters
instead of a mixed-integer process model.

Training (thesis Ch. 4; Lyu et al., IEEE TSG 16(3), 2025)
--------------------------------------------------------
Solve the full model under many prices (the data), then find ALF parameters whose
*optimal* response reproduces it: inverse optimization with the ALF's KKT
conditions, complementary slackness linearized with big-M binaries
(Fortuny-Amat). There are no bilinear terms, so:

* squared-error loss -> mixed-integer QP (Gurobi), as in the paper;
* absolute-error loss -> MILP, solvable with the free HiGHS solver.

The paper code (https://github.com/ruikelyu/Data-Driven-Dimension-Reduction)
solves small random batches of days and averages the parameters (a zeroth-order
stochastic method); :func:`train_d3r` transcribes it. Small cases can be fitted
in a single solve with :func:`fit_alf`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo

from ..solve import solve


@dataclass
class ALFParams:
    """Adjustable load fleet: one entry per adjustable load."""

    p_min: np.ndarray
    p_max: np.ndarray
    e_min: np.ndarray
    e_max: np.ndarray

    @property
    def n_loads(self) -> int:
        return len(self.p_max)

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame({"P_min": self.p_min, "P_max": self.p_max, "E_min": self.e_min, "E_max": self.e_max},
                            index=[f"load {i + 1}" for i in range(self.n_loads)]).round(3)


# --------------------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------------------
def build_alf(params: ALFParams, price: Sequence[float], dt: float = 1.0) -> pyo.ConcreteModel:
    """The ALF's cost-minimizing response to a price profile (an LP)."""
    price = np.asarray(price, dtype=float)
    T, I = len(price), params.n_loads
    m = pyo.ConcreteModel(name="ALF")
    m.T = pyo.RangeSet(0, T - 1)
    m.p = pyo.Var(range(T), range(I), bounds=lambda m, t, i: (params.p_min[i], params.p_max[i]))
    m.energy = pyo.ConstraintList()
    for i in range(I):
        daily = sum(m.p[t, i] * dt for t in range(T))
        m.energy.add(daily <= params.e_max[i])
        m.energy.add(daily >= params.e_min[i])
    m.net_load = pyo.Expression(m.T, rule=lambda m, t: sum(m.p[t, i] for i in range(I)) * dt)
    m.cost = pyo.Objective(expr=sum(price[t] * m.net_load[t] for t in m.T))
    m._meta = {"kind": "alf", "price": price, "dt": dt}
    return m


def simulate_alf(params: ALFParams, prices: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Optimal ALF load (days x T) under each day's price."""
    return np.array([solve(build_alf(params, p, dt)).load for p in np.atleast_2d(prices)])


# --------------------------------------------------------------------------------------
# Inverse problem
# --------------------------------------------------------------------------------------
def build_inverse_alf(prices: np.ndarray, loads: np.ndarray, n_loads: int, M: float = 1e3,
                      loss: str = "l2") -> pyo.ConcreteModel:
    """Inverse optimization: ALF parameters whose optimal response fits ``loads``.

    ``prices`` and ``loads`` are (days x T). One copy of the ALF's primal variables,
    dual variables and complementarity binaries is created per day.
    """
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    loads = np.atleast_2d(np.asarray(loads, dtype=float))  # (meter data may come as integers)
    B, T = loads.shape
    I = n_loads
    Bs, Ts, Is = range(B), range(T), range(I)
    m = pyo.ConcreteModel(name="D3R inverse problem")

    # ALF parameters θ, with the paper's priors
    m.P_max = pyo.Var(Is, bounds=(None, M))
    m.P_min = pyo.Var(Is, bounds=(0, None))
    m.E_max = pyo.Var(Is, bounds=(None, M * T))
    m.E_min = pyo.Var(Is, bounds=(0, None))
    m.order = pyo.ConstraintList()  # sort loads by P_max to remove symmetric solutions
    for i in range(I - 1):
        m.order.add(m.P_max[i] <= m.P_max[i + 1])

    # primal and dual variables, one set per day
    m.p = pyo.Var(Bs, Ts, Is)
    m.mu_pmin = pyo.Var(Bs, Ts, Is, bounds=(0, M / 2))
    m.mu_pmax = pyo.Var(Bs, Ts, Is, bounds=(0, M / 2))
    m.mu_emin = pyo.Var(Bs, Is, bounds=(0, M / 2))
    m.mu_emax = pyo.Var(Bs, Is, bounds=(0, M / 2))
    m.z_pmin = pyo.Var(Bs, Ts, Is, domain=pyo.Binary)
    m.z_pmax = pyo.Var(Bs, Ts, Is, domain=pyo.Binary)
    m.z_emin = pyo.Var(Bs, Is, domain=pyo.Binary)
    m.z_emax = pyo.Var(Bs, Is, domain=pyo.Binary)

    c = m.kkt = pyo.ConstraintList()
    for b in Bs:
        for i in Is:
            daily = sum(m.p[b, t, i] for t in Ts)
            for t in Ts:
                # stationarity
                c.add(prices[b, t] - m.mu_pmin[b, t, i] + m.mu_pmax[b, t, i]
                      - m.mu_emin[b, i] + m.mu_emax[b, i] == 0)
                # primal feasibility
                c.add(m.p[b, t, i] <= m.P_max[i])
                c.add(m.p[b, t, i] >= m.P_min[i])
                # complementary slackness (big-M): a multiplier is zero unless its limit binds
                c.add(m.P_max[i] - m.p[b, t, i] <= M * m.z_pmax[b, t, i])
                c.add(m.mu_pmax[b, t, i] <= M * (1 - m.z_pmax[b, t, i]))
                c.add(m.p[b, t, i] - m.P_min[i] <= M * m.z_pmin[b, t, i])
                c.add(m.mu_pmin[b, t, i] <= M * (1 - m.z_pmin[b, t, i]))
            c.add(daily <= m.E_max[i])
            c.add(daily >= m.E_min[i])
            c.add(m.E_max[i] - daily <= M * m.z_emax[b, i])
            c.add(m.mu_emax[b, i] <= M * (1 - m.z_emax[b, i]))
            c.add(daily - m.E_min[i] <= M * m.z_emin[b, i])
            c.add(m.mu_emin[b, i] <= M * (1 - m.z_emin[b, i]))

    m.fitted = pyo.Expression(Bs, Ts, rule=lambda m, b, t: sum(m.p[b, t, i] for i in Is))
    if loss == "l2":
        m.obj = pyo.Objective(expr=sum((m.fitted[b, t] - loads[b, t]) ** 2 for b in Bs for t in Ts))
    elif loss == "l1":
        m.err = pyo.Var(Bs, Ts, domain=pyo.NonNegativeReals)
        m.abs = pyo.ConstraintList()
        for b in Bs:
            for t in Ts:
                m.abs.add(m.err[b, t] >= m.fitted[b, t] - loads[b, t])
                m.abs.add(m.err[b, t] >= loads[b, t] - m.fitted[b, t])
        m.obj = pyo.Objective(expr=sum(m.err[b, t] for b in Bs for t in Ts))
    else:
        raise ValueError("loss must be 'l2' or 'l1'")
    m._n_loads = I
    return m


def alf_of(m: pyo.ConcreteModel) -> ALFParams:
    I = m._n_loads
    get = lambda v: np.array([pyo.value(v[i]) for i in range(I)])  # noqa: E731
    return ALFParams(p_min=get(m.P_min), p_max=get(m.P_max), e_min=get(m.E_min), e_max=get(m.E_max))


def _solver(name: str, time_limit: float | None):
    from pyomo.contrib.appsi.solvers import Gurobi, Highs

    opt = {"highs": Highs, "gurobi": Gurobi}[name]()
    opt.config.load_solution = False
    if time_limit is not None:
        opt.config.time_limit = time_limit
    return opt


def fit_alf(prices: np.ndarray, loads: np.ndarray, n_loads: int = 1, solver: str = "highs",
            loss: str | None = None, time_limit: float | None = 60) -> tuple[ALFParams, dict]:
    """Fit an ALF to all given days in a single inverse-optimization solve.

    Default loss: ``"l1"`` with HiGHS (a MILP), ``"l2"`` with Gurobi (a MIQP, as in the paper).
    """
    loss = loss or ("l1" if solver == "highs" else "l2")
    m = build_inverse_alf(prices, loads, n_loads, loss=loss)
    opt = _solver(solver, time_limit)
    t0 = time.perf_counter()
    res = opt.solve(m)
    if res.best_feasible_objective is None:
        raise RuntimeError(f"no feasible ALF found ({res.termination_condition.name})")
    res.solution_loader.load_vars()
    info = {"status": res.termination_condition.name, "seconds": time.perf_counter() - t0,
            "objective": res.best_feasible_objective, "binaries": sum(1 for v in m.component_data_objects(pyo.Var) if v.is_binary())}
    return alf_of(m), info


@dataclass
class D3RHistory:
    params: list[ALFParams] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_d3r(prices: np.ndarray, loads: np.ndarray, n_loads: int = 1, batch_size: int = 1,
              max_iter: int = 40, solver: str = "gurobi", loss: str | None = None,
              time_limit: float = 120, seed: int = 0) -> tuple[ALFParams, D3RHistory]:
    """Iterative training as in the paper code (``model_training.m``).

    Each iteration solves the inverse problem on ``batch_size`` random training days
    and moves θ toward the solution: ``θ <- (1 - a) θ + a θ*`` with ``a = 0.05``,
    decaying as ``0.05 / sqrt(k - 20)`` after 20 iterations. θ starts at zero.
    """
    rng = np.random.default_rng(seed)
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    loads = np.atleast_2d(np.asarray(loads, dtype=float))
    D = len(loads)
    loss = loss or ("l1" if solver == "highs" else "l2")
    theta = ALFParams(*(np.zeros(n_loads) for _ in range(4)))
    hist = D3RHistory()
    t0 = time.perf_counter()
    for k in range(max_iter + 1):
        days = rng.integers(0, D, batch_size)
        m = build_inverse_alf(prices[days], loads[days], n_loads, loss=loss)
        opt = _solver(solver, time_limit)
        res = opt.solve(m)
        if res.best_feasible_objective is None:
            continue
        res.solution_loader.load_vars()
        star = alf_of(m)
        a = 0.05 if k <= 20 else 0.05 / (k - 20) ** 0.5
        theta = ALFParams(*((1 - a) * getattr(theta, n) + a * getattr(star, n)
                            for n in ("p_min", "p_max", "e_min", "e_max")))
        hist.params.append(theta)
        hist.loss.append(float(res.best_feasible_objective))
        if k > D and np.mean(np.abs(hist.loss[-D:])) < 1e-3:
            break
    hist.seconds = time.perf_counter() - t0
    return theta, hist


def nrmse_paper(pred: np.ndarray, actual: np.ndarray) -> float:
    """Error metric of the D3R paper: RMSE over all hours and days / max load of the first day."""
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    actual = np.atleast_2d(np.asarray(actual, dtype=float))
    return float(np.sqrt(np.mean((pred - actual) ** 2)) / np.max(actual[0]))
