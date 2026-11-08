"""STN and LSTN models of serial continuous production lines.

Both models share one formulation. They differ only in the mode variable:

* **STN** (MILP): ``x[n, p, t]`` is binary, so machine ``n`` runs in exactly one
  operating mode ``p`` for the whole interval ``t``.
* **LSTN** (LP): ``x[n, p, t]`` is the *fraction of interval t* that machine ``n``
  spends in mode ``p``. Market intervals (e.g. one hour) are much longer than the
  time a machine needs to switch modes, so time-in-state is a good approximation,
  and the scheduling problem becomes a linear program.

Material flow (stage ``n`` produces material ``n``, stage ``n+1`` consumes it 1:1)::

    raw -> [stage 1] -> S_1 -> [stage 2] -> S_2 -> ... -> [stage N] -> S_N (final product)

References
----------
STN case data: R. Lu et al., Appl. Energy 283 (2021) 116291.
LSTN: R. Lyu et al., "LSTN: A Linear Model of Industrial Production Process for
Demand Response", IEEE ISGT Europe 2023; thesis Ch. 2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyomo.environ as pyo

from ..data import Plant


def _add_plant(blk: pyo.Block, plant: Plant, n_intervals: int, dt: float, binary: bool) -> None:
    """Add one plant's variables and constraints to a Pyomo block."""
    stages = plant.stages
    N = range(len(stages))
    T = range(n_intervals)
    NP = [(n, p) for n in N for p in range(len(stages[n].modes))]

    blk.x = pyo.Var(NP, T, domain=pyo.Binary if binary else pyo.UnitInterval)
    blk.S = pyo.Var(N, range(n_intervals + 1), domain=pyo.NonNegativeReals)

    def energy(b, n, t):  # kWh used by machine n in interval t
        return dt * sum(m.e * b.x[n, p, t] for p, m in enumerate(stages[n].modes))

    def processed(b, n, t):  # material produced by machine n in interval t
        return dt * sum(m.g * b.x[n, p, t] for p, m in enumerate(stages[n].modes))

    blk.E = pyo.Expression(N, T, rule=energy)
    blk.G = pyo.Expression(N, T, rule=processed)

    # (1) each machine is in exactly one mode (STN) / its mode times fill the interval (LSTN)
    blk.one_mode = pyo.Constraint(N, T, rule=lambda b, n, t: sum(b.x[n, p, t] for p in range(len(stages[n].modes))) == 1)

    # (2) buffer balance: + produced by stage n, - consumed by stage n+1
    def balance(b, n, t):
        consumed = b.G[n + 1, t] if n + 1 < len(stages) else 0
        return b.S[n, t + 1] == b.S[n, t] + b.G[n, t] - consumed

    blk.balance = pyo.Constraint(N, T, rule=balance)

    # (3) initial levels, capacity, and end-of-horizon targets
    blk.init = pyo.Constraint(N, rule=lambda b, n: b.S[n, 0] == stages[n].s_init)
    blk.cap = pyo.Constraint(N, range(n_intervals + 1), rule=lambda b, n, t: b.S[n, t] <= stages[n].s_max)
    blk.target = pyo.Constraint(
        N, rule=lambda b, n: b.S[n, n_intervals] >= stages[n].s_init + stages[n].s_target_change
    )

    blk.net_load = pyo.Expression(T, rule=lambda b, t: sum(b.E[n, t] for n in N))


def build_stn(plant: Plant, price: Sequence[float], dt: float = 1.0, relax: bool = False) -> pyo.ConcreteModel:
    """Cost-minimizing schedule of one plant under a price profile.

    Parameters
    ----------
    plant : Plant
    price : $/MWh for each interval (its length sets the horizon).
    dt : interval length in hours.
    relax : if True, build the LSTN (continuous time-in-state) instead of the STN.
    """
    price = np.asarray(price, dtype=float)
    m = pyo.ConcreteModel(name=("LSTN" if relax else "STN") + f" {plant.name}")
    m.T = pyo.RangeSet(0, len(price) - 1)
    m.plant = pyo.Block()
    _add_plant(m.plant, plant, len(price), dt, binary=not relax)
    m.net_load = pyo.Expression(m.T, rule=lambda m, t: m.plant.net_load[t])
    # kWh * $/MWh / 1000 = $
    m.cost = pyo.Objective(expr=sum(price[t] / 1000 * m.net_load[t] for t in m.T))
    m._meta = {"kind": "lstn" if relax else "stn", "plants": [plant], "price": price, "dt": dt,
               "detail": _single_plant_detail}
    return m


def _single_plant_detail(m):
    """Energy and buffer level per stage (rows), for single-plant models."""
    import pandas as pd

    plant = m._meta["plants"][0]
    names = [f"{k + 1}. {s.name}" for k, s in enumerate(plant.stages)]  # unique even if names repeat
    T, N = list(m.T), range(plant.n_stages)
    energy = pd.DataFrame([[pyo.value(m.plant.E[n, t]) for t in T] for n in N], index=names, columns=T)
    storage = pd.DataFrame([[pyo.value(m.plant.S[n, t]) for t in range(len(T) + 1)] for n in N], index=names)
    return energy, storage


def build_lstn(plant: Plant, price: Sequence[float], dt: float = 1.0) -> pyo.ConcreteModel:
    """LSTN model: the STN with on/off binaries replaced by time-in-state fractions (an LP)."""
    return build_stn(plant, price, dt=dt, relax=True)


def build_fleet(plants: Sequence[Plant], price: Sequence[float], dt: float = 1.0, relax: bool = True) -> pyo.ConcreteModel:
    """Many independent plants scheduled together against one price (LSTN by default).

    The fleet objective decomposes by plant, so the aggregate is what an aggregator
    or a grid model would see: the sum of the plants' optimal responses.
    """
    price = np.asarray(price, dtype=float)
    m = pyo.ConcreteModel(name=("LSTN" if relax else "STN") + f" fleet of {len(plants)}")
    m.T = pyo.RangeSet(0, len(price) - 1)
    m.F = pyo.RangeSet(0, len(plants) - 1)
    m.plant = pyo.Block(m.F)
    for f, plant in enumerate(plants):
        _add_plant(m.plant[f], plant, len(price), dt, binary=not relax)
    m.net_load = pyo.Expression(m.T, rule=lambda m, t: sum(m.plant[f].net_load[t] for f in m.F))
    m.cost = pyo.Objective(expr=sum(price[t] / 1000 * m.net_load[t] for t in m.T))
    m._meta = {"kind": "lstn" if relax else "stn", "plants": list(plants), "price": price, "dt": dt}
    return m
