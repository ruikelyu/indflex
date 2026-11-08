"""RTN and cRTN models of batch (discrete) production lines, e.g. steelmaking.

A batch line processes discrete *heats* through serial stages (EAF → AOD → LF → CC).
Two features make it hard for STN/LSTN:

* **Uninterruptible tasks:** once a heat starts melting, it must finish.
* **Waiting-time limits:** a hot heat may only wait so long between stages.

Two formulations are provided:

* :func:`build_rtn`: the conventional Resource-Task Network (Zhang et al. 2017),
  with a flexible EAF. Binary *start* variables per **task and heat**, so the model
  grows with the production target (number of heats).
* :func:`build_crtn`: the continuous RTN (cRTN). Tasks and resources are decoupled
  from heats; the decision variable is *how long* each task runs in each operating
  point within a time slot (as in the LSTN). Only status binaries remain, and their
  number does **not** depend on the number of heats.

Both are transcriptions of the MATLAB code in
https://github.com/ruikelyu/Efficient-Scheduling-of-Discrete-Industrial-Processes
(tests compare them with that code).

References
----------
R. Lyu, X. Su, E. Du, H. Guo, Q. Chen, C. Kang, "Efficient Scheduling of Discrete
Industrial Processes through Continuous Modeling", IEEE Trans. Smart Grid, in press.
X. Zhang, G. Hug, I. Harjunkoski, "Cost-Effective Scheduling of Steel Plants With
Flexible EAFs", IEEE Trans. Smart Grid 8(1), 2017.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ..data import BatchPlant


# --------------------------------------------------------------------------------------
# cRTN
# --------------------------------------------------------------------------------------
def crtn_parameters(plant: BatchPlant, dt: float):
    """Rates G[i, k] and powers P[i, k] of the cRTN tasks (0-based; k = 0 idle, 1, 2).

    Each stage j has four tasks (and four resources, indexed like the task that
    produces them): processing ``4j``, output ``4j+1``, waiting ``4j+2``, input ``4j+3``.
    """
    n = plant.n_stages
    n_task = 4 * n
    proc = np.arange(0, n_task, 4)
    out, wait, inp = proc + 1, proc + 2, proc + 3
    lo, hi = plant.power_ratio

    # minutes: [processing, transfer, max waiting, 0] per stage, at least one slot
    minutes = np.zeros(n_task)
    minutes[proc] = [s.processing_min for s in plant.stages]
    minutes[out] = [s.transfer_min for s in plant.stages]
    minutes[wait] = [s.max_wait_min for s in plant.stages]
    minutes = np.maximum(minutes, dt * 60)

    G = np.zeros((n_task, 3))
    G[proc, 1] = dt / (minutes[proc] / 60)  # fraction of the task done per slot
    G[0, 2] = G[0, 1]
    G[out, 1] = 1
    G[wait, 2] = dt / (minutes[wait - 1] / 60 + dt)  # fast: transfer time
    G[wait, 1] = dt / (minutes[wait] / 60 + dt)  # slow: maximum waiting time
    if n > 1:
        G[wait[-1], 1:3] = 1  # final product does not wait
    G[inp, 1] = 1
    G = np.minimum(G, 1)
    G[0, 1:3] = G[0, 1] * np.array([lo, hi])  # flexible EAF: low / high power point

    P = np.zeros_like(G)
    P[proc, 1] = [s.power_mw for s in plant.stages]
    P[0, 1:3] = P[0, 1] * np.array([lo, hi])
    return G, P, dict(proc=proc, out=out, wait=wait, inp=inp)


def build_crtn(plant: BatchPlant, price: Sequence[float], n_heats: int, dt: float = 1.0) -> pyo.ConcreteModel:
    """Continuous RTN: cost-minimizing schedule for ``n_heats`` heats.

    Parameters
    ----------
    price : $/MWh per slot (its length sets the horizon).
    dt : slot length in hours.
    """
    price = np.asarray(price, dtype=float)
    T = len(price)
    n = plant.n_stages
    G, Pw, idx = crtn_parameters(plant, dt)
    proc, out, wait, inp = idx["proc"], idx["out"], idx["wait"], idx["inp"]
    n_task = 4 * n
    I, K, Ts = range(n_task), range(3), range(T)

    m = pyo.ConcreteModel(name=f"cRTN {plant.name}")
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.Var(I, range(T + 1), domain=pyo.Reals)  # resource level / task progress
    m.D = pyo.Var(I, K, Ts, bounds=(0, 1))  # fraction of slot t task i spends at point k
    m.u = pyo.Var(range(2 * n), Ts, domain=pyo.Binary)  # task in progress (no idling)
    m.on = pyo.Var(range(2 * n), Ts, domain=pyo.Binary)  # output / input event

    def produced(i, t):
        return sum(G[i, k] * m.D[i, k, t] for k in K if G[i, k] != 0)

    c = m.cons = pyo.ConstraintList()
    for t in Ts:
        for i in I:
            # (a) resource balance: produced by task i, consumed by task i + 1
            consumed = produced(i + 1, t) if i + 1 < n_task else 0
            c.add(m.R[i, t + 1] <= m.R[i, t] + produced(i, t) - consumed)
            c.add(m.R[i, t + 1] >= 0)
            if i <= proc[-1]:
                c.add(m.R[i, t + 1] <= 1)
            # (b) slot is fully allocated across operating points
            c.add(sum(m.D[i, k, t] for k in K) == 1)
        # uninterruptible processing: no idling (k = 0) while a heat is in progress
        c.add(m.D[proc[0], 0, t] <= 1 - m.u[0, t])
        c.add(m.R[proc[0], t] <= m.u[0, t])
        for j in range(1, n):
            c.add(m.D[proc[j], 0, t] <= 1 - m.u[j, t])
            c.add(m.R[proc[j] - 1, t] <= m.u[j, t])
            c.add(m.R[proc[j], t] <= m.u[j, t])
        # waiting tasks also run without pause once started
        for j in range(n):
            c.add(m.D[wait[j], 0, t] <= 1 - m.u[n + j, t])
            c.add(m.R[wait[j] - 1, t] <= m.u[n + j, t])
            c.add(m.R[wait[j], t] <= m.u[n + j, t])
        # only the EAF has a second (high-power) operating point
        for i in list(proc[1:]) + list(out) + list(inp):
            m.D[i, 2, t].fix(0)
        # output / input tasks happen as whole-slot events
        for j in range(n):
            c.add(m.D[out[j], 1, t] == m.on[j, t])
            c.add(m.D[inp[j], 1, t] == m.on[n + j, t])
    for i in I:
        m.R[i, 0].fix(0)
    c.add(m.R[n_task - 1, T] == n_heats)  # deliver all heats

    m.stage_energy = pyo.Expression(
        range(n), Ts, rule=lambda m, j, t: dt * sum(Pw[proc[j], k] * m.D[proc[j], k, t] for k in K if Pw[proc[j], k])
    )
    m.net_load = pyo.Expression(m.T, rule=lambda m, t: sum(m.stage_energy[j, t] for j in range(n)))
    m.cost = pyo.Objective(expr=sum(price[t] * m.net_load[t] for t in Ts))  # MWh * $/MWh
    m._crtn = {"G": G, "proc": proc}
    m._meta = {"kind": "crtn", "plants": [plant], "price": price, "dt": dt, "n_heats": n_heats,
               "detail": _stage_energy_detail, "energy_unit": "MWh"}
    return m


# --------------------------------------------------------------------------------------
# Conventional RTN with a flexible EAF
# --------------------------------------------------------------------------------------
def build_rtn(plant: BatchPlant, price: Sequence[float], n_heats: int, dt: float = 1.0) -> pyo.ConcreteModel:
    """Conventional RTN (binary task starts per heat), with a power-flexible first stage."""
    price = np.asarray(price, dtype=float)
    T = len(price)
    n, H = plant.n_stages, n_heats
    Ts = range(T)

    # slots: rows = processing, transfer, max waiting; the final product may wait all day
    ps = np.array([[ceil(round(v / dt / 60, 9)) for v in (s.processing_min, s.transfer_min, s.max_wait_min)]
                   for s in plant.stages]).T
    ps[2, n - 1] = T

    def task_proc(j, h):
        return j * 2 * H + 2 * h

    def res_dev(j):
        return j * (1 + 2 * H)

    def res_s(j, h):  # material just finished at stage j (must be transferred at once)
        return res_dev(j) + 1 + 2 * h

    def res_d(j, h):  # material delivered to the next stage (may wait)
        return res_dev(j) + 2 + 2 * h

    n_task, n_res = n * 2 * H, n * (1 + 2 * H)
    max_theta = int(max(ps[0].max(), ps[1].max()))
    MU = np.zeros((n_res, n_task, max_theta + 1))
    for j in range(n):
        for h in range(H):
            pt, tt = task_proc(j, h), task_proc(j, h) + 1
            MU[res_dev(j), pt, 0] = -1  # occupy the device...
            MU[res_dev(j), pt, ps[0, j]] = 1  # ...and release it after processing
            if j > 0:
                MU[res_d(j - 1, h), pt, 0] = -1  # consume the heat from stage j-1
            MU[res_s(j, h), pt, ps[0, j]] = 1
            MU[res_s(j, h), tt, 0] = -1  # transfer it away...
            MU[res_d(j, h), tt, ps[1, j]] = 1  # ...arriving after the transfer time
    # the flexible EAF (stage 0) is modelled by the melting constraints below instead
    for h in range(H):
        pt = task_proc(0, h)
        MU[res_dev(0), pt, :] = 0
        MU[res_s(0, h), pt, :] = 0
        MU[res_s(0, h), pt + 1, :] = 0

    lo, hi = plant.power_ratio
    p0, t0 = plant.stages[0].power_mw, plant.stages[0].processing_min
    tau_L = ceil(round(t0 / dt / 60 / hi, 9))
    tau_U = ceil(round(t0 / dt / 60 / lo, 9))

    m = pyo.ConcreteModel(name=f"RTN {plant.name}")
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.Var(range(n_res), range(T + 1), domain=pyo.Binary)
    m.N = pyo.Var(range(n_task), Ts, domain=pyo.Binary)  # task i starts in slot t
    m.S = pyo.Var(range(H), Ts, domain=pyo.Binary)  # heat h is melting in slot t
    m.Pm = pyo.Var(range(H), Ts, domain=pyo.NonNegativeReals)  # EAF power for heat h, MW

    nz = [(r, i, th) for r, i, th in zip(*np.nonzero(MU))]
    c = m.cons = pyo.ConstraintList()
    for j in range(n):
        m.R[res_dev(j), 0].fix(1)
        for h in range(H):
            m.R[res_s(j, h), 0].fix(0)
            m.R[res_d(j, h), 0].fix(0)
    for t in Ts:
        change = {r: 0 for r in range(n_res)}
        for r, i, th in nz:
            if th <= t:
                change[r] = change[r] + MU[r, i, th] * m.N[i, t - th]
        for r in range(n_res):
            c.add(m.R[r, t + 1] == m.R[r, t] + change[r])
    for i in range(n_task):
        c.add(sum(m.N[i, t] for t in Ts) == 1)
    for j in range(n):
        for h in range(H):
            for t in range(T + 1):
                m.R[res_s(j, h), t].fix(0)  # immediate transfer
            c.add(sum(m.R[res_d(j, h), t] for t in range(1, T + 1)) + ps[1, j] <= ps[2, j])  # max waiting
    for h in range(H):
        c.add(m.R[res_d(n - 1, h), T] == 1)  # deliver all heats

    # flexible melting (Zhang et al. 2017, eqs. 12-15), per heat
    for h in range(H):
        mel, tr = task_proc(0, h), task_proc(0, h) + 1
        for t in range(T):
            if t + tau_L >= T:
                m.N[mel, t].fix(0)  # cannot finish melting within the horizon
            else:
                end = min(t + tau_U, T - 1)
                c.add(sum(m.N[tr, s] for s in range(t + tau_L, end + 1)) >= m.N[mel, t])
            c.add(lo * p0 * m.S[h, t] <= m.Pm[h, t])
            c.add(m.Pm[h, t] <= hi * p0 * m.S[h, t])
            if t == 0:
                c.add(m.S[h, 0] == m.N[mel, 0])
            else:
                c.add(m.S[h, t] - m.S[h, t - 1] == m.N[mel, t] - m.N[tr, t])
        c.add(sum(m.Pm[h, t] * dt for t in Ts) >= p0 * t0 / 60)
    for t in Ts:
        c.add(sum(m.S[h, t] for h in range(H)) <= 1)

    def stage_energy(m, j, t):
        if j == 0:
            return dt * sum(m.Pm[h, t] for h in range(H))
        return dt * plant.stages[j].power_mw * (1 - m.R[res_dev(j), t + 1])

    m.stage_energy = pyo.Expression(range(n), Ts, rule=stage_energy)
    m.net_load = pyo.Expression(m.T, rule=lambda m, t: sum(m.stage_energy[j, t] for j in range(n)))
    m.cost = pyo.Objective(expr=sum(price[t] * m.net_load[t] for t in Ts))
    m._rtn = {"task_proc": task_proc, "slots": ps, "n_heats": H}
    m._meta = {"kind": "rtn", "plants": [plant], "price": price, "dt": dt, "n_heats": n_heats,
               "detail": _stage_energy_detail, "energy_unit": "MWh"}
    return m


def _stage_energy_detail(m):
    plant = m._meta["plants"][0]
    names = [f"{k + 1}. {s.name}" for k, s in enumerate(plant.stages)]
    T = list(m.T)
    energy = pd.DataFrame(
        [[pyo.value(m.stage_energy[j, t]) for t in T] for j in range(plant.n_stages)], index=names, columns=T
    )
    return energy, None


# --------------------------------------------------------------------------------------
# Reading schedules back out of solved models (for plots)
# --------------------------------------------------------------------------------------
def _crtn_segments(m, j: int):
    """Piecewise-linear cumulative progress of stage j's processing task, in hours.

    The cRTN only says *how long* a task runs within each slot, not *when*. For
    drawing, a task that continues from the previous slot runs at the start of the
    slot; one that continues into the next slot runs at its end; otherwise at the start.
    """
    dt, T = m._meta["dt"], len(m.T)
    G, i = m._crtn["G"], m._crtn["proc"][j]
    active = [sum(pyo.value(m.D[i, k, t]) for k in (1, 2)) for t in range(T)]
    gain = [sum(G[i, k] * pyo.value(m.D[i, k, t]) for k in (1, 2)) for t in range(T)]
    segs, c = [], 0.0
    for t in range(T):
        a = active[t]
        if a < 1e-9 or gain[t] < 1e-12:
            continue
        if t > 0 and active[t - 1] > 1e-9:
            start = t
        elif t + 1 < T and active[t + 1] > 1e-9:
            start = t + 1 - a
        else:
            start = t
        segs.append((start * dt, (start + a) * dt, c, c + gain[t]))
        c += gain[t]
    return segs


def crtn_progress(m, stage: int = 0):
    """(times, progress) of one stage's processing task, resetting to 0 when a heat completes."""
    times, prog = [0.0], [0.0]
    done = 0
    for t0, t1, c0, c1 in _crtn_segments(m, stage):
        times.append(t0); prog.append(c0 - done)
        while c1 >= done + 1 - 1e-6:  # a heat completes inside this segment
            tf = t0 + (t1 - t0) * (done + 1 - c0) / (c1 - c0)
            times += [tf, tf]; prog += [1.0, 0.0]
            done += 1
        times.append(t1); prog.append(c1 - done)
    times.append(len(m.T) * m._meta["dt"]); prog.append(prog[-1])
    return np.array(times), np.clip(np.array(prog), 0, 1)


def heat_schedule(m) -> pd.DataFrame:
    """When each heat is processed at each stage (hours), from a solved RTN or cRTN model."""
    plant, dt, T = m._meta["plants"][0], m._meta["dt"], len(m.T)
    rows = []
    if m._meta["kind"] == "crtn":
        for j, st in enumerate(plant.stages):
            segs = _crtn_segments(m, j)
            total = segs[-1][3] if segs else 0
            for h in range(int(round(total))):
                def when(level, first):
                    for t0, t1, c0, c1 in segs:
                        if c0 - 1e-6 <= level <= c1 + 1e-6 and (c1 - c0) > 1e-9 and (c1 > level + 1e-6 if first else True):
                            return t0 + (t1 - t0) * (min(max(level, c0), c1) - c0) / (c1 - c0)
                    return None
                start, end = when(h, True), when(h + 1, False)
                if start is not None and end is not None:
                    rows.append({"stage": st.name, "heat": h + 1, "start": start, "end": end})
    else:
        info = m._rtn
        for h in range(info["n_heats"]):
            on = [t for t in range(T) if pyo.value(m.S[h, t]) > 0.5]
            if on:
                rows.append({"stage": plant.stages[0].name, "heat": h + 1, "start": on[0] * dt, "end": (on[-1] + 1) * dt})
            for j in range(1, plant.n_stages):
                i = info["task_proc"](j, h)
                s0 = [t for t in range(T) if pyo.value(m.N[i, t]) > 0.5]
                if s0:
                    rows.append({"stage": plant.stages[j].name, "heat": h + 1, "start": s0[0] * dt,
                                 "end": (s0[0] + info["slots"][0, j]) * dt})
    return pd.DataFrame(rows, columns=["stage", "heat", "start", "end"])
