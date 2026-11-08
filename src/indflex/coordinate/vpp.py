"""Virtual power plant (VPP): bidding for regulation, then fast power disaggregation.

A VPP aggregates many small, different resources (EVs, storage, air conditioning,
industrial production stages, PV) and sells energy and frequency regulation. Two
decisions at two time scales:

1. **Bidding** (every 30 min): an LP over the rest of the day chooses hourly energy and
   regulation bids. The regulation signal is uncertain, so the LP plans each resource's
   power under discretized signal scenarios ``d_s`` with their probabilities.
2. **Disaggregation** (every 2 s): when the grid operator sends a regulation signal
   ``δ``, the VPP must deliver ``P_req = Bid_P + Bid_R · δ`` *now*. Re-solving an
   optimization every 2 s is too slow. Instead, the bidding LP's **dual variables**
   ``λ`` (the value of each resource's stored energy) turn each resource into
   power segments with a constant marginal cost; sorting them gives a **merit order**,
   and filling it up to ``P_req`` is optimal for the disaggregation problem.

All resources share one *standardized model*: discharge/charge power limits, an
energy state with self-loss ``θ``, efficiency matrices ``η`` (off-diagonal terms couple
industrial stages through their buffers), external drift ``w`` and usage costs.

Reference: Q. Chen, R. Lyu, H. Guo, X. Su, "Real-Time Operation Strategy of Virtual Power
Plants With Optimal Power Disaggregation Among Heterogeneous Resources", Applied Energy
361 (2024) 122876; code: https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo

from ..data import DATA_DIR

STEPS_PER_SLOT = 1800  # 2-s signal steps per 1-h slot


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class VPPCase:
    kind: list[str]
    price_e: np.ndarray  # $/MWh, per slot
    price_reg: np.ndarray  # (T, 2): capacity, mileage
    s_perf: float
    hourly_mileage: np.ndarray
    hourly_distribution: np.ndarray  # (T, S) probability of each signal scenario
    d_s: np.ndarray  # (S,) signal scenarios
    index_none_reg: np.ndarray
    energy_init: np.ndarray
    energy_upper: np.ndarray  # (N, T)
    energy_lower: np.ndarray
    p_dis_upper: np.ndarray
    p_dis_lower: np.ndarray
    p_ch_upper: np.ndarray
    p_ch_lower: np.ndarray
    theta: np.ndarray
    eta_dis: np.ndarray  # (N, N)
    eta_ch: np.ndarray
    pr_dis: np.ndarray
    pr_ch: np.ndarray
    w: np.ndarray  # (N, T)
    signal: np.ndarray  # RegD, one value per 2 s
    delta_t: float = 1.0
    delta_t_req: float = 0.5
    penalty: float = 1e6
    description: str = ""

    @property
    def n(self) -> int:
        return len(self.kind)

    @property
    def T(self) -> int:
        return len(self.price_e)


def load_vpp_case(name: str = "day21") -> VPPCase:
    raw = json.loads((DATA_DIR / "vpp" / f"vpp_{name}.json").read_text())
    n = len(raw["kind"])

    def dense(entries):
        A = np.zeros((n, n))
        for i, j, v in entries:
            A[i, j] = v
        return A

    arr = lambda k: np.asarray(raw[k], dtype=float)  # noqa: E731
    return VPPCase(
        kind=raw["kind"], price_e=arr("price_e"), price_reg=arr("price_reg"), s_perf=raw["s_perf"],
        hourly_mileage=arr("hourly_mileage"), hourly_distribution=arr("hourly_distribution"), d_s=arr("d_s"),
        index_none_reg=np.asarray(raw["index_none_reg"], dtype=int),
        energy_init=arr("energy_init"), energy_upper=arr("energy_upper"), energy_lower=arr("energy_lower"),
        p_dis_upper=arr("p_dis_upper"), p_dis_lower=arr("p_dis_lower"),
        p_ch_upper=arr("p_ch_upper"), p_ch_lower=arr("p_ch_lower"),
        theta=arr("theta"), eta_dis=dense(raw["eta_dis"]), eta_ch=dense(raw["eta_ch"]),
        pr_dis=arr("pr_dis"), pr_ch=arr("pr_ch"), w=arr("w"),
        signal=np.loadtxt(DATA_DIR / "vpp" / f"regd_{name}.csv"),
        delta_t=raw["delta_t"], delta_t_req=raw["delta_t_req"], penalty=raw["penalty"],
        description=raw.get("description", ""),
    )


# --------------------------------------------------------------------------------------
# Bidding LP (maxProfit_1 / maxProfit_t)
# --------------------------------------------------------------------------------------
@dataclass
class BidResult:
    bid_p: np.ndarray  # energy bids for the slots planned
    bid_r: np.ndarray  # regulation capacity bids
    energy: np.ndarray  # planned energy states (N, slots + 1)
    lam: np.ndarray  # value of each resource's current stored energy, d(profit)/dE
    p_none_reg: np.ndarray  # planned power of non-regulating resources in the current slot
    profit: float
    seconds: float
    slots: list[int] = field(default_factory=list)


def _nz(A):
    return {i: [(j, A[i, j]) for j in np.nonzero(A[i])[0]] for i in range(A.shape[0])}


def bid(case: VPPCase, cur_slot: int | None = None, E_cur: np.ndarray | None = None,
        bid_cur: tuple[float, float] = (0.0, 0.0), rest: float = 1.0) -> BidResult:
    """Solve the bidding LP.

    ``cur_slot=None``: day-ahead bidding over all slots (``maxProfit_1``).
    Otherwise, re-bid during slot ``cur_slot`` (0-based) for the slots after it, given the
    current energy ``E_cur``, the bids already accepted for the current slot ``bid_cur``
    and the remaining fraction ``rest`` of the current slot (``maxProfit_t``).
    """
    from pyomo.contrib.appsi.solvers import Highs

    c = case
    N, S, dt, dtr = c.n, len(c.d_s), c.delta_t, c.delta_t_req
    realtime = cur_slot is not None
    first = cur_slot if realtime else 0
    slots = list(range(first, c.T))  # slots with power variables (current one first if real-time)
    bid_slots = slots[1:] if realtime else slots
    K = len(slots)
    E0 = c.energy_init if E_cur is None else np.asarray(E_cur, float)
    ed, ec = _nz(c.eta_dis), _nz(c.eta_ch)
    none_reg = set(c.index_none_reg.tolist())
    Is, Ss, Ks = range(N), range(S), range(K)

    m = pyo.ConcreteModel(name="VPP bidding")
    m.bid_p = pyo.Var(bid_slots)
    m.bid_r = pyo.Var(bid_slots)
    m.p_dis = pyo.Var(Is, Ks, Ss, bounds=lambda m, i, k, s: (c.p_dis_lower[i, slots[k]], c.p_dis_upper[i, slots[k]]))
    m.p_ch = pyo.Var(Is, Ks, Ss, bounds=lambda m, i, k, s: (c.p_ch_lower[i, slots[k]], c.p_ch_upper[i, slots[k]]))
    m.E = pyo.Var(Is, range(K + 1))
    m.init = pyo.Constraint(Is, rule=lambda m, i: m.E[i, 0] == E0[i])  # duals -> λ
    cons = m.cons = pyo.ConstraintList()

    dist = c.hourly_distribution
    exp = lambda var, i, k: sum(dist[slots[k], s] * var[i, k, s] for s in Ss)  # noqa: E731  (expected power)

    def cost(k, s):
        return sum(c.pr_dis[i] * m.p_dis[i, k, s] + c.pr_ch[i] * m.p_ch[i, k, s] for i in Is)

    # power balance in every signal scenario
    for k in Ks:
        t = slots[k]
        for s in Ss:
            total = sum(m.p_dis[i, k, s] - m.p_ch[i, k, s] for i in Is)
            if realtime and k == 0:
                cons.add(bid_cur[0] + bid_cur[1] * c.d_s[s] - total == 0)
            else:
                cons.add(m.bid_p[t] + m.bid_r[t] * c.d_s[s] - total == 0)
        for i in none_reg:  # same output in every scenario
            for s in range(1, S):
                cons.add(m.p_dis[i, k, s] == m.p_dis[i, k, 0])
                cons.add(m.p_ch[i, k, s] == m.p_ch[i, k, 0])

    # energy dynamics (expected power) and limits
    pen = []
    for k in Ks:
        t = slots[k]
        length = rest if (realtime and k == 0) else dt
        decay = (1 - rest * (1 - c.theta)) if (realtime and k == 0) else c.theta
        for i in Is:
            cons.add(m.E[i, k + 1] == decay[i] * m.E[i, k]
                     + sum(v * exp(m.p_ch, j, k) for j, v in ec[i]) * length
                     - sum(v * exp(m.p_dis, j, k) for j, v in ed[i]) * length + c.w[i, t] * length)
    if realtime:
        m.dE1 = pyo.Var(Is, range(1, K + 1), domain=pyo.NonNegativeReals)
        m.dE2 = pyo.Var(Is, range(1, K + 1), domain=pyo.NonNegativeReals)
        m.dE3 = pyo.Var(Is, range(1, K), domain=pyo.NonNegativeReals)
        m.dE4 = pyo.Var(Is, range(1, K), domain=pyo.NonNegativeReals)
        pen = [m.dE1, m.dE2, m.dE3, m.dE4]
    for i in Is:
        for k in range(1, K + 1):  # E at the end of slot slots[k-1]
            t = slots[k - 1]
            lo = c.energy_lower[i, t] - (m.dE1[i, k] if realtime else 0)
            hi = c.energy_upper[i, t] + (m.dE2[i, k] if realtime else 0)
            cons.add(m.E[i, k] >= lo)
            cons.add(m.E[i, k] <= hi)
        # sustain the full regulation signal for delta_t_req hours (paper's duration constraints)
        dur = [(k, k) for k in Ks] if not realtime else [(k, k) for k in range(1, K)]
        for k_e, k_p in dur:
            t = slots[k_p]
            t_prev = max(t - 1, 0) if not realtime else t - 1
            keep = 1 - dtr * (1 - c.theta[i])
            cons.add(keep * m.E[i, k_e] - dtr * sum(v * m.p_dis[j, k_p, S - 1] for j, v in ed[i]) + dtr * c.w[i, t]
                     >= c.energy_lower[i, t_prev] - (m.dE3[i, k_e] if realtime else 0))
            # NOTE: transcribed as in the paper code (charging energy enters with a minus sign)
            cons.add(keep * m.E[i, k_e] - dtr * sum(v * m.p_ch[j, k_p, 0] for j, v in ec[i]) + dtr * c.w[i, t]
                     <= c.energy_upper[i, t_prev] + (m.dE4[i, k_e] if realtime else 0))

    k_bid = [(k, slots[k]) for k in Ks if slots[k] in bid_slots]
    revenue = sum(c.price_e[t] * m.bid_p[t] + c.price_reg[t, 0] * m.bid_r[t] * c.s_perf
                  + c.price_reg[t, 1] * c.hourly_mileage[t] * m.bid_r[t] * c.s_perf
                  + float(dist[t] @ c.d_s) * c.price_e[t] * m.bid_r[t]
                  - sum(dist[t, s] * cost(k, s) for s in Ss) for k, t in k_bid) * dt
    if realtime:
        t0 = slots[0]
        revenue -= float(dist[t0] @ c.d_s) * c.price_e[t0] * bid_cur[1] * rest
        revenue -= sum(dist[t0, s] * cost(0, s) for s in Ss) * rest
        revenue -= c.penalty * sum(sum(v.values()) for v in [dict(p.items()) for p in pen]) if pen else 0
    m.profit = pyo.Objective(expr=revenue, sense=pyo.maximize)

    opt = Highs()
    opt.config.load_solution = False
    t_start = time.perf_counter()
    res = opt.solve(m)
    seconds = time.perf_counter() - t_start
    if res.termination_condition.name != "optimal":
        raise RuntimeError(f"bidding LP: {res.termination_condition.name}")
    res.solution_loader.load_vars()
    duals = res.solution_loader.get_duals([m.init[i] for i in Is])
    lam = np.array([duals[m.init[i]] for i in Is])  # d(profit)/dE_cur
    return BidResult(
        bid_p=np.array([pyo.value(m.bid_p[t]) for t in bid_slots]),
        bid_r=np.array([pyo.value(m.bid_r[t]) for t in bid_slots]),
        energy=np.array([[pyo.value(m.E[i, k]) for k in range(K + 1)] for i in Is]),
        lam=lam,
        p_none_reg=np.array([pyo.value(m.p_ch[i, 0, 0]) for i in c.index_none_reg]),
        profit=float(pyo.value(m.profit)), seconds=seconds, slots=bid_slots,
    )


# --------------------------------------------------------------------------------------
# Fast disaggregation (fastControl_prepare / fastControl_implement)
# --------------------------------------------------------------------------------------
@dataclass
class Segments:
    """Power segments in merit order: resource, charge flag, marginal cost, power bounds.

    Discharge segments have power in [p_lo, p_hi] >= 0; charge segments are negative.
    """

    der: np.ndarray
    charge: np.ndarray
    cost: np.ndarray
    p_lo: np.ndarray
    p_hi: np.ndarray


def merit_order(case: VPPCase, lam: np.ndarray, E_cur: np.ndarray, slot: int, p_none_reg: np.ndarray,
                eta_convention: str = "transpose") -> Segments:
    """Build and sort the power segments of every resource.

    Marginal cost of discharging resource i: ``pr_dis_i + (η_disᵀ λ)_i`` (usage cost plus
    the value of the stored energy it uses); of charging, per unit of (negative) power:
    ``-pr_ch_i + (η_chᵀ λ)_i``. ``eta_convention="paper"`` uses ``η λ`` (rows) as in the
    paper code; the two differ only for coupled resources (industrial stages).
    """
    c, N = case, case.n
    if eta_convention == "transpose":
        v_dis, v_ch = c.eta_dis.T @ lam, c.eta_ch.T @ lam
    elif eta_convention == "paper":
        v_dis, v_ch = c.eta_dis @ lam, c.eta_ch @ lam
    else:
        raise ValueError("eta_convention must be 'transpose' or 'paper'")
    t = slot
    low, high = E_cur < c.energy_lower[:, t], E_cur > c.energy_upper[:, t]
    d_lo, d_hi = c.p_dis_lower[:, t].copy(), np.where(low, c.p_dis_lower[:, t], c.p_dis_upper[:, t])
    ch_lo = np.where(high, -c.p_ch_lower[:, t], -c.p_ch_upper[:, t])
    ch_hi = np.where(low, -c.p_ch_upper[:, t], -c.p_ch_lower[:, t])  # empty storage: charge at full power
    ch_lo[c.index_none_reg] = -p_none_reg  # non-regulating resources follow their schedule
    ch_hi[c.index_none_reg] = -p_none_reg
    ipp = [i for i in range(N) if c.kind[i] == "IPP"]
    for i in ipp[:-1]:  # an empty buffer stops the next stage
        if E_cur[i] < c.energy_lower[i, t]:
            ch_lo[i + 1] = -c.p_ch_lower[i + 1, t]
    der = np.r_[np.arange(N), np.arange(N)]
    seg = Segments(der=der, charge=np.r_[np.zeros(N, bool), np.ones(N, bool)],
                   cost=np.r_[c.pr_dis + v_dis, -c.pr_ch + v_ch], p_lo=np.r_[d_lo, ch_lo], p_hi=np.r_[d_hi, ch_hi])
    order = np.argsort(seg.cost, kind="stable")
    return Segments(*(getattr(seg, f)[order] for f in ("der", "charge", "cost", "p_lo", "p_hi")))


def allocate(seg: Segments, p_req: float) -> np.ndarray:
    """Merit-order fill: start every segment at its lower bound, raise the cheapest first."""
    p = seg.p_lo.copy()
    gap = p_req - p.sum()
    if gap > 0:
        room = seg.p_hi - seg.p_lo
        cum = np.cumsum(room)
        k = np.searchsorted(cum, gap)  # first segment that is not completely filled
        p[:k] = seg.p_hi[:k]
        if k < len(p):
            p[k] += gap - (cum[k - 1] if k > 0 else 0.0)
    return p


def allocate_lp(seg: Segments, p_req: float) -> np.ndarray:
    """The disaggregation LP that :func:`allocate` solves in closed form (for comparison)."""
    from pyomo.contrib.appsi.solvers import Highs

    m = pyo.ConcreteModel()
    K = range(len(seg.cost))
    m.p = pyo.Var(K, bounds=lambda m, k: (seg.p_lo[k], seg.p_hi[k]))
    m.balance = pyo.Constraint(expr=sum(m.p[k] for k in K) == p_req)
    m.obj = pyo.Objective(expr=sum(seg.cost[k] * m.p[k] for k in K))
    Highs().solve(m)
    return np.array([pyo.value(m.p[k]) for k in K])


def to_resources(seg: Segments, p: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Segment powers -> (discharge, charge) power per resource (both >= 0)."""
    p_dis, p_ch = np.zeros(n), np.zeros(n)
    np.add.at(p_dis, seg.der[~seg.charge], p[~seg.charge])
    np.add.at(p_ch, seg.der[seg.charge], -p[seg.charge])
    return p_dis, p_ch


# --------------------------------------------------------------------------------------
# Simulating real-time operation
# --------------------------------------------------------------------------------------
@dataclass
class HourResult:
    p_req: np.ndarray  # requested power per 2-s step
    p_dis: np.ndarray  # (N, steps)
    p_ch: np.ndarray
    energy: np.ndarray  # (N, control updates + 1)
    alloc_seconds: float  # mean time per allocation


def simulate_hour(case: VPPCase, slot: int, bid_p: float, bid_r: float, lam: np.ndarray, E_cur: np.ndarray,
                  p_none_reg: np.ndarray, ctrl_steps: int = 30, eta_convention: str = "transpose",
                  n_steps: int = STEPS_PER_SLOT) -> HourResult:
    """Follow the RegD signal for (the first ``n_steps`` of) one slot with the merit order.

    The merit order is rebuilt and the energy states updated every ``ctrl_steps`` steps (1 min).
    """
    c, N = case, case.n
    sig = c.signal[slot * STEPS_PER_SLOT:slot * STEPS_PER_SLOT + n_steps]
    E = np.asarray(E_cur, float).copy()
    p_dis, p_ch, energies, p_req, spent = [], [], [E.copy()], [], 0.0
    dtc = ctrl_steps / STEPS_PER_SLOT
    for start in range(0, n_steps, ctrl_steps):
        seg = merit_order(c, lam, E, slot, p_none_reg, eta_convention)
        win_dis, win_ch = [], []
        for delta in sig[start:start + ctrl_steps]:
            req = bid_p + bid_r * delta
            t0 = time.perf_counter()
            p = allocate(seg, req)
            spent += time.perf_counter() - t0
            d, ch = to_resources(seg, p, N)
            win_dis.append(d); win_ch.append(ch); p_req.append(req)
        win_dis, win_ch = np.array(win_dis).T, np.array(win_ch).T
        E = ((1 - dtc * (1 - c.theta)) * E + c.eta_ch @ win_ch.sum(1) / STEPS_PER_SLOT
             - c.eta_dis @ win_dis.sum(1) / STEPS_PER_SLOT + c.w[:, slot] * dtc)
        energies.append(E.copy()); p_dis.append(win_dis); p_ch.append(win_ch)
    return HourResult(np.array(p_req), np.hstack(p_dis), np.hstack(p_ch), np.array(energies).T,
                      spent / n_steps)
