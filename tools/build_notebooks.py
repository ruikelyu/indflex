"""Build the teaching notebooks from source cells (keeps notebook diffs reviewable).

    python tools/build_notebooks.py [03 ...]   # all notebooks, or only those listed
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NB01 = [
    md(r"""
# 01 · STN vs. LSTN: scheduling a continuous industrial process

**Where this sits in the chain:** *process model* → parameter identification → dimension reduction → grid/aggregator optimization.

This notebook covers the first step for **continuous** processes (crushing, grinding, cement, steel powder):

1. Build a **State-Task Network (STN)** for a 10-stage steel powder plant and let it respond to an hourly electricity price.
2. Replace the STN's on/off binaries with **time-in-state** fractions to get the **LSTN**, which is a linear program.
3. Compare accuracy (cost, load profile) and **solve time as the number of factories grows**.

Every cell runs in seconds with the free HiGHS solver. The scaling demo (Section 4) takes about 1–2 minutes.

*References:* thesis Ch. 2; R. Lyu et al., "LSTN: A Linear Model of Industrial Production Process for Demand Response," ISGT Europe 2023. Plant data: R. Lu et al., *Appl. Energy* 283 (2021).
"""),
    code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indflex as fx
from indflex import viz

plt.rcParams["figure.dpi"] = 110
"""),
    md(r"""
## 1. The plant: a serial production line

Each **stage** is a machine with a few **operating modes**. Mode $p$ of machine $n$ processes $g_{n,p}$ t/h and draws $e_{n,p}$ kW (mode 0 is *off*). Between consecutive stages sits a **buffer** (storage) with capacity $S^{\max}_n$. That buffer is what allows flexibility: upstream stages can run ahead while power is cheap, and downstream stages catch up later.

```
raw → [1 Atomizer] → S₁ → [2 Dehydrator] → S₂ → … → [10 Blender] → S₁₀ (final product)
```

The plant must deliver its daily output target $S^{\rm tar}$ by the end of the day.
"""),
    code(r"""
plant = fx.load_plant("steel_powder_lu2021")
print(plant.description, "\nSource:", plant.source)
pd.DataFrame(
    [{"stage": f"{k+1}. {s.name}",
      "modes (g t/h, e kW)": ", ".join(f"({m.g:g}, {m.e:g})" for m in s.modes),
      "buffer max (t)": s.s_max, "initial (t)": s.s_init, "daily target (t)": s.s_target_change}
     for k, s in enumerate(plant.stages)]
)
"""),
    md(r"""
## 2. STN: on/off scheduling (a MILP)

Decision variable $x_{n,p,t}\in\{0,1\}$: machine $n$ runs in mode $p$ during hour $t$.

$$
\begin{aligned}
\min\ & \textstyle\sum_t \pi_t \sum_n E_{n,t}, \qquad E_{n,t}=\sum_p e_{n,p}\,x_{n,p,t}\,\Delta t \\
\text{s.t. }& \textstyle\sum_p x_{n,p,t}=1 &&\text{one mode at a time}\\
& S_{n,t+1}=S_{n,t}+G_{n,t}-G_{n+1,t},\quad G_{n,t}=\sum_p g_{n,p}\,x_{n,p,t}\,\Delta t &&\text{buffer balance}\\
& 0\le S_{n,t}\le S^{\max}_n,\qquad S_{n,T}\ge S_{n,0}+S^{\rm tar}_n &&\text{capacity and target}
\end{aligned}
$$

Use the sliders to change the day (PJM real-time prices, July 2022) and the daily production target. Watch the plant move its load into cheap hours.
"""),
    code(r"""
from ipywidgets import interact, IntSlider

prices = fx.load_prices("2022-07")

@interact(day=IntSlider(23, 1, 31, description="July day"),
          target=IntSlider(240, 120, 300, step=20, description="target (t)"))
def show_stn(day, target):
    price = prices.iloc[day - 1].to_numpy()
    r = fx.solve(fx.build_stn(plant.with_target(target), price))
    if r.status != "optimal":
        print("No feasible schedule:", r.status)
        return
    viz.plot_schedule(r, price, title=f"STN, July {day}: cost ${r.cost:,.2f}, {r.n_binary} binaries, {r.solve_seconds:.2f} s")
    plt.show()
"""),
    md(r"""
## 3. LSTN: model *how long* each machine runs in each mode

Market intervals (1 h) are much longer than the time a machine needs to switch modes. So instead of asking *which* mode the machine is in for the whole hour, the LSTN asks *how long* it spends in each mode:

$$x_{n,p,t}\in[0,1]\quad\text{(fraction of hour } t \text{ spent in mode } p\text{)},\qquad \textstyle\sum_p x_{n,p,t}=1.$$

All other constraints are unchanged. The model becomes a **linear program**. In code, the only difference is `relax=True`:
"""),
    code(r"""
day = 23
price = prices.iloc[day - 1].to_numpy()
stn = fx.solve(fx.build_stn(plant, price))
lstn = fx.solve(fx.build_lstn(plant, price))
print("STN :", stn.summary())
print("LSTN:", lstn.summary())
print(f"Cost gap: {(stn.cost - lstn.cost) / stn.cost:.2%}  (the LSTN is a lower bound on the STN)")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
viz.plot_schedule(stn, price, title=f"STN  (MILP): ${stn.cost:,.2f}", ax=axes[0])
viz.plot_schedule(lstn, price, title=f"LSTN (LP):   ${lstn.cost:,.2f}", ax=axes[1])
plt.tight_layout(); plt.show()
"""),
    md(r"""
**How close is it on every day?** Solve both models for all 31 days of July (about 5 s):
"""),
    code(r"""
rows = []
for d in range(1, 32):
    p = prices.iloc[d - 1].to_numpy()
    a, b = fx.solve(fx.build_stn(plant, p)), fx.solve(fx.build_lstn(plant, p))
    rows.append({"day": d, "STN cost": a.cost, "LSTN cost": b.cost,
                 "cost gap %": 100 * (a.cost - b.cost) / a.cost,
                 "load profile diff %": 100 * np.abs(a.load - b.load).sum() / a.load.sum()})
gap = pd.DataFrame(rows).set_index("day")
display(gap.describe().loc[["mean", "max"]].round(3))
gap[["cost gap %", "load profile diff %"]].plot(marker="o", figsize=(9, 3), title="LSTN vs. STN, July 2022"); plt.show()
"""),
    md(r"""
## 4. Scaling: many factories at once

An aggregator or a grid model needs **many** factories at once. Here we schedule fleets of randomized steel-powder factories (parameters scaled by random factors in [0.8, 1.2], as in the LSTN paper) against one price.

* **STN**: binaries grow with the fleet, and HiGHS soon hits the time limit.
* **LSTN**: an LP that stays fast for hundreds of factories.

The sizes below are chosen so that the whole cell finishes in about 1–2 minutes: the STN reaches the time limit at around 5 factories, while the LSTN solves 200 in about 10 s. Change `TIME_LIMIT` or the size lists to explore. With a commercial solver (Gurobi or CPLEX), the paper solves 2,000 LSTN factories in minutes, while the STN fails to converge within 2 h for 20 factories.
"""),
    code(r"""
TIME_LIMIT = 30                       # seconds per solve
STN_SIZES = [1, 2, 3, 5]
LSTN_SIZES = [1, 2, 5, 10, 20, 50, 100, 200]

fleet = fx.load_fleet(200)            # take the first n factories of one 200-factory fleet
price_aug5 = fx.daily_price("2022-08", 5)
rows = []
for model, sizes, relax in [("STN", STN_SIZES, False), ("LSTN", LSTN_SIZES, True)]:
    for n in sizes:
        r = fx.solve(fx.build_fleet(fleet[:n], price_aug5, relax=relax), time_limit=TIME_LIMIT)
        rows.append({"model": model, "n": n, "seconds": r.solve_seconds, "status": r.status,
                     "binaries": r.n_binary, "cost": r.cost})
        print(f"{model:4s} n={n:4d}: {r.status:10s} {r.solve_seconds:6.1f} s, {r.n_binary} binaries")

scaling = pd.DataFrame(rows)
viz.plot_scaling(rows, time_limit=TIME_LIMIT); plt.title("Solve time (HiGHS)"); plt.show()
"""),
    md(r"""
## 5. Things to try

1. Shrink a buffer: `fx.load_plant(...)` returns a frozen dataclass; build a modified copy with `dataclasses.replace` and see how much flexibility disappears.
2. Raise the target to 300 t in Section 2. When does the plant lose its ability to avoid the evening peak?
3. Use 15-minute intervals: `fx.build_lstn(plant, np.repeat(price, 4), dt=0.25)`.

**Takeaways**
* For continuous processes, the LSTN keeps the STN's material-flow logic but drops the binaries, so it scales to fleets.
* Its cost is a lower bound on the STN's. For this plant it is within about 1% on every day of July 2022 (0.4% on average), and the hourly load profiles differ by about 4% on average.
* Batch processes (e.g. steelmaking heats that cannot be interrupted) need more structure → **notebook 02: RTN vs. cRTN** 
"""),
]

NB02 = [
    md(r"""
# 02 · RTN vs. cRTN: scheduling a batch process (steelmaking)

**Where this sits in the chain:** *process model* → parameter identification → dimension reduction → grid/aggregator optimization.

Notebook 01 handled **continuous** processes. Many energy-intensive plants instead process discrete **batches** (in steelmaking, *heats*):

* **Uninterruptible tasks:** once a heat starts melting in the electric arc furnace (EAF), it must finish.
* **Waiting limits:** a hot heat may wait only so long before the next stage.
* **Flexible power:** the EAF can melt faster at high power or slower at low power.

This notebook compares two models of the same line:

| | Conventional **RTN** (Zhang et al. 2017) | **cRTN** (this thesis) |
|---|---|---|
| Decision | binary: *does task i for heat h start in slot t?* | continuous: *how long does task i run at each operating point in slot t?* |
| Grows with the number of heats? | yes, there is one set of tasks per heat | no, tasks are shared by all heats |
| Task durations | rounded up to whole slots | exact (fractions of a slot) |

*References:* thesis Ch. 2; R. Lyu et al., "Efficient Scheduling of Discrete Industrial Processes through Continuous Modeling," *IEEE Trans. Smart Grid*, in press; paper code: [Efficient-Scheduling-of-Discrete-Industrial-Processes](https://github.com/ruikelyu/Efficient-Scheduling-of-Discrete-Industrial-Processes). Plant data: X. Zhang et al., *IEEE Trans. Smart Grid* 8(1), 2017.
"""),
    code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indflex as fx
from indflex import viz

plt.rcParams["figure.dpi"] = 110
"""),
    md(r"""
## 1. A one-furnace example: why rounding matters

One EAF of 1 MW must melt **2 heats** in 6 hours. A heat takes 2 h at nominal power, and the furnace can run between 50% and 133% of nominal power, so a heat takes between 1.5 h and 4 h. Two 6-hour price profiles:
"""),
    code(r"""
small = fx.load_batch_plant("eaf_small_case")
scenarios = {"scenario 1": np.array([100, 100, 100, 100, 200, 200.]),
             "scenario 2": np.array([100, 100, 100, 200, 200, 100.])}

fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharey=True)
for col, (label, price) in enumerate(scenarios.items()):
    for row, (name, build) in enumerate([("RTN", fx.build_rtn), ("cRTN", fx.build_crtn)]):
        r = fx.solve(build(small, price, n_heats=2))
        viz.plot_schedule(r, price, ax=axes[row, col],
                          title=f"{label}, {name}: ${r.cost:,.2f} ({r.n_binary} binaries)")
        axes[row, col].get_legend().remove()
plt.tight_layout(); plt.show()
"""),
    md(r"""
In **scenario 2**, the cheap hours are 0–2 and 5. The cRTN melts both heats at high power in the first three hours (1.5 h each, so two heats in 3 h) for **\$400**. The RTN can only start and finish tasks on slot boundaries: a 1.5 h melt is rounded up to 2 slots, so the second heat spills into an expensive hour and the cost rises to **\$466.67**. In the paper, this rounding error grows with the number of stages.

## 2. The steelmaking line

Four serial stages: EAF (85 MW, power-flexible 75–125%) → AOD → LF → continuous caster (CC). Use the sliders to change the day (PJM real-time prices, July 2022) and the number of heats. Both models are solved at **30-minute** slots so that each solve takes a few seconds.
"""),
    code(r"""
steel = fx.load_batch_plant("steelmaking_zhang2017")
pd.DataFrame([vars(s) for s in steel.stages]).rename(columns={
    "power_mw": "power (MW)", "processing_min": "processing (min)",
    "transfer_min": "transfer (min)", "max_wait_min": "max wait (min)"})
"""),
    code(r"""
from ipywidgets import interact, IntSlider

DT = 0.5  # hours per slot
prices = fx.load_prices("2022-07")

@interact(day=IntSlider(26, 1, 31, description="July day"), heats=IntSlider(2, 1, 3, description="heats"))
def show(day, heats):
    price = np.repeat(prices.iloc[day - 1].to_numpy(), int(1 / DT))
    rtn = fx.solve(fx.build_rtn(steel, price, heats, dt=DT))
    crtn = fx.solve(fx.build_crtn(steel, price, heats, dt=DT))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    viz.plot_schedule(rtn, price, dt=DT, ax=axes[0],
                      title=f"RTN:  ${rtn.cost:,.0f}, {rtn.n_binary} binaries, {rtn.solve_seconds:.1f} s")
    viz.plot_schedule(crtn, price, dt=DT, ax=axes[1],
                      title=f"cRTN: ${crtn.cost:,.0f}, {crtn.n_binary} binaries, {crtn.solve_seconds:.1f} s")
    plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3. Model size: binaries vs. the number of heats

This is the core idea of the cRTN. The conventional RTN creates tasks and resources **for every heat**, so its binary variables grow with the production target. The cRTN shares one set of tasks across all heats and tracks *progress* with continuous variables, so its binary count does not depend on the number of heats.

Counting variables only requires building the models, not solving them, so we can use the paper's **5-minute** resolution here:
"""),
    code(r"""
price_5min = np.repeat(fx.daily_price("2022-07", 26), 12)
rows = []
for heats in range(1, 11):
    for name, build in [("RTN", fx.build_rtn), ("cRTN", fx.build_crtn)]:
        nb, nc = fx.count_vars(build(steel, price_5min, heats, dt=1 / 12))
        rows.append({"model": name, "heats": heats, "binary": nb, "continuous": nc})
size = pd.DataFrame(rows)
display(size.pivot(index="heats", columns="model", values="binary"))

ax = size.pivot(index="heats", columns="model", values="binary").plot(
    marker="o", color={"RTN": "tab:red", "cRTN": "tab:blue"}, figsize=(6.5, 4))
ax.set_ylabel("Binary variables"); ax.set_title("Steelmaking, 5-min slots (288 per day)")
ax.set_ylim(0); ax.grid(alpha=0.3); plt.show()
"""),
    md(r"""
## 4. Solve time as the number of heats grows

Fewer binaries do not guarantee a faster solve on every small case. With 1–3 heats, the RTN is small too and often solves faster. The difference appears as the production target grows. The cell below solves both models with the free HiGHS solver at 30-minute slots for 1–6 heats (about 2 minutes in total; each solve is capped at `TIME_LIMIT`). In our runs on a laptop, the cRTN stays under about 10 s, while the RTN overtakes it at around 4 heats and reaches the time limit at 6. The RTN's schedules also cost 3–8% more, because task durations are rounded up to whole slots. (At 6 heats the RTN was stopped at the time limit, so part of that gap is unfinished optimization.)
"""),
    code(r"""
TIME_LIMIT = 45
HEATS = [1, 2, 3, 4, 6]
price_30min = np.repeat(fx.daily_price("2022-07", 26), 2)

rows = []
for heats in HEATS:
    for name, build in [("RTN", fx.build_rtn), ("cRTN", fx.build_crtn)]:
        r = fx.solve(build(steel, price_30min, heats, dt=0.5), time_limit=TIME_LIMIT)
        rows.append({"model": name, "n": heats, "seconds": r.solve_seconds, "status": r.status,
                     "binaries": r.n_binary, "cost": r.cost})
        print(f"{name:4s} heats={heats}: {r.status:10s} {r.solve_seconds:5.1f} s, cost ${r.cost:,.0f}")

timing = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
viz.plot_scaling(rows, time_limit=TIME_LIMIT, ax=axes[0], xlabel="Number of heats", log_x=False)
axes[0].set_title("Solve time (HiGHS, 30-min slots)")
gap = timing.pivot(index="n", columns="model", values="cost")
(100 * (gap["RTN"] - gap["cRTN"]) / gap["cRTN"]).plot.bar(ax=axes[1], color="tab:grey")
axes[1].set_ylabel("RTN cost above cRTN (%)"); axes[1].set_xlabel("Number of heats")
axes[1].set_title("Cost of rounding task durations to whole slots")
plt.tight_layout(); plt.show()
"""),
    md(r"""
**At the paper's scale.** The thesis uses 5-minute slots, Gurobi, and up to 8 heats. There, the RTN's solve time grows roughly exponentially with the number of heats while the cRTN's grows roughly linearly; for 8 heats, the mean solve time over 15 July days falls from **156.7 to 17.1 minutes (−89%)**. This is a different test case from the LSTN scaling result in notebook 01. To try it yourself with Gurobi (`pip install gurobipy` plus a license), set `RUN_GUROBI = True`. Expect minutes per solve.
"""),
    code(r"""
RUN_GUROBI = False

if RUN_GUROBI:
    price_5min = np.repeat(fx.daily_price("2022-07", 26), 12)
    for heats in [2, 4, 8]:
        for name, build in [("RTN", fx.build_rtn), ("cRTN", fx.build_crtn)]:
            r = fx.solve(build(steel, price_5min, heats, dt=1 / 12), solver="gurobi", mip_gap=1e-4, time_limit=1800)
            print(f"{name:4s} heats={heats}: {r.summary()}")
"""),
    md(r"""
## 5. How the cRTN relates to the LSTN

Both models replace **"which state is the machine in during this slot?"** (binary) with **"how long does it spend in each state?"** (continuous). For continuous processes that removes every binary (LSTN). Batch processes still need a few status binaries for the uninterruptible and waiting logic, but no longer one per heat (cRTN). In the thesis (Ch. 2 §2.5.5), the STN is shown to be a special case of the RTN, and the LSTN a special case of the cRTN.

## 6. Things to try

1. Make the EAF inflexible: `dataclasses.replace(steel, power_ratio=(1.0, 1.0))`. How much of the cost saving disappears?
2. Change `DT` in Section 2 to `0.25`. The RTN's rounding error shrinks, but its model grows.
3. Tighten the waiting limit of the LF stage and watch the schedule compress.

**Takeaways**
* The cRTN keeps the batch logic (uninterruptible melting, waiting limits) but its size does not grow with the number of heats.
* It avoids the RTN's rounding of task durations to whole slots, which changes the cost even in a one-furnace example.
* The cRTN reduces binaries; it does not remove them all. Its solve-time advantage is a statement about growing problems (many heats, fine slots), not every small case.
* Next in the chain: what if we *don't know* the plant's parameters? → **notebook 03: parameter identification (PSI)**.
"""),
]

NB03 = [
    md(r"""
# 03 · PSI: fitting a factory's private parameters with inverse optimization

**Where this sits in the chain:** process model → *parameter identification* → dimension reduction → grid/aggregator optimization.

Notebooks 01–02 assumed we **know** the plant's parameters. A grid operator or aggregator usually doesn't: it sees only the factory's **hourly meter data** and the **prices** it faced. Production Scheduling Identification (PSI) asks:

> If the factory schedules production to minimize its electricity cost, which plant parameters make that optimal schedule reproduce the meter data?

This notebook shows the idea on a **2-machine plant whose true parameters we know**, so we can check that inverse optimization recovers them. A single solve takes about a second.

*References:* thesis Ch. 3; R. Lyu et al., "Production Scheduling Identification: An Inverse Optimization Approach for Industrial Load Modeling Using Smart Meter Data," *IEEE Trans. Smart Grid* 16(2), 2025; paper code: [Production-Scheduling-Identification](https://github.com/ruikelyu/Production-Scheduling-Identification).

**Solver note:** the inverse problem is nonconvex (bilinear), so this notebook uses **Gurobi** (`pip install gurobipy` plus a license; academic licenses are free). The size-limited license bundled with `gurobipy` is too small for it.
"""),
    code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indflex as fx
from indflex.identify import MSTNParams, simulate, train_psi, nrmse

plt.rcParams["figure.dpi"] = 110
"""),
    md(r"""
## 1. The plant (a modified STN) and its meter data

Raw material → **machine 1** → buffer → **machine 2** → final product. Machine $i$ draws power $P_{t,i}\in[0,P^{\max}_i]$ and processes $g_i P_{t,i}$ units of material per hour. Buffer $j$ holds between 0 and $S^{\max}_j$, starts the day at $S^0_j$, and must change by $S^{\rm tar}_j$ by the end of the day. The plant schedules its machines to minimize its electricity bill (an LP).

* **Public:** $g_i$ (energy per unit of product is an industry figure), prices, meter data.
* **Private, to be identified:** $\theta = (P^{\max}, S^0, S^{\rm tar}, S^{\max})$.

We pick "true" values, let the plant respond to 10 days of PJM prices (July 2022), and keep only what an outsider would see: the hourly load.
"""),
    code(r"""
true = MSTNParams(
    g=np.array([0.5, 0.25]),                # public
    p_max=np.array([4.0, 8.0]),             # rated power of machines 1 and 2 (MW)
    s_init=np.array([200.0, 10.0, 0.0]),    # raw material, intermediate, final product
    s_target=np.array([-130.0, 0.0, 30.0]), # daily change: use raw, make 30 units of product
    s_max=np.array([400.0, 20.0, 100.0]),
)
prices = fx.load_prices("2022-07").to_numpy()[:10] / 1000  # $/kWh, 10 training days
meter = simulate(true, prices)                              # what the meter records (MWh)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
for d in range(4):
    axes[0].plot(meter[d], label=f"July {d + 1}")
    axes[1].plot(prices[d] * 1000)
axes[0].set(title="Meter data (observed)", xlabel="Hour", ylabel="MWh"); axes[0].legend(fontsize=8)
axes[1].set(title="Prices", xlabel="Hour", ylabel="$/MWh")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 2. Inverse optimization

Given θ, the plant's response $E(\theta, \pi)$ is the solution of an LP. To go backwards, write down **what it means for a schedule to be optimal**, and let θ be a decision variable:

$$
\begin{aligned}
\min_{\theta,\,P,\,S,\,\lambda,\,\mu}\ & \textstyle\sum_t \big(E_t - E^{\rm meter}_t\big)^2 \\
\text{s.t. }& \text{primal feasibility: } P, S \text{ satisfy the mSTN with parameters } \theta \\
& \text{dual feasibility: } \lambda, \mu \text{ satisfy the LP's dual constraints} \\
& \text{strong duality: primal cost} = \text{dual cost} \quad(\text{dual variables} \times \theta \Rightarrow \text{bilinear})
\end{aligned}
$$

Together, the three conditions say that $(P,S)$ is an *optimal* schedule for θ. Two public priors from the paper help: only the raw-material and final-product buffers change over the day, and the rated powers add up to the peak observed load.

For this small plant, **one day of data and one solve are enough** (about 1 s):
"""),
    code(r"""
from pyomo.contrib.appsi.solvers import Gurobi
from indflex.identify import build_inverse_day
from indflex.identify.psi import theta_of

day = 0  # July 1
m = build_inverse_day(true.g, T=24, dt=1.0, max_load=float(meter.max()))
for t in range(24):
    m.price[t], m.E_obs[t] = prices[day, t], meter[day, t]

opt = Gurobi()
opt.gurobi_options["NonConvex"] = 2  # bilinear strong-duality constraint
opt.config.time_limit = 10
res = opt.solve(m)
theta = theta_of(m)
print(f"{res.termination_condition.name}, squared misfit {m.fit():.2e}")

rows = []
for name, label in [("p_max", "rated power"), ("s_init", "initial level"), ("s_target", "daily change"), ("s_max", "capacity")]:
    for k, (t, e) in enumerate(zip(getattr(true, name), getattr(theta, name))):
        what = f"machine {k + 1}" if name == "p_max" else ["raw material", "intermediate", "final product"][k]
        rows.append({"parameter": f"{label}, {what}", "true": t, "identified": round(e, 2)})
pd.DataFrame(rows).set_index("parameter")
"""),
    md(r"""
The rated powers and the daily production target are recovered. Some buffer values can differ from the truth without changing the plant's behavior, for example a buffer that never fills: the meter data cannot distinguish them, so any value consistent with the data is equally correct.

## 3. Check on unseen prices

The model identified from July 1 alone should predict the plant's response to other prices (August 1–5, 2022):
"""),
    code(r"""
test_prices = fx.load_prices("2022-08").to_numpy()[:5] / 1000
actual, predicted = simulate(true, test_prices), simulate(theta, test_prices)
print(f"test nRMSE: {100 * nrmse(predicted, actual):.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, d in zip(axes, [0, 2, 4]):
    ax.plot(actual[d], "k", lw=3, label="true plant")
    ax.plot(predicted[d], "--", color="tab:orange", lw=2, label="identified model")
    ax.set(title=f"August {d + 1}", xlabel="Hour")
axes[0].set_ylabel("MWh"); axes[0].legend(fontsize=8); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 4. When is iteration needed?

With many days of data, or a larger plant, two things change:

* **Solving all days at once is too large.** Each day adds its own primal and dual variables and its own bilinear constraint.
* **A single day can mislead.** Some days can be explained by the wrong parameters (for example, swapping which machine is large).

PSI therefore iterates: it solves one day at a time under a short time limit, and after a first pass it also penalizes the distance from θ to the average of recent estimates, until the estimates agree across days (a zeroth-order stochastic method). Set `RUN_ITERATIVE = True` to watch this on the same plant (about 20 s). Each point is the solution of one day's inverse problem, not an initial guess.
"""),
    code(r"""
RUN_ITERATIVE = False

if RUN_ITERATIVE:
    theta_it, hist = train_psi(true.g, prices, meter, n_iter=22, time_limit=1.0)
    pm = np.array([h.p_max for h in hist.theta])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i in range(2):
        ax.plot(pm[:, i], marker="o", ms=3, label=f"machine {i + 1} (true {true.p_max[i]:g} MW)")
        ax.axhline(true.p_max[i], color=f"C{i}", ls=":", lw=1)
    ax.axvline(len(prices), color="grey", ls="--", lw=1)
    ax.text(len(prices) + 0.3, ax.get_ylim()[1] * 0.92, "consensus penalty on", fontsize=8, color="grey")
    ax.set(xlabel="Iteration (= one day's inverse problem)", ylabel="Rated power (MW)",
           title="Identified rated powers over the iterations"); ax.legend(fontsize=8, loc="center right")
    plt.show()
"""),
    md(r"""
## 5. Optional: a real plant (cement, 4 stages)

The paper applies PSI to plants simulated with the STNs of notebook 01, whose on/off machines the mSTN can only approximate. This is much harder to solve. The paper's runs give Gurobi 120 s per iteration for about 210 iterations (hours of solver time) and reach 5.2% error on the cement plant. With 1 s per iteration, the solver often stops at a trivial "produce nothing" model. A prior from public data avoids that: every unit of product passes every machine, so the daily production target is about *daily energy* $/\sum_i 1/g_i$. Set `RUN_CEMENT = True` to try it (about 45 s; we saw 13–14% error).
"""),
    code(r"""
RUN_CEMENT = False

if RUN_CEMENT:
    from indflex.identify import make_meter_dataset
    from indflex.identify.psi import target_from_energy
    ds = make_meter_dataset(fx.load_plant("cement_golmohamadi2020"))
    g = np.array([1000 / 2200, 250 / 11000, 300 / 11550, 350 / 11220])
    target = target_from_energy(g, ds.load_train)
    th, h = train_psi(g, ds.price_train, ds.load_train, n_iter=43, target_bounds=(target, target))
    print(f"{h.seconds:.0f} s, test nRMSE {100 * nrmse(simulate(th, ds.price_test), ds.load_test):.1f}%")
"""),
    md(r"""
**Takeaways**
* PSI fits a *structured* model: the factory's optimal response under process constraints, not a free-form curve. Inverse optimization turns "the schedule must be optimal" into constraints, and the private parameters become decision variables.
* Strong duality makes the problem nonconvex. A small plant needs one solve; with many days or larger plants, PSI cycles through the days and averages the estimates, which keeps each solve small.
* Parameters the data never exercise (such as a buffer that never fills) cannot be identified, but they also do not change the predicted response.
* Next in the chain: thousands of such models are too complex for a grid model, so we compress them → **notebook 04: dimension reduction (D3R)**.
"""),
]

def main(only=None):
    """Write the notebooks (without outputs). ``only``: list of name prefixes, e.g. ["03"]."""
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for fname, cells in [("01_stn_vs_lstn.ipynb", NB01), ("02_rtn_vs_crtn.ipynb", NB02), ("03_psi.ipynb", NB03)]:
        if only and not any(fname.startswith(o) for o in only):
            continue
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, out / fname)
        print(f"wrote notebooks/{fname}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
