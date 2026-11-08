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

**Inside the two models.** The figure below follows the illustration in the thesis (Ch. 2, Figs. "Scheduling results of cRTN" and "Rounding error in the RTN model"). The RTN marks the furnace as busy for whole slots. The cRTN tracks the *progress* of the current heat, which rises from 0 to 1 and resets when the heat is output, so a heat can finish in the middle of a slot.

For these two figures, scenario 2 gets two extra expensive hours (200 $/MWh) at the end, so that every heat's delivery fits inside the plot; the schedules and costs are unchanged. The RTN also gets a tiny tie-breaker that prefers earlier task starts, because the timing of a free transfer is otherwise arbitrary (the paper code uses a quadratic term for the same purpose).
"""),
    code(r"""
import pyomo.environ as pyo
from indflex.models import crtn_progress, heat_schedule

price = np.r_[scenarios["scenario 2"], 200, 200]  # 8 hours
T = len(price)
m_rtn, m_crtn = fx.build_rtn(small, price, 2, tie_break=1e-2), fx.build_crtn(small, price, 2)
rtn_res, crtn_res = fx.solve(m_rtn, mip_gap=0), fx.solve(m_crtn)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax in axes:  # price background
    ax2 = ax.twinx(); ax2.fill_between(range(T + 1), np.append(price, price[-1]), step="post", color="grey", alpha=0.15)
    ax2.set_ylim(0, 220); ax2.set_ylabel("price ($/MWh)", color="grey")

# RTN: melting status of each heat, whole slots only
for h in range(2):
    status = [pyo.value(m_rtn.S[h, t]) for t in range(T)]
    axes[0].step(range(T + 1), status + [status[-1]], where="post", lw=2.5, label=f"heat {h + 1}")
axes[0].set(ylabel="melting\n(RTN status)", ylim=(-0.1, 1.3), title=f"RTN: ${rtn_res.cost:,.2f}")
axes[0].legend(fontsize=8, loc="upper right")

# cRTN: melting progress of the current heat
t, prog = crtn_progress(m_crtn, stage=0)
axes[1].plot(t, prog, color="tab:orange", lw=2.5)
for r in heat_schedule(m_crtn).itertuples():
    axes[1].annotate(f"heat {r.heat} done", (r.end, 1), xytext=(r.end - 0.2, 1.15), fontsize=8, ha="right")
axes[1].set(ylabel="melting progress\n(cRTN)", ylim=(-0.1, 1.3), xlabel="Hour", title=f"cRTN: ${crtn_res.cost:,.2f}")

# rounding error: the second heat really finishes at 3.0 h, the RTN can only finish at a slot boundary
rtn_end = heat_schedule(m_rtn)["end"].max()
axes[0].annotate("", xy=(rtn_end, 1.15), xytext=(3.0, 1.15), arrowprops=dict(arrowstyle="<->", color="red"))
axes[0].text((3.0 + rtn_end) / 2, 1.2, "rounding error", color="red", ha="center", fontsize=8)
plt.tight_layout(); plt.show()
"""),
    md(r"""
**Why the cRTN does not grow with the number of heats.** The same two heats, drawn as the variables each model tracks. The RTN needs a separate melting task, transfer task and pair of material resources **for every heat** (2 heats → 8 lines). The cRTN tracks **one** melting progress, output, waiting progress and input, shared by all heats (4 lines, whatever the production target; 3 heats would need 12 RTN lines). The cRTN panels follow the thesis figure "Scheduling results of cRTN in Scenario 2".

How to read it: the RTN's molten iron-s stays at 0 because a finished heat is transferred at once. The cRTN's resource levels are defined at slot boundaries (dots). Only the melting progress is also drawn inside the slots, reconstructed from how long the furnace runs at each power level.
"""),
    code(r"""
from matplotlib.gridspec import GridSpec

hours = np.arange(T + 1)
fig = plt.figure(figsize=(12, 7.5))
gs = GridSpec(8, 2, figure=fig, hspace=0.15, wspace=0.25)

# RTN: one set of tasks and resources per heat
info = m_rtn._rtn
rtn_rows = []
for h in range(2):
    melt, trans = info["task_proc"](0, h), info["task_proc"](0, h) + 1
    status = [pyo.value(m_rtn.S[h, t]) for t in range(T)]
    rtn_rows += [
        (f"melting task\nheat {h + 1}", status + [status[-1]], "step"),
        (f"transfer task\nheat {h + 1}", [pyo.value(m_rtn.N[trans, t]) for t in range(T)] + [0], "start"),
        (f"molten iron-s\nheat {h + 1}", [pyo.value(m_rtn.R[1 + 2 * h, t]) for t in hours], "step"),
        (f"molten iron-d\nheat {h + 1}", [pyo.value(m_rtn.R[2 + 2 * h, t]) for t in hours], "step"),
    ]
for k, (label, values, kind) in enumerate(rtn_rows):
    ax = fig.add_subplot(gs[k, 0])
    color = "tab:blue" if k < 4 else "tab:orange"
    if kind == "start":
        starts = [t for t, v in enumerate(values) if v > 0.5]
        ax.vlines(starts, 0, 1, color=color, lw=3); ax.plot(starts, [1] * len(starts), "v", color=color)
    else:
        ax.step(hours, values, where="post", color=color, lw=2)
    ax.set_ylim(-0.2, 1.4); ax.set_yticks([]); ax.set_xlim(0, T)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    if k < 7:
        ax.set_xticklabels([])
    if k == 0:
        ax.set_title(f"RTN: one set per heat (8 lines, {fx.count_vars(m_rtn)[0]} binaries)")
ax.set_xlabel("Hour")

# cRTN: one set of tasks and resources, shared by all heats
t_melt, melt = crtn_progress(m_crtn, stage=0)
crtn_rows = [
    ("melting\nprogress", (t_melt, melt), "tab:orange"),  # within-slot reconstruction
    ("molten iron-s\n(output)", (hours, [pyo.value(m_crtn.R[1, t]) for t in hours]), "tab:blue"),
    ("waiting\nprogress", (hours, [pyo.value(m_crtn.R[2, t]) for t in hours]), "tab:red"),
    ("molten iron-d\n(input, heats)", (hours, [pyo.value(m_crtn.R[3, t]) for t in hours]), "tab:green"),
]
for k, (label, (x, y), color) in enumerate(crtn_rows):
    ax = fig.add_subplot(gs[k, 1])  # same row height as the RTN: the column is half as tall
    if k == 0:
        ax.plot(x, y, color=color, lw=2); ax.set_ylim(-0.1, 1.2); ax.set_yticks([0, 1])
    else:  # levels at slot boundaries
        ax.plot(x, y, "o--", color=color, lw=1, ms=6)
        ax.set_ylim(-0.2, 2.4 if k == 3 else 1.2); ax.set_yticks([0, 1, 2] if k == 3 else [0, 1])
    ax.set_xlim(0, T)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    if k < 3:
        ax.set_xticklabels([])
    if k == 0:
        ax.set_title(f"cRTN: shared by all heats (4 lines, {fx.count_vars(m_crtn)[0]} binaries)")
ax.set_xlabel("Hour")
fig.text(0.73, 0.28, "no extra lines needed:\nthe cRTN's size does not\ndepend on the number of heats",
         ha="center", va="center", fontsize=10, color="grey", style="italic")
plt.show()
"""),
    md(r"""
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
**Which heat is where, and when?** A Gantt chart of the same day (3 heats) shows the schedules directly. Both models push the energy-hungry EAF melting into the cheaper hours, but the cRTN can start and finish tasks within a slot, so it packs the heats more tightly. (The cRTN decides how long each task runs within a slot, not its exact start time; the chart places that time at the slot edge next to the neighboring activity.)
"""),
    code(r"""
price = np.repeat(prices.iloc[25].to_numpy(), int(1 / DT))  # July 26
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
for ax, (name, build) in zip(axes, [("RTN", fx.build_rtn), ("cRTN", fx.build_crtn)]):
    m = build(steel, price, 3, dt=DT)
    r = fx.solve(m)
    viz.plot_gantt(heat_schedule(m), [st.name for st in steel.stages], ax=ax, price=price, dt=DT,
                   title=f"{name}: ${r.cost:,.0f}")
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

**Solver note:** the inverse problem is nonconvex (bilinear), so this notebook uses **Gurobi** (`pip install gurobipy` plus a license; academic licenses are free). The size-limited license bundled with `gurobipy` is too small for it, so on Colab you need to add your own license (e.g. a Gurobi WLS license).
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
    p_max=np.array([6.0, 8.0]),             # rated power of machines 1 and 2 (MW)
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
**What the identified model does, machine by machine.** The meter only shows the total load. The identified model also explains *how* that load arises. Machine 1 can process 3 units/h but machine 2 only 2, so machine 1 runs ahead in cheap hours and fills the intermediate buffer, and machine 2 drains it later:
"""),
    code(r"""
from indflex.identify import build_mstn

def machine_schedule(params, price):
    mm = build_mstn(params, price)
    fx.solve(mm)
    P = np.array([[mm.plant.P[t, i]() for i in range(2)] for t in range(24)])
    S = np.array([mm.plant.S[t, 1]() for t in range(25)])  # intermediate buffer
    return P, S

price = test_prices[0]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.3))
for ax, (label, params) in zip(axes[:2], [("true plant", true), ("identified model", theta)]):
    P, _ = machine_schedule(params, price)
    ax.bar(range(24), P[:, 0], label="machine 1", color="tab:blue")
    ax.bar(range(24), P[:, 1], bottom=P[:, 0], label="machine 2", color="tab:orange")
    ax.set(title=label, xlabel="Hour", ylabel="MW"); ax.legend(fontsize=8)
for label, params, style in [("true", true, "k-"), ("identified", theta, "--")]:
    axes[2].plot(machine_schedule(params, price)[1], style, lw=2, label=label)
axes[2].set(title="intermediate buffer level", xlabel="Hour"); axes[2].legend(fontsize=8)
ax2 = axes[2].twinx(); ax2.step(range(24), price * 1000, where="post", color="grey", alpha=0.5); ax2.set_ylabel("$/MWh", color="grey")
plt.tight_layout(); plt.show()
"""),
    md(r"""
The identified model runs the same machines at the same hours, and its buffer follows the same pattern, only shifted: the meter data reveals how much the buffer *changes*, not its absolute level. That shift is the unidentifiable part noted above, and it does not change the load.

**Using the model: a demand-response event.** This is the use case in the thesis (Ch. 3, Fig. "Use cases for load modeling"). Before an event, an aggregator asks how the plant would respond if 06:00–11:00 became expensive. The identified model gives both the *baseline* (no event) and the *response*:
"""),
    code(r"""
base = test_prices[0]
event = base.copy(); event[6:11] = 0.3  # $/kWh during the event
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(6, 11, color="orange", alpha=0.15, label="DR event")
ax.plot(simulate(theta, base[None])[0], "o-", color="tab:red", ms=4, label="identified model: baseline (no event)")
ax.plot(simulate(theta, event[None])[0], "s--", color="tab:blue", ms=4, label="identified model: response to the event")
ax.plot(simulate(true, event[None])[0], "k:", lw=2, label="true plant: response to the event")
ax.set(xlabel="Hour", ylabel="MWh", title="Estimating a plant's demand response before the event")
ax.legend(fontsize=8, loc="upper right"); plt.show()
"""),
    md(r"""
## 4. When is iteration needed?

With many days of data, or a larger plant, two things change:

* **Solving all days at once is too large.** Each day adds its own primal and dual variables and its own bilinear constraint.
* **A single day can mislead.** Some days can be explained by the wrong parameters (for example, swapping which machine is large).

PSI therefore iterates: it solves one day at a time under a short time limit, and after a first pass it also penalizes the distance from θ to the average of recent estimates, until the estimates agree across days (a zeroth-order stochastic method). Set `RUN_ITERATIVE = True` to watch this on the same plant (about 30 s). Each point is the solution of one day's inverse problem, not an initial guess.
"""),
    code(r"""
RUN_ITERATIVE = False

if RUN_ITERATIVE:
    theta_it, hist = train_psi(true.g, prices, meter, n_iter=33, time_limit=1.0)
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
NB04 = [
    md(r"""
# 04 · D3R: compressing a plant model into a few numbers

**Where this sits in the chain:** process model → parameter identification → *dimension reduction* → grid/aggregator optimization.

A grid or planning model would like to include thousands of factories, but each detailed process model (STN/RTN) brings its own binaries, and exact aggregation of many such models is computationally hard. Data-driven dimension reduction (D3R) replaces each detailed model with **reduced constraints** that are simple and linear, chosen so that their **optimal response to prices** mimics the detailed model's.

*References:* thesis Ch. 4; R. Lyu et al., "Data-Driven Dimension Reduction for Industrial Load Modeling Using Inverse Optimization," *IEEE Trans. Smart Grid* 16(3), 2025; paper code: [Data-Driven-Dimension-Reduction](https://github.com/ruikelyu/Data-Driven-Dimension-Reduction).

Everything here runs with the free HiGHS solver, in seconds.
"""),
    code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indflex as fx
from indflex.identify import make_meter_dataset
from indflex.reduce import ALFParams, build_alf, fit_alf, simulate_alf, nrmse_paper

plt.rcParams["figure.dpi"] = 110
"""),
    md(r"""
## 1. The detailed model and its responses

We take the **cement plant STN** from notebook 01 (4 machines with on/off modes, buffers between them) and record its optimal hourly load under PJM prices. We use these days for training and later check on July 22–31. The data is in MWh and $/kWh, as in the paper.
"""),
    code(r"""
plant = fx.load_plant("cement_golmohamadi2020")
ds = make_meter_dataset(plant)
n_bin, n_cont = fx.count_vars(fx.build_stn(plant, ds.price_train[0] * 1000))
print(f"detailed model (STN): {n_bin} binary + {n_cont} continuous variables per day")

fig, ax = plt.subplots(figsize=(9, 3.2))
for d in range(3):
    ax.plot(ds.load_train[d], label=f"July {d + 1}")
ax.set(title="Optimal load of the detailed model", xlabel="Hour", ylabel="MWh"); ax.legend(fontsize=8); plt.show()
"""),
    md(r"""
## 2. The reduced form: an adjustable load fleet (ALF)

The plant is approximated by $I$ adjustable loads in parallel. Load $i$ draws $p_{t,i}$ with

$$\underline P_i \le p_{t,i} \le \overline P_i \quad\text{(every hour)},\qquad \underline E_i \le \textstyle\sum_t p_{t,i}\,\Delta t \le \overline E_i \quad\text{(over the day)},$$

and the plant's load is $\sum_i p_{t,i}$. That is **4 numbers per load**, and the plant's response to a price becomes a tiny LP. The ALF keeps three features of industrial loads: several pieces of equipment, rated powers, and a daily production goal (energy).

## 3. Learning the ALF by inverse optimization

We want ALF parameters whose **optimal** response reproduces the detailed model's. As in notebook 03, we write the ALF's optimality (KKT) conditions as constraints and let the parameters be variables. Here the complementary-slackness conditions are linearized with big-M binaries, so the problem has no bilinear terms:

* squared-error loss → mixed-integer QP (Gurobi, as in the paper);
* absolute-error loss → **MILP, which the free HiGHS solver handles**.

For one adjustable load and three days of data, **one solve** is enough:
"""),
    code(r"""
alf, info = fit_alf(ds.price_train[:3], ds.load_train[:3], n_loads=1, solver="highs")
print(f"{info['status']} in {info['seconds']:.1f} s ({info['binaries']} binaries in the inverse problem)")
alf.as_frame()
"""),
    md(r"""
The learned ALF says: the plant always draws between $\underline P$ and $\overline P$ and must use a fixed amount of energy per day (a production target, so $\underline E \approx \overline E$). Its flexibility is *when* it uses that energy.

## 4. Check on unseen prices (July 22–31)

First, what the learned ALF allows: hourly power anywhere in the band $[\underline P, \overline P]$, with a fixed daily energy. Under each price it draws as little as possible in the expensive hours, as long as it still uses its daily energy. Below, the band and the ALF's response to three unseen days (top), and the detailed model's response to the same days (bottom):
"""),
    code(r"""
pred = simulate_alf(alf, ds.price_cv)
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for ax, (label, loads) in zip(axes, [("1-load ALF", pred), ("detailed STN", ds.load_cv)]):
    ax.axhspan(alf.p_min.sum(), alf.p_max.sum(), color="tab:green", alpha=0.12, label="ALF power band")
    for d, c in zip([0, 4, 8], ["C0", "C1", "C3"]):
        ax.plot(loads[d], color=c, lw=2, label=f"July {22 + d}")
    ax.set(ylabel="MWh", title=label, ylim=(0, alf.p_max.sum() * 1.15))
axes[0].legend(fontsize=8, ncol=4, loc="lower left"); axes[1].set_xlabel("Hour")
plt.tight_layout(); plt.show()
"""),
    md(r"""
How close is the ALF's response to the detailed model on each held-out day?
"""),
    code(r"""
pred = simulate_alf(alf, ds.price_cv)
print(f"nRMSE vs. the detailed model: {100 * nrmse_paper(pred, ds.load_cv):.1f}%   (paper, 1 load: 8.9%)")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, d in zip(axes, [0, 4, 8]):
    ax.plot(ds.load_cv[d], "k", lw=2.5, label="detailed STN")
    ax.plot(pred[d], "--", color="tab:green", lw=2, label="1-load ALF")
    ax.set(title=f"July {22 + d}", xlabel="Hour")
axes[0].set_ylabel("MWh"); axes[0].legend(fontsize=8); plt.tight_layout(); plt.show()
"""),
    md(r"""
The errors are errors in the **optimal response** under new prices. The ALF does not guarantee that its schedules are feasible for the detailed plant; in practice, the detailed model (or the plant itself) turns the reduced model's decision into an executable schedule.

**More loads, better fit.** A second adjustable load lets the ALF represent equipment with different flexibility. Set `RUN_TWO_LOADS = True` to fit it: about a minute with HiGHS, or about 20 s with `solver="gurobi"`. In our runs the error fell to about 3%.
"""),
    code(r"""
RUN_TWO_LOADS = False

if RUN_TWO_LOADS:
    alf2, info2 = fit_alf(ds.price_train[:3], ds.load_train[:3], n_loads=2, solver="highs", time_limit=120)
    print(f"{info2['status']} in {info2['seconds']:.0f} s, nRMSE {100 * nrmse_paper(simulate_alf(alf2, ds.price_cv), ds.load_cv):.1f}%")
    display(alf2.as_frame())
"""),
    md(r"""
## 5. Why this matters for the grid

The reduced model is an LP with $4I$ parameters, so many plants are cheap to handle together. Compare the size of the two models for one plant-day:
"""),
    code(r"""
alf_bin, alf_cont = fx.count_vars(build_alf(alf, ds.price_cv[0]))
pd.DataFrame({"binary variables": [n_bin, alf_bin], "continuous variables": [n_cont, alf_cont]},
             index=["detailed STN", f"ALF, {alf.n_loads} load"])
"""),
    md(r"""
For the steelmaking line of notebook 02, the paper replaces **10,208 integer variables with 24 continuous ones** at 3.6% error.

**When is iteration needed?** One solve becomes too large with many days or many loads, since each day adds its own binaries. The paper's code then solves small random batches of days and moves the parameters a small step toward each batch's solution (`indflex.reduce.train_d3r`).

**Takeaways**
* D3R learns reduced constraints from the detailed model's *optimal responses*, using the same inverse-optimization idea as PSI.
* An adjustable load fleet captures the main features of industrial flexibility with a handful of numbers, and it fits in any LP.
* Next in the chain: operate many flexible resources together → **notebook 05: virtual power plant**.
"""),
]

NB05 = [
    md(r"""
# 05 · Virtual power plant: bidding for regulation and splitting the signal in microseconds

**Where this sits in the chain:** process model → parameter identification → dimension reduction → *grid/aggregator optimization*.

A virtual power plant (VPP) pools many small, different resources (EVs, storage, air conditioning, industrial production stages, PV) and sells **energy** and **frequency regulation** in the market. It decides on two time scales:

1. **Bidding (every 30 min).** An LP over the rest of the day chooses hourly energy and regulation bids. The regulation signal is uncertain, so the LP plans every resource's power under signal scenarios $d_s \in [-1, 1]$ with their probabilities.
2. **Disaggregation (every 2 s).** The grid operator sends a signal $\delta$, and the VPP must deliver $P^{\rm req} = P^{\rm bid} + R^{\rm bid}\,\delta$ immediately, split among its resources. Re-solving an optimization every 2 s is too slow. The trick: the bidding LP's **dual variables** $\lambda_i$ (the value of the energy stored in resource $i$) give every resource a constant marginal cost. Sorting resources by that cost gives a **merit order**, and filling it up to $P^{\rm req}$ solves the split exactly.

*Reference:* Q. Chen, R. Lyu, H. Guo, X. Su, "Real-Time Operation Strategy of Virtual Power Plants With Optimal Power Disaggregation Among Heterogeneous Resources," *Applied Energy* 361 (2024) 122876; thesis Ch. 5; paper code: [Real-time-operation-strategy-of-virtual-power-plants](https://github.com/ruikelyu/Real-time-operation-strategy-of-virtual-power-plants).

Everything here runs with the free HiGHS solver.
"""),
    code(r"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indflex.coordinate import load_vpp_case, bid, merit_order, allocate, allocate_lp, simulate_hour

plt.rcParams["figure.dpi"] = 110
KINDS = ["EV", "ES", "TCL", "IPP", "PV"]
COLORS = {"EV": "tab:blue", "ES": "tab:purple", "TCL": "tab:green", "IPP": "tab:orange", "PV": "gold"}
LABELS = {"EV": "EVs", "ES": "storage", "TCL": "air conditioning", "IPP": "industrial stages", "PV": "PV"}
"""),
    md(r"""
## 1. The resources

One day of the paper's case (PJM prices and RegD signal). Every resource uses the same *standardized model*: discharge and charge power limits, an energy state with self-loss and efficiencies, external drift (heat gains for air conditioning), and usage costs. For industrial stages, the "energy" is the material in the buffer after the stage: running stage $i$ fills its buffer and drains the one before it.
"""),
    code(r"""
case = load_vpp_case("day21")
kind = np.array(case.kind)
print(case.description)
display(pd.Series(kind).value_counts().rename("count").to_frame())

hours = np.arange(case.T)
fig, ax = plt.subplots(figsize=(10, 3.6))
up, down = np.zeros(case.T), np.zeros(case.T)
for k in KINDS:
    dis = case.p_dis_upper[kind == k].sum(axis=0)
    ch = case.p_ch_upper[kind == k].sum(axis=0)
    ax.bar(hours, dis, bottom=up, color=COLORS[k], label=LABELS[k]); up += dis
    ax.bar(hours, -ch, bottom=-down, color=COLORS[k]); down += ch
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="Hour", ylabel="MW", title="Maximum discharge (up) and charge (down) power by resource type")
ax.set_ylim(-down.max() * 1.1, up.max() * 1.35)
ax.legend(fontsize=8, ncol=5); plt.show()
"""),
    md(r"""
## 2. Bidding: one LP for the day

The LP maximizes energy revenue plus regulation capacity and mileage revenue, minus usage costs, over 22 signal scenarios. It keeps every resource within its limits in every scenario, and keeps enough stored energy to sustain the full signal for 30 minutes.
"""),
    code(r"""
day_ahead = bid(case)
print(f"solved in {day_ahead.seconds:.1f} s, expected profit ${day_ahead.profit:,.0f}")

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.bar(hours - 0.2, day_ahead.bid_p, width=0.4, color="tab:gray", label="energy bid (MW, + = sell)")
ax.bar(hours + 0.2, day_ahead.bid_r, width=0.4, color="tab:red", label="regulation capacity bid (MW)")
ax.axhline(0, color="black", lw=0.8); ax.set(xlabel="Hour", ylabel="MW", ylim=(-7, 9)); ax.legend(fontsize=8, loc="upper left")
ax2 = ax.twinx()
ax2.plot(hours, case.price_e, "k--", lw=1.5, label="energy price")
ax2.plot(hours, case.price_reg[:, 0], ":", color="tab:red", lw=2, label="regulation capacity price")
ax2.set_ylabel("$/MWh"); ax2.legend(fontsize=8, loc="upper right")
plt.title("Hourly bids"); plt.show()
"""),
    md(r"""
The VPP buys energy (negative bids) mainly to charge EVs, and sells regulation capacity in every hour.

## 3. From duals to a merit order

At the start of each half hour the VPP re-solves the bidding LP from its current state. The dual of "energy now = measured energy" is $\lambda_i = \partial\,\text{profit}/\partial E_i$, the value of one more unit stored in resource $i$. With it, every resource becomes two **power segments** with constant marginal costs:

* discharging: usage cost $+$ the value of the stored energy it uses;
* charging (negative power): $-$usage cost $+$ the value of the energy it stores.

Sorted by cost, the segments form a supply curve. Following the signal means moving along this curve. In hour 1 (midnight) no EVs are connected yet, so air conditioning, storage and the industrial stages provide the flexibility.
"""),
    code(r"""
rebid = bid(case, cur_slot=0, E_cur=case.energy_init, bid_cur=(day_ahead.bid_p[0], day_ahead.bid_r[0]), rest=1.0)
seg = merit_order(case, rebid.lam, case.energy_init, slot=0, p_none_reg=rebid.p_none_reg)

base = seg.p_lo.sum()  # every segment at its lower bound
x = base + np.r_[0, np.cumsum(seg.p_hi - seg.p_lo)]
fig, ax = plt.subplots(figsize=(10, 4))
for k in range(len(seg.cost)):
    if seg.p_hi[k] - seg.p_lo[k] > 1e-9:
        ax.fill_between([x[k], x[k + 1]], 0, seg.cost[k], step="pre", color=COLORS[kind[seg.der[k]]], alpha=0.85, lw=0)
for k in KINDS:  # legend only for types with flexible power this hour
    if any((kind[seg.der] == k) & (seg.p_hi - seg.p_lo > 1e-9)):
        ax.fill_between([], [], color=COLORS[k], label=LABELS[k])
bp, br = day_ahead.bid_p[0], day_ahead.bid_r[0]
ax.axvspan(bp - br, bp + br, color="grey", alpha=0.15, label="range of $P^{req}$ this hour")
ax.axvline(bp, color="black", ls="--", lw=1, label="energy bid ($\\delta = 0$)")
ax.set(xlabel="VPP output (MW, + = injection)", ylabel="marginal cost ($/MWh)",
       title="Merit order of the VPP's resources (hour 1)")
ax.legend(fontsize=8, loc="upper left"); plt.show()
"""),
    md(r"""
## 4. Following the regulation signal for one hour

Every 2 s: $P^{\rm req} = P^{\rm bid} + R^{\rm bid}\delta$, filled along the merit order. Every minute the merit order is rebuilt from the updated energy states. (The paper also re-bids every 30 minutes; for clarity this hour uses one re-bid.)
"""),
    code(r"""
hour = simulate_hour(case, 0, bp, br, rebid.lam, case.energy_init, rebid.p_none_reg)
net = hour.p_dis - hour.p_ch
t_min = np.arange(len(hour.p_req)) * 2 / 60

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t_min, hour.p_req, color="black", lw=1.2, label="requested $P^{req}$")
axes[0].plot(t_min, net.sum(axis=0), "--", color="tab:red", lw=1, label="delivered (sum of resources)")
axes[0].set(ylabel="MW", title=f"Signal following, {1e6 * hour.alloc_seconds:.0f} µs per allocation"); axes[0].legend(fontsize=8)
for k in KINDS:
    axes[1].plot(t_min, net[kind == k].sum(axis=0), color=COLORS[k], lw=1.2, label=LABELS[k])
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set(xlabel="Minute", ylabel="MW", title="Who follows the signal"); axes[1].legend(fontsize=8, ncol=5)
plt.tight_layout(); plt.show()
"""),
    md(r"""
**Why not just solve an LP every 2 s?** The merit-order fill *is* the solution of the disaggregation LP (minimize the total marginal cost subject to delivering $P^{\rm req}$), computed without a solver:
"""),
    code(r"""
rows = []
for delta in case.signal[:20]:
    req = bp + br * delta
    t0 = time.perf_counter(); fast = allocate(seg, req); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); lp = allocate_lp(seg, req); t_lp = time.perf_counter() - t0
    rows.append({"cost gap": seg.cost @ fast - seg.cost @ lp, "merit order (ms)": 1e3 * t_fast, "LP (ms)": 1e3 * t_lp})
pd.DataFrame(rows).describe().loc[["mean", "max"]].round(4)
"""),
    md(r"""
**A note on the paper code.** For coupled resources (industrial stages), the energy update is $E' = \eta\,p$, so the value of running resource $j$ is $(\eta^\top\lambda)_j$. The paper code uses $(\eta\lambda)_j$. The toolkit uses $\eta^\top\lambda$ by default, and `merit_order(..., eta_convention="paper")` reproduces the paper code exactly (see `tests/test_vpp.py`). The two differ only for the industrial stages; in our tests the transposed version is never more expensive.

**Takeaways**
* A VPP turns heterogeneous resources into one market participant through a standardized resource model and a scenario-based bidding LP.
* The bidding LP's duals price each resource's stored energy, which turns real-time disaggregation into a merit order: optimal, and microseconds instead of milliseconds.
* In the chain, reduced industrial models (notebook 04) are exactly the kind of resource such a VPP, or a grid model, can carry.
"""),
]


REPO = "ruikelyu/indflex"


def colab_cells(fname, extras="dev"):
    """An 'Open in Colab' badge and an install cell that only runs on Colab."""
    badge = md(f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
               f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{fname})")
    install = code(f"""# On Google Colab, install indflex first (locally: pip install -e ".[{extras}]")
import sys
if "google.colab" in sys.modules:
    !pip install -q "indflex[{extras}] @ git+https://github.com/{REPO}"
""")
    return [badge, install]


def main(only=None):
    """Write the notebooks (without outputs). ``only``: list of name prefixes, e.g. ["03"]."""
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    books = [("01_stn_vs_lstn.ipynb", NB01, "dev"), ("02_rtn_vs_crtn.ipynb", NB02, "dev"),
             ("03_psi.ipynb", NB03, "dev,gurobi"), ("04_d3r.ipynb", NB04, "dev"), ("05_vpp.ipynb", NB05, "dev")]
    for fname, cells, extras in books:
        if only and not any(fname.startswith(o) for o in only):
            continue
        cells = [cells[0], *colab_cells(fname, extras), *cells[1:]]  # title, then Colab, then the rest
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, out / fname)
        print(f"wrote notebooks/{fname}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
