"""Plots shared by the notebooks (and video figures)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .solve import Result


def plot_schedule(result: Result, price: Sequence[float], title: str = "", ax=None, dt: float = 1.0):
    """Stacked energy per slot by stage, with the price on the right axis.

    ``dt`` is the slot length in hours; the x axis is labelled in hours.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    hours = np.arange(len(price)) * dt
    if result.energy is not None:
        bottom = np.zeros(len(hours))
        cmap = plt.get_cmap("tab20")
        for k, (name, row) in enumerate(result.energy.iterrows()):
            ax.bar(hours, row.to_numpy(), bottom=bottom, width=0.8 * dt, align="edge", label=name, color=cmap(k % 20))
            bottom += row.to_numpy()
        ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8, frameon=False)
    else:
        ax.bar(hours, result.load, width=0.8 * dt, align="edge", color="tab:blue", label="total load")
    ax.set_xlabel("Hour")
    ax.set_ylabel(f"Energy per slot ({result.energy_unit})")
    ax2 = ax.twinx()
    ax2.step(np.append(hours, hours[-1] + dt), np.append(price, price[-1]), where="post",
             color="black", lw=1.8, ls="--", label="price")
    ax2.set_ylabel("Price ($/MWh)")
    ax.set_title(title or f"Cost ${result.cost:,.0f}")
    span = len(price) * dt
    ax.set_xticks(np.arange(0, span + 1e-9, 1 if span <= 24 else 2))
    ax.set_xlim(0, span)
    return ax


def plot_storage(result: Result, ax=None):
    """Buffer level of each stage over the horizon, as a fraction of its capacity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    for name, row in result.storage.iterrows():
        ax.plot(row.index, row.to_numpy() / max(row.max(), 1e-9), label=name, lw=1.5)
    ax.set_xlabel("Hour (end of interval)")
    ax.set_ylabel("Level / max level")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    return ax


def plot_scaling(rows, time_limit: float | None = None, ax=None, xlabel: str = "Number of factories",
                 log_x: bool = True):
    """Solve time vs. problem size. ``rows``: dicts with model, n, seconds, status."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    styles = {"STN": dict(color="tab:red", marker="s"), "LSTN": dict(color="tab:blue", marker="o"),
              "RTN": dict(color="tab:red", marker="s"), "cRTN": dict(color="tab:blue", marker="o")}
    for model in dict.fromkeys(r["model"] for r in rows):
        pts = [r for r in rows if r["model"] == model]
        if not pts:
            continue
        ax.plot([r["n"] for r in pts], [r["seconds"] for r in pts], label=model, **styles[model])
        for r in pts:
            if r["status"] != "optimal":
                ax.annotate("time limit", (r["n"], r["seconds"]), textcoords="offset points",
                            xytext=(4, 6), fontsize=8, color=styles[model]["color"])
    if time_limit:
        ax.axhline(time_limit, color="grey", ls=":", lw=1)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Solve time (s)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_gantt(schedule, stages, ax=None, price=None, dt: float = 1.0, title: str = "", horizon: float | None = None):
    """Gantt chart of heats (bars) through serial stages; ``schedule`` from ``models.heat_schedule``.

    Optionally shades the price profile behind the bars.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.6 * len(stages) + 1.2))
    cmap = plt.get_cmap("tab10")
    y = {s: k for k, s in enumerate(reversed(list(stages)))}
    for r in schedule.itertuples():
        ax.barh(y[r.stage], r.end - r.start, left=r.start, height=0.6, color=cmap((r.heat - 1) % 10),
                edgecolor="black", lw=0.6)
        if r.end - r.start > 0.35:
            ax.text((r.start + r.end) / 2, y[r.stage], str(r.heat), ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
    ax.set_yticks(list(y.values()), list(y.keys()))
    if price is not None:
        price = np.asarray(price)
        ax2 = ax.twinx()
        t = np.arange(len(price) + 1) * dt
        ax2.fill_between(t, np.append(price, price[-1]), step="post", color="grey", alpha=0.15, lw=0)
        ax2.set_ylabel("Price ($/MWh)", color="grey")
        ax2.set_ylim(0, price.max() * 1.1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        horizon = horizon or len(price) * dt
    if horizon:
        ax.set_xlim(0, horizon)
    ax.set_xlabel("Hour")
    ax.set_title(title)
    return ax
