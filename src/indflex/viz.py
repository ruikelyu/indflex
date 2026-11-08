"""Plots shared by the notebooks (and video figures)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .solve import Result


def plot_schedule(result: Result, price: Sequence[float], title: str = "", ax=None):
    """Stacked hourly energy by stage, with the price on the right axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    hours = np.arange(len(price))
    if result.energy is not None:
        bottom = np.zeros(len(hours))
        cmap = plt.get_cmap("tab20")
        for k, (name, row) in enumerate(result.energy.iterrows()):
            ax.bar(hours, row.to_numpy(), bottom=bottom, width=0.8, label=name, color=cmap(k % 20))
            bottom += row.to_numpy()
        ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8, frameon=False)
    else:
        ax.bar(hours, result.load, width=0.8, color="tab:blue", label="total load")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Energy (kWh)")
    ax2 = ax.twinx()
    ax2.plot(hours, price, color="black", lw=2, ls="--", label="price")
    ax2.set_ylabel("Price ($/MWh)")
    ax.set_title(title or f"Cost ${result.cost:,.0f}")
    ax.set_xticks(hours)
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


def plot_scaling(rows, time_limit: float | None = None, ax=None):
    """Solve time vs. number of factories. ``rows``: dicts with model, n, seconds, status."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    styles = {"STN": dict(color="tab:red", marker="s"), "LSTN": dict(color="tab:blue", marker="o")}
    for model in ["STN", "LSTN"]:
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
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of factories")
    ax.set_ylabel("Solve time (s)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax
