"""Explainer video: D3R, data-driven dimension reduction for industrial load modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import *  # noqa: E402,F403

from export_web_demos import d3r_demo  # noqa: E402  (same computation as the web demo)


def video() -> Video:
    R = d3r_demo()
    A = R["alf"]
    hours = np.arange(24)
    day = max(R["days"], key=lambda d: np.ptp(d["stn"]))  # a day with a visible shift

    def problem_scene(fig, t):
        heading(fig, "Thousands of plants, each with its own binaries", "exact aggregation of detailed models is computationally hard")
        n = int(round(stage(t, 0.0, 0.7) * 120))
        for i in range(n):
            x, y = 0.08 + (i % 20) * 0.042, 0.72 - (i // 20) * 0.075
            fig.patches.append(plt.Rectangle((x, y), 0.032, 0.05, transform=fig.transFigure, color=BLUE, alpha=0.75))
        a = stage(t, 0.6, 0.85)
        fig.text(0.08, 0.2, f"each plant: {R['stn_binaries']} binary variables per day (cement STN)", fontsize=30, color=INK, alpha=a)
        fig.text(0.08, 0.14, "grid models need something much smaller", fontsize=28, color=RED, alpha=stage(t, 0.75, 0.95))
        footer(fig)

    def alf_scene(fig, t):
        heading(fig, "An adjustable load fleet: four numbers per load", "power band [P_min, P_max] and daily energy [E_min, E_max]")
        ax = fig.add_axes([0.08, 0.2, 0.6, 0.56])
        ax.axhspan(A["p_min"], A["p_max"], color=GREEN, alpha=0.12)
        ax.axhline(A["p_min"], color=GREEN, lw=2); ax.axhline(A["p_max"], color=GREEN, lw=2)
        a = stage(t, 0.15, 0.8)
        k = max(1, int(round(a * 24)))
        alf = np.array(day["alf"])
        ax.fill_between(hours[:k], 0, alf[:k], step="mid", color=GREEN, alpha=0.25)
        ax.step(hours[:k], alf[:k], where="mid", color=GREEN, lw=3.5)
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, A["p_max"] * 1.3); ax.set_xlabel("Hour"); ax.set_ylabel("MWh")
        ax.text(0.3, A["p_max"] + 1, f"P_max = {A['p_max']:g} MW", color=GREEN, fontsize=22)
        ax.text(0.3, A["p_min"] - 3, f"P_min = {A['p_min']:g} MW", color=GREEN, fontsize=22)
        b = stage(t, 0.6, 0.85)
        fig.text(0.72, 0.66, "daily energy budget", fontsize=26, color=MUTED, alpha=b)
        fig.text(0.72, 0.6, f"{A['e_min']:g} MWh", fontsize=40, color=INK, fontweight="demibold", alpha=b)
        fig.text(0.72, 0.5, "flexibility = when to\nuse that energy", fontsize=26, color=INK, alpha=b)
        footer(fig)

    def train_scene(fig, t):
        heading(fig, "Learned by inverse optimization", "match the detailed model's optimal response, not its constraints")
        steps = ["many prices", "detailed model\n(STN, binaries)", "optimal\nresponses", "inverse\noptimization", "fleet:\n4 numbers"]
        for i, label in enumerate(steps):
            a = stage(t, 0.05 + 0.12 * i, 0.15 + 0.12 * i)
            x = 0.06 + i * 0.19
            fig.patches.append(plt.Rectangle((x, 0.5), 0.15, 0.16, transform=fig.transFigure, fc="#eef3fb", ec=BLUE, lw=2,
                                             alpha=a))
            fig.text(x + 0.075, 0.58, label, fontsize=24, ha="center", va="center", color=INK, alpha=a)
            if i < len(steps) - 1:
                fig.text(x + 0.168, 0.58, "→", fontsize=34, color=BLUE, va="center", alpha=a)
        a = stage(t, 0.7, 0.9)
        fig.text(0.06, 0.32, "KKT conditions with big-M binaries: no products of unknowns", fontsize=28, color=INK, alpha=a)
        fig.text(0.06, 0.26, "absolute-error loss  →  a MILP that the free HiGHS solver handles", fontsize=28, color=GREEN,
                 alpha=stage(t, 0.8, 1.0))
        footer(fig)

    def result_scene(fig, t):
        heading(fig, f"Cement plant: fitted on 3 days, checked on 10 unseen days",
                f"July {day['day']}, 2022 · error on all unseen days: {R['overall_nrmse']}%")
        ax = fig.add_axes([0.08, 0.2, 0.84, 0.56])
        ax.axhspan(A["p_min"], A["p_max"], color=GREEN, alpha=0.1)
        k = max(1, int(round(stage(t, 0.1, 0.75) * 24)))
        ax.step(hours[:k], np.array(day["stn"])[:k], where="mid", color=INK, lw=3, label="detailed STN")
        ax.step(hours[:k], np.array(day["alf"])[:k], where="mid", color=GREEN, lw=3.5, ls="--", label="reduced fleet")
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, A["p_max"] * 1.3); ax.set_xlabel("Hour"); ax.set_ylabel("MWh")
        ax.legend(loc="upper right", ncol=2)
        ax2 = ax.twinx(); ax2.plot(hours, day["price"], ":", color=GRAY, lw=2); ax2.grid(False)
        ax2.set_ylabel("price ($/MWh)", color=GRAY); ax2.spines["right"].set_visible(True)
        footer(fig)

    def paper_scene(fig, t):
        heading(fig, "In the paper: a steelmaking plant", "IEEE Transactions on Smart Grid 16(3), 2025")
        a = stage(t, 0.05, 0.3)
        fig.text(0.12, 0.62, "10,208", fontsize=90, color=RED, fontweight="demibold", alpha=a)
        fig.text(0.12, 0.54, "integer variables", fontsize=30, color=INK, alpha=a)
        fig.text(0.46, 0.62, "→", fontsize=90, color=MUTED, alpha=stage(t, 0.2, 0.35))
        b = stage(t, 0.25, 0.45)
        fig.text(0.6, 0.62, "24", fontsize=90, color=GREEN, fontweight="demibold", alpha=b)
        fig.text(0.6, 0.54, "continuous variables", fontsize=30, color=INK, alpha=b)
        fig.text(0.12, 0.40, "3.6% error in the optimal response on test days", fontsize=32, color=INK,
                 alpha=stage(t, 0.4, 0.6))
        fig.text(0.12, 0.30, "not a feasibility guarantee: the detailed model turns decisions into executable schedules",
                 fontsize=24, color=MUTED, alpha=stage(t, 0.65, 0.85))
        footer(fig)

    return Video("04_d3r", "D3R: data-driven dimension reduction for industrial load modeling", [
        Scene([("D3R: data-driven dimension reduction for industrial load modeling, using inverse optimization.",
                "D three R: data driven dimension reduction for industrial load modeling, using inverse optimization."),
               "Here is the idea behind the paper, in two minutes."],
              title_card("IEEE Transactions on Smart Grid, 2025", "D3R: a few numbers per factory",
                         "R. Lyu, H. Guo, G. Strbac, C. Kang")),
        Scene(["A power system can have thousands of factories.",
               "Each detailed process model brings its own binary variables, and combining many of them exactly is computationally hard.",
               "Grid models need something much smaller."], problem_scene),
        Scene([("D3R replaces each detailed model with an adjustable load fleet.", "D three R replaces each detailed model with an adjustable load fleet."),
               "A few loads, each with a power band and a daily energy budget: four numbers per load.",
               "The goal is not to copy the plant's constraints, but to respond to prices the same way."], alf_scene),
        Scene([("The fleet is learned by inverse optimization, like PSI.", "The fleet is learned by inverse optimization, like P S I."),
               "We solve the detailed model under many prices, then find the fleet whose optimal response matches those results.",
               "There are no products of unknowns here, so with an absolute-error loss it is a mixed-integer linear program that free solvers can handle."],
              train_scene),
        Scene([f"For a cement plant, a one-load fleet was fitted to three days of responses in {R['fit_seconds']:g} seconds.",
               f"On ten unseen days, it reproduces the detailed model's optimal load with {R['overall_nrmse']} percent error."],
              result_scene),
        Scene([("In the paper, for a steelmaking plant, D3R replaces 10,208 integer variables with 24 continuous ones, at 3.6 percent error.",
                "In the paper, for a steelmaking plant, D three R replaces 10,208 integer variables with 24 continuous ones, at 3.6 percent error."),
               "These are errors in the optimal response to prices, not feasibility guarantees: the detailed model still turns decisions into executable schedules."],
              paper_scene),
        Scene(["The code and an interactive notebook are open source in indflex.",
               "Next: operating many flexible resources together in a virtual power plant."],
              outro_card(["Notebook 04: D3R (free solver; runs on Google Colab)",
                          "Interactive demos: ruikelyu.github.io",
                          "Paper: IEEE Transactions on Smart Grid 16(3), 2025"]), static=True),
    ])


if __name__ == "__main__":
    main(video)
