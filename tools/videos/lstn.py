"""Explainer video: LSTN, a linear model of industrial production for demand response."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: E402,F403

import pyomo.environ as pyo  # noqa: E402

import indflex as fx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def data():
    plant = fx.load_plant("steel_powder_lu2021")
    price = fx.daily_price("2022-07", 23)
    m = fx.build_stn(plant, price)
    stn = fx.solve(m)
    # operating mode of each machine in each hour (index of the mode with x = 1)
    modes = np.array([[max(range(len(s.modes)), key=lambda p: pyo.value(m.plant.x[n, p, t])) for t in range(24)]
                      for n, s in enumerate(plant.stages)])
    levels = np.array([[s.modes[p].e for p in row] for s, row in zip(plant.stages, modes)])
    lstn = fx.solve(fx.build_lstn(plant, price))
    # scaling: measured now, so the narrated numbers match the chart (as in notebook 01)
    fleet = fx.load_fleet(200)
    p5 = fx.daily_price("2022-08", 5)
    scaling = {"STN": [], "LSTN": []}
    for model, sizes, relax in [("STN", [1, 2, 3, 5], False), ("LSTN", [1, 2, 5, 10, 20, 50, 100, 200], True)]:
        for n in sizes:
            r = fx.solve(fx.build_fleet(fleet[:n], p5, relax=relax), time_limit=30)
            scaling[model].append((n, r.solve_seconds, r.status))
    return plant, price, stn, lstn, levels, scaling


def video() -> Video:
    plant, price, stn, lstn, levels, scaling = data()
    names = [s.name for s in plant.stages]
    stn_limit = next((n for n, _, st in scaling["STN"] if st != "optimal"), None)
    lstn200 = scaling["LSTN"][-1][1]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    hours = np.arange(24)

    # --- scene: the plant
    def plant_scene(fig, t):
        heading(fig, "A factory is a chain of machines and buffers", "Steel powder plant (Lu et al., 2021): 10 stages in series")
        ax = fig.add_axes([0.05, 0.38, 0.9, 0.4])
        ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
        show = names[:5]
        phase = np.sin(2 * np.pi * t)  # buffers fill (cheap) then drain (expensive)
        for i, name in enumerate(show):
            x = 0.3 + i * 1.95
            ax.add_patch(plt.Rectangle((x, 1.6), 1.1, 0.9, color=BLUE, alpha=0.85))
            ax.text(x + 0.55, 2.05, name, ha="center", va="center", color="white", fontsize=17)
            if i < len(show) - 1:
                level = 0.5 + 0.4 * phase * (1 if i % 2 == 0 else -1)
                bx = x + 1.25
                ax.add_patch(plt.Rectangle((bx, 0.3), 0.55, 1.0, fill=False, ec=MUTED, lw=2))
                ax.add_patch(plt.Rectangle((bx, 0.3), 0.55, level, color=ORANGE, alpha=0.7))
                ax.annotate("", xy=(bx + 0.62, 2.05), xytext=(x + 1.12, 2.05),
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
        ax.text(9.85, 2.05, "…", fontsize=30, color=MUTED, va="center")
        ax.text(0.3, 0.0, "buffers (orange) let upstream machines run ahead while power is cheap",
                fontsize=22, color=MUTED)
        footer(fig)

    # --- scene: STN modes grid
    def stn_scene(fig, t):
        heading(fig, "STN: one operating mode per machine per hour", "binary decision: off / partial / full")
        ax = fig.add_axes([0.16, 0.22, 0.62, 0.56])
        shown = int(round(stage(t, 0.05, 0.8) * 24))
        grid = np.where(hours[None, :] < shown, levels, np.nan)
        ax.imshow(grid, aspect="auto", cmap="Blues", vmin=-10, vmax=80)
        ax.set_yticks(range(len(names)), names, fontsize=17); ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour"); ax.grid(False)
        n_bin = sum(len(s.modes) for s in plant.stages) * 24
        fig.text(0.81, 0.62, f"{min(shown, 24) * n_bin // 24:,}", fontsize=64, color=RED, fontweight="demibold")
        fig.text(0.81, 0.55, "binary variables", fontsize=24, color=MUTED)
        fig.text(0.81, 0.40, "darker = more power", fontsize=20, color=MUTED)
        footer(fig)

    # --- scene: the key idea
    def idea_scene(fig, t):
        heading(fig, "LSTN: how long in each mode, not which mode", "time-in-state variables make the problem linear")
        a = stage(t, 0.35, 0.75)
        for row, (label, y) in enumerate([("STN", 0.62), ("LSTN", 0.40)]):
            fig.text(0.08, y + 0.04, label, fontsize=34, fontweight="demibold", color=INK, va="center")
            ax = fig.add_axes([0.2, y, 0.62, 0.09]); ax.set_axis_off(); ax.set_xlim(0, 60); ax.set_ylim(0, 1)
            if label == "STN":
                ax.add_patch(plt.Rectangle((0, 0), 60, 1, color=BLUE)); ax.text(30, 0.5, "full power, all hour", color="white",
                                                                             ha="center", va="center", fontsize=24)
                fig.text(0.85, y + 0.045, "x ∈ {0, 1}", fontsize=32, color=RED, va="center")
            else:
                split = 60 - 20 * a
                ax.add_patch(plt.Rectangle((0, 0), split, 1, color=BLUE))
                ax.add_patch(plt.Rectangle((split, 0), 60 - split, 1, color=LIGHT))
                ax.text(split / 2, 0.5, f"full: {split:.0f} min", color="white", ha="center", va="center", fontsize=24)
                if a > 0.3:
                    ax.text(split + (60 - split) / 2, 0.5, f"off: {60 - split:.0f} min", color=INK, ha="center",
                            va="center", fontsize=22)
                fig.text(0.85, y + 0.045, "x ∈ [0, 1]", fontsize=32, color=GREEN, va="center", alpha=max(a, 0.2))
        fig.text(0.2, 0.3, "one market hour", fontsize=20, color=MUTED)
        fig.text(0.2, 0.24, "Σ (time in each mode) = 1 hour   →   an LP instead of a MILP", fontsize=28, color=INK,
                 alpha=stage(t, 0.7, 0.9))
        footer(fig)

    # --- scene: schedules and costs
    def result_scene(fig, t):
        heading(fig, "Same schedule, almost the same cost", "July 23, 2022 · PJM real-time prices")
        ax = fig.add_axes([0.08, 0.2, 0.78, 0.58])
        k = max(1, int(round(stage(t, 0.0, 0.6) * 24)))
        ax.bar(hours[:k], stn.load[:k], color=BLUE, alpha=0.35, label="STN (MILP)")
        ax.step(hours[:k], lstn.load[:k], where="mid", color=ORANGE, lw=3.5, label="LSTN (LP)")
        ax.set_xlim(-0.6, 23.6); ax.set_ylim(0, max(stn.load) * 1.15); ax.set_xlabel("Hour"); ax.set_ylabel("Energy (kWh)")
        ax2 = ax.twinx(); ax2.plot(hours[:k], price[:k], "--", color=GRAY, lw=2.5, label="price")
        ax2.set_ylim(0, max(price) * 1.2); ax2.set_ylabel("$/MWh", color=GRAY); ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.legend(loc="upper left", ncol=2)
        a = stage(t, 0.55, 0.75)
        fig.text(0.62, 0.73, f"STN  ${stn.cost:,.2f}", fontsize=30, color=BLUE, alpha=a)
        fig.text(0.62, 0.68, f"LSTN ${lstn.cost:,.2f}", fontsize=30, color=ORANGE, alpha=a)
        footer(fig)

    # --- scene: scaling
    def scaling_scene(fig, t):
        heading(fig, "The payoff: scale", "solve time vs. number of factories (free HiGHS solver)")
        ax = fig.add_axes([0.1, 0.2, 0.55, 0.58])
        for model, color, mk in [("STN", RED, "s"), ("LSTN", BLUE, "o")]:
            pts = scaling[model]
            k = max(1, int(round(stage(t, 0.0, 0.35) * len(pts))))
            ax.plot([p[0] for p in pts[:k]], [p[1] for p in pts[:k]], marker=mk, ms=11, lw=3, color=color, label=model)
        ax.axhline(30, ls=":", color=GRAY, lw=2); ax.text(1.1, 34, "30 s limit", color=GRAY, fontsize=20)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, 260); ax.set_ylim(0.02, 80)
        plain_log_axes(ax, xticks=[1, 2, 5, 10, 20, 50, 100, 200], yticks=[0.1, 1, 10, 30])
        ax.set_xlabel("Number of factories"); ax.set_ylabel("Solve time (s)"); ax.legend(loc="lower right")
        a = stage(t, 0.3, 0.45)
        fig.text(0.7, 0.66, f"STN: time limit at {stn_limit} factories", fontsize=28, color=RED, alpha=a)
        fig.text(0.7, 0.59, f"LSTN: 200 factories in {lstn200:.0f} s", fontsize=28, color=BLUE, alpha=a)
        fig.text(0.7, 0.45, "Paper (commercial solver):", fontsize=24, color=MUTED, alpha=stage(t, 0.65, 0.8))
        fig.text(0.7, 0.40, "2,000 factories in minutes", fontsize=28, color=INK, alpha=stage(t, 0.65, 0.8))
        footer(fig)

    return Video("01_lstn", "LSTN: a linear model of industrial production for demand response", [
        Scene(["LSTN: a linear model of industrial production for demand response.",
               "Here is the idea behind the paper, in two minutes."],
              title_card("IEEE ISGT Europe, 2023", "LSTN: a linear model of industrial production",
                         "R. Lyu, H. Guo, Y. Zheng, Y. Bai, Q. Chen")),
        Scene(["Many factories can shift when they use electricity.",
               "A steel powder plant, for example, has machines in series, with storage buffers between them.",
               "When power is cheap, upstream machines run ahead and fill the buffers. When power is expensive, they slow down."],
              plant_scene),
        Scene(["The classic way to schedule this is a state-task network, or STN.",
               "In every hour, each machine picks exactly one operating mode: off, partial, or full.",
               "That is a yes-or-no decision, so the model needs a binary variable for every machine, mode and hour."],
              stn_scene),
        Scene(["But market prices change hourly, while machines can switch modes within minutes.",
               "So the LSTN asks a different question: how long does each machine spend in each mode during the hour?",
               "Those durations are continuous, so the scheduling problem becomes a linear program."],
              idea_scene),
        Scene(["Under real PJM prices, both models move production away from the expensive afternoon.",
               "The LSTN cost is a lower bound on the STN cost. For this plant, it is within one percent on every day of July."],
              result_scene),
        Scene(["The payoff is scale.",
               f"With a free solver, the STN hits a thirty second limit at {words.get(stn_limit, stn_limit)} factories, "
               f"while the LSTN schedules two hundred in about {lstn200:.0f} seconds.",
               "In the paper, with a commercial solver, the LSTN schedules two thousand factories in minutes."],
              scaling_scene),
        Scene(["The code and an interactive notebook are open source in indflex.",
               "Next: batch processes like steelmaking, and the continuous resource-task network."],
              outro_card(["Notebook 01: STN vs. LSTN (runs on Google Colab)",
                          "Interactive demos: ruikelyu.github.io",
                          "Paper: IEEE ISGT Europe 2023"]), static=True),
    ])


if __name__ == "__main__":
    main(video)
