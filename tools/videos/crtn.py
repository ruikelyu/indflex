"""Explainer video: cRTN, efficient scheduling of discrete industrial processes through continuous modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: E402,F403

import pyomo.environ as pyo  # noqa: E402

import indflex as fx  # noqa: E402
from indflex.models import crtn_progress, heat_schedule  # noqa: E402

HEAT_COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def data():
    steel = fx.load_batch_plant("steelmaking_zhang2017")
    price30 = np.repeat(fx.daily_price("2022-07", 26), 2)
    m = fx.build_crtn(steel, price30, 3, dt=0.5)
    fx.solve(m)
    gantt = heat_schedule(m)

    small = fx.load_batch_plant("eaf_small_case")
    price = np.array([100, 100, 100, 200, 200, 100, 200, 200.0])
    m_rtn, m_crtn = fx.build_rtn(small, price, 2, tie_break=1e-2), fx.build_crtn(small, price, 2)
    rtn, crtn = fx.solve(m_rtn, mip_gap=0), fx.solve(m_crtn)
    status = np.array([[pyo.value(m_rtn.S[h, t]) for t in range(len(price))] for h in range(2)])
    t_prog, prog = crtn_progress(m_crtn)

    price5 = np.repeat(fx.daily_price("2022-07", 26), 12)
    size = [(h, fx.count_vars(fx.build_rtn(steel, price5, h, dt=1 / 12))[0],
             fx.count_vars(fx.build_crtn(steel, price5, h, dt=1 / 12))[0]) for h in range(1, 11)]
    return steel, gantt, price, status, rtn, crtn, t_prog, prog, size


def video() -> Video:
    steel, gantt, price, status, rtn, crtn, t_prog, prog, size = data()
    stages = [s.name for s in steel.stages]

    def line_scene(fig, t):
        heading(fig, "Steelmaking is a batch process", "heats move through EAF → AOD → LF → CC (one day, three heats)")
        ax = fig.add_axes([0.1, 0.22, 0.8, 0.5])
        now = 2 + stage(t, 0.05, 0.95) * 10  # hours shown
        for _, r in gantt.iterrows():
            y = len(stages) - 1 - stages.index(r.stage)
            end = min(r.end, now)
            if end > r.start:
                ax.barh(y, end - r.start, left=r.start, height=0.6, color=HEAT_COLORS[(r.heat - 1) % 4], ec="white")
                if end - r.start > 0.4:
                    ax.text((r.start + end) / 2, y, f"heat {r.heat}", color="white", ha="center", va="center", fontsize=17)
        ax.axvline(now, color=RED, lw=2)
        ax.set_yticks(range(len(stages)), stages[::-1]); ax.set_xlim(1.5, 12); ax.set_xlabel("Hour"); ax.grid(axis="y")
        fig.text(0.1, 0.79, "• melting cannot stop once started", fontsize=24, color=INK, alpha=stage(t, 0.3, 0.5))
        fig.text(0.45, 0.79, "• hot steel can only wait so long", fontsize=24, color=INK, alpha=stage(t, 0.45, 0.65))
        footer(fig)

    def rtn_scene(fig, t):
        heading(fig, "RTN: tasks and resources for every heat", "binary start variables, durations rounded to whole slots")
        ax = fig.add_axes([0.1, 0.2, 0.5, 0.58])
        k = max(1, int(round(stage(t, 0.1, 0.8) * len(size))))
        ax.plot([s[0] for s in size[:k]], [s[1] for s in size[:k]], "o-", color=RED, lw=3, ms=10)
        ax.set_xlim(0.5, 10.5); ax.set_ylim(0, 42000); ax.set_xlabel("Number of heats"); ax.set_ylabel("Binary variables")
        ax.set_title("RTN, steelmaking at 5-minute slots", loc="left", color=MUTED, fontsize=22)
        heats = size[k - 1][0]
        fig.text(0.66, 0.74, "per heat:", fontsize=26, color=MUTED)
        for i, lab in enumerate(["melting task", "transfer task", "molten steel (ready)", "molten steel (delivered)"]):
            fig.text(0.66, 0.68 - i * 0.05, lab, fontsize=24, color=INK)
        fig.text(0.66, 0.42, f"{heats} heat{'s' if heats > 1 else ''} → {4 * heats} lines", fontsize=34, color=RED,
                 fontweight="demibold")
        footer(fig)

    def idea_scene(fig, t):
        heading(fig, "cRTN: one shared set of tasks, tracked by progress", "how long each task runs at each power level, in every slot")
        ax = fig.add_axes([0.1, 0.42, 0.55, 0.3])
        k = max(2, int(round(stage(t, 0.05, 0.7) * len(t_prog))))
        ax.plot(t_prog[:k], prog[:k], color=RED, lw=4)
        ax.set_xlim(0, 8); ax.set_ylim(-0.05, 1.25); ax.set_xlabel("Hour"); ax.set_ylabel("progress")
        ax.set_title("progress of the heat in the furnace: 0 → 1, then the next heat", loc="left", color=MUTED, fontsize=20)
        fig.text(0.7, 0.68, "4 lines,", fontsize=40, color=BLUE, fontweight="demibold", alpha=stage(t, 0.4, 0.6))
        fig.text(0.7, 0.62, "whatever the number of heats", fontsize=26, color=INK, alpha=stage(t, 0.4, 0.6))
        fig.text(0.1, 0.26, "A few status binaries keep the batch logic (no interruption, waiting limits).", fontsize=26,
                 color=INK, alpha=stage(t, 0.7, 0.9))
        fig.text(0.1, 0.21, "The cRTN reduces binaries; it does not remove them all.", fontsize=26, color=MUTED,
                 alpha=stage(t, 0.75, 0.95))
        footer(fig)

    def rounding_scene(fig, t):
        heading(fig, "Rounding error: the paper's one-furnace example", "two heats, flexible melting power (50–133%)")
        cursor = stage(t, 0.05, 0.85) * 8
        T = len(price)
        for row, (label, y) in enumerate([("RTN", 0.5), ("cRTN", 0.2)]):
            ax = fig.add_axes([0.1, y, 0.58, 0.24])
            ax.fill_between(range(T + 1), np.append(price, price[-1]) / 200 * 1.2, step="post", color="#000000", alpha=0.06)
            if label == "RTN":
                for h in range(2):
                    st = np.append(status[h], status[h][-1])
                    xs = np.arange(T + 1)
                    keep = xs <= cursor
                    ax.step(xs[keep], st[keep], where="post", lw=4, color=[BLUE, ORANGE][h], label=f"heat {h + 1}")
                ax.legend(loc="upper right", ncol=2, fontsize=18)
                ax.set_ylabel("melting")
            else:
                keep = t_prog <= cursor
                ax.plot(t_prog[keep], prog[keep], lw=4, color=RED)
                ax.set_ylabel("progress")
                ax.set_xlabel("Hour (shading = price; darker hours are expensive)")
            ax.set_xlim(0, 8); ax.set_ylim(-0.05, 1.3); ax.axvline(cursor, color=GRAY, lw=1.5)
            ax.text(0.15, 1.12, label, fontsize=26, fontweight="demibold")
        a = stage(t, 0.85, 1.0)
        fig.text(0.72, 0.66, f"RTN  ${rtn.cost:,.2f}", fontsize=34, color=RED, alpha=a)
        fig.text(0.72, 0.36, f"cRTN ${crtn.cost:,.2f}", fontsize=34, color=BLUE, alpha=a)
        footer(fig)

    def scale_scene(fig, t):
        heading(fig, "The model no longer grows with the production target", "binary variables, steelmaking at 5-minute slots")
        ax = fig.add_axes([0.1, 0.2, 0.52, 0.58])
        k = max(1, int(round(stage(t, 0.05, 0.6) * len(size))))
        ax.plot([s[0] for s in size[:k]], [s[1] for s in size[:k]], "o-", color=RED, lw=3, ms=10, label="RTN")
        ax.plot([s[0] for s in size[:k]], [s[2] for s in size[:k]], "o-", color=BLUE, lw=3, ms=10, label="cRTN")
        ax.set_xlim(0.5, 10.5); ax.set_ylim(0, 42000); ax.set_xlabel("Number of heats"); ax.set_ylabel("Binary variables")
        ax.legend(loc="upper left")
        a = stage(t, 0.65, 0.85)
        fig.text(0.66, 0.66, "Paper (8 heats, 15 days):", fontsize=26, color=MUTED, alpha=a)
        fig.text(0.66, 0.59, "156.7 → 17.1 min", fontsize=44, color=INK, fontweight="demibold", alpha=a)
        fig.text(0.66, 0.53, "mean solution time (−89%)", fontsize=26, color=INK, alpha=a)
        footer(fig)

    n10 = size[-1]
    return Video("02_crtn", "cRTN: efficient scheduling of discrete industrial processes through continuous modeling", [
        Scene([("cRTN: efficient scheduling of discrete industrial processes through continuous modeling.",
                "C R T N: efficient scheduling of discrete industrial processes, through continuous modeling."),
               "Here is the idea behind the paper, in two minutes."],
              title_card("IEEE Transactions on Smart Grid (in press)", "cRTN: continuous scheduling of batch processes",
                         "R. Lyu, X. Su, E. Du, H. Guo, Q. Chen, C. Kang")),
        Scene(["Steelmaking is a batch process.",
               "Heats of molten steel move through an electric arc furnace, then refining stages, then a continuous caster.",
               "Once a heat starts melting, it cannot stop, and hot steel can only wait so long between stages."],
              line_scene),
        Scene([("The classic model is the resource-task network, or RTN.", "The classic model is the resource task network, or R T N."),
               "It creates a separate set of tasks and resources for every heat, with a binary variable for when each task starts.",
               "So the model grows with the production target, and every task duration is rounded up to whole time slots."],
              rtn_scene),
        Scene([("The continuous RTN, or cRTN, shares one set of tasks across all heats.",
                "The continuous R T N, or C R T N, shares one set of tasks across all heats."),
               "Instead of asking when each heat's task starts, it tracks the progress of the heat in the furnace, from zero to one, and how long each task runs at each power level in every slot.",
               "A few status binaries keep the batch logic, but their number does not depend on the number of heats."],
              idea_scene),
        Scene(["Here is the paper's example: one furnace, two heats, and flexible melting power.",
               ("The cRTN melts each heat in one and a half hours at high power, and finishes both before prices rise.",
                "The C R T N melts each heat in one and a half hours at high power, and finishes both before prices rise."),
               (f"The RTN can only finish tasks at whole hours, so the second heat runs into an expensive hour: ${rtn.cost:,.2f} instead of ${crtn.cost:,.0f}.",
                f"The R T N can only finish tasks at whole hours, so the second heat runs into an expensive hour: {rtn.cost:.2f} dollars instead of {crtn.cost:.0f}.")],
              rounding_scene),
        Scene(["With more heats, the difference grows.",
               (f"At five-minute resolution, the RTN needs {n10[1]:,} binaries for ten heats, while the cRTN stays at {n10[2]:,}.",
                f"At five minute resolution, the R T N needs {n10[1]:,} binaries for ten heats, while the C R T N stays at {n10[2]:,}."),
               ("In the paper, for eight heats, the cRTN cuts the mean solution time from 156.7 to 17.1 minutes.",
                "In the paper, for eight heats, the C R T N cuts the mean solution time from 157 to 17 minutes.")],
              scale_scene),
        Scene(["The code and an interactive notebook are open source in indflex.",
               "Next: what if we do not know the plant's parameters, and only see its meter data?"],
              outro_card(["Notebook 02: RTN vs. cRTN (runs on Google Colab)",
                          "Interactive demos: ruikelyu.github.io",
                          "Paper: IEEE Transactions on Smart Grid (in press)"]), static=True),
    ])


if __name__ == "__main__":
    main(video)
