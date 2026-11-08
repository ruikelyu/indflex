"""Explainer video: PSI, production scheduling identification from smart meter data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import *  # noqa: E402,F403

from export_web_demos import psi_demo  # noqa: E402  (same computation as the web demo; needs Gurobi)

# Paper results (thesis Ch. 3, Table "Load model accuracy comparison"): nRMSE on test days
PAPER = {"cement": (5.2, 13.4), "steel powder": (8.5, 18.5)}  # (PSI, best machine-learning model)


def video() -> Video:
    P = psi_demo()
    hours = np.arange(24)
    meter, mprice = np.array(P["meter"]), np.array(P["meter_price"])
    ev = next(e for e in P["events"] if e["start"] == 6)
    base, resp_id, resp_true = np.array(P["baseline"]), np.array(ev["identified"]), np.array(ev["true"])

    def problem_scene(fig, t):
        heading(fig, "What the grid can see", "an aggregator sees prices and meter data, not the process")
        ax = fig.add_axes([0.07, 0.24, 0.5, 0.52])
        k = max(1, int(round(stage(t, 0.0, 0.5) * 24)))
        ax.bar(hours[:k], meter[:k], color=BLUE, alpha=0.6, label="meter data (MWh)")
        ax.set_xlim(-0.6, 23.6); ax.set_ylim(0, meter.max() * 1.3); ax.set_xlabel("Hour"); ax.legend(loc="upper left")
        ax2 = ax.twinx(); ax2.plot(hours[:k], mprice[:k], "--", color=GRAY, lw=2.5); ax2.grid(False)
        ax2.set_ylabel("price ($/MWh)", color=GRAY); ax2.set_ylim(0, mprice.max() * 1.3)
        a = stage(t, 0.55, 0.8)
        fig.text(0.64, 0.72, "Private:", fontsize=32, color=RED, fontweight="demibold", alpha=a)
        for i, item in enumerate(["rated power of each machine", "buffer sizes and levels", "daily production target"]):
            fig.text(0.64, 0.64 - i * 0.07, "–  " + item, fontsize=28, color=INK, alpha=a)
        footer(fig)

    def idea_scene(fig, t):
        heading(fig, "Run the optimization backwards", "the factory is assumed to minimize its electricity cost")
        boxes = [(0.08, "plant parameters θ"), (0.40, "cost-minimizing\nschedule (an LP)"), (0.72, "hourly load")]
        for x, label in boxes:
            fig.patches.append(plt.Rectangle((x, 0.5), 0.2, 0.16, transform=fig.transFigure, fc="#eef3fb", ec=BLUE, lw=2))
            fig.text(x + 0.1, 0.58, label, fontsize=26, ha="center", va="center", color=INK)
        a1 = stage(t, 0.1, 0.35)
        for x0, x1 in [(0.29, 0.39), (0.61, 0.71)]:
            fig.add_artist(plt.Line2D([x0, x0 + (x1 - x0) * a1], [0.6, 0.6], transform=fig.transFigure, color=BLUE, lw=4))
        fig.text(0.5, 0.71, "forward: parameters → schedule", fontsize=26, color=BLUE, ha="center", alpha=a1)
        a2 = stage(t, 0.55, 0.85)
        for x0, x1 in [(0.71, 0.61), (0.39, 0.29)]:
            fig.add_artist(plt.Line2D([x0, x0 + (x1 - x0) * a2], [0.54, 0.54], transform=fig.transFigure, color=ORANGE,
                                      lw=4))
        fig.text(0.5, 0.42, "PSI, backward: meter data → parameters", fontsize=30, color=ORANGE, ha="center",
                 fontweight="demibold", alpha=a2)
        footer(fig)

    def method_scene(fig, t):
        heading(fig, "Inverse optimization", "make the observed schedule optimal; the parameters become variables")
        items = [("primal feasibility", "the schedule respects the plant's constraints"),
                 ("dual feasibility", "a matching set of shadow prices exists"),
                 ("strong duality", "both give the same cost  →  schedule is optimal")]
        for i, (head, text) in enumerate(items):
            a = stage(t, 0.05 + 0.15 * i, 0.2 + 0.15 * i)
            y = 0.68 - i * 0.12
            fig.text(0.08, y, head, fontsize=32, color=BLUE, fontweight="demibold", alpha=a)
            fig.text(0.36, y, text, fontsize=28, color=INK, alpha=a)
        a = stage(t, 0.6, 0.8)
        fig.text(0.08, 0.3, "minimize   Σ (model load − meter data)²", fontsize=32, color=INK, alpha=a)
        fig.text(0.08, 0.23, "parameters × shadow prices → bilinear, nonconvex (Gurobi NonConvex)", fontsize=26,
                 color=RED, alpha=stage(t, 0.75, 0.95))
        footer(fig)

    def toy_scene(fig, t):
        heading(fig, "Recovering a plant from one day of data", "two machines; true parameters known, so we can check")
        ax = fig.add_axes([0.1, 0.22, 0.42, 0.55])
        a = stage(t, 0.2, 0.75)
        true_p, id_p = P["params"]["true_p_max"], P["params"]["identified_p_max"]
        x = np.arange(2)
        ax.bar(x - 0.18, true_p, width=0.34, color=GRAY, alpha=0.5, label="true")
        ax.bar(x + 0.18, np.array(id_p) * a, width=0.34, color=ORANGE, label="identified")
        ax.set_xticks(x, ["machine 1", "machine 2"]); ax.set_ylabel("rated power (MW)"); ax.set_ylim(0, 10)
        ax.legend(loc="upper left")
        b = stage(t, 0.6, 0.85)
        fig.text(0.6, 0.66, f"rated powers: {id_p[0]:g} and {id_p[1]:g} MW", fontsize=30, color=INK, alpha=b)
        fig.text(0.6, 0.59, f"daily output: {P['params']['identified_target']:g} units", fontsize=30, color=INK, alpha=b)
        fig.text(0.6, 0.50, "one inverse solve, about a second", fontsize=26, color=MUTED, alpha=b)
        fig.text(0.6, 0.40, "buffer levels the data never exercise\ncannot be identified, but do not change the load",
                 fontsize=22, color=MUTED, alpha=stage(t, 0.8, 1.0))
        footer(fig)

    def whatif_scene(fig, t):
        heading(fig, "Using the model: a demand-response event", "prices spike from 6:00 to 11:00")
        ax = fig.add_axes([0.08, 0.2, 0.84, 0.56])
        ax.axvspan(5.5, 10.5, color=ORANGE, alpha=0.12)
        ax.plot(hours, base, "o-", color=RED, lw=3, ms=7, label="baseline (identified model)")
        a = stage(t, 0.3, 0.8)
        k = max(1, int(round(a * 24)))
        ax.plot(hours[:k], resp_id[:k], "--", color=BLUE, lw=3.5, label="response (identified model)")
        ax.plot(hours[:k], resp_true[:k], ":", color=INK, lw=3, label="response (true plant)")
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, base.max() * 1.3); ax.set_xlabel("Hour"); ax.set_ylabel("MWh")
        ax.legend(loc="upper right", ncol=3, fontsize=18)
        footer(fig)

    def paper_scene(fig, t):
        heading(fig, "Real plants: many days, iterate", "solve one day at a time, average the estimates until they agree")
        ax = fig.add_axes([0.1, 0.2, 0.5, 0.56])
        a = stage(t, 0.35, 0.7)
        x = np.arange(2)
        ax.bar(x - 0.18, [v[0] * a for v in PAPER.values()], width=0.34, color=ORANGE, label="PSI")
        ax.bar(x + 0.18, [v[1] * a for v in PAPER.values()], width=0.34, color=GRAY, label="best machine learning")
        ax.set_xticks(x, list(PAPER)); ax.set_ylabel("test error, nRMSE (%)"); ax.set_ylim(0, 22); ax.legend(loc="upper left")
        for i, (p, m) in enumerate(PAPER.values()):
            ax.text(i - 0.18, p * a + 0.5, f"{p}%", ha="center", fontsize=22, alpha=a)
            ax.text(i + 0.18, m * a + 0.5, f"{m}%", ha="center", fontsize=22, alpha=a)
        fig.text(0.65, 0.66, "Paper: 21 days of hourly data", fontsize=28, color=INK, alpha=stage(t, 0.4, 0.6))
        fig.text(0.65, 0.59, "structured model + few data", fontsize=26, color=MUTED, alpha=stage(t, 0.45, 0.65))
        fig.text(0.65, 0.53, "beats black-box models", fontsize=26, color=MUTED, alpha=stage(t, 0.45, 0.65))
        footer(fig)

    return Video("03_psi", "PSI: production scheduling identification from smart meter data", [
        Scene([("PSI: production scheduling identification.", "P S I: production scheduling identification."),
               "An inverse optimization approach for modeling industrial loads from smart meter data."],
              title_card("IEEE Transactions on Smart Grid, 2025", "PSI: learning a factory from its meter",
                         "R. Lyu, H. Guo, Q. Tang, Q. Chen, C. Kang")),
        Scene(["A grid operator or aggregator rarely knows a factory's process.",
               "It sees two things: the prices the factory faced, and its hourly meter data.",
               "Rated powers, buffer sizes and production targets stay private."], problem_scene),
        Scene([("PSI assumes the factory schedules production to minimize its electricity cost.",
                "P S I assumes the factory schedules production to minimize its electricity cost."),
               "Normally we go forward: given the parameters, solve for the optimal schedule.",
               ("PSI goes backward: given the meter data, find the parameters that make it optimal.",
                "P S I goes backward: given the meter data, find the parameters that make it optimal.")], idea_scene),
        Scene(["This is inverse optimization.",
               "We write down what it means for a schedule to be optimal: it is feasible, a matching set of shadow prices exists, and both give the same cost.",
               "The private parameters then become decision variables. Because they multiply the shadow prices, the problem is nonconvex, so we use Gurobi."],
              method_scene),
        Scene(["Here is a two-machine plant whose parameters we know.",
               f"From one day of meter data, a single solve recovers its rated powers, {P['params']['identified_p_max'][0]:g} and {P['params']['identified_p_max'][1]:g} megawatts, and its daily output, exactly."],
              toy_scene),
        Scene(["The identified model can then answer what-if questions.",
               "If prices spike from six to eleven in the morning, it predicts that the plant stops and shifts production later, just as the true plant does."],
              whatif_scene),
        Scene([("For real plants and many days of data, PSI solves one day at a time and averages the estimates until they agree.",
                "For real plants and many days of data, P S I solves one day at a time, and averages the estimates until they agree."),
               ("In the paper, PSI predicts a cement plant and a steel powder plant with 5.2 and 8.5 percent error, against 13.4 and 18.5 percent for the best machine learning models.",
                "In the paper, P S I predicts a cement plant and a steel powder plant with 5.2 and 8.5 percent error, against 13.4 and 18.5 percent for the best machine learning models.")],
              paper_scene),
        Scene(["The code and an interactive notebook are open source in indflex.",
               "Next: compressing thousands of plant models into a few numbers each."],
              outro_card(["Notebook 03: PSI (needs a Gurobi license; free for academics)",
                          "Interactive demos: ruikelyu.github.io",
                          "Paper: IEEE Transactions on Smart Grid 16(2), 2025"]), static=True),
    ])


if __name__ == "__main__":
    main(video)
