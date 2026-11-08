"""Explainer video: real-time operation of a virtual power plant with optimal power disaggregation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import *  # noqa: E402,F403

from export_web_demos import vpp_demo  # noqa: E402  (same computation as the web demo)

KIND = {"IPP": ("industrial stages", ORANGE), "ES": ("storage", PURPLE), "TCL": ("air conditioning", GREEN),
        "EV": ("EVs", BLUE), "PV": ("PV", GOLD)}


def video() -> Video:
    V = vpp_demo(step=1)
    mins = np.array(V["minutes"])
    preq = np.array(V["p_req"])
    M = V["merit"]
    widths, costs, kinds = np.array(M["width"]), np.array(M["cost"]), M["kind"]
    lefts = M["base"] + np.r_[0, np.cumsum(widths)[:-1]]
    hours = np.arange(len(V["bid_p"]))

    def resources_scene(fig, t):
        heading(fig, "A virtual power plant pools small, different resources", "one day of the paper's case")
        items = [("EV", V["counts"]["EV"]), ("IPP", V["counts"]["IPP"]), ("TCL", V["counts"]["TCL"]),
                 ("ES", V["counts"]["ES"]), ("PV", V["counts"]["PV"])]
        for i, (k, n) in enumerate(items):
            a = stage(t, 0.05 + 0.1 * i, 0.15 + 0.1 * i)
            x = 0.08 + i * 0.18
            fig.text(x, 0.62, str(n), fontsize=72, color=KIND[k][1], fontweight="demibold", alpha=a)
            fig.text(x, 0.54, KIND[k][0], fontsize=26, color=INK, alpha=a)
        ax = fig.add_axes([0.08, 0.2, 0.84, 0.22])
        k = max(2, int(round(stage(t, 0.55, 0.95) * len(mins))))
        ax.plot(mins[:k], (preq[:k] - V["bid_p"][0]) / V["bid_r"][0], color=INK, lw=1.5)
        ax.set_xlim(0, 60); ax.set_ylim(-1.1, 1.1); ax.set_ylabel("signal δ"); ax.set_xlabel("Minute (RegD, one value every 2 s)")
        footer(fig)

    def timescale_scene(fig, t):
        heading(fig, "Two time scales", "bid every half hour; follow the signal every 2 seconds")
        ax = fig.add_axes([0.08, 0.3, 0.5, 0.45])
        k = max(1, int(round(stage(t, 0.1, 0.5) * len(hours))))
        ax.bar(hours[:k] - 0.2, V["bid_p"][:k], width=0.4, color=GRAY, label="energy bid")
        ax.bar(hours[:k] + 0.2, V["bid_r"][:k], width=0.4, color=RED, label="regulation capacity bid")
        ax.axhline(0, color=MUTED, lw=1); ax.set_xlim(-0.6, 23.6); ax.set_xlabel("Hour"); ax.set_ylabel("MW")
        ax.legend(loc="upper left", fontsize=17); ax.set_title("bidding LP, under 22 signal scenarios", loc="left", fontsize=20, color=MUTED)
        a = stage(t, 0.55, 0.75)
        fig.text(0.63, 0.66, "every 2 seconds:", fontsize=28, color=MUTED, alpha=a)
        fig.text(0.63, 0.58, "P_req = P_bid + R_bid · δ", fontsize=36, color=INK, alpha=a)
        fig.text(0.63, 0.48, "split among 135 resources, now", fontsize=28, color=INK, alpha=stage(t, 0.65, 0.85))
        fig.text(0.63, 0.41, "an optimization here is too slow", fontsize=26, color=RED, alpha=stage(t, 0.8, 1.0))
        footer(fig)

    def merit(ax, req=None, reveal=1.0):
        n = max(1, int(round(reveal * len(costs))))
        for i in range(n):
            col = KIND[kinds[i]][1]
            ax.bar(lefts[i], costs[i], width=widths[i], align="edge", color=col, alpha=0.3, ec="white")
            if req is not None:
                filled = min(max(req - lefts[i], 0), widths[i])
                if filled > 0:
                    ax.bar(lefts[i], costs[i], width=filled, align="edge", color=col, alpha=0.95, ec="white")
        if req is not None:
            ax.axvline(req, color=RED, lw=3)
        ax.set_xlim(M["base"], M["base"] + widths.sum()); ax.set_ylim(0, costs.max() * 1.15)
        ax.set_xlabel("VPP output (MW)"); ax.set_ylabel("marginal cost ($/MWh)")

    def duals_scene(fig, t):
        heading(fig, "Duals turn the split into a merit order", "λ = value of one more unit of stored energy, from the bidding LP")
        ax = fig.add_axes([0.08, 0.2, 0.55, 0.56])
        merit(ax, reveal=stage(t, 0.3, 0.85))
        shown = sorted(set(kinds), key=list(KIND).index)
        for i, k in enumerate(shown):
            fig.text(0.68, 0.68 - i * 0.06, "■ " + KIND[k][0], fontsize=26, color=KIND[k][1])
        fig.text(0.68, 0.4, "marginal cost = usage cost", fontsize=24, color=INK, alpha=stage(t, 0.5, 0.7))
        fig.text(0.68, 0.35, "± value of stored energy", fontsize=24, color=INK, alpha=stage(t, 0.5, 0.7))
        fig.text(0.68, 0.27, "cheapest flexibility first", fontsize=28, color=BLUE, fontweight="demibold",
                 alpha=stage(t, 0.75, 0.95))
        footer(fig)

    def follow_scene(fig, t):
        heading(fig, "Following the signal", "fill the merit order up to the requested power")
        i = min(int(t * (len(mins) - 1)), len(mins) - 1)
        ax = fig.add_axes([0.06, 0.2, 0.44, 0.56])
        ax.plot(mins, preq, color=LIGHT, lw=1.5)
        ax.plot(mins[:i + 1], preq[:i + 1], color=INK, lw=1.8, label="requested power")
        for k in ["IPP", "ES", "TCL"]:
            ys = np.array(V["by_kind"][k])
            if np.abs(ys).max() > 1e-6:
                ax.plot(mins[:i + 1], ys[:i + 1], color=KIND[k][1], lw=1.6, label=KIND[k][0])
        ax.axvline(mins[i], color=RED, lw=1.5)
        ax.set_xlim(0, 60); ax.set_xlabel("Minute"); ax.set_ylabel("MW")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=16, ncol=4, handlelength=1.2, columnspacing=0.8)
        ax2 = fig.add_axes([0.56, 0.2, 0.4, 0.56])
        merit(ax2, req=preq[i])
        fig.text(0.56, 0.79, f"requested now: {preq[i]:.2f} MW", fontsize=24, color=RED)
        footer(fig)

    return Video("05_vpp", "Real-time operation of virtual power plants with optimal power disaggregation", [
        Scene(["Real-time operation of virtual power plants, with optimal power disaggregation among heterogeneous resources.",
               "Here is the idea behind the paper, in two minutes."],
              title_card("Applied Energy 361, 2024", "Virtual power plants: bid, then split in microseconds",
                         "Q. Chen, R. Lyu, H. Guo, X. Su")),
        Scene(["A virtual power plant pools many small resources: electric vehicles, industrial production stages, air conditioning, storage, and solar.",
               "It sells energy and frequency regulation.",
               ("Regulation means following a signal from the grid operator, updated every two seconds.",
                "Regulation means following a signal from the grid operator, updated every two seconds.")], resources_scene),
        Scene(["So the virtual power plant decides on two time scales.",
               "Every half hour, a bidding optimization chooses how much energy and regulation capacity to offer, planning every resource under many possible signals.",
               "Every two seconds, it must split the requested power among its resources at once, and solving an optimization there would be too slow."],
              timescale_scene),
        Scene(["The key idea is to reuse the dual variables of the bidding problem.",
               "They tell us what one more unit of stored energy is worth in each resource.",
               "That gives every resource a marginal cost. Sorted, they form a merit order: use the cheapest flexibility first."],
              duals_scene),
        Scene(["When the signal arrives, the virtual power plant simply fills the merit order up to the requested power.",
               "This is the exact solution of the split problem, computed in a few microseconds instead of milliseconds with a solver.",
               "Here, storage, air conditioning and the industrial stages take turns as the signal moves."],
              follow_scene),
        Scene(["The code and an interactive notebook are open source in indflex.",
               "Together, the five steps go from process physics to grid decisions."],
              outro_card(["Notebook 05: VPP (free solver; runs on Google Colab)",
                          "Interactive demos: ruikelyu.github.io",
                          "Paper: Applied Energy 361, 2024"]), static=True),
    ])


if __name__ == "__main__":
    main(video)
