"""Draw the results figure from the CSV files written by run_sweeps.py.

Usage (from the repository root):

    python experiments/plot_figure.py

Outputs: results/figure_policies.png, results/figure_policies.pdf,
         results/summary.md (the safety numbers quoted with the figure)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parent.parent / "results"

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Colour follows the policy in every panel. The two unsafe references are drawn
# in grey with broken lines, so safety is never carried by colour alone.
STYLE = {
    "conservative":    dict(color="#2a78d6", ls="-",  marker="o", label="conservative"),
    "hybrid(w=2)":     dict(color="#eb6834", ls="-",  marker="s", label="hybrid, window 2"),
    "optimistic+gate": dict(color="#1baf7a", ls="-",  marker="^", label="optimistic, effects gated"),
    "optimistic":      dict(color=INK_2,     ls="--", marker=None, label="optimistic, not gated (unsafe)"),
    "unsynchronized":  dict(color=INK_3,     ls=":",  marker=None, label="unsynchronized (unsafe)"),
}
WINDOW_COLOR = {"0": "#2a78d6", "2": "#eb6834", "inf": "#1baf7a"}
SLACK_LABEL = {0: "exact footprints", 3: "coarse footprints", 6: "very coarse"}


def read(name: str) -> list[dict]:
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line_panel(ax, rows, metric, ylabel, title) -> None:
    order = ["unsynchronized", "optimistic", "optimistic+gate", "hybrid(w=2)", "conservative"]
    for name in order:
        pts = sorted((float(r["contention"]), float(r[f"{metric}_mean"]), float(r[f"{metric}_sem"]))
                     for r in rows if r["policy"] == name)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        lo = [p[1] - 1.96 * p[2] for p in pts]
        hi = [p[1] + 1.96 * p[2] for p in pts]
        st = STYLE[name]
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=st["color"], linestyle=st["ls"], linewidth=1.4,
                marker=st["marker"], markersize=3.6, markeredgecolor=SURFACE,
                markeredgewidth=0.6)
    ax.set_xlabel("contention", fontsize=7, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=7, color=INK_2)
    ax.set_title(title, fontsize=8, color=INK, loc="left", pad=6)
    ax.set_xlim(0, 1)
    style_axes(ax)


def frontier_panel(ax, rows) -> None:
    for slack in (0, 3, 6):
        pts = [r for r in rows if int(r["footprint_slack"]) == slack]
        x = [float(r["token_ratio_mean"]) for r in pts]
        y = [float(r["speedup_mean"]) for r in pts]
        ax.plot(x, y, color=INK_3, linewidth=1.0, zorder=1)
        for r, xi, yi in zip(pts, x, y):
            w = r["window"]
            if w in WINDOW_COLOR:
                marker = {"0": "o", "2": "s", "inf": "^"}[w]
                ax.plot(xi, yi, marker=marker, markersize=5, color=WINDOW_COLOR[w],
                        markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
            else:
                ax.plot(xi, yi, marker="o", markersize=2.6, color=INK_2, zorder=2)
        ax.annotate(SLACK_LABEL[slack], (x[-1], y[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.annotate("window: 0, 1, 2, 4, 8, unbounded", (0.98, 0.03),
                xycoords="axes fraction", fontsize=6.3, color=INK_2, ha="right", va="bottom")
    ax.set_xlabel("token cost (relative to serial)", fontsize=7, color=INK_2)
    ax.set_ylabel("speedup over serial", fontsize=7, color=INK_2)
    ax.set_title("(c) Speed against cost", fontsize=8, color=INK, loc="left", pad=6)
    ax.set_xlim(0.98, 1.46)
    ax.set_ylim(1, 4.5)
    style_axes(ax)


def write_summary(rows) -> None:
    lines = [
        "# Safety summary (from the policy-independent checker)",
        "",
        "Mean violations per run, 200 workloads per setting, coarse footprints.",
        "",
        "| contention | policy | stale or future reads | leaked irreversible effects | runs with any violation |",
        "|---|---|---|---|---|",
    ]
    for c in ("0.1", "0.3", "0.6"):
        for name in ("conservative", "hybrid(w=2)", "optimistic+gate", "optimistic", "unsynchronized"):
            r = next(r for r in rows if r["policy"] == name and r["contention"] == c)
            reads = float(r["stale_reads_mean"]) + float(r["future_reads_mean"])
            lines.append(
                f"| {c} | {name} | {reads:.2f} | {float(r['leaked_irreversible_mean']):.2f} "
                f"| {100 * float(r['unsafe_run_mean']):.0f}% |"
            )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=3.0, help="figure height in inches")
    parser.add_argument("--name", default="figure_policies", help="output file stem")
    args = parser.parse_args()
    contention = read("sweep_contention.csv")
    window = read("sweep_window.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, args.height), facecolor=SURFACE)
    line_panel(axes[0], contention, "speedup", "speedup over serial",
               "(a) Speed, coarse footprints")
    axes[0].set_ylim(0, 7)
    line_panel(axes[1], contention, "token_ratio", "token cost (relative to serial)",
               "(b) Cost, coarse footprints")
    axes[1].set_ylim(0.9, 3.0)
    frontier_panel(axes[2], window)

    handles = [
        Line2D([0], [0], color=st["color"], linestyle=st["ls"], linewidth=1.4,
               marker=st["marker"], markersize=4, markeredgecolor=SURFACE, label=st["label"])
        for st in STYLE.values()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.4,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.39 / args.height), w_pad=1.6)
    fig.savefig(RESULTS / f"{args.name}.png", dpi=300, facecolor=SURFACE)
    fig.savefig(RESULTS / f"{args.name}.pdf", facecolor=SURFACE)
    write_summary(contention)
    print("wrote figure and summary to", RESULTS)


if __name__ == "__main__":
    main()
