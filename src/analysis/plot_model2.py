"""Season calendar figure for Model 2.

Usage:
    python -m src.analysis.plot_model2   (after src.models.run_model2)

Output:
    results/figures/model2_calendar.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CLASS_Y = {"sprint": 0, "olympic": 1, "70.3": 2}
CLASS_COLOR = {"sprint": "#0ea5e9", "olympic": "#f59e0b", "70.3": "#ef4444"}


def plot_calendar(sel: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for _, r in sel.iterrows():
        y = CLASS_Y[str(r["distance_class"])]
        color = CLASS_COLOR[str(r["distance_class"])]
        if r["selected"]:
            ax.scatter(r["week"], y, s=180, color=color, zorder=3, edgecolor="black", lw=1.2)
            ax.annotate(r["id"], (r["week"], y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
        else:
            marker = "x" if not r["ready"] else "o"
            ax.scatter(r["week"], y, s=70, color=color, alpha=0.35, marker=marker, zorder=2)
            ax.annotate(r["id"], (r["week"], y), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=7, color="#94a3b8")

    ax.set_yticks(list(CLASS_Y.values()), list(CLASS_Y.keys()))
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("Season week (2027)")
    ax.set_title("Model 2 — optimal race calendar "
                 "(solid = selected, faded = rejected, x = not race-ready)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv("results/tables/model2_selection.csv")
    plot_calendar(sel, out / "model2_calendar.png")
    print("[plots] wrote results/figures/model2_calendar.png")
