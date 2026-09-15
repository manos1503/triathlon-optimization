"""Pacing-plan figure for Model 3.

Usage:
    python -m src.analysis.plot_model3   (after src.models.run_model3)

Output:
    results/figures/model3_pacing.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LEGS = ["swim", "bike", "run"]
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
ZONE_COLOR = {"Z1": "#bfdbfe", "Z2": "#93c5fd", "Z3": "#fbbf24", "Z4": "#f87171", "Z5": "#dc2626"}


def plot_pacing(plan: pd.DataFrame, path: Path):
    per = plan.pivot_table(index="leg", columns="zone", values="minutes",
                           aggfunc="sum", fill_value=0.0).reindex(LEGS)
    fig, ax = plt.subplots(figsize=(9, 4))
    left = pd.Series(0.0, index=per.index)
    for z in ZONES:
        if z not in per.columns:
            continue
        ax.barh(per.index, per[z], left=left, color=ZONE_COLOR[z], label=z, height=0.55)
        for leg in per.index:
            w = per.loc[leg, z]
            if w > 3:
                ax.text(left[leg] + w / 2, leg, f"{z}\n{w:.0f}'", ha="center",
                        va="center", fontsize=13)
        left += per[z]
    ax.set_xlabel("Minutes")
    ax.set_title("Model 3 — optimal pacing: time in zone per leg "
                 f"(total {plan['minutes'].sum():.0f}' + transitions)")
    ax.legend(loc="lower right", ncols=5, fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv("results/tables/model3_pacing_plan.csv")
    plot_pacing(plan, out / "model3_pacing.png")
    print("[plots] wrote results/figures/model3_pacing.png")
