"""Figures for Model 1 results.

Usage:
    python -m src.analysis.plot_model1   (after src.models.run_model1)

Outputs:
    results/figures/model1_weekly_hours.png    stacked hours by discipline
    results/figures/model1_trajectory.png      CTL/ATL/TSB with phase shading
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASES = {"base": (1, 6), "build": (7, 12), "peak": (13, 14), "taper": (15, 16)}
PHASE_COLOR = {"base": "#dbeafe", "build": "#fde68a", "peak": "#fecaca", "taper": "#d1fae5"}
DISC_COLOR = {"swim": "#0ea5e9", "bike": "#f59e0b", "run": "#ef4444"}


def _shade_phases(ax):
    for name, (lo, hi) in PHASES.items():
        ax.axvspan(lo - 0.5, hi + 0.5, color=PHASE_COLOR[name], alpha=0.5, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.97, name,
                ha="center", va="top", fontsize=9, color="#334155")


def plot_weekly_hours(plan: pd.DataFrame, path: Path):
    per = plan.groupby(["week", "discipline"])["hours"].sum().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = None
    for d in ("swim", "bike", "run"):
        ax.bar(per.index, per[d], bottom=bottom, label=d, color=DISC_COLOR[d], width=0.7)
        bottom = per[d] if bottom is None else bottom + per[d]
    ax.set_xlabel("Week")
    ax.set_ylabel("Training hours")
    ax.set_title("Model 1 — optimal weekly hours by discipline")
    ax.legend(loc="upper right")
    _shade_phases(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectory(traj: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(traj["week"], traj["ctl"], "o-", label="CTL (fitness)", color="#2563eb")
    ax.plot(traj["week"], traj["atl"], "s-", label="ATL (fatigue)", color="#dc2626")
    ax.plot(traj["week"], traj["ctl"] - traj["atl"], "^-", label="TSB (form)", color="#16a34a")
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("Week")
    ax.set_ylabel("TRIMP (daily-scale)")
    ax.set_title("Model 1 — optimal CTL/ATL/TSB trajectory")
    ax.legend(loc="center left")
    _shade_phases(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv("results/tables/model1_plan.csv")
    traj = pd.read_csv("results/tables/model1_trajectory.csv")
    plot_weekly_hours(plan, out / "model1_weekly_hours.png")
    plot_trajectory(traj, out / "model1_trajectory.png")
    print("[plots] wrote results/figures/model1_weekly_hours.png, model1_trajectory.png")
