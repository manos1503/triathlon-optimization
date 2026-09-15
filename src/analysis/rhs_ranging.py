"""RHS ranging: race-day performance as a function of the weekly hours ceiling.

LP theory: the optimal value of an LP is a piecewise-linear concave function
of a <= constraint's right-hand side, with slope equal to the constraint's
shadow price on each linear piece. This module demonstrates the result
empirically by re-solving Model 1 over a grid of weekly hour ceilings.

Usage:
    python -m src.analysis.rhs_ranging
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.model1_training import build_model, solve

from .common import FIGURES, TABLES, load_inputs, with_param


def sweep(profile: dict, state: dict, rates: pd.DataFrame,
          grid=None) -> pd.DataFrame:
    import pulp
    grid = grid if grid is not None else np.arange(8.0, 16.01, 0.5)
    rows = []
    for h_max in grid:
        prob, _ = build_model(with_param(profile, ["constraints", "weekly_hours_max"], float(h_max)),
                              state, rates)
        status = solve(prob)
        rows.append({"weekly_hours_max": float(h_max),
                     "status": status,
                     "objective": pulp.value(prob.objective) if status == "Optimal" else np.nan})
    df = pd.DataFrame(rows)
    # empirical slope between grid points ~ shadow price of the hours cap
    df["slope"] = df["objective"].diff() / df["weekly_hours_max"].diff()
    return df


def plot(df: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["weekly_hours_max"], df["objective"], "o-", color="#2563eb")
    ax.axvline(12, color="#94a3b8", ls="--", lw=1)
    ax.annotate("baseline (12h)", (12, df["objective"].min()), fontsize=14,
                textcoords="offset points", xytext=(6, 4), color="#475569")
    ax.set_xlabel("Weekly hours ceiling $H^{max}$")
    ax.set_ylabel("Optimal race-day performance $p_T$")
    ax.set_title("RHS ranging — $p_T$ is piecewise-linear concave in $H^{max}$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    df = sweep(profile, state, rates)
    df.round(4).to_csv(TABLES / "model1_rhs_ranging.csv", index=False)
    plot(df, FIGURES / "model1_rhs_ranging.png")
    print(df.round(3).to_string(index=False))
    print(f"[rhs_ranging] wrote {TABLES/'model1_rhs_ranging.csv'} and {FIGURES/'model1_rhs_ranging.png'}")
