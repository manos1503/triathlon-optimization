"""Budget sweep for Model 2: season value as a function of the money budget.

IP theory demonstration: the MIP value function of a knapsack is a
non-decreasing STEP function (races are indivisible), while the LP
relaxation's value function is piecewise-linear concave and bounds it
from above. The gap between the curves is the price of integrality.

Usage:
    python -m src.analysis.budget_sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from src.models.model2_calendar import build_model, extract_solution, solve

from .common import FIGURES, TABLES, load_inputs, with_param


def sweep(races: pd.DataFrame, profile: dict, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else np.arange(500, 4001, 250)
    rows = []
    for budget in grid:
        p = with_param(profile, ["model2", "budget_eur"], float(budget))
        prob, v = build_model(races, p)
        assert solve(prob) == "Optimal"
        sel = extract_solution(races, v)
        chosen = sel[sel["selected"]]

        prob_lp, _ = build_model(races, p, relax=True)
        assert solve(prob_lp) == "Optimal"

        rows.append({
            "budget_eur": int(budget),
            "z_mip": round(pulp.value(prob.objective), 1),
            "z_lp": round(pulp.value(prob_lp.objective), 1),
            "n_races": int(len(chosen)),
            "spent_eur": int(chosen["cost"].sum()),
            "days_used": int(chosen["vacation_days"].sum()),
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(df["budget_eur"], df["z_mip"], where="post", color="#2563eb",
            lw=2, label="MIP optimum (step function)")
    ax.plot(df["budget_eur"], df["z_lp"], "o--", ms=4, color="#16a34a",
            label="LP relaxation (concave upper bound)")
    ax.axvline(2500, color="#94a3b8", ls="--", lw=1)
    ax.annotate("baseline budget", (2500, df["z_mip"].min()), fontsize=14,
                textcoords="offset points", xytext=(6, 4), color="#475569")
    ax.set_xlabel("Season budget (EUR)")
    ax.set_ylabel("Optimal season value")
    ax.set_title("Model 2 — value of budget: integer steps vs LP relaxation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    races = pd.read_csv("data/candidate_races.csv")
    df = sweep(races, profile)
    df.to_csv(TABLES / "model2_budget_sweep.csv", index=False)
    plot(df, FIGURES / "model2_budget_sweep.png")
    print(df.to_string(index=False))
