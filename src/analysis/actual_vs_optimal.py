"""Validation: the athlete's ACTUAL last 16 weeks vs. Model 1's optimal plan.

Parameters were estimated from the training history, but estimation is not
validation. This module compares what the athlete actually did (final 16 weeks
of the Strava export) against the LP's optimal allocation on the same metrics:
weekly hours, intensity polarization, discipline split, and load trajectory.

Usage:
    python -m src.analysis.actual_vs_optimal
"""

from __future__ import annotations

import pandas as pd

from .common import FIGURES, TABLES

WEEKS = 16


def actual_weekly(activities_csv: str) -> pd.DataFrame:
    df = pd.read_csv(activities_csv, parse_dates=["date"])
    end = df["date"].max()
    start = end - pd.Timedelta(weeks=WEEKS)
    df = df[df["date"] > start].copy()
    df["week"] = ((df["date"] - start).dt.days // 7 + 1).clip(upper=WEEKS)

    weekly = df.groupby("week").agg(
        hours=("duration_min", lambda s: s.sum() / 60.0),
        load=("trimp", "sum"),
    )
    hard = df[df["bucket"].isin(["moderate", "hard"])].groupby("week")["duration_min"].sum() / 60.0
    weekly["hard_hours"] = hard.reindex(weekly.index, fill_value=0.0)
    return weekly.reindex(range(1, WEEKS + 1), fill_value=0.0)


def comparison(actual: pd.DataFrame, optimal_plan: pd.DataFrame,
               optimal_traj: pd.DataFrame) -> pd.DataFrame:
    opt_hours = optimal_plan.groupby("week")["hours"].sum()
    opt_hard = (optimal_plan[optimal_plan["bucket"].isin(["moderate", "hard"])]
                .groupby("week")["hours"].sum())
    rows = [{
        "plan": "actual (last 16w)",
        "avg_weekly_hours": round(actual["hours"].mean(), 1),
        "total_hours": round(actual["hours"].sum(), 1),
        "hard_share_pct": round(100 * actual["hard_hours"].sum() / actual["hours"].sum(), 1),
        "avg_weekly_trimp": round(actual["load"].mean(), 0),
        "hours_cv_pct": round(100 * actual["hours"].std() / actual["hours"].mean(), 1),
    }, {
        "plan": "Model 1 optimal",
        "avg_weekly_hours": round(opt_hours.mean(), 1),
        "total_hours": round(opt_hours.sum(), 1),
        "hard_share_pct": round(100 * opt_hard.sum() / opt_hours.sum(), 1),
        "avg_weekly_trimp": round(optimal_traj["load"].mean(), 0),
        "hours_cv_pct": round(100 * opt_hours.std() / opt_hours.mean(), 1),
    }]
    return pd.DataFrame(rows)


def plot(actual: pd.DataFrame, optimal_plan: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opt_hours = optimal_plan.groupby("week")["hours"].sum()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(actual.index - 0.2, actual["hours"], width=0.4,
           color="#94a3b8", label="actual (last 16 weeks)")
    ax.bar(opt_hours.index + 0.2, opt_hours, width=0.4,
           color="#2563eb", label="Model 1 optimal")
    ax.set_xlabel("Week")
    ax.set_ylabel("Training hours")
    ax.set_title("Actual training vs. optimal plan — weekly hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    actual = actual_weekly("data/processed/activities_clean.csv")
    plan = pd.read_csv("results/tables/model1_plan.csv")
    traj = pd.read_csv("results/tables/model1_trajectory.csv")

    table = comparison(actual, plan, traj)
    table.to_csv(TABLES / "model1_actual_vs_optimal.csv", index=False)
    plot(actual, plan, FIGURES / "model1_actual_vs_optimal.png")
    print(table.to_string(index=False))
