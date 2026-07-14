"""Distance scenarios: re-solve Model 1 for Sprint, Olympic and 70.3 goals.

Different race distances change the PROBLEM SIZE (12/16/20 weeks -> 108/144/180
hour variables) and the parameters (volume ceiling, taper depth), per the
assignment requirement to exercise the model across sizes and parameter sets.

Usage:
    python -m src.analysis.distance_scenarios
"""

from __future__ import annotations

import copy

import pandas as pd
import pulp

from src.models.model1_training import build_model, extract_solution, solve

from .common import FIGURES, TABLES, load_inputs

SCENARIOS = {
    "sprint": {
        "weeks": 12, "hours_max": 10.0,
        "phases": {"base": [1, 5], "build": [6, 9], "peak": [10, 10], "taper": [11, 12]},
        "taper_load_fraction": {11: 0.60, 12: 0.45},
    },
    "olympic": {   # baseline
        "weeks": 16, "hours_max": 12.0,
        "phases": {"base": [1, 6], "build": [7, 12], "peak": [13, 14], "taper": [15, 16]},
        "taper_load_fraction": {15: 0.60, 16: 0.40},
    },
    "70.3": {
        "weeks": 20, "hours_max": 14.0,
        "phases": {"base": [1, 8], "build": [9, 16], "peak": [17, 18], "taper": [19, 20]},
        "taper_load_fraction": {19: 0.60, 20: 0.40},
    },
}


def scenario_profile(profile: dict, spec: dict) -> dict:
    p = copy.deepcopy(profile)
    p["macrocycle"]["weeks"] = spec["weeks"]
    p["macrocycle"]["phases"] = {k: {"weeks": v} for k, v in spec["phases"].items()}
    p["constraints"]["weekly_hours_max"] = spec["hours_max"]
    p["model1"]["taper_load_fraction"] = spec["taper_load_fraction"]
    return p


def run_scenarios(profile: dict, state: dict, rates: pd.DataFrame):
    rows, trajectories = [], {}
    for name, spec in SCENARIOS.items():
        prob, v = build_model(scenario_profile(profile, spec), state, rates)
        assert solve(prob) == "Optimal", f"{name} infeasible"
        sol = extract_solution(prob, v, state)
        traj = sol["trajectory"]
        trajectories[name] = traj
        rows.append({
            "scenario": name,
            "weeks": spec["weeks"],
            "hours_ceiling": spec["hours_max"],
            "n_variables": prob.numVariables(),
            "n_constraints": prob.numConstraints(),
            "objective_pT": round(sol["objective"], 2),
            "total_hours": round(traj["hours"].sum(), 1),
            "avg_weekly_hours": round(traj["hours"].mean(), 1),
            "peak_ctl": round(traj["ctl"].max(), 1),
            "race_day_ctl": round(traj["ctl"].iloc[-1], 1),
            "race_day_tsb": round(traj["ctl"].iloc[-1] - traj["atl"].iloc[-1], 1),
        })
    return pd.DataFrame(rows), trajectories


def plot(trajectories: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sprint": "#0ea5e9", "olympic": "#f59e0b", "70.3": "#ef4444"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, traj in trajectories.items():
        ax.plot(traj["week"], traj["ctl"], "o-", ms=4, color=colors[name],
                label=f"{name} CTL ({len(traj)}w)")
        ax.plot(traj["week"], traj["ctl"] - traj["atl"], "--", lw=1, color=colors[name])
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("Week")
    ax.set_ylabel("CTL (solid) / TSB (dashed)")
    ax.set_title("Model 1 across race distances — fitness build and taper")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    table, trajectories = run_scenarios(profile, state, rates)
    table.to_csv(TABLES / "model1_distance_scenarios.csv", index=False)
    plot(trajectories, FIGURES / "model1_distance_scenarios.png")
    print(table.to_string(index=False))
