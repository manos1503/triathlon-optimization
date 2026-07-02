"""Solve Model 1 with real pipeline inputs and export results.

Usage:
    python -m src.models.run_model1 [--profile ...] [--state ...] [--rates ...] [--outdir results]

Outputs:
    results/tables/model1_plan.csv          hours by (week, discipline, bucket)
    results/tables/model1_trajectory.csv    weekly load/hours/CTL/ATL/TSB
    results/tables/model1_duals.csv         shadow prices of binding constraints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .model1_training import build_model, extract_solution, solve


def run(profile_path: str, state_path: str, rates_path: str, outdir: str) -> dict:
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    with open(state_path) as f:
        state = yaml.safe_load(f)
    rates = pd.read_csv(rates_path)

    prob, v = build_model(profile, state, rates)
    status = solve(prob)
    if status != "Optimal":
        raise RuntimeError(f"Model 1 not optimal: {status}")

    sol = extract_solution(prob, v, state)

    out = Path(outdir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    sol["plan"].round(2).to_csv(out / "model1_plan.csv", index=False)
    sol["trajectory"].round(2).to_csv(out / "model1_trajectory.csv", index=False)
    duals = pd.DataFrame(
        [(k, round(p, 4)) for k, p in sorted(sol["duals"].items()) if abs(p) > 1e-9],
        columns=["constraint", "shadow_price"],
    )
    duals.to_csv(out / "model1_duals.csv", index=False)

    traj = sol["trajectory"]
    print(f"[model1] status={status}  objective p_T={sol['objective']:.2f}")
    print(f"[model1] start: CTL={state['ctl']} ATL={state['atl']}")
    print(traj.round(1).to_string(index=False))
    weekly_hours = sol["plan"].groupby(["week"])["hours"].sum()
    print(f"[model1] avg weekly hours {weekly_hours.mean():.1f}, "
          f"race-day CTL={traj['ctl'].iloc[-1]:.1f} ATL={traj['atl'].iloc[-1]:.1f} "
          f"TSB={traj['ctl'].iloc[-1]-traj['atl'].iloc[-1]:.1f}")
    return sol


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="docs/athlete_profile.yaml")
    p.add_argument("--state", default="data/processed/fitness_state.yaml")
    p.add_argument("--rates", default="data/processed/trimp_rates.csv")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    run(a.profile, a.state, a.rates, a.outdir)
