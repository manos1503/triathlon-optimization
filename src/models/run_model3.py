"""Solve Model 3 with the Model 1 fitness link and export results.

Race-day CTL is read from Model 1's optimal trajectory (last week).

Usage:
    python -m src.models.run_model3

Outputs:
    results/tables/model3_pacing_plan.csv    minutes/km/kJ per (leg, zone)
    results/tables/model3_duals.csv          shadow prices with interpretation
    results/tables/model3_summary.csv        leg splits + totals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .model3_pacing import build_model, extract_solution, solve

DUAL_MEANING = {
    "dist_swim": "marginal min per extra swim km",
    "dist_bike": "marginal min per extra bike km",
    "dist_run": "marginal min per extra run km",
    "energy_budget": "min saved per extra kJ of energy (fitness value)",
    "hard_cap_swim": "min saved per unit relaxed swim intensity cap",
    "hard_cap_bike": "min saved per unit relaxed bike intensity cap",
    "hard_cap_run": "min saved per unit relaxed run intensity cap",
}


def race_day_ctl(trajectory_csv: str) -> float:
    traj = pd.read_csv(trajectory_csv)
    return float(traj["ctl"].iloc[-1])


def run(profile_path: str, trajectory_csv: str, outdir: str) -> dict:
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    ctl = race_day_ctl(trajectory_csv)

    prob, v = build_model(profile, ctl)
    status = solve(prob)
    if status != "Optimal":
        raise RuntimeError(f"Model 3 not optimal: {status}")
    sol = extract_solution(prob, v)

    out = Path(outdir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    sol["plan"].round(2).to_csv(out / "model3_pacing_plan.csv", index=False)

    duals = pd.DataFrame(
        [{"constraint": k, "shadow_price": round(p, 4),
          "meaning": DUAL_MEANING.get(k, k)}
         for k, p in sol["duals"].items() if abs(p) > 1e-9]
    )
    duals.to_csv(out / "model3_duals.csv", index=False)

    lt = sol["leg_times"]
    summary = pd.DataFrame([{
        "ctl_race_day": round(ctl, 1),
        "swim_min": round(lt["swim"], 1),
        "bike_min": round(lt["bike"], 1),
        "run_min": round(lt["run"], 1),
        "transitions_min": round(sol["total_time_min"] - lt.sum(), 1),
        "total_min": round(sol["total_time_min"], 1),
        "energy_used_kj": round(sol["energy_used_kj"]),
        "energy_budget_kj": round(sol["e_tot"]),
    }])
    summary.to_csv(out / "model3_summary.csv", index=False)

    print(f"[model3] CTL={ctl:.1f} -> E_tot={sol['e_tot']:.0f} kJ")
    print(f"[model3] total {sol['total_time_min']:.1f} min "
          f"(swim {lt['swim']:.1f} / bike {lt['bike']:.1f} / run {lt['run']:.1f})")
    print(sol["plan"].round(2).to_string(index=False))
    print("\nshadow prices:")
    print(duals.to_string(index=False))
    return sol


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="docs/athlete_profile.yaml")
    p.add_argument("--trajectory", default="results/tables/model1_trajectory.csv")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    run(a.profile, a.trajectory, a.outdir)
