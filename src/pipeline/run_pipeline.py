"""End-to-end data pipeline: Strava export -> TRIMP -> CTL/ATL -> Model 1 inputs.

Usage:
    python -m src.pipeline.run_pipeline \
        --input data/raw/activities.csv \
        --profile docs/athlete_profile.yaml \
        --outdir data/processed

Outputs (in --outdir):
    activities_clean.csv   one row per activity with TRIMP and bucket
    ctl_atl_daily.csv      daily load/CTL/ATL/TSB trajectory
    trimp_rates.csv        estimated TRIMP-per-hour by (sport, bucket)  [r_db]
    fitness_state.yaml     current CTL/ATL/TSB                          [CTL_0, ATL_0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .banister import ctl_atl, current_state, daily_load
from .load_strava import load_activities
from .trimp import add_trimp, trimp_rates


def run(input_csv: str, profile_path: str, outdir: str) -> dict:
    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    activities = add_trimp(load_activities(input_csv), profile)
    activities.to_csv(out / "activities_clean.csv", index=False)

    traj = ctl_atl(
        daily_load(activities),
        tau_ctl=profile["banister"]["tau_ctl_days"],
        tau_atl=profile["banister"]["tau_atl_days"],
    )
    traj.round(2).to_csv(out / "ctl_atl_daily.csv")

    rates = trimp_rates(activities)
    rates.round(1).to_csv(out / "trimp_rates.csv", index=False)

    state = current_state(traj)
    with open(out / "fitness_state.yaml", "w") as f:
        yaml.safe_dump(state, f)

    print(f"[pipeline] {len(activities)} activities, "
          f"{traj.index[0].date()} -> {traj.index[-1].date()}")
    print(f"[pipeline] current state: CTL={state['ctl']}  "
          f"ATL={state['atl']}  TSB={state['tsb']}")
    print("[pipeline] TRIMP/hour by sport & bucket:")
    print(rates.round(1).to_string(index=False))
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/activities.csv")
    p.add_argument("--profile", default="docs/athlete_profile.yaml")
    p.add_argument("--outdir", default="data/processed")
    a = p.parse_args()
    run(a.input, a.profile, a.outdir)
