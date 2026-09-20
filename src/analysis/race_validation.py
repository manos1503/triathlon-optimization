"""Model 3 validated against the athlete's three real races.

Race days were identified in the raw Strava export as the only three days
containing a swim, a bike and a run. Race-day fitness is the CTL/ATL of the
preceding day from the data pipeline.

DISTANCES ARE THE GPS TRACE FOR BIKE AND RUN, AND THE NOMINAL RACE DISTANCE
FOR THE SWIM. The watch loses satellite lock underwater, so the open-water swim
trace is unusable (2.26 km recorded for a 1.5 km leg); on land the trace is
reliable and is what the athlete actually covered. Climbing likewise comes from
the trace.

The bike speed curve accounts for each course's climbing through
``model3_pacing.grade_factor``, which is derived from the cycling power balance
and fits NOTHING to these races -- so the comparison below is a genuine
out-of-sample test, not a curve fit. For contrast the script also reports a
two-parameter empirical fit (a linear penalty c together with a re-fitted k_E),
which is fitted ON these three races and is therefore an upper bound on how
well any such correction could look.

Note on Spetsathlon: its bike leg is 25 km, not the 20 km of a standard sprint,
which is why it is marked sprint* throughout.

    Epidavros 2025      7 Sep 2025   CTL 55.6   bike 15.5 m/km
    Costa Navarino 70.3 26 Oct 2025  CTL 71.4   bike  9.1 m/km
    Spetsathlon 2026    17 May 2026  CTL 64.9   bike 15.2 m/km

Three variants are compared:

    flat      pretending every course is flat (the model before the fix)
    physics   grade_factor from the power balance -- nothing fitted  [DEFAULT]
    fitted    linear penalty c = 0.014 with k_E = 190, fitted on these races

Total bike error: 23.0 min flat, 10.6 min physics, 8.2 min fitted. The physics
model recovers most of the available improvement with zero free parameters, and
its errors fall on both sides of zero; the flat model is slow on every course.

Usage:
    python -m src.analysis.race_validation
"""

from __future__ import annotations

import copy

import pandas as pd

from src.models.model3_pacing import build_model, extract_solution, solve

from .common import TABLES, load_inputs

K_E = 142            # feasibility-calibrated on Epidavros; unchanged throughout
K_E_FITTED = 190     # only for the empirical-fit comparison column
C_FITTED = 0.014     # only for the empirical-fit comparison column

RACES = [
    # name, date, CTL, ATL, distances (bike/run = GPS, swim = nominal), m/km, fuel, actual
    ("Spetsathlon 2026 (sprint*)", "2026-05-17", 64.94, 64.49,
     {"swim": 0.75, "bike": 24.40, "run": 4.70}, 15.2, 0.0, (13.2, 50.9, 17.7)),
    ("Epidavros 2025 (Olympic)", "2025-09-07", 55.60, 64.06,
     {"swim": 1.50, "bike": 38.69, "run": 9.22}, 15.5, 0.0, (25.8, 91.5, 42.0)),
    ("IM 70.3 Costa Navarino 2025", "2025-10-26", 71.40, 58.56,
     {"swim": 1.90, "bike": 88.46, "run": 21.13}, 9.1, 21.0, (32.9, 180.3, 115.2)),
]


def predict(profile: dict, ctl: float, distances: dict, gpk: float,
            fuelling: float, mode: str):
    """mode: 'flat' | 'physics' | 'fitted'."""
    p = copy.deepcopy(profile)
    p["model3"]["distances_km"] = distances
    p["model3"]["energy_budget_kj_per_ctl"] = K_E_FITTED if mode == "fitted" else K_E
    if mode == "fitted":
        # emulate the old linear penalty by scaling the reference speed directly
        p["model3"]["bike_speed_at_ftp_kmh"] *= max(0.25, 1.0 - C_FITTED * gpk)
        g = 0.0
    else:
        g = gpk if mode == "physics" else 0.0
    prob, v = build_model(p, ctl_race_day=ctl, fueling_kj_min=fuelling,
                          gradient_m_per_km=g)
    if solve(prob) != "Optimal":
        return None
    return extract_solution(prob, v)["leg_times"]


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    rows = []
    for name, date, ctl, atl, dist, gpk, fuel, (a_sw, a_bk, a_rn) in RACES:
        flat = predict(profile, ctl, dist, gpk, fuel, "flat")
        phys = predict(profile, ctl, dist, gpk, fuel, "physics")
        fit = predict(profile, ctl, dist, gpk, fuel, "fitted")
        rows.append({
            "race": name, "date": date, "ctl": ctl, "bike_m_per_km": gpk,
            # the shipped model: gradient-aware, nothing fitted to these races
            "swim_model": round(phys["swim"], 1) if phys is not None else None,
            "swim_actual": a_sw,
            "bike_model": round(phys["bike"], 1) if phys is not None else None,
            "bike_actual": a_bk,
            "run_model": round(phys["run"], 1) if phys is not None else None,
            "run_actual": a_rn,
            # comparators
            "bike_flat": round(flat["bike"], 1) if flat is not None else None,
            "bike_fitted": round(fit["bike"], 1) if fit is not None else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "model3_race_validation.csv", index=False)

    print(df.to_string(index=False))

    def total_err(col):
        d = df.dropna(subset=[col])
        return (d[col] - d["bike_actual"]).abs().sum()

    for col, label in (("bike_flat", "flat (no gradient)"),
                       ("bike_model", "physics (shipped, 0 fitted)"),
                       ("bike_fitted", "empirical fit (2 fitted)")):
        e = (df[col] - df["bike_actual"])
        print(f"[validation] bike error {label:28} {e.abs().sum():5.1f} min"
              f"   all one sign: {bool((e < 0).all())}")
