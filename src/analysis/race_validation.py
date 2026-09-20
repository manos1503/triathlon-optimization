"""Model 3 validated against the athlete's three real races.

Race days were identified in the raw Strava export as the only three days
containing a swim, a bike and a run. Race-day fitness is the CTL/ATL of the
preceding day from the data pipeline.

DISTANCES USE THE OFFICIAL RACE DISTANCE, NOT THE GPS TRACE. Open-water GPS
drift inflates the swim badly (2.26 km recorded for a 1.5 km leg, because the
watch loses signal underwater), and the bike trace reads short on all three
courses by 1.7-3.3% -- a consistent under-read rather than three short courses.
Only the bike CLIMBING is taken from the trace, where GPS/barometer is reliable.

The choice matters: fitting the gradient coefficient on GPS distances gives
c = 0.014, on official distances c = 0.008. That factor-of-two spread is
itself a reason to report the correction as an alternative formulation rather
than a calibrated constant.

    Epidavros 2025      7 Sep 2025   CTL 55.6   bike 15.5 m/km
    Costa Navarino 70.3 26 Oct 2025  CTL 71.4   bike  9.1 m/km
    Spetsathlon 2026    17 May 2026  CTL 64.9   bike 15.2 m/km

Two model variants are compared:

    flat      the baseline speed curve (35 km/h at FTP on a flat course)
    gradient  reference speed scaled by (1 - c * m/km), c = 0.014

c and k_E were fitted jointly on the three races (official distances). The gradient variant roughly
halves the total bike error, and — more importantly — removes its systematic
sign: the flat model is slow on every course, the gradient model errs in both
directions.

Usage:
    python -m src.analysis.race_validation
"""

from __future__ import annotations

import copy

import pandas as pd

from src.models.model3_pacing import build_model, extract_solution, solve

from .common import TABLES, load_inputs

K_E_FLAT = 142       # calibrated on Epidavros with the flat speed curve
K_E_GRADIENT = 160   # re-calibrated jointly with the gradient penalty
C_GRADIENT = 0.008   # speed loss per metre of climbing per km

RACES = [
    # name, date, CTL, ATL, distances (swim nominal), bike m/km, fuelling, actual legs
    ("Spetsathlon 2026 (sprint*)", "2026-05-17", 64.94, 64.49,
     {"swim": 0.75, "bike": 25.00, "run": 5.00}, 15.2, 0.0, (13.2, 50.9, 17.7)),
    ("Epidavros 2025 (Olympic)", "2025-09-07", 55.60, 64.06,
     {"swim": 1.50, "bike": 40.00, "run": 10.00}, 15.5, 0.0, (25.8, 91.5, 42.0)),
    ("IM 70.3 Costa Navarino 2025", "2025-10-26", 71.40, 58.56,
     {"swim": 1.90, "bike": 90.00, "run": 21.10}, 9.1, 21.0, (32.9, 180.3, 115.2)),
]


def predict(profile: dict, ctl: float, distances: dict, gpk: float,
            fuelling: float, gradient: bool):
    p = copy.deepcopy(profile)
    p["model3"]["distances_km"] = distances
    p["model3"]["energy_budget_kj_per_ctl"] = K_E_GRADIENT if gradient else K_E_FLAT
    p["model3"]["bike_gradient_penalty"] = C_GRADIENT if gradient else 0.0
    prob, v = build_model(p, ctl_race_day=ctl, fueling_kj_min=fuelling,
                          gradient_m_per_km=gpk if gradient else 0.0)
    if solve(prob) != "Optimal":
        return None
    return extract_solution(prob, v)["leg_times"]


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    rows = []
    for name, date, ctl, atl, dist, gpk, fuel, (a_sw, a_bk, a_rn) in RACES:
        flat = predict(profile, ctl, dist, gpk, fuel, gradient=False)
        grad = predict(profile, ctl, dist, gpk, fuel, gradient=True)
        rows.append({
            "race": name, "date": date, "ctl": ctl, "bike_m_per_km": gpk,
            "swim_model": round(grad["swim"], 1) if grad is not None else None,
            "swim_actual": a_sw,
            "bike_flat": round(flat["bike"], 1) if flat is not None else None,
            "bike_gradient": round(grad["bike"], 1) if grad is not None else None,
            "bike_actual": a_bk,
            "run_model": round(grad["run"], 1) if grad is not None else None,
            "run_actual": a_rn,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "model3_race_validation.csv", index=False)

    print(df.to_string(index=False))

    def total_err(col):
        d = df.dropna(subset=[col])
        return (d[col] - d["bike_actual"]).abs().sum()

    print(f"\n[validation] bike error, flat model:     {total_err('bike_flat'):5.1f} min")
    print(f"[validation] bike error, gradient model: {total_err('bike_gradient'):5.1f} min")
    signs = [(r["bike_flat"] or 0) - r["bike_actual"] for _, r in df.iterrows()]
    print(f"[validation] flat errors all same sign:  {all(s < 0 for s in signs)}")
    signs_g = [(r["bike_gradient"] or 0) - r["bike_actual"] for _, r in df.iterrows()]
    print(f"[validation] gradient errors mixed sign: {not all(s < 0 for s in signs_g)}")
