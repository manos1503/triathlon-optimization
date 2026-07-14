"""Model 3 validated against the athlete's three real races.

Three races at three distances, with race-day fitness reconstructed from the
data pipeline, probe different parts of the model:

    Spetsathlon 2026 (sprint*)   CTL 59.0  -- speed caps bind, energy slack
    Epidavros 2025 (Olympic)     CTL 55.6  -- energy budget binds
    Half-Ironman 2025 (70.3)     CTL 71.4  -- infeasible WITHOUT in-race fueling

The 70.3 requires the fueling extension: intake at r kJ/min gives net drain
sum (e - r) t <= E_tot (still an LP). r = 21 kJ/min (~75 g carbs/h, the
standard guideline) reproduces the actual swim and bike within one minute.

Usage:
    python -m src.analysis.race_validation
"""

from __future__ import annotations

import pandas as pd
import pulp

from src.models.model3_pacing import build_model, extract_solution, solve

from .common import TABLES, load_inputs

K_E = 142   # energy constant calibrated on Epidavros (a priori value: 115)

RACES = [
    # name, date, ctl, distances, fueling kJ/min, actual legs (min): swim/bike/run
    ("Spetsathlon 2026 (sprint*)", "2026-04-25", 59.0,
     {"swim": 0.75, "bike": 25.0, "run": 4.75}, 0.0, (13.2, 50.9, 17.7)),
    ("Epidavros 2025 (Olympic)", "2025-09-06", 55.6,
     {"swim": 1.5, "bike": 40.0, "run": 10.0}, 0.0, (25.8, 91.5, 42.0)),
    ("Half-Ironman 2025 (70.3)", "2025-10-26", 71.4,
     {"swim": 1.9, "bike": 90.0, "run": 21.1}, 21.0, (32.9, 180.3, 115.2)),
]


def predict(profile: dict, ctl: float, distances: dict, fueling: float):
    import copy
    p = copy.deepcopy(profile)
    p["model3"]["distances_km"] = distances
    p["model3"]["energy_budget_kj_per_ctl"] = K_E
    prob, v = build_model(p, ctl_race_day=ctl, fueling_kj_min=fueling)
    if solve(prob) != "Optimal":
        return None
    sol = extract_solution(prob, v)
    return sol["leg_times"]


if __name__ == "__main__":
    profile, _, _ = load_inputs()
    rows = []
    for name, date, ctl, dist, fuel, (a_swim, a_bike, a_run) in RACES:
        lt = predict(profile, ctl, dist, fuel)
        rows.append({
            "race": name, "date": date, "ctl": ctl, "fueling_kj_min": fuel,
            "swim_model": round(lt["swim"], 1), "swim_actual": a_swim,
            "bike_model": round(lt["bike"], 1), "bike_actual": a_bike,
            "run_model": round(lt["run"], 1), "run_actual": a_run,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "model3_race_validation.csv", index=False)
    print(df.to_string(index=False))

    # the 70.3 is infeasible without fueling -- demonstrate
    lt = predict(profile, 71.4, {"swim": 1.9, "bike": 90.0, "run": 21.1}, 0.0)
    print(f"\n70.3 without in-race fueling: "
          f"{'feasible' if lt is not None else 'INFEASIBLE (as expected)'}")
