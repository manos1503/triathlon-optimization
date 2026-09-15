"""Closing the loop: a fixed point between Model 1 and Model 2.

The two models are mutually dependent, which the sequential presentation hides:

    Model 1 needs a GOAL RACE  -> its macrocycle length and volume ceiling
                                  follow from the A-race distance class
    Model 2 needs a FITNESS CURVE -> its readiness gate is driven by the
                                  sustained training load Model 1 produces

Solving them in a fixed order resolves this by fiat. Instead we iterate:

    T, H  <- macrocycle for the current A-race class
    solve Model 1        -> average build-phase load  Lbar
    sustained_load <- Lbar
    solve Model 2        -> selected A-race           -> new class
    repeat until the A-race class stops changing

Each step is a solve of an already-validated model, so the iteration inherits
their guarantees; only the coupling parameters move. In practice it converges
in two or three iterations, and the fixed point is reported as the
self-consistent season plan.

Usage:
    python -m src.analysis.pipeline_loop
"""

from __future__ import annotations

import copy

import pandas as pd
import pulp

from src.models.model1_training import build_model as m1_build
from src.models.model1_training import extract_solution as m1_extract
from src.models.model1_training import solve as m1_solve
from src.models.model2_calendar import build_model as m2_build
from src.models.model2_calendar import extract_solution as m2_extract
from src.models.model2_calendar import solve as m2_solve

from .common import TABLES, load_inputs
from .distance_scenarios import SCENARIOS, scenario_profile

MAX_ITERS = 6


def _macrocycle_for(cls: str) -> str:
    """Map an A-race distance class onto a macrocycle scenario."""
    return {"sprint": "sprint", "olympic": "olympic", "70.3": "70.3"}[str(cls)]


def iterate(profile: dict, state: dict, rates: pd.DataFrame,
            races: pd.DataFrame, start_class: str = "olympic") -> pd.DataFrame:
    cls = start_class
    history, seen = [], {}

    for k in range(1, MAX_ITERS + 1):
        # --- Model 1 under the macrocycle implied by the current A-race class
        p1 = scenario_profile(profile, SCENARIOS[_macrocycle_for(cls)])
        prob1, v1 = m1_build(p1, state, rates)
        assert m1_solve(prob1) == "Optimal", f"Model 1 infeasible for {cls}"
        traj = m1_extract(prob1, v1, state)["trajectory"]

        lo, hi = SCENARIOS[_macrocycle_for(cls)]["phases"]["build"]
        lbar = float(traj.loc[(traj["week"] >= lo) & (traj["week"] <= hi), "load"].mean())

        # --- Model 2 with Model 1's sustained load driving the readiness gate
        p2 = copy.deepcopy(profile)
        p2["model2"]["sustained_weekly_load"] = lbar
        prob2, v2 = m2_build(races, p2)
        assert m2_solve(prob2) == "Optimal", "Model 2 infeasible"
        sel = m2_extract(races, v2)
        chosen = sel[sel["selected"]]
        a_races = chosen[chosen["a_race"] == 1]
        # the A-race is the latest championship-quality target in the calendar
        a_race = a_races.sort_values("week").iloc[-1]
        new_cls = str(a_race["distance_class"])

        history.append({
            "iteration": k,
            "macrocycle_class": cls,
            "weeks": SCENARIOS[_macrocycle_for(cls)]["weeks"],
            "build_load_Lbar": round(lbar, 1),
            "a_race": a_race["id"],
            "a_race_name": a_race["name"],
            "a_race_class": new_cls,
            "n_races": int(len(chosen)),
            "season_value": round(pulp.value(prob2.objective), 1),
            "race_day_ctl": round(traj["ctl"].iloc[-1], 1),
        })

        if new_cls == cls:
            history[-1]["status"] = "FIXED POINT"
            break
        if new_cls in seen:
            history[-1]["status"] = "cycle detected"
            break
        seen[cls] = k
        history[-1]["status"] = f"-> re-plan as {new_cls}"
        cls = new_cls

    return pd.DataFrame(history)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    races = pd.read_csv("data/candidate_races.csv")

    hist = iterate(profile, state, rates, races)
    hist.to_csv(TABLES / "pipeline_loop.csv", index=False)
    print(hist.to_string(index=False))

    last = hist.iloc[-1]
    if last["status"] == "FIXED POINT":
        print(f"\n[loop] converged in {int(last['iteration'])} iterations: "
              f"train a {last['macrocycle_class']} macrocycle "
              f"({int(last['weeks'])} weeks) toward {last['a_race']} "
              f"({last['a_race_name']})")
    else:
        print(f"\n[loop] terminated: {last['status']}")
