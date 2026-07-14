"""Reproduce every result in the repository with one command.

    python -m src.run_all

Runs: data pipeline -> Model 1 -> Model 2 -> Model 3 -> all analyses and
figures, and writes a model-size/solve-time table (results/tables/model_sizes.csv).

Requires data/raw/activities.csv (personal Strava export, not committed);
without it, the pipeline step is skipped and committed processed data is used.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pulp
import yaml


def main():
    sizes = []

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        print(f"=== {name} done in {time.perf_counter() - t0:.2f}s\n")
        return out

    # 1. data pipeline (optional if raw export absent)
    if Path("data/raw/activities.csv").exists():
        from src.pipeline.run_pipeline import run as pipeline_run
        timed("pipeline", lambda: pipeline_run(
            "data/raw/activities.csv", "docs/athlete_profile.yaml", "data/processed"))
    else:
        print("=== pipeline skipped (no raw export); using committed processed data\n")

    with open("docs/athlete_profile.yaml") as f:
        profile = yaml.safe_load(f)
    with open("data/processed/fitness_state.yaml") as f:
        state = yaml.safe_load(f)
    rates = pd.read_csv("data/processed/trimp_rates.csv")
    races = pd.read_csv("data/candidate_races.csv")

    # 2. models (capture size + solve time)
    from src.models.model1_training import build_model as m1_build
    from src.models.model1_training import solve as m1_solve
    from src.models.model2_calendar import build_model as m2_build
    from src.models.model2_calendar import solve as m2_solve
    from src.models.model3_pacing import build_model as m3_build
    from src.models.model3_pacing import solve as m3_solve
    from src.models.run_model1 import run as run1
    from src.models.run_model2 import run as run2
    from src.models.run_model3 import run as run3

    for name, build, solve_fn, kind in (
        ("Model 1 (training LP)", lambda: m1_build(profile, state, rates), m1_solve, "LP"),
        ("Model 2 (calendar MIP)", lambda: m2_build(races, profile), m2_solve, "0-1 MIP"),
        ("Model 3 (pacing LP)", lambda: m3_build(profile, 81.6), m3_solve, "LP"),
    ):
        prob = build()[0]
        t0 = time.perf_counter()
        status = solve_fn(prob)
        sizes.append({"model": name, "type": kind,
                      "variables": prob.numVariables(),
                      "constraints": prob.numConstraints(),
                      "solve_seconds": round(time.perf_counter() - t0, 3),
                      "status": status,
                      "objective": round(pulp.value(prob.objective), 2)})

    timed("model 1", lambda: run1("docs/athlete_profile.yaml",
                                  "data/processed/fitness_state.yaml",
                                  "data/processed/trimp_rates.csv", "results"))
    timed("model 2", lambda: run2("data/candidate_races.csv",
                                  "docs/athlete_profile.yaml", "results"))
    timed("model 3", lambda: run3("docs/athlete_profile.yaml",
                                  "results/tables/model1_trajectory.csv", "results"))

    # 3. analyses + figures
    import subprocess
    import sys
    for mod in ("src.analysis.plot_model1", "src.analysis.plot_model2",
                "src.analysis.plot_model3", "src.analysis.rhs_ranging",
                "src.analysis.shadow_prices", "src.analysis.alt_optima",
                "src.analysis.heuristic_benchmark", "src.analysis.ftp_sweep",
                "src.analysis.distance_scenarios", "src.analysis.budget_sweep",
                "src.analysis.k2_sensitivity"):
        timed(mod, lambda m=mod: subprocess.run([sys.executable, "-m", m], check=True))

    df = pd.DataFrame(sizes)
    df.to_csv("results/tables/model_sizes.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
