"""Solve Model 2 (MIP + LP relaxation) and export results.

Usage:
    python -m src.models.run_model2

Outputs:
    results/tables/model2_selection.csv      all candidates with x values (MIP + LP)
    results/tables/model2_summary.csv        objective, budget use, integrality gap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pulp
import yaml

from .model2_calendar import build_model, extract_solution, solve


def run(races_path: str, profile_path: str, outdir: str) -> dict:
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    races = pd.read_csv(races_path)

    # --- full MIP ---
    prob, v = build_model(races, profile)
    status = solve(prob)
    if status != "Optimal":
        raise RuntimeError(f"Model 2 not optimal: {status}")
    mip = extract_solution(races, v)
    z_mip = pulp.value(prob.objective)

    # --- LP relaxation ---
    prob_lp, v_lp = build_model(races, profile, relax=True)
    assert solve(prob_lp) == "Optimal"
    lp = extract_solution(races, v_lp)
    z_lp = pulp.value(prob_lp.objective)
    mip["x_lp_relaxation"] = lp["x"]

    out = Path(outdir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "name", "date", "week", "distance_class", "cost", "value",
            "ready", "x", "selected", "x_lp_relaxation"]
    mip[cols].to_csv(out / "model2_selection.csv", index=False)

    sel = mip[mip["selected"]]
    summary = pd.DataFrame([{
        "z_mip": round(z_mip, 1),
        "z_lp_relaxation": round(z_lp, 1),
        "integrality_gap_pct": round(100 * (z_lp - z_mip) / z_mip, 2),
        "n_selected": int(sel.shape[0]),
        "budget_used_eur": int(sel["cost"].sum()),
        "budget_eur": profile["model2"]["budget_eur"],
        "vacation_days_used": int(sel["vacation_days"].sum()),
        "vacation_days_budget": profile["model2"].get("vacation_days_budget"),
        "n_not_ready": int((~mip["ready"]).sum()),
    }])
    summary.to_csv(out / "model2_summary.csv", index=False)

    # LP-relaxation duals of the two knapsack constraints: marginal season
    # value of one more euro vs. one more vacation day
    duals = {name: c.pi for name, c in prob_lp.constraints.items()
             if name in ("budget", "vacation_days") and c.pi is not None}
    pd.DataFrame([
        {"constraint": k, "shadow_price": round(p, 4),
         "meaning": {"budget": "season value per extra EUR",
                     "vacation_days": "season value per extra day off"}[k]}
        for k, p in duals.items()
    ]).to_csv(out / "model2_lp_duals.csv", index=False)

    frac = lp[(lp["x"] > 0.01) & (lp["x"] < 0.99)]
    print(f"[model2] MIP z={z_mip:.1f}  LP relaxation z={z_lp:.1f}  "
          f"gap {100*(z_lp-z_mip)/z_mip:.2f}%")
    print(f"[model2] budget {int(sel['cost'].sum())}/{profile['model2']['budget_eur']} EUR, "
          f"vacation days {int(sel['vacation_days'].sum())}/{profile['model2'].get('vacation_days_budget')}, "
          f"{len(sel)} races selected, {int((~mip['ready']).sum())} excluded by readiness")
    print("\nselected calendar:")
    print(sel[["id", "name", "date", "distance_class", "cost", "value"]].to_string(index=False))
    if not frac.empty:
        print("\nfractional in LP relaxation (rounded off by B&B):")
        print(frac[["id", "name", "x"]].to_string(index=False))
    if duals:
        print("\nLP-relaxation shadow prices:")
        for k, p in duals.items():
            print(f"  {k}: {p:.4f}")
    return {"mip": mip, "summary": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--races", default="data/candidate_races.csv")
    p.add_argument("--profile", default="docs/athlete_profile.yaml")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    run(a.races, a.profile, a.outdir)
