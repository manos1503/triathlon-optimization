"""Shadow price extraction and physiological interpretation for Model 1.

The dual value (pi) of each binding constraint measures the marginal change
in race-day performance p_T per unit relaxation of that constraint's RHS.

Usage:
    python -m src.analysis.shadow_prices
"""

from __future__ import annotations

import re

import pandas as pd

from src.models.model1_training import build_model, extract_solution, solve

from .common import TABLES, load_inputs

INTERPRETATION = {
    "hours_cap":    "+1 training hour available in week {w}",
    "atl_cap":      "+1 TRIMP/day fatigue tolerance in week {w}",
    "min_swim":     "swim minimum forced in week {w} (negative = costs performance)",
    "min_bike":     "bike minimum forced in week {w}",
    "min_run":      "run minimum forced in week {w}",
    "max_swim":     "+1 swim hour allowed in week {w}",
    "max_bike":     "+1 bike hour allowed in week {w}",
    "max_run":      "+1 run hour allowed in week {w}",
    "polarization": "+1 unit of allowed moderate/hard fraction in week {w}",
    "ramp":         "relaxed ramp-rate limit in week {w}",
    "taper":        "+1 TRIMP of allowed taper-week load in week {w}",
}

_PAT = re.compile(r"^([a-z_]+?)_w(\d+)$")


def shadow_price_table(duals: dict) -> pd.DataFrame:
    rows = []
    for name, pi in duals.items():
        if name.startswith("def_") or abs(pi) < 1e-9:
            continue  # skip equality definitions and non-binding constraints
        m = _PAT.match(name)
        if not m:
            continue
        kind, week = m.group(1), int(m.group(2))
        meaning = INTERPRETATION.get(kind, kind).format(w=week)
        rows.append({"constraint": name, "type": kind, "week": week,
                     "shadow_price": round(pi, 4), "meaning": meaning})
    return (pd.DataFrame(rows)
            .sort_values("shadow_price", key=abs, ascending=False)
            .reset_index(drop=True))


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    prob, v = build_model(profile, state, rates)
    assert solve(prob) == "Optimal"
    sol = extract_solution(prob, v, state)

    table = shadow_price_table(sol["duals"])
    table.to_csv(TABLES / "model1_shadow_prices.csv", index=False)
    print(f"[shadow_prices] {len(table)} binding constraints, top 15 by |pi|:")
    print(table.head(15).to_string(index=False))
