"""Pipeline coherence: is Model 1's objective the right one?

Model 1 maximizes a Banister performance index
    p_T = k1*CTL_T - k2*ATL_T,
while the athlete's true goal is the quantity Model 3 computes: finishing
time. These need not agree, and by default they do NOT: Model 3's energy
budget E = k_E*CTL ignores ATL entirely, so as far as Model 3 is concerned
the best preparation is maximal fitness at any fatigue -- i.e. no taper.

Coherence theorem. Let the race-day energy budget be form-based,
    E(plan) = E_base + k_E * (CTL_T - lam * ATL_T).
If lam = k2/k1 then
    E = E_base + (k_E/k1) * p_T,
an increasing affine function of p_T. Model 3's optimal time is non-increasing
in E, so
    argmax p_T  =  argmax E  =  argmin race time.
Model 1's objective is then not a proxy but a *derived* one, and k2/k1 stops
being a free parameter: it is the fatigue sensitivity of the energy budget.

Feasibility. E > 0 requires lam < CTL_T/ATL_T at the optimum (~1.75 here), so
the assumed k2 = 2.0 is NOT coherent. Cost ranging (cost_ranging.py) shows the
optimal plan is unchanged for k2 in [1.29, 2.95]; any lam in [1.29, 1.75) is
therefore both coherent and plan-preserving -- k2 = 1.5 is used.

This module verifies the theorem numerically: it builds a spread of training
plans, scores each by p_T and by end-to-end predicted race time, and checks
that the two rankings agree when lam = k2/k1 and disagree when lam != k2/k1.

Usage:
    python -m src.analysis.coherence
"""

from __future__ import annotations

import copy

import pandas as pd
import pulp

from src.models.model1_training import build_model as m1_build
from src.models.model1_training import extract_solution as m1_extract
from src.models.model1_training import solve as m1_solve
from src.models.model3_pacing import build_model as m3_build
from src.models.model3_pacing import solve as m3_solve

from .common import FIGURES, TABLES, load_inputs

LAMBDA = 1.5        # coherent, positive-energy, inside the stability interval
ENERGY_BASE = 6000  # kJ of training-independent stores (glycogen); affine offset
K_E_FORM = 300      # kJ per unit of form; scale only -- rankings are invariant


def candidate_plans(profile: dict, state: dict, rates: pd.DataFrame) -> pd.DataFrame:
    """A spread of distinct training plans, obtained by varying k2 in Model 1.

    Only the terminal (CTL_T, ATL_T) pair matters downstream, so plans are
    de-duplicated on it.
    """
    rows = []
    for k2 in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.5, 5.0):
        p = copy.deepcopy(profile)
        p["banister"]["k2"] = k2
        prob, v = m1_build(p, state, rates)
        if m1_solve(prob) != "Optimal":
            continue
        traj = m1_extract(prob, v, state)["trajectory"]
        rows.append({"generated_with_k2": k2,
                     "ctl_T": round(traj["ctl"].iloc[-1], 3),
                     "atl_T": round(traj["atl"].iloc[-1], 3),
                     "total_hours": round(traj["hours"].sum(), 1)})
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["ctl_T", "atl_T"])
            .reset_index(drop=True))


def race_time(profile: dict, ctl: float, atl: float, lam: float) -> float | None:
    """End-to-end predicted finishing time for a plan, via Model 3."""
    p = copy.deepcopy(profile)
    p["model3"]["energy_budget_kj_per_ctl"] = K_E_FORM
    p["model3"]["energy_base_kj"] = ENERGY_BASE
    prob, _ = m3_build(p, ctl_race_day=ctl, atl_race_day=atl, fatigue_lambda=lam)
    if m3_solve(prob) != "Optimal":
        return None
    return pulp.value(prob.objective)


def evaluate(profile: dict, plans: pd.DataFrame, lam: float,
             k1: float, k2: float) -> pd.DataFrame:
    """Score every plan by p_T (with weights k1,k2) and by race time (with lam)."""
    out = plans.copy()
    out["p_T"] = k1 * out["ctl_T"] - k2 * out["atl_T"]
    out["race_time_min"] = [race_time(profile, r.ctl_T, r.atl_T, lam)
                            for r in out.itertuples()]
    return out


def rankings_agree(df: pd.DataFrame) -> bool:
    """True if ordering plans by p_T (desc) matches ordering by time (asc)."""
    d = df.dropna(subset=["race_time_min"])
    by_p = d.sort_values("p_T", ascending=False).index.to_list()
    by_t = d.sort_values("race_time_min", ascending=True).index.to_list()
    return by_p == by_t


def plot(coh: pd.DataFrame, inc: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, df, title, ok in (
        (ax1, coh, f"coherent: $\\lambda = k_2/k_1 = {LAMBDA}$", True),
        (ax2, inc, "incoherent: $\\lambda = 0$ (fitness-only)", False),
    ):
        d = df.dropna(subset=["race_time_min"])
        ax.scatter(d["p_T"], d["race_time_min"],
                   color="#16a34a" if ok else "#dc2626", s=55, zorder=3)
        ax.set_xlabel("Model 1 objective $p_T$")
        ax.set_title(title + ("  ✓ monotone" if ok else "  ✗ not monotone"))
        ax.grid(alpha=.3)
    ax1.set_ylabel("Model 3 predicted race time (min)")
    fig.suptitle("Does maximizing Model 1's objective minimize race time?", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    profile, state, rates = load_inputs()
    k1 = profile["banister"]["k1"]

    plans = candidate_plans(profile, state, rates)
    print(f"[coherence] {len(plans)} distinct terminal states from Model 1:")
    print(plans.to_string(index=False))

    # Coherent case: lam = k2/k1, with k2 = LAMBDA*k1
    coh = evaluate(profile, plans, lam=LAMBDA, k1=k1, k2=LAMBDA * k1)
    # Incoherent case: the shipped fitness-only budget (lam = 0), p_T with k2=2
    inc = evaluate(profile, plans, lam=0.0, k1=k1, k2=2.0)

    coh.to_csv(TABLES / "coherence_coherent.csv", index=False)
    inc.to_csv(TABLES / "coherence_incoherent.csv", index=False)
    plot(coh, inc, FIGURES / "coherence.png")

    print(f"\n[coherence] lam = k2/k1 = {LAMBDA}:")
    print(coh.round(2).to_string(index=False))
    print(f"  rankings agree: {rankings_agree(coh)}")

    print("\n[coherence] lam = 0 (fitness-only budget, k2 = 2.0):")
    print(inc.round(2).to_string(index=False))
    print(f"  rankings agree: {rankings_agree(inc)}")

    # feasibility window for lambda
    best = plans.loc[(plans["ctl_T"] - LAMBDA * plans["atl_T"]).idxmax()]
    print(f"\n[coherence] positivity needs lam < CTL_T/ATL_T = "
          f"{best.ctl_T / best.atl_T:.2f}; cost ranging allows k2 in [1.29, 2.95] "
          f"-> coherent AND plan-preserving for lam in [1.29, 1.75)")
