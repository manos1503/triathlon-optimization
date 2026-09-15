"""Two-variable reduction of Model 3, solved graphically — Simplex appendix.

Restrict Model 3 to the run leg with two zones (Z3, Z5):

    min  t3 + t5
    s.t. 0.2317 t3 + 0.2704 t5 >= 10      (cover 10 km)
         71.3   t3 + 95.1   t5 <= 3300    (run-leg energy share, kJ)
         t3, t5 >= 0

Small enough to solve graphically and by hand with the Simplex method;
this module draws the feasible region, enumerates the vertices, and
confirms the optimum with CBC. Used by the report's Simplex appendix.

Usage:
    python -m src.analysis.simplex_toy
"""

from __future__ import annotations

import numpy as np
import pulp

from .common import FIGURES

S3, S5 = 0.2317, 0.2704      # km/min at Z3, Z5 (run leg, from thresholds)
E3, E5 = 71.3, 95.1          # kJ/min
DIST, ENERGY = 10.0, 3300.0


def solve_cbc() -> tuple[float, float, float]:
    prob = pulp.LpProblem("simplex_toy", pulp.LpMinimize)
    t3 = pulp.LpVariable("t3", lowBound=0)
    t5 = pulp.LpVariable("t5", lowBound=0)
    prob += t3 + t5
    prob += S3 * t3 + S5 * t5 >= DIST, "distance"
    prob += E3 * t3 + E5 * t5 <= ENERGY, "energy"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return t3.varValue, t5.varValue, pulp.value(prob.objective)


def vertices() -> dict[str, tuple[float, float]]:
    """Corner points of the feasible region."""
    # A: all-Z3 (distance binding, t5 = 0)
    a = (DIST / S3, 0.0)
    # B: distance & energy both binding
    m = np.array([[S3, S5], [E3, E5]])
    b = np.linalg.solve(m, np.array([DIST, ENERGY]))
    # C: all-Z5 (energy binding, t3 = 0) -- check distance feasibility
    c = (0.0, ENERGY / E5)
    return {"A (all Z3)": a, "B (mix)": tuple(b), "C (all Z5)": c}


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t3 = np.linspace(0, 50, 400)
    dist_line = (DIST - S3 * t3) / S5          # distance boundary
    energy_line = (ENERGY - E3 * t3) / E5      # energy boundary

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot(t3, dist_line, color="#2563eb", label="distance: $0.232t_3+0.270t_5=10$")
    ax.plot(t3, energy_line, color="#dc2626", label="energy: $71.3t_3+95.1t_5=3300$")

    upper = np.minimum(energy_line, 60)
    lower = np.maximum(dist_line, 0)
    feas = upper >= lower
    ax.fill_between(t3[feas], lower[feas], upper[feas], color="#bfdbfe", alpha=.6,
                    label="feasible region")

    vs = vertices()
    for name, (x, y) in vs.items():
        if y >= -1e-9 and x >= -1e-9:
            ax.plot(x, y, "ko", ms=6)
            ax.annotate(f"{name}\n$t={x+y:.1f}$ min", (x, y), fontsize=14,
                        textcoords="offset points", xytext=(8, 6))

    # objective level set through the optimum
    x_opt, y_opt = vs["B (mix)"]
    z = x_opt + y_opt
    ax.plot(t3, z - t3, "--", color="#475569", lw=1,
            label=f"objective $t_3+t_5={z:.1f}$")

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 45)
    ax.set_xlabel("$t_3$ (min at Z3)")
    ax.set_ylabel("$t_5$ (min at Z5)")
    ax.set_title("Two-zone pacing toy LP — graphical solution")
    ax.legend(fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t3, t5, z = solve_cbc()
    print(f"[simplex_toy] CBC optimum: t3={t3:.2f}, t5={t5:.2f}, total={z:.2f} min")
    for name, (x, y) in vertices().items():
        e = E3 * x + E5 * y
        feasible = e <= ENERGY + 1e-6 and S3 * x + S5 * y >= DIST - 1e-6
        print(f"   {name}: t=({x:.2f},{y:.2f})  time={x+y:.2f}  energy={e:.0f}"
              f"  {'feasible' if feasible else 'INFEASIBLE'}")
    plot(FIGURES / "simplex_toy.png")
    print("[simplex_toy] wrote results/figures/simplex_toy.png")
