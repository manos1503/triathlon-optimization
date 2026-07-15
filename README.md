# Triathlon Optimization

**Optimal Athletic Programming in Triathlon using Linear and Integer Programming**

## Author

| ΑΜ      | Name             | GitHub      |
| :------ | :--------------- | :---------- |
| 1100830 | Emmanouil Vichos | [manos1503] |

[manos1503]: https://github.com/manos1503

**Course:** Linear and Combinatorial Optimization
**University:** University of Patras
**Supervisor:** Prof. S. Daskalaki

---

## Overview

This project applies Linear Programming (LP) and Integer Programming (IP) techniques to optimize three interconnected decision problems in triathlon — a multidisciplinary endurance sport combining swimming, cycling, and running. The models are built on real training data (~440 activities, 15-month Strava export) and cover the full scope of the course material.

---

## Models

### Model 1 — Optimal Training Load Allocation (LP)
Optimal distribution of weekly training hours across the three disciplines (swim / bike / run) and intensity buckets (easy / moderate / hard) over a 16-week macrocycle.

- **Objective:** Maximize predicted race-day performance (Banister fitness–fatigue model, linearized)
- **Key constraints:** Maximum weekly hours, discipline minimums, 80/20 polarization, physiological fatigue limits (ATL), ramp-rate guard, macrocycle phases (base, build, peak, taper)

### Model 2 — Optimal Race Calendar Selection (0-1 MIP)
Binary selection of races within a competitive season, balancing race value against cost and recovery constraints.

- **Objective:** Maximize total season "value" (ranking points, strategic importance, preference)
- **Variables:** xᵢ ∈ {0,1} for each candidate race i
- **Key constraints:** Total budget (entry fees + travel), maximum races per month, minimum recovery weeks between races, fitness-readiness link from Model 1
- **Method:** Branch & Bound; LP relaxation vs. integer optimum compared

### Model 3 — Optimal Race Pacing Strategy (LP with Linearization)
Optimal distribution of time across intensity zones per race leg, minimizing total finishing time.

- **Objective:** Minimize total race time
- **Key constraints:** Total energy budget (from Model 1 fitness state), maximum intensity per leg, bike→run metabolic fatigue carry-over
- **Linearization:** Non-linear speed/energy relationship discretized via intensity zones (Z1–5)
- **Extension:** Dual theory — shadow prices interpreted physiologically (e.g. "how much faster if FTP increases by 5W?")

---

## Model Pipeline

```
Model 1 (LP)     →     Model 2 (IP)     →     Model 3 (LP)
Training Load          Race Calendar          Pacing Strategy
Allocation             Selection              Optimization
(fitness level)        (which races?)         (how to race?)
```

---

## Course Topics Covered

- LP and IP Modeling
- Simplex Algorithm (Models 1 & 3)
- Dual Theory & Sensitivity Analysis (Model 3)
- Branch & Bound (Model 2)
- Network Flow (Model 2 extension)

---

## Repository Structure

```
data/raw/          Strava export (not committed)
data/processed/    Cleaned activities, TRIMP, CTL/ATL series
docs/              Mathematical formulations, athlete profile
src/pipeline/      Strava → TRIMP → CTL/ATL data pipeline
src/models/        PuLP implementations of Models 1–3
src/analysis/      Sensitivity analysis and scenario runs
results/           Figures and tables for the report
report/            Final report (10–15 pages)
presentation/      Slides (~20 min)
```

---

## Data Sources

- Personal training data: Garmin Connect / Strava export
- Race calendar: Hellenic Triathlon Federation & World Triathlon
- Physiological parameters: Sports science literature

---

## Setup

```bash
pip install -r requirements.txt
```

Python ≥3.10. Optimization via PuLP/CBC; data handling via pandas; plots via matplotlib.

## Reproduce everything

```bash
python -m src.run_all    # pipeline -> Models 1-3 -> all analyses and figures
python -m pytest tests/  # 44 tests
```

Without the personal Strava export (`data/raw/activities.csv`, not committed),
the pipeline step is skipped and the committed processed data is used.

## Documentation

- [`docs/formulations.md`](docs/formulations.md) — complete mathematical formulation of all three models
- [`docs/athlete_profile.yaml`](docs/athlete_profile.yaml) — athlete-specific parameters used throughout
