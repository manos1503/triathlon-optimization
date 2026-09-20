# Triathlon Optimization

**Optimal Resource Allocation and Scheduling of Athletic Activities, with an Application to Triathlon**

*Linear and 0–1 Integer Programming models for training load, race calendar and race pacing*

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

This project applies Linear Programming (LP) and Integer Programming (IP) techniques to optimize three interconnected decision problems in triathlon — a multidisciplinary endurance sport combining swimming, cycling, and running. The models are built on real training data (574 activities from a 32-month Strava export) and cover the full scope of the course material.

---

## Models

### Model 1 — Optimal Training Load Allocation (LP)
Optimal distribution of weekly training hours across the three disciplines (swim / bike / run) and intensity buckets (easy / moderate / hard) over a 16-week macrocycle.

- **Objective:** Maximize predicted race-day performance (Banister fitness–fatigue model, linearized)
- **Key constraints:** Maximum weekly hours, discipline minimums, 80/20 polarization, physiological fatigue limits (ATL), ramp-rate guard, macrocycle phases (base, build, peak, taper)

### Model 2 — Optimal Race Calendar Selection (0-1 MIP)
Binary selection of races within a competitive season, balancing race value against cost and recovery constraints.

- **Objective:** Maximize weather-risk-adjusted season "value" (ranking points, strategic importance, enjoyment, course suitability)
- **Variables:** xᵢ ∈ {0,1} for each of 17 candidate races
- **Key constraints:** Two-dimensional knapsack (money budget + vacation days), max races per month, recovery spacing between races, ≥1 A-race (set covering), fitness-readiness gate from Model 1
- **Method:** Branch & Bound; LP relaxation compared (5.04% integrality gap), tightened to 3.38% with clique cuts

### Model 3 — Optimal Race Pacing Strategy (LP with Linearization)
Optimal distribution of time across intensity zones per race leg, minimizing total finishing time.

- **Objective:** Minimize total race time
- **Key constraints:** Total energy budget (from Model 1 fitness state), maximum intensity per leg, bike→run metabolic fatigue carry-over
- **Linearization:** Non-linear speed/energy relationship discretized via intensity zones (Z1–5)
- **In-race fuelling:** carbohydrate intake as a fixed rate or as a per-leg decision variable bounded by gut absorption (stays an LP)
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

- LP and IP modelling; standard integer-modelling templates (either–or, k-of-N, set covering, variable fixing)
- Simplex algorithm (CBC; worked two-variable example in the report appendix)
- Duality: explicit primal–dual pair, shadow-price interpretation, strong duality and complementary slackness verified numerically
- Sensitivity analysis: RHS ranging **and** objective-coefficient ranging (basic / non-basic cases)
- Branch & Bound, LP-relaxation gaps, clique cuts (valid inequalities)
- Network models: longest path in a DAG, total unimodularity, Network Simplex structure

---

## Validation

Model 3 is validated against three of the author's real races across the full
distance spectrum — Spetsathlon 2026 (sprint), Epidavros 2025 (Olympic) and
Ironman 70.3 Costa Navarino 2025 — with race-day fitness reconstructed from the
data pipeline. Swim predictions land within 0.4 min in all three; the calibrated
in-race fuelling rate independently recovers the standard 75 g carbs/hour
guideline.

---

## Repository Structure

```
data/raw/          Strava export (not committed)
data/processed/    Cleaned activities, TRIMP, CTL/ATL series
docs/              Mathematical formulations, athlete profile
src/pipeline/      Strava → TRIMP → CTL/ATL data pipeline
src/models/        PuLP implementations of Models 1–3
src/analysis/      Sensitivity analysis and scenario runs
src/run_all.py     One-command reproduction of every result
tests/             60 tests (models, duality theorems, analyses)
results/           Figures, tables and captured solver output
report/            Final report (15 pp + appendices with code and output)
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
python -m pytest tests/  # 60 tests
```

Building the documents:

```bash
cd report       && pdflatex report.tex && pdflatex report.tex
cd presentation && pdflatex presentation.tex && pdflatex presentation.tex
```

The deck can also be built with the speaker notes shown. The notes live in
`presentation/speaker-notes.tex`; they are in Greek, are a personal rehearsal
aid, and are not tracked, so the deck compiles with or without them:

```bash
cd presentation && xelatex -jobname=presentation-notes '\def\shownotes{}\input{presentation}'
```

Without the personal Strava export (`data/raw/activities.csv`, not committed),
the pipeline step is skipped and the committed processed data is used.

## Documentation

- [`docs/formulations.md`](docs/formulations.md) — complete mathematical formulation of all three models
- [`docs/athlete_profile.yaml`](docs/athlete_profile.yaml) — athlete-specific parameters used throughout
