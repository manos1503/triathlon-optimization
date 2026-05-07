# Triathlon Optimization
### Optimal Athletic Programming in Triathlon using Linear and Integer Programming

## Author
| ΑΜ      | Name             | GitHub      |
| :------ | :--------------- | :---------- |
| 1100830 | Emmanouil Vichos | [manos1503] |

[manos1503]: https://github.com/manos1503

**Course:** Linear and Combinatorial Optimization  
**University:** University of Patras

---

## Overview
This project applies Linear Programming (LP) and Integer Programming (IP) 
techniques to optimize three interconnected decision problems in triathlon — 
a multidisciplinary endurance sport combining swimming, cycling, and running.

The triathlon's triple nature naturally generates three optimization problems
that are modeled and solved within the framework of Linear and Integer 
Programming, covering the full scope of the course material.

---

## Models

### Model 1 — Optimal Training Load Allocation (LP)
Optimal distribution of weekly training hours across the three disciplines 
(swim / bike / run) and training types (aerobic base, high intensity, recovery)
throughout the annual training cycle (macrocycle).

- **Objective:** Maximize estimated athletic performance
- **Key constraints:** Maximum weekly hours, load/recovery ratios, 
  physiological fatigue limits (ATL), macrocycle phases (base, build, peak, taper)
- **Performance model:** Banister CTL/ATL/TSB model (linearized)

### Model 2 — Optimal Race Calendar Selection (0-1 MIP)
Binary selection of races within a competitive season, balancing race value 
against cost and recovery constraints.

- **Objective:** Maximize total season "value" (ranking points, strategic importance)
- **Variables:** xᵢ ∈ {0,1} for each candidate race i
- **Key constraints:** Total budget (entry fees + travel), maximum races per month,
  minimum recovery weeks between races, connection to fitness level from Model 1

### Model 3 — Optimal Race Pacing Strategy (LP with Linearization)
Optimal distribution of energy and time across the three race legs and intensity
zones, minimizing total race completion time.

- **Objective:** Minimize total race time
- **Key constraints:** Total energy budget (from Model 1 fitness level), 
  maximum intensity per leg, metabolic fatigue carry-over between legs
- **Linearization:** Non-linear speed/energy relationship approximated via 
  discrete intensity zones (Zone 1–5), enabling LP formulation
- **Extension:** Dual theory and sensitivity analysis — shadow prices interpreted 
  physiologically (e.g. "how much faster if FTP increases by 5W?")

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

## Data Sources
- Personal training data: Garmin Connect / Strava export
- Race calendar: Hellenic Triathlon Federation & World Triathlon
- Physiological parameters: Sports science literature

---

## Tools
- Python
- PuLP (LP/IP solver)
- pandas (data handling)
- matplotlib (visualization)

---


