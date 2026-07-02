# Mathematical Formulations

Complete formulation of the three models. Notation is shared across models; the pipeline is sequential: Model 1 → fitness state → Models 2 and 3.

---

## Model 1 — Optimal Training-Load Allocation (LP)

### Sets

| Symbol | Definition |
|--------|-----------|
| $D = \{\text{swim}, \text{bike}, \text{run}\}$ | disciplines |
| $B = \{\text{easy}, \text{mod}, \text{hard}\}$ | intensity buckets (easy = Z1–2, mod = Z3, hard = Z4+) |
| $W = \{1, \dots, T\}$ | weeks of the macrocycle ($T = 16$ for Olympic distance) |

### Parameters

| Symbol | Definition |
|--------|-----------|
| $r_{db}$ | TRIMP per hour of discipline $d$ at bucket $b$ (from HR data) |
| $H^{\max}_w$ | weekly hour ceiling (phase-dependent; baseline 12 h) |
| $h^{\min}_d$ | minimum weekly hours in discipline $d$ |
| $\pi$ | maximum fraction of hours at mod+hard intensity (polarized rule, $\pi = 0.2$) |
| $\lambda_c = e^{-7/42}$, $\lambda_a = e^{-7/7}$ | weekly Banister decay factors (CTL, ATL) |
| $\text{CTL}_0, \text{ATL}_0$ | initial fitness/fatigue state (from data pipeline) |
| $A^{\max}$ | ATL ceiling (overtraining guard) |
| $\delta$ | maximum weekly load ramp rate (e.g. 0.10) |
| $k_1, k_2$ | Banister performance weights ($k_2 > k_1$) |

### Decision variables

$h_{dbw} \ge 0$ — hours of discipline $d$ at bucket $b$ in week $w$ ($3 \times 3 \times 16 = 144$ variables).

Auxiliary (defined by equality constraints, so the model stays LP):

$$L_w = \sum_{d \in D} \sum_{b \in B} r_{db}\, h_{dbw} \qquad \text{(weekly TRIMP load)}$$

$$\text{CTL}_w = \lambda_c\, \text{CTL}_{w-1} + (1-\lambda_c)\, L_w, \qquad
\text{ATL}_w = \lambda_a\, \text{ATL}_{w-1} + (1-\lambda_a)\, L_w$$

Exponential smoothing is linear in the loads $L_w$, so the Banister dynamics enter the LP exactly — no approximation is needed beyond the weekly (rather than daily) time step.

### Objective

Maximize predicted race-day performance (Banister fitness–fatigue model):

$$\max \; p_T = k_1\, \text{CTL}_T - k_2\, \text{ATL}_T$$

> **Design note.** Maximizing TSB alone is degenerate (optimum = no training). The
> Banister form rewards accumulating CTL early — whose contribution persists via slow
> decay — while ATL from early weeks vanishes by race day. The optimal solution
> naturally exhibits a build-then-taper shape, which the model *discovers* rather than
> has imposed.

### Constraints

$$\sum_{d,b} h_{dbw} \le H^{\max}_w \quad \forall w \qquad \text{(weekly hours)}$$

$$\sum_{b} h_{dbw} \ge h^{\min}_d \quad \forall d, w \qquad \text{(discipline minimums)}$$

$$\sum_{d} \left( h_{d,\text{mod},w} + h_{d,\text{hard},w} \right) \le \pi \sum_{d,b} h_{dbw} \quad \forall w \qquad \text{(80/20 polarization)}$$

$$\text{ATL}_w \le A^{\max} \quad \forall w \qquad \text{(fatigue ceiling)}$$

$$L_w \le (1+\delta)\, L_{w-1} \quad \forall w \ge 2 \qquad \text{(ramp-rate guard)}$$

$$L_w \le \theta_w \cdot \bar{L} \quad \forall w \in \text{taper} \qquad \text{(taper volume reduction, } \theta_w < 1\text{)}$$

where $\bar{L} = \frac{1}{|W_{\text{build}}|} \sum_{w \in W_{\text{build}}} L_w$ is the average build-phase load (linear expression, so the constraint remains linear).

Phase structure (base/build/peak/taper) enters through phase-dependent $H^{\max}_w$ and $\theta_w$.

### Outputs consumed downstream

$\text{CTL}_w, \text{ATL}_w, \text{TSB}_w = \text{CTL}_{w-1} - \text{ATL}_{w-1}$ for every week → Models 2 and 3.

---

## Model 2 — Race-Calendar Selection (0-1 MIP)

### Sets and parameters

| Symbol | Definition |
|--------|-----------|
| $R$ | candidate races; race $i$ has week $t_i$, distance class $c_i \in \{\text{sprint}, \text{oly}, 70.3\}$ |
| $v_i = \alpha\, \text{pts}_i + \beta\, \text{strat}_i + \gamma\, \text{pref}_i$ | race value (weighted score) |
| $\kappa_i$ | total cost (entry fee + travel) |
| $\mathcal{B}$ | season budget |
| $M$ | max races per month; $\text{month}(i)$ maps race to month |
| $\rho_c$ | minimum recovery weeks after a race of class $c$ |
| $E \subseteq R \times R$ | explicit exclusion pairs (geographic/scheduling conflicts) |
| $\text{TSB}^{\text{req}}_c$ | minimum form required to race class $c$ (from Model 1 output) |

### Decision variables

$x_i \in \{0, 1\}$ — race $i$ is entered.

### Formulation

$$\max \sum_{i \in R} v_i\, x_i$$

subject to

$$\sum_{i} \kappa_i\, x_i \le \mathcal{B} \qquad \text{(budget — knapsack core)}$$

$$\sum_{i : \text{month}(i) = m} x_i \le M \quad \forall m \qquad \text{(monthly cap)}$$

$$x_i + x_j \le 1 \quad \forall\, i \ne j \text{ with } 0 \le t_j - t_i < \rho_{c_i} \qquad \text{(recovery spacing)}$$

$$x_i + x_j \le 1 \quad \forall (i,j) \in E \qquad \text{(exclusions)}$$

$$x_i = 0 \quad \text{if } \text{TSB}_{t_i} < \text{TSB}^{\text{req}}_{c_i} \qquad \text{(fitness-readiness link, pre-filter from Model 1)}$$

Solved by Branch & Bound (CBC). Analysis compares the LP relaxation bound with the integer optimum and reports the integrality gap.

> **Design note.** The fitness link is a *sequential* decomposition: Model 1 is solved
> first and its trajectory is treated as data here. A joint model would be a larger MIP;
> the decomposition is discussed as a limitation/extension in the report.

---

## Model 3 — Race Pacing Strategy (LP)

### Sets and parameters

| Symbol | Definition |
|--------|-----------|
| $\Lambda = \{\text{swim}, \text{bike}, \text{run}\}$ | race legs, with fixed distances $d_\ell$ |
| $Z = \{1, \dots, 5\}$ | intensity zones |
| $s_{\ell z}$ | sustainable speed in leg $\ell$ at zone $z$ (from threshold data: FTP, pace, LTHR) |
| $e_{\ell z}$ | energy cost per unit time in leg $\ell$ at zone $z$ (kJ/min) |
| $E^{\text{tot}}$ | total energy budget, $E^{\text{tot}} = g(\text{CTL}_T)$ from Model 1 |
| $\rho_\ell$ | max fraction of leg $\ell$ time spent at zones 4–5 |
| $\varphi$ | run-speed penalty per unit of high-intensity bike energy (fatigue carry-over) |
| $\tau_1, \tau_2$ | fixed transition times |

### Decision variables

$t_{\ell z} \ge 0$ — time spent in zone $z$ during leg $\ell$.

The nonlinear speed–energy curve is **linearized by discretization**: within each zone, speed and energy rate are constants, so distance and energy are linear in the $t_{\ell z}$. The athlete's continuous pacing choice becomes a mix over zones.

### Objective

$$\min \; \sum_{\ell \in \Lambda} \sum_{z \in Z} t_{\ell z} \; + \; \tau_1 + \tau_2$$

### Constraints

Distance coverage (with bike→run fatigue coupling):

$$\sum_{z} s_{\text{swim},z}\, t_{\text{swim},z} \ge d_{\text{swim}}, \qquad
\sum_{z} s_{\text{bike},z}\, t_{\text{bike},z} \ge d_{\text{bike}}$$

$$\sum_{z} s_{\text{run},z}\, t_{\text{run},z} - \varphi \underbrace{\sum_{z \ge 4} e_{\text{bike},z}\, t_{\text{bike},z}}_{\text{high-intensity bike energy } G} \ge d_{\text{run}}$$

Energy budget:

$$\sum_{\ell, z} e_{\ell z}\, t_{\ell z} \le E^{\text{tot}}$$

Per-leg intensity caps:

$$\sum_{z \ge 4} t_{\ell z} \le \rho_\ell \sum_{z} t_{\ell z} \quad \forall \ell$$

### Duality and sensitivity analysis

The shadow prices of this LP have direct physiological meaning:

- **Energy budget dual** $y_E$: seconds of race time saved per additional kJ available — i.e., the marginal value of fitness. Parametric analysis on $E^{\text{tot}}(\text{FTP})$ answers *"how much faster per +5 W of FTP?"*
- **Run-distance dual**: marginal cost of the run leg, revealing whether the run or bike leg is the binding bottleneck.
- **Intensity-cap duals** $y_{\rho_\ell}$: value of being able to sustain more time at high intensity in each leg.

---

## Experiment plan (per assignment: vary size and parameters)

1. **Base case** — Olympic distance, real personal data, $T=16$, 12 h/week.
2. **Distance sensitivity** — re-solve Model 1 for Sprint ($T=12$, lower volume) and 70.3 ($T=20$, higher volume); compare optimal allocations.
3. **Constraint sensitivity** — $H^{\max} \in \{10, 12, 14\}$; vary $A^{\max}$; vary $\mathcal{B}$ in Model 2.
4. **Methodological** — LP relaxation vs. MIP (Model 2); optimal plan vs. 80/20 heuristic (Model 1); Model 3 shadow prices vs. coaching intuition.
