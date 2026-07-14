# Candidate Race Dataset — Assumptions

`data/candidate_races.csv` is a constructed dataset with documented assumptions,
as allowed by the project proposal ("κατασκευασμένα δεδομένα με τεκμηριωμένες
παραδοχές"). It is a *scenario* for the 2027 season, not the official calendar:
costs, scores and structural conflicts are the modeled quantities; names anchor
the scenario to reality where possible.

## Verification status (checked July 2026)

**Verified — 2025 and/or 2026 edition confirmed:**

- Ironman 70.3 Kraichgau: 31 May 2026 (annual; matches R04's late-May slot)
- Ironman 70.3 Nice: 29 Jun 2025 and 28 Jun 2026; hosts the 2026 70.3 World
  Championship; official bike gain ~1367 m (matches R08's "mountainous" 1300 m)
- WTCS Hamburg: 11–12 Jul 2026, ~2500 age-group athletes (matches R06's slot)
- London T100 (age-group Olympic offered): 9–10 Aug 2025; 25–26 Jul 2026 (R10)
- Chicago Triathlon: 23–24 Aug 2025, Olympic on Sunday (matches R15 exactly)
- Ironman 70.3 Greece, Costa Navarino: 26 Oct 2025 — the author raced it (R14)
- Copenhagen: Olympic-distance racing listed for Aug 2026 (R12)
- Ironman 70.3 Dubai: recurring February race (2025 edition held; 2026
  unconfirmed at check time) — R00's early-March placement is one week off its
  usual slot
- MTRI / Mediterranean Triathlon Valencia: 2025 edition with sprint distance
  (R16); note the separate 2025 World Triathlon Cup Valencia was cancelled
- Epidavros Legacy (Whynot events, running since 2014): 2026 edition held in
  early June with Sprint/Standard/Half distances — matches R05's June slot;
  the athlete raced the 2025 Standard edition (the Model 3 validation race)
- Hellenic Championship, Olympic distance: on the federation's published 2026
  calendar (hellastriathlon.gr), Attica, venue pending — matches R13

**Active organizer, edition dates not centrally listed:** TRIMORE Syros,
TRIMORE Nafplio, TRIMORE Rethymno — all on the organizer's current
Multisports Tour (trimore.gr), with recent participant reviews on racecheck.com.

**Representative placements (no verified counterpart in that slot):**

- R11 is modeled as a flat, late-August European 70.3 with moderate travel
  cost. Its namesake (Ironman 70.3 Budapest) is real but races in **April**
  (20 Apr 2026); the true late-August alternative (Ironman 70.3 Vichy,
  23 Aug 2026) has a hillier bike (~1030 m). Read R11 as a representative
  scenario slot, not a specific event.
- The Athens season-opener sprint (R01) and Thessaloniki sprint (R07) are
  representative of the Greek domestic race scene (the federation publishes
  its calendar year by year; e.g. the real Psathathlon sprint races in
  Attica each May, and AXD Sprint in Alexandroupoli each September).

## Cost assumptions (EUR)

| Item | Assumption |
|------|-----------|
| Sprint/Olympic entry (GR) | 40–110, per Hellenic federation race norms |
| Olympic entry (international) | 140–220 |
| Ironman 70.3 entry | 320–360 (official 2025-26 price range) |
| Travel, domestic | 0 (Athens), 80–250 (islands / other cities) |
| Travel, Europe | 450–650 (flights + 2-3 hotel nights) |
| Travel, transatlantic | ~1100 |

## Value scoring

`points` (0–100): proxy for ranking value — World-Series AG events 90, 70.3s 80–85,
national champs 60, local sprints 20–25.
`strategic` (0–10): qualification/championship relevance.

Personal preference (enjoyment) is decomposed into three subjective 1–10 scores:
`organization` (event logistics, aid stations, transition setup — from race
reviews), `scenery` (views, atmosphere), `swim_quality` (water clarity,
temperature, typical chop for the venue). enjoyment = 0.3·org + 0.3·scenery +
0.4·swim (weights in profile).

## Course terrain (objective) → suitability score

`bike_elev_gain_m` / `run_elev_gain_m`: total elevation gain per leg, from
official course profiles (constructed here at realistic values — e.g. Nice 70.3
bike ≈ 1300 m is famously mountainous; Budapest ≈ 400 m is flat).

Suitability converts terrain into athlete fit. Gain per km is normalized to a
flatness score, then matched to the athlete's terrain preference per leg:

    gpk    = elev_gain_m / leg_km            (leg_km from distance class)
    flat   = 10 · (1 − min(gpk, 15)/15)      (0 = alpine, 10 = pancake)
    score  = flat if preference is 'flat', 10 − flat if 'hilly', 7 if 'any'
    suitability = 0.5·bike_score + 0.5·run_score

The athlete (strong steady-power cyclist, flat-course runner) prefers flat on
both legs, so e.g. Nice (14.4 m/km bike) scores poorly despite high enjoyment.

`weather_prob`: probability of good race-day conditions given location and month
(e.g. Aegean in May 0.85, Copenhagen late August 0.65, Dubai March 0.90).
Race value is risk-adjusted: a ruined race delivers no value.

    v_i = weather_prob_i · (w_pts·points + w_str·strategic + w_pref·preference)

## Vacation-days budget (second knapsack dimension)

`vacation_days`: days off work each race consumes — Athens 0, domestic 1–2,
Europe 3, transatlantic 5. Season cap: 12 days (profile). Together with the
money budget this makes Model 2 a two-dimensional knapsack: a race can be
cheap in euros but expensive in days (and vice versa), so the two constraints
bind differently — compare their shadow prices in the LP relaxation.

## Structural features (intentional)

- R12/R15 same weekend (Copenhagen vs Chicago) — explicit exclusion pair.
- R13 and R14 two weeks apart, both marked A-race quality — recovery constraint
  forces a choice (70.3 requires 4 recovery weeks).
- Cheap domestic races create high value-per-euro; internationals create
  budget tension — the knapsack trade-off.
- Week numbers count from 2027-01-04 (season week 1).

## A-race requirement

`a_race` flags championship-quality targets (WT Hamburg, Hellenic Champs,
Costa Navarino 70.3). A season plan must include at least one:
Σ_{i∈A} x_i ≥ 1 — a set-covering side constraint on the knapsack.

## Recovery requirements

sprint: 1 week · olympic: 2 weeks · 70.3: 4 weeks (after the race,
before the next one), per standard coaching guidance.
