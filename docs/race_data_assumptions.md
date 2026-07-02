# Candidate Race Dataset — Assumptions

`data/candidate_races.csv` is a constructed dataset with documented assumptions,
as allowed by the project proposal ("κατασκευασμένα δεδομένα με τεκμηριωμένες
παραδοχές"). Race names mix real recurring events (Challenge Heraklion,
Ironman 70.3 Kraichgau/Nice/Budapest, WT Hamburg) with representative domestic
races; 2027 dates are plausible placements based on each event's usual slot.

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

Personal preference is decomposed into three subjective 1–10 scores rather than
one opaque number: `course_quality` (technical interest, road surface, profile),
`scenery` (views, atmosphere), `swim_quality` (water clarity, temperature, chop).
preference = 0.4·course + 0.3·scenery + 0.3·swim (weights in profile).

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

## Recovery requirements

sprint: 1 week · olympic: 2 weeks · 70.3: 4 weeks (after the race,
before the next one), per standard coaching guidance.
