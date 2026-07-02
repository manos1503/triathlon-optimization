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
`preference` (0–10): athlete's personal interest.

Race value in Model 2: v_i = w_pts·points + w_str·strategic + w_pref·preference
with weights in `athlete_profile.yaml`.

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
