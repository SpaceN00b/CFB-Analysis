# Defensive player impact scores (test bench, 2026-11-20)

Question (Josiah): use defensive players in the player impact scores (PR 35's depth-chart scores).

**Result:** a defender priced by the plays he wins (sacks, tackles for loss, interceptions, pass
breakups, hurries, at their EPA swing) over the player who fills in adds nothing measurable to
model_v2 or to PR 35's `dc_total`. Added to `dc_total`: 2026 listed +0.0001 (SE 0.0007), 2025
realized +0.0009 (SE 0.0008), pooled +0.0005 (SE 0.0005), P4/Notre Dame pooled +0.0014 (SE 0.0009)
(positive = worse). The best defensive version on its own, 2018-2024 realized with cleaned absences,
is -0.0004 (SE 0.0003) vs model_v2.

The measure does carry some signal: on 2018-2025 realized absences the residual moves 0.63 points per
point of defensive cost (SE 0.32), and 1.54 (SE 0.41) when an isolated blank box-score game is read as
played. The costs are small (mean game term 0.3 points; 10 listed-out defenders cost 0.5+ points in
week 12), so the log-loss effect stays inside noise. A version scaled by the 2018-2024 slope (x1.83,
SE 0.45) does not help 2025 or 2026 either (pooled +0.0007 vs `dc_total`, SE 0.0010). Nothing in a
model config or the week-12 picks changes.

Code: `src/cfb/lab/defense_impact.py`, `scripts/defense_impact_select.py`. Tests:
`tests/test_defense_impact.py`. Output: `data/exploration/defense_impact/` (copy in
`/mnt/project-files/reports/defense-impact/`).

```bash
python -m cfb.core.build
python scripts/impact_base.py
python scripts/injury_select.py --build
python scripts/depth_impact_select.py --week 12     # PR 35; writes this week's chart
python scripts/defense_impact_select.py --week 12   # ~1 min
```

## Data reviewed

| Source | Covers | Use here |
| --- | --- | --- |
| CFBD box score (`games_players`), defensive + interceptions | TOT, SOLO, SACKS, TFL (includes sacks), PD (excludes INTs), QB HUR, INT; every FBS game 2016 on (2014-15 INT only) | per-game disruption (larger dataset) |
| CFBD `/plays/stats` | Sack, Pass Breakup, Interception, Fumble Forced, QB Hurry per play; hurries only 2025 on and only on incompletions; no tackles | EPA value of sacks and PBUs |
| CFBD play-by-play | EPA (ppa) per play | event values |
| Sportradar 2026 per-play stats | tackles, sacks, TFL, QB hits, pass defended, 2026 only | not used: box score covers the same events for every season |
| CFBDepth exports (2026-11-18) | DL / LB / DB snaps by alignment, games, pressures, targets / catch% allowed, ratings; 2026 season totals | not point in time (cannot test on 2025); used to measure how often a defender who played has no box-score line |
| Team depth charts (68 P4 + Notre Dame) | defensive labels and depth | 2026 group (IDL / EDGE -> DL, LB, CB / S / NB -> DB); next man up for this week |

PR 26 (tackle share) and PR 34 (CFBDepth snap share, Overall rating) found no defensive cost. Tackles
mostly count how often the ball carrier reached a player, and snap share says who plays, not how well,
so this version prices disruption instead.

## Method

**Event values** (points, defense's view: mean EPA of the play type minus mean EPA of plays with the
event, 2018-2024, non-garbage scrimmage plays): sack 1.45, interception 4.80, pass breakup 0.93,
non-sack tackle for loss 1.29 (rushes for negative yards). Hurries 0.5 by assumption: `/plays/stats`
records hurries only on incompletions, so their swing is not measurable from it.

**Disruption per player-game** = 1.45 x sacks + 1.29 x (TFL - sacks) + 4.80 x INT + 0.93 x PD + 0.5 x
hurries. Involvement = tackles + PD + hurries + INT (a proxy for snaps). Groups: roster position of
that season (DL / LB / DB); for 2026 the defensive chart role where charted.

**Regulars per team-game (pregame):** DL 4, LB 3, DB 5 with the most involvement so far this season
and at least 2 appearances (`dc_total`'s rule: a starter out long enough for his backup to pass him
drops out).

**Starter rate:** (disruption this season + 0.5 x last season + K x league starter mean) / (games +
0.5 x last-season games + K). K from odd / even game split-half reliability of regulars, 2018-2024:
DL 5.9, LB 9.7, DB 24.4 games. League per-game disruption, 2018-2024: regulars DL 0.98, LB 1.18, DB
1.18; the player who filled in for an out regular, in that game: DL 0.61 (n 4,940), LB 0.76 (2,447),
DB 0.84 (5,452).

**Replacement:** `fill` = the league fill-in rate of the group. `own` = the next healthy non-regular in
the group, his disruption per involvement times the starter's involvement per game (at most 4x his
own), shrunk toward the fill rate with K. This week: the next man up at the starter's label on this
week's chart (current chart less listed players), priced by the `own` rule; a starter not on the
chart falls back to the usage pick. `count` = every regular out at the league mean (starter - fill
rate), no player value.

Cost = max(starter rate - replacement rate, 0) x P(out), against the team's margin; game term = home
cost minus away cost through model_v2's calibration slope, nothing fitted (as PR 34 / PR 35).

**Absence information:** as PR 35 (2025 realized: no box-score line in a regular-season game = out;
2026 listed: PR 34 listings, 2,404 defender rows; 2026 realized). `streak`: realized, but out only
when he also has no line in the previous or next game. Reason: CFBDepth 2026 snaps show that a
defender who played has no box-score line in 4.4% of his games at 45+ snaps per game, 11% at 30-45,
28% at 15-30, so single blank games are often not absences.

## Results

Log loss change (paired bootstrap by week, SE). Negative = better.

| Version | 2018-24 realized (n 4,943) vs v2 | 2025 realized FBS (n 762) vs v2 | vs dc_total | 2026 listed FBS (n 632) vs v2 | vs dc_total | 2026 listed P4/ND (n 353) vs v2 | vs dc_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dc_total (PR 35) | | -0.0005 (0.0031) | | -0.0082 (0.0032) | | -0.0144 (0.0050) | |
| def_fill | -0.0002 (0.0004) | +0.0009 (0.0009) | | +0.0001 (0.0007) | | +0.0001 (0.0010) | |
| def_own | +0.0000 (0.0004) | +0.0005 (0.0008) | | +0.0001 (0.0008) | | +0.0002 (0.0010) | |
| def_count | +0.0002 (0.0004) | +0.0010 (0.0012) | | -0.0004 (0.0007) | | -0.0012 (0.0009) | |
| def_fill, streak absences | -0.0004 (0.0003) | +0.0004 (0.0008) | | | | | |
| def_fill x 1.83 (2018-24 fit) | | +0.0012 (0.0015) | | +0.0003 (0.0013) | | +0.0003 (0.0018) | |
| dc_total + def_fill | | +0.0004 (0.0032) | +0.0009 (0.0008) | -0.0082 (0.0034) | +0.0001 (0.0007) | -0.0143 (0.0053) | +0.0001 (0.0010) |
| dc_total + def_own | | +0.0000 (0.0031) | +0.0005 (0.0008) | -0.0081 (0.0035) | +0.0001 (0.0007) | -0.0142 (0.0052) | +0.0002 (0.0010) |
| dc_total + def_count | | +0.0006 (0.0030) | +0.0010 (0.0012) | -0.0085 (0.0033) | -0.0003 (0.0007) | -0.0154 (0.0054) | -0.0010 (0.0009) |
| dc_total + def_fill, streak | | -0.0001 (0.0030) | +0.0003 (0.0008) | | | | |
| dc_total + def_fill x 1.83 | | +0.0006 (0.0030) | +0.0010 (0.0014) | -0.0079 (0.0037) | +0.0003 (0.0013) | -0.0141 (0.0056) | +0.0004 (0.0018) |

model_v2 log loss: 2018-2024 0.5191, 2025 0.5115, 2026 0.5453 (P4/ND 0.5326).

**Pooled, 2025 realized + 2026 listed (n 1,394; P4/ND 786):**

| Version | vs model_v2 | vs dc_total | P4/ND vs model_v2 | P4/ND vs dc_total |
| --- | --- | --- | --- | --- |
| dc_total (PR 35) | -0.0040 (0.0024) | | -0.0097 (0.0032) | |
| dc_total + def_fill | -0.0035 (0.0026) | +0.0005 (0.0005) | -0.0082 (0.0035) | +0.0014 (0.0009) |
| dc_total + def_own | -0.0037 (0.0025) | +0.0003 (0.0005) | -0.0090 (0.0034) | +0.0007 (0.0008) |
| dc_total + def_count | -0.0036 (0.0025) | +0.0004 (0.0007) | -0.0085 (0.0035) | +0.0011 (0.0009) |
| dc_total + def_fill x 1.83 | -0.0033 (0.0026) | +0.0007 (0.0010) | -0.0078 (0.0037) | +0.0019 (0.0014) |

2026 realized (FBS): dc_total -0.0062, dc_total + def_fill -0.0066 (vs dc_total -0.0004, SE 0.0011).

**Points lost per point of defensive cost** (residual vs model_v2 on the game term, in sample; 1.0 =
costs right on average):

| Version | 2018-2025 realized (games adjusted) | 2026 listed |
| --- | --- | --- |
| def_fill | 0.63 (0.32), 4,745 | 0.57 (2.16), 263 |
| def_fill, streak absences | 1.54 (0.41), 4,036 | |
| def_own | 0.59 (0.32) | 1.00 (2.17) |
| def_count | 0.53 (0.32) | 1.95 (2.09) |
| DL only | 0.02 (0.54) | -1.36 (3.49) |
| LB only | 1.86 (0.65) | -0.87 (4.00) |
| DB only | 0.66 (0.61) | 4.19 (3.81) |

Pass rushers' disruption does not price their absence (DL 0.02); linebackers' does (1.86, SE 0.65).
Player values (`fill`, `own`) and a flat count of regulars out score about the same, so the player
measure adds little over knowing that a regular is out. 2026 listings cover too few defensive games
(263) to measure a slope.

Fourteen versions on the same seasons; the scaled and streak numbers carry selection optimism.

## This week (week 12, 2026)

`def_impact_scores_week12_2026.csv`: every defensive regular for each FBS team's week 12 game: games,
involvement and disruption per game, rate, the replacement (next man up on this week's chart for the
68 charted teams, usage pick for the rest) and his rate, value over the fill rate and over that
replacement, P(out), and both costs. 1,632 regulars, 104 listed out; 10 cost at least 0.5 points
(fill). Largest: Marshall DL Braydin Ward 1.22, Texas DL Colin Simmons (Questionable) 1.03 (0.68 over
chart next man Colton Vasek), Oregon LB Teitum Tuioti 0.92 (1.10 over Cole Thomas), Indiana LB Rolijah
Hardy 0.78. Team totals: `def_team_costs_week12_2026.csv`. Game terms: 12 of 70 games move 0.5+ points,
2 move 1+ (largest 1.34). Not added to the picks.

## Limits

- Realized absences for defenders are noisy (see streak above); the 2026 listings are the clean test
  but small.
- Box scores do not show coverage or run fits; PD and INT are the only coverage signal, and they are
  noisy (DB K 24 games).
- Hurry value is an assumption (0.5).
- Model_v2's ratings already contain earlier games without the player; no discount here (`dc_total`
  has none either).
- CFBDepth exports are season totals at download, so snaps, pressures and coverage numbers cannot be
  tested on earlier weeks or 2025.
