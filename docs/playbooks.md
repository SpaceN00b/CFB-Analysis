# Team playbooks (test bench)

`cfb.lab.playbooks` is an experiment: no feature group, model or weekly run reads it. It builds
rough team playbooks from the detailed stat-crew play text that `cfb.core.playtext` parses (every
game from 2025 week 9 on), joined to the players and positions the player maps use. The default
scope is `P4`: ACC, Big Ten, Big 12 and SEC plus Notre Dame, the 68 teams with depth charts.
`--conference SEC` (or any conference name) builds one conference.

```bash
python -m cfb.core.build && python -m cfb.core.playtext   # core tables and play_detail / play_players
python -m cfb.depth_charts ...                             # or restore data/tables/depth_charts from the project snapshot
python -m cfb.lab.player_maps --conference P4              # optional: style, defender-role and injury labels
python -m cfb.lab.playbooks                                # about 30 s; --season, --conference (default P4), --player-maps, --out
python scripts/playbook_check.py --ridge 5,50,200          # second stage on model_v2 (needs data/predictions/model_v2); --scope P4
```

Output in `data/lab/`: `playbooks_<season>_<scope>.json` and `.html` (the viewer,
`src/cfb/lab/playbooks_viewer.html` with the JSON embedded; team list grouped by conference, compare
table filtered by conference).

## What the text has and does not have

| has | does not have |
| --- | --- |
| "Shotgun" and "No Huddle" before the play (no word = under center, or the crew did not write it) | personnel groupings, formation names (trips, empty, pistol, I), motion |
| run direction, pass depth and direction, air yards | play action, RPOs (Sportradar's flags are false on every 2026 play), run schemes |
| ball carrier / intended receiver, with roster position | designed QB runs versus scrambles (both are a QB rush) |
| tacklers, sacks, hurries, breakups | defensive fronts and coverages |

2026 scan of all play text: "Pistol", "Under Center", "Wildcat", "Empty", "Trips", "play action",
"RPO", "screen" and "scramble" never appear. So a "play" is formation x play type x direction x
position of the ball carrier or target. "Screen / behind LOS" is any pass caught at or behind the line.

## Stat-crew effects

Per (game, offense) shares over 2025 week 9 on and 2026 (2,298 rows with 30+ plays), weighted R²:

| share | offense only | venue only | both |
| --- | ---: | ---: | ---: |
| shotgun | 0.67 | 0.26 | 0.70 |
| no huddle | 0.59 | 0.44 | 0.73 |

No huddle depends heavily on the home crew (median home-away gap for a team 10.6 points, 90th
percentile 29). The viewer's adjusted share is the offense effect of a weighted two-way fit
(offense + venue, ridge 1) over every FBS game of the season, at an average venue. Under center
gets the same adjustment, which moves it little.

## Per team

- Formation: shotgun / under-center shares, run rate and success in each, against FBS.
- Pass rate over expected: dropback share minus an FBS logistic model on down, distance, field
  position, score, quarter and clock (fit on the season's FBS plays).
- Play-type mix: RB run left / middle / right, QB run, WR/TE run, screen or behind the line, short
  1-9, intermediate 10-19, deep 20+, sack; share, success and EPA against FBS.
- Situations: 1st and 10, 2nd and 1-3 / 4-7 / 8+, 3rd-4th and 1-2 / 3-6 / 7+, red zone, goal to go
  inside the 5, last 2:00 of a half.
- Most-called plays (8+ plays, sacks and throwaways left out) with the player who gets the ball most.
- Ball carriers and targets (10+ touches) with the player-map style label.
- Defense: success allowed by play type (shrunk toward FBS with 20 plays), sack and hurry rates per
  dropback, most active defenders with the player-map role model's role.

Garbage time, kneels, spikes and no-plays are left out of everything.

Depth charts and injuries: receivers who are on their team's depth chart (cfb.depth_charts, downloaded
2026-11-19) are labelled by alignment (X, Z, Slot) instead of WR, so the play mix and top plays read
"Short 1-9 left to Slot". The player and defender tables show each player's chart position and string
(e.g. "WR-X 1", "MLB 2") and his status on the CFBDepth injury list (2026-11-18), which the 68-team
player maps attach. Defenders on teams the player maps do not cover come from the play credits, with
no role-model columns.

2026 SEC (weeks 1-11): Georgia (19%) and Alabama (13%) are the only offenses under center on more
than 8% of plays (FBS 6%); under center, SEC teams run 67% to 100% of the time. Crew-adjusted no
huddle ranges from 27% (Kentucky) and 28% (Alabama) to 98-99% (LSU, Mississippi State). Pass rate
over expected: Alabama +7.6, Ole Miss +6.7, LSU +6.1 points; Tennessee -6.1, Florida -4.1.

2026 P4 + Notre Dame (68 teams, weeks 1-11): 13 offenses are under center on more than 8% of plays
(FBS 6%); crew-adjusted: Iowa 30%, Minnesota 27%, Ohio State 23%, Northwestern 21%. Crew-adjusted no
huddle runs from Wisconsin 9% and North Carolina 10% to Mississippi State 99%, LSU 98%, USC 97%.
Pass rate over expected: Washington +11.8, Arizona +9.9, California +8.5 points; West Virginia -17.0
(also the most QB runs, 26% of plays), Tennessee, Michigan and Kansas -6.

## Does it help the model? (2026-11-18)

`scripts/playbook_check.py`: nine columns as home minus away (under center, raw no huddle, pass rate
over expected, screen, deep, QB-run share, play-mix entropy, defense sack and hurry rates), each from
the team's earlier games only (2025 plays weighted 0.5 in 2026, shrunk toward FBS with 150 plays,
situation model fit on 2025 only). Rows: 900 FBS vs FBS games, 2025 week 10 on (275) and 2026 (625).
Second stage: logistic regression with offset logit(model_v2 p_home_win), no intercept, ridge on
standardised columns, refit before every 2026 week. Paired SE over weeks.

| ridge | 2026 FBS log loss change (n 625) | SE | SEC games (n 89) | SE |
| ---: | ---: | ---: | ---: | ---: |
| 5 | +0.0120 | 0.0036 | +0.0150 | 0.0106 |
| 50 | +0.0061 | 0.0023 | +0.0057 | 0.0053 |
| 200 | +0.0025 | 0.0012 | +0.0016 | 0.0023 |
| no huddle only, 50 | +0.0006 | 0.0011 | +0.0002 | 0.0065 |

model_v2 alone: 0.5454 (FBS), 0.5567 (SEC games).

Same check reported for the 68 teams (2026-11-19, `--scope P4`; the columns were already built for
every FBS team, so only the subsets are new):

| ridge | games with a P4 or Notre Dame team (n 351) | SE | P4 vs P4 (n 271) | SE |
| ---: | ---: | ---: | ---: | ---: |
| 5 | +0.0090 | 0.0046 | +0.0117 | 0.0053 |
| 50 | +0.0038 | 0.0023 | +0.0050 | 0.0027 |
| 200 | +0.0012 | 0.0010 | +0.0017 | 0.0012 |

model_v2 alone: 0.5324 and 0.6294. No-huddle slope on the 68-team games: -1.09 points per SD (t -1.5). One-column slopes of the margin residual (home
margin minus model_v2 pred_margin), all 900 rows: no huddle -1.41 points per SD (t -2.69), under
center +0.73 (t 1.39), every other |t| below 0.7. The no-huddle slope does not hold out of sample and
no huddle is crew-dependent. Conclusion: playbook tendencies, as far as this text records them, add
nothing to model_v2. This matches W4 (pure style columns about 0; docs/matchup.md).

## Kneels in the model (2026-11-18)

CFBD types kneels as `Rush` and gives them a PPA, so `core.pbp` counts them as scrimmage rushes:
every rush and offense feature (unit EPA and success, run rate, pace) includes them unless the play
is in garbage time. No feature on main removes them. The detailed-text feature groups on the player
maps branch drop them through `play_detail.kneel`.

| season | kneels found in text | likely kneels (heuristic) |
| --- | ---: | ---: |
| 2014-2019 | 0-1 a season | about 1,100 a season |
| 2020-2021 | 7, 22 | 778, 1,003 |
| 2022-2024 | 317, 617, 637 | 1,113-1,405 |
| 2025, 2026 | 679, 876 | 1,517, 1,196 |

Text marks kneels ("Kneel down by ...") only from 2022, and in part until 2025 week 9. Heuristic:
rush for 0 or fewer yards in the last 2:30 of the 2nd or 4th quarter by a leading team (or a tied
team before the half). On detailed games it finds 86% of text kneels, and 61-65% of its plays are
text kneels. Detailed games average 1.1-1.4 kneels.

2026 effect (non-garbage scrimmage plays): kneels are 0.77% of plays (672 of 87,459), PPA -0.91
against +0.12 for other rushes, success 0.1%. Removing them raises a team's offensive EPA per play
by 0.008 on average (FBS SD across teams 0.006, max 0.028; SD of team offensive EPA 0.10). The bias
correlates with offensive quality (r 0.28), so it slightly understates good offenses and, on the
other side, flatters the defenses that face kneels. SEC 2026: Alabama +0.027 EPA/play and +2.1
points of rush success with kneels removed (12 kneels), Texas A&M +0.001 (1).

Not tested: rebuilding the strength features without kneels and re-scoring model_v2. A consistent
fix for 2014-2021 needs the heuristic, since the text has no kneels there.

## Kneel-removal test and starter games (2026-11-19)

Josiah's rule: a CFBD `Rush` of -2 to -6 yards with under 1:00 on the game clock (2nd or 4th
quarter) while the offense leads is a kneel; drop it. Against the text kneels in detailed games
(895), it is right 91.5% of the time but finds 25% of them: most kneels lose 1 yard (475) or 2
(287). A widened rule (-1 to -6 yards) is right 91.6% and finds 61%. Both were tested.

`scripts/kneel_experiment.py` builds a separate data folder with those plays dropped before
`core.pbp` and `team_game` (3,654 plays 2014-2026 for the rule, 6,698 widened), then unit ratings,
`str_power`, `str_power2` and `str_program` were rebuilt (about 10 minutes each) and model_v2 re-run
(walk-forward 2026, season-forward 2018-2025). The baseline folder is built the same way with
nothing dropped.

| games | n | baseline log loss | rule | widened |
| --- | ---: | ---: | ---: | ---: |
| 2026 with a starter-affected team | 274 | 0.5911 | 0.5911 (+0.0000, SE 0.0001) | 0.5911 (+0.0000) |
| 2026 SEC starter-affected | 33 | 0.6464 | 0.6466 (+0.0002, SE 0.0004) | 0.6463 (-0.0001) |
| 2026 other games | 358 | 0.5100 | 0.5100 | 0.5099 |
| 2026 all | 632 | 0.5452 | 0.5451 (-0.0000) | 0.5451 (-0.0000) |
| 2018-2025 season-forward | 6,024 | 0.5255 | +0.00004 (SE 0.00003) | +0.00002 |

Win probabilities move 0.04-0.06 points on average (max 0.4), predicted margins 0.04-0.06 points.
Kneels only enter the EPA and success unit columns; the points-based power ratings carry the model.

`scripts/starter_response.py` sorts the 2026 games. Regulars come from a team's last 4 earlier games:
QB1 (70%+ of dropbacks), RB1 (30%+ of carries), WR1/WR2 (15%/10%+ of targets), TE1 (8%+), and five
defenders (6%+ of defensive credits). In a game a regular is out (no stat line) or reduced (under 40%
of his usual share, garbage time left out), or listed Out on the conference availability report
(ACC, Big Ten, Big 12, SEC conference games; 170 listings matched to a regular). A team-game is
affected when QB1 is out or reduced, 2+ offensive regulars are, or a regular is listed. A TE or
defender with no stat line may still have played; there are no snap counts.

| affected team-games (2026) | n | margin minus model_v2 | offense EPA/play vs expected | success |
| --- | ---: | ---: | ---: | ---: |
| QB1 out or reduced | 155 | -2.48 (SE 1.16) | -0.054 | -1.8 pts |
| 2+ offensive regulars | 102 | -2.34 (SE 1.58) | -0.049 | -2.1 pts |
| only listed on the report | 75 | -1.00 (SE 2.05) | -0.016 | 0.0 |
| not affected | 552 | +1.02 (SE 0.67) | -0.016 | -0.3 pts |

How teammates responded (same-position teammate with the largest share gain):

| regular | n | his usual share | replacement usual -> game share | other teammates' gain | EPA: regular usual / replacement |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB1 out | 124 | 91% | 31% -> 96% | +2 pts | 0.106 / 0.071 per dropback |
| QB1 reduced | 29 | 92% | 23% -> 74% | +3 pts | 0.106 / -0.037 |
| RB1 out | 68 | 45% | 17% -> 47% | +9 pts | 0.139 / 0.015 per carry |
| WR1 out | 49 | 27% | 11% -> 29% | +9 pts | 0.537 / 0.349 per target |
| WR2 out | 86 | 18% | 11% -> 27% | +8 pts | 0.351 / 0.448 per target |
| TE1 out | 103 | 13% | 5% -> 10% | +1 pt | 0.365 / 0.258 per target |

The backup QB takes nearly all dropbacks; a missing RB1's carries go mostly to one back, with less
efficiency; a missing WR1's targets spread over the next receiver and others. Rows, SEC examples and
CSVs: /mnt/project-files/exploration/kneels/.

## 68 teams, depth charts and the CFBDepth injury list (2026-11-19)

Josiah asked to use the larger dataset and expand from the SEC to all 68 teams (P4 + Notre Dame).
`scripts/starter_response.py --scope P4` (default) now reports the 68 teams apart from all FBS.

Injury listings: the CFBDepth player-updates list (downloaded 2026-11-18; 1,449 rows, 137 FBS
teams, non-conference games and Notre Dame included; 1,445 matched to a roster id) is the first
source. It is a current-status list, so a listing covers games from its Last Update date to the
download, plus each "Official report <date>: <status>. Game <date>" entry in its update text
(1,654 windows). The conference availability reports (HDI, conference games only) fill the
gaps: Nebraska, which the list lacks, and earlier weeks, which a current-status list cannot hold.
For the 68 teams, 363 regulars were out (no stat line); 63 (17%) are listed with the list alone
(`--list-only`), 175 (48%) with the HDI gap fill.

| team-games (2026) | 68 teams n | margin minus model_v2 | off. EPA/play vs exp. | all FBS n | margin minus model_v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB1 out or reduced | 65 | -3.73 (SE 1.92) | -0.069 | 155 | -2.48 (SE 1.16) |
| 2+ offensive regulars | 49 | -0.13 (SE 2.20) | -0.047 | 102 | -2.34 (SE 1.58) |
| only listed injured | 90 | -1.06 (SE 1.84) | -0.034 | 124 | -1.15 (SE 1.47) |
| not affected | 221 | +1.23 (SE 1.04) | -0.003 | 505 | +1.27 (SE 0.71) |

How teammates responded, 68 teams (regular out; replacement = same-position teammate with the
largest share gain):

| regular out | n | listed injured | replacement usual -> game share | other teammates' gain | EPA: regular usual / replacement |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB1 | 51 | 49% | 28% -> 95% | +3 pts | 0.165 / 0.116 per dropback |
| RB1 | 42 | 64% | 17% -> 48% | +9 pts | 0.160 / -0.016 per carry |
| WR1 | 17 | 76% | 12% -> 29% | +12 pts | 0.766 / 0.414 per target |
| WR2 | 39 | 54% | 9% -> 23% | +7 pts | 0.407 / 0.555 per target |
| TE1 | 51 | 41% | 6% -> 11% | +0.5 pts | 0.425 / 0.358 per target |

Injured or not: when a listed-injured QB1 was out (25), the replacement averaged 0.016 EPA per
dropback against the starter's usual 0.205; when an unlisted QB1 was out (26: benchings, rotations,
unreported injuries), the replacement averaged 0.212 against 0.127. RB1 listed out (27): -0.035
against 0.124.

Depth chart check (chart downloaded 2026-11-19, after every game here): when the missing regular is
still first team on the chart (a short absence, n 96), the replacement is the chart's second man at
that position in 56 (58%): QB 14 of 20, RB 12 of 20, TE 17 of 27. Receivers spread (WR1 out: the
replacement is another position's starter in 5 of 7). The chart's Bench unit holds the injured
(Ohio State's Julian Sayin, Pittsburgh's Mason Heintschel).

Kneel removal on the 68-team games (rebuilt folders as above, model_v2 walk-forward 2026):

| games | n | baseline log loss | rule | widened |
| --- | ---: | ---: | ---: | ---: |
| with an affected P4/ND team | 161 | 0.6405 | 0.6405 (+0.0000, SE 0.0001) | 0.6405 (+0.0000) |
| P4/ND games, not affected | 192 | 0.4421 | 0.4420 (-0.0001) | 0.4420 (-0.0001) |
| all P4/ND games | 353 | 0.5326 | 0.5326 (-0.0000) | 0.5325 (-0.0001) |
| all 2026 | 632 | 0.5452 | 0.5451 | 0.5451 |

Win probabilities move 0.04-0.06 points on average (max 0.4). Kneels still do not matter to
model_v2. Outputs: /mnt/project-files/exploration/kneels/ (`starter_response_p4_output.txt`,
`starters/*.csv`).
