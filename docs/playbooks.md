# Team playbooks (test bench)

`cfb.lab.playbooks` is an experiment: no feature group, model or weekly run reads it. It builds
rough team playbooks from the detailed stat-crew play text that `cfb.core.playtext` parses (every
game from 2025 week 9 on), joined to the players and positions the SEC player maps use. The default
is the 2026 SEC.

```bash
python -m cfb.core.build && python -m cfb.core.playtext   # core tables and play_detail / play_players
python -m cfb.lab.player_maps                              # optional: style and defender-role labels
python -m cfb.lab.playbooks                                # about 10 s; --season, --conference, --player-maps, --out
python scripts/playbook_check.py --ridge 5,50,200          # second stage on model_v2 (needs data/predictions/model_v2)
```

Output in `data/lab/`: `playbooks_<season>_<conf>.json` and `.html` (the viewer,
`src/cfb/lab/playbooks_viewer.html` with the JSON embedded).

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

2026 SEC (weeks 1-11): Georgia (19%) and Alabama (13%) are the only offenses under center on more
than 8% of plays (FBS 6%); under center, SEC teams run 67% to 100% of the time. Crew-adjusted no
huddle ranges from 27% (Kentucky) and 28% (Alabama) to 98-99% (LSU, Mississippi State). Pass rate
over expected: Alabama +7.6, Ole Miss +6.7, LSU +6.1 points; Tennessee -6.1, Florida -4.1.

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

model_v2 alone: 0.5454 (FBS), 0.5567 (SEC games). One-column slopes of the margin residual (home
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
