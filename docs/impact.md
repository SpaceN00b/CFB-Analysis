# Player impact scores from usage share (research, 2026-11-18)

Question (Josiah): can an absent player's cost scale with his share of production (a receiver
with 40% of targets matters more than a back with 20 yards a game), and does it improve picks?

**Result:** the share-scaled cost of an absence is measurable but small (about 1 to 3 points
for a team's lead skill player, near zero for defenders), and no pregame version improves
model_v2 log loss in 2018-2025. Nothing is added to any model.

Scripts (diagnostic only, nothing in `src/` changed): `scripts/impact_base.py` (model_v2
season-forward margins 2018-2025 plus 2026 fit on 2015-2025), `scripts/impact_players.py`
(roles and absences), `scripts/impact_select.py` (regressions and log-loss tests),
`scripts/impact_cases.py` (lead-player table). Outputs go to `data/exploration/`.

## Method

- **Role** (from CFBD box scores, `games_players`): a player's share of team production in
  the earlier games of the season in which he had a box-score line, at least 2 such games:
  pass attempts, carries, receptions, rushing + receiving yards, tackles, and TFL + sacks +
  passes defended + interceptions. Receptions stand in for targets: box scores have no targets.
  Position from the season roster.
- **Absent** = no box-score line in the game. This misreads a healthy receiver with no catch
  or a defender with no tackle as absent, and an injury early in a game as present.
  - *New out*: absent, played the previous game (known only after the game; sizes the effect).
  - *Missed last game*: no line in the previous game (known before kickoff). It predicts the
    next absence: QB with >= 50% of attempts 75% absent again vs 10% if he played; WR1 (>= 20% of catches)
    58% vs 7%.
- Residual = actual margin minus model_v2's season-forward predicted margin, from the team's
  side. 6,024 baseline games 2018-2025; 629 in 2026 through week 11.

## Cost of a lead player's absence (`impact_cases.py`)

Lead player = the team's highest share at the position. Points vs model_v2 (SE), 2018-2025.

| Lead player | New out | n | Missed only the last game | n |
| --- | --- | ---: | --- | ---: |
| QB, >= 80% of attempts | **-2.73 (0.71)** | 507 | -0.98 (0.79) | 413 |
| RB, >= 40% of carries | -0.65 (0.94) | 230 | -1.22 (1.19) | 174 |
| WR, >= 25% of catches | -1.75 (1.04) | 280 | -0.17 (1.06) | 215 |
| TE, >= 15% of catches | -1.87 (0.93) | 336 | -1.37 (0.98) | 304 |
| DL, >= 6% of tackles | -0.37 (0.86) | 344 | -0.37 (0.95) | 288 |
| LB, >= 10% of tackles | -0.22 (0.82) | 390 | -1.41 (0.84) | 280 |
| DB, >= 8% of tackles | +0.23 (0.75) | 471 | -0.59 (0.82) | 370 |

2026 samples are 15 to 87 team-games per row (SE 2 to 6 points): not informative alone.

Per 100% of share (all role players, new absences, one regression): QB -1.56 (0.62),
RB -1.94 (1.25), WR -4.12 (1.45), TE -4.42 (2.72); defensive positions are unstable
(DL +9.5 (4.4), LB -4.0 (3.4), DB -1.4 (3.2)). So a receiver with 40% of catches out costs
about 4.12 x 0.4 = **1.6 points (SE 0.6)**; a back with 15% of carries about 0.3.

The oracle term (every actual absence known, slopes fitted in sample) moves model_v2 log
loss 0.5255 to 0.5258: absences explain almost none of the 15.8-point residual SD.

## Pregame test (`impact_select.py`)

model_v2 columns plus home-minus-away "missing share of players who missed the last game",
same ridge (lam 0.03), fit 2015..S-1. Paired bootstrap over weeks.

| Added columns | 2018-25 log loss (v2 0.5255) | diff (SE) | seasons better | 2026 diff (SE) |
| --- | --- | --- | --- | --- |
| missed last: pass, carries, catches, tackles | 0.5256 | +0.0001 (0.0005) | 4/8 | +0.0009 (0.0013) |
| missed only last: same | 0.5254 | -0.0000 (0.0003) | 5/8 | -0.0015 (0.0014) |
| missed last: QB share | 0.5256 | +0.0002 (0.0002) | 3/8 | -0.0001 (0.0001) |
| missed last: pass, yards, tackles | 0.5252 | -0.0003 (0.0005) | 6/8 | +0.0005 (0.0017) |
| missed last: share by position | 0.5258 | +0.0003 (0.0006) | 3/8 | +0.0003 (0.0015) |
| missed only last: share by position | 0.5257 | +0.0002 (0.0004) | 4/8 | -0.0022 (0.0013) |

None passes the rule used for model changes (gain over 2 SE, or >= 0.0005 and better in >= 5
of 8 seasons). The 2026 gain for "missed only last game, by position" (-0.0022, 1.7 SE) is
not matched in 2018-2025 (+0.0002).

## Why the signal is weak

- The power and Elo ratings already include the games the player missed.
- Missing the last game predicts the next absence only 55-75% of the time.
- Box-score absence is noisy for receivers and defenders.
- Lead-player absences are a few percent of team-games, and a 1-3 point effect is small
  against a 15.8-point residual SD.

## Re-run with play-by-play stats (2026-11-18, `--plays-stats`)

CFBD `/plays/stats` (table `plays_stats`, snapshot `cfbd_data_2014-2026_2026-11-18_playstats.tar`, PR #27):
target share (Target + Reception rows) replaces reception share for WR/TE, and any plays_stats
row also counts as presence. Run: `impact_players.py --plays-stats`, then `impact_select.py` and
`impact_cases.py` with `--plays-stats`. Limits: intended receivers on incompletions are recorded
on 19% (2024) to 95% (2026) of incompletions; no Tackle rows, so defenders are unchanged.
Presence changed little: 3,818 player-games added to 671,443 box-score lines.

| 2018-2025 | Reception share | Target share |
| --- | --- | --- |
| WR1 (>= 25%) new out | -1.75 (1.04), n 280 | -1.22 (1.00), n 307 |
| TE (>= 15%) new out | -1.87 (0.93), n 336 | -0.98 (1.05), n 280 |
| WR per 100% share | -4.12 (1.45) | -3.24 (1.44) |
| TE per 100% share | -4.42 (2.72) | -3.77 (2.98) |
| Best pregame variant (pass, yards, tackles) | -0.0003 (0.0005), 6/8 | -0.0003 (0.0005), 7/8 |

Pregame variants still fail the rule (2026: best -0.0015, SE 0.0011, not matched in 2018-2025).
A 40%-target receiver out: about 3.24 x 0.4 = 1.3 points (SE 0.6). Conclusion unchanged.

## Open option

Extend the manual QB-out override (`cfb.model.qb_out`) to any player, using the per-share
points above. That works for known-out players (e.g. conference availability reports), with
the stated SEs.
