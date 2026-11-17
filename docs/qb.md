# Quarterback features and the QB-out override (P1)

Code: `src/cfb/features/strength/qb.py` (group `str_qb`), `src/cfb/model/qb_out.py` (manual
override). Scripts: `scripts/qb_fit.py` (constants), `scripts/qb_select.py` (ablation),
`scripts/qb_oracle.py` (oracle backtest), `scripts/build_groups_parallel.py` (builds a group
with one process per season, about 4 minutes for `str_qb` on 4 cores). Tests: `tests/test_qb.py`.
Data: the player-level datasets in `docs/data-coverage.md`.

**Result:** `str_qb` is built and leak-tested but **not used by any model**: no column or
subset improves season-forward log loss beyond `model_v1` by the acceptance rule. The oracle
backtest puts a primary starter's absence at about 2 to 3 points, under the 3-point bar of the
brief; the manual override was built anyway at Josiah's option (see the end).

## Player-game rows and quality

`player_success_games` (passing and rushing plays per player and game, with `gameId`) joined
to `ppa_players_games` (average PPA as passer and rusher) on season, season type, week,
athlete, team and opponent. Rows with at least one passing play. Value per game = PPA over
passing plus rushing plays, minus the opponent defense's pass (rush) EPA allowed per play
relative to the mean FBS defense, from `unit_ratings` at that game's cutoff.

Quality at a cutoff, over every visible game at any school (transfers keep their history):

    q = (sum_w adj + K * prior) / (sum_w plays + K),   w = DECAY ** (seasons ago)

Prior = 0.2163 + 0.8401 x (recruiting rating - 0.85) [rated] or 0.2163 - 0.0400 [unrated],
+ 0.0309 if a junior-college recruit, + 0.0055 x (class year - 2). The rating is the latest
high-school or JUCO class found through `recruiting_players.athleteId` or `roster.recruitIds`
(65% of player-games have one).

### Fit (`scripts/qb_fit.py`, seasons 2014-2025)

Predict each player-game's adjusted PPA per play from the player's earlier games only; score
the play-weighted squared error. 34,842 player-games, 5,763 players.

| Setting | MSE |
| --- | --- |
| Constant (0.2586) | 0.10734 |
| K 150, no decay | 0.10101 |
| K 75 / **150** / 500, decay 0.4 | 0.10066 / **0.10038** / 0.10173 |
| K 150, decay 1.0 / 0.8 / 0.6 / 0.5 / 0.4 / 0.3 / 0.2 | 0.10101 / 0.10067 / 0.10044 / 0.10039 / **0.10038** / 0.10044 / 0.10060 |

Chosen: K 150 plays, DECAY 0.4 per season. The per-game target is noisy; the shrunk career
value explains about 6.5% of its variance.

## Group `str_qb` (team level, per cutoff)

| Column | Meaning |
| --- | --- |
| `str_qb_q` | Expected starter's quality |
| `str_qb_mix_q` | Play-weighted quality (today's values) of the QBs behind the team's rating: this season's games, before the first game last season's |
| `str_qb_diff` | `str_qb_q - str_qb_mix_q`: a starter change the rating has not seen |
| `str_qb_plays` | log(1 + the starter's career QB plays) |
| `str_qb_share` | Starter's share of the team's QB plays behind the rating |
| `str_qb_prior` | Starter's prior |
| `str_qb_from_roster` | 1 before the team's first game of the season, else 0 |

Expected starter: the player with the most passing plays in the team's last visible game;
before the first game, the season roster's QB with the most passing plays last season (any
school), then career, then the best recruiting rating. Roster is the final roster of the
season, used as if known before it. Player PPA rows are visible when their calendar week ends,
so a bowl's player data is visible only after the last bowl.

Leak test on the snapshot passes (`tests/test_qb.py::test_no_leak_on_real_data`).

## Ablation against model_v1 (`scripts/qb_select.py`)

Season-forward 2018-2025, 6,024 FBS vs FBS games, margin ridge lam 0.03 as frozen, columns as
home-minus-away differences. Paired bootstrap over whole weeks.

| Configuration | Log loss | vs model_v1 (SE) | Seasons better |
| --- | --- | --- | --- |
| model_v1 | 0.5318 | | |
| + all 7 | 0.5321 | +0.0003 (0.0007) | 4/8 |
| + diff | 0.5318 | -0.0000 (0.0003) | 5/8 |
| + q | 0.5315 | -0.0003 (0.0006) | 6/8 |
| + q, mix | 0.5314 | -0.0004 (0.0006) | 6/8 |
| + diff, share | 0.5318 | -0.0000 (0.0005) | 4/8 |
| + diff, plays | 0.5319 | +0.0001 (0.0004) | 3/8 |
| + diff, from_roster | 0.5318 | +0.0000 (0.0003) | 4/8 |

Weeks 1-4 only (1,467 games, model_v1 0.4970): + q -0.0001 (SE 0.0011), + diff -0.0004
(SE 0.0004), + all 7 +0.0015 (SE 0.0013). None passes the rule (gain over 2 SE, or gain
>= 0.0005 and better in >= 5 of 8 seasons). Model_v1 margin residual on `d_str_qb_diff`:
4.3 points per unit (SE 4.6), so a 0.1 per-play gap is worth about 0.4 points. The power
and Elo ratings already move with the team's results after a QB change, and the last game's
leading passer is a poor guess of the next starter (below).

## Oracle backtest (`scripts/qb_oracle.py`, diagnostic only)

Actual starter = most passing plays in the game, known only afterwards. 12,048 team-games of
the 2018-2025 evaluation games; both starters known in 11,905; the actual starter differed
from the expected one in 15.8% (14.9% in season, 28.0% before the first game).

Margin residual against model_v1's season-forward margin, from the team's side (points):

| Case | n | Residual (SE) |
| --- | --- | --- |
| Starter as expected | 10,027 | +0.51 (0.16) |
| Changed, any | 1,878 | -2.81 (0.39) |
| Primary (share >= 0.5) replaced, in season | 1,010 | -4.16 (0.51) |
| ... in-game change (primary played, someone else led) | 407 | -6.04 (0.80) |
| **Clean QB out**: primary threw no pass, replacement >= 80% of passing plays | 529 | **-1.83 (0.68)** |
| Clean, primary had >= 80% of the plays | 317 | -2.90 (0.89) |
| Clean, primary had 50-80% | 212 | -0.23 (1.05) |

In-game changes are partly a result of the game (injury in a loss, benching), so the clean
cases size a pregame "QB out". Clean residual on the quality gap dq (replacement minus
starter): -1.23 (SE 0.71) + 24.2 (SE 8.1) x dq; dq quartiles -0.083, -0.024, +0.034. By
season (clean): 2018 +2.2, 2019 -2.7, 2020 +1.4, 2021 -5.6, 2022 -1.8, 2023 -2.3, 2024 -2.7,
2025 -3.0.

Log loss with the oracle term (home-minus-away dq x 22.6 points, fitted in sample): 0.5315
to 0.5295, an upper bound for perfect starter knowledge.

The brief's bar for building an override was 3 points for a typical starter-to-backup drop.
The clean effect is 1.8 points overall and 2.9 for a dominant starter, so the bar is not met.

## Manual QB-out override (`cfb.model.qb_out`)

Built as the recommended option of the open decision in the P1 thread. A user who knows a starter will not play names the team. The
team's margin moves by `-1.23 + 24.2 x (q_replacement - q_starter)`, clipped to [-6, 0]
points, and the win probability by the model's log-odds per point (fitted calibration in a
matchup; 0.109 in the weekly run, the model_v1 slope fitted through 2021 and 2025 is 0.1085 /
0.1097). The starter is the expected starter above; the default replacement is the team's QB
with the most passing plays this season, then the roster QB with the most career plays. A
reason line "QB out (manual): ..." is added.

    python -m cfb.predict matchup "Ohio State" "Michigan" --qb-out "Michigan"
    python -m cfb.predict matchup "Texas" "Texas A&M" --qb-out "Texas=Arch Manning"

Weekly run: entries under `[qb_out]` in `dashboard/config.toml` apply to the target week's
games of the primary model in the report. Stored predictions (`data/predictions/`) stay the
model's own, so scoring is unaffected. Clear the entries after the week.
