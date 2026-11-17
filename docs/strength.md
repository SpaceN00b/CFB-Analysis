# Team strength (W3)

Code: `src/cfb/features/strength/`. Prefix `str_`. Scripts: `scripts/strength_*.py`.

## unit_ratings

`cfb.features.strength.unit_ratings(view)` replaces the W1 placeholder. The driver
(`python -m cfb.features.build`) picks it up; the schema, units and scale are unchanged
(`docs/foundations.md` section 5). Readers keep using
`view.latest("unit_ratings", ["team_id", "unit"])`.

### Method

1. Plays: scrimmage plays (`pbp.is_scrimmage`) outside garbage time (`pbp.garbage_time`),
   summed per game, offense and defense with `cfb.core.pbp.unit_sums`. Units `all`,
   `rush`, `pass` use the shared `is_rush` / `is_pass` flags.
2. Model, per unit kind and metric (`ridge.fit`):

       y = mu + hfa * h + off[offense] + def[defense]

   `y` is the game's EPA per play, success rate or EPA per successful play; the weight is
   plays (successes for explosiveness). `h` is +1 when the offense is at home, -1 when the
   defense is, 0 at a neutral site. Ratings are reported for a neutral field:
   `off_*` = mu + off, `def_*` = mu + def (what the defense allows; lower is better).
3. Penalty: each team value is pulled toward its prior with strength `lam` (in plays).
   With no games a team sits at its prior; the prior's share fades as plays accumulate,
   roughly lam / (lam + plays).
4. Prior: last season's final fit of the same model, times `rho` (shrink toward the league
   mean). League mean and home field are pulled toward last season's values.
5. Teams below FBS in the cutoff season are pooled into one team. Each FCS team is reported
   with the pooled rating and its own play count. A team new to FBS starts from last
   season's pooled FCS value.
6. Recency: a half-life in weeks is supported (`MetricParams.half_life`); the tuned value
   is none (see below).

Everything comes from the view: the cutoff season's games visible at the cutoff and the
whole previous season. Build time: about 0.12 s per cutoff, 83 s for all 660 (855,696 rows; 0.15% empty, all in
2014 week 1, which has no prior season).

### Differences from the placeholder contract

- Ratings are filled before a team has played (prior values with `n_plays == 0`). Section 5
  of foundations.md says `n_plays` 0 means empty ratings; that holds only for the first
  season in the data (2014), which has no prior season.
- `n_plays` counts non-garbage scrimmage plays only, so it is lower than the placeholder's.

### Parameters (chosen on 2015 to 2025, never 2026)

`scripts/strength_tune_units.py` predicts every FBS vs FBS offense-defense pairing of
each cutoff's slate from the ratings at that cutoff and scores the play-weighted squared
error against the game's actual value. The skill column is 1 - MSE / MSE of the league
mean so far.

| Metric | lam (plays) | half-life | rho | Skill all / rush / pass |
| --- | --- | --- | --- | --- |
| EPA per play | 300 | none | 0.8 | 0.179 / 0.092 / 0.096 |
| Success rate | 300 | none | 0.8 | 0.224 / 0.131 / 0.143 |
| EPA per successful play | 300 | none (successes) | 0.7 | 0.047 / 0.042 / 0.038 |

Grids searched: lam 150 to 1200, half-life 4, 8, 16, 32 weeks and none, rho 0.2 to 1.0.
Explosiveness is mostly noise game to game (skill under 5%).

Recency weighting did not help: every half-life tried (4 to 32 weeks) scored worse than
none (16 weeks: +0.2% MSE for EPA). The brief asked for recent games to be weighted more; the code supports it, but the
tuned setting is no decay. Recent form is reported separately by the `form` group.

## Feature groups

All are level `team` (keys `game_id, team_id`, one row per side), mode `per_cutoff`, and
built from the view only. `python -m cfb.features.build --list` prints every column.

| Group | Columns | Notes |
| --- | --- | --- |
| `str_power` | `str_pwr_off`, `str_pwr_def`, `str_pwr_net`, `str_pwr_games`, `str_pwr_prior_share`, `str_pwr_net_se`, `str_unit_*` | Points per game against an average FBS team on a neutral field. Defense: points allowed (lower is better). `str_unit_*` repeat the all-plays unit ratings for the model. |
| `str_prior` | `str_prior_off/def/net`, last season's in-house rating, SP+, FPI, SRS, CFBD preseason Elo, talent, returning production, 4-year recruiting, portal in/out/net | Raw inputs are kept as columns next to the projection. |
| `str_elo` | `str_elo`, `str_elo_win_prob` | In-house Elo, recomputed from all visible games at each cutoff. |
| `str_schedule` | `str_sos_played`, `str_sos_remaining`, `str_sor`, `str_sor_per_game`, `str_wins`, `str_losses` | SOS uses today's power ratings of the opponents. SOR: wins minus the expected wins of the 25th-best FBS team against the same schedule and sites. |
| `str_luck` | `str_luck_fumble_rec`, `str_luck_int`, `str_luck_turnover_margin`, `str_luck_win_prob`, `str_luck_close_record` | Season to date. |
| `str_special_teams` | `str_st_fg_poe`, `str_st_kr_avg`, `str_st_pr_avg`, `str_st_kr_allowed`, `str_st_pr_allowed`, `str_st_net_punt` | Shrunk toward the league average. Net punting from 2015 (no punting data for 2014). |
| `str_form` | `str_form_pts3`, `str_form_epa3`, `str_form_games` | Last 3 games against the season mean, after opponent adjustment with today's ratings. Empty until a team has 4 games. |
| `str_polls` | AP and Coaches rank and points, `str_poll_cfp_rank`, `str_poll_ranked` | The latest poll visible at the cutoff. Committee ranking lagged one week in the regular season. Unranked: rank empty, points 0. |

### Power rating

Same ridge model as the unit ratings, on points: each team-game is one observation
(the team's points against the opponent's defense), weight 1, home field estimated,
lower division pooled. The prior is the preseason projection below; the prior's share is
about lam / (lam + games) with lam = 4 games. Games are down-weighted with a 16-week
half-life (slightly better than none: 0.5334 against 0.5340 in the exploration).

Preseason projection (`prior.py`, fitted by `scripts/strength_fit_prior.py` on 2015 to
2025): offense and defense are each linear in last season's final in-house offense and
defense, last season's SP+ offense and defense, talent (z-score), 4-year recruiting
(z-score) and returning production. Leave-one-season-out R^2 of the season's final
rating: offense 0.436, defense 0.472 (last season alone: 0.372 and 0.417). Teams new to FBS
start from last season's pooled lower-division value.

### Check against CFBD Elo (`scripts/strength_check.py`)

Season-forward, 2018 to 2025, the 6,024 FBS vs FBS games with a closing spread and CFBD
pregame Elo. Logistic on the home indicator plus one rating gap, fitted on seasons 2015 to
S-1 and scored on S. Ratings come from the feature code at each game's cutoff.

| Season | Games | Power gap log loss | Accuracy | CFBD Elo log loss | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 2018 | 772 | 0.5070 | 0.738 | 0.5219 | 0.737 |
| 2019 | 774 | 0.5015 | 0.748 | 0.5186 | 0.744 |
| 2020 | 534 | 0.5367 | 0.728 | 0.5535 | 0.712 |
| 2021 | 770 | 0.5497 | 0.704 | 0.5551 | 0.697 |
| 2022 | 776 | 0.5655 | 0.707 | 0.5860 | 0.691 |
| 2023 | 792 | 0.5296 | 0.731 | 0.5500 | 0.730 |
| 2024 | 798 | 0.5503 | 0.706 | 0.5667 | 0.672 |
| 2025 | 808 | 0.5339 | 0.730 | 0.5427 | 0.726 |
| All | 6,024 | **0.5342** | 0.724 | 0.5492 | 0.714 |

Better than CFBD Elo in 8 of 8 seasons. Other gaps on the same games: in-house Elo
0.5545, preseason projection alone 0.5909. Games where either team has played fewer than
4 games: 0.5159; the rest 0.5434. Fitted on all of 2015 to 2025, 10 points of net gap are
worth 1.19 in log-odds (`WIN_SCALE` 8.4 points) and home field 0.32; residual SD of one
team's points is 11.7 (`POINTS_SD`).

### Known limits

- The projection coefficients and lam were chosen on 2015 to 2025, which includes the
  check seasons 2018 to 2025. There are 9 coefficients per side and one lam; the
  leave-one-season-out R^2 above is within 0.01 of the in-sample fit.
- `str_luck_int` needs passes defensed (`box_passes_deflected`), which the team box scores
  lack in 2014 and 2015; it is empty there. `str_luck_fumble_rec` comes from the
  play-by-play recovery plays, because `box_total_fumbles` is missing in 2014, 2015 and
  2020.
- In-house Elo parameters (K 45, home 45, regression 0.15) were chosen on 2015 to 2025;
  the grid was flat within 0.003 log loss.
- SOS and SOR use the current ratings of past opponents, not their ratings when played.

- Player-based prior (version 2, `str_power2`) for model_v2: [priors.md](priors.md).
