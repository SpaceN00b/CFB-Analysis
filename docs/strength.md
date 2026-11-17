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
