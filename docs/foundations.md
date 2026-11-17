# Foundations

The contracts every feature, evaluation and model module builds on. Code: `src/cfb/core/`
and `src/cfb/features/base.py` (W1). A change to anything on this page is a contract
change: write it into the build plan first.

## 1. Cutoffs and visibility

Each game has a `cutoff_utc`. A prediction for the game may use only what was known then.

| Game | Cutoff |
| --- | --- |
| Regular season | Start of the game's CFBD calendar week. All games of a week share it. |
| Postseason | 24 hours before kickoff. Each bowl and playoff game has its own. |

A game's results become visible 5 hours after kickoff (`RESULT_LAG`), not at kickoff. The
plan said "games that started before the cutoff"; the lag is stricter and only matters in
the postseason, where 223 cutoffs fall while another bowl is being played. No
regular-season cutoff is affected.

The snapshot has 660 distinct cutoffs: 198 weekly and 462 postseason.

## 2. Core tables

Built by `python -m cfb.core.build` into `data/core/`. Timestamps are `datetime64[ns, UTC]`.

**`game_index`**: one row per game, in kickoff order.
`game_id, season, season_type, week, start_time_utc, cutoff_utc, neutral, conference_game,
venue_id, home_id, away_id, home_fbs, away_fbs, completed, home_points, away_points`,
then the extras `home_team, away_team, start_time_tbd`. Points are empty until a game is
completed. Team names are there because several raw tables are keyed by name.

**`team_game`**: two rows per game. Keys `game_id, team_id, opp_id, is_home`, then:

| Columns | Source | Note |
| --- | --- | --- |
| `season, season_type, week, start_time_utc, neutral, team_fbs, opp_fbs` | `game_index` | |
| `points, opp_points, margin, win, postgame_win_prob, pregame_elo, postgame_elo` | `games` | |
| `box_*` (39) | `/games/teams` | `"5-13"` values are split: `box_third_down_conv`, `box_third_down_att` |
| `adv_off_*`, `adv_def_*` (56) | `/stats/game/advanced` | per-game rates; do not sum |
| `ppa_off_*`, `ppa_def_*` (12) | `/ppa/games` | per-game averages; FBS teams only |
| `pbp_off_*`, `pbp_def_*` (24) | `pbp` | additive totals for the units `all`, `rush`, `pass`: `plays`, `epa_sum`, `successes`, `expl_sum` |

EPA per play is `epa_sum / plays`, success rate `successes / plays`, explosiveness
`expl_sum / successes`. Use the `pbp_` totals for anything spanning several games.

**`pbp`**: one row per play with team ids and shared definitions.
`game_id, play_id, drive_id, drive_number, play_number, season, season_type, week,
offense_id, defense_id, offense_is_home, period, clock_seconds, down, distance,
yards_to_goal, yards_gained, offense_score, defense_score, score_diff, play_type,
play_class, is_rush, is_pass, is_scrimmage, ppa, success, passing_down, turnover, scoring,
garbage_time`

| Flag | Definition |
| --- | --- |
| `is_rush`, `is_pass` | By play type. Fumble plays go by the play text ("pass", "sacked" mean pass). |
| `is_scrimmage` | Rush or pass with a PPA value from CFBD. |
| `success` | Offensive touchdown, or 50% of the distance on first down, 70% on second, 100% on third or fourth. Turnovers fail. Empty for non-scrimmage plays. |
| `passing_down` | Second down with 8 or more to go, third or fourth with 5 or more. |
| `garbage_time` | Margin above 38 in the second quarter, 28 in the third, 22 in the fourth. Nothing is filtered; the flag is for the reader to apply. |

Agreement with CFBD's own per-game numbers over 22,136 team-games: success rate r = 0.994,
EPA per play r = 0.996. CFBD counts every play that has a PPA value, about 0.7 more per
team-game (some field goals and period markers). CFBD also counts every fumble play as a
rush and leaves "Interception" plays out of passing; with those two conventions the play
counts match exactly in 98.8% (rush) and 100% (pass) of team-games from 2019 on.

**`teams`**: `season, team_id, team, conference, fbs` for every team that appears in a game.
**`weeks`**: `season, season_type, week, start_utc, end_utc` from the CFBD calendar.
**`game_pregame`**: `game_id, home_pregame_elo, away_pregame_elo`.

## 3. AsOfView

```python
from cfb.core import Warehouse

wh = Warehouse("data")
view = wh.view(cutoff)                      # any timestamp; normally a game's cutoff_utc
view.cutoff, view.season
view.table("team_game")                     # rows known at the cutoff, oldest first
view.table("pbp", seasons=view.season, columns=["game_id", "offense_id", "ppa"])
view.prior_seasons("ratings_sp")            # finals of seasons that ended before the cutoff
view.schedule()                             # past games and the slate, no results
view.schedule(future=True)                  # plus later regular-season games of this season
view.latest("unit_ratings", ["team_id", "unit"])
view.has("weather")                         # policy exists and data is present
view.data_key                               # number of visible games; equal keys, equal results
```

Frames share memory with the cache (pandas copy-on-write, hence pandas 3). Slicing by
time costs microseconds; a caller may modify what it gets.

Errors: `UnknownTableError` (no policy), `BannedTableError` (season table, own season),
`MissingTableError` (policy but no data in this warehouse).

### When each table is visible

| Kind | Rule | Tables |
| --- | --- | --- |
| game | The game kicked off at least 5 hours before the cutoff. | `game_index`, `team_game`, `pbp`, `games`, `drives`, `plays`, `games_teams`, `stats_game_advanced`, `ppa_games`, `games_players`, `stats_game_havoc`, `punting_games` |
| pregame | The game's own cutoff is at or before the view's: past games and the slate. | `game_pregame`, `lines`, `media`, `weather` |
| preseason | Season up to the cutoff season. | `teams`, `teams_fbs`, `talent`, `returning_production`, `recruiting_teams`, `ratings_elo_preseason` |
| season final | `view.table` raises. `view.prior_seasons` returns seasons whose last calendar week ended before the cutoff. | `ratings_sp`, `ratings_fpi`, `ratings_srs`, `ratings_elo`, `ratings_core`, `records`, `stats_season_advanced`, `coaches` |
| dated | The row's own timestamp is at or before the cutoff. | `unit_ratings` (`cutoff_utc`), `transfer_portal` (`transferDate`), `ratings_sp_asof`, `ratings_fpi_asof`, `ratings_srs_asof`, `ratings_core_asof` (`asof_utc`) |
| weekly | "pre": the calendar week has started. "post": it has ended. | `rankings` (regular pre, postseason post), `ratings_elo_weekly` (post) |
| static | Always. | `venues`, `calendar`, `weeks`, `coach_tenures` (see below) |

Special handling:

- `lines`: `homeScore` and `awayScore` are removed. The table holds closing lines, which
  are later than a week-start cutoff. It is for the market group and baselines only.
- `weather`: observed values for played games, forecasts for upcoming ones. Visible for the
  slate by design: training uses observed weather, live runs use forecasts.
- `game_pregame`: point-in-time for FBS vs FBS. Unreliable for FBS vs FCS from 2022. When
  a team plays twice inside one calendar week (week 0 is filed under week 1), the Elo
  before its second game already includes the first.
- `transfer_portal`: `destination` is filled in later at the source. It is blanked until
  the portal season has started.
- `coach_tenures`: a tenure appears once the hire date has passed (or, without one, once
  its start year is the cutoff season). `record.*`, `seasons`, `active` and
  `attributionComplete` are removed. An `endYear` in the cutoff season or later is blanked.
- `rankings`: the playoff committee ranking is published Tuesday evening US time, about 1.5
  days after the week-start cutoff. It is served as "pre" like the other polls.
- Rows of games that are not in `game_index` (non-FBS games in `lines`,
  `stats_game_advanced`) are dropped.
- `view.schedule(future=True)` leaves out postseason games and December conference games
  from week 14 on, because title games and bowls are set by results. The future schedule
  is as of the pull date; games moved during a season (2020 above all) appear as played.

A new table needs a policy in `cfb/core/policies.py`. Ask the W1 owner.

`Warehouse.game_index` and `Warehouse.outcomes()` include results. They are for drivers
and evaluation code, never for feature code.

## 4. Feature groups

```python
# src/cfb/features/context/__init__.py
import pandas as pd
from ..base import FeatureColumn, FeatureGroup

class Rest(FeatureGroup):
    name = "rest"
    prefix = "ctx_"
    level = "team"            # keys game_id, team_id.  "game": key game_id
    mode = "per_cutoff"       # default
    columns = [
        FeatureColumn("ctx_rest_days", "Days since the team's previous game",
                      sources=("schedule",), first_season=2014),
    ]

    def build(self, view, games):
        ...
        return frame          # game_id, team_id, ctx_rest_days

GROUPS = [Rest()]
```

- Prefixes: `str_` strength, `mat_` matchup, `coach_` coaching, `ctx_` context, `mkt_` market.
- `games` holds the schedule columns of the games to describe, never results. Take the
  slate from `games`, not from a table: an on-demand matchup is in no table.
- Return at most one row per key. Missing rows and unknown values stay empty; the runner
  completes the output to one row per key. Never fill from later data.
- Columns are numeric or boolean, start with the prefix, and each has one catalog line.
  `sources` names tables from section 3, or `"schedule"`.
- `build` is a pure function of its arguments. No state between calls, no module-level caches.
- `per_cutoff`: one call per distinct cutoff, every game in `games` has
  `cutoff_utc == view.cutoff`. Leak-safe as long as data comes from the view.
- `batch`: one call with a view at the last cutoff and all games. Faster for running
  totals, but the code must itself use only rows visible at each game's `cutoff_utc`
  (`start_time_utc + RESULT_LAG <= cutoff_utc`). The leak test is the guard.

The driver finds every `GROUPS` list in a subpackage of `cfb.features`; nothing else has
to be registered. `python -m cfb.features.build` writes `data/features/<group>.parquet`
and `catalog.json`.

### Leak test

```python
from cfb.features.base import assert_group_no_leak

def test_rest_does_not_leak(synthetic):          # fixture from tests/conftest.py
    assert_group_no_leak(Rest(), synthetic)

def test_rest_does_not_leak_on_real_data(snapshot):   # skipped when data/ is absent
    assert_group_no_leak(Rest(), snapshot, n=3)
```

For each tested cutoff the group is built on the real data and on a copy in which every
row not visible at that cutoff is shuffled and perturbed (results, later seasons of
preseason tables, later dated rows, same-season rows of season tables). Identifiers and
the schedule stay. Any difference in the group's values for that cutoff is a leak. A
per-cutoff group is first run on later slates, so remembered state shows up too. Sources
under `cfb/features/<area>/` are linted for file access, `Store`, `Warehouse` and
network imports.

`tests/test_features.py` already runs the lint and the synthetic leak test on every
registered group, so a new group is covered in CI without extra wiring. A group that needs
a table the synthetic store lacks is skipped there and must be tested on the snapshot:
`python -m cfb.features.build --leak-test`.

What the test cannot see: code that reads files directly (the lint covers the usual
ways), module-level caches, and leaks through the schedule itself.

## 5. unit_ratings

`data/derived/unit_ratings.parquet`, one row per team, cutoff and unit:
`team_id, cutoff_utc, unit, epa_adj, success_adj, explosive_adj, n_plays`, with `unit` one
of `off_rush, off_pass, def_rush, def_pass, off_all, def_all`.

Scale: EPA per play, success rate, EPA per successful play. Defensive units hold what the
defense allowed, so lower is better. `n_plays` 0 means no data yet and empty ratings.

Today it is a placeholder: unadjusted season-to-date averages from `team_game`, for every
team in the season, at all 660 cutoffs (855,696 rows, 8 seconds to build). W3 replaces it
by defining `unit_ratings(view) -> DataFrame` in `cfb/features/strength/__init__.py`; the
driver uses that function when it exists. Readers do not change:

```python
ratings = view.latest("unit_ratings", ["team_id", "unit"])
```

## 6. Predictions and metrics

Defined in `cfb.core.schemas` with `validate_*`, `write_*` and `read_*` functions.
Examples are in `samples/`.

**Predictions**: `data/predictions/<model_id>/<season>_<season_type>_<week>.parquet`,
week as two digits. One row per `(game_id, model_id)`. Baselines use the same schema.

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | int64 | |
| `model_id` | string | lower-case letters, digits, `_`, `-`, `.` |
| `p_home_win` | float64 | probability the home team wins; never empty. At a neutral site "home" is the team CFBD lists as home. |
| `pred_margin` | float64 | home minus away points. Empty until phase 2. |
| `pred_total` | float64 | home plus away points. Empty until phase 2. |
| `cutoff_utc` | timestamp | cutoff of the view the inputs came through |
| `created_utc` | timestamp | when the row was written |
| `reasons` | string | JSON list, strongest first, `"[]"` if none. Items: `{"text", "feature"?, "effect"?, "value"?}`; `effect` is signed toward the home team. |

**Metrics**: `data/eval/<model_id>.json`

```json
{
  "schema_version": 1, "model_id": "sample_elo", "created_utc": "2026-11-17T06:42:54+00:00",
  "scope": "fbs_vs_fbs",
  "overall":    {"n": 98, "log_loss": 0.522, "brier": 0.176, "accuracy": 0.745, "calibration": [...]},
  "by_season":  [{"season": 2025, "n": 98, "...": "..."}],
  "by_week":    [{"season": 2025, "season_type": "regular", "week": 1, "n": 48, "...": "..."}],
  "by_segment": [{"segment": "conference", "n": 6, "...": "..."}]
}
```

A calibration bin is `{"p_lo", "p_hi", "n", "p_mean", "win_rate"}`. `calibration` is
required in `overall` and `by_season`. Metrics are null when `n` is 0. Segment names in
use: `fbs_vs_fbs`, `fbs_vs_fcs`, `conference`, `non_conference`, `neutral`, `regular`,
`postseason`.

The `sample_elo` files cover 2025 weeks 1 and 2 (98 FBS vs FBS games) and show the format
only. The real baselines belong to `cfb.eval`.

## 7. Tests without the snapshot

CI has no data. `cfb.core.synthetic.synthetic_warehouse(path)` writes a fake store shaped
like the real one (8 FBS and 2 FCS teams, three seasons, the last one in progress) and
builds core tables and unit ratings from it. The `synthetic` fixture in
`tests/conftest.py` provides it; the `snapshot` fixture provides the real data and skips
the test when `data/core` is missing.

The synthetic store has the step 1 tables only. For another table, build a frame in the
test and add it: `Warehouse(synthetic.root).put("weather", frame)`.
