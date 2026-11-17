# CFB-Analysis

Predictive model for NCAA Division I FBS football, built on the
[CollegeFootballData.com](https://collegefootballdata.com/) API.

## Plan

1. **Data pipeline** (this package, `cfb.ingest`): pull 2014 through 2026 into a local store.
2. **Features**: one row per game from pre-kickoff information only. Opponent-adjusted
   efficiency, SOS, strength of record, home/road effects, run/pass matchup, and coaching
   (record vs. the spread, 4th-down aggressiveness, run/pass mix, offensive flexibility).
3. **Model**: winner and win probability. Train on seasons through 2025, test on 2026
   walk-forward (predict week N from data through week N-1). Baselines: closing-line
   implied probability and a simple rating model.
4. **Outputs**: dashboard and weekly report. Spread and totals come after step 3 performs well.

## Setup

```bash
pip install -e '.[dev]'
export CFBD_API_KEY=...        # never commit the key; omit where a proxy injects it
python -m cfb.ingest --list    # dataset names
python -m cfb.ingest --start 2014 --end 2026
python -m cfb.ingest --start 2026 --end 2026 --force   # refresh the current season
pytest
```

## Data layout

`data/` is git-ignored.

- `data/raw/<dataset>/<key>.json.gz`: exact API responses (source of truth).
- `data/tables/<dataset>/<key>.parquet`: flattened copy. Nested objects become dotted
  columns; lists inside a record are stored as JSON strings.

Load a dataset with `cfb.store.Store("data").load("games")`.

## Status

Verified against the live API (v5.33.3) on 2026-11-16: the full 2014 through 2026 pull
completed with 700 requests and 0 failures, and no endpoint or parameter changes were
needed. One season is 56 requests. The unit tests cover retry logic, storage, and task
planning only.

Result of that pull: 11,260 FBS-involved games, 1,995,021 plays, 276,210 drives,
251 MB on disk. 2026 was complete through week 11.

### Data notes

Observed in the 2014 through 2026 pull. Relevant to feature building.

- **Scope.** `games`, `drives`, `plays`, and `games_teams` hold games with at least one FBS
  team, so FBS vs. FCS games are included. `lines`, `records`, `stats_game_advanced`,
  `ratings_srs`, and `talent` also contain non-FBS rows. Join to `games` or `teams_fbs`.
- **Not point-in-time.** Year-level datasets (`ratings_*`, `records`, `coaches`,
  `stats_season_advanced`) are whatever the API holds on the pull date: end-of-season values
  for past seasons, season-to-date for the current one. Using a season's own values to
  predict that season's games leaks results. Pregame Elo on `games` is point-in-time.
- **Current season.** Weeks not yet played are cached as empty. Refresh with `--force`.
- **Plays.** 41 completed FBS vs. FBS games have no plays (2014: 14, 2015: 7, 2016: 13,
  2017: 4, 2018: 2, 2019: 1). None are missing from 2020 on.
- **Lines.** Every FBS vs. FBS game from 2014 through 2025 has a spread except one in 2024.
  Moneylines start in 2021. No provider spans all seasons (`consensus` 2014 to 2022,
  Bovada 2019 on, DraftKings 2023 on).
- **Elo.** `homePregameElo` is null only in FBS vs. FCS games, almost all before 2022.
- **Gaps at the source.** `talent` is empty for 2014. `ratings_srs` has 77 teams for 2020.
  `ratings_sp` includes a `nationalAverages` row. `talent` lacks Air Force and Navy, and
  `returning_production` lacks first-year FBS teams (checked for 2025).
