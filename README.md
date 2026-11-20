# CFB-Analysis

Predictive model for NCAA Division I FBS football, built on the
[CollegeFootballData.com](https://collegefootballdata.com/) API. Phase 1 predicts the
winner and a win probability for every game with an FBS team. Phase 2 adds a predicted
spread and total for every game, scored against the closing lines.

## Status

| Part | State | Where |
| --- | --- | --- |
| Data pipeline, 2014 to 2026 | done | `cfb.ingest`, coverage in [docs/data-coverage.md](docs/data-coverage.md) |
| Sportradar play-by-play, 2013 on (players per play) | 2026 season pulled | `cfb.sportradar`, [docs/sportradar.md](docs/sportradar.md) |
| Foundations: core tables, point-in-time access, leak test, feature interface, file contracts | done | `cfb.core`, `cfb.features.base`, [docs/foundations.md](docs/foundations.md) |
| Feature groups: strength, matchup, coaching, context, market | done | `cfb.features.<area>`, docs/strength.md, matchup.md, coaching.md, context.md |
| Evaluation and baselines | done | `cfb.eval`, [docs/eval.md](docs/eval.md) |
| Win-probability model (`model_v1`) | done | `cfb.model`, `cfb.predict`, [docs/model.md](docs/model.md) |
| Spread and total (phase 2) | done | `cfb.model.points_config`, `cfb.eval.points`, [docs/points.md](docs/points.md) |
| Weekly report, dashboard | done | `cfb.report`, `cfb.run_week`, `dashboard/`, [docs/report.md](docs/report.md) |
| Play text parsed into run direction, pass depth, air yards, targets, tacklers, penalties | done | `cfb.core.playtext`, [docs/play-text.md](docs/play-text.md) |
| Depth chart import (dated snapshots, all 68 P4 + Notre Dame on 2026-11-19) | done | `cfb.depth_charts`, [docs/depth-charts.md](docs/depth-charts.md) |
| Local engine on the PC: CFBD pulls, build, weekly run, R2 backup, Task Scheduler; main and experimental editions | done | `cfb.engine`, `cfb.schedule`, `cfb.edition`, [docs/local-app.md](docs/local-app.md) |

Baselines to beat, season-forward on 2018 to 2025 (6,024 FBS vs FBS games): closing spread
log loss 0.520, CFBD pregame Elo plus home field 0.549, home team always 0.679.

## Setup

```bash
pip install -e '.[dev]'
export CFBD_API_KEY=...        # never commit the key; omit where a proxy injects it

# data: restore the snapshot (cloud threads) or pull it (about 1,350 API calls)
tar -xf /mnt/project-files/cfbd-data/$(cat /mnt/project-files/cfbd-data/LATEST)
python -m cfb.ingest --start 2014 --end 2026
python -m cfb.ingest --start 2026 --end 2026 --force   # refresh the current season

python -m cfb.core.build       # data/tables -> data/core            (about 10 seconds)
python -m cfb.features.build   # unit_ratings and every feature group
pytest                         # snapshot tests are skipped when data/ is absent
```

Paths and keys come from `cfb.settings` (`src/cfb/settings.py`). Defaults are the values
above: data dir `data`, run config `dashboard/config.toml`, snapshots and the dated-copy archive
under `/mnt/project-files/cfbd-data` when that folder exists. The packaged Windows app uses
`%LOCALAPPDATA%\cfb` and `%APPDATA%\cfb` instead. Overrides: `CFB_DATA_DIR`, `CFB_CONFIG`,
`CFB_PROJECT_FILES`, `CFB_SNAPSHOT_DIR`, `CFB_ASOF_ARCHIVE`, `CFB_WAREHOUSE_DIR`. Keys
(`CFBD_API_KEY`, `R2_*`) are read from the environment first, then from the OS keyring
(Windows Credential Manager) with `pip install -e '.[keyring]'`.

Local Windows app (Streamlit UI, packaged installer, R2 sync, and the local engine that runs the
whole weekly pipeline on the PC on a Task Scheduler schedule): `docs/local-app.md`.

Other commands:

```bash
python -m cfb.ingest --list              # datasets
python -m cfb.features.build --list      # feature catalog: column, description, sources, first season
python -m cfb.features.build --tables    # which tables feature code may read, and from when
python -m cfb.features.build --leak-test # leak-test the registered feature groups on the real data
python -m cfb.core.samples               # rewrite samples/ from the snapshot
python -m cfb.core.playtext              # play text -> data/core/play_{detail,players,penalties} (about 65 s)
python -m cfb.depth_charts import PATH --asof <download time>   # team depth chart CSVs, docs/depth-charts.md
```

## Layout

```
src/cfb/
  client.py, store.py, ingest.py   CFBD client, local store, pull            (ingest: W2)
  engine.py, schedule.py           local engine on the PC (bootstrap, all, light, check, backup); Task Scheduler
  edition.py                       main / experimental edition: names, folders, keys, bucket
  warehouse_sync.py                R2 publish and sync (slim and full channels)
  depth_charts.py                  import team depth chart CSVs (dated)
  core/                            foundations                               (W1)
    game_index.py                  game_index, weeks, teams, game_pregame
    pbp.py, team_game.py           plays with shared definitions; two rows per game
    playtext.py                    play text -> direction, depth, air yards, players, penalties
    policies.py, asof.py           per-table access rules; Warehouse and AsOfView
    leak.py                        scramble-after-cutoff leak test and source lint
    unit_ratings.py                unit_ratings contract and placeholder
    schemas.py, samples.py         prediction and metrics contracts; sample writer
    synthetic.py                   small fake store for tests without the snapshot
  features/
    base.py, build.py              feature group interface, runner, driver   (W1)
    strength/ matchup/ coaching/ context/ market/      feature groups        (W3 to W7)
samples/                           sample prediction and metrics files
docs/foundations.md                contracts in detail
docs/data-coverage.md              what the snapshot holds, by dataset and season
docs/sportradar.md                 Sportradar play-by-play source, limits, coverage
```

`data/` is git-ignored:

| Path | Content |
| --- | --- |
| `data/raw/<dataset>/<key>.json.gz` | exact API responses (source of truth) |
| `data/tables/<dataset>/<key>.parquet` | flattened copy; nested objects become dotted columns, lists become JSON strings |
| `data/core/<table>.parquet` | `game_index`, `team_game`, `pbp`, `teams`, `weeks`, `game_pregame`; `play_detail`, `play_players`, `play_penalties` from `cfb.core.playtext` |
| `data/derived/unit_ratings.parquet` | unit ratings at every cutoff |
| `data/features/<group>.parquet`, `catalog.json` | feature group outputs and their catalog |
| `data/predictions/<model_id>/<season>_<type>_<week>.parquet` | predictions, baselines included |
| `data/eval/<model_id>.json` | metrics |

## Rules for feature code

1. A prediction for a game may use only what was known at the game's cutoff. Regular
   season: the start of the game's calendar week, so week N sees data through week N-1.
   Postseason: 24 hours before kickoff.
2. Feature code reads data only through the `AsOfView` it is given. No `Store`, no
   `read_parquet`, no files. A source lint enforces this in the tests.
3. Season tables as of the pull date (SP+, FPI, SRS, CORE, records, season stats,
   coaches' season stats) are banned for their own season. `view.prior_seasons(name)`
   returns seasons that had ended.
4. Every feature group passes the leak test: results after a cutoff are scrambled, the
   group is rebuilt, and its values for that cutoff must not change.
5. 2026 results are not used for feature selection or tuning. The model is frozen on
   seasons through 2025 and 2026 is scored once.

How to write a group, the table list and the file contracts are in
[docs/foundations.md](docs/foundations.md).

## Data notes

Observed in the 2014 through 2026 snapshot. More in [docs/data-coverage.md](docs/data-coverage.md).

- **Size.** 11,260 games with an FBS team, 11,123 completed, 9,714 of them FBS vs FBS
  (9,085 through 2025). 1,995,021 plays. 660 distinct cutoffs.
- **Scope.** `games`, `drives`, `plays`, and `games_teams` hold games with at least one FBS
  team, so FBS vs. FCS games are included. `lines`, `records`, `stats_game_advanced`,
  `ratings_srs`, and `talent` also contain non-FBS rows; rows of games outside
  `game_index` are dropped by `AsOfView`.
- **Not point-in-time.** Year-level datasets (`ratings_*`, `records`, `coaches`,
  `stats_season_advanced`) are whatever the API holds on the pull date. Dated copies of
  SP+, FPI, SRS and CORE are kept from 2026-11-17 on (`ratings_*_asof`).
- **Week 1.** CFBD files week 0 under week 1, so about six FBS teams a season play twice
  in week 1 and the second game cannot see the first.
- **Pregame Elo.** Point-in-time for FBS vs. FBS games. Null or unreliable in FBS vs. FCS
  games (from 2022 the pregame value equals the postgame value there).
- **Conference flag.** The raw `conferenceGame` flag misses some games (12 in 2025, the
  title games among them). `game_index.conference_game` also counts a shared conference.
- **Plays.** 41 completed FBS vs. FBS games have no plays (2014: 14, 2015: 7, 2016: 13,
  2017: 4, 2018: 2, 2019: 1). None are missing from 2020 on.
- **Lines.** Every FBS vs. FBS game from 2014 through 2025 has a spread except one in 2024.
  Moneylines start in 2021. No provider spans all seasons (`consensus` 2014 to 2022,
  Bovada 2019 on, DraftKings 2023 on).
- **Gaps at the source.** `talent` is empty for 2014. `ratings_srs` has 77 teams for 2020.
  `ratings_sp` includes a `nationalAverages` row. `talent` lacks Air Force and Navy, and
  `returning_production` lacks first-year FBS teams (checked for 2025).
