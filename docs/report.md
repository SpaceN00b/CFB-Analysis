# Weekly report and dashboard (W9)

Code: `src/cfb/report/` (bundle, Markdown report, dashboard page), `src/cfb/run_week.py`
(the weekly pipeline), `dashboard/config.toml` (which model the report is about).

## Commands

```bash
python -m cfb.run_week                                   # the current calendar week, full pipeline
python -m cfb.run_week --publish /mnt/project-files/reports
python -m cfb.run_week --season 2026 --week 12 --no-refresh --no-features   # from local data, ~5 s
python -m cfb.run_week --season 2025 --week 5            # a past week (no refresh for past seasons)
python -m cfb.run_week --primary model_v1                # report on another model for this run
```

```bash
python -m cfb.rankings --season 2026 --week 12           # model top 50 vs polls, Markdown to stdout
python -m cfb.rankings --season 2026 --week 12 --source elo --top 25
```

`--now 2026-11-17T16:00:00Z` pretends the run happens at that time (picks the week).
`--out DIR` writes the report elsewhere. `--no-write-predictions` leaves `data/predictions` alone.

## What a run does

| Step | What | Notes |
| --- | --- | --- |
| 1 data | Restores the snapshot tar when `data/raw` is empty | Default tar: `cfbd-data/LATEST` in project files; `--snapshot PATH` |
| 2 refresh | `cfb.ingest` for the target season, `--force`, with the dated-copy archive | Default for the current season only. Refuses without an archive (`--asof-archive`, default `$CFB_ASOF_ARCHIVE` or `/mnt/project-files/cfbd-data/asof`), because weekly SP+/FPI/SRS/CORE values cannot be pulled again. About 240 API calls. |
| 3 core | `build_core` | Always after a refresh |
| 4 features | `unit_ratings` (all cutoffs) and every feature group for the season | `run.build_features` in the config; about 2.5 minutes on the 2026 snapshot |
| 5 predict | Every model in the config, for the season's games up to the target week | Built-in baselines and producers write `data/predictions/<model_id>/`; other models are read from there |
| 6 report | `data/reports/<season>_<type>_<week>/` | `report.md`, `dashboard.html`, `bundle.json`, `metrics/<model_id>.json` |
| 7 publish | Copies the folder to `--publish/<key>/` and the newest files to `--publish/latest/` | Optional |

Season-to-date metrics go to the report folder, not `data/eval/`, which holds the 2018-2025
validation documents written by `cfb.eval`.

## Which week, and "as of" when

- Regular season: the CFBD calendar week containing the run time. The report is as of the
  week's cutoff (start of the week, Monday 08:00 UTC): predictions use nothing later, and
  "last week" and the season metrics use results visible then.
- Postseason (every bowl and playoff game is CFBD postseason week 1, each with its own
  cutoff): a live run takes the postseason games that kick off in the next 8 days and calls
  the previous 8 days "last week". `--season Y --season-type postseason --week 1` for a past
  season takes the whole postseason.
- Lines for games not yet played are the lines at the last data pull, not closing lines.
  The report says when that pull was.

## Switching the report to the model

Edit `dashboard/config.toml`:

```toml
[models]
primary = "model_v1"

[producers]
model_v1 = "cfb.predict:produce"
```

A producer is called as `produce(warehouse, games)` and returns the frame
`cfb.eval.splits.walk_forward` returns (prediction schema plus `season`, `season_type`,
`week`); `games` are schedule rows of the season through the target week. Without a
producer the run reads `data/predictions/<model_id>/` as it stands. Nothing else changes:
the report, dashboard and metrics follow `primary`. `tests/test_report.py` checks this.

## Report contents (`report.md`)

1. Season to date, FBS vs FBS: every model on the games all of them predicted (log loss,
   Brier, accuracy), then the model against each comparison with a paired bootstrap
   (difference, SE, 95% interval, share of resamples in which the model is better).
   Validation numbers from `data/eval/<model_id>.json` when they exist.
2. Last week: per-model record and log loss, the five biggest misses, all results with the
   closing spread.
3. This week: largest differences from the market, every game (kickoff in Eastern time,
   pick, model probability, market probability, edge, spread, top reason), then the top
   three reasons for every pick.
4. Spread and total: only when the model fills `pred_margin` (phase 2). Columns: model
   margin against the spread, model total against the line total.

Market probability for a game: the de-vigged moneyline where it exists, else the closing-spread
baseline (`run.market_models`). Edge = model probability for its pick minus the market's,
in percentage points.

## Dashboard (`dashboard.html`)

One static page, data embedded, no network access needed; light and dark themes follow the
system. Tabs:

- **This week**: sortable table with search, conference, FBS-only and ranked-only filters.
  Clicking a game opens its matchup.
- **Model vs baselines**: season tiles, cumulative log loss by week (crosshair tooltip),
  paired comparisons, weekly table, validation table.
- **Calibration**: reliability chart and bins for any model, season to date.
- **Last week**: results with the pick, whether it was right, market and log loss.
- **Teams**: record, AP rank, profile features with FBS percentiles (power rating, offense,
  defense, Elo, schedule strength, strength of record, form, style, coaching, QB continuity),
  season game log with the model's win probability.
- **Rankings**: model top 50 next to AP, Coaches and CFP ranks (see below); sortable, with
  a poll selector for the gap column and the disagreement lists.
- **Matchup**: every model's home win probability, the model's reasons as signed bars,
  game-specific edges (unit matchups, expected run-rate shift, rest, travel) and both
  teams' profiles side by side.

Team profile and matchup features come from `data/features/*.parquet` when they were built
(step 4); the columns shown are listed in `PROFILE_FEATURES` and `MATCHUP_FEATURES` in
`cfb/report/bundle.py`. Missing columns are skipped.

## Model rankings vs polls

Code: `src/cfb/report/rankings.py`; `[rankings]` in `dashboard/config.toml`. The model's top
50 FBS teams by a team rating at the report's cutoff, next to the latest AP, Coaches and
playoff committee top 25 visible then. It is the last section of `report.md`, the
**Rankings** tab of the dashboard, and `rankings` in the bundle.

For reading only. The rankings are built after every model has predicted, live in
`cfb.report` (feature discovery reads `cfb.features` only) and are written nowhere but the
report, so they cannot change a prediction. `tests/test_rankings.py` checks this.

| Column | Meaning |
| --- | --- |
| Rank, Move | Model rank among FBS teams; places gained since the ranking one week earlier |
| Rating, Off, Def | The source's rating. `power`: net points per game against an average FBS team on a neutral field, and its offense and defense parts |
| SOR rank | Strength of record: wins minus the wins a 25th-best FBS team would expect against the same schedule and sites (same definition as `str_sor`) |
| AP, Coaches, CFP | Poll rank, each from its own latest week. The committee ranking comes out on Tuesday evening, so a Monday run shows the previous week's |
| Gap vs poll (dashboard) | Poll rank minus model rank; positive means the model rates the team higher |

Teams in a poll's top 25 that the model ranks below 50 are listed after the top 50, so every
poll team can be compared. Per poll: how many of its top 25 are in the model's top 25, the
Spearman correlation between poll rank and model rank over the poll's 25 teams, the mean
absolute rank gap, the poll teams the model ranks lowest, and the model's top 25 teams the
poll leaves out.

**Rating source.** `source = "power"` (W3 power rating, default), `"elo"` (in-house Elo) or
`"module:function"`. A function source is called as `rate(view)` with the AsOfView at the
report's cutoff and returns `team_id`, `rating` (higher is better) and optionally `off`,
`def`, `se`, `games`; `label` and `unit` attributes on the function name it in the report.
That is how the trained model replaces the power rating once it can rate a team on its own
(for example, expected margin against an average FBS team on a neutral field).
`source = ""` turns the section off. A source that fails leaves the section out and adds a
note to the report; it does not stop the run.

The power rating is predictive (opponent-adjusted points margin); SOR is résumé-based (wins
against the schedule played), which is closer to what polls describe.

## Bundle (`bundle.json`)

Everything the report and dashboard show, as one JSON document (`bundle_version` 1):
`meta`, `models`, `games` (this week), `previous` (last week with results), `season`
(own-game blocks, common-game blocks, weekly series, comparisons, calibration),
`validation`, `teams`, `feature_catalog`, `rankings` (null when turned off). It is the record of what a run predicted.

## Room for phase 2

`pred_margin` and `pred_total` are carried for every game. When the primary model fills
them, the report adds the spread-and-total section and the dashboard adds model margin and
total columns. Last week's results already carry `home_cover_margin` (home margin plus the
closing spread) and `total_vs_line`, so scoring against the spread needs no new data.
