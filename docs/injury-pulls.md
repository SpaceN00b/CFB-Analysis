# Scheduled injury report pulls

Josiah (2026-11-20): pull the injury reports Wednesday, Thursday and Friday evenings, plus 90 minutes
before kickoff for every game where a high-impact player is on the pregame report and is not known to be
out for the season; update the database each time.

## Sources

- **Conference availability reports (HD Intelligence)**: SEC, ACC, Big 12, Big Ten; public JSON, no
  sign-in (`cfb.hdi`). Conference games only: no Notre Dame, no non-conference games, no "out for
  season" status. On Thursday 2026-11-19 the updates went up at 8:00 PM ET (Big Ten, ACC), 8:10 PM ET
  (SEC) and 10:00 PM ET (Big 12), so the evening pulls run at **10:30 PM ET** (Josiah chose it over 7 PM).
- **CFBDepth player-updates list** (all FBS teams, the larger dataset): needs Josiah's paid sign-in, so he
  downloads it by hand about weekly to `D:\claude\cfbdepth-player-updates.csv`. The evening runs copy a newer
  file, when his PC is reachable, to `/mnt/project-files/injury-reports/cfbdepth_<YYYY-MM-DD>/`; the pull
  uses the newest such folder. Where both sources list a player for this week's game, the HDI pull wins if it
  is from the list's download date or later (newer report); otherwise the list wins. HDI fills the list's gaps.

## Run

    python scripts/injury_pull.py run --kind scheduled                         # evening pulls
    python scripts/injury_pull.py run --kind kickoff --slot 2026-11-21T17:00:00Z
    python scripts/injury_pull.py run --kind manual [--week N] [--no-publish] [--no-package]

Steps (see the script docstring): pull HDI into `injury-reports/hdi_pulls/<UTC stamp>/` (raw current
reports and archive, `merged/` = archive with the current reports written in, `summary.md`, `plan.json`);
prepare `data/` from the week's cached base (`injury-reports/pull-base/base_<season>_week<N>.tar`, built
by the first pull of the week: LATEST snapshot, CFBD refresh when last week's results are missing, using
a scratch copy of the asof archive, features, depth charts, impact inputs); write
`tables/availability_reports/<stamp>.parquet` for every pull and publish only that table onto the
bucket's current warehouse manifest (`warehouse_sync.publish(only=...)`; a later full publish keeps it);
re-run `depth_impact_select.py`, `defense_impact_select.py`, `depth_impact_week.py` and
`availability_export.py --publish` (extras/availability/); plan the kickoff pulls.

**High impact** = value over the replacement of at least 1.0 point per game: offensive starters from the
PR 35 depth-chart scores (`vor`), defensive regulars from the PR 37 scores (larger of `vor_fill`,
`vor_own`). Week 12 2026: 149 of 816 offensive starters (mostly QBs) and 43 of 1,632 defenders.
**Out for the season** = CFBDepth status "Out for Season" in the newest list, or an expected return of
"season"/2027 in the 2026-11-19 per-player checks. Any listed status counts as "on the report" (Out,
Out - 1st half, Doubtful, Questionable, Probable). One kickoff pull per kickoff time (games at the same
time share it); the evening runs create them as one-time scheduled tasks.

## Table `availability_reports`

One partition per pull, listed players only: `pulled_utc, season, conference, hdi_game_id, report_type,
posted_utc, game_date, kickoff_utc, team, opponent, home, player, pos, number, status, exempt_status,
team_id, athlete_id`. Policy DATED on `posted_utc` (when the conference posted the report).

## Limits

- The 7:00 PM ET reports were not available at 7:00 PM on 2026-11-19 (one day measured).
- When the HDI game-day report posts relative to the 90-minute pull is unknown; the summary shows each
  report's posted time.
- "Out - (1st Half)" (targeting) has no absence rate, so the model treats that player as available.
- The scheduled runs check out this branch; if it is renamed or merged, update the scheduled task prompts.
