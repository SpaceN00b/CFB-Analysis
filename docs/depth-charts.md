# Depth charts

Team depth chart CSVs that Josiah downloads from a sign-in site, imported as a dated table.
The raw CSVs and the table are not in the repo; they live in project files (see Snapshot).

## Import

```bash
python -m cfb.depth_charts import PATH [PATH ...] --asof 2026-11-19T04:06:01Z
python -m cfb.depth_charts summary
```

`PATH` is a CSV or a folder of them. `--asof` is when the charts were downloaded (UTC);
without it the file's modified time is used. Requires `teams_fbs` and `roster` for the
season (restore the CFBD snapshot first).

Input columns: `Team, Unit, Scheme, Position, Depth, Number, Player`, for example
`Georgia Bulldogs,Offense,Pro Spread,WR-X,First Team,6,"Canion, Isiah"`.

Output:

| Path | Content |
| --- | --- |
| `data/raw/depth_charts/<asof_date>/<file>.csv` | the CSV exactly as given |
| `data/tables/depth_charts/<asof_date>_<team>.parquet` | one row per listed player |

Importing a team again on the same date replaces its partition; a later date adds one, so
weekly downloads build a history.

## Table `depth_charts`

| Column | Meaning |
| --- | --- |
| `season`, `asof_utc`, `asof_date` | season, and when the chart was downloaded |
| `team`, `source_team`, `source_file` | CFBD school name; name and file as given |
| `unit` | `offense`, `defense`, `special_teams`, `bench` |
| `scheme` | as given: "Pro Spread", "Air Raid", "3-4", "Multiple 4-2-5", ... |
| `position` | the team's own label: WR-X, WR-Z, WR-SL, LT ... RT, DE, NT, JACK, WLB, MLB, LCB, SS, FS, RCB, NB, PK, KR ... |
| `pos_group` | QB, RB, WR, TE, OL, DL, EDGE, LB, DB, ST, BENCH (`POS_GROUPS` in `cfb/depth_charts.py`) |
| `depth`, `depth_label` | 1 to 5 ("First Team" to "Fifth Team"); null on the bench |
| `starter` | depth 1 on offense or defense |
| `jersey`, `player_raw`, `first_name`, `last_name`, `suffix`, `player` | as given, and split ("Greene III, Earnest" -> Earnest / Greene / III) |
| `athlete_id`, `match`, `roster_position` | CFBD roster id and position; `match` is how it was found |

A player can appear more than once (a starter who also returns kicks, a two-way player).

**Team names.** "School Mascot" or "alternate name Mascot" from `teams_fbs`
("Central Florida Knights" -> UCF). Exceptions go in `TEAM_OVERRIDES`
("Mississippi Rebels" -> Ole Miss). An unknown team stops the import.

**Positions.** Labels are team-specific. EDGE is the fourth rusher of a four-man front or a
3-4 outside backer (JACK, BUCK, LEO, RAZOR, WOLF, RUSH, DOG, BAN, LOLB, ROLB). A label not in
`POS_GROUPS` is reported and takes its group from the matched player's CFBD roster position.
Vanderbilt labels its line by quick and strong side (QT QG C SG ST), not left and right.

**Player match** (same team, 2026 CFBD roster): jersey and last name, then full name (first
and last run together), then a last name unique on the team with the same first initial,
then jersey and first name. Two roster rows that fit equally are left unmatched unless
`PLAYER_OVERRIDES` names the id (`match` = `manual`).

## Point in time

Policy `DATED` on `asof_utc`: a row is visible to a view whose cutoff is at or after the
download time. The model's regular-season cutoff is the start of the game's calendar week,
so a chart downloaded mid-week is first visible to the next week's predictions (the
2026-11-19 download is after week 12's cutoff of 2026-11-16 08:00 UTC and is visible from
week 13). Using it for the current week would need a manual path like the QB-out override.

The charts are the site's current depth charts, not past ones, so there is no history
before the first download and no backtest yet.

## State (2026-11-19)

| Batch | Downloaded (asof_utc) | Teams | Rows | Roster match |
| --- | --- | --- | --- | --- |
| 1 | 2026-11-19T04:01:16Z (upload time) | 15: SEC except Alabama and Arkansas, plus Illinois | 1,339 | 100% |
| 2 | 2026-11-19T04:06:01Z (latest file time) | 17: Alabama, Arkansas, Big 12 except Arizona | 1,395 | 99.9% |

Unmatched: West Virginia's NaQuari Rogers (not on the CFBD roster). Utah's Nick Brown (#13,
defense) has two identical 2026 CFBD roster rows; the override picks 5226164, the one with
2026 Utah game stats (518520 has none).

**Compared with the playing-time starters** built from the CFBDepth stat exports
(`starters.csv`, 87% against the 7 lineup PNGs): they hold 599 of the 710 first-team
starters on these 32 charts (84%). By group: DB 89%, QB 88% (28 of 32), OL 87%, DL 82%,
LB 82%, WR 82%, RB 79%, TE 78%, EDGE 77%. The four QB differences: Arizona State (chart
Jake Fette, usage Cutter Boley), Kansas State (Avery Johnson / Blake Barnett), Mississippi
State (AJ Swann / Parker Puckett), Texas Tech (Thomas Castellanos / Will Hammond). Names are
compared by first initial and last name, so spelling differences between the sources count
as misses. Script: `exploration/depth-charts/compare_starters.py` in project files.

## Snapshot

`/mnt/project-files/depth-charts/` holds the raw CSVs and the table as a tar (file name in
`LATEST`). Restore from the repo root: `tar -xf /mnt/project-files/depth-charts/$(cat /mnt/project-files/depth-charts/LATEST)`.
The table is under `data/tables/`, so `cfb.warehouse_sync` publishes it with the rest.

## Source terms

Josiah reports the site builds its own depth charts and that its terms do not ban use in a
model. Not verified here: the site renders its pages in the browser, so its terms page
could not be read. Ourlads data is not used (its terms ban AI and ML use).
