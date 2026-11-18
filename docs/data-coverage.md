# Data coverage

What the data snapshot holds, by dataset and season, and when each row becomes known.
Owned by the data workstream (W2). Regenerate the row counts with `python -m cfb.ingest --coverage`.

## Snapshot

| | |
| --- | --- |
| File | the tar named in `cfbd-data/LATEST` in project files (since 2026-11-17 22:10 UTC `cfbd_data_2014-2026_2026-11-17_players.tar`, which adds the player-level datasets below; table of snapshots in `cfbd-data/README.md`) |
| W2 snapshot | `cfbd_data_2014-2026_2026-11-17.tar`, 323,645,440 bytes, 2,740 files, SHA-256 `7e92dc864383af8a041053db511e0bfd67b6cebf4a291bba26873f31be382dda`; every file is byte-identical in the players tar |
| Pulled | Step 1 datasets 2026-11-16 19:12 UTC. Datasets below 2026-11-17 06:17 to 06:27 UTC. API v5.33.3 |
| Current season | 2026 complete through week 11 (last completed game started 2026-11-15 18:00 UTC) |
| Result of the pull | 651 requests, 0 failures |

Restore from the repo root:

```bash
tar -xf /mnt/project-files/cfbd-data/$(cat /mnt/project-files/cfbd-data/LATEST)
```

Each snapshot holds every file of the one before it. Only the newest is needed; older ones may be removed from project files.

## Datasets added in W2

Tables are in `data/tables/<name>/`, raw responses in `data/raw/<name>/`. Column names are the
API's (camelCase). Ingest adds `season`, and `season_type` and `week` where the key has them.

| Name | Endpoint | One row per | Key columns | First season with rows |
| --- | --- | --- | --- | --- |
| `games_players` | `/games/players` | game, team, category, stat type, athlete | `gameId`, `team`, `category`, `statType`, `athleteId` | 2014 |
| `stats_game_havoc` | `/stats/game/havoc` | game and team | `gameId`, `team` | 2014 (partial until 2016) |
| `punting_games` | `/punting/teams/games` | game and team | `gameId`, `team` | 2015 |
| `weather` | `/games/weather` | game | `id` (game id) | 2014 |
| `media` | `/games/media` | game and outlet | `id` (game id), `mediaType`, `outlet` | 2014 |
| `rankings` | `/rankings` | season, week, poll, team | `season`, `seasonType`, `week`, `poll`, `teamId` | 2014 |
| `ratings_elo_weekly` | `/ratings/elo?week=N` | season, regular week, team | `season`, `week`, `team` (name) | 2014 |
| `ratings_elo_preseason` | `/ratings/elo?preseason=true` | season, team | `season`, `team` (name) | 2014 (partial until 2022) |
| `recruiting_teams` | `/recruiting/teams` | class year, team | `year`, `team` (name) | class of 2010 |
| `transfer_portal` | `/player/portal` | portal entry | none; `season`, `firstName`, `lastName`, `origin`, `transferDate` | 2021 |
| `coach_tenures` | `/coaches/tenures` | head-coach tenure | `id`, `coach.id`, `team.id` | all history of 139 teams |
| `ratings_core` | `/ratings/core` | season, team | `season`, `team` (name) | 2016 |
| `ratings_sp_asof`, `ratings_fpi_asof`, `ratings_srs_asof`, `ratings_core_asof` | same as the year tables | season, team, copy date | `season`, `team`, `asof_date` | 2026-11-17 |

`rankings` and `games_players` tables are reshaped from nested responses; the raw files keep
the original nesting. `games_players.stat` is text ("15/22", "5.3").

## Player-level datasets (P1, 2026-11-17)

| Name | Endpoint | One row per | Key columns | Access policy | First season with rows |
| --- | --- | --- | --- | --- | --- |
| `roster` | `/roster?year=` | season, player (FBS and FCS) | `id` (athlete), `team`, `position`, `year` (class year), `recruitIds` | PRESEASON | 2014 |
| `player_usage` | `/player/usage?year=` | season, player | `id`, `team`, `usage.*` | SEASON_FINAL (prior seasons only) | 2014 |
| `ppa_players_season` | `/ppa/players/season?year=` | season, player | `id`, `team`, `averagePPA.*`, `totalPPA.*` | SEASON_FINAL | 2013 (lag year for the 2014 prior) |
| `recruiting_players` | `/recruiting/players?year=` | class year, high-school recruit | `id` (recruit), `athleteId` (55% filled), `stars`, `rating` | PRESEASON (season = class year) | class of 2010 |
| `recruiting_players_juco` | same, `classification=JUCO` | class year, junior-college recruit | as above | PRESEASON | class of 2010 |
| `draft_picks` | `/draft/picks?year=` | draft year, pick | `collegeAthleteId`, `collegeTeam`, `round`, `overall` | PRESEASON (season = draft year; held in April) | 2014 |
| `ppa_players_games` | `/ppa/players/games?year=&week=` | week, player, opponent | `id`, `team`, `opponent`, `averagePPA.all/pass/rush` | WEEKLY, rows known when the calendar week ends | 2014 |
| `player_success_games` | `/stats/player/success/game?year=&week=` | game, passer or rusher | `gameId`, `id`, `passing.plays`, `rushing.plays`, `*.successes` | GAME | 2014 |
| `passing_players_games` | `/passing/players/games?year=&week=` | game, FBS passer | `gameId`, `playerId`, `cpoe`, `ppa`, `averageDepthOfTarget` | GAME | 2025 (empty 2014-2024) |

Notes:

- Athlete ids are strings and match across `roster.id`, the player game tables, `ppa_players_season.id`,
  `games_players.athleteId` and `recruiting_players.athleteId`. A recruit without `athleteId`
  links through `roster.recruitIds` (recruit `id`).
- `roster` is the season's final roster as of the pull; it is used as if known before the
  season (late additions and position changes are not dated). It holds FCS rosters too: about
  16,000 to 19,000 rows a season through 2021, 22,000 to 31,500 from 2022.
- `/ppa/players/games` has no `gameId`. Join it to `player_success_games` on season, season
  type, week, athlete, team and opponent (97.5% of passer rows match in 2014-2015; a few
  hundred duplicate keys exist, drop them). Because rows are dated by week end, a bowl's player
  PPA is visible only after the last bowl.
- `player_success_games` lists passers and rushers only (no receivers). `passing.plays`
  includes sacks and scrambles counted as dropbacks by CFBD (not checked play by play).
- `/stats/player/success/game` answers about one request per 12 seconds and refuses concurrent
  requests (HTTP 429 "Too many concurrent requests for this endpoint"); pull it in one process.

## When a row is known

Feature code must respect these. The cutoff rule is in the build plan.

| Name | Known when |
| --- | --- |
| `games_players`, `stats_game_havoc`, `punting_games` | After the game. |
| `weather` | Completed games hold observed values, which were not known before kickoff. Games not yet played hold forecasts (72 rows for 2026 weeks 12 and 13 in this snapshot); a later pull replaces them with observed values. |
| `media` | Before the game (broadcast schedule). |
| `rankings` | Regular season: the week N poll is published before week N's games. Checked on Notre Dame 2024: AP 5 in week 2 after the week 1 win, 18 in week 3 after the week 2 loss. Postseason week 1 is the final poll after the bowls and playoff, so it is not known before any postseason game. |
| `rankings`, poll `Playoff Committee Rankings` | Published on Tuesday evening US time, after the calendar week has started. In this snapshot (pulled Tuesday 06:19 UTC) 2026 has AP and Coaches for week 12 but the committee ranking only through week 11. |
| `ratings_elo_weekly` | Week N is the rating after week N's games. Use it for week N+1 or later. |
| `ratings_elo_preseason` | Before the team's first game of the season. |
| `recruiting_teams` | Class of year Y is signed before season Y. Values are as of the pull date. |
| `transfer_portal` | `transferDate` (UTC) is the entry date. `destination`, `rating`, `stars` and `eligibility` are filled in later and are as of the pull date. |
| `coach_tenures` | `hireDate`, `startYear` and `isInterim` describe the tenure. `record.*`, `seasons`, `active` and an open `endYear` are as of the pull date and include later games. |
| `ratings_core` | As of the pull date, like SP+, FPI and SRS: final for past seasons, through `throughWeek` for the current one. Same-season use leaks results. |
| `*_asof` | `asof_utc` is the time of the pull. See "Dated rating copies". |

## Rows by season

New datasets. `coach_tenures` is not keyed by season: 2,546 rows. The four `_asof` tables hold
one copy, for 2026: 139, 138, 268 and 138 rows (SP+, FPI, SRS, CORE). `transfer_portal` was
requested for every season 2014 to 2027; 2014 to 2020 and 2027 returned no rows.

| Season | weather | media | rankings | ratings_elo_weekly | ratings_elo_preseason | recruiting_teams | transfer_portal | games_players | stats_game_havoc | punting_games | ratings_core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010 | | | | | | 124 | | | | | |
| 2011 | | | | | | 137 | | | | | |
| 2012 | | | | | | 183 | | | | | |
| 2013 | | | | | | 210 | | | | | |
| 2014 | 858 | 949 | 1,801 | 2,023 | 85 | 231 | 0 | 171,250 | 558 | 0 | 0 |
| 2015 | 863 | 845 | 1,577 | 1,894 | 82 | 231 | 0 | 170,723 | 554 | 1,619 | 0 |
| 2016 | 866 | 846 | 1,603 | 1,898 | 83 | 238 | 0 | 441,459 | 1,714 | 1,600 | 128 |
| 2017 | 866 | 812 | 1,600 | 1,932 | 91 | 233 | 0 | 437,218 | 1,730 | 1,641 | 130 |
| 2018 | 871 | 854 | 1,600 | 1,918 | 88 | 229 | 0 | 435,568 | 1,708 | 1,651 | 130 |
| 2019 | 875 | 860 | 1,704 | 2,050 | 92 | 223 | 0 | 440,765 | 1,774 | 1,658 | 130 |
| 2020 | 560 | 598 | 925 | 1,478 | 116 | 206 | 0 | 275,704 | 1,136 | 1,102 | 127 |
| 2021 | 853 | 886 | 1,400 | 1,916 | 95 | 191 | 1,770 | 443,077 | 1,770 | 1,657 | 130 |
| 2022 | 887 | 881 | 1,950 | 1,965 | 131 | 184 | 2,273 | 435,101 | 2,099 | 1,672 | 131 |
| 2023 | 901 | 933 | 1,902 | 1,995 | 133 | 177 | 2,502 | 446,525 | 2,117 | 1,702 | 133 |
| 2024 | 900 | 1,042 | 2,028 | 2,143 | 134 | 194 | 3,378 | 455,245 | 2,197 | 1,715 | 134 |
| 2025 | 913 | 993 | 2,052 | 2,176 | 135 | 232 | 4,499 | 468,966 | 2,250 | 1,736 | 136 |
| 2026 | 824 | 945 | 1,552 | 1,518 | 138 | 221 | 4,484 | 385,514 | 2,792 | 1,365 | 138 |

Step 1 datasets, unchanged since the 2026-11-16 pull except the four 2026 rating tables.
`venues` is not keyed by season: 852 rows.

| Season | calendar | coaches | drives | games | games_teams | lines | plays | ppa_games | ratings_elo | ratings_fpi | ratings_sp | ratings_srs | records | returning_production | stats_game_advanced | stats_season_advanced | talent | teams_fbs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2014 | 17 | 132 | 22,491 | 868 | 867 | 868 | 158,315 | 1,597 | 130 | 128 | 129 | 128 | 687 | 125 | 1,698 | 128 | 0 | 128 |
| 2015 | 16 | 132 | 22,855 | 870 | 870 | 870 | 160,127 | 1,621 | 130 | 128 | 129 | 128 | 687 | 128 | 1,726 | 128 | 232 | 128 |
| 2016 | 16 | 134 | 22,515 | 873 | 872 | 873 | 158,487 | 1,604 | 130 | 128 | 129 | 128 | 687 | 128 | 1,714 | 128 | 237 | 128 |
| 2017 | 16 | 139 | 22,519 | 874 | 874 | 874 | 158,464 | 1,641 | 132 | 130 | 129 | 130 | 686 | 128 | 1,738 | 130 | 157 | 130 |
| 2018 | 16 | 139 | 22,717 | 884 | 884 | 884 | 160,503 | 1,651 | 131 | 130 | 131 | 130 | 685 | 130 | 1,762 | 130 | 236 | 130 |
| 2019 | 17 | 134 | 22,392 | 888 | 888 | 888 | 159,829 | 1,660 | 131 | 130 | 131 | 130 | 680 | 130 | 1,774 | 130 | 231 | 130 |
| 2020 | 29 | 137 | 14,146 | 568 | 568 | 568 | 102,706 | 1,102 | 127 | 127 | 131 | 77 | 635 | 130 | 1,136 | 128 | 219 | 128 |
| 2021 | 16 | 152 | 21,611 | 887 | 887 | 887 | 157,634 | 1,657 | 130 | 130 | 131 | 130 | 668 | 128 | 1,774 | 130 | 224 | 130 |
| 2022 | 16 | 146 | 21,917 | 896 | 896 | 1,463 | 158,543 | 1,672 | 131 | 131 | 132 | 261 | 670 | 130 | 2,918 | 131 | 233 | 131 |
| 2023 | 16 | 143 | 21,866 | 910 | 910 | 1,416 | 158,745 | 1,702 | 133 | 133 | 134 | 261 | 670 | 131 | 2,984 | 133 | 238 | 133 |
| 2024 | 17 | 152 | 21,766 | 920 | 919 | 1,573 | 162,406 | 1,711 | 134 | 134 | 135 | 265 | 677 | 133 | 3,212 | 134 | 134 | 134 |
| 2025 | 17 | 161 | 21,762 | 934 | 934 | 1,597 | 165,384 | 1,742 | 136 | 136 | 137 | 266 | 679 | 134 | 3,316 | 136 | 134 | 136 |
| 2026 | 16 | 138 | 17,653 | 888 | 752 | 1,477 | 133,878 | 1,381 | 138 | 138 | 139 | 268 | 682 | 136 | 2,814 | 138 | 138 | 138 |

## Per-game coverage

Percent of completed FBS vs FBS games with at least one row. Coverage of all completed games
(including FBS vs FCS) differs from these figures by at most 2.1 points (media, 2016).

| Season | FBS vs FBS games | weather | media | games_players | stats_game_havoc | punting_games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2014 | 760 | 98.9 | 89.2 | 100.0 | 32.1 | 0.0 |
| 2015 | 765 | 99.1 | 89.3 | 100.0 | 32.7 | 99.0 |
| 2016 | 760 | 99.2 | 91.1 | 99.9 | 98.3 | 98.0 |
| 2017 | 776 | 99.1 | 87.8 | 100.0 | 99.0 | 99.5 |
| 2018 | 772 | 98.6 | 91.6 | 100.0 | 96.9 | 99.7 |
| 2019 | 774 | 98.6 | 92.1 | 100.0 | 99.9 | 99.7 |
| 2020 | 534 | 98.5 | 96.4 | 100.0 | 100.0 | 100.0 |
| 2021 | 770 | 96.2 | 95.1 | 100.0 | 99.7 | 100.0 |
| 2022 | 776 | 99.0 | 96.0 | 100.0 | 98.6 | 100.0 |
| 2023 | 792 | 99.0 | 97.5 | 100.0 | 98.7 | 100.0 |
| 2024 | 798 | 98.0 | 99.2 | 100.0 | 99.1 | 99.9 |
| 2025 | 808 | 97.8 | 98.5 | 100.0 | 99.9 | 99.6 |
| 2026 | 629 | 100.0 | 99.8 | 100.0 | 100.0 | 98.7 |

## Season-level coverage

| Season | FBS teams | Elo weekly: weeks | Elo weekly: teams | Elo preseason: teams | CORE: teams | FBS teams with a recruiting class | AP: regular weeks | CFP ranking: weeks | Portal: entries | Portal: % with destination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2014 | 128 | 16 | 130 | 85 | 0 | 128 | 16 | 10-16 | 0 | |
| 2015 | 128 | 15 | 130 | 82 | 0 | 128 | 15 | 10-15 | 0 | |
| 2016 | 128 | 15 | 130 | 83 | 128 | 128 | 15 | 10-15 | 0 | |
| 2017 | 130 | 15 | 132 | 91 | 130 | 130 | 15 | 10-15 | 0 | |
| 2018 | 130 | 15 | 131 | 88 | 130 | 130 | 15 | 10-15 | 0 | |
| 2019 | 130 | 16 | 131 | 92 | 130 | 130 | 16 | 11-16 | 0 | |
| 2020 | 128 | 16 | 127 | 116 | 127 | 128 | 15 | 13-16 | 0 | |
| 2021 | 130 | 15 | 130 | 95 | 130 | 130 | 15 | 10-15 | 1,770 | 59.5 |
| 2022 | 131 | 15 | 131 | 131 | 131 | 130 | 15 | 10-15 | 2,273 | 60.1 |
| 2023 | 133 | 15 | 133 | 133 | 133 | 133 | 15 | 10-15 | 2,502 | 64.2 |
| 2024 | 134 | 16 | 134 | 134 | 134 | 134 | 16 | 11-16 | 3,378 | 78.6 |
| 2025 | 136 | 16 | 136 | 135 | 136 | 136 | 16 | 11-16 | 4,499 | 83.8 |
| 2026 | 138 | 11 | 138 | 138 | 138 | 138 | 12 | 10-11 | 4,484 | 78.1 |

## Notes by dataset

Every figure below was measured on this snapshot.

**Player box scores (`games_players`).** 5,007,115 rows. One completed FBS vs FBS game has no
rows (Duke vs Army, 2016 week 6). Categories `passing`, `rushing`, `receiving`, `kicking` and
`punting` are present for every season. `defensive` and `fumbles` start in 2016. Stat type
`QBR` is missing from most passing lines in 2018 (3% present), 2020 (2%) and 2019 (64%).

**Havoc (`stats_game_havoc`).** 2014 and 2015 cover 32% of FBS vs FBS games. From 2016 the
lowest season is 96.9% (2018). One row has a rate above 1. From 2022 the endpoint also returns
games without an FBS team.

**Punting (`punting_games`).** No rows for 2014. FBS teams only.

**Weather.** Temperature and wind are filled for every row on a completed game. 4.1% of games
are flagged indoors. `pressure` is 0 in 588 rows (568 in 2026, 20 in 2025) and null in 339.
`weatherCondition` is null for 94% to 100% of rows in 2014 to 2017, about half in 2018 to
2021, and under 10% from 2022. Units are not stated in the API spec. Inferred from the values: temperature and dew
point in °F (median 68.9), wind in mph (median 7.0, max 34.5), precipitation in inches,
pressure in hPa.

**Media.** 9% to 12% of FBS vs FBS games in 2014 to 2017 have no row; under 2% from 2024.

**Polls (`rankings`).** AP and Coaches polls exist for every regular week of 2014 to 2025
except 2020 week 2, plus a final postseason poll. The committee ranking starts in week 10 or
11 (13 in 2020). The table also holds FCS, Division II and Division III polls.

**Weekly Elo.** The week N rating equals the postgame Elo on `games` of the team's last game
in weeks 1 to N in all 24,759 team-weeks checked. Week 1 includes "week 0" games, so 78 FBS
team-weeks (all in week 1, 2016 to 2026) contain two games. There are no postseason rows; the
season-final value is in `ratings_elo`.

**Elo around FBS vs FCS games, 2022 on.** In `games`, every FBS vs FCS row from 2022 on (605
team-games) has pregame Elo equal to postgame Elo, and in 366 of 390 that value differs from
the team's previous postgame Elo (median gap 11 points, 90th percentile 65, maximum 342). The
gap has no relation to the game's result (its sign matches win or loss in 49% of cases).
Weekly Elo carries the same value for that week. Consequence: for an FBS vs FBS game, weekly
Elo of week N-1 equals the pregame Elo on `games` in 100% of cases through 2020 and 97.8%
overall; 363 of the 380 mismatches follow an FCS game. After an FCS game, the next game's
pregame Elo on `games` equals the FCS-game value in 40% of cases and the earlier value in the
rest. For an FBS vs FCS game from 2022 on, the pregame field is therefore not a reliable
rating entering the game; the team's previous postgame Elo is the last consistent value.

**Preseason Elo.** Present for every team that opened against an FBS opponent and for 42% of
those that opened against an FCS opponent. 35 to 46 FBS teams per season are missing in 2014
to 2019 and 2021, 11 in 2020, at most 1 from 2022.

**Recruiting.** Classes 2010 to 2026. Every FBS team-season from 2014 has a class row except
Florida International 2022. Rows include non-FBS teams.

**Transfer portal.** No rows before the 2021 season. The middle 98% of a portal season's
entries fall between August to December of the previous year and May to July of the season
year (2026: first entry 2025-09-25, last 2026-10-14, 84% dated January 2026). Season 2027 had no rows on 2026-11-17. Destination is filled for
59.5% (2021) to 83.8% (2025) of entries.

**Coach tenures.** 2,546 tenures for the 139 teams that were FBS in any season from 2014; 558
overlap 2014 or later. `isInterim` is set on 62 of those, 60 of them starting in 2022 or
later (2022: 14, 2023: 9, 2024: 15, 2025: 21, 2026: 1), so earlier interim coaches are not
flagged: LSU 2016 lists Ed Orgeron as one non-interim tenure dated from his permanent hire.
`coaches` (step 1) shows 4 to 9 team-seasons per year with two head coaches in 2014 to 2020,
which is the source to use for in-season changes before 2022. `effectiveStart` (first game
coached) is filled for 28 of the 558, 23 of them interim. All 30 tenures starting in 2026
show 0 games and `coaches` shows no 2026 team with two coaches, so 2026 games were not yet
attributed on the pull date. Whether CFBD records an in-season change during the season is
unknown.

**CORE.** Empty for 2014 and 2015. Past seasons are through the postseason; 2026 is through
regular week 11.

## Dated rating copies

SP+, FPI, SRS and CORE are stored by CFBD as one current value per team and season. From
2026-11-17 every ingest run pulls them for the current season (August starts a new season)
and keeps a copy in `ratings_sp_asof`, `ratings_fpi_asof`, `ratings_srs_asof` and
`ratings_core_asof`, keyed `<season>_<YYYYMMDD>`, with columns `asof_date` and `asof_utc`.
A second run on the same UTC date replaces that date's copy.

Containers are ephemeral, so the copies also live in project files under `cfbd-data/asof/`.
Pass that folder on every run:

```bash
python -m cfb.ingest --start 2026 --end 2026 --force --asof-archive /mnt/project-files/cfbd-data/asof
```

The run first restores archived copies missing from `data/`, then saves the copies it made.
Without `--asof-archive` (or `$CFB_ASOF_ARCHIVE`) the copies exist only in the local `data/`.

Limits:

- There are no copies before 2026-11-17, so seasons through 2025 and 2026 weeks 1 to 11 have
  no point-in-time SP+, FPI, SRS or CORE.
- A copy shows what CFBD held at `asof_utc`. A copy pulled on Monday or Tuesday is after the
  start of that calendar week, even when no game of the week has been played. Only
  `ratings_core_asof` states the week it covers (`throughSeasonType`, `throughWeek`).
- The first copy (`asof_utc` 2026-11-17T06:19:18Z) was taken after all week 11 games and
  before any week 12 game.

## Refreshing

- Weeks that have not started are not requested. `ratings_elo_weekly` is requested only for
  weeks that have ended, also under `--force`, because the API answers a later week with the
  latest ratings.
- A week in progress is stored as it stands and is not requested again without `--force`.
- One forced current-season run is about 240 requests at full season: 56 for the step 1
  datasets, 45 for the new season and week datasets, 138 for `coach_tenures`.
- The full 2014 to 2026 pull of the datasets added here was 651 requests.
