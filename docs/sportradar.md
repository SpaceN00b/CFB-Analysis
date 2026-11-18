# Sportradar play-by-play (`cfb.sportradar`)

Second play-by-play source next to CFBD: Sportradar NCAA Football API v7
(`https://api.sportradar.com/ncaafb/<trial|production>/v7/en/...`). Its plays carry the
players involved, with roles and per-player stats.

```
python -m cfb.sportradar --estimate                   # hierarchy + schedules, then the pending calls
python -m cfb.sportradar --start 2013 --end 2026 --max-calls 900
python -m cfb.sportradar --coverage
```

## Auth and limits

- Key: `settings.secret("sportradar_api_key")` (`SPORTRADAR_API_KEY`, then the OS keyring), sent
  as `x-api-key`. In the cloud environment the proxy injects it, so no header is sent.
- Sportradar's docs give trial keys 1 request per second and 1,000 requests per rolling 30 days
  (counted per product; the account portal shows usage). The API returns no quota headers, so
  the client cannot read the remaining calls.
- The client spaces requests by `--interval` (default 1.2 s). On 2026-11-18, 2 of 33 requests
  at that spacing needed a retry (status not logged; an unspaced request earlier got 429 "Too
  Many Requests"). The client now counts 429 answers (`throttled`). Every answer counts as a
  call in the `--max-calls` budget (default 900; 0 = no limit).
- 429 "Limit Exceeded" (quota) and 403 "Authentication Error" stop the run. A stopped or partial
  run resumes where it left off: finished games are cached.

## Datasets

| Store name | Key | Table rows |
| --- | --- | --- |
| `sr_hierarchy` | `all` | one per team: division, conference, subdivision (as of the pull) |
| `sr_schedule` | `<year>_REG` | one per game: status, coverage, scheduled time, venue, home/away ids |
| `sr_pbp` | `<year>_REG_<game id>` | one per play: down, distance, yardline, possession, clock, wall clock, score, drive id and offense/defense ids; `details` (players and roles per play step) as JSON text |
| `sr_play_stats` | same as `sr_pbp` | one per play and statistics entry: player id/name/position, team, `stat_type` (pass, rush, receive, defense, return, kick, punt, penalty, ...) and its fields (yards, target, reception, yards_after_catch, tackle, ast_tackle, sack, qb_hit, ...) |

- Bowls and playoff games are weeks of the REG schedule (2025: weeks 17 to 21). `PST` answers
  404, so it is not requested.
- Games pulled: status `closed`, coverage `full`, and at least one team in the current
  hierarchy's FBS division (138 teams on 2026-11-18). Teams that moved up to FBS recently bring
  their FCS-era games too; filter by season at join time.
- Caching: a closed game is never re-pulled; past schedules are final; the current season's
  schedule is re-pulled on every run (1 call).
- Sportradar ids are UUIDs. There is no crosswalk to CFBD game or team ids yet.

## Coverage and call count (schedules pulled 2026-11-18)

Games with play-by-play by the rule above (`--estimate`):

| 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 to date |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 575 | 879 | 883 | 881 | 884 | 894 | 898 | 582 | 976 | 904 | 918 | 923 | 935 | 754 |

Total 11,886 play-by-play calls for 2013 through 2026 week 12, plus 14 schedule calls and
1 hierarchy call. A weekly update is about 65 games plus 1 schedule call. Earliest season is
2013 (the 2012 schedule answers 404). In 2013 only 516 of 750 FBS vs FBS games have full
coverage; from 2014, 762 of 767 or more.

Size per game (2025 sample): raw 46 KB gzip, `sr_pbp` table about 90 KB, `sr_play_stats` about
40 KB, so about 2 GB for all seasons. At 1.2 s per call the full pull takes about 4 to 5 hours.

## Field notes (one game per season, 2026-11-18)

- Iowa State vs Kansas State 2025: 161 plays, 119 tackle roles, intended receiver on 22 of 24
  incompletions, yards after catch on 35 of 57 receptions.
- Ohio State at Navy 2014: 155 plays, 1 tackle role (defense mostly missing), intended receiver
  on 5 of 5 incompletions, yards after catch on 19 of 19.
- `screen_pass`, `play_action`, `run_pass_option` were false on every play of both games:
  probably not charted.
- At most 5 players per play: this is not an on-field participation list.
