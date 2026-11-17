# Context features (W6)

Package `src/cfb/features/context/`, prefix `ctx_`. Ten feature groups, all `per_cutoff`,
all read through the `AsOfView` only. Check script: `scripts/context_check.py`.

## Groups and columns

### ctx_schedule (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_site` | +1 at home, -1 on the road, 0 at a neutral site | schedule | 2014 |
| `ctx_rest_days` | Days since the team's previous game this season (empty for its first game) | schedule | 2014 |
| `ctx_off_bye` | 1 if rest is 13 days or more (coming off a bye) | schedule | 2014 |
| `ctx_short_week` | 1 if rest is 5.5 days or less | schedule | 2014 |
| `ctx_games_last_14d` | Games the team played in the 14 days before kickoff | schedule | 2014 |
| `ctx_game_number` | Game number in the team's season, 1 for its first game | schedule | 2014 |
| `ctx_days_since_regular` | Postseason only: days since the team's last regular-season game | schedule | 2014 |

### ctx_season_phase (game)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_postseason` | 1 for bowls and playoff games | schedule | 2014 |
| `ctx_neutral` | 1 at a neutral site | schedule | 2014 |
| `ctx_conference_game` | 1 for a conference game | schedule | 2014 |
| `ctx_start_time_tbd` | 1 if the kickoff time was still to be decided | schedule | 2014 |
| `ctx_season_week` | Week of the season counted from the first calendar week, by kickoff date (continues through the postseason; week 0 games count as week 1) | weeks, schedule | 2014 |
| `ctx_season_2020` | 1 in the 2020 season (limited crowds, irregular schedules) | schedule | 2014 |

### ctx_travel (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_travel_km` | Great-circle distance from the team's home stadium to the venue (km); at neutral sites this is the team's distance from the venue | schedule, venues, teams_fbs | 2014 |
| `ctx_tz_shift_h` | Venue UTC offset minus the team's home UTC offset at kickoff (hours; positive = the team travelled east) | schedule, venues, teams_fbs | 2014 |
| `ctx_elev_gain_m` | Venue elevation minus the team's home elevation (m) | schedule, venues, teams_fbs | 2014 |
| `ctx_surface_change` | 1 if the venue surface (grass or turf) differs from the team's home surface | schedule, venues, teams_fbs | 2014 |
| `ctx_dome_change` | 1 if the venue is indoor and the team's home is not, or the reverse | schedule, venues, teams_fbs | 2014 |
| `ctx_body_clock_hour` | Kickoff hour on the team's home clock (0-24; empty when the kickoff time was TBD) | schedule, venues, teams_fbs | 2014 |
| `ctx_body_clock_early` | 1 if kickoff is before 11:00 on the team's home clock | schedule, venues, teams_fbs | 2014 |
| `ctx_at_own_stadium` | 1 if the game is at the team's own home stadium | schedule, venues, teams_fbs | 2014 |

### ctx_venue (game)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_venue_elev_m` | Venue elevation (m) | schedule, venues, teams_fbs | 2014 |
| `ctx_venue_dome` | 1 if the venue is indoor | schedule, venues, teams_fbs | 2014 |
| `ctx_venue_grass` | 1 if the venue has natural grass | schedule, venues, teams_fbs | 2014 |
| `ctx_kickoff_local_hour` | Kickoff hour at the venue (0-24; empty when TBD) | schedule, venues, teams_fbs | 2014 |
| `ctx_night_game` | 1 if kickoff is at or after 18:00 local time | schedule, venues, teams_fbs | 2014 |
| `ctx_away_travel_km` | Distance the away team travels to the venue (km) | schedule, venues, teams_fbs | 2014 |
| `ctx_travel_km_diff` | Away team's travel minus home team's travel (km); at neutral sites positive means the venue is closer to the home-listed team | schedule, venues, teams_fbs | 2014 |

### ctx_home_road (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_home_resid` | Team's mean home margin above expectation (Elo + league home edge), last 4 seasons, shrunk toward 0 (points) | team_game | 2014 |
| `ctx_road_resid` | Team's mean road margin above expectation, same window and shrinkage (points) | team_game | 2014 |
| `ctx_home_games_n` | On-campus FBS home games behind ctx_home_resid | team_game | 2014 |
| `ctx_road_games_n` | On-campus FBS road games behind ctx_road_resid | team_game | 2014 |

### ctx_home_field (game)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_league_hfa` | League home edge (points) fitted at the cutoff on FBS vs FBS games of the last 4 seasons, controlling for the Elo gap | team_game | 2014 |

### ctx_crowd (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_crowd_prior_att` | Mean attendance (thousands) at the team's on-campus home games in the previous season (2019 for 2021) | games, venues, teams_fbs | 2015 |
| `ctx_crowd_prior_fill` | Same games: attendance over venue capacity | games, venues, teams_fbs | 2015 |
| `ctx_crowd_season_att` | Mean attendance (thousands) at the team's on-campus home games already played this season | games, venues, teams_fbs | 2014 |
| `ctx_crowd_capacity` | Capacity of the team's home stadium (thousands) | games, venues, teams_fbs | 2014 |

### ctx_portal (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_portal_out_since_last` | Players from the team who entered the portal after the kickoff of its previous game and by the cutoff | transfer_portal, teams, schedule | 2020 |
| `ctx_portal_out_rated_since_last` | Same, players rated 3 stars or more | transfer_portal, teams, schedule | 2020 |
| `ctx_portal_out_since_regular` | Players from the team who entered the portal after its last regular-season game and by the cutoff (postseason only) | transfer_portal, teams, schedule | 2020 |

### ctx_qb (team)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_qb_leader_share` | Share of the team's season pass attempts thrown by the passer who led its last game | games_players, schedule | 2014 |
| `ctx_qb_changed` | 1 if the leading passer of the last game differs from the one of the game before | games_players, schedule | 2014 |
| `ctx_qb_passers_10` | Passers with 10 or more attempts this season | games_players, schedule | 2014 |
| `ctx_qb_leader_games` | Games this season led in attempts by the passer who led the last game | games_players, schedule | 2014 |
| `ctx_qb_games_seen` | Team games this season with a visible passing box score | games_players, schedule | 2014 |

### ctx_weather (game)

| Column | Meaning | Sources | From |
| --- | --- | --- | --- |
| `ctx_wx_indoors` | 1 if the game is played indoors | weather | 2014 |
| `ctx_wx_temp_f` | Temperature at kickoff (deg F) | weather | 2014 |
| `ctx_wx_wind_mph` | Wind speed (mph) | weather | 2014 |
| `ctx_wx_precip_in` | Precipitation (inches) | weather | 2014 |
| `ctx_wx_snow_in` | Snowfall (inches) | weather | 2014 |
| `ctx_wx_humidity` | Relative humidity (%) | weather | 2014 |

## Definitions and choices

- **Site and rest** come from the schedule (kickoff times are known in advance), not from
  results. Rest counts only games of the same season, so a team's first game has no rest
  value. A game of the same slate that kicks off earlier counts (week 0 games are filed under
  week 1 and share its cutoff). Bye = rest of 13 days or more, short week = 5.5 days or
  less, the rules `effects.py` used. One scheduled game was never played (2024 week 5,
  App State vs Liberty); it still counts as the previous game for those two teams.
- **Stadiums.** FBS teams: `teams_fbs` `location.*` of the latest season up to the cutoff
  season. Other teams: the venue of most of their on-campus home games in their latest
  season with one in the visible schedule. **Game venue:** `venues` by `venue_id`; for an
  on-campus game a missing fact (time zone, elevation, surface) comes from the home team's
  stadium. A time zone still missing is taken from the nearest venue or stadium with a known
  zone within 400 km. UTC offsets are computed at kickoff, so daylight saving time is
  handled (`zoneinfo`).
- **Indoor** in `ctx_venue_dome` and `ctx_dome_change` is the stadium's `dome` flag. The
  weather table's `gameIndoors` is kept separately in `ctx_wx_indoors`.
- **Kickoff hours** are empty when `start_time_tbd` is set.
- **Home and road residuals.** At each cutoff, OLS of the home margin on 1, on-campus
  indicator and the CFBD pregame Elo gap over visible FBS vs FBS games of the current and
  three previous seasons (at least 200 games, else empty). A team's home (road) residual is
  the sum of its on-campus home (road) residuals over n + 146. k = 146 comes from seasons
  2014 to 2025: per-game residual SD 16.7 points, between-team variance of 4-season home
  means about 1.9 points², so k = 16.7² / 1.9. Same model gives `ctx_league_hfa`.
- **Crowd.** Attendance of the game itself is a result and is never used. Prior-season
  values skip 2020 (2021 games look back to 2019). Fill = attendance over the capacity in
  `venues`. Zero or missing attendance is treated as unknown.
- **Portal.** Entries whose `origin` is the team (name matched to `teams`), with
  `transferDate` after the kickoff of the team's previous game (or last regular-season
  game) and at or before the cutoff. Empty before 2020-08-01, the first portal date in
  the data; zero after that means no entries.
- **QB continuity.** Passing `C/ATT` rows of the team's games this season whose box score
  is visible at the cutoff; team rows (negative athlete ids) dropped. The "leader" of a
  game is the passer with the most attempts. All empty for a team's first game of a season.
- **Weather.** As CFBD reports it: observed for played games, forecast for upcoming ones
  (only as fresh as the last ingest). Stored for phase 2.

## Coverage on the 2026-11-17 snapshot

Share of team rows (or games) with a value, all games 2014 to 2026 and FBS vs FBS games.

| Group | All | FBS vs FBS | Why empty |
| --- | --- | --- | --- |
| `ctx_schedule` rest columns | 88% | 94% | first game of a team's season |
| `ctx_season_phase` | 100% | 100% | |
| `ctx_travel` | 94% | 100% (elevation, surface 99.8%) | FCS teams without a known home venue |
| `ctx_venue` | 99.6% to 100% | 99.6% to 100% | TBD kickoffs, a few venues without elevation |
| `ctx_home_road`, `ctx_home_field` | 97% | 98% | early 2014 cutoffs (under 200 games to fit) |
| `ctx_crowd` prior season | 86% | 91% | 2014 (no 2013 data), teams without home games |
| `ctx_crowd_season_att` | 75% | 83% | no home game played yet this season |
| `ctx_portal` since last game | 47% | 50% | before 2020-08-01; `since_regular` is postseason only (2.4%) |
| `ctx_qb` | 87% (changed: 79%) | 94% (87%) | first (second) game of a team's season |
| `ctx_weather` | 98% | 98% | CFBD has no weather row |

Build time on the full snapshot, all 660 cutoffs: about 7 minutes for the ten groups
(QB continuity about 100 s, the rest 7 to 40 s each).

## Check against the plan doc (seasons 2014 to 2025)

`python scripts/context_check.py` rebuilds the groups for every FBS vs FBS game of 2014 to
2025 and reruns the regressions of `effects.py` with the `ctx_` columns. All 13 measured
effects are reproduced: largest difference 0.11 standard errors. Twelve are identical to
the third decimal or within 0.02 points; the indoor row differs a little because it uses
the stadium's dome flag where `effects.py` used the weather table's `gameIndoors`. The
weather effects on totals match too.

Team-specific home and road edges are noise on this data: split-half correlation (odd
against even seasons) is -0.05 for home and -0.10 for road residuals, and the road
between-team variance is below zero. The columns are kept, heavily shrunk, with that
warning for W8. New measurements without a plan-doc value: prior-season home fill rate
+0.52 points per 10 percentage points (t=5.3) and prior-season attendance +0.52 per 10,000
(t=7.1), both beyond Elo; last-game QB's share of attempts +0.12 per 10 points (t=2.4);
bowl portal departures and layoff length show nothing (t about -0.5, 240 and 480 games).

```
n games (FBS vs FBS, 2014-2025, Elo present) = 9085
base: margin = 1.03 + 2.19*home + 4.24*(elo_diff/100); n=9085

effect on home margin (points)                     ctx_ value         plan doc      z  n
home field (on-campus)                       +2.191 (se 0.655)   +2.191                 9085
rest diff (days, capped 21)                  +0.109 (se 0.058)   +0.107 (0.058)  +0.03  8506  ok
home off bye, away not                       +1.881 (se 0.645)   +1.864 (0.644)  +0.03  8506  ok
away off bye, home not                       -0.535 (se 0.665)   -0.515 (0.664)  -0.03  8506  ok
away travel (per 1000 km), on-campus         +0.682 (se 0.213)   +0.682 (0.213)  -0.00  8364  ok
away travel > 2000 km, on-campus             +1.469 (se 0.626)   +1.469 (0.626)  -0.00  8364  ok
away crosses >= 2 tz eastward                +1.515 (se 0.900)   +1.515 (0.900)  +0.00  8364  ok
away crosses >= 2 tz westward                +2.097 (se 0.888)   +2.097 (0.888)  +0.00  8364  ok
venue >= 1200 m, away home < 600 m           +1.557 (se 0.969)   +1.557 (0.969)  -0.00  8364  ok
away surface mismatch                        +0.686 (se 0.392)   +0.686 (0.392)  +0.00  8364  ok
indoor venue, away home outdoor              -0.621 (se 1.186)   -0.756 (1.274)  +0.11  8364  ok
away body clock kickoff before 11:00         +3.030 (se 1.479)   +3.033 (1.479)  -0.00  8363  ok
night kickoff (local >= 18:00)               +0.514 (se 0.373)   +0.517 (0.373)  -0.01  8363  ok
conference game (home edge shift)            -2.965 (se 0.432)   -2.973 (0.431)  +0.02  8364  ok
largest |z| = 0.11

== ctx_ variables the plan doc did not measure (same model; home minus away unless noted) ==
QB leader changed in last game (diff)                coef= -0.573 se=0.381 t= -1.5 n=7876
QB last-game leader share of attempts (diff, per 0.1) coef= +0.121 se=0.052 t= +2.4 n=8504
passers with 10+ attempts (diff)                     coef= -0.202 se=0.202 t= -1.0 n=8504
home team prior-season fill rate (per 0.1), on-campus coef= +0.520 se=0.099 t= +5.3 n=7597
home team prior-season attendance (per 10k), on-campus coef= +0.522 se=0.074 t= +7.1 n=7597
team home resid minus opp road resid, on-campus      coef= -0.195 se=0.373 t= -0.5 n=8143
bowls: portal entries since last game (diff)         coef= -0.117 se=0.250 t= -0.5 n=240
bowls: rated (3*+) portal entries since last game (diff) coef= -0.154 se=0.282 t= -0.5 n=240
bowls: days since last regular-season game (diff)    coef= -0.099 se=0.163 t= -0.6 n=480
travel km difference at neutral sites (per 1000)     coef= +1.188 se=0.560 t= +2.1 n=718
weather: wind (per 10 mph) on total points           coef= -1.046 se=0.399 t= -2.6 n=8550
weather: precipitation > 0 on total points           coef= -2.671 se=0.676 t= -4.0 n=8550
weather: temperature < 32F on total points           coef= -4.481 se=1.469 t= -3.0 n=8550
plan doc, weather on totals (outdoor games with an O/U): wind -1.04 (t=-2.6), precip>0 -2.68 (t=-4.0), temp<32F -4.49 (t=-3.1)

== Team home/road residuals (shrinkage and reliability) ==
per-game residual SD = 16.66
home: 392 team-blocks, between-team variance +1.93 pts^2 -> k = 144 games; split-half (odd vs even seasons) r = -0.048 over 135 teams
road: 393 team-blocks, between-team variance -0.28 pts^2 -> k = inf games; split-half (odd vs even seasons) r = -0.102 over 135 teams
```

## Open issues

- Team home/road residuals have no measurable reliability (above). W8 should test them and
  will probably drop them; the league home edge and site columns carry the signal.
- `ctx_league_hfa` is noisy in the first weeks of 2014 (window starts with 2014 data).
- Portal origin names are matched to `teams` by exact name; unmatched entries are ignored
  (not measured how many).
- QB continuity cannot see a change announced during the week (injury, benching). It is the
  approved v1 substitute for injury data.
- Weather for upcoming games is a forecast only as fresh as the last ingest; W9's weekly run
  decides when to pull it.
- 2026 values exist for every column but were not looked at for any choice.
