# Play text

`cfb.core.playtext` parses the `playText` column of CFBD's `plays` into three tables. It needs no
API calls: CFBD's play text is ESPN's play-by-play text, already in the snapshot for 2014 to 2026.

```bash
python -m cfb.core.playtext          # writes data/core/play_{detail,players,penalties}.parquet (about 65 s)
python -m cfb.core.playtext --check  # parse and print coverage without writing
```

It is a separate command from `python -m cfb.core.build` because no feature group reads these
tables yet. The tables have `game` access policies, so feature code can read them through
`AsOfView` once a group needs them.

## Two text formats

| format | where | example |
| --- | --- | --- |
| detailed | every play from 2025 week 9 on; before that, about 1% of plays (mostly penalty plays, where ESPN kept the stat crew's text) | `(13:24) No Huddle-Shotgun #13 A.Simmons pass complete short middle to #0 N.Montgomery caught at MIZ27, for 4 yards loss to the MIZ27 (#19 J.Williams)` |
| condensed | 2014 to 2025 week 8 | `Drake Lindsey pass complete to Frank Bierman for 5 yds to the MINN 30` |

ESPN still serves the condensed text for games before 2025 week 9, so direction, depth and
tacklers cannot be recovered for those games from ESPN either (checked 2026-11-18 on games
from 2014 to 2025 week 6).

## Tables

**`play_detail`**: one row per play, for every play in `plays` whose game is in `game_index`.

| column | detailed | condensed | values |
| --- | --- | --- | --- |
| `text_format` | | | `detailed`, `condensed`, `none` |
| `action` | yes | yes | `rush`, `pass`, `sack`, `kneel`, `spike`, `penalty` (penalty only), `other`, `none` |
| `shotgun`, `no_huddle` | yes | NA | formation words before the play; no word means under center |
| `run_dir` | yes | NA | `left`, `middle`, `right` |
| `pass_depth`, `pass_dir` | yes | NA | `short` / `deep`; `left`, `middle`, `right` |
| `completed`, `intercepted` | yes | yes | |
| `kneel`, `spike` | yes | kneel only | CFBD types kneels as `Rush` (3,157 in the snapshot) |
| `text_yards` | yes | yes | yards in the text (negative for losses and sacks) |
| `air_yards` | yes | NA | line of scrimmage to the catch, throw or interception spot |
| `yac` | yes | NA | `text_yards - air_yards` on completions |
| `n_tacklers` | yes | NA | rushes and completions: 0 (out of bounds, touchdown), 1, or 2 (shared tackle) |
| `broken_up`, `qb_hurried` | yes | breakup only | |
| `fumbled`, `penalties`, `no_play` | yes | yes | |

**`play_players`**: one row per named player per role: `passer`, `rusher`, `target`, `tackler`,
`sacker`, `interceptor`, `breakup`, `hurry`, `fumbler`, `forced_fumble`. `side` is offense or
defense, `team_id` the team. `athlete_id` comes from that season's `roster`: the same team and
jersey with a matching last name (`match = jersey`), else a unique match on team, last name and
first initial or full name (`match = name`), else NA (`match = none`).

**`play_penalties`**: one row per penalty: `team_text` as written, `side` and `team_id`,
`penalty` (for example `Holding`), `status` (`accepted`, `declined`, `offsetting`), `yards`,
`no_play`, and the player with `athlete_id` when the text names one.

How spots become yards: a spot such as `MIZ31` is either the offense's or the defense's 31. For
each game, the end spot of every detailed rush and pass is checked against CFBD's `yardsToGoal`
and the text's yards, which tells which team each abbreviation stands for. An abbreviation with
no evidence in its game uses its evidence from other games for one of the game's two teams.

## Coverage (snapshot 2026-11-18, all games)

Detailed rushes and passes, 2025 and 2026: run direction 99.9% of 78,224 rushes; pass depth
97.5% and direction 97.4% of 73,720 passes (85% short, median 5 air yards; 15% deep, median
28); air yards 99.5%; at least one tackler on 91% of rushes and 82% of completions. Shotgun on
94.5% of plays, no huddle on 59%.

Named intended receiver on incompletions (both formats), by season:

| 2014-2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 76-78% | 54% | 47% | 42% | 21% | 92% | 95% |

Roster ids: 99.6% of player mentions in detailed text, 96% in condensed text. Penalties:
124,934 rows, team side known for 94% to 99% per season, 61% name a player and 90% of those match
a roster id.

Known gaps:
- A few stat-crew grammars are parsed only in part (for example `pass deep to the right
  complete for 31 yards to GILL, Jacob`); those plays get `action = other`.
- Scoring-summary text (`Maurki James 1 Yd Run (Nathanial Eichner Kick)`) gives the players
  and yards only.
- Roster ids are matched by name and jersey; transfers between the roster pull and the game,
  and jersey changes, fall back to name matching or stay NA.
