# Early-season priors from player data (P2, model_v2)

Code: `src/cfb/features/strength/players.py` (player components), `prior.py`
(`components_v2`, `COMPONENTS_V2`, `COEF_*_V2`), `power.py` (`power_state(version=2)`),
`groups_v2.py` (groups `str_power2`, `str_prior2`, `str_program`), `src/cfb/model/config.py`
(`model_v2`). Scripts: `scripts/strength_fit_prior.py --version 2` (projection refit),
`scripts/prior_harness.py` (experiment harness), `scripts/prior_select.py final` (record).
Tests: `tests/test_priors.py`.

`model_v1` and its groups are unchanged. `model_v2` is a second model id; the weekly report's
primary stays `model_v1` until Josiah approves a switch.

## Why

Season-forward 2018-2025, `model_v1` trails the closing line most in week 1 and in the
postseason, and the week 1-4 gap grew in the transfer-portal years:

RESULTS_DIAG

## Player components (`players.py`)

For season S and team T, players are matched by CFBD athlete id across the S roster
(`roster`), last season's player PPA (`ppa_players_season`, S-1) and last season's box scores
(`games_players`, S-1):

| Column | Meaning |
| --- | --- |
| `pl_ret_ppa` | Total PPA in S-1 of players on T's S roster who played for T, per 100 PPA |
| `pl_in_ppa` | Total PPA in S-1 at other schools of players on T's S roster (transfers keep their production) |
| `pl_lost_ppa` | T's S-1 total PPA minus `pl_ret_ppa` |
| `pl_ret_def_tkl`, `pl_ret_def_hav` | Share of T's S-1 tackles / havoc plays (TFL + passes defended + interceptions) made by players on T's S roster |
| `pl_in_def_tkl`, `pl_in_def_hav` | Incoming players' shares of their previous team's tackles / havoc plays, summed |
| `pl_portal_in3/out3/in4/out4` | Transfers in and out rated 3+ and 4+ stars (portal 2021 on, 0 before) |

They enter the projection centred on the season's FBS mean (`<name>_c`). Unknown values
(no roster or PPA rows) count as average.

Point in time: the roster table is as of the pull date, so a past season's list is its final
roster. Players added during a season are included; they rarely have S-1 production at
another FBS school. PPA and box scores are of the finished season S-1. Season-level PPA is
offense only (CFBD has no defensive PPA), which is why defense uses box-score shares.

## Projection refit

RESULTS_PROJ

## Model

RESULTS_MODEL

## Everything tried (season-forward 2018-2025, 6,024 games; differences against model_v1, paired SE over whole weeks)

RESULTS_TRIED

## Known limits

- Roster lists are final rosters (see above); a weekly 2026 pull is current.
- Defensive value is box-score shares, not opponent-adjusted.
- The projection coefficients are fitted on 2015-2025, which includes the check seasons
  2018-2025 (as for model_v1). The search used leave-one-season-out coefficients; the frozen
  configuration uses the full-sample fit. Both are reported.
- 2014 has no prior season, so no projection; the program residual starts with 2014 games.
