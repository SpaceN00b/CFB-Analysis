# Early-season priors from player data (P2, model_v2)

Code: `src/cfb/features/strength/players.py` (player components), `prior.py`
(`components_v2`, `COMPONENTS_V2`, `COEF_*_V2`), `power.py` (`power_state(version=2)`),
`groups_v2.py` (groups `str_power2`, `str_prior2`, `str_program`), `src/cfb/model/config.py`
(`model_v2`). Scripts: `scripts/strength_fit_prior.py --version 2` (projection refit),
`scripts/prior_harness.py` (experiment harness), `scripts/prior_select.py final` (record).
Tests: `tests/test_priors.py`.

`model_v1` and its groups are unchanged. `model_v2` is a second model id. Josiah approved it
as the weekly report's primary on 2026-11-17 (23:18 UTC); `model_v1` is still shown next to it.

## Why

Season-forward 2018-2025, `model_v1` trails the closing line most in week 1 and in the
postseason, and the week 1-4 gap grew in the transfer-portal years:

| Weeks | Games | model_v1 | Closing spread | Gap |
| --- | --- | --- | --- | --- |
| 1 | 337 | 0.4858 | 0.4527 | 0.033 |
| 2 | 347 | 0.4914 | 0.4797 | 0.012 |
| 3-4 | 783 | 0.5044 | 0.4885 | 0.016 |
| 5-8 | 1,632 | 0.5510 | 0.5381 | 0.013 |
| 9+ | 2,606 | 0.5237 | 0.5189 | 0.005 |
| Postseason | 319 | 0.6602 | 0.6247 | 0.036 |

Weeks 1-4 gap by season: 2018 -0.005, 2019 0.009, 2020 -0.015, 2021 0.015, 2022 0.022, 2023
0.027, 2024 0.038, 2025 0.035. Calibration in weeks 1-4 was fine (every bin within 0.05),
and refitting how fast the rating leaves the prior (lam) did not help, so the prior's
content was the limit: last season's rating, talent and team-level returning production do
not see who left and who arrived.

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

`python scripts/strength_fit_prior.py --version 2`: offense and defense linear in
`COMPONENTS_V2` (the 7 version-1 components plus the 11 centred player components), OLS on
2015-2025 (1,424 team-seasons), target = the season's final points rating.

| | Offense R^2 | Defense R^2 |
| --- | --- | --- |
| Last season only | 0.372 | 0.417 |
| Version 1 (model_v1) | 0.436 | 0.472 |
| **Version 2** | **0.457** | **0.503** |

Leave-one-season-out. The largest new terms: returning offense PPA (+1.42 points of offense
per 100 PPA), incoming PPA (+1.05), returning tackle and havoc shares on defense (-4.8 and
-4.3 points allowed per full share) and incoming tackle share (-6.0).

The power rating keeps lam 4 games and the 16-week half-life: with the v2 prior, lam 3, 5,
6 and half-lives 8, 32, none all scored worse (table below).

## Model

`model_v2` (`src/cfb/model/config.py`): `model_v1`'s family (ridge margin regression, lam
0.03, sigmoid calibration on out-of-fold margins), training rows (2015 on, FCS included) and
refits, with these inputs as home minus away differences: the `str_power2` rating columns
(off, def, net, prior share, net SE) in place of `str_power`'s, `str_power`'s game count and
7 unit-rating columns, `str_elo` (2), `str_prog_resid`, plus `site` and `fcs_game`.

`str_prog_resid` (group `str_program`, batch mode): for every completed game since 2014
visible at the cutoff, the team's margin minus the v2 power-rating expectation at that
game's own cutoff (net gap + 2 x home field at a home game); summed and divided by
(games + 8). Each season's expectations use the v2 projection as it stood at the season's
first cutoff.

Frozen in commit 61fbcf1 (pushed 2026-11-17 22:13 UTC) before 2026 was scored.

### Season-forward 2018-2025 (6,024 games; `python scripts/prior_select.py final`)

| Model | Pooled | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_v1 | 0.5318 | 0.5047 | 0.4997 | 0.5365 | 0.5447 | 0.5664 | 0.5264 | 0.5483 | 0.5290 |
| **model_v2** | **0.5255** | 0.5024 | 0.4932 | 0.5282 | 0.5459 | 0.5610 | 0.5167 | 0.5394 | 0.5178 |
| Closing spread | 0.5198 | 0.5011 | 0.4901 | 0.5294 | 0.5252 | 0.5516 | 0.5097 | 0.5356 | 0.5183 |

model_v2 minus model_v1: -0.0064 (paired SE 0.0011, bootstrap over whole weeks), better in
7 of 8 seasons. Weeks 1-4: -0.0139 (SE 0.0029). Brier 0.1769 (v1 0.1793), accuracy 0.7292
(v1 0.7259).

| Weeks | Games | model_v1 | model_v2 | Closing spread |
| --- | --- | --- | --- | --- |
| 1 | 337 | 0.4858 | 0.4669 | 0.4527 |
| 2 | 347 | 0.4914 | 0.4788 | 0.4797 |
| 3-4 | 783 | 0.5044 | 0.4921 | 0.4885 |
| 5-8 | 1,632 | 0.5510 | 0.5424 | 0.5381 |
| 9+ | 2,606 | 0.5237 | 0.5224 | 0.5189 |
| Postseason | 319 | 0.6602 | 0.6579 | 0.6247 |

These use the full-sample projection coefficients (as model_v1 does). With
leave-one-season-out coefficients the same inputs gave -0.0042 for the prior and a further
-0.0017 for the program residual (about -0.0059 together), so roughly 0.0005 of the gain may be
in-sample fit of the projection.

### 2026 test (walk-forward, weeks 1-11, 629 games; scored once after the freeze)

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| baseline_close | 0.5155 | 0.1731 | 0.7424 |
| **model_v2** | **0.5470** | **0.1842** | **0.7170** |
| model_v1 | 0.5586 | 0.1881 | 0.7170 |
| baseline_elo | 0.5605 | 0.1902 | 0.7186 |

model_v2 minus model_v1: -0.0116 (SE 0.0046, 95% CI -0.0205 to -0.0026). Weeks 1-4 (215
games): model_v1 0.4679, model_v2 0.4387, closing spread 0.3955. Weeks 5-11 (414): 0.6057,
0.6033, 0.5778.

### Weekly run

`dashboard/config.toml` has the producer `model_v2 = "cfb.predict:produce_v2"`;
`primary = "model_v2"` since Josiah's approval on 2026-11-17, with `model_v1` among the
compared models. `python -m cfb.run_week --primary model_v1` reports on the old model for one run. `python -m cfb.predict --model model_v2 matchup "A" "B"` works too. The
stored-features tar `features_2014-2026_2026-11-17_p2.tar` (FEATURES_LATEST) holds the new
groups for every season.

## Everything tried (season-forward 2018-2025, 6,024 games; differences against model_v1, paired SE over whole weeks)

From `scripts/prior_harness.py` with the exploration scripts in project files
`exploration/priors/` (leave-one-season-out projection coefficients; R^2 is
offense / defense). Week 1-4 differences in brackets.

| Change | Prior R^2 | Log loss | vs model_v1 |
| --- | --- | --- | --- |
| model_v1 | 0.436 / 0.472 (full-sample) | 0.5318 | |
| Version 1 components, LOSO coefficients | 0.436 / 0.472 | 0.5321 | +0.0003 (+0.0007) |
| + last 2 and 3 seasons' ratings | 0.435 / 0.473 | 0.5321 | +0.0003 |
| + returning usage by unit (pass, rush, receiving) | 0.438 / 0.473 | 0.5319 | +0.0000 (+0.0024) |
| + returning PPA totals by unit (z) | 0.453 / 0.472 | 0.5320 | +0.0002 |
| + last season's FPI, SRS, CFBD preseason Elo | 0.433 / 0.471 | 0.5322 | +0.0004 |
| + portal counts in/out | 0.436 / 0.471 | 0.5318 | +0.0000 (-0.0011) |
| + portal 3+/4+ star counts | 0.438 / 0.473 | 0.5315 | -0.0003 (-0.0031) |
| + player offense PPA (ret, in, lost) | 0.462 / 0.475 | 0.5300 | -0.0018 (-0.0045) |
| + QB PPA (ret, in) | 0.439 / 0.472 | 0.5316 | -0.0002 |
| + player defense shares (ret, in) | 0.436 / 0.503 | 0.5293 | -0.0025 (-0.0040) |
| + draft picks (all, rounds 1-2) | 0.438 / 0.471 | 0.5320 | +0.0001 |
| + roster recruiting (4-5 stars, top-22/44 rating) | 0.433 / 0.473 | 0.5320 | +0.0002 |
| + all player inputs | 0.463 / 0.503 | 0.5275 | -0.0043 (-0.0078) |
| **+ offense + defense + portal stars (chosen)** | 0.457 / 0.503 | 0.5276 | **-0.0042 (SE 0.0010; -0.0090)** |
| chosen + previous rating x returning share | 0.455 / 0.502 | 0.5277 | -0.0041 |
| chosen, ridge 20 on the projection | 0.468 / 0.500 | 0.5285 | -0.0033 |
| chosen, lam 3 / 5 / 6 games | | 0.5278 / 0.5280 / 0.5287 | |
| chosen, half-life 8 / 32 / none | | 0.5279 / 0.5276 / 0.5278 | |
| model_v1, lam 2 / 3 / 5 / 6 / 8 (old prior) | | 0.5322 / 0.5316 / 0.5326 / 0.5335 / 0.5353 | |
| model_v1 + net gap x prior share | | 0.5320 | +0.0002 |

Program residual on top of the chosen prior (difference against the chosen prior alone):

| Window, shrink | Difference | Seasons better |
| --- | --- | --- |
| **all since 2014, 8 games (chosen)** | **-0.0017 (SE 0.0008), weeks 1-4 -0.0039** | 6 / 8 |
| all since 2014, 16 | -0.0015 (SE 0.0007) | 6 / 8 |
| last 5 seasons, 8 | -0.0012 (SE 0.0007) | 6 / 8 |
| last 3 seasons, 8 | -0.0006 (SE 0.0005) | 5 / 8 |

Growing-window check: the window lengthens from 2014 on, so later seasons see more games.
A fixed 5-season window gives a similar gain in 2023-2025 (-0.0014, -0.0034, -0.0030
against -0.0048, -0.0027, -0.0025 for the full window), so the gain is not only the window
growing. It meets the stop rule (gain >= 0.0005, better in >= 5 of 8 seasons) and is 2.1 SE.
The setting (all games, 8) was the best of 15 in the planning thread's test and was not
re-chosen here.

Not used and why: QB-only PPA, draft picks and roster recruiting ratings added nothing on
top of talent and 4-year recruiting; interactions overfit slightly.

## Known limits

- Roster lists are final rosters (see above); a weekly 2026 pull is current.
- Defensive value is box-score shares, not opponent-adjusted.
- The projection coefficients are fitted on 2015-2025, which includes the check seasons
  2018-2025 (as for model_v1). The search used leave-one-season-out coefficients; the frozen
  configuration uses the full-sample fit. Both are reported.
- 2014 has no prior season, so no projection; the program residual starts with 2014 games.
- The weekly run rebuilds `str_program` (batch) over all seasons each time, about 6 minutes.
- The analysis pull that preceded the P1 ingest used the same endpoints; the code reads only
  the ingested tables (`roster`, `ppa_players_season`, `games_players`, `transfer_portal`).
