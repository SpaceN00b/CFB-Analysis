# Spread and total (phase 2)

Code: `src/cfb/model/points_config.py` (frozen configuration), `PointsModel` in
`src/cfb/model/models.py`, `src/cfb/predict.py` (fills `pred_margin` and `pred_total` of
`model_v1`), `src/cfb/eval/points.py` (scoring against the closing lines). Selection script:
`scripts/points_select.py`. Tests: `tests/test_points.py`.

## Commands

```bash
python -m cfb.features.build                          # stored features, all groups
python -m cfb.predict season-forward                  # 2018-2025: p_home_win, pred_margin, pred_total
python -m cfb.predict walk-forward --seasons 2026     # the 2026 test, refit at every cutoff
python -m cfb.eval points model_v1                    # MAE and hit rate vs the closing lines, 2018-2025
python -m cfb.eval points model_v1 --seasons 2026     # the same for the test season
python scripts/points_select.py margin|total|forward|tune|final   # selection record (seasons <= 2025)
```

`python -m cfb.eval points` writes `data/eval/<model_id>_points.json` unless `--no-write`.

## What is predicted

- `pred_margin`: home points minus away points. As a spread for the home team it is
  `-pred_margin` (a home margin of +7 is home -7).
- `pred_total`: home plus away points.

Both come from ridge regressions on points, refit with the win-probability model (once per
season in the validation, at every cutoff for 2026 and the weekly runs) on every completed
game from 2015 visible at the cutoff, FBS vs FCS and ties included. `p_home_win` is still the
frozen W8 model (docs/model.md) and is unchanged: 0.5318 season-forward, 0.5586 on 2026.
The margin model uses more inputs than the win model, so in a close game the spread can
favour one team while the win probability is just under 0.5 for it.

No market data goes in (`market_lines` is the benchmark only). Weather (`ctx_weather`) is
left out: CFBD holds observed weather for played games and forecasts for upcoming ones, so a
model trained on it would be scored on better information than a weekly run has.

## Frozen configuration

Frozen on 2026-11-17 using seasons through 2025 only, committed before 2026 was scored
(commit 881cd64).

| Item | Spread (`pred_margin`) | Total (`pred_total`) |
| --- | --- | --- |
| Inputs | `str_power`, `ctx_crowd`, `str_prior`, `str_elo`, home minus away; `site`, `fcs_game` | `str_power` home plus away; the coaches' fourth-down and run rates above expected and flexibility (`coach_fourth_go_oe`, `coach_run_oe`, `coach_flex_slope`, `coach_flex_shift`); pace and explosiveness from `matchup_style` (`mat_pace_plays`, `mat_off_plays`, `mat_off_sec_per_play`, `mat_off_expl_rate`, `mat_def_expl_rate`, `mat_off_run_rate_neutral`), all home plus away; `site`, `fcs_game` |
| Family | Ridge least squares, lam 0.03 per observation on standardised inputs, missing values at the training mean with indicators for columns missing in 5% or more of rows | Same |
| Target cap, era weighting | none | none |
| Training rows | Completed games from 2015 | Same |

## Validation (season-forward, 2018-2025)

The 6,024 FBS vs FBS games with a closing spread and total (the W7 baseline set). Fit on
2015 to S-1, score S. MAE and RMSE in points. Hit rate = share of games where the model's
side of the closing line won (home cover when `pred_margin` > minus the spread, over when
`pred_total` > the total), pushes left out; at -110 a side needs 52.4% to break even.
"Edge 3+" keeps games where the model is at least 3 points from the line.

| Target | Model MAE | Closing line MAE | Model RMSE | Line RMSE | Bias | Hit rate, all | Edge 3+ | Edge 7+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spread | 12.64 | 12.25 | 15.94 | 15.49 | +0.10 | 50.5% (5,908) | 50.6% (2,644) | 51.7% (530) |
| Total | 13.02 | 12.72 | 16.38 | 16.04 | -0.07 | 50.3% (5,940) | 52.5% (2,473) | 53.2% (498) |

Bias = mean of actual minus predicted. The SE of a hit rate is about 0.65 points at 5,900
games, 1.0 at 2,500 and 2.2 at 500. No row is above 52.4% by more than its SE; every
spread row is within 1 SE of 50%.

| Season | Spread MAE | Line | Total MAE | Line |
| --- | --- | --- | --- | --- |
| 2018 | 12.79 | 12.66 | 13.31 | 12.93 |
| 2019 | 12.45 | 12.15 | 13.07 | 12.83 |
| 2020 | 13.30 | 12.89 | 14.06 | 13.62 |
| 2021 | 13.07 | 12.56 | 13.01 | 12.58 |
| 2022 | 12.47 | 12.03 | 12.58 | 12.07 |
| 2023 | 12.47 | 12.09 | 13.00 | 12.78 |
| 2024 | 12.57 | 12.05 | 12.81 | 12.73 |
| 2025 | 12.24 | 11.82 | 12.69 | 12.52 |

The closing line is more accurate in every season for both targets. The closing line is
set at kickoff, the model at the start of the week, so part of the gap is information the
model cannot have (injuries, weather, late news). Opening lines exist from 2021 (3,937 of
these games): opening spread MAE 12.25, opening total MAE 12.69; the model's hit rate
against the opening total is 52.0% (SE 0.8), against the opening spread 50.5%. These
opening-line numbers were computed after the freeze and were not used for any choice.

## 2026 test (walk-forward, weeks 1 to 11)

One pass with the frozen configuration, refit at every cutoff (predict week N with results
through week N-1). 629 completed FBS vs FBS games with closing lines.

| Target | Model MAE | Closing line MAE | Model RMSE | Line RMSE | Bias | Hit rate, all | Edge 3+ | Edge 7+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spread | 12.72 | 11.75 | 16.04 | 14.98 | -0.27 | 46.1% (610) | 46.1% (293) | 46.9% (81) |
| Total | 12.82 | 12.42 | 15.99 | 15.61 | +1.02 | 48.2% (622) | 45.6% (180) | 40.0% (15) |

| Weeks | Games | Spread MAE | Line | Hit rate | Total MAE | Line | Hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-4 | 215 | 12.31 | 11.02 | 45.5% | 12.26 | 11.81 | 48.6% |
| 5-8 | 229 | 12.37 | 11.52 | 47.7% | 13.33 | 13.12 | 48.9% |
| 9-11 | 185 | 13.64 | 12.89 | 44.7% | 12.83 | 12.28 | 47.0% |

Against the opening lines (all 629 games have one): spread hit rate 47.5%, total 49.7%.

The spread gap to the closing line (0.97 points of MAE) is larger than in any validation
season (0.13 to 0.52), the same pattern as the win probability (docs/model.md: 0.043 log
loss behind the close in 2026, 0.012 in validation). The spread hit rate of 46.1% is 1.9
SE below 50%. One season cannot separate a weaker model from a season where the market
had more information than usual; the numbers were not used for any choice.

**Reading for betting use:** on this evidence the model does not beat the closing spread
or total. Its lines are a market-independent opinion; the weekly report shows where they
differ from the market and keeps score, so the record builds up week by week.

## Selection record (seasons through 2025)

`python scripts/points_select.py`. Every comparison uses the same 6,024 games; differences
under about 0.03 points of MAE are inside the noise (paired SE with whole weeks as
clusters: 0.03 to 0.04).

### Spread

| Configuration | MAE | Hit rate vs close |
| --- | --- | --- |
| `str_power` only | 12.708 | 49.7% |
| `str_power` + `str_elo` (the W8 win-model inputs) | 12.696 | 50.2% |
| Same, home and away values instead of the difference | 12.719 | 50.3% |
| W8 inputs + each other group | 12.665 (`str_prior`, `ctx_crowd`) to 12.715 | 49.9% to 50.8% |
| All 21 candidate groups, lam 0.03 / 0.3 | 12.703 / 12.691 | 51.6% / 51.0% |
| Forward selection (frozen): `str_power`, `ctx_crowd`, `str_prior`, `str_elo` | **12.641** | 50.5% |
| Frozen set, lam 0.01 / 0.1 / 0.3 | 12.634 / 12.665 / 12.702 | |
| Frozen set, margins capped at 28 / 35 / 42 | 12.825 / 12.699 / 12.656 | |
| Frozen set, era half-life 3 / 6 seasons | 12.638 / 12.639 | |
| W8 inputs + observed weather (diagnostic) | 12.696 | 49.7% |

Forward selection (stop rule fixed before running: MAE gain under 0.01 or fewer than 5 of
8 seasons better): `+ ctx_crowd` 12.683 (5/8 seasons), `+ str_prior` 12.665 (5/8),
`+ str_elo` 12.641 (6/8), then `+ ctx_travel` 12.631 failed the rule (4/8). The frozen set
against the W8 inputs: -0.055 MAE (SE 0.033). `ctx_crowd` matches W6's measured effect
(prior-season home fill rate, +0.52 points per 10 percentage points, t = 5.3); `str_prior`
carries the preseason projection. Capping blowout margins made the MAE worse; lam 0.01
was 0.007 better than 0.03, inside the noise, so 0.03 (the W8 value) was kept.

### Total

| Configuration | MAE | Bias | Hit rate vs close |
| --- | --- | --- | --- |
| `str_power` home plus away | 13.097 | -0.64 | 50.2% |
| `str_power` home and away separately / sum and difference | 13.109 / 13.121 | | |
| `str_power` + `matchup_style` (all 48 columns) | 13.036 | +0.45 | 50.4% |
| `str_power` + `coach_style` (all 7 columns) | 13.006 | -0.02 | 51.0% |
| `str_power` + the 6 pace columns | 13.040 | -0.08 | 50.6% |
| `str_power` + the 4 coach rate columns | 13.043 | -0.65 | 50.4% |
| **`str_power` + 4 coach rates + 6 pace columns (frozen)** | **13.022** | -0.07 | 50.3% |
| Same + all of `matchup_style`, lam 0.03 / 0.1 / 0.3 | 13.042 / 13.032 / 13.027 | | |
| All 21 groups, lam 0.03 / 0.3 | 13.172 / 13.060 | | |
| League scoring level (previous season, or season to date blended with it) added | 13.080 to 13.181 | -0.8 to -1.7 | |
| `str_power` + observed weather (diagnostic) | 13.063 | -0.70 | 51.3% |

Forward selection over whole groups picked `coach_style` (13.006, better in 8 of 8
seasons) and stopped. Three of its seven columns are counts (`coach_fourth_n`,
`coach_run_n`, `coach_flex_games`); they grow with the length of the data window since
2014 (mean `coach_flex_games` 3 in 2014, 46 in 2025, 51 in 2026), so their weight is a
calendar trend that would extrapolate past the training range. The frozen set keeps the
coach rate columns only and adds the pace columns, which `matchup_style` contributed
alone (13.036). The frozen set against `str_power` alone: -0.075 MAE (SE 0.037).
Capping totals at 80 or 90 lowered the MAE by up to 0.03 and moved the bias to +0.8 to
+1.2 points, so it was not used.

## Known limits

- No injury, depth-chart or weather input; the closing line has all three.
- The selection compared about 60 configurations on the same 2018-2025 games, so the
  frozen sets' validation numbers are slightly optimistic.
- W3's preseason projection (in `str_prior` and `str_power`) was fitted on 2015-2025,
  which overlaps the validation seasons (docs/strength.md). The 2026 result is free of this.
- Scoring fell from about 57 points per FBS game in 2014-2020 to 53 in 2025; the total
  model has no explicit era term and is low in some seasons and high in others (bias -3.9
  in 2020, +1.7 in 2023).
