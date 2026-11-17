# Win-probability model (W8)

Code: `src/cfb/model/` (matrix builder, model families, frozen configuration) and
`src/cfb/predict.py` (predictions and CLI). Selection scripts: `scripts/model_select.py`,
`scripts/model_opponent_tier.py`. Tests: `tests/test_model.py`.

## Commands

```bash
python -m cfb.features.build                              # stored features, all groups
python -m cfb.predict season-forward                      # 2018-2025, refit once per season
python -m cfb.predict walk-forward --seasons 2026         # the 2026 test, refit at every cutoff
python -m cfb.predict matchup "Ohio State" "Michigan" [--neutral] [--cutoff 2026-11-23]
python -m cfb.eval score model_v1 --common-with baseline_close baseline_elo --compare baseline_close baseline_elo
python scripts/model_select.py ablation|tune|gbt|final    # selection record (seasons <= 2025)
```

`cfb.predict:produce(warehouse, games)` is the producer for `cfb.run_week` (W9): it returns
the `walk_forward` frame (prediction schema plus season, season_type, week), refitting at
every cutoff. Enable it in `dashboard/config.toml` under `[producers]`.

## Frozen configuration (`model_id = model_v1`)

Frozen on 2026-11-17 using seasons through 2025 only, and committed before 2026 was scored
(commit e6861d2).

| Item | Value |
| --- | --- |
| Inputs | `str_power` (13 columns) and `str_elo` (2), as home minus away differences, plus `site` (1 at a true home game) and `fcs_game` |
| Family | Ridge regression on the home margin. Probability = sigmoid(a + s x predicted margin), with a and s fitted on out-of-fold margins (one fold per training season) |
| Penalty | lam 0.03 per observation, on standardised features. Missing values are set to the training mean, with missing-value indicators for columns missing in 5% or more of rows |
| Training rows | Every completed game with a winner from 2015 on, FBS vs FCS included, no era weighting |
| Refit | Once per season for the validation, at every cutoff for 2026 and the weekly runs |
| `pred_margin`, `pred_total` | Left empty (phase 2). The margin fit is an intermediate step, not a validated spread |
| `reasons` | The five largest per-feature contributions in log-odds toward the home team, with the home and away values |

## Validation (season-forward, 2018-2025)

The 6,024 FBS vs FBS games with a closing spread and CFBD pregame Elo. Fit on seasons
2015 to S-1, score S.

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| `baseline_close` | 0.5198 | 0.1749 | 0.7312 |
| **`model_v1`** | **0.5318** | **0.1793** | **0.7259** |
| W3 power rating + home (logit) | 0.5332 | | |
| `baseline_elo` | 0.5493 | 0.1862 | 0.7131 |
| `baseline_home` | 0.6785 | 0.2427 | 0.5830 |

Paired bootstrap: against Elo -0.0175 (SE 0.0025); against the closing spread +0.0120
(SE 0.0024). The plan's acceptance gate (pooled log loss at or below 0.535 and better
than Elo in at least 6 of 8 seasons) is met: better than Elo in 8 of 8 seasons.

| Season | n | model_v1 | Elo | Close |
| --- | --- | --- | --- | --- |
| 2018 | 772 | 0.5047 | 0.5221 | 0.5011 |
| 2019 | 774 | 0.4997 | 0.5191 | 0.4901 |
| 2020 | 534 | 0.5365 | 0.5532 | 0.5294 |
| 2021 | 770 | 0.5447 | 0.5553 | 0.5252 |
| 2022 | 776 | 0.5664 | 0.5862 | 0.5516 |
| 2023 | 792 | 0.5264 | 0.5499 | 0.5097 |
| 2024 | 798 | 0.5483 | 0.5669 | 0.5356 |
| 2025 | 808 | 0.5290 | 0.5427 | 0.5183 |

Calibration (10 bins, 6,024 games): every bin's win rate is within 0.04 of its mean
probability (largest gap 0.3-0.4: mean 0.350, win rate 0.388, n 578). Segments: conference
0.5457 (4,208), non-conference 0.4996 (1,816), neutral 0.6183 (476), postseason 0.6602
(319). Bowls are the weakest segment (opt-outs and layoffs are not in the data).

## 2026 test (walk-forward, weeks 1 to 11)

One pass with the frozen configuration, refit at every cutoff (predict week N with results
through week N-1). 629 completed FBS vs FBS games with a spread and Elo.

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| `baseline_close` | 0.5155 | 0.1731 | 0.7424 |
| **`model_v1`** | **0.5586** | **0.1881** | **0.7170** |
| `baseline_elo` | 0.5605 | 0.1902 | 0.7186 |
| `baseline_home` | 0.6730 | 0.2400 | 0.6010 |

Against Elo -0.0019 (SE 0.0089): not distinguishable in one season. Against the closing
spread +0.0431 (SE 0.0090), larger than the 0.0120 of the validation years. By week band:

| Weeks | n | model_v1 | Close | Elo |
| --- | --- | --- | --- | --- |
| 1-4 | 215 | 0.4679 | 0.3955 | 0.4863 |
| 5-8 | 229 | 0.6279 | 0.5935 | 0.6287 |
| 9-11 | 185 | 0.5782 | 0.5585 | 0.5622 |

These numbers were not used for any choice. One season resolves log-loss differences of
about 0.02, so 2026 does not show whether the gap to the market grew or 2026 is a hard
season (Elo is also 0.011 worse than its 2018-2025 mean).

## Selection record (seasons through 2025)

### Group ablation (logit, lam 0.01, all groups as home-minus-away differences)

| Configuration | Pooled log loss | Seasons better than Elo |
| --- | --- | --- |
| Site only | 0.6811 | 0/8 |
| `str_power` only | 0.5332 | 8/8 |
| All 21 candidate groups | 0.5396 | 7/8 |
| All 21, lam 0.1 | 0.5339 | 8/8 |
| All 21, margin model, lam 0.1 to 0.3 | 0.5330 | 8/8 |

`str_power` plus one other group (logit, lam 0.01): every result lies between 0.5322
(`str_elo`) and 0.5351 (`matchup_style`), against 0.5332 for `str_power` alone. Removing
one group from the full set changes it by at most 0.0022 except `str_power` (+0.0047).
Candidates were every group except `market_lines` (kept out by the approved default) and
`ctx_weather` (phase 2: observed in training, forecast live).

### Forward selection (margin model, lam 0.03)

Start `str_power` 0.5327. Step 1 `+ str_elo` 0.5318 (better in 5 of 8 seasons), step 2
`+ ctx_home_road` 0.5312 (6 of 8), step 3 `+ coach_tenure` 0.5306 (3 of 8; stop rule:
gain under 0.0005 or fewer than 5 seasons better).

Paired bootstrap of the steps: `+ str_elo` -0.0009 (SE 0.0007); `+ ctx_home_road` -0.0006
(SE 0.0006); margin against logit on `str_power` -0.0005 (SE 0.0004). No step clears two
standard errors. The frozen set keeps `str_elo`, a second rating method fitted
differently. It leaves out `ctx_home_road`, which W6 measured as noise (split-half r
-0.05 home, -0.10 road), so its gain here is most likely the best of 20 noisy tries.

### Other settings (margin, `str_power` + `str_elo`, lam 0.03: 0.5318)

| Variant | Log loss |
| --- | --- |
| Era half-life 3 / 6 seasons | 0.5321 / 0.5319 |
| Training from 2014 | 0.5324 |
| FBS vs FBS training rows only | 0.5318 |
| Average with the logit on `str_power` | 0.5320 |
| Gradient-boosted trees, all groups, home/away/difference (scikit-learn, 3 settings) | 0.5365 to 0.5447 |

### What the matchup, coaching and context groups add

Measured by the ablation above, none of them improves season-forward log loss beyond the
power rating by more than its standard error. Their effects are real in the W4 to W6
checks (for example the rush-success edge, t = 8.9 after Elo), but the opponent-adjusted
power and unit ratings already carry that information. The groups stay stored and
documented. A margin model for phase 2 (spread and totals) may use them differently.

### Opponent tier: do teams play down or up to opponents? (`scripts/model_opponent_tier.py`)

Residual = margin minus (power-rating gap + league home edge x site), all at each game's own
cutoff. 20,574 team-games, 2014-2025, residual SD 16.65 points.

| Game type for the team | n | Mean residual |
| --- | --- | --- |
| Favourite by more than 7 | 6,100 | +1.42 (SE 0.21) |
| Within 7 | 8,374 | 0.00 |
| Underdog by more than 7 | 6,100 | -1.42 (SE 0.21) |

League-wide, favourites beat the rating gap (slope +0.063 points per point of gap): the
rating is slightly compressed, and the fitted gap coefficient corrects it. A non-linear
gap term changes log loss by 0.0001. Team-specific: the difference between a team's
residual as favourite and as underdog does not repeat (split-half r -0.11 over 90
team-windows). A shrunk same-tier residual feature changes log loss from 0.5332 to 0.5333.
Not used.

### W5 coaching check rerun with W3's unit ratings

`python scripts/coaching_check.py` after PR #9 (adjusted unit ratings): ATS, fourth-down
and run-mix figures unchanged (reliability 0.03, 0.86, 0.97). Flexibility: pooled slope
+0.098 run share per EPA of gap (was +0.034 on the placeholder), slope variance observed
0.0502, sampling 0.0483, true 0.0018 (constant `FLEX_SLOPE_VAR` 0.0006), split-half
reliability of the raw slope 0.028 (80 coaches). Game-level variance 0.0017 (constant
0.002). Coach-specific flexibility is still not distinguishable from noise, so the
constants in `cfb.features.coaching.metrics` were left as they are; `coach_style` is not in
the model.

## Leakage

- Training rows are the completed games the runner passes to `fit` (results visible at the
  cutoff). Features of those games come from the stored files, where each game was built
  at its own cutoff.
- `tests/test_model.py::test_model_does_not_leak`: predictions at a cutoff do not change
  when every result after it is scrambled (synthetic store, features rebuilt on the
  scrambled data).
- Snapshot leak test of the two model groups: `python -m cfb.features.build --leak-test
  --groups str_power,str_elo --leak-cutoffs 4`: ok.
- `test_live_features_match_stored`: the live path for on-demand games gives the stored
  values.

## Known limits

- W3's preseason projection coefficients and power-rating lam were chosen on 2015-2025,
  which overlaps the validation seasons (docs/strength.md). The 2026 result is free of this.
- The selection compared about 40 configurations on the same 2018-2025 games; differences
  under 0.001 between them are not meaningful.
- No injury or depth-chart input; bowls (opt-outs) are the weakest segment.
- On-demand matchups use the home team's stadium (or no venue at a neutral site) and a
  kickoff at the cutoff.
