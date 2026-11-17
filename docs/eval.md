# Evaluation and baselines (W7)

Code: `src/cfb/eval/` and the market feature group in `src/cfb/features/market/`.
Contracts for prediction and metrics files: `docs/foundations.md` section 6.

## Commands

```bash
python -m cfb.eval baselines                       # predict 2018-2026, score 2018-2025, print the reference table
python -m cfb.eval score <model_id>                # writes data/eval/<model_id>.json, scores 2018-2025
python -m cfb.eval score <model_id> --compare baseline_close baseline_elo     # paired bootstrap
python -m cfb.eval score <model_id> --common-with baseline_close baseline_elo # same games as the baselines
python -m cfb.eval score <model_id> --seasons 2026 # the test season, only when asked for
```

Scoring defaults to 2018-2025. 2026 is the test season and is never in a default.

## Reference numbers (snapshot 2026-11-17)

Season-forward (fit on seasons before S, score S, S = 2018 to 2025), FBS vs FBS games
with a closing spread and CFBD pregame Elo, n = 6,024. Same numbers as the exploration
(`effects.py`) to four decimals, per season too.

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| `baseline_close` | 0.5198 | 0.1749 | 0.7312 |
| `baseline_elo` | 0.5493 | 0.1862 | 0.7131 |
| `baseline_home` | 0.6785 | 0.2427 | 0.5830 |
| `baseline_moneyline` (3,771 games, 2021-2025) | 0.5478 | 0.1854 | 0.7131 |
| `baseline_close` on those 3,771 games | 0.5450 | 0.1844 | 0.7163 |

| Season | n | close | elo | home |
| --- | --- | --- | --- | --- |
| 2018 | 772 | 0.5011 | 0.5221 | 0.6774 |
| 2019 | 774 | 0.4901 | 0.5191 | 0.6747 |
| 2020 | 534 | 0.5294 | 0.5532 | 0.6857 |
| 2021 | 770 | 0.5252 | 0.5553 | 0.6800 |
| 2022 | 776 | 0.5516 | 0.5862 | 0.6795 |
| 2023 | 792 | 0.5097 | 0.5499 | 0.6806 |
| 2024 | 798 | 0.5356 | 0.5669 | 0.6765 |
| 2025 | 808 | 0.5183 | 0.5427 | 0.6762 |

Paired bootstrap, Elo minus closing spread, log loss: +0.0295, SE 0.0033 over 8 seasons.
`baseline_home` accuracy is 0.5830 here and 0.582 in the exploration: a probability of
exactly 0.5 (neutral sites) counts as a home pick here, as in `cfb.core.samples`.

## Baselines

| model_id | Method | Games predicted | `cutoff_utc` |
| --- | --- | --- | --- |
| `baseline_close` | Logistic without intercept on minus the closing home spread | Games with a spread | kickoff |
| `baseline_moneyline` | Closing moneylines, vig removed proportionally; no fit | Games with both moneylines (2021 on) | kickoff |
| `baseline_elo` | Logistic without intercept on [home, (home Elo - away Elo) / 100], Elo from `game_pregame` | Games with both Elos, except FBS vs FCS from 2022 | game cutoff |
| `baseline_home` | Logistic without intercept on the home indicator; 0.5 at neutral sites | Every game | game cutoff |

Fits are refreshed once per season on completed FBS vs FBS games of earlier seasons that
have the model's inputs (`refit="season"`). Inputs are read through the view at each
cutoff: lines through the `market_lines` group, Elo through `game_pregame`.

Closing lines are set at kickoff, later than the week-start cutoff the `lines` table is
served at, so the two market baselines record kickoff as `cutoff_utc`. They are the
benchmark to measure against, not a model that could have run at the cutoff.

FBS vs FCS predictions exist for close, moneyline and home (and Elo before 2022) but the
fits use FBS vs FBS games only; scores outside the headline scope are in `by_segment`.

## Metrics (`cfb.eval.metrics`)

- Log loss with probabilities clipped to [1e-6, 1 - 1e-6] (the exploration's clip),
  Brier, accuracy (0.5 counts as a home pick), 10 equal-width calibration bins.
- Games are scored against `Warehouse.outcomes()`; ties and unplayed games are left out.
- `overall`, `by_season`, `by_week`: the scope (default FBS vs FBS; `--scope all`).
  `by_segment`: `fbs_vs_fbs` and `fbs_vs_fcs` over all scored games; `conference`,
  `non_conference`, `neutral`, `regular`, `postseason` inside the scope.
- `--common-with` restricts to games every named model predicted and says so in `scope`.

## Splits and runner (`cfb.eval.splits`)

- `season_forward(frame, seasons=2018..2025)` yields `Split(season, train, test)`;
  `season_forward_predict(frame, fit, predict)` returns out-of-season predictions.
- `walk_forward(warehouse, fit, predict, model_id, seasons, refit="season"|"cutoff")`:
  one call of `predict(model, view, slate)` per cutoff; `fit(view, train)` gets the
  completed games visible at the cutoff (kickoff + 5 h before it). Use `refit="cutoff"`
  for the 2026 week-by-week run. It checks that every row belongs to the slate and that no
  `cutoff_utc` is earlier than the game's cutoff, then validates the prediction schema.
- `write_weekly(preds, "data", replace=True)` writes one file per model and week and
  removes the model's older files first.

## Paired bootstrap (`cfb.eval.bootstrap`)

`paired_bootstrap(y, p_a, p_b, metric)` resamples per-game differences (exact, because all
three metrics are per-game means); `clusters=` resamples whole groups such as weeks.
Returns the two scores, the difference a - b, its SE, a 95% interval and the share of
resamples in which a is better. `compare(pred_a, pred_b, outcomes)` aligns two prediction
sets on their common completed games first.

## Market feature group (`market_lines`, prefix `mkt_`, level `game`)

Not an input of the primary model. Values are for the home team. Provider priority:
consensus, Bovada, DraftKings, ESPN Bet, then other providers in CFBD's order. Each value
comes from the first provider that has it; the moves and the moneyline pair come from the
first provider that has both parts.

| Column | Coverage, all 11,260 games |
| --- | --- |
| `mkt_spread_close`, `mkt_spread_range`, `mkt_n_providers` | 11,110 with a spread (2014 on) |
| `mkt_total_close` | 10,725 |
| `mkt_spread_open`, `mkt_spread_move` | 5,173 (2021 on) |
| `mkt_total_open`, `mkt_total_move` | 5,179 / 5,178 (2021 on) |
| `mkt_ml_home`, `mkt_ml_away`, `mkt_p_home_ml`, `mkt_overround` | 4,693 (2021 on) |

Leak test: passes on the synthetic store (CI) and on the snapshot at 6 cutoffs
(`python -m cfb.features.build --leak-test --groups market_lines --leak-cutoffs 6`).

Source quirks left as they are: a few moves and provider ranges above 30 points (a line
entered with the wrong sign at one provider), one game with both moneylines positive, and
moneylines down to -100000. Odds between -100 and 100 are treated as missing.
