"""Choose the matchup shrinkage constants on seasons through 2025.

    python scripts/matchup_tune.py [--data-dir data]

1. PRIOR_GAMES (cfb.features.matchup.stats): how many games the previous season's style
   counts for. For every team-season 2015-2025 and every point after j >= 1 games, the
   blended rate predicts the rate over the rest of that season. Prints the mean squared
   error, each metric scaled by its spread across teams, for a grid of values.
2. HISTORY_SHRINK (cfb.features.matchup.history): per-game noise variance of the style
   residuals over the variance of the true team-by-class effect, the latter estimated
   from the covariance of odd-game and even-game means (split half).

Reads the snapshot directly; 2026 is never used.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.features.matchup.history import FAMILIES, _classes, _with_tilt, game_residuals
from cfb.features.matchup.stats import RATES, game_stats, season_profiles

LAST_SEASON = 2025
METRICS = ["off_run_rate_neutral", "off_run_rate", "off_plays", "off_sec_per_play", "off_expl_share",
           "off_pd_rate", "off_sd_success", "def_havoc", "off_line_yards", "def_stuff_rate"]
GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


def full_stats(wh: Warehouse) -> pd.DataFrame:
    games = wh.schedule(range(2014, LAST_SEASON + 1))
    view = wh.view(games["cutoff_utc"].max())
    stats = game_stats(view, list(range(2014, LAST_SEASON + 1)))
    return stats.sort_values(["team_id", "start_time_utc"]).reset_index(drop=True)


def tune_prior(stats: pd.DataFrame) -> None:
    cols = sorted({c for m in METRICS for c in RATES[m]})
    prior = stats.groupby(["season", "team_id"])[cols].sum()
    prior_games = stats.groupby(["season", "team_id"]).size()
    errs = {k: {m: [] for m in METRICS} for k in GRID}
    spread = season_profiles(stats)[METRICS].groupby(level="season").std().mean()
    for (season, team), g in stats[stats["season"] >= 2015].groupby(["season", "team_id"]):
        if (season - 1, team) not in prior.index or len(g) < 4:
            continue
        p = prior.loc[(season - 1, team)] / prior_games.loc[(season - 1, team)]
        cum = g[cols].cumsum().to_numpy()
        total = cum[-1]
        idx = {c: i for i, c in enumerate(cols)}
        for j in range(1, len(g) - 1):
            seen, rest = cum[j - 1], total - cum[j - 1]
            for m in METRICS:
                num, den = RATES[m]
                if rest[idx[den]] <= 0:
                    continue
                truth = rest[idx[num]] / rest[idx[den]]
                for k in GRID:
                    d = seen[idx[den]] + k * p[den]
                    if d <= 0:
                        continue
                    est = (seen[idx[num]] + k * p[num]) / d
                    errs[k][m].append(((est - truth) / spread[m]) ** 2)
    print("PRIOR_GAMES: mean squared error / cross-team SD^2, rest of season, j >= 1 games played")
    table = pd.DataFrame({k: {m: np.mean(v) if v else np.nan for m, v in errs[k].items()} for k in GRID})
    table.loc["mean"] = table.mean()
    print(table.round(4).to_string())


def tune_history(wh: Warehouse, stats: pd.DataFrame) -> None:
    view = wh.view(wh.schedule(range(2014, LAST_SEASON + 1))["cutoff_utc"].max())
    res = game_residuals(view, stats)
    parts = []
    for s in range(2014, LAST_SEASON + 1):
        sub = stats[stats["season"] == s]
        prof = season_profiles(sub).droplevel("season")
        fbs = sub.loc[sub["team_fbs"].astype(bool), "team_id"].unique()
        cls = _classes(_with_tilt(prof, fbs), fbs)
        part = res[res["season"] == s].copy()
        for fam in FAMILIES:
            part[fam] = cls[fam].reindex(part["opp_id"]).to_numpy()
        # placebo: thirds by opponent id, unrelated to style; its split-half covariance
        # measures the bias of the estimator (residuals share rating updates)
        part["placebo"] = (part["opp_id"] * 2654435761 % 2**32 % 3).astype(float)
        parts.append(part)
    res = pd.concat(parts, ignore_index=True)

    def split_half(fam: str, col: str) -> tuple[float, float, int]:
        r = res.dropna(subset=[fam, col])
        a_means, b_means = [], []
        for end in range(2015, LAST_SEASON + 1):  # two-season windows, like the feature
            w = r[r["season"].isin([end - 1, end])].copy()
            w["k"] = w.groupby(["team_id", fam]).cumcount()
            for _, grp in w.groupby(["team_id", fam]):
                if len(grp) < 4:
                    continue
                a_means.append(grp[grp["k"] % 2 == 0][col].mean())
                b_means.append(grp[grp["k"] % 2 == 1][col].mean())
        a, b = np.array(a_means), np.array(b_means)
        se = float(np.std((a - a.mean()) * (b - b.mean())) / np.sqrt(len(a)))
        return float(np.cov(a, b)[0, 1]), float(r[col].var()), len(a), se

    print("\nHISTORY_SHRINK: split-half covariance of odd/even-game means per team and class,")
    print("two-season windows ending 2015-2025, minus the placebo covariance (bias)")
    rows = []
    for side in ("off", "def"):
        bias, _, _, bias_se = split_half("placebo", f"res_{side}")
        for fam, (fside, *_rest) in FAMILIES.items():
            if fside != side:
                continue
            cov, noise, n, se = split_half(fam, f"res_{side}")
            tau2 = cov - bias
            rows.append({"family": fam, "side": side, "groups": n, "noise_var": noise, "cov": cov,
                         "placebo_cov": bias, "tau2": tau2, "tau2_se": float(np.hypot(se, bias_se)), "shrink_k": noise / tau2 if tau2 > 0 else np.inf})
    print(pd.DataFrame(rows).round(6).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    stats = full_stats(wh)
    tune_prior(stats)
    tune_history(wh, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
