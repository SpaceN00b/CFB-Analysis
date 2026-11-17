"""Fit the quarterback quality constants of cfb.features.strength.qb (K, DECAY, PRIOR).

For every player-game with a passing play in seasons <= 2025, predict the player's
opponent-adjusted PPA per play from his earlier games only:

    q = (sum_w adj + K * prior) / (sum_w plays + K),   w = DECAY ** (seasons ago)

and score the play-weighted squared error. The prior is linear in (recruiting rating,
no-rating flag, JUCO flag, class year), so for each (K, DECAY) its coefficients are a
weighted least-squares fit. Prints the grid and the chosen constants.

    python scripts/qb_fit.py [--data-dir data]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cfb.core.asof import Warehouse
from cfb.features.strength import qb as Q

LAST_SEASON = 2025


def history(wh: Warehouse) -> pd.DataFrame:
    """Player-game rows (through 2025, ordered) with prior features."""
    cut = [c for c in wh.cutoffs() if wh.season_of(c) <= LAST_SEASON + 1][-1]
    view = wh.view(cut)
    pg = Q.player_games(view)
    pg = pg[(pg["season"] <= LAST_SEASON) & (pg["plays"] > 0)].copy()
    rec = Q.recruit_info(view)
    pg = pg.merge(rec, on="athlete", how="left")
    ros = []
    for s in sorted(pg["season"].unique()):
        r = Q.roster_frame(view, int(s)).drop_duplicates("athlete")[["athlete", "class_year"]]
        ros.append(r.assign(season=s))
    pg = pg.merge(pd.concat(ros), on=["athlete", "season"], how="left")
    return pg.sort_values(["athlete", "start_time_utc", "game_id"]).reset_index(drop=True)


def design(pg: pd.DataFrame) -> np.ndarray:
    rating = pg["rating"].to_numpy(dtype=np.float64)
    has = np.isfinite(rating)
    cy = np.clip(np.nan_to_num(pg["class_year"].to_numpy(dtype=np.float64), nan=2.0), 1, 5)
    return np.column_stack([
        np.ones(len(pg)),
        np.where(has, np.nan_to_num(rating) - Q.PRIOR["rating_center"], 0.0),
        (~has).astype(float),
        np.nan_to_num(pg["juco"].to_numpy(dtype=np.float64)),
        cy - 2.0,
    ])


def prior_sums(pg: pd.DataFrame, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row: decayed sums of adj and plays over the player's earlier games."""
    a = np.zeros(len(pg))
    p = np.zeros(len(pg))
    ath = pg["athlete"].to_numpy()
    season = pg["season"].to_numpy(dtype=np.float64)
    adj, plays = pg["adj"].to_numpy(), pg["plays"].to_numpy()
    start = 0
    n = len(pg)
    while start < n:
        end = start
        while end < n and ath[end] == ath[start]:
            end += 1
        for i in range(start + 1, end):
            w = decay ** (season[i] - season[start:i])
            a[i] = np.sum(w * adj[start:i])
            p[i] = np.sum(w * plays[start:i])
        start = end
    return a, p


def fit(pg: pd.DataFrame, k: float, decay: float, sums=None):
    a, p = sums if sums is not None else prior_sums(pg, decay)
    X = design(pg)
    y = pg["adj"].to_numpy() / pg["plays"].to_numpy()
    wt = pg["plays"].to_numpy()
    b = k / (p + k)
    # q = a/(p+k) + b * X beta;  solve WLS of (y - a/(p+k)) on b*X.
    base = a / (p + k)
    Z = X * b[:, None]
    W = np.sqrt(wt)
    beta, *_ = np.linalg.lstsq(Z * W[:, None], (y - base) * W, rcond=None)
    q = base + Z @ beta
    mse = float(np.sum(wt * (y - q) ** 2) / wt.sum())
    return beta, mse, q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    pg = history(wh)
    print(f"player-games {len(pg):,}, players {pg['athlete'].nunique():,}, plays {pg['plays'].sum():,.0f}, "
          f"seasons {pg['season'].min()}-{pg['season'].max()}, rating known {pg['rating'].notna().mean():.2f}")
    y = pg["adj"] / pg["plays"]
    wt = pg["plays"]
    mean = float(np.sum(wt * y) / wt.sum())
    print(f"constant (play-weighted mean {mean:.4f}): mse {float(np.sum(wt * (y - mean) ** 2) / wt.sum()):.5f}")
    best = None
    for decay in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        sums = prior_sums(pg, decay)
        for k in (75, 100, 150, 200, 300, 500):
            beta, mse, _ = fit(pg, k, decay, sums)
            print(f"decay {decay:.1f} K {k:5d}: mse {mse:.5f}  beta {np.round(beta, 4).tolist()}")
            if best is None or mse < best[0]:
                best = (mse, decay, k, beta)
    mse, decay, k, beta = best
    print(f"\nbest: decay {decay}, K {k}, mse {mse:.5f}")
    print("PRIOR = {" + f'"const": {beta[0]:.4f}, "rating": {beta[1]:.4f}, "rating_center": '
          f'{Q.PRIOR["rating_center"]}, "no_rating": {beta[2]:.4f}, "juco": {beta[3]:.4f}, '
          f'"class_year": {beta[4]:.4f}' + "}")


if __name__ == "__main__":
    main()
