"""Choose unit rating parameters on seasons through 2025 (never 2026).

For every cutoff of seasons 2015 to 2025 the ratings are fitted on what was visible
and used to predict each FBS vs FBS offense-defense pairing of that cutoff's slate:

    pred = mu + hfa * h + off[offense] + def[defense]

The score is the play-weighted mean squared error against the game's actual value
(non-garbage scrimmage plays). Parameters are searched one metric at a time, since a
metric's parameters affect only its own fit.

    python scripts/strength_tune_units.py [--data-dir data] [--seasons 2015-2025]
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.core.asof import RESULT_LAG
from cfb.features.strength import units as U


def load(wh: Warehouse, seasons: list[int]):
    gi = wh.game_index
    pbp = wh.prepared("pbp").frame[U.PBP_COLUMNS + ["season"]]
    pbp = pbp[pbp["season"].isin([min(seasons) - 1] + seasons)]
    obs = U.observations(pbp.drop(columns="season"), gi)
    teams = wh.prepared("teams").frame
    return gi, obs, teams


def evaluate(gi, obs, teams, seasons, params: U.UnitParams, kinds=("all", "rush", "pass")):
    """Weighted MSE per (kind, metric), summed over the seasons' cutoffs."""
    lag = int(pd.Timedelta(RESULT_LAG).value)
    sse = {}
    wsum = {}
    for season in seasons:
        o_cur = obs[obs["season"] == season]
        o_prev = obs[obs["season"] == season - 1]
        t_cur = teams[teams["season"] == season]
        t_prev = teams[teams["season"] == season - 1]
        fbs = set(t_cur.loc[t_cur["fbs"].astype(bool), "team_id"])
        cut = gi[gi["season"] == season][["game_id", "cutoff_utc"]]
        o_cur = o_cur.merge(cut, on="game_id")
        prior_idx = U.TeamIndex.from_teams(t_prev) if len(o_prev) else None
        prior = None
        if prior_idx is not None:
            prior = U.season_fits(o_prev, prior_idx, int(o_prev["start_ns"].max()) + U.DAY_NS, params)
        idx = U.TeamIndex.from_teams(t_cur)
        for cutoff, slate in o_cur.groupby("cutoff_utc"):
            c_ns = int(cutoff.value)
            seen = o_cur[o_cur["start_ns"] + lag <= c_ns]
            slate = slate[slate["offense_id"].isin(fbs) & slate["defense_id"].isin(fbs)]
            if slate.empty:
                continue
            fits = U.season_fits(seen, idx, c_ns, params, prior, prior_idx)
            oi, di = idx.index(slate["offense_id"]), idx.index(slate["defense_id"])
            h = slate["h"].to_numpy()
            for (kind, metric), f in fits.items():
                if kind not in kinds:
                    continue
                num_col, w_col = U._METRIC_SOURCES[metric]
                w = slate[f"{kind}_{w_col}"].to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = slate[f"{kind}_{num_col}"].to_numpy(float) / w
                pred = f.mu + f.hfa * h + f.off[oi] + f.deff[di]
                ok = (w > 0) & np.isfinite(pred) & np.isfinite(y)
                sse[(kind, metric)] = sse.get((kind, metric), 0.0) + float((w[ok] * (y[ok] - pred[ok]) ** 2).sum())
                wsum[(kind, metric)] = wsum.get((kind, metric), 0.0) + float(w[ok].sum())
    return {k: sse[k] / wsum[k] for k in sse}


def null_mse(gi, obs, teams, seasons):
    """MSE of predicting the league mean of the season so far (no team information)."""
    p = U.UnitParams(epa=U.MetricParams(1e12, np.inf, 0.0), success=U.MetricParams(1e12, np.inf, 0.0),
                     explosive=U.MetricParams(1e12, np.inf, 0.0), lam_hfa=1e12)
    return evaluate(gi, obs, teams, seasons, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seasons", default="2015-2025")
    ap.add_argument("--quick", action="store_true", help="every other season only")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    assert hi <= 2025, "2026 is the test season"
    seasons = list(range(lo, hi + 1))
    if args.quick:
        seasons = seasons[::2]
    wh = Warehouse(args.data_dir)
    gi, obs, teams = load(wh, seasons)

    base = U.DEFAULT
    t = time.time()
    ref = evaluate(gi, obs, teams, seasons, base)
    nul = null_mse(gi, obs, teams, seasons)
    print(f"default params ({time.time() - t:.0f}s)")
    for k in sorted(ref):
        print(f"  {k}: mse {ref[k]:.6f}  null {nul[k]:.6f}  skill {1 - ref[k] / nul[k]:.4f}")

    grids = {
        "epa": dict(lam=[300, 450], half_life=[32, np.inf], rho=[0.8, 0.9, 1.0]),
        "success": dict(lam=[300, 450], half_life=[32, np.inf], rho=[0.8, 0.9, 1.0]),
        "explosive": dict(lam=[300, 600, 1200], half_life=[np.inf], rho=[0.6, 0.7, 0.8]),
    }
    best = {}
    for metric, grid in grids.items():
        rows = []
        for lam, hl, rho in itertools.product(grid["lam"], grid["half_life"], grid["rho"]):
            mp = U.MetricParams(lam, hl, rho)
            p = replace(base, **{metric: mp}, metrics=(metric,))
            res = evaluate(gi, obs, teams, seasons, p)
            score = np.mean([res[(k, metric)] / nul[(k, metric)] for k in ("all", "rush", "pass")])
            rows.append({"lam": lam, "half_life": hl, "rho": rho, "rel_mse": score,
                         **{f"{k}": res[(k, metric)] for k in ("all", "rush", "pass")}})
        table = pd.DataFrame(rows).sort_values("rel_mse")
        print(f"\n{metric}: best 8 of {len(table)} (rel_mse = mean of mse/null over all, rush, pass)")
        print(table.head(8).to_string(index=False))
        best[metric] = table.iloc[0]
    print("\nchosen:")
    for metric, row in best.items():
        print(f"  {metric}: lam={row['lam']} half_life={row['half_life']} rho={row['rho']} rel_mse={row['rel_mse']:.5f}")


if __name__ == "__main__":
    main()
