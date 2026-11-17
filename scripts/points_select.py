"""Season-forward selection for the phase 2 points models: home margin (spread) and game
total. Seasons through 2025 only; 2026 is never read.

Fit on seasons [TRAIN_START, S-1], score S, S = 2018..2025. Scored games: completed FBS vs
FBS games with a closing spread (margin) or closing total (total). The closing line is the
benchmark, never an input.

    python scripts/points_select.py margin [--lam 0.03]   # candidate feature sets for the spread
    python scripts/points_select.py total  [--lam 0.03]   # candidate feature sets for the total
    python scripts/points_select.py tune                  # lam, era weighting, target caps
    python scripts/points_select.py final                 # the frozen configuration, per season

Columns: MAE and RMSE in points, bias (mean actual minus predicted), the closing line's MAE
on the same games, and the hit rate against the closing line (pushes left out) for all
games and for games where the model and the line differ by 3 and 7 points or more.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.model import points_config as P
from cfb.model.matrix import feature_columns, game_matrix, groups_by_name, stored_features
from cfb.model.models import PointsModel

SEASONS = range(2018, 2026)
LAST_SEASON = 2025
THRESHOLDS = (0.0, 3.0, 7.0)


def load(data_dir="data"):
    wh = Warehouse(data_dir)
    gi = wh.game_index
    gi = gi[gi["season"] <= LAST_SEASON].reset_index(drop=True)   # 2026 never enters selection
    groups = [g for g in groups_by_name() if (Path(data_dir) / "features" / f"{g.name}.parquet").exists()]
    feats = stored_features(data_dir, groups)
    X = game_matrix(gi, feats, groups)
    X["completed"] = gi["completed"].to_numpy()
    X["margin"] = (gi["home_points"] - gi["away_points"]).to_numpy(dtype="float64")
    X["total"] = (gi["home_points"] + gi["away_points"]).to_numpy(dtype="float64")
    X.loc[~X["completed"].astype(bool), ["margin", "total"]] = np.nan
    fbs = X["fcs_game"] == 0
    X["eval_margin"] = fbs & X["margin"].notna() & X["mkt_spread_close"].notna()
    X["eval_total"] = fbs & X["total"].notna() & X["mkt_total_close"].notna()
    return X, groups


def cols(groups: list[str], forms: tuple[str, ...], base=("site", "fcs_game"), extra=()) -> list[str]:
    return list(base) + feature_columns(groups_by_name(groups), forms) + list(extra)


def season_forward(X, target: str, make, train_start=P.TRAIN_START, cap=None, seasons=SEASONS) -> pd.Series:
    """Out-of-season predictions of `target` on the eval games of `seasons`."""
    preds = []
    for s in seasons:
        tr = X[(X["season"] >= train_start) & (X["season"] < s) & X[target].notna()]
        te = X[(X["season"] == s) & X[f"eval_{target}"]]
        y = tr[target].to_numpy(float)
        if cap is not None:
            y = np.clip(y, -cap, cap) if target == "margin" else np.minimum(y, cap)
        m = make().fit(tr, y)
        preds.append(pd.Series(m.predict(te), index=te.index))
    return pd.concat(preds)


def line_of(d: pd.DataFrame, target: str) -> np.ndarray:
    """The closing line as a prediction of the target: -spread for the margin."""
    return -d["mkt_spread_close"].to_numpy(float) if target == "margin" else d["mkt_total_close"].to_numpy(float)


def scores(X, target: str, pred: pd.Series) -> dict:
    d = X.loc[pred.index]
    y = d[target].to_numpy(float)
    p = pred.to_numpy(float)
    line = line_of(d, target)
    out = {"n": len(y), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "bias": float(np.mean(y - p)), "line_mae": float(np.mean(np.abs(y - line)))}
    edge = p - line            # model above the line: home covers / over
    res = y - line             # actual above the line
    for t in THRESHOLDS:
        k = (np.abs(edge) >= max(t, 1e-9)) & (res != 0)
        hit = np.sign(edge[k]) == np.sign(res[k])
        out[f"hit{t:g}"] = (float(hit.mean()) if k.any() else float("nan"), int(k.sum()))
    out["per_season_mae"] = {int(s): float(np.mean(np.abs(y[d["season"].to_numpy() == s] - p[d["season"].to_numpy() == s])))
                             for s in sorted(d["season"].unique())}
    return out


def report(name: str, sc: dict, t0=None):
    hits = "  ".join(f"{100 * sc[f'hit{t:g}'][0]:.1f}%/{sc[f'hit{t:g}'][1]}" for t in THRESHOLDS)
    extra = f" ({time.time() - t0:.0f}s)" if t0 else ""
    print(f"{name:<52} MAE {sc['mae']:.3f} RMSE {sc['rmse']:.3f} bias {sc['bias']:+.2f} "
          f"line {sc['line_mae']:.3f} | ATS/OU all,3+,7+ {hits}{extra}", flush=True)


def run(X, target, name, columns, lam, **kw):
    t0 = time.time()
    columns = [c for c in columns if c in X.columns]
    sc = scores(X, target, season_forward(X, target, lambda: PointsModel(columns, lam=lam, era_half_life=kw.pop("hl", None)), **kw))
    report(f"{name} [{len(columns)}]", sc, t0)
    return sc


def cmd_margin(X, groups, args):
    lam = args.lam
    have = {g.name for g in groups}
    v1 = ["str_power", "str_elo"]
    run(X, "margin", "model_v1 inputs (str_power+str_elo d)", cols(v1, ("d",)), lam)
    run(X, "margin", "str_power only (d)", cols(["str_power"], ("d",)), lam)
    run(X, "margin", "v1 inputs h,a", cols(v1, ("h", "a")), lam)
    cands = [g for g in P.CANDIDATE_GROUPS if g in have and g not in v1]
    for g in cands:
        run(X, "margin", f"  v1 + {g}", cols(v1 + [g], ("d",)), lam)
    run(X, "margin", "all candidates (d)", cols(v1 + cands, ("d",)), lam)
    run(X, "margin", "all candidates (d), lam 0.3", cols(v1 + cands, ("d",)), 0.3)
    run(X, "margin", "v1 + ctx_weather (observed; diagnostic)", cols(v1 + ["ctx_weather"], ("d",)), lam)


def cmd_total(X, groups, args):
    lam = args.lam
    have = {g.name for g in groups}
    base = ["str_power"]
    run(X, "total", "str_power (s)", cols(base, ("s",)), lam)
    run(X, "total", "str_power (h,a)", cols(base, ("h", "a")), lam)
    run(X, "total", "str_power (s,d)", cols(base, ("s", "d")), lam)
    cands = [g for g in P.CANDIDATE_GROUPS if g in have and g not in base]
    for g in cands:
        run(X, "total", f"  str_power + {g} (s)", cols(base + [g], ("s",)), lam)
    run(X, "total", "all candidates (s)", cols(base + cands, ("s",)), lam)
    run(X, "total", "all candidates (s), lam 0.3", cols(base + cands, ("s",)), 0.3)
    run(X, "total", "str_power + ctx_weather (observed; diagnostic)", cols(base + ["ctx_weather"], ("s",)), lam)


def cmd_forward(X, groups, args):
    """Greedy forward selection of groups for args.target, stop when MAE gain < 0.01."""
    target = args.target
    have = {g.name for g in groups}
    form = ("d",) if target == "margin" else ("s",)
    chosen = ["str_power"]
    cur = run(X, target, f"start {chosen}", cols(chosen, form, extra=P.EXTRA[target]), args.lam)
    pool = [g for g in P.CANDIDATE_GROUPS if g in have and g not in chosen]
    while pool:
        best, best_sc = None, None
        for g in pool:
            sc = run(X, target, f"  + {g}", cols(chosen + [g], form, extra=P.EXTRA[target]), args.lam)
            if best_sc is None or sc["mae"] < best_sc["mae"]:
                best, best_sc = g, sc
        better = sum(best_sc["per_season_mae"][s] < cur["per_season_mae"][s] for s in SEASONS)
        print(f"best {best}: MAE {best_sc['mae']:.3f} (gain {cur['mae'] - best_sc['mae']:.3f}), "
              f"better in {better}/8 seasons", flush=True)
        if cur["mae"] - best_sc["mae"] < 0.01 or better < 5:
            break
        chosen.append(best)
        pool.remove(best)
        cur = best_sc
    print(f"selected {chosen}")


def cmd_tune(X, groups, args):
    for target in ("margin", "total"):
        c = [c for c in P.model_columns(target) if c in X.columns]
        print(f"== {target}: {P.GROUPS[target]} {P.FORMS[target]}")
        for lam in (0.003, 0.01, 0.03, 0.1, 0.3):
            run(X, target, f"lam {lam}", c, lam)
        for hl in (2, 4, 8):
            run(X, target, f"lam {P.LAM[target]} era half-life {hl}", c, P.LAM[target], hl=hl)
        for start in (2014, 2016, 2018):
            if start < 2018:
                run(X, target, f"lam {P.LAM[target]} train from {start}", c, P.LAM[target], train_start=start)
        caps = (28, 35, 42) if target == "margin" else (80, 90, 100)
        for cap in caps:
            run(X, target, f"lam {P.LAM[target]} target capped at {cap}", c, P.LAM[target], cap=cap)


def paired_mae(X, target: str, a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Mean difference in absolute error (a minus b) and its SE with whole weeks as clusters."""
    d = X.loc[a.index]
    y = d[target].to_numpy(float)
    diff = np.abs(y - a.to_numpy()) - np.abs(y - b.loc[a.index].to_numpy())
    key = (d["season"].astype(str) + d["season_type"] + d["week"].astype(str)).to_numpy()
    sums = pd.Series(diff).groupby(key).sum()
    return float(diff.mean()), float(np.sqrt(len(sums)) * sums.std() / len(diff))


REFERENCE = {"margin": (["str_power", "str_elo"], "model_v1 inputs"), "total": (["str_power"], "str_power only")}


def cmd_final(X, groups, args):
    for target in ("margin", "total"):
        pred = season_forward(X, target, lambda: P.make_model(target), cap=P.CAP[target])
        sc = scores(X, target, pred)
        report(f"FINAL {target} {P.GROUPS[target]}", sc)
        print("  per season MAE:", " ".join(f"{s}:{v:.2f}" for s, v in sc["per_season_mae"].items()))
        gs, name = REFERENCE[target]
        ref = season_forward(X, target, lambda: PointsModel(cols(gs, P.FORMS[target]), lam=P.LAM[target]))
        diff, se = paired_mae(X, target, pred, ref)
        print(f"  vs {name}: MAE {diff:+.4f} (SE {se:.4f}, weeks as clusters)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["margin", "total", "forward", "tune", "final"])
    ap.add_argument("--target", choices=["margin", "total"], default="margin")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lam", type=float, default=0.03)
    args = ap.parse_args()
    t = time.time()
    X, groups = load(args.data_dir)
    print(f"matrix {X.shape}; eval games 2018-2025: margin {int((X['eval_margin'] & (X['season'] >= 2018)).sum())}, "
          f"total {int((X['eval_total'] & (X['season'] >= 2018)).sum())} ({time.time() - t:.0f}s)", flush=True)
    {"margin": cmd_margin, "total": cmd_total, "forward": cmd_forward, "tune": cmd_tune,
     "final": cmd_final}[args.cmd](X, groups, args)


if __name__ == "__main__":
    sys.exit(main())
