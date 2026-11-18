"""Player-impact study: how much an absent player's usage share costs, and whether it helps model_v2.

Inputs: data/exploration/impact_base.parquet (scripts/impact_base.py) and
data/exploration/impact_roles.parquet (scripts/impact_players.py).

Per team-game, "missing share" in category c = sum of sh_c over role players with no box-score
line. Three versions:
    oracle  absent in this game (known only afterwards; sizes the effect)
    new     oracle, only players who played the previous game (streak 0)
    prev    missed the previous game (known before kickoff): the pregame signal

Parts:
  1. oracle: margin residual vs model_v2 (season-forward) on home-minus-away missing shares
  2. pregame: season-forward 2018-2025 log loss of model_v2 + d_prev columns (ridge, frozen
     lam), paired bootstrap over weeks; then 2026 (fit 2015-2025)

    python scripts/impact_select.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select  # noqa: E402

from cfb.eval.bootstrap import paired_bootstrap  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel, log_loss, sigmoid  # noqa: E402

CATS = ["pass", "car", "rec", "ryd", "tkl", "dplay"]
OFF = ["QB", "RB", "WR", "TE"]
# Position -> the share that measures its role.
PRIMARY = {"QB": "pass", "RB": "car", "WR": "rec", "TE": "rec", "DL": "tkl", "LB": "tkl", "DB": "tkl"}


def team_features(r: pd.DataFrame) -> pd.DataFrame:
    """Missing-share columns per (game_id, team_id)."""
    cols = {}
    masks = {"orc": r["absent"] == 1,
             "new": (r["absent"] == 1) & (r["streak"] == 0),
             "old": (r["absent"] == 1) & (r["streak"] >= 1),
             "prev": r["absent_prev"] == 1,
             "prev1": (r["absent_prev"] == 1) & (r["streak"] == 1),     # missed only the last game
             "prev2": r["streak"] >= 2,                                  # missed 2+ in a row
             # diagnostic: a new absence that continued into the next game (likely injury) or not
             "sure": (r["absent"] == 1) & (r["streak"] == 0) & (r["absent_next"] == 1),
             "blip": (r["absent"] == 1) & (r["streak"] == 0) & (r["absent_next"] == 0)}
    off = r["pos"].isin(OFF)
    for k, m in masks.items():
        for c in CATS:
            # offensive categories from offensive players, defensive from everyone else
            sel = m & (off if c in ("pass", "car", "rec", "ryd") else ~off)
            cols[f"{k}_{c}"] = r["sh_" + c].where(sel, 0.0)
        cols[f"{k}_qbpass"] = r["sh_pass"].where(m & (r["pos"] == "QB"), 0.0)
        for pos, c in PRIMARY.items():
            cols[f"{k}_{pos}"] = r["sh_" + c].where(m & r["pos"].isin(pos.split("_")), 0.0)
    f = pd.DataFrame(cols)
    f["game_id"] = r["game_id"].to_numpy()
    f["team_id"] = r["team_id"].to_numpy()
    return f.groupby(["game_id", "team_id"]).sum().reset_index()


def add_diffs(X: pd.DataFrame, f: pd.DataFrame, names) -> pd.DataFrame:
    h = X[["game_id", "home_id"]].merge(f, left_on=["game_id", "home_id"], right_on=["game_id", "team_id"], how="left")
    a = X[["game_id", "away_id"]].merge(f, left_on=["game_id", "away_id"], right_on=["game_id", "team_id"], how="left")
    d = pd.DataFrame({f"d_imp_{n}": h[n].fillna(0.0).to_numpy() - a[n].fillna(0.0).to_numpy() for n in names},
                     index=X.index)
    return pd.concat([X, d], axis=1)


def ols(Z, y):
    Z = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    s2 = np.sum((y - Z @ b) ** 2) / (len(y) - Z.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
    return b[1:], se[1:]


def main():
    base = pd.read_parquet("data/exploration/impact_base.parquet")
    roles = pd.read_parquet("data/exploration/impact_roles.parquet")
    f = team_features(roles)
    names = [c for c in f.columns if c not in ("game_id", "team_id")]
    model_select.LAST_SEASON = 2026
    X, _ = model_select.load()
    X = add_diffs(X, f, names)

    # ---- 1. oracle sizes ---------------------------------------------------------------
    ev = base[base.season <= 2025].merge(X[["game_id"] + [f"d_imp_{n}" for n in names]], on="game_id")
    res = (ev["margin"] - ev["pred"]).to_numpy()
    print(f"eval games 2018-2025 {len(ev):,}; resid SD {res.std():.2f}")
    print("\n1. oracle: residual (points, home side) on home-minus-away missing share; coef = points per 100% share")
    sets = {
        "orc offense (pass, car, rec)": ["orc_pass", "orc_car", "orc_rec"],
        "orc offense (pass, ryd)": ["orc_pass", "orc_ryd"],
        "orc qb only": ["orc_qbpass"],
        "orc defense (tkl)": ["orc_tkl"],
        "orc defense (dplay)": ["orc_dplay"],
        "orc all (pass, car, rec, tkl)": ["orc_pass", "orc_car", "orc_rec", "orc_tkl"],
        "new all (pass, car, rec, tkl)": ["new_pass", "new_car", "new_rec", "new_tkl"],
        "old all (pass, car, rec, tkl)": ["old_pass", "old_car", "old_rec", "old_tkl"],
        "new by position": [f"new_{p}" for p in PRIMARY],
        "sure by position (diag.)": [f"sure_{p}" for p in PRIMARY],
        "blip by position (diag.)": [f"blip_{p}" for p in PRIMARY],
        "old by position": [f"old_{p}" for p in PRIMARY],
        "prev by position": [f"prev_{p}" for p in PRIMARY],
        "prev all (pass, car, rec, tkl)": ["prev_pass", "prev_car", "prev_rec", "prev_tkl"],
        "prev1 all (pass, car, rec, tkl)": ["prev1_pass", "prev1_car", "prev1_rec", "prev1_tkl"],
        "prev2 all (pass, car, rec, tkl)": ["prev2_pass", "prev2_car", "prev2_rec", "prev2_tkl"],
    }
    for lab, cs in sets.items():
        b, se = ols(ev[[f"d_imp_{c}" for c in cs]].to_numpy(), res)
        print(f"  {lab:<34} " + "  ".join(f"{c} {bb:+6.2f} ({ss:.2f})" for c, bb, ss in zip(cs, b, se)))
    # Oracle log loss upper bound: in-sample slope on the calibrated scale.
    cs = ["orc_pass", "orc_car", "orc_rec", "orc_tkl"]
    Z = ev[[f"d_imp_{c}" for c in cs]].to_numpy()
    b, _ = ols(Z, res)
    lin2 = ev["lin"] + ev["calib_s"] * (Z @ b)
    print(f"  log loss model_v2 {log_loss(ev['y'], sigmoid(ev['lin'])):.4f} -> oracle term {log_loss(ev['y'], sigmoid(lin2)):.4f}"
          " (in-sample, upper bound for knowing every absence)")

    # ---- 2. pregame signal in the model ---------------------------------------------------
    cols0 = list(C.MODEL_V2_COLUMNS)
    seasons = list(range(2018, 2026))

    def sf(extra, s_list=seasons):
        cols = cols0 + [f"d_imp_{c}" for c in extra]
        preds = []
        for s in s_list:
            tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < s) & X["y"].notna()]
            te = X[(X["season"] == s) & X["in_eval"] & X["y"].notna()]
            m = MarginModel(cols, lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
            preds.append(pd.Series(m.predict_proba(te), index=te.index))
            last = m
        return pd.concat(preds), last

    p0, _ = sf([])
    d = X.loc[p0.index]
    cl = (d["season"].astype(str) + "_" + d["season_type"].astype(str) + "_" + d["week"].astype(str)).to_numpy()
    per0 = {s: log_loss(d.loc[d.season == s, "y"], p0[d.season == s]) for s in seasons}
    print(f"\n2. pregame, season-forward 2018-2025: model_v2 {log_loss(d['y'], p0):.4f} ({len(d):,} games)")
    variants = {
        "prev pass,car,rec,tkl": ["prev_pass", "prev_car", "prev_rec", "prev_tkl"],
        "prev1 pass,car,rec,tkl": ["prev1_pass", "prev1_car", "prev1_rec", "prev1_tkl"],
        "prev qbpass": ["prev_qbpass"],
        "prev1 qbpass": ["prev1_qbpass"],
        "prev pass,ryd": ["prev_pass", "prev_ryd"],
        "prev pass,ryd,tkl": ["prev_pass", "prev_ryd", "prev_tkl"],
        "prev by position": [f"prev_{p}" for p in PRIMARY],
        "prev1 by position": [f"prev1_{p}" for p in PRIMARY],
        "prev1 + prev2 (pass,ryd,tkl)": ["prev1_pass", "prev1_ryd", "prev1_tkl", "prev2_pass", "prev2_ryd", "prev2_tkl"],
    }
    out = {}
    for lab, extra in variants.items():
        p, m = sf(extra)
        b = paired_bootstrap(d["y"].to_numpy(), p.loc[d.index].to_numpy(), p0.to_numpy(), clusters=cl)
        per = {s: log_loss(d.loc[d.season == s, "y"], p[d.season == s]) for s in seasons}
        better = sum(per[s] < per0[s] for s in seasons)
        coefs = dict(zip(m.prep.names, m.coef))
        cpts = " ".join(f"{c}={coefs.get('d_imp_' + c, np.nan):+.2f}" for c in extra)
        print(f"  + {lab:<30} {log_loss(d['y'], p):.4f}  diff {b['diff']:+.4f} (SE {b['se']:.4f})  better {better}/8"
              f"   ridge coef (std units, fit 2015-2024): {cpts}")
        out[lab] = extra

    # ---- 3. 2026 walk-forward (fit 2015-2025) ---------------------------------------------
    q0, _ = sf([], [2026])
    d6 = X.loc[q0.index]
    cl6 = (d6["week"].astype(str)).to_numpy()
    print(f"\n3. 2026 walk-forward ({len(d6)} games): model_v2 {log_loss(d6['y'], q0):.4f}")
    for lab, extra in out.items():
        q, _ = sf(extra, [2026])
        b = paired_bootstrap(d6["y"].to_numpy(), q.loc[d6.index].to_numpy(), q0.to_numpy(), clusters=cl6)
        print(f"  + {lab:<30} {log_loss(d6['y'], q):.4f}  diff {b['diff']:+.4f} (SE {b['se']:.4f})")


if __name__ == "__main__":
    main()
