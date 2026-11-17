"""Fit the preseason projection coefficients in cfb.features.strength.prior (2015-2025 only).

For each season S the components come from a view at S's first cutoff (exactly what the
feature code sees); the target is S's final in-house points rating (offense and defense
deviations, ridge without a prior). Ordinary least squares, pooled over seasons, with a
leave-one-season-out R^2 for reference.

    python scripts/strength_fit_prior.py [--data-dir data] [--version 2]

Version 1 is the projection of model_v1 (COMPONENTS); version 2 adds the player components
(COMPONENTS_V2, model_v2).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cfb.core import Warehouse
from cfb.features.strength import points as P
from cfb.features.strength import prior as PR
from cfb.features.strength.units import TeamIndex


def frame(wh: Warehouse, seasons, version: int = 1) -> pd.DataFrame:
    rows = []
    tg = wh.prepared("team_game").frame
    teams = wh.prepared("teams").frame
    for s in seasons:
        first = min(c for c in wh.cutoffs() if wh.season_of(c) == s)
        v = wh.view(first)
        comp = PR.components(v) if version == 1 else PR.components_v2(v)
        idx = TeamIndex.from_teams(teams[teams["season"] == s])
        fin = P.final_fit(P.observations(tg[tg["season"] == s][P.TG_COLUMNS]), idx, P.DEFAULT)
        pos = idx.index(comp.index.to_numpy())
        comp = comp.assign(season=s, fin_off=fin.off[pos], fin_def=fin.deff[pos])
        rows.append(comp.reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--version", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    wh = Warehouse(args.data_dir)
    X = frame(wh, range(2015, 2026), args.version)
    X = X[X["prev_off"].notna()].copy()
    names = PR.COMPONENTS if args.version == 1 else PR.COMPONENTS_V2
    suffix = "" if args.version == 1 else "_V2"
    cols = ["one"] + names
    X[names] = X[names].fillna(0.0)
    X["one"] = 1.0
    for target, name in (("fin_off", "COEF_OFF"), ("fin_def", "COEF_DEF")):
        b = np.linalg.lstsq(X[cols].to_numpy(), X[target].to_numpy(), rcond=None)[0]
        err = tot = 0.0
        for s in X["season"].unique():
            tr, te = X[X["season"] != s], X[X["season"] == s]
            bb = np.linalg.lstsq(tr[cols].to_numpy(), tr[target].to_numpy(), rcond=None)[0]
            err += float(((te[target] - te[cols].to_numpy() @ bb) ** 2).sum())
            tot += float(((te[target] - te[target].mean()) ** 2).sum())
        prev_only = ["one", "prev_off", "prev_def"]
        err0 = 0.0
        for s in X["season"].unique():
            tr, te = X[X["season"] != s], X[X["season"] == s]
            bb = np.linalg.lstsq(tr[prev_only].to_numpy(), tr[target].to_numpy(), rcond=None)[0]
            err0 += float(((te[target] - te[prev_only].to_numpy() @ bb) ** 2).sum())
        print(f"{name}{suffix} = {{{', '.join(f'{c!r}: {v:.4f}' for c, v in zip(cols, b))}}}")
        print(f"  n={len(X)}  leave-one-season-out R^2 {1 - err / tot:.3f}  (last season only: {1 - err0 / tot:.3f})")


if __name__ == "__main__":
    main()
