"""Do the play-text groups add to model_v2? (str_pass_ci, matchup_run_dir, matchup_pass_zone)

The detailed play text starts in 2025 week 9, so the groups cannot enter the 2018-2025
season-forward selection. Test instead:

  Base     model_v2 as the weekly report runs it: 2025 from the fit before the season,
           2026 refit at every cutoff (cfb.predict.run). Cached in data/lab/.
  Stage 2  logistic regression with offset logit(p_v2) and no intercept, on the new columns
           as standardized home-minus-away differences (missing = 0 after standardizing),
           ridge LAM per observation. Walk-forward over 2026: at each cutoff, fit on every
           earlier baseline-set game from 2025 week 9 on, predict the slate.
  Score    2026 baseline-set games (FBS vs FBS, closing spread, both Elos), log loss against
           model_v2, paired bootstrap over weeks.

    python scripts/playtext_select.py [--lam 0.03]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_select import eval_ids  # noqa: E402

from cfb.core import Warehouse  # noqa: E402
from cfb.eval.bootstrap import paired_bootstrap  # noqa: E402
from cfb.model import config as C  # noqa: E402
from cfb.model.matrix import game_matrix, groups_by_name, stored_features  # noqa: E402
from cfb.model.models import log_loss  # noqa: E402
from cfb.predict import run  # noqa: E402

GROUPS = {
    "str_pass_ci": ["str_pci_qb", "str_pci_rec", "str_pci_def", "str_pci_n"],
    "matchup_run_dir": ["mat_rdir_fit", "mat_rdir_exp", "mat_rdir_n"],
    "matchup_pass_zone": ["mat_pzone_fit", "mat_pzone_exp", "mat_pzone_n"],
}
ALL = [c for cs in GROUPS.values() for c in cs]
VARIANTS = {
    "str_pass_ci (4)": GROUPS["str_pass_ci"],
    "  qb": ["str_pci_qb"],
    "  qb, rec": ["str_pci_qb", "str_pci_rec"],
    "  def": ["str_pci_def"],
    "matchup_run_dir (3)": GROUPS["matchup_run_dir"],
    "  rdir fit": ["mat_rdir_fit"],
    "  rdir exp": ["mat_rdir_exp"],
    "matchup_pass_zone (3)": GROUPS["matchup_pass_zone"],
    "  pzone fit": ["mat_pzone_fit"],
    "  pzone exp": ["mat_pzone_exp"],
    "fits only (rdir, pzone)": ["mat_rdir_fit", "mat_pzone_fit"],
    "all 10": ALL,
}


def sig(z):
    return 1 / (1 + np.exp(-z))


def base_predictions(wh: Warehouse, data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    cache = data_dir / "lab" / "model_v2_2025_2026.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    p25 = run(wh, [2025], "season", model_id=C.MODEL_V2_ID)
    p26 = run(wh, [2026], "cutoff", model_id=C.MODEL_V2_ID)
    out = pd.concat([p25, p26])[["game_id", "p_home_win"]].rename(columns={"p_home_win": "p_base"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def frame(data_dir="data", refresh=False) -> pd.DataFrame:
    data_dir = Path(data_dir)
    wh = Warehouse(data_dir)
    gi = wh.game_index
    gi = gi[gi["season"].isin([2025, 2026])].reset_index(drop=True)
    groups = groups_by_name(list(GROUPS))
    X = game_matrix(gi, stored_features(data_dir, groups), groups)
    X["cutoff_utc"] = gi["cutoff_utc"].to_numpy()
    X["margin"] = (gi["home_points"] - gi["away_points"]).to_numpy(dtype="float64")
    X["y"] = (X["margin"] > 0).astype("float64")
    X.loc[~gi["completed"].to_numpy() | (X["margin"] == 0).to_numpy(), "y"] = np.nan
    X["in_eval"] = X["game_id"].isin(eval_ids(wh)) & gi["completed"].to_numpy() & (X["fcs_game"] == 0)
    X = X.merge(base_predictions(wh, data_dir, refresh), on="game_id", how="left")
    have = X[[f"d_{c}" for c in ALL]].notna().any(axis=1)
    return X[have & X["in_eval"] & X["y"].notna() & X["p_base"].notna()].reset_index(drop=True)


def fit_offset_logit(Z: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float, iters: int = 50):
    """Ridge logistic regression with an offset and no intercept (Newton). Returns coef and SE."""
    n, k = Z.shape
    b = np.zeros(k)
    pen = lam * n * np.eye(k)
    for _ in range(iters):
        p = sig(off + Z @ b)
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, Z.T @ (y - p) - pen @ b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    p = sig(off + Z @ b)
    H = Z.T @ (Z * (p * (1 - p))[:, None]) + pen
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1).fillna(1)
    zt = ((train[cols] - mu) / sd).fillna(0.0).to_numpy()
    zs = ((test[cols] - mu) / sd).fillna(0.0).to_numpy()
    return zt, zs


def walk_forward(D: pd.DataFrame, cols, lam: float) -> pd.Series:
    logit_base = np.log(D["p_base"] / (1 - D["p_base"]))
    out = pd.Series(np.nan, index=D.index)
    test = D[D["season"] == 2026]
    for cutoff, te in test.groupby("cutoff_utc"):
        tr = D[D["cutoff_utc"] < cutoff]
        if len(tr) < 100:
            continue
        zt, zs = standardize(tr, te, cols)
        b, _ = fit_offset_logit(zt, tr["y"].to_numpy(), logit_base[tr.index].to_numpy(), lam)
        out[te.index] = sig(logit_base[te.index].to_numpy() + zs @ b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--refresh", action="store_true", help="recompute the cached model_v2 predictions")
    a = ap.parse_args(argv)
    D = frame(refresh=a.refresh)
    te = D[D["season"] == 2026]
    print(f"games from 2025 week 9 on in the baseline set with play-text columns: {len(D)} "
          f"(2025: {int((D.season == 2025).sum())}, 2026: {len(te)}); lam {a.lam}")
    base_ll = log_loss(te["y"], te["p_base"])
    print(f"{'model_v2':<26} 2026 log loss {base_ll:.4f}")
    clusters = (te["season"].astype(str) + "_" + te["season_type"].astype(str) + "_" + te["week"].astype(str)).to_numpy()
    for name, cols in VARIANTS.items():
        dcols = [f"d_{c}" for c in cols]
        p = walk_forward(D, dcols, a.lam)[te.index]
        ok = p.notna().to_numpy()
        b = paired_bootstrap(te["y"].to_numpy()[ok], p.to_numpy()[ok], te["p_base"].to_numpy()[ok], clusters=clusters[ok])
        weeks = pd.DataFrame({"w": clusters[ok], "d": (
            -(te["y"].to_numpy()[ok] * np.log(p.to_numpy()[ok]) + (1 - te["y"].to_numpy()[ok]) * np.log(1 - p.to_numpy()[ok]))
            + (te["y"].to_numpy()[ok] * np.log(te["p_base"].to_numpy()[ok]) + (1 - te["y"].to_numpy()[ok]) * np.log(1 - te["p_base"].to_numpy()[ok])))})
        wk = weeks.groupby("w")["d"].mean()
        print(f"{name:<26} {log_loss(te['y'].to_numpy()[ok], p.to_numpy()[ok]):.4f}  vs v2 {b['diff']:+.4f} "
              f"(SE {b['se']:.4f})  better in {(wk < 0).sum()}/{len(wk)} weeks  n {ok.sum()}")
    # Descriptive: one fit on every game (in sample), lam ~0, coefficient per SD of the difference.
    logit_base = np.log(D["p_base"] / (1 - D["p_base"])).to_numpy()
    print("\nIn-sample, one column at a time (all games from 2025 week 9 on): log-odds per SD, z")
    for c in ALL:
        z, _ = standardize(D, D, [f"d_{c}"])
        b, se = fit_offset_logit(z, D["y"].to_numpy(), logit_base, 1e-6)
        print(f"  {c:<16} {b[0]:+.3f}  z {b[0] / se[0]:+.2f}")


if __name__ == "__main__":
    main()
