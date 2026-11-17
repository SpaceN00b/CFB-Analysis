"""Experiment harness for the preseason prior (P2): swap the power rating's prior and lam,
then score model_v1's configuration season-forward (2018-2025 only; 2026 is never read).

Caches per cutoff the visible team-game observations, so a new prior costs one ridge fit
per cutoff (about 10 s for all 2015-2025 cutoffs).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model_select import load, season_forward  # noqa: E402

from cfb.core import Warehouse
from cfb.features.strength import points as P
from cfb.features.strength import prior as PR
from cfb.features.strength.common import team_keys
from cfb.features.strength.power import POINTS_SD, POWER
from cfb.features.strength.units import TeamIndex
from cfb.model import config as C
from cfb.model.models import log_loss

SEASONS = list(range(2015, 2026))
PWR_COLS = ["str_pwr_off", "str_pwr_def", "str_pwr_net", "str_pwr_prior_share", "str_pwr_net_se"]


def build_cache(data_dir="data", path=None):
    """Per season: components at the first cutoff, previous final fit, final fit target, and
    per cutoff the observations visible and the slate's team keys."""
    if path and Path(path).exists():
        return pickle.loads(Path(path).read_bytes())
    wh = Warehouse(data_dir)
    gi = wh.game_index
    teams_all = wh.prepared("teams").frame
    tg_all = wh.prepared("team_game").frame
    out = {}
    for s in SEASONS:
        cuts = [c for c in wh.cutoffs() if wh.season_of(c) == s]
        first = wh.view(cuts[0])
        comp = PR.components(first)
        prev, _ = PR.previous_final(first, s)
        idx = TeamIndex.from_teams(teams_all[teams_all["season"] == s])
        fin = P.final_fit(P.observations(tg_all[tg_all["season"] == s][P.TG_COLUMNS]), idx, P.DEFAULT)
        per_cut = []
        for c in cuts:
            v = wh.view(c)
            obs = P.observations(v.table("team_game", columns=P.TG_COLUMNS, seasons=s))
            games = gi[(gi["season"] == s) & (gi["cutoff_utc"] == c)]
            if games.empty:
                continue
            per_cut.append((c, obs, team_keys(games)[["game_id", "team_id"]]))
        out[s] = dict(comp=comp, prev=prev, idx=idx, fin=fin, cuts=per_cut)
        print(f"cache {s}: {len(per_cut)} cutoffs", flush=True)
    if path:
        Path(path).write_bytes(pickle.dumps(out))
    return out


def targets(cache) -> pd.DataFrame:
    """Components + final off/def deviations, one row per FBS team-season with a previous season."""
    rows = []
    for s, c in cache.items():
        comp = c["comp"]
        pos = c["idx"].index(comp.index.to_numpy())
        rows.append(comp.assign(season=s, fin_off=c["fin"].off[pos], fin_def=c["fin"].deff[pos]).reset_index())
    X = pd.concat(rows, ignore_index=True)
    return X[X["prev_off"].notna()].reset_index(drop=True)


def loso_projection(T: pd.DataFrame, cols: list[str], ridge: float = 0.0) -> pd.DataFrame:
    """Leave-one-season-out projected (off, def) per team-season, plus R^2 per side."""
    Z = T[cols].fillna(0.0).to_numpy()
    Z = np.column_stack([np.ones(len(T)), Z])
    pen = ridge * np.eye(Z.shape[1]); pen[0, 0] = 0.0
    out = pd.DataFrame({"team_id": T["team_id"], "season": T["season"]})
    r2 = {}
    for side in ("off", "def"):
        y = T[f"fin_{side}"].to_numpy()
        pred = np.empty(len(T))
        for s in T["season"].unique():
            tr = (T["season"] != s).to_numpy()
            b = np.linalg.solve(Z[tr].T @ Z[tr] + pen, Z[tr].T @ y[tr])
            pred[~tr] = Z[~tr] @ b
        out[f"p_{side}"] = pred
        tot = sum(((y[T.season == s] - y[T.season == s].mean()) ** 2).sum() for s in T.season.unique())
        r2[side] = 1 - ((y - pred) ** 2).sum() / tot
    out.attrs["r2"] = r2
    return out


def power_columns(cache, proj: pd.DataFrame | None, lam: float = POWER.lam,
                  half_life: float = POWER.half_life) -> pd.DataFrame:
    """str_power's prior-dependent columns per (game, team), with the prior from `proj`
    (team_id, season, p_off, p_def deviations) or the frozen projection when None."""
    params = P.PointsParams(lam=lam, half_life=half_life, lam_mu=POWER.lam_mu, lam_hfa=POWER.lam_hfa)
    frames = []
    for s, c in cache.items():
        idx, comp, prev = c["idx"], c["comp"], c["prev"]
        if prev is None:
            continue
        if proj is None:
            o, d = PR.projection(comp.reindex(idx.fbs_ids))
        else:
            pj = proj[proj["season"] == s].set_index("team_id")
            o = pj["p_off"].reindex(idx.fbs_ids).to_numpy()
            d = pj["p_def"].reindex(idx.fbs_ids).to_numpy()
            # Teams new to FBS start from last season's pooled value (as PR.projection).
            o = np.where(np.isnan(o), comp["pool_off"].iloc[0], o)
            d = np.where(np.isnan(d), comp["pool_def"].iloc[0], d)
        off0 = np.append(o, comp["pool_off"].iloc[0])
        def0 = np.append(d, comp["pool_def"].iloc[0])
        for cut, obs, keys in c["cuts"]:
            f = P.fit(obs, idx, int(cut.value), params, off0, def0, prev.mu, prev.hfa)
            ids = keys["team_id"].to_numpy(dtype=np.int64)
            pos = idx.index(ids)
            games = f.games[pos]
            own = np.isin(ids, idx.fbs_ids)
            if (~own).any() and len(obs):
                cnt = obs.groupby("offense_id").size()
                games = np.where(own, games, cnt.reindex(ids).fillna(0.0).to_numpy())
            off = f.mu + f.off[pos]
            dd = f.mu + f.deff[pos]
            frames.append(pd.DataFrame({
                "game_id": keys["game_id"].to_numpy(), "team_id": ids,
                "str_pwr_off": off, "str_pwr_def": dd, "str_pwr_net": off - dd,
                "str_pwr_prior_share": lam / (lam + games),
                "str_pwr_net_se": POINTS_SD * np.sqrt(2.0 / (lam + games)),
            }))
    return pd.concat(frames, ignore_index=True)


def swap(X: pd.DataFrame, gi: pd.DataFrame, pw: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """X with the h_/a_/d_ power columns replaced (rows of seasons in pw), and team-level
    `extra` columns (game_id, team_id, cols...) added as d_ differences."""
    X = X.copy()
    g = X[["game_id"]].merge(gi[["game_id", "home_id", "away_id"]], on="game_id", how="left")
    for frame, cols in ((pw, PWR_COLS), (extra, None)):
        if frame is None:
            continue
        cols = cols or [c for c in frame.columns if c not in ("game_id", "team_id")]
        f = frame.drop_duplicates(["game_id", "team_id"])
        h = g.merge(f, left_on=["game_id", "home_id"], right_on=["game_id", "team_id"], how="left")
        a = g.merge(f, left_on=["game_id", "away_id"], right_on=["game_id", "team_id"], how="left")
        have = h[cols[0]].notna().to_numpy() | (frame is extra)
        for col in cols:
            hv, av = h[col].to_numpy(), a[col].to_numpy()
            for pre, v in (("h", hv), ("a", av), ("d", hv - av)):
                name = f"{pre}_{col}"
                if name in X.columns:
                    X.loc[have, name] = v[have]
                else:
                    X[name] = v
    return X


def score(X, extra_cols=(), train_start=C.TRAIN_START):
    from cfb.model.models import MarginModel
    cols = C.model_columns() + list(extra_cols)
    return season_forward(X, lambda: MarginModel(cols, lam=C.LAM), train_start=train_start)


def buckets(X, p) -> pd.Series:
    d = X.loc[p.index]
    wk = np.where(d["season_type"] == "regular", d["week"], 99)
    b = pd.cut(wk, [0, 1, 2, 4, 8, 98, 100], labels=["w1", "w2", "w3-4", "w5-8", "w9+", "post"])
    y = d["y"].to_numpy()
    pp = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    s = pd.Series(ll, index=p.index).groupby(b, observed=True).mean()
    s["w1-4"] = ll[wk <= 4].mean()
    s["all"] = ll.mean()
    return s


def paired(X, p_a, p_b, mask=None):
    """Mean log-loss diff (a - b) and SE from a bootstrap over whole weeks."""
    d = X.loc[p_a.index]
    y = d["y"].to_numpy()
    def ll(p):
        p = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = ll(p_a) - ll(p_b.loc[p_a.index])
    key = d["season"].astype(str) + "_" + d["season_type"] + "_" + d["week"].astype(str)
    if mask is not None:
        m = mask.loc[p_a.index].to_numpy()
        diff, key = diff[m], key[m]
    g = pd.DataFrame({"k": key.to_numpy(), "d": diff}).groupby("k")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g))
        bs.append(g["sum"].to_numpy()[i].sum() / g["size"].to_numpy()[i].sum())
    return float(diff.mean()), float(np.std(bs))
