"""Preseason prior: what is known about a team before its season starts.

Components (per FBS team of the cutoff season):
    prev_off, prev_def  last season's final points ratings (in-house ridge, no prior)
    sp_off_c, sp_def_c  last season's final SP+ offense and defense, centred on the season mean
    tal_z               247 talent composite, z-score within the season's FBS teams
    rec_z               mean recruiting points of the last four classes, z-score
    ret                 returning production (share of last season's PPA), centred

The projected offense and defense (points per game against an average FBS team) are
linear in the components. Coefficients were fitted by `scripts/strength_fit_prior.py`
on seasons 2015 to 2025: the target is the season's final points rating. Missing
components count as average (0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import players as PL
from . import points as P
from .common import by_name, zscore
from .units import TeamIndex

COMPONENTS = ["prev_off", "prev_def", "sp_off_c", "sp_def_c", "tal_z", "rec_z", "ret"]

# Fitted by scripts/strength_fit_prior.py on 2015-2025 (never 2026). Intercept first.
COEF_OFF = {"one": 0.074, "prev_off": 0.369, "prev_def": -0.015, "sp_off_c": 0.053, "sp_def_c": -0.038,
            "tal_z": 0.709, "rec_z": 1.220, "ret": 4.554}
COEF_DEF = {"one": -0.125, "prev_off": 0.084, "prev_def": 0.309, "sp_off_c": -0.163, "sp_def_c": 0.165,
            "tal_z": 1.260, "rec_z": -2.586, "ret": -3.074}


# Version 2 (model_v2, P2): the components above plus the player components of
# `players.py`, centred within the season. Fitted by scripts/strength_fit_prior.py --version 2
# on 2015-2025 (never 2026). Leave-one-season-out R^2: offense 0.457, defense 0.503
# (version 1: 0.436, 0.472).
COMPONENTS_V2 = COMPONENTS + [f"{c}_c" for c in PL.PLAYER_COLUMNS]
COEF_OFF_V2 = {
    "one": 0.0594, "prev_off": 0.2862, "prev_def": -0.0061, "sp_off_c": 0.015, "sp_def_c": -0.0782,
    "tal_z": 0.1748, "rec_z": 1.4219, "ret": 0.7565, "pl_ret_ppa_c": 1.4239, "pl_in_ppa_c": 1.0545,
    "pl_lost_ppa_c": -0.1293, "pl_ret_def_tkl_c": 1.7479, "pl_ret_def_hav_c": 0.8959,
    "pl_in_def_tkl_c": 0.7082, "pl_in_def_hav_c": -0.0313, "pl_portal_in3_c": 0.0371,
    "pl_portal_out3_c": 0.0441, "pl_portal_in4_c": 0.0588, "pl_portal_out4_c": 0.0924,
}
COEF_DEF_V2 = {
    "one": -0.108, "prev_off": 0.0717, "prev_def": 0.3011, "sp_off_c": -0.2075, "sp_def_c": 0.2086,
    "tal_z": 1.5105, "rec_z": -2.2691, "ret": -1.055, "pl_ret_ppa_c": 0.1257,
    "pl_in_ppa_c": -0.2973, "pl_lost_ppa_c": 0.588, "pl_ret_def_tkl_c": -4.789,
    "pl_ret_def_hav_c": -4.2754, "pl_in_def_tkl_c": -5.9527, "pl_in_def_hav_c": -0.202,
    "pl_portal_in3_c": 0.0235, "pl_portal_out3_c": -0.0105, "pl_portal_in4_c": -0.1783,
    "pl_portal_out4_c": 0.0096,
}


def _table(view, name, seasons, prior=False):
    if not view.has(name):
        return None
    if prior:
        return view.prior_seasons(name, seasons=seasons)
    return view.table(name, seasons=seasons)


def previous_final(view, season: int):
    """(PointsFit, TeamIndex) of last season's final points ratings, or (None, None)."""
    tg = view.table("team_game", columns=P.TG_COLUMNS, seasons=season - 1)
    if tg.empty:
        return None, None
    teams = view.table("teams", seasons=season - 1)
    idx = TeamIndex.from_teams(teams)
    return P.final_fit(P.observations(tg), idx, P.DEFAULT), idx


def components(view, season: int | None = None) -> pd.DataFrame:
    """One row per FBS team of the season (index team_id): the components above plus raw
    inputs kept as features. NaN where unknown."""
    season = view.season if season is None else season
    teams = view.table("teams", seasons=[season - 1, season])
    cur = teams[(teams["season"] == season) & teams["fbs"].fillna(False).astype(bool)]
    ids = np.sort(cur["team_id"].drop_duplicates().to_numpy(dtype=np.int64))
    out = pd.DataFrame(index=pd.Index(ids, name="team_id"))
    names_cur = teams[teams["season"] == season]
    names_prev = teams[teams["season"] == season - 1]

    prev, idx_prev = previous_final(view, season)
    if prev is not None:
        pos = idx_prev.index(ids)
        known = np.isin(ids, idx_prev.fbs_ids)
        out["prev_off"] = np.where(known, prev.off[pos], np.nan)
        out["prev_def"] = np.where(known, prev.deff[pos], np.nan)
        out["pool_off"], out["pool_def"] = prev.off[-1], prev.deff[-1]
    else:
        out["prev_off"] = out["prev_def"] = out["pool_off"] = out["pool_def"] = np.nan

    def attach(frame, cols, names, rename=None):
        if frame is None or frame.empty:
            for c in (rename or {}).values() if rename else cols:
                out[c] = np.nan
            return
        f = by_name(frame[["season", "team"] + cols], names).drop_duplicates("team_id").set_index("team_id")
        f = f.rename(columns=rename or {})
        for c in (rename or {c: c for c in cols}).values():
            out[c] = f[c].reindex(ids).to_numpy(dtype=np.float64)

    attach(_table(view, "ratings_sp", season - 1, prior=True), ["rating", "offense.rating", "defense.rating"],
           names_prev, {"rating": "sp_prev", "offense.rating": "sp_off_prev", "defense.rating": "sp_def_prev"})
    attach(_table(view, "ratings_fpi", season - 1, prior=True), ["fpi"], names_prev, {"fpi": "fpi_prev"})
    attach(_table(view, "ratings_srs", season - 1, prior=True), ["rating"], names_prev, {"rating": "srs_prev"})
    attach(_table(view, "talent", season), ["talent"], names_cur, {"talent": "talent"})
    attach(_table(view, "returning_production", season), ["percentPPA", "usage"], names_cur,
           {"percentPPA": "ret_ppa", "usage": "ret_usage"})
    attach(_table(view, "ratings_elo_preseason", season), ["elo"], names_cur, {"elo": "elo_pre"})
    rec = _table(view, "recruiting_teams", list(range(season - 3, season + 1)))
    if rec is not None and not rec.empty:
        r = rec.groupby("team", as_index=False)["points"].mean().assign(season=season)
        attach(r, ["points"], names_cur, {"points": "recruit4"})
    else:
        out["recruit4"] = np.nan

    out["sp_off_c"] = out["sp_off_prev"] - out["sp_off_prev"].mean()
    out["sp_def_c"] = out["sp_def_prev"] - out["sp_def_prev"].mean()
    out["tal_z"] = zscore(out["talent"])
    out["rec_z"] = zscore(out["recruit4"])
    out["ret"] = out["ret_ppa"] - out["ret_ppa"].mean()
    return out


def components_v2(view, season: int | None = None, asof=None) -> pd.DataFrame:
    """`components` plus the player components (raw and centred `<name>_c`). `asof`: see
    players.components."""
    season = view.season if season is None else season
    comp = components(view, season)
    pl = PL.components(view, season, comp.index.to_numpy(), asof=asof)
    comp = comp.join(pl)
    cen = PL.centred(pl)
    for c in PL.PLAYER_COLUMNS:
        comp[f"{c}_c"] = cen[c]
    return comp


def portal(view, season: int, ids: np.ndarray) -> pd.DataFrame:
    """Transfers of the season visible at the cutoff (2021 on): counts and star-weighted net.
    Unrated transfers count as two stars. Empty before the data starts."""
    out = pd.DataFrame(index=pd.Index(ids, name="team_id"), columns=["portal_in", "portal_out", "portal_net"],
                       dtype="float64")
    tp = _table(view, "transfer_portal", season)
    if tp is None or tp.empty:
        return out
    names = view.table("teams", seasons=season)
    stars = pd.to_numeric(tp["stars"], errors="coerce").fillna(2.0)
    tp = tp.assign(_stars=stars.to_numpy())
    res = {}
    for col, key in (("origin", "out"), ("destination", "in")):
        x = tp[tp[col].notna()][[col, "_stars"]].rename(columns={col: "team"}).assign(season=season)
        x = by_name(x, names)
        res[key] = x.groupby("team_id")["_stars"].agg(["size", "sum"]).reindex(ids).fillna(0.0)
    out["portal_in"] = res["in"]["size"].to_numpy()
    out["portal_out"] = res["out"]["size"].to_numpy()
    out["portal_net"] = (res["in"]["sum"] - res["out"]["sum"]).to_numpy()
    return out


def projection(comp: pd.DataFrame, version: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Projected (off, def) deviations in points from the components; NaN components count as 0.
    A team with no previous FBS season starts from last season's pooled lower-division value.
    `version` 2 uses COMPONENTS_V2 (needs `components_v2`)."""
    names, co, cd = (COMPONENTS, COEF_OFF, COEF_DEF) if version == 1 else (COMPONENTS_V2, COEF_OFF_V2, COEF_DEF_V2)
    x = comp.reindex(columns=names).fillna(0.0)
    off = co["one"] + sum(co[c] * x[c] for c in names)
    deff = cd["one"] + sum(cd[c] * x[c] for c in names)
    new = comp["prev_off"].isna() & comp["pool_off"].notna()
    off = off.where(~new, comp["pool_off"])
    deff = deff.where(~new, comp["pool_def"])
    if comp["prev_off"].isna().all() and comp["pool_off"].isna().all():
        # No earlier season in the data: no prior.
        return np.full(len(comp), np.nan), np.full(len(comp), np.nan)
    return off.to_numpy(dtype=np.float64), deff.to_numpy(dtype=np.float64)
