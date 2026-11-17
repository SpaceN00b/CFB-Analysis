"""Player-based preseason components (P2): what last season's players are worth to this
season's roster.

For season S and each FBS team T (index team_id):
    pl_ret_ppa      offense: last season's total PPA (CFBD `ppa_players_season`, S-1) of the
                    players on T's S roster who played for T in S-1, per 100 PPA
    pl_in_ppa       the same for players on T's S roster who played for another team in S-1
                    (transfers keep their previous-school production)
    pl_lost_ppa     T's S-1 total PPA minus pl_ret_ppa (production that left)
    pl_ret_def_tkl  defense: share of T's S-1 tackles (box scores, `games_players`) made by
                    players on T's S roster who played for T
    pl_ret_def_hav  the same for havoc plays (TFL + passes defended + interceptions)
    pl_in_def_tkl   incoming players' shares of their previous team's tackles, summed
    pl_in_def_hav   the same for havoc plays
    pl_portal_in3, pl_portal_out3, pl_portal_in4, pl_portal_out4
                    transfers in and out rated 3 stars and up / 4 stars and up (portal 2021 on)

Players are matched by CFBD athlete id across roster, PPA and box scores. Values enter the
projection centred on the season's FBS mean (`centred`). Unknown inputs (no roster or PPA
table) stay NaN and count as average in the projection.

Point in time: the roster table is as of the pull date, so a past season's list is its final
roster (players who joined during the season are included; rarely players with S-1
production). PPA and box scores are of the finished season S-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import by_name

PLAYER_COLUMNS = ["pl_ret_ppa", "pl_in_ppa", "pl_lost_ppa", "pl_ret_def_tkl", "pl_ret_def_hav",
                  "pl_in_def_tkl", "pl_in_def_hav", "pl_portal_in3", "pl_portal_out3",
                  "pl_portal_in4", "pl_portal_out4"]

PORTAL_FIRST_SEASON = 2021

_DEF_STATS = {("defensive", "TOT"): "tkl", ("defensive", "TFL"): "hav", ("defensive", "PD"): "hav",
              ("interceptions", "INT"): "hav"}


def _ids(s: pd.Series) -> pd.Series:
    """Athlete ids as strings without a trailing '.0' (tables mix ints, floats and text)."""
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def _defense(view, season: int) -> pd.DataFrame:
    """athlete id, team, tkl, hav for season `season` (box scores)."""
    if not view.has("games_players"):
        return pd.DataFrame(columns=["id", "team", "tkl", "hav"])
    gp = view.table("games_players", columns=["team", "category", "statType", "athleteId", "stat"], seasons=season)
    if gp.empty:
        return pd.DataFrame(columns=["id", "team", "tkl", "hav"])
    key = pd.Series(list(zip(gp["category"], gp["statType"])), index=gp.index).map(_DEF_STATS)
    gp = gp[key.notna()].assign(k=key[key.notna()])
    gp = gp.assign(v=pd.to_numeric(gp["stat"], errors="coerce").fillna(0.0).astype("float64"),
                   id=_ids(gp["athleteId"]).astype(object))
    out = gp.pivot_table(index=["id", "team"], columns="k", values="v", aggfunc="sum", fill_value=0.0)
    return out.reindex(columns=["tkl", "hav"], fill_value=0.0).reset_index()


def components(view, season: int, ids: np.ndarray, asof=None) -> pd.DataFrame:
    """Raw (uncentred) player components per team id in `ids`; NaN where unknown.
    `asof` (a timestamp at or before the view's cutoff) limits the dated portal rows; batch
    groups use it to see a past season as it stood at that season's first cutoff."""
    out = pd.DataFrame(np.nan, index=pd.Index(ids, name="team_id"), columns=PLAYER_COLUMNS)
    names = view.table("teams", seasons=season)
    if view.has("roster"):
        ros = view.table("roster", columns=["id", "team"], seasons=season)
    else:
        ros = pd.DataFrame(columns=["id", "team"])
    if not ros.empty:
        ros = ros.assign(id=_ids(ros["id"])).drop_duplicates("id")
        # offense
        if view.has("ppa_players_season"):
            ppa = view.prior_seasons("ppa_players_season", columns=["id", "team", "totalPPA.all"], seasons=season - 1)
        else:
            ppa = pd.DataFrame()
        if not ppa.empty:
            ppa = pd.DataFrame({"id": _ids(ppa["id"]), "pteam": ppa["team"],
                                "ppa": pd.to_numeric(ppa["totalPPA.all"], errors="coerce").fillna(0.0)})
            ppa = ppa.drop_duplicates("id")
            m = ros.merge(ppa, on="id", how="inner")
            same = m["pteam"] == m["team"]
            prev_tot = ppa.groupby("pteam")["ppa"].sum()
            t = pd.DataFrame({
                "team": prev_tot.index.union(m["team"].unique()),
            }).set_index("team")
            t["pl_ret_ppa"] = m[same].groupby("team")["ppa"].sum()
            t["pl_in_ppa"] = m[~same].groupby("team")["ppa"].sum()
            t = t.fillna(0.0)
            t["pl_lost_ppa"] = prev_tot.reindex(t.index).fillna(0.0) - t["pl_ret_ppa"]
            t = t / 100.0
            f = by_name(t.reset_index().assign(season=season), names).drop_duplicates("team_id").set_index("team_id")
            for c in ("pl_ret_ppa", "pl_in_ppa", "pl_lost_ppa"):
                out[c] = f[c].reindex(ids).fillna(0.0).to_numpy()
        # defense
        d = _defense(view, season - 1)
        if not d.empty:
            tot = d.groupby("team")[["tkl", "hav"]].sum()
            for c in ("tkl", "hav"):
                d[f"{c}_sh"] = d[c] / d["team"].map(tot[c]).replace(0.0, np.nan)
            m = ros.merge(d.rename(columns={"team": "dteam"}), on="id", how="inner")
            same = m["dteam"] == m["team"]
            t = pd.DataFrame({
                "pl_ret_def_tkl": m[same].groupby("team")["tkl_sh"].sum(),
                "pl_ret_def_hav": m[same].groupby("team")["hav_sh"].sum(),
                "pl_in_def_tkl": m[~same].groupby("team")["tkl_sh"].sum(),
                "pl_in_def_hav": m[~same].groupby("team")["hav_sh"].sum(),
            })
            if len(t):
                f = by_name(t.rename_axis("team").reset_index().assign(season=season), names)
                f = f.drop_duplicates("team_id").set_index("team_id")
                for c in t.columns:
                    out[c] = f[c].reindex(ids).fillna(0.0).to_numpy()
    # portal quality counts (dated rows visible at the cutoff)
    tp = view.table("transfer_portal", seasons=season) if view.has("transfer_portal") else pd.DataFrame()
    if not tp.empty:
        if asof is not None:
            td = pd.to_datetime(tp["transferDate"], utc=True, errors="coerce")
            tp = tp[(td.isna() | (td <= asof)).to_numpy()]
        st = pd.to_numeric(tp["stars"], errors="coerce").fillna(2.0)
        tp = tp.assign(_st=st.to_numpy())
        for col, key in (("destination", "in"), ("origin", "out")):
            x = tp[tp[col].notna()][[col, "_st"]].rename(columns={col: "team"}).assign(season=season)
            x = by_name(x, names) if len(x) else x.assign(team_id=pd.Series(dtype="int64"))
            for lo in (3, 4):
                out[f"pl_portal_{key}{lo}"] = x[x["_st"] >= lo].groupby("team_id").size().reindex(ids) \
                    .fillna(0.0).to_numpy(dtype=np.float64)
    elif season < PORTAL_FIRST_SEASON:
        # Before the portal data starts no transfers are counted (0, not unknown).
        for c in ("pl_portal_in3", "pl_portal_out3", "pl_portal_in4", "pl_portal_out4"):
            out[c] = 0.0
    # A portal season without portal rows stays NaN (counts as average).
    return out


def centred(comp: pd.DataFrame) -> pd.DataFrame:
    """Player components centred on the mean of the FBS teams in `comp` (NaN stays NaN)."""
    return comp[PLAYER_COLUMNS] - comp[PLAYER_COLUMNS].mean()
