"""Direction and zone matchups from the detailed play text (groups `matchup_run_dir`,
`matchup_pass_zone`). Empty before 2025 week 9, when the detailed text starts.

For the team's offense against this opponent's defense, over visible plays of the cutoff season
and the one before it (garbage time left out), with CFBD success:

    share_o[z]   the offense's share of plays to zone z, shrunk toward the FBS share
    off_o[z]     the offense's success in z minus the FBS rate in z, shrunk (K plays)
    def_x[z]     the defense's success allowed in z minus the FBS rate in z, shrunk (K plays)

    *_fit  = sum_z (share_o[z] - share_fbs[z]) * def_x[z]
             positive when the offense goes where this defense gives up more
    *_exp  = sum_z share_o[z] * (off_o[z] + def_x[z])
             expected success over the FBS rate for these plays against this defense
    *_n    = log(1 + min(offense plays, defense plays))

Run zones: run direction (left, middle, right). Pass zones: air yards (behind the line, 1-9,
10-19, 20+) by direction, on passes with a named target (incompletions count as failures).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ..strength.common import team_keys
from ..strength.playtext import detailed_plays, flag

K_SHARE = 30.0   # plays: share shrinkage toward the FBS share
K_RATE = 40.0    # plays: success shrinkage toward the FBS rate
DEPTHS = ((-99, 0), (1, 9), (10, 19), (20, 999))
SOURCES = ("play_detail", "pbp")


def _zone_profiles(plays: pd.DataFrame, zone: pd.Series, k_share=K_SHARE, k_rate=K_RATE):
    """FBS shares and rates per zone; per offense share and success adjustments; per defense adjustments."""
    d = plays.assign(zone=zone.to_numpy(), succ=flag(plays["success"]).astype(float))
    d = d[d["zone"].notna().to_numpy()]
    zones = sorted(d["zone"].unique())
    fbs_share = d["zone"].value_counts(normalize=True).reindex(zones).fillna(0.0)
    fbs_rate = d.groupby("zone")["succ"].mean().reindex(zones)

    def side(col):
        cnt = d.groupby([col, "zone"]).size().unstack("zone").reindex(columns=zones).fillna(0.0)
        suc = d.groupby([col, "zone"])["succ"].sum().unstack("zone").reindex(columns=zones).fillna(0.0)
        tot = cnt.sum(axis=1)
        share = (cnt + k_share * fbs_share.to_numpy()).div(tot + k_share, axis=0)
        adj = (suc + k_rate * fbs_rate.to_numpy()) / (cnt + k_rate) - fbs_rate.to_numpy()
        return share, adj, tot

    off_share, off_adj, off_n = side("offense_id")
    _, def_adj, def_n = side("defense_id")
    return {"zones": zones, "fbs_share": fbs_share, "off_share": off_share, "off_adj": off_adj,
            "def_adj": def_adj, "off_n": off_n, "def_n": def_n}


def _matchup(keys: pd.DataFrame, prof: dict | None, prefix: str) -> pd.DataFrame:
    out = keys[["game_id", "team_id"]].reset_index(drop=True).copy()
    cols = [f"{prefix}_fit", f"{prefix}_exp", f"{prefix}_n"]
    if prof is None:
        for c in cols:
            out[c] = np.nan
        return out
    zones = prof["zones"]
    fs = prof["fbs_share"].to_numpy()
    zero = np.full(len(zones), np.nan)
    rows = []
    for t, o in zip(keys["team_id"].astype("int64"), keys["opp_id"].astype("int64")):
        sh = prof["off_share"].loc[t].to_numpy() if t in prof["off_share"].index else zero
        oa = prof["off_adj"].loc[t].to_numpy() if t in prof["off_adj"].index else zero
        da = prof["def_adj"].loc[o].to_numpy() if o in prof["def_adj"].index else zero
        n = min(prof["off_n"].get(t, 0), prof["def_n"].get(o, 0))
        rows.append(((sh - fs) @ da, sh @ (oa + da), np.log1p(n) if n else np.nan))
    arr = np.array(rows, dtype=float) if rows else np.empty((0, 3))
    for i, c in enumerate(cols):
        out[c] = arr[:, i] if len(arr) else np.nan
    return out


def _plays(view):
    p = detailed_plays(view, [view.season - 1, view.season])
    if p.empty:
        return p
    return p[~flag(p["garbage_time"])].reset_index(drop=True)


class RunDirection(FeatureGroup):
    name = "matchup_run_dir"
    prefix = "mat_"
    level = "team"
    columns = [
        FeatureColumn("mat_rdir_fit", "Run direction fit: the offense's run-direction tilt times this defense's "
                      "success allowed by direction (detailed play text, 2025 week 9 on)", SOURCES, 2025),
        FeatureColumn("mat_rdir_exp", "Expected run success over the FBS rate for this offense's direction mix "
                      "against this defense", SOURCES, 2025),
        FeatureColumn("mat_rdir_n", "log(1 + the smaller of the offense's and the defense's detailed runs)",
                      SOURCES, 2025),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        p = _plays(view)
        prof = None
        if not p.empty:
            r = p[(p["action"] == "rush").to_numpy()]
            if len(r) >= 200:
                prof = _zone_profiles(r, r["run_dir"].astype("object"))
        return _matchup(keys, prof, "mat_rdir")


def pass_zone(plays: pd.DataFrame) -> pd.Series:
    air = pd.to_numeric(plays["air_yards"], errors="coerce")
    depth = pd.Series(np.nan, index=plays.index, dtype="object")
    for i, (lo, hi) in enumerate(DEPTHS):
        depth[(air >= lo) & (air <= hi)] = str(i)
    z = depth + "_" + plays["pass_dir"].astype("object")
    return z.where(depth.notna() & plays["pass_dir"].notna())


class PassZone(FeatureGroup):
    name = "matchup_pass_zone"
    prefix = "mat_"
    level = "team"
    columns = [
        FeatureColumn("mat_pzone_fit", "Pass zone fit: the offense's target tilt by depth and direction times "
                      "this defense's success allowed there (detailed play text, 2025 week 9 on)", SOURCES, 2025),
        FeatureColumn("mat_pzone_exp", "Expected pass success over the FBS rate for this offense's target mix "
                      "against this defense", SOURCES, 2025),
        FeatureColumn("mat_pzone_n", "log(1 + the smaller of the offense's and the defense's detailed targets)",
                      SOURCES, 2025),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        p = _plays(view)
        prof = None
        if not p.empty:
            q = p[(p["action"] == "pass").to_numpy()]
            if len(q) >= 200:
                prof = _zone_profiles(q, pass_zone(q))
        return _matchup(keys, prof, "mat_pzone")
