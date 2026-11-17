"""Feature groups for model_v2 (P2, early-season priors): the power rating with the
player-based preseason projection, its components, and a long-window program residual.

model_v1's groups (str_power, str_prior) are unchanged; these are separate groups so the
frozen model keeps its inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ...core.timeutil import RESULT_LAG
from . import players as PL
from . import points as P
from . import prior as PR
from .common import team_keys
from .power import POINTS_SD, POWER, power_state
from .units import TeamIndex

# Program residual: shrinkage in pseudo-games (chosen in the planning thread's B5 test on
# 2018-2025 and confirmed on the v2 ratings; scripts/prior_select.py).
PROGRAM_SHRINK = 8.0
PROGRAM_FIRST_SEASON = 2014


def _col(name, description, sources, first_season=2015):
    return FeatureColumn(name, description, tuple(sources), first_season)


def _out(keys: pd.DataFrame, values: dict) -> pd.DataFrame:
    out = keys[["game_id", "team_id"]].reset_index(drop=True).copy()
    for k, v in values.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


_PL_SOURCES = ["roster", "ppa_players_season", "games_players", "transfer_portal"]


class Power2(FeatureGroup):
    name = "str_power2"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_pwr2_off", "Power rating v2 offense: points per game against an average FBS defense, neutral "
             "field; prior = player-based preseason projection", ["team_game", "teams"] + _PL_SOURCES),
        _col("str_pwr2_def", "Power rating v2 defense: points per game allowed to an average FBS offense (lower "
             "is better)", ["team_game", "teams"] + _PL_SOURCES),
        _col("str_pwr2_net", "Power rating v2 net: str_pwr2_off minus str_pwr2_def", ["team_game", "teams"] + _PL_SOURCES),
        _col("str_pwr2_prior_share", "Share of the v2 power rating that is still preseason prior: lam / (lam + games)",
             ["team_game"]),
        _col("str_pwr2_net_se", "Approximate standard error of str_pwr2_net in points", ["team_game"]),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        p = power_state(view, version=2).lookup(keys["team_id"])
        g = p["games"].to_numpy()
        return _out(keys, {
            "str_pwr2_off": p["off"], "str_pwr2_def": p["def"], "str_pwr2_net": p["net"],
            "str_pwr2_prior_share": np.where(np.isfinite(p["prior_net"]), POWER.lam / (POWER.lam + g), 0.0),
            "str_pwr2_net_se": np.where(np.isfinite(p["net"]), POINTS_SD * np.sqrt(2.0 / (POWER.lam + g)), np.nan),
        })


_PRIOR2_DESCR = {
    "pl_ret_ppa": "Returning offense: last season's total PPA of players on this season's roster who played "
                  "for the team, per 100 PPA",
    "pl_in_ppa": "Incoming offense: last season's total PPA at other schools of players on this season's roster, "
                 "per 100 PPA",
    "pl_lost_ppa": "Departed offense: last season's team total PPA minus the returning part, per 100 PPA",
    "pl_ret_def_tkl": "Returning defense: share of last season's tackles made by players on this season's roster",
    "pl_ret_def_hav": "Returning defense: share of last season's havoc plays (TFL, passes defended, "
                      "interceptions) by players on this season's roster",
    "pl_in_def_tkl": "Incoming defense: incoming players' shares of their previous team's tackles, summed",
    "pl_in_def_hav": "Incoming defense: incoming players' shares of their previous team's havoc plays, summed",
    "pl_portal_in3": "Transfers in rated 3 stars or more (portal 2021 on; 0 before)",
    "pl_portal_out3": "Transfers out rated 3 stars or more (portal 2021 on; 0 before)",
    "pl_portal_in4": "Transfers in rated 4 stars or more (portal 2021 on; 0 before)",
    "pl_portal_out4": "Transfers out rated 4 stars or more (portal 2021 on; 0 before)",
}


class Prior2(FeatureGroup):
    name = "str_prior2"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_prior2_off", "Preseason projection v2 (player-based): offense, points per game against an "
             "average FBS team", ["team_game", "teams"] + _PL_SOURCES),
        _col("str_prior2_def", "Preseason projection v2: points allowed per game against an average FBS team",
             ["team_game", "teams"] + _PL_SOURCES),
        _col("str_prior2_net", "Preseason projection v2: net (str_prior2_off minus str_prior2_def)",
             ["team_game", "teams"] + _PL_SOURCES),
    ] + [_col(f"str_{c}", d, _PL_SOURCES) for c, d in _PRIOR2_DESCR.items()]

    def build(self, view, games):
        keys = team_keys(games)
        comp = PR.components_v2(view)
        off, deff = PR.projection(comp, version=2)
        prev, _ = PR.previous_final(view, view.season)
        mu = prev.mu if prev is not None else np.nan
        comp = comp.assign(p_off=mu + off, p_def=mu + deff)
        c = comp.reindex(keys["team_id"])
        vals = {"str_prior2_off": c["p_off"], "str_prior2_def": c["p_def"], "str_prior2_net": c["p_off"] - c["p_def"]}
        for col in PL.PLAYER_COLUMNS:
            vals[f"str_{col}"] = c[col]
        return _out(keys, vals)


class Program(FeatureGroup):
    """Long-window program residual: how much a team's results have beaten its v2 power
    rating, over every visible game since 2014."""
    name = "str_program"
    prefix = "str_"
    level = "team"
    mode = "batch"
    columns = [
        _col("str_prog_resid", "Program residual: shrunk mean (8 pseudo-games) over every visible game since 2014 "
             "of the team's margin minus the v2 power-rating expectation at that game's cutoff, in points",
             ["team_game", "teams"] + _PL_SOURCES, 2014),
        _col("str_prog_games", "Games behind str_prog_resid", ["team_game"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        res = residuals(view, int(games["season"].max()) if len(games) else view.season)
        vals = np.zeros(len(keys))
        cnt = np.zeros(len(keys))
        if len(res):
            by_team = {t: d for t, d in res.groupby("team_id")}
            cut = games.set_index("game_id")["cutoff_utc"]
            kc = cut.reindex(keys["game_id"]).to_numpy()
            avail = {t: (d["avail_ns"].to_numpy(), np.cumsum(d["resid"].to_numpy())) for t, d in by_team.items()}
            for i, (t, c) in enumerate(zip(keys["team_id"].to_numpy(), kc)):
                a = avail.get(int(t))
                if a is None:
                    continue
                n = int(np.searchsorted(a[0], pd.Timestamp(c).value, side="right"))
                if n:
                    vals[i] = a[1][n - 1] / (n + PROGRAM_SHRINK)
                    cnt[i] = n
        return _out(keys, {"str_prog_resid": vals, "str_prog_games": cnt})


def residuals(view, last_season: int) -> pd.DataFrame:
    """One row per team per completed game (seasons 2014..last_season, visible at the view):
    team_id, avail_ns (when the result became visible), resid = team's margin minus the v2
    power-rating expectation at the game's own cutoff. Sorted by avail_ns.

    Expectations use the v2 projection as it stood at each season's first cutoff and the
    results visible at each cutoff (start + RESULT_LAG <= cutoff)."""
    sched = view.schedule()
    lag = RESULT_LAG.value
    rows = []
    for s in range(PROGRAM_FIRST_SEASON, last_season + 1):
        ss = sched[sched["season"] == s]
        if ss.empty:
            continue
        cuts = np.sort(ss["cutoff_utc"].unique())
        first = pd.Timestamp(cuts[0])
        teams = view.table("teams", seasons=s)
        idx = TeamIndex.from_teams(teams)
        tg = view.table("team_game", columns=P.TG_COLUMNS, seasons=s)
        obs = P.observations(tg)
        if obs.empty:
            continue
        prev, _ = PR.previous_final(view, s)
        off0 = def0 = None
        mu0, hfa0 = None, 0.0
        if prev is not None:
            comp = PR.components_v2(view, s, asof=first)
            o, d = PR.projection(comp.reindex(idx.fbs_ids), version=2)
            pool_o, pool_d = comp["pool_off"].iloc[0], comp["pool_def"].iloc[0]
            off0, def0 = np.append(o, pool_o), np.append(d, pool_d)
            mu0, hfa0 = prev.mu, prev.hfa
        # completed games of the season with both sides' points
        g = tg[tg["is_home"].astype(bool) & tg["points"].notna()][["game_id", "team_id", "opp_id", "neutral",
                                                                      "start_time_utc", "points"]]
        opp = tg[["game_id", "team_id", "points"]].rename(columns={"team_id": "opp_id", "points": "opp_points"})
        g = g.merge(opp, on=["game_id", "opp_id"], how="inner").merge(
            ss[["game_id", "cutoff_utc"]], on="game_id", how="inner")
        if g.empty:
            continue
        start = obs["start_ns"].to_numpy()
        for c in cuts:
            gc = g[g["cutoff_utc"] == c]
            if gc.empty:
                continue
            cn = pd.Timestamp(c).value
            f = P.fit(obs[start + lag <= cn], idx, cn, POWER, off0, def0, mu0, hfa0)
            ph = idx.index(gc["team_id"].to_numpy())
            pa = idx.index(gc["opp_id"].to_numpy())
            site = 1.0 - gc["neutral"].fillna(False).astype(bool).to_numpy().astype(float)
            exp = (f.off[ph] - f.deff[ph]) - (f.off[pa] - f.deff[pa]) + 2.0 * f.hfa * site
            r = (gc["points"] - gc["opp_points"]).to_numpy(dtype=float) - exp
            av = pd.to_datetime(gc["start_time_utc"], utc=True).astype("int64").to_numpy() + lag
            rows.append(pd.DataFrame({"team_id": gc["team_id"].to_numpy(dtype=np.int64), "avail_ns": av, "resid": r}))
            rows.append(pd.DataFrame({"team_id": gc["opp_id"].to_numpy(dtype=np.int64), "avail_ns": av, "resid": -r}))
    if not rows:
        return pd.DataFrame(columns=["team_id", "avail_ns", "resid"])
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out["resid"])]
    return out.sort_values("avail_ns", kind="stable").reset_index(drop=True)


GROUPS_V2 = [Power2(), Prior2(), Program()]
