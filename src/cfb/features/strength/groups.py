"""The str_ feature groups. All are level "team", mode "per_cutoff"."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from . import elo as E
from . import prior as PR
from .common import team_keys
from .power import WIN_SCALE, POINTS_SD, POWER, power_state, win_prob


def _col(name, description, sources, first_season=2014):
    return FeatureColumn(name, description, tuple(sources), first_season)


def _team_game(view, cols):
    """Season-to-date team_game rows with `cols`; columns the store lacks come back empty."""
    tg = view.table("team_game", seasons=view.season)
    return tg.reindex(columns=cols)


def _out(keys: pd.DataFrame, values: dict) -> pd.DataFrame:
    out = keys[["game_id", "team_id"]].reset_index(drop=True).copy()
    for k, v in values.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


# ---- power ---------------------------------------------------------------------------

class Power(FeatureGroup):
    name = "str_power"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_pwr_off", "Power rating offense: points per game against an average FBS defense on a neutral "
             "field (opponent-adjusted ridge on points, preseason projection as prior)", ["team_game", "teams"], 2015),
        _col("str_pwr_def", "Power rating defense: points per game allowed to an average FBS offense on a neutral "
             "field (lower is better)", ["team_game", "teams"], 2015),
        _col("str_pwr_net", "Power rating net: str_pwr_off minus str_pwr_def (points margin against an average "
             "FBS team, neutral field)", ["team_game", "teams"], 2015),
        _col("str_pwr_games", "Completed games this season behind the power rating", ["team_game"], 2014),
        _col("str_pwr_prior_share", "Share of the power rating that is still preseason prior: lam / (lam + games)",
             ["team_game"], 2015),
        _col("str_pwr_net_se", "Approximate standard error of str_pwr_net in points", ["team_game"], 2015),
        _col("str_unit_off_epa", "Opponent-adjusted offensive EPA per play, all non-garbage scrimmage plays "
             "(unit_ratings off_all)", ["unit_ratings"], 2015),
        _col("str_unit_def_epa", "Opponent-adjusted EPA per play allowed (unit_ratings def_all; lower is better)",
             ["unit_ratings"], 2015),
        _col("str_unit_off_success", "Opponent-adjusted offensive success rate (off_all)", ["unit_ratings"], 2015),
        _col("str_unit_def_success", "Opponent-adjusted success rate allowed (def_all)", ["unit_ratings"], 2015),
        _col("str_unit_off_expl", "Opponent-adjusted EPA per successful play, offense (off_all)", ["unit_ratings"], 2015),
        _col("str_unit_def_expl", "Opponent-adjusted EPA per successful play allowed (def_all)", ["unit_ratings"], 2015),
        _col("str_unit_net_epa", "str_unit_off_epa minus str_unit_def_epa", ["unit_ratings"], 2015),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        st = power_state(view)
        p = st.lookup(keys["team_id"])
        se = POINTS_SD * np.sqrt(2.0 / (POWER.lam + p["games"].to_numpy()))
        ur = view.latest("unit_ratings", ["team_id", "unit"])
        vals = {}
        for side in ("off", "def"):
            u = ur[ur["unit"] == f"{side}_all"].set_index("team_id")
            for metric, col in (("epa", "epa_adj"), ("success", "success_adj"), ("expl", "explosive_adj")):
                vals[f"str_unit_{side}_{metric}"] = u[col].reindex(keys["team_id"]).to_numpy(dtype=np.float64) \
                    if len(u) else np.full(len(keys), np.nan)
        return _out(keys, {
            "str_pwr_off": p["off"], "str_pwr_def": p["def"], "str_pwr_net": p["net"],
            "str_pwr_games": p["games"],
            "str_pwr_prior_share": np.where(np.isfinite(p["prior_net"]), POWER.lam / (POWER.lam + p["games"]), 0.0),
            "str_pwr_net_se": np.where(np.isfinite(p["net"]), se, np.nan),
            **vals,
            "str_unit_net_epa": vals["str_unit_off_epa"] - vals["str_unit_def_epa"],
        })


# ---- preseason prior -------------------------------------------------------------------

class Prior(FeatureGroup):
    name = "str_prior"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_prior_off", "Preseason projected offense, points per game against an average FBS team",
             ["team_game", "ratings_sp", "talent", "recruiting_teams", "returning_production"], 2015),
        _col("str_prior_def", "Preseason projected points allowed per game against an average FBS team",
             ["team_game", "ratings_sp", "talent", "recruiting_teams", "returning_production"], 2015),
        _col("str_prior_net", "Preseason projected net rating (str_prior_off minus str_prior_def)",
             ["team_game", "ratings_sp", "talent", "recruiting_teams", "returning_production"], 2015),
        _col("str_prior_prev_net", "Last season's final in-house points rating, net", ["team_game"], 2015),
        _col("str_prior_sp_prev", "Last season's final SP+ rating", ["ratings_sp"], 2015),
        _col("str_prior_fpi_prev", "Last season's final FPI", ["ratings_fpi"], 2015),
        _col("str_prior_srs_prev", "Last season's final SRS", ["ratings_srs"], 2015),
        _col("str_prior_elo_pre", "CFBD preseason Elo (missing for many teams before 2022)",
             ["ratings_elo_preseason"], 2014),
        _col("str_prior_talent", "247 team talent composite for the season", ["talent"], 2015),
        _col("str_prior_ret_ppa", "Returning production: share of last season's PPA", ["returning_production"], 2014),
        _col("str_prior_ret_usage", "Returning production: share of last season's usage", ["returning_production"], 2014),
        _col("str_prior_recruit4", "Mean recruiting points of the last four classes", ["recruiting_teams"], 2014),
        _col("str_prior_portal_in", "Transfers into the team this season visible at the cutoff (2021 on)",
             ["transfer_portal"], 2021),
        _col("str_prior_portal_out", "Transfers out of the team this season visible at the cutoff (2021 on)",
             ["transfer_portal"], 2021),
        _col("str_prior_portal_net", "Star-weighted transfers in minus out (unrated count as 2 stars; 2021 on)",
             ["transfer_portal"], 2021),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        comp = PR.components(view)
        off, deff = PR.projection(comp)
        st_mu = None
        prev, _ = PR.previous_final(view, view.season)
        if prev is not None:
            st_mu = prev.mu
        comp = comp.assign(p_off=(st_mu if st_mu is not None else np.nan) + off,
                           p_def=(st_mu if st_mu is not None else np.nan) + deff)
        port = PR.portal(view, view.season, comp.index.to_numpy())
        c = comp.reindex(keys["team_id"])
        pt = port.reindex(keys["team_id"])
        return _out(keys, {
            "str_prior_off": c["p_off"], "str_prior_def": c["p_def"], "str_prior_net": c["p_off"] - c["p_def"],
            "str_prior_prev_net": c["prev_off"] - c["prev_def"], "str_prior_sp_prev": c["sp_prev"],
            "str_prior_fpi_prev": c["fpi_prev"], "str_prior_srs_prev": c["srs_prev"],
            "str_prior_elo_pre": c["elo_pre"], "str_prior_talent": c["talent"], "str_prior_ret_ppa": c["ret_ppa"],
            "str_prior_ret_usage": c["ret_usage"], "str_prior_recruit4": c["recruit4"],
            "str_prior_portal_in": pt["portal_in"], "str_prior_portal_out": pt["portal_out"],
            "str_prior_portal_net": pt["portal_net"],
        })


# ---- Elo -------------------------------------------------------------------------------

class Elo(FeatureGroup):
    name = "str_elo"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_elo", "In-house Elo before the game (margin-of-victory multiplier, season regression)",
             ["game_index"], 2014),
        _col("str_elo_win_prob", "In-house Elo win probability against this opponent, with home field",
             ["game_index"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        teams = view.table("teams", seasons=view.season)
        r = E.ratings_at(view, view.season, teams)
        own = r.reindex(keys["team_id"]).to_numpy()
        opp = r.reindex(keys["opp_id"]).to_numpy()
        h = np.where(keys["neutral"], 0.0, np.where(keys["is_home"], 1.0, -1.0))
        p = np.array([E.expected(x) for x in own + h * E.DEFAULT.hfa - opp])
        return _out(keys, {"str_elo": own, "str_elo_win_prob": p})


# ---- schedule strength and record ------------------------------------------------------

class Schedule(FeatureGroup):
    name = "str_schedule"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_sos_played", "Mean power rating (net) of opponents already played this season, at today's "
             "ratings", ["team_game", "schedule"], 2015),
        _col("str_sos_remaining", "Mean power rating (net) of this season's regular-season opponents from this "
             "game on (includes this game)", ["schedule"], 2015),
        _col("str_sor", "Strength of record: wins minus the wins a 25th-best FBS team would expect against the "
             "same schedule and sites", ["team_game", "schedule"], 2015),
        _col("str_sor_per_game", "str_sor divided by games played", ["team_game", "schedule"], 2015),
        _col("str_wins", "Wins this season before the game", ["team_game"], 2014),
        _col("str_losses", "Losses this season before the game", ["team_game"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        st = power_state(view)
        played = view.table("team_game", columns=["game_id", "team_id", "opp_id", "is_home", "neutral", "win",
                                                    "points", "start_time_utc"], seasons=view.season)
        teams = keys["team_id"].unique()
        played = played[played["team_id"].isin(teams) & played["points"].notna()]
        net = lambda ids: st.lookup(ids)["net"].to_numpy()
        hfa2 = 2.0 * st.fit.hfa if np.isfinite(st.fit.hfa) else 0.0
        fbs_net = np.sort(st.fit.off[:-1] - st.fit.deff[:-1])[::-1]
        ref = fbs_net[24] if len(fbs_net) > 24 else (fbs_net[-1] if len(fbs_net) else np.nan)
        res = pd.DataFrame(index=pd.Index(teams, name="team_id"))
        if len(played):
            opp_net = net(played["opp_id"])
            h = np.where(played["neutral"].fillna(False), 0.0, np.where(played["is_home"], 1.0, -1.0))
            exp_ref = win_prob(ref - opp_net + h * hfa2)
            p = played.assign(opp_net=opp_net, exp_ref=exp_ref, w=played["win"].astype("float64"))
            agg = p.groupby("team_id").agg(sos=("opp_net", "mean"), exp=("exp_ref", "sum"), wins=("w", "sum"),
                                           n=("w", "size"))
            res = res.join(agg)
        else:
            res = res.assign(sos=np.nan, exp=np.nan, wins=np.nan, n=np.nan)
        sched = view.schedule(future=True)
        sched = sched[(sched["season"] == view.season) & (sched["season_type"] == "regular")
                      & (sched["start_time_utc"] >= view.cutoff)]
        fut = team_keys(sched)
        fut = fut[fut["team_id"].isin(teams)]
        if len(fut):
            fut = fut.assign(opp_net=net(fut["opp_id"]))
            res = res.join(fut.groupby("team_id")["opp_net"].mean().rename("sos_rem"))
        else:
            res["sos_rem"] = np.nan
        r = res.reindex(keys["team_id"])
        n = r["n"].fillna(0.0).to_numpy()
        wins = r["wins"].fillna(0.0).to_numpy()
        sor = np.where(n > 0, wins - r["exp"].fillna(0.0).to_numpy(), 0.0)
        return _out(keys, {
            "str_sos_played": r["sos"], "str_sos_remaining": r["sos_rem"],
            "str_sor": np.where(np.isfinite(ref), sor, np.nan),
            "str_sor_per_game": np.where(n > 0, sor / np.maximum(n, 1), np.nan),
            "str_wins": wins, "str_losses": n - wins,
        })


# ---- luck ------------------------------------------------------------------------------

# Share of passes defensed (INT + PBU) that are intercepted, FBS vs FBS 2016-2025.
INT_SHARE = 0.196
FUMBLE_PLAYS = ("Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown")


class Luck(FeatureGroup):
    name = "str_luck"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_luck_fumble_rec", "Share of fumble-recovery plays in the team's games this season won by the "
             "team, minus 0.5 (play-by-play)", ["pbp"], 2014),
        _col("str_luck_int", "Interceptions above expectation per game: (defensive INTs minus 19.6% of passes "
             "defensed) minus the same for the opponents (needs passes defensed, from 2016)", ["team_game"], 2016),
        _col("str_luck_turnover_margin", "Turnover margin per game this season", ["team_game"], 2014),
        _col("str_luck_win_prob", "Wins minus the sum of CFBD postgame win probabilities this season",
             ["team_game"], 2014),
        _col("str_luck_close_record", "Wins minus losses in one-score games (margin 8 or less) this season",
             ["team_game"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        cols = ["game_id", "team_id", "opp_id", "win", "margin", "postgame_win_prob", "box_interceptions",
                "box_passes_deflected", "box_turnovers"]
        tg = _team_game(view, cols + ["points"])
        tg = tg[tg["points"].notna()].drop(columns="points")
        if tg.empty:
            return _out(keys, {c.name: np.nan for c in self.columns})
        opp = tg[["game_id", "team_id", "box_interceptions", "box_passes_deflected", "box_turnovers"]].rename(
            columns={"team_id": "opp_id", "box_interceptions": "o_ints", "box_passes_deflected": "o_pd",
                     "box_turnovers": "o_to"})
        t = tg.merge(opp, on=["game_id", "opp_id"], how="left")
        # Defensive INTs are the opponent's interceptions thrown.
        int_def = t["o_ints"] - INT_SHARE * (t["o_ints"] + t["box_passes_deflected"])
        int_off = t["box_interceptions"] - INT_SHARE * (t["box_interceptions"] + t["o_pd"])
        t = t.assign(int_luck=int_def - int_off, to_margin=t["o_to"] - t["box_turnovers"],
                     wp_luck=t["win"].astype("float64") - t["postgame_win_prob"],
                     close=np.where(t["margin"].abs() <= 8, np.sign(t["margin"]), 0.0))
        g = t.groupby("team_id").agg(int_luck=("int_luck", "mean"), to=("to_margin", "mean"), wp=("wp_luck", "sum"),
                                     close=("close", "sum"))
        pbp = view.table("pbp", columns=["game_id", "offense_id", "defense_id", "play_type"], seasons=view.season)
        f = pbp[pbp["play_type"].isin(FUMBLE_PLAYS) & pbp["game_id"].isin(tg["game_id"])]
        own = (f["play_type"] == "Fumble Recovery (Own)").to_numpy()
        rec = pd.concat([pd.DataFrame({"team_id": f["offense_id"].to_numpy(), "won": own.astype(float)}),
                         pd.DataFrame({"team_id": f["defense_id"].to_numpy(), "won": (~own).astype(float)})])
        fr = rec.groupby("team_id")["won"].agg(["sum", "size"])
        r = g.reindex(keys["team_id"])
        fr = fr.reindex(keys["team_id"])
        played = keys["team_id"].isin(g.index).to_numpy()
        share = np.where(fr["size"].fillna(0) > 0, fr["sum"] / fr["size"].where(fr["size"] > 0) - 0.5, 0.0)
        return _out(keys, {
            "str_luck_fumble_rec": np.where(played, share, np.nan),
            "str_luck_int": r["int_luck"], "str_luck_turnover_margin": r["to"],
            "str_luck_win_prob": r["wp"], "str_luck_close_record": r["close"],
        })


# ---- special teams ---------------------------------------------------------------------

FG_PRIOR = 6.0      # pseudo-attempts at the league rate, by distance
RET_PRIOR = 10.0    # pseudo-returns at the league average
PUNT_PRIOR = 15.0   # pseudo-punts at the league average


def _fg_expected(dist: np.ndarray) -> np.ndarray:
    """League make probability by kick distance (logistic fitted on 2014-2025 FBS kicks)."""
    return 1.0 / (1.0 + np.exp(-(5.9 - 0.105 * dist)))


class SpecialTeams(FeatureGroup):
    name = "str_special_teams"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_st_fg_poe", "Field goal points over expected per game this season (by kick distance, shrunk "
             "toward 0)", ["pbp"], 2014),
        _col("str_st_kr_avg", "Kick return yards per return this season, shrunk toward the league average",
             ["team_game"], 2014),
        _col("str_st_pr_avg", "Punt return yards per return this season, shrunk toward the league average",
             ["team_game"], 2014),
        _col("str_st_kr_allowed", "Opponents' kick return yards per return, shrunk", ["team_game"], 2014),
        _col("str_st_pr_allowed", "Opponents' punt return yards per return, shrunk", ["team_game"], 2014),
        _col("str_st_net_punt", "Net yards per punt this season, shrunk toward the league average",
             ["punting_games"], 2015),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        ids = keys["team_id"]
        vals = {}
        pbp = view.table("pbp", columns=["offense_id", "play_type", "yards_to_goal"], seasons=view.season)
        fg = pbp[pbp["play_type"].isin(["Field Goal Good", "Field Goal Missed", "Blocked Field Goal"])]
        tg = _team_game(view, ["game_id", "team_id", "opp_id", "box_kick_returns", "box_kick_return_yards",
                               "box_punt_returns", "box_punt_return_yards", "points"])
        tg = tg[tg["points"].notna()]
        n_games = tg.groupby("team_id").size()
        if len(fg):
            dist = fg["yards_to_goal"].to_numpy(dtype=np.float64) + 17.0
            exp = _fg_expected(dist)
            made = (fg["play_type"] == "Field Goal Good").to_numpy(dtype=np.float64)
            f = pd.DataFrame({"team_id": fg["offense_id"].to_numpy(), "poe": 3.0 * (made - exp), "n": 1.0})
            g = f.groupby("team_id").sum()
            shrunk = g["poe"] * g["n"] / (g["n"] + FG_PRIOR)
            per_game = (shrunk / n_games.reindex(g.index)).reindex(ids)
            vals["str_st_fg_poe"] = per_game.fillna(0.0).where(ids.isin(n_games.index).to_numpy()).to_numpy()
        else:
            vals["str_st_fg_poe"] = np.full(len(keys), np.nan)

        def shrink(num, den, prior_n):
            league = num.sum() / den.sum() if den.sum() > 0 else np.nan
            return (num + prior_n * league) / (den + prior_n)

        if len(tg):
            opp = tg.rename(columns={"team_id": "opp_id", "opp_id": "team_id"})
            s = tg.groupby("team_id")[["box_kick_returns", "box_kick_return_yards", "box_punt_returns",
                                       "box_punt_return_yards"]].sum(min_count=1)
            a = opp.groupby("team_id")[["box_kick_returns", "box_kick_return_yards", "box_punt_returns",
                                        "box_punt_return_yards"]].sum(min_count=1)
            for name, frame in (("", s), ("_allowed", a)):
                kr = shrink(frame["box_kick_return_yards"].fillna(0), frame["box_kick_returns"].fillna(0), RET_PRIOR)
                pr = shrink(frame["box_punt_return_yards"].fillna(0), frame["box_punt_returns"].fillna(0), RET_PRIOR)
                vals[f"str_st_kr{'_allowed' if name else '_avg'}"] = kr.reindex(ids).to_numpy()
                vals[f"str_st_pr{'_allowed' if name else '_avg'}"] = pr.reindex(ids).to_numpy()
        else:
            for c in ("str_st_kr_avg", "str_st_pr_avg", "str_st_kr_allowed", "str_st_pr_allowed"):
                vals[c] = np.full(len(keys), np.nan)

        vals["str_st_net_punt"] = np.full(len(keys), np.nan)
        if view.has("punting_games"):
            pg = view.table("punting_games", seasons=view.season)
            if len(pg):
                names = view.table("teams", seasons=view.season)[["team", "team_id"]]
                pg = pg.merge(names, on="team", how="inner")
                punts = pd.to_numeric(pg["punting.punts"], errors="coerce").fillna(0)
                net = pd.to_numeric(pg["punting.netPuntYards"], errors="coerce").fillna(0)
                g = pd.DataFrame({"team_id": pg["team_id"], "p": punts, "y": net}).groupby("team_id").sum()
                vals["str_st_net_punt"] = shrink(g["y"], g["p"], PUNT_PRIOR).reindex(ids).to_numpy()
        return _out(keys, vals)


# ---- recent form ------------------------------------------------------------------------

class Form(FeatureGroup):
    name = "str_form"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_form_pts3", "Mean points margin above the power-rating expectation over the last 3 games, "
             "minus the season mean of the same (today's ratings)", ["team_game"], 2014),
        _col("str_form_epa3", "Mean opponent-adjusted net EPA per play over the last 3 games minus the season "
             "mean (today's unit ratings)", ["team_game", "unit_ratings"], 2014),
        _col("str_form_games", "Games in the season mean behind the form columns", ["team_game"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        st = power_state(view)
        tg = view.table("team_game", columns=["game_id", "team_id", "opp_id", "is_home", "neutral", "margin",
                                              "start_time_utc", "pbp_off_all_plays", "pbp_off_all_epa_sum",
                                              "pbp_def_all_plays", "pbp_def_all_epa_sum"], seasons=view.season)
        tg = tg[tg["team_id"].isin(keys["team_id"].unique()) & tg["margin"].notna()]
        if tg.empty:
            return _out(keys, {c.name: np.nan for c in self.columns})
        own, opp = st.lookup(tg["team_id"]), st.lookup(tg["opp_id"])
        h = np.where(tg["neutral"].fillna(False), 0.0, np.where(tg["is_home"], 1.0, -1.0))
        hfa2 = 2.0 * st.fit.hfa if np.isfinite(st.fit.hfa) else 0.0
        resid = tg["margin"].to_numpy() - (own["net"].to_numpy() - opp["net"].to_numpy() + h * hfa2)
        ur = view.latest("unit_ratings", ["team_id", "unit"])
        u = ur.pivot_table(index="team_id", columns="unit", values="epa_adj") if len(ur) else pd.DataFrame()
        with np.errstate(divide="ignore", invalid="ignore"):
            off_epa = tg["pbp_off_all_epa_sum"].to_numpy() / tg["pbp_off_all_plays"].to_numpy()
            def_epa = tg["pbp_def_all_epa_sum"].to_numpy() / tg["pbp_def_all_plays"].to_numpy()
        if len(u) and {"off_all", "def_all"} <= set(u.columns):
            opp_def = u["def_all"].reindex(tg["opp_id"]).to_numpy()
            opp_off = u["off_all"].reindex(tg["opp_id"]).to_numpy()
            epa_resid = (off_epa - opp_def) - (def_epa - opp_off)
        else:
            epa_resid = np.full(len(tg), np.nan)
        t = pd.DataFrame({"team_id": tg["team_id"].to_numpy(), "t": tg["start_time_utc"].to_numpy(),
                          "r": resid, "e": epa_resid}).sort_values("t")
        last = t.groupby("team_id").tail(3).groupby("team_id")[["r", "e"]].mean()
        season = t.groupby("team_id")[["r", "e"]].mean()
        n = t.groupby("team_id").size()
        d = (last - season).reindex(keys["team_id"])
        n = n.reindex(keys["team_id"])
        enough = (n >= 4).to_numpy()
        return _out(keys, {
            "str_form_pts3": np.where(enough, d["r"], np.nan), "str_form_epa3": np.where(enough, d["e"], np.nan),
            "str_form_games": n.fillna(0.0),
        })


# ---- polls -----------------------------------------------------------------------------

AP, COACHES, CFP = "AP Top 25", "Coaches Poll", "Playoff Committee Rankings"


class Polls(FeatureGroup):
    name = "str_polls"
    prefix = "str_"
    level = "team"
    columns = [
        _col("str_poll_ap_rank", "AP rank in the latest poll before the game (empty if unranked)", ["rankings"], 2014),
        _col("str_poll_ap_points", "AP poll points in the latest poll (0 if not listed)", ["rankings"], 2014),
        _col("str_poll_coaches_rank", "Coaches Poll rank in the latest poll (empty if unranked)", ["rankings"], 2014),
        _col("str_poll_coaches_points", "Coaches Poll points in the latest poll (0 if not listed)", ["rankings"], 2014),
        _col("str_poll_cfp_rank", "Playoff committee rank, lagged one week in the regular season because it is "
             "published after the week starts (empty if unranked or none published)", ["rankings"], 2014),
        _col("str_poll_ranked", "Ranked in the AP poll (1) or not (0)", ["rankings"], 2014),
    ]

    def build(self, view, games):
        keys = team_keys(games)
        vals = {c.name: np.full(len(keys), np.nan) for c in self.columns}
        if not view.has("rankings"):
            return _out(keys, vals)
        r = view.table("rankings", seasons=view.season)
        r = r[r["poll"].isin([AP, COACHES, CFP])]
        if r.empty:
            return _out(keys, vals)
        r = r.assign(_order=np.where(r["seasonType"] == "postseason", 100, 0) + r["week"])
        regular = (games["season_type"] == "regular").all()
        for poll, rank_col, pts_col in ((AP, "str_poll_ap_rank", "str_poll_ap_points"),
                                        (COACHES, "str_poll_coaches_rank", "str_poll_coaches_points"),
                                        (CFP, "str_poll_cfp_rank", None)):
            p = r[r["poll"] == poll]
            if poll == CFP and regular and len(p):
                cur_week = int(games["week"].max())
                p = p[p["week"] < cur_week]
            if p.empty:
                if pts_col:
                    vals[pts_col] = np.full(len(keys), np.nan)
                continue
            latest = p[p["_order"] == p["_order"].max()].drop_duplicates("teamId").set_index("teamId")
            vals[rank_col] = latest["rank"].reindex(keys["team_id"]).to_numpy(dtype=np.float64)
            if pts_col:
                vals[pts_col] = pd.to_numeric(latest["points"], errors="coerce").reindex(keys["team_id"]).fillna(0.0).to_numpy()
        ap = vals["str_poll_ap_rank"]
        has_ap = r["poll"].eq(AP).any()
        vals["str_poll_ranked"] = np.where(has_ap, np.isfinite(ap).astype(float), np.nan)
        return _out(keys, vals)


GROUPS = [Power(), Prior(), Elo(), Schedule(), Luck(), SpecialTeams(), Form(), Polls()]
