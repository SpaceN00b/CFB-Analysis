"""matchup_style: style profile, unit edges and variance inputs, one row per team and game.

Style profile: the team's own tendencies at the cutoff (season to date, with the
previous season counted as PRIOR_GAMES games). Unit edges: the team's units against
the opponent's from `unit_ratings` (offense against defense in both directions),
plus style edges (havoc, line yards). Variance inputs: the two teams' pace and
reliance on explosive plays, the same on both rows of a game.

Sign of every edge: positive is good for the row's team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from .stats import BUCKETS, PRIOR_GAMES, blended_profiles, fbs_teams, game_stats

_PROFILE_SOURCES = ("pbp", "team_game")

_PROFILE = [
    ("mat_style_games", "games", "Games of the current season behind the style profile", ("team_game",), 2014),
    ("mat_off_run_rate", "off_run_rate", "Share of offensive scrimmage plays that are rushes (garbage time removed)",
     _PROFILE_SOURCES, 2014),
    ("mat_off_run_rate_neutral", "off_run_rate_neutral",
     "Rush share on first and second down with the score within 14 (garbage time removed)", _PROFILE_SOURCES, 2014),
    *[(f"mat_off_run_rate_{b}", f"off_run_rate_{b}",
       f"Rush share in the {b.replace('_', ' ')} bucket (short 1-3, med 4-7, long 8+ on 2nd; long 4+ on 3rd/4th), "
       "score within 14", _PROFILE_SOURCES, 2014) for b in BUCKETS],
    ("mat_off_plays", "off_plays", "Offensive scrimmage plays per game, garbage time included", ("pbp",), 2014),
    ("mat_pace_plays", "pace_plays", "Scrimmage plays per game by both teams in the team's games", ("pbp",), 2014),
    ("mat_off_sec_per_play", "off_sec_per_play", "Drive time per offensive drive play, seconds", ("drives",), 2014),
    ("mat_off_expl_share", "off_expl_share",
     "Share of the offense's positive EPA from explosive plays (rush 12+ yards, pass 16+)", ("pbp",), 2014),
    ("mat_off_expl_rate", "off_expl_rate", "Explosive plays per offensive scrimmage play", ("pbp",), 2014),
    ("mat_def_expl_share", "def_expl_share", "Share of positive EPA allowed that came on explosive plays",
     ("pbp",), 2014),
    ("mat_def_expl_rate", "def_expl_rate", "Explosive plays allowed per scrimmage play", ("pbp",), 2014),
    ("mat_off_sd_success", "off_sd_success", "Offensive success rate on standard downs", ("pbp",), 2014),
    ("mat_off_pd_success", "off_pd_success", "Offensive success rate on passing downs", ("pbp",), 2014),
    ("mat_off_pd_rate", "off_pd_rate", "Share of offensive plays run on passing downs", ("pbp",), 2014),
    ("mat_def_sd_success", "def_sd_success", "Success rate allowed on standard downs", ("pbp",), 2014),
    ("mat_def_pd_success", "def_pd_success", "Success rate allowed on passing downs", ("pbp",), 2014),
    ("mat_off_havoc_allowed", "off_havoc_allowed", "Havoc events against the offense per play (thin before 2016)",
     ("stats_game_havoc",), 2014),
    ("mat_def_havoc", "def_havoc", "Havoc events created by the defense per play (thin before 2016)",
     ("stats_game_havoc",), 2014),
    ("mat_off_line_yards", "off_line_yards", "Offensive line yards per rush, rush-weighted", ("team_game",), 2014),
    ("mat_off_stuff_rate", "off_stuff_rate", "Share of the offense's rushes stuffed, rush-weighted", ("team_game",), 2014),
    ("mat_def_line_yards", "def_line_yards", "Line yards per rush allowed, rush-weighted", ("team_game",), 2014),
    ("mat_def_stuff_rate", "def_stuff_rate", "Share of opponents' rushes stuffed, rush-weighted", ("team_game",), 2014),
]

_EDGE_KINDS = [
    ("rush_epa", "rush EPA per play", ("unit_ratings",)),
    ("pass_epa", "pass EPA per play", ("unit_ratings",)),
    ("rush_success", "rush success rate", ("unit_ratings",)),
    ("pass_success", "pass success rate", ("unit_ratings",)),
    ("explosive", "EPA per successful play (all plays)", ("unit_ratings",)),
    ("mix_epa", "EPA per play weighted by the offense's run rate", ("unit_ratings", "pbp")),
    ("align", "run-rate lean times (rush edge minus pass edge)", ("unit_ratings", "pbp")),
    ("havoc", "havoc rate (style, unadjusted)", ("stats_game_havoc",)),
    ("line_yards", "line yards per rush (style, unadjusted)", ("team_game",)),
]

_VARIANCE = [
    ("mat_var_pace", "Mean offensive scrimmage plays per game of the two teams", ("pbp",), 2014),
    ("mat_var_sec_per_play", "Mean seconds per drive play of the two offenses", ("drives",), 2014),
    ("mat_var_expl_share", "Mean explosive share of positive EPA of the two offenses", ("pbp",), 2014),
]


def _edge_columns() -> list[FeatureColumn]:
    cols = []
    for side, who in (("off", "the team's offense against the opponent's defense"),
                      ("def", "the team's defense against the opponent's offense")):
        for kind, what, sources in _EDGE_KINDS:
            cols.append(FeatureColumn(f"mat_{side}_edge_{kind}",
                                      f"Edge of {who}: {what}, against league average; positive favors the team",
                                      sources, 2014))
    return cols


_UNIT_METRICS = {"epa": "epa_adj", "success": "success_adj", "explosive": "explosive_adj"}


def _unit_wide(view, season_start) -> pd.DataFrame:
    """Latest unit ratings of this season, wide: index team_id, columns '<unit>_<metric>'."""
    ur = view.latest("unit_ratings", ["team_id", "unit"])
    ur = ur[(ur["cutoff_utc"] >= season_start) & (ur["n_plays"] > 0)]
    if ur.empty:
        return pd.DataFrame()
    wide = ur.pivot(index="team_id", columns="unit", values=list(_UNIT_METRICS.values()))
    short = {v: k for k, v in _UNIT_METRICS.items()}
    wide.columns = [f"{unit}_{short[metric]}" for metric, unit in wide.columns]
    return wide.astype(float)


class MatchupStyle(FeatureGroup):
    name = "matchup_style"
    prefix = "mat_"
    level = "team"
    mode = "per_cutoff"
    columns = (
        [FeatureColumn(c, d, s, f) for c, _, d, s, f in _PROFILE]
        + _edge_columns()
        + [FeatureColumn(c, d, s, f) for c, d, s, f in _VARIANCE]
    )

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        season = view.season
        stats = game_stats(view, [season - 1, season])
        prof = blended_profiles(stats, season, PRIOR_GAMES)
        prof["pace_plays"] = prof["off_plays"] + prof["def_plays"]

        sched = view.schedule()
        season_start = sched.loc[sched["season"] == season, "cutoff_utc"].min()
        if pd.isna(season_start):
            season_start = view.cutoff
        units = _unit_wide(view, season_start)
        fbs = fbs_teams(view, season)

        # league averages over FBS teams
        def lg(frame: pd.DataFrame, col: str) -> float:
            if frame.empty or col not in frame.columns:
                return np.nan
            vals = frame[col].reindex(fbs).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if len(vals) else np.nan

        rows = pd.concat([
            pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["home_id"].to_numpy(),
                          "opp_id": games["away_id"].to_numpy()}),
            pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["away_id"].to_numpy(),
                          "opp_id": games["home_id"].to_numpy()}),
        ], ignore_index=True)

        def team(col: str, frame=prof) -> np.ndarray:
            if frame.empty or col not in frame.columns:
                return np.full(len(rows), np.nan)
            return frame[col].reindex(rows["team_id"]).to_numpy(dtype=float)

        def opp(col: str, frame=prof) -> np.ndarray:
            if frame.empty or col not in frame.columns:
                return np.full(len(rows), np.nan)
            return frame[col].reindex(rows["opp_id"]).to_numpy(dtype=float)

        out = rows[["game_id", "team_id"]].copy()
        for col, src, *_ in _PROFILE:
            out[col] = team(src)
        out["mat_style_games"] = np.nan_to_num(out["mat_style_games"].to_numpy(dtype=float))

        # unit edges: offense(T) vs defense(O), and offense(O) vs defense(T), centered on the league
        def centered(unit_col: str, who) -> np.ndarray:
            return who(unit_col, units) - lg(units, unit_col)

        for side, o_side, d_side, sign in (("off", team, opp, 1.0), ("def", opp, team, -1.0)):
            e = {}
            for kind, unit, metric in (("rush_epa", "rush", "epa"), ("pass_epa", "pass", "epa"),
                                       ("rush_success", "rush", "success"), ("pass_success", "pass", "success"),
                                       ("explosive", "all", "explosive")):
                e[kind] = sign * (centered(f"off_{unit}_{metric}", o_side) + centered(f"def_{unit}_{metric}", d_side))
            rr = o_side("off_run_rate")
            e["mix_epa"] = rr * e["rush_epa"] + (1 - rr) * e["pass_epa"]
            e["align"] = (rr - lg(prof, "off_run_rate")) * (e["rush_epa"] - e["pass_epa"])
            # havoc against the offense is bad for it; line yards for it are good
            havoc = (o_side("off_havoc_allowed") - lg(prof, "off_havoc_allowed")
                     + d_side("def_havoc") - lg(prof, "def_havoc"))
            e["havoc"] = -sign * havoc
            ly = (o_side("off_line_yards") - lg(prof, "off_line_yards")
                  + d_side("def_line_yards") - lg(prof, "def_line_yards"))
            e["line_yards"] = sign * ly
            for kind, *_ in _EDGE_KINDS:
                out[f"mat_{side}_edge_{kind}"] = e[kind]

        out["mat_var_pace"] = (team("off_plays") + opp("off_plays")) / 2
        out["mat_var_sec_per_play"] = (team("off_sec_per_play") + opp("off_sec_per_play")) / 2
        out["mat_var_expl_share"] = (team("off_expl_share") + opp("off_expl_share")) / 2
        out = out.drop_duplicates(["game_id", "team_id"])
        return out[["game_id", "team_id"] + self.column_names]
