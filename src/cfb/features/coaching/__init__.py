"""Coaching features (W5), prefix coach_. One row per team per game.

Two groups:
    coach_tenure   who coaches the team (tenure, first-year and interim flags, games in
                   the data) and the coach's record against the closing spread in past
                   games, career and at the current school.
    coach_style    fourth-down go rate above expected, neutral-situation run rate above
                   expected, and offensive flexibility: how far the coach's run rate
                   moves from their own tendency as the opponent's run-versus-pass
                   defensive gap widens, plus the shift that implies for this game.

Coach metrics follow the coach across schools by CFBD coach id. Games before 2014 are
not in the data, so "career" means career since 2014. Details: docs/coaching.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from . import metrics as M
from .mapping import coach_game_map

TEAM_KEYS = ["game_id", "team_id"]


def _slate_and_history(view, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(slate rows with coach facts, played team-games with coach and result columns)."""
    cmap = coach_game_map(view, games)
    slate_ids = set(games["game_id"])
    slate = cmap[cmap["game_id"].isin(slate_ids)].reset_index(drop=True)
    tg = view.table("team_game", columns=["game_id", "team_id", "opp_id", "is_home", "margin"])
    history = tg.merge(cmap.drop(columns=["start_time_utc"]), on=["game_id", "team_id"], how="inner")
    history = history.dropna(subset=["coach_id"])
    return slate, history


def _attach(slate: pd.DataFrame, frame: pd.DataFrame, on: list[str], cols: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return slate.assign(**{new: np.nan for new in cols.values()})
    part = frame[on + list(cols)].rename(columns=cols)
    return slate.merge(part, on=on, how="left")


def _finish(out: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keys plus the group's columns, all float64 (empty inputs leave object columns)."""
    res = out[TEAM_KEYS + columns].copy()
    res[columns] = res[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    return res


class CoachTenure(FeatureGroup):
    name = "coach_tenure"
    prefix = "coach_"
    level = "team"
    columns = [
        FeatureColumn("coach_tenure_seasons", "Head coach's season at the school: 1 in the first full season, "
                      "0 while finishing a season as an in-season replacement", ("coach_tenures", "coaches"), 2014),
        FeatureColumn("coach_first_year", "1 if the head coach is in a first season at the school (tenure 0 or 1)",
                      ("coach_tenures", "coaches"), 2014),
        FeatureColumn("coach_interim", "1 if the head coach is an interim or took over during this season",
                      ("coach_tenures", "coaches"), 2014),
        FeatureColumn("coach_games_career", "Games the head coach has led since 2014, any school",
                      ("coach_tenures", "coaches", "team_game"), 2014),
        FeatureColumn("coach_games_school", "Games the head coach has led at this school since 2014",
                      ("coach_tenures", "coaches", "team_game"), 2014),
        FeatureColumn("coach_ats_career", "Coach's cover rate against the closing spread in past games since 2014, "
                      f"shrunk toward 0.5 with {M.ATS_PRIOR_GAMES:.0f} pseudo-games",
                      ("coach_tenures", "coaches", "team_game", "lines"), 2014),
        FeatureColumn("coach_ats_career_n", "Past games with a closing spread behind coach_ats_career (pushes excluded)",
                      ("coach_tenures", "coaches", "team_game", "lines"), 2014),
        FeatureColumn("coach_ats_school", "Coach's cover rate at this school, shrunk like coach_ats_career",
                      ("coach_tenures", "coaches", "team_game", "lines"), 2014),
        FeatureColumn("coach_ats_school_n", "Past games with a closing spread behind coach_ats_school",
                      ("coach_tenures", "coaches", "team_game", "lines"), 2014),
    ]

    def build(self, view, games):
        slate, history = _slate_and_history(view, games)
        out = slate[TEAM_KEYS + ["coach_id", "tenure_seasons", "interim"]].copy()
        out["coach_tenure_seasons"] = out["tenure_seasons"]
        out["coach_first_year"] = (out["tenure_seasons"] <= 1).astype(float).where(out["tenure_seasons"].notna())
        out["coach_interim"] = out["interim"]

        career = history.groupby("coach_id").size().rename("coach_games_career").reset_index()
        school = history.groupby(["coach_id", "team_id"]).size().rename("coach_games_school").reset_index()
        out = out.merge(career, on="coach_id", how="left").merge(school, on=["coach_id", "team_id"], how="left")
        known = out["coach_id"].notna()
        for col in ("coach_games_career", "coach_games_school"):
            out[col] = out[col].astype("float64").where(~known | out[col].notna(), 0.0)

        lines = view.table("lines", columns=["id", "lines"]) if view.has("lines") else pd.DataFrame(
            columns=["id", "lines"])
        spreads = M.closing_spreads(lines)
        ats = M.ats_results(history, spreads)
        ats = ats.assign(resid=ats["cover"] - 0.5)
        c = M.shrunk_rate(ats, ["coach_id"], view.season, None, M.ATS_PRIOR_GAMES)
        s = M.shrunk_rate(ats, ["coach_id", "team_id"], view.season, None, M.ATS_PRIOR_GAMES)
        out = _attach(out, c, ["coach_id"], {"value": "coach_ats_career", "n": "coach_ats_career_n"})
        out = _attach(out, s, ["coach_id", "team_id"], {"value": "coach_ats_school", "n": "coach_ats_school_n"})
        for rate, n in (("coach_ats_career", "coach_ats_career_n"), ("coach_ats_school", "coach_ats_school_n")):
            out[n] = out[n].astype("float64").where(~known | out[n].notna(), 0.0)
            out[rate] = (0.5 + out[rate].astype("float64")).where(known)
            out.loc[known & (out[n] == 0), rate] = 0.5
        return _finish(out, self.column_names)


class CoachStyle(FeatureGroup):
    name = "coach_style"
    prefix = "coach_"
    level = "team"
    columns = [
        FeatureColumn("coach_fourth_go_oe", "Coach's fourth-down go rate above expected (distance, field position, "
                      f"score, time; outside garbage time), half-life {M.FOURTH_HALF_LIFE:g} seasons, shrunk "
                      "toward 0", ("coach_tenures", "coaches", "pbp"), 2014),
        FeatureColumn("coach_fourth_n", "Fourth-down decisions behind coach_fourth_go_oe (unweighted)",
                      ("coach_tenures", "coaches", "pbp"), 2014),
        FeatureColumn("coach_run_oe", "Coach's run rate above expected on first and second down, score within 14, "
                      f"outside garbage time; half-life {M.RUN_HALF_LIFE:g} season, shrunk toward 0",
                      ("coach_tenures", "coaches", "pbp"), 2014),
        FeatureColumn("coach_run_n", "Neutral-situation plays behind coach_run_oe (unweighted)",
                      ("coach_tenures", "coaches", "pbp"), 2014),
        FeatureColumn("coach_flex_slope", "Change in the coach's run rate (vs own season tendency) per unit of the "
                      "opponent's rush-minus-pass EPA allowed gap, shrunk toward the pooled slope",
                      ("coach_tenures", "coaches", "pbp", "unit_ratings"), 2014),
        FeatureColumn("coach_flex_games", "Past games with a rated opponent behind coach_flex_slope",
                      ("coach_tenures", "coaches", "pbp", "unit_ratings"), 2014),
        FeatureColumn("coach_flex_shift", "coach_flex_slope times this opponent's centred rush-minus-pass gap: the "
                      "run-rate shift from tendency the coach's history implies for this game",
                      ("coach_tenures", "coaches", "pbp", "unit_ratings"), 2014),
    ]

    def build(self, view, games):
        slate, history = _slate_and_history(view, games)
        out = slate[TEAM_KEYS + ["coach_id", "season"]].copy()
        coach_of = history[["game_id", "team_id", "coach_id"]]

        pbp = view.table("pbp", columns=M.PBP_COLUMNS)
        fourth = M.add_expectation(M.fourth_down_plays(pbp))
        fourth = fourth.merge(coach_of, left_on=["game_id", "offense_id"], right_on=["game_id", "team_id"])
        f = M.shrunk_rate(fourth, ["coach_id"], view.season, M.FOURTH_HALF_LIFE, M.FOURTH_PRIOR)
        out = _attach(out, f, ["coach_id"], {"value": "coach_fourth_go_oe", "n": "coach_fourth_n"})

        neutral = M.add_expectation(M.neutral_plays(pbp))
        run = neutral.merge(coach_of, left_on=["game_id", "offense_id"], right_on=["game_id", "team_id"])
        r = M.shrunk_rate(run, ["coach_id"], view.season, M.RUN_HALF_LIFE, M.RUN_PRIOR)
        out = _attach(out, r, ["coach_id"], {"value": "coach_run_oe", "n": "coach_run_n"})

        # Flexibility: game-level run residuals against the opponent's gap at that game's cutoff.
        gaps = M.defensive_gaps(view.table("unit_ratings", columns=["team_id", "cutoff_utc", "unit", "epa_adj",
                                                                    "n_plays"]))
        cut = view.schedule()[["game_id", "cutoff_utc"]]
        g = M.game_run_residuals(neutral).merge(history[["game_id", "team_id", "opp_id", "coach_id", "season"]],
                                                on=["game_id", "team_id"])
        g = g.merge(cut, on="game_id").merge(gaps.rename(columns={"team_id": "opp_id", "gap": "x"}),
                                             on=["opp_id", "cutoff_utc"], how="left")
        slopes, pooled = M.flex_slopes(g)
        out = _attach(out, slopes, ["coach_id"], {"slope": "coach_flex_slope", "games": "coach_flex_games"})
        out["coach_flex_slope"] = out["coach_flex_slope"].astype("float64").where(
            out["coach_id"].isna() | out["coach_flex_slope"].notna(), pooled)

        # This game's opponent gap, from the latest ratings visible at this cutoff.
        opp = pd.concat([games[["game_id", "home_id", "away_id"]].rename(columns={"home_id": "team_id",
                                                                                 "away_id": "opp_id"}),
                         games[["game_id", "away_id", "home_id"]].rename(columns={"away_id": "team_id",
                                                                                 "home_id": "opp_id"})])
        latest = gaps["cutoff_utc"].max() if len(gaps) else None
        now = gaps[gaps["cutoff_utc"] == latest][["team_id", "gap"]].rename(columns={"team_id": "opp_id"})
        out = out.merge(opp, on=TEAM_KEYS, how="left").merge(now, on="opp_id", how="left")
        out["coach_flex_shift"] = out["coach_flex_slope"] * out["gap"]

        known = out["coach_id"].notna()
        for n in ("coach_fourth_n", "coach_run_n", "coach_flex_games"):
            out[n] = out[n].astype("float64").where(~known | out[n].notna(), 0.0)
        for v in ("coach_fourth_go_oe", "coach_run_oe"):
            out[v] = out[v].astype("float64").where(~known | out[v].notna(), 0.0)
        return _finish(out, self.column_names)


GROUPS = [CoachTenure(), CoachStyle()]
