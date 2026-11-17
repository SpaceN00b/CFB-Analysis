"""Coach metrics from past games: record against the spread, fourth-down aggressiveness,
run/pass mix above expected, and offensive flexibility.

Everything here takes frames read through an AsOfView, so a value only uses what the
view shows. Expectations are cell rates fitted on visible plays of the play's own season
and the season before (`window` seasons), so they never use later plays than the view's.
The shrinkage constants and half-lives below were chosen on seasons through 2025 with
scripts/coaching_check.py; docs/coaching.md has the numbers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

# ---- constants chosen on seasons <= 2025 (scripts/coaching_check.py) -------------------
ATS_PRIOR_GAMES = 200.0  # pseudo-games at 50% for the ATS rate
FOURTH_PRIOR = 75.0  # pseudo-decisions at 0 for go rate over expected
FOURTH_HALF_LIFE = 1.0  # seasons
RUN_PRIOR = 600.0  # pseudo-plays at 0 for run rate over expected
RUN_HALF_LIFE = 1.0  # seasons
FLEX_GAME_VAR = 0.002  # extra game-level variance of a game's run rate over expected
FLEX_SLOPE_VAR = 0.0006  # between-coach variance of the true flexibility slope (MoM, <= 2025)
FLEX_MIN_UNIT_PLAYS = 60  # opponent defensive plays needed before its gap counts

LINE_PROVIDERS = ("consensus", "Bovada", "DraftKings", "ESPN Bet")
NEUTRAL_MAX_SCORE = 14
EXPECT_WINDOW = 2  # seasons: the play's season and the one before
CELL_PRIOR = 20.0  # pseudo-plays pulling a cell rate toward its coarse cell

PBP_COLUMNS = ["game_id", "season", "offense_id", "period", "clock_seconds", "down", "distance",
               "yards_to_goal", "score_diff", "play_class", "is_rush", "is_pass", "is_scrimmage", "garbage_time"]


# ---- lines -----------------------------------------------------------------------------

def closing_spreads(lines: pd.DataFrame) -> pd.DataFrame:
    """game_id, spread (home perspective: negative means the home team is favoured),
    taken from the first provider in LINE_PROVIDERS that has one."""
    rank = {p: i for i, p in enumerate(LINE_PROVIDERS)}
    out = []
    for gid, raw in zip(lines["id"].to_numpy(), lines["lines"].to_numpy()):
        items = json.loads(raw) if isinstance(raw, str) else list(raw if raw is not None else [])
        best = None
        for item in items:
            r = rank.get(item.get("provider"))
            if r is None or item.get("spread") is None:
                continue
            if best is None or r < best[0]:
                best = (r, float(item["spread"]))
        if best is not None:
            out.append((int(gid), best[1]))
    return pd.DataFrame(out, columns=["game_id", "spread"])


def ats_results(history: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    """history (game_id, team_id, is_home, margin, ...) with `cover` 1/0, pushes and
    games without a line dropped."""
    h = history.merge(spreads, on="game_id", how="inner")
    team_spread = np.where(h["is_home"], h["spread"], -h["spread"])
    edge = h["margin"].to_numpy(dtype="float64") + team_spread
    h = h.assign(ats_edge=edge)
    h = h[np.isfinite(edge) & (edge != 0)]
    return h.assign(cover=(h["ats_edge"] > 0).astype(float))


# ---- plays -----------------------------------------------------------------------------

def _bucket(values: pd.Series, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype="float64"), values.to_numpy(dtype="float64"), side="right")


def fourth_down_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Fourth-down decisions outside garbage time: go (rush or pass, fakes included)
    against punt or field goal, with the situation cells."""
    p = pbp[(pbp["down"] == 4) & pbp["play_class"].isin(["rush", "pass", "punt", "field_goal"])
            & ~pbp["garbage_time"] & (pbp["distance"] > 0)]
    go = p["play_class"].isin(["rush", "pass"]).astype(float)
    late = (p["period"] >= 4) & (p["clock_seconds"] <= 300)
    half_end = (p["period"] == 2) & (p["clock_seconds"] <= 120)
    time_b = np.where(p["period"] >= 5, 4, np.where(late, 3, np.where(half_end, 2, np.where(p["period"] >= 4, 1, 0))))
    coarse = _bucket(p["distance"], [2, 3, 4, 6, 10]) * 10 + _bucket(p["yards_to_goal"], [4, 11, 21, 36, 51, 66])
    fine = coarse * 100 + _bucket(p["score_diff"], [-16, -8, -3, 0, 1, 4, 9, 17]) * 10 + time_b
    return pd.DataFrame({"game_id": p["game_id"].to_numpy(), "season": p["season"].to_numpy(),
                         "offense_id": p["offense_id"].to_numpy(), "y": go.to_numpy(),
                         "coarse": coarse, "cell": fine})


def neutral_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Scrimmage plays on first and second down, score within 14, outside garbage time,
    with y = 1 for a rush (pbp `is_rush`) and situation cells."""
    p = pbp[pbp["is_scrimmage"] & pbp["down"].isin([1, 2]) & (pbp["score_diff"].abs() <= NEUTRAL_MAX_SCORE)
            & ~pbp["garbage_time"]]
    coarse = (p["down"].to_numpy() * 10 + _bucket(p["distance"], [3, 6, 10, 11]))
    late = ((p["period"] >= 4) & (p["clock_seconds"] <= 300)).to_numpy()
    period = np.where(late, 5, np.minimum(p["period"].to_numpy(), 4))
    fine = (coarse * 1000 + _bucket(p["yards_to_goal"], [6, 11, 21, 41, 81]) * 100
            + _bucket(p["score_diff"], [-7, 0, 1, 8]) * 10 + period)
    return pd.DataFrame({"game_id": p["game_id"].to_numpy(), "season": p["season"].to_numpy(),
                         "offense_id": p["offense_id"].to_numpy(), "y": p["is_rush"].to_numpy(dtype="float64"),
                         "coarse": coarse, "cell": fine})


def _window_rates(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sum and count of y per (season, key) over the window of seasons ending at season."""
    g = frame.groupby(["season", key])["y"].agg(["sum", "count"]).reset_index()
    parts = []
    for lag in range(EXPECT_WINDOW):
        parts.append(g.assign(season=g["season"] + lag))
    w = pd.concat(parts).groupby(["season", key])[["sum", "count"]].sum().reset_index()
    return w


def add_expectation(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `exp` (expected y) and `resid` (y - exp) from cell rates of the play's own
    and previous season(s), each cell pulled toward its coarse cell."""
    if frame.empty:
        return frame.assign(exp=pd.Series(dtype="float64"), resid=pd.Series(dtype="float64"))
    coarse = _window_rates(frame, "coarse")
    coarse["c_rate"] = coarse["sum"] / coarse["count"]
    fine = _window_rates(frame, "cell")
    out = frame.merge(coarse[["season", "coarse", "c_rate"]], on=["season", "coarse"], how="left")
    out = out.merge(fine.rename(columns={"sum": "f_sum", "count": "f_n"}), on=["season", "cell"], how="left")
    exp = (out["f_sum"] + CELL_PRIOR * out["c_rate"]) / (out["f_n"] + CELL_PRIOR)
    out = out.drop(columns=["c_rate", "f_sum", "f_n"]).assign(exp=exp.to_numpy())
    return out.assign(resid=out["y"] - out["exp"])


# ---- coach aggregates ------------------------------------------------------------------

def season_weights(seasons: pd.Series, current: int, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(seasons))
    return 0.5 ** ((current - seasons.to_numpy(dtype="float64")) / half_life)


def shrunk_rate(plays: pd.DataFrame, by: list[str], current: int, half_life: float | None,
                prior: float) -> pd.DataFrame:
    """Weighted mean residual per `by`, shrunk toward 0 with `prior` pseudo-plays.
    Returns by + [value, n] where n is the unweighted count."""
    if plays.empty:
        return pd.DataFrame(columns=by + ["value", "n"])
    w = season_weights(plays["season"], current, half_life)
    tmp = plays[by].assign(_wr=w * plays["resid"].to_numpy(), _w=w, _n=1.0)
    agg = tmp.groupby(by)[["_wr", "_w", "_n"]].sum().reset_index()
    agg["value"] = agg["_wr"] / (agg["_w"] + prior)
    return agg[by + ["value"]].assign(n=agg["_n"].to_numpy())


def game_run_residuals(neutral: pd.DataFrame) -> pd.DataFrame:
    """Per offense per game: neutral plays, mean run rate over expected."""
    g = neutral.groupby(["game_id", "offense_id"]).agg(n=("resid", "size"), r=("resid", "mean")).reset_index()
    return g.rename(columns={"offense_id": "team_id"})


def defensive_gaps(unit_ratings: pd.DataFrame) -> pd.DataFrame:
    """team_id, cutoff_utc, gap: rush EPA allowed minus pass EPA allowed, minus the mean
    of that difference over teams rated at the same cutoff. Positive means the defense
    is weaker against the run than against the pass, relative to the league."""
    u = unit_ratings[unit_ratings["unit"].isin(["def_rush", "def_pass"])]
    u = u[u["n_plays"] >= FLEX_MIN_UNIT_PLAYS]
    wide = u.pivot_table(index=["team_id", "cutoff_utc"], columns="unit", values="epa_adj", aggfunc="last")
    if wide.empty or not {"def_rush", "def_pass"} <= set(wide.columns):
        return pd.DataFrame({"team_id": pd.Series(dtype="int64"),
                             "cutoff_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                             "gap": pd.Series(dtype="float64")})
    wide = wide.dropna(subset=["def_rush", "def_pass"]).reset_index()
    raw = wide["def_rush"] - wide["def_pass"]
    wide["gap"] = raw - raw.groupby(wide["cutoff_utc"]).transform("mean")
    return wide[["team_id", "cutoff_utc", "gap"]]


def flex_slopes(games: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Flexibility per coach from past games.

    games: coach_id, season, n (neutral plays), r (run rate over expected), x (opponent
    defensive gap at that game's cutoff). Within each coach-season, r and x are centred,
    so the slope says how far the run rate moved from the coach's own tendency that
    season as the opponent's gap moved. Weights 1 / (0.25 / n + FLEX_GAME_VAR).
    The slope is shrunk toward the pooled slope of all coaches by its precision.
    Returns (coach_id, slope, games, raw_slope), pooled slope.
    """
    g = games.dropna(subset=["coach_id", "x", "r"])
    g = g[g["n"] > 0]
    if g.empty:
        return pd.DataFrame(columns=["coach_id", "slope", "games", "raw_slope"]), np.nan
    w = 1.0 / (0.25 / g["n"] + FLEX_GAME_VAR)
    g = g.assign(w=w, wx=w * g["x"], wr=w * g["r"])
    grp = g.groupby(["coach_id", "season"])
    sw = grp["w"].transform("sum")
    xc = g["x"] - grp["wx"].transform("sum") / sw
    rc = g["r"] - grp["wr"].transform("sum") / sw
    g = g.assign(sxx=g["w"] * xc * xc, sxy=g["w"] * xc * rc, one=1.0)
    per = g.groupby("coach_id")[["sxx", "sxy", "one"]].sum()
    pooled = float(per["sxy"].sum() / per["sxx"].sum()) if per["sxx"].sum() > 0 else np.nan
    raw = per["sxy"] / per["sxx"].where(per["sxx"] > 0)
    shrink = per["sxx"] / (per["sxx"] + 1.0 / FLEX_SLOPE_VAR)
    slope = pooled + (raw.fillna(pooled) - pooled) * shrink
    out = pd.DataFrame({"coach_id": per.index, "slope": slope.to_numpy(), "games": per["one"].to_numpy(),
                        "raw_slope": raw.to_numpy()})
    return out.reset_index(drop=True), pooled
