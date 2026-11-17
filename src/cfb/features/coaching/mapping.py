"""Which head coach led each team in each game, as known at the view's cutoff.

Sources, all through the view:
    coach_tenures   one row per tenure: coach, team, start and end year, hire date, and
                    (recent rows only) an effective start date and an interim flag.
                    The view shows a tenure once it has begun and blanks end years that
                    had not happened yet.
    coaches         per finished season, the number of games each coach led each team.
                    Used to place in-season changes of past seasons, because tenure
                    rows before 2022 rarely flag interims and a few have no hire date.

Rules
    * Candidates for a team-season are tenures with start year <= season and an end year
      that is empty or >= season. A tenure that started on or after 25 August of its
      season (by effective start, else hire date) is an in-season start: it takes over
      from that moment. Of the candidates that had started by kickoff, the latest
      in-season start wins; without one, the season-start tenure with the latest start
      year wins. A tenure that starts in the same season as another candidate but has no
      date cannot be placed, so it never takes over in the current season.
    * Finished seasons: when `coaches` lists one coach for the team-season, that coach
      led every game. When it lists several, they are ordered by when the date rule first
      assigns them and given their game counts in kickoff order; the last takes the rest.
    * Interim means the tenure is flagged interim or the coach took over during the
      season (that includes a new permanent coach who coaches only the bowl).
    * Without coach_tenures in the warehouse every coach is unknown (empty values).
    * Tenure in seasons counts from the first full season at the school; rows of the
      same coach at the same school that follow on without a gap are joined. It is 0
      during an in-season debut, 1 in the first full season.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

IN_SEASON_FROM = (8, 25)  # month, day: a start on or after this date is an in-season start

MAP_COLUMNS = ["game_id", "team_id", "season", "start_time_utc", "coach_id", "tenure_seasons", "interim"]


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")


def _season_open(years: pd.Series) -> pd.Series:
    """25 August of each year, UTC (NaT for missing years)."""
    y = pd.to_numeric(years, errors="coerce")
    text = y.astype("Int64").astype("string") + f"-{IN_SEASON_FROM[0]:02d}-{IN_SEASON_FROM[1]:02d}"
    return pd.to_datetime(text, utc=True, errors="coerce")


def tenures(view) -> pd.DataFrame:
    """Tenure rows as known at the cutoff, with derived start facts.

    Columns: tenure_id, coach_id, team_id, start_year, end_year (NaN = open), start_at,
    in_season (started during its start year's season), first_full (first full season at
    the school, joining back-to-back rows), is_interim.
    """
    raw = view.table("coach_tenures") if view.has("coach_tenures") else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame(columns=["tenure_id", "coach_id", "team_id", "start_year", "end_year", "start_at",
                                     "in_season", "first_full", "is_interim"])
    t = pd.DataFrame({
        "tenure_id": raw["id"].to_numpy(),
        "coach_id": pd.to_numeric(raw["coach.id"]).to_numpy(),
        "team_id": pd.to_numeric(raw["team.id"]).to_numpy(),
        "start_year": pd.to_numeric(raw["startYear"], errors="coerce").to_numpy(),
        "end_year": pd.to_numeric(raw["endYear"], errors="coerce").to_numpy(dtype="float64"),
    })
    start_at = _utc(raw["effectiveStart"]) if "effectiveStart" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    hire = _utc(raw["hireDate"]) if "hireDate" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    t["start_at"] = start_at.fillna(hire).to_numpy()
    opens = _season_open(t["start_year"])
    nxt = _season_open(t["start_year"] + 1)
    t["in_season"] = ((t["start_at"] >= opens) & (t["start_at"] < nxt)).fillna(False).to_numpy(dtype=bool)
    interim = raw["isInterim"] if "isInterim" in raw.columns else pd.Series(False, index=raw.index)
    t["is_interim"] = interim.fillna(False).astype(bool).to_numpy()

    # First full season, joining rows of the same coach at the same school without a gap.
    t = t.sort_values(["coach_id", "team_id", "start_year", "tenure_id"]).reset_index(drop=True)
    first = t["start_year"] + t["in_season"].astype(int)
    span_first = first.copy()
    prev = t.shift(1)
    joined = ((t["coach_id"] == prev["coach_id"]) & (t["team_id"] == prev["team_id"])
              & (prev["end_year"].isna() | (t["start_year"] <= prev["end_year"] + 1)))
    vals = span_first.to_numpy(dtype="float64")
    for i in np.flatnonzero(joined.to_numpy()):
        vals[i] = vals[i - 1]
    t["first_full"] = vals
    return t


def _team_rows(view, games: pd.DataFrame | None) -> pd.DataFrame:
    """game_id, team_id, season, start_time_utc for every past game and the slate."""
    sched = view.schedule()
    if games is not None and len(games):
        sched = pd.concat([sched, games[[c for c in sched.columns if c in games.columns]]], ignore_index=True)
        sched = sched.drop_duplicates("game_id", keep="last")
    cols = ["game_id", "season", "start_time_utc"]
    long = pd.concat([
        sched[cols + ["home_id"]].rename(columns={"home_id": "team_id"}),
        sched[cols + ["away_id"]].rename(columns={"away_id": "team_id"}),
    ], ignore_index=True)
    return long.sort_values(["start_time_utc", "game_id", "team_id"]).reset_index(drop=True)


def _date_rule(rows: pd.DataFrame, ten: pd.DataFrame) -> pd.DataFrame:
    """Coach per team-game by tenure dates. Returns rows with tenure columns (NaN if none)."""
    keep = ["tenure_id", "coach_id", "team_id", "start_year", "end_year", "start_at", "in_season", "first_full",
            "is_interim"]
    cand = rows.reset_index().merge(ten[keep], on="team_id", how="inner")
    ok = (cand["start_year"] <= cand["season"]) & (cand["end_year"].isna() | (cand["end_year"] >= cand["season"]))
    cand = cand[ok]
    # In-season start for this row's season: dated start inside that season's window.
    opens, nxt = _season_open(cand["season"]), _season_open(cand["season"] + 1)
    in_season_start = cand["start_at"].where((cand["start_at"] >= opens) & (cand["start_at"] < nxt))
    cand = cand.assign(_iss=in_season_start)
    started = cand["_iss"].isna() | (cand["_iss"] <= cand["start_time_utc"])
    cand = cand[started]
    # An undated tenure starting this season, when an older candidate covers the season,
    # cannot be placed: rank it below every other candidate.
    older = cand.groupby("index")["start_year"].transform("min") < cand["season"]
    undated_new = cand["start_at"].isna() & (cand["start_year"] == cand["season"]) & older
    big_neg = pd.Timestamp("1900-01-01", tz="UTC")
    cand = cand.assign(
        _k1=cand["_iss"].fillna(big_neg),
        _k2=(~undated_new).astype(int),
        _k3=cand["start_year"],
        _k4=cand["start_at"].fillna(big_neg),
    )
    cand = cand.sort_values(["index", "_k1", "_k2", "_k3", "_k4", "tenure_id"])
    best = cand.drop_duplicates("index", keep="last").set_index("index")
    out = rows.copy()
    for col in ["coach_id", "first_full", "is_interim"]:
        out[col] = best[col].reindex(out.index)
    out["in_season_start"] = best["_iss"].notna().reindex(out.index).fillna(False).to_numpy(dtype=bool)
    return out


def season_counts(view) -> pd.DataFrame:
    """coach_id, team_id, season, games from `coaches` for finished seasons."""
    rows = []
    if not view.has("coaches"):
        return pd.DataFrame(rows, columns=["coach_id", "team_id", "season", "games"])
    raw = view.prior_seasons("coaches")
    for cid, seasons, season in zip(raw["id"], raw["seasons"], raw["season"]):
        items = json.loads(seasons) if isinstance(seasons, str) else list(seasons if seasons is not None else [])
        for item in items:
            if item.get("year") == season and item.get("teamId") is not None:
                rows.append((int(cid), int(item["teamId"]), int(season), int(item.get("games") or 0)))
    out = pd.DataFrame(rows, columns=["coach_id", "team_id", "season", "games"])
    return out.drop_duplicates(["coach_id", "team_id", "season"])


def _apply_counts(mapped: pd.DataFrame, counts: pd.DataFrame, played: pd.Series) -> pd.DataFrame:
    """Overwrite past seasons' coach assignment with the game counts of `coaches`."""
    counts = counts[counts["games"] > 0]
    if counts.empty:
        return mapped
    mapped = mapped.copy()
    mapped["_count_interim"] = np.nan
    n_coaches = counts.groupby(["team_id", "season"])["coach_id"].transform("size")
    single = counts[n_coaches == 1][["team_id", "season", "coach_id"]].rename(columns={"coach_id": "_c1"})
    m = mapped.reset_index().merge(single, on=["team_id", "season"], how="inner").set_index("index")
    mapped.loc[m.index, "coach_id"] = m["_c1"].to_numpy()
    mapped.loc[m.index, "_count_interim"] = 0.0

    multi = counts[n_coaches > 1]
    for (team, season), part in multi.groupby(["team_id", "season"]):
        idx = mapped.index[(mapped["team_id"] == team) & (mapped["season"] == season) & played]
        if len(idx) == 0:
            continue
        seq = mapped.loc[idx].sort_values(["start_time_utc", "game_id"])
        first_seen = {c: i for i, c in reversed(list(enumerate(seq["coach_id"].tolist())))}
        order = sorted(part.itertuples(index=False),
                       key=lambda r: (first_seen.get(r.coach_id, len(seq)), r.coach_id))
        assigned, pos = [], 0
        for j, r in enumerate(order):
            n = len(seq) - pos if j == len(order) - 1 else min(r.games, len(seq) - pos)
            assigned += [(r.coach_id, 0.0 if j == 0 else 1.0)] * max(n, 0)
            pos += max(n, 0)
        mapped.loc[seq.index, "coach_id"] = [a[0] for a in assigned]
        mapped.loc[seq.index, "_count_interim"] = [a[1] for a in assigned]
    return mapped


def coach_game_map(view, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team per game (past games and the slate) with the head coach.

    Columns: game_id, team_id, season, start_time_utc, coach_id (NaN when unknown),
    tenure_seasons, interim (1.0/0.0, NaN when unknown).
    """
    rows = _team_rows(view, games)
    ten = tenures(view)
    mapped = _date_rule(rows, ten)
    prior = season_counts(view)
    played_ids = set(view.table("team_game", columns=["game_id"])["game_id"])
    played = mapped["game_id"].isin(played_ids)
    mapped = _apply_counts(mapped, prior, played)

    # Tenure facts for the final coach: the tenure row of that coach at that team
    # covering the season (the one the date rule chose when it agrees).
    cover = ten[["coach_id", "team_id", "start_year", "end_year", "first_full", "is_interim", "in_season"]]
    j = mapped.reset_index()[["index", "coach_id", "team_id", "season"]].merge(cover, on=["coach_id", "team_id"])
    j = j[(j["start_year"] <= j["season"]) & (j["end_year"].isna() | (j["end_year"] >= j["season"] - 0))]
    j = j.sort_values(["index", "start_year"]).drop_duplicates("index", keep="last").set_index("index")
    first_full = j["first_full"].reindex(mapped.index)
    tenure = mapped["season"] - first_full
    in_debut = (j["in_season"] & (j["start_year"] == j["season"])).reindex(mapped.index)
    mapped["tenure_seasons"] = np.where(in_debut.fillna(False).to_numpy(dtype=bool), 0.0, tenure + 1)
    mapped.loc[mapped["tenure_seasons"] < 0, "tenure_seasons"] = 0.0

    flag = j["is_interim"].reindex(mapped.index).fillna(False).astype(bool) | in_debut.fillna(False).astype(bool)
    if "_count_interim" in mapped.columns:
        flag = flag | (mapped["_count_interim"] == 1.0)
    interim = flag.astype(float)
    mapped["interim"] = interim.where(mapped["coach_id"].notna())
    mapped["tenure_seasons"] = mapped["tenure_seasons"].where(mapped["coach_id"].notna())
    return mapped[MAP_COLUMNS].reset_index(drop=True)
