"""Helpers shared by the context groups: team rows, locations, time zones, distances.

Everything here takes frames that came through the AsOfView; nothing reads data itself.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
DAY = pd.Timedelta(days=1)

TEAM_ROW_COLUMNS = [
    "game_id", "season", "season_type", "week", "start_time_utc", "start_time_tbd", "neutral",
    "venue_id", "team_id", "opp_id", "is_home",
]


def team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home side first): the schedule from each team's point of view."""
    keep = [c for c in ("game_id", "season", "season_type", "week", "start_time_utc", "start_time_tbd",
                        "neutral", "venue_id") if c in games.columns]
    base = games[keep].reset_index(drop=True)
    home = base.assign(team_id=games["home_id"].to_numpy(), opp_id=games["away_id"].to_numpy(), is_home=True)
    away = base.assign(team_id=games["away_id"].to_numpy(), opp_id=games["home_id"].to_numpy(), is_home=False)
    out = pd.concat([home, away], ignore_index=True)
    for col in TEAM_ROW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["neutral"] = out["neutral"].fillna(False).astype(bool)
    return out[TEAM_ROW_COLUMNS]


def history_with_targets(view, games: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """Team rows of the visible schedule plus the target games (which may be in no table).

    A target game replaces a schedule row with the same game id. `_target` marks the
    rows to describe.
    """
    sched = view.schedule()
    if seasons is not None:
        sched = sched[sched["season"].isin(list(seasons))]
    hist = team_rows(sched[~sched["game_id"].isin(games["game_id"])])
    hist["_target"] = False
    tgt = team_rows(games)
    tgt["_target"] = True
    return pd.concat([hist, tgt], ignore_index=True)


def _at(values: pd.Series, keys: pd.Series) -> pd.Series:
    """values[key] for each key, aligned with `keys` (missing keys give empty values)."""
    out = values.reindex(keys.to_numpy())
    out.index = keys.index
    return out


def site(rows: pd.DataFrame) -> np.ndarray:
    """+1 home, -1 away, 0 neutral."""
    return np.where(rows["neutral"].to_numpy(dtype=bool), 0.0, np.where(rows["is_home"].to_numpy(dtype=bool), 1.0, -1.0))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def utc_offset_hours(tz: pd.Series, when: pd.Series) -> np.ndarray:
    """UTC offset in hours of time zone `tz` at instant `when` (DST aware). NaN if unknown."""
    tz = pd.Series(tz).reset_index(drop=True)
    when = pd.Series(when).reset_index(drop=True)
    out = np.full(len(tz), np.nan)
    ok = tz.notna() & when.notna()
    if not ok.any():
        return out
    pairs = pd.DataFrame({"tz": tz[ok].astype(str), "day": when[ok].dt.floor("h")})
    cache: dict = {}
    for i, (name, ts) in zip(pairs.index, pairs.itertuples(index=False, name=None)):
        key = (name, ts)
        if key not in cache:
            try:
                cache[key] = ts.tz_convert(ZoneInfo(name)).utcoffset().total_seconds() / 3600.0
            except (ZoneInfoNotFoundError, ValueError, KeyError):
                cache[key] = np.nan
        out[i] = cache[key]
    return out


def _num(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").astype("float64")


def _flag(series) -> pd.Series:
    """Booleans that may be missing, as float 1/0/NaN."""
    s = pd.Series(series)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    known = s.notna()
    if known.any():
        out[known] = s[known].map(lambda v: 1.0 if bool(v) else 0.0).astype("float64")
    return out


LOC_COLUMNS = ["lat", "lon", "elev", "tz", "grass", "dome", "capacity"]


def venue_table(view) -> pd.DataFrame:
    """venue_id with lat, lon, elev (m), tz, grass, dome, capacity."""
    v = view.table("venues")
    out = pd.DataFrame({
        "venue_id": _num(v["id"]).to_numpy(),
        "lat": _num(v.get("latitude")).to_numpy(),
        "lon": _num(v.get("longitude")).to_numpy(),
        "elev": _num(v.get("elevation")).to_numpy(),
        "tz": v["timezone"].astype("object").where(v["timezone"].notna(), None).to_numpy(),
        "grass": _flag(v.get("grass")).to_numpy(),
        "dome": _flag(v.get("dome")).to_numpy(),
        "capacity": _num(v.get("capacity")).to_numpy(),
    })
    out.loc[out["capacity"] <= 0, "capacity"] = np.nan
    return out.dropna(subset=["venue_id"]).drop_duplicates("venue_id", keep="last").reset_index(drop=True)


def fbs_homes(view) -> pd.DataFrame:
    """Home stadium of each FBS team from teams_fbs, latest season up to the view's season."""
    tf = view.table("teams_fbs")
    tf = tf[tf["season"] <= view.season].sort_values("season")
    out = pd.DataFrame({
        "team_id": _num(tf["id"]).to_numpy(),
        "home_venue_id": _num(tf.get("location.id")).to_numpy(),
        "lat": _num(tf.get("location.latitude")).to_numpy(),
        "lon": _num(tf.get("location.longitude")).to_numpy(),
        "elev": _num(tf.get("location.elevation")).to_numpy(),
        "tz": tf["location.timezone"].astype("object").where(tf["location.timezone"].notna(), None).to_numpy(),
        "grass": _flag(tf.get("location.grass")).to_numpy(),
        "dome": _flag(tf.get("location.dome")).to_numpy(),
        "capacity": _num(tf.get("location.capacity")).to_numpy(),
    })
    out.loc[out["capacity"] <= 0, "capacity"] = np.nan
    return out.drop_duplicates("team_id", keep="last").reset_index(drop=True)


def team_homes(view, venues: pd.DataFrame | None = None) -> pd.DataFrame:
    """Home location for every team: teams_fbs for FBS teams; for others the venue of most
    of their non-neutral home games in their latest season with one (visible schedule)."""
    venues = venue_table(view) if venues is None else venues
    fbs = fbs_homes(view)
    sched = view.schedule()
    home = sched[~sched["neutral"].fillna(False).astype(bool) & sched["venue_id"].notna()]
    home = home[~home["home_id"].isin(fbs["team_id"])]
    if len(home):
        last = home.groupby("home_id")["season"].transform("max")
        home = home[home["season"] == last]
        counts = home.groupby(["home_id", "venue_id"]).size().rename("n").reset_index()
        counts = counts.sort_values(["home_id", "n", "venue_id"], ascending=[True, False, True])
        pick = counts.drop_duplicates("home_id")[["home_id", "venue_id"]]
        other = pick.rename(columns={"home_id": "team_id", "venue_id": "home_venue_id"}).merge(
            venues.rename(columns={"venue_id": "home_venue_id"}), on="home_venue_id", how="left")
        other["team_id"] = other["team_id"].astype("float64")
        other["home_venue_id"] = other["home_venue_id"].astype("float64")
        fbs = pd.concat([fbs, other[fbs.columns]], ignore_index=True)
    return fbs


def tz_reference(venues: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    ref = pd.concat([venues[["lat", "lon", "tz"]], homes[["lat", "lon", "tz"]]], ignore_index=True)
    return ref.dropna(subset=["lat", "lon", "tz"]).reset_index(drop=True)


def fill_tz(frame: pd.DataFrame, ref: pd.DataFrame, max_km: float = 400.0) -> pd.DataFrame:
    """Fill a missing `tz` with the zone of the nearest reference point within max_km."""
    need = frame["tz"].isna() & frame["lat"].notna() & frame["lon"].notna()
    if not need.any() or ref.empty:
        return frame
    out = frame.copy()
    rlat, rlon, rtz = ref["lat"].to_numpy(), ref["lon"].to_numpy(), ref["tz"].to_numpy()
    for i in np.flatnonzero(need.to_numpy()):
        d = haversine_km(out["lat"].iat[i], out["lon"].iat[i], rlat, rlon)
        j = int(np.argmin(d))
        if d[j] <= max_km:
            out.iat[i, out.columns.get_loc("tz")] = rtz[j]
    return out


def game_venues(games: pd.DataFrame, venues: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    """One row per game: where it is played. Missing venue facts of a non-neutral game come
    from the home team's stadium; a missing time zone from the nearest known location."""
    g = games[["game_id", "venue_id", "home_id", "neutral"]].copy()
    g["venue_id"] = _num(g["venue_id"]).to_numpy()
    g = g.merge(venues, on="venue_id", how="left")
    h = homes.rename(columns={c: f"h_{c}" for c in homes.columns}).rename(columns={"h_team_id": "home_id"})
    g["home_id"] = g["home_id"].astype("float64")
    g = g.merge(h, on="home_id", how="left")
    on_campus = ~g["neutral"].fillna(False).astype(bool)
    for col in LOC_COLUMNS:
        g[col] = g[col].where(g[col].notna() | ~on_campus, g[f"h_{col}"])
    g = fill_tz(g, tz_reference(venues, homes))
    return g[["game_id", "venue_id", *LOC_COLUMNS]]


def local_hour(start: pd.Series, offset: np.ndarray) -> np.ndarray:
    utc = start.dt.hour.to_numpy() + start.dt.minute.to_numpy() / 60.0
    return np.mod(utc + offset, 24.0)
