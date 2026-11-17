"""Travel, time zones, altitude, surface, dome and kickoff time.

Team stadiums come from teams_fbs (`location.*`); teams outside FBS get the venue of
most of their home games. The game venue comes from `venues`; for an on-campus game a
missing venue fact is taken from the home team's stadium, and a missing time zone from
the nearest venue or stadium with a known zone (within 400 km).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ._util import game_venues, haversine_km, local_hour, team_homes, team_rows, utc_offset_hours, venue_table

NIGHT_HOUR = 18.0
EARLY_BODY_HOUR = 11.0
SOURCES = ("schedule", "venues", "teams_fbs")


def _unknown_time(games: pd.DataFrame) -> np.ndarray:
    if "start_time_tbd" not in games.columns:
        return np.zeros(len(games), dtype=bool)
    return games["start_time_tbd"].fillna(False).astype(bool).to_numpy()


class Travel(FeatureGroup):
    name = "ctx_travel"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_travel_km", "Great-circle distance from the team's home stadium to the venue (km); "
                      "at neutral sites this is the team's distance from the venue", SOURCES, 2014),
        FeatureColumn("ctx_tz_shift_h", "Venue UTC offset minus the team's home UTC offset at kickoff (hours; "
                      "positive = the team travelled east)", SOURCES, 2014),
        FeatureColumn("ctx_elev_gain_m", "Venue elevation minus the team's home elevation (m)", SOURCES, 2014),
        FeatureColumn("ctx_surface_change", "1 if the venue surface (grass or turf) differs from the team's home "
                      "surface", SOURCES, 2014),
        FeatureColumn("ctx_dome_change", "1 if the venue is indoor and the team's home is not, or the reverse",
                      SOURCES, 2014),
        FeatureColumn("ctx_body_clock_hour", "Kickoff hour on the team's home clock (0-24; empty when the "
                      "kickoff time was TBD)", SOURCES, 2014),
        FeatureColumn("ctx_body_clock_early", f"1 if kickoff is before {EARLY_BODY_HOUR:.0f}:00 on the team's "
                      "home clock", SOURCES, 2014),
        FeatureColumn("ctx_at_own_stadium", "1 if the game is at the team's own home stadium", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        venues = venue_table(view)
        homes = team_homes(view, venues)
        gv = game_venues(games, venues, homes)
        rows = team_rows(games)
        rows["team_id"] = rows["team_id"].astype("float64")
        rows = rows.merge(gv.rename(columns=lambda c: c if c == "game_id" else f"v_{c}"), on="game_id", how="left")
        rows = rows.merge(homes.rename(columns=lambda c: c if c == "team_id" else f"t_{c}"), on="team_id", how="left")
        start = rows["start_time_utc"]
        v_off = utc_offset_hours(rows["v_tz"], start)
        t_off = utc_offset_hours(rows["t_tz"], start)
        tbd = np.concatenate([_unknown_time(games)] * 2)
        body = np.where(tbd, np.nan, local_hour(start, t_off))
        differ = lambda a, b: np.where(rows[a].isna() | rows[b].isna(), np.nan,
                                       (rows[a] != rows[b]).astype("float64"))
        own = np.where(rows["t_home_venue_id"].isna() | rows["v_venue_id"].isna(), np.nan,
                       (rows["t_home_venue_id"] == rows["v_venue_id"]).astype("float64"))
        out = pd.DataFrame({
            "game_id": rows["game_id"],
            "team_id": games["home_id"].tolist() + games["away_id"].tolist(),
            "ctx_travel_km": haversine_km(rows["t_lat"], rows["t_lon"], rows["v_lat"], rows["v_lon"]),
            "ctx_tz_shift_h": v_off - t_off,
            "ctx_elev_gain_m": (rows["v_elev"] - rows["t_elev"]).astype("float64"),
            "ctx_surface_change": differ("v_grass", "t_grass"),
            "ctx_dome_change": differ("v_dome", "t_dome"),
            "ctx_body_clock_hour": body,
            "ctx_body_clock_early": np.where(np.isnan(body), np.nan, (body < EARLY_BODY_HOUR).astype("float64")),
            "ctx_at_own_stadium": own,
        })
        return out.drop_duplicates(["game_id", "team_id"])


class Venue(FeatureGroup):
    name = "ctx_venue"
    prefix = "ctx_"
    level = "game"
    columns = [
        FeatureColumn("ctx_venue_elev_m", "Venue elevation (m)", SOURCES, 2014),
        FeatureColumn("ctx_venue_dome", "1 if the venue is indoor", SOURCES, 2014),
        FeatureColumn("ctx_venue_grass", "1 if the venue has natural grass", SOURCES, 2014),
        FeatureColumn("ctx_kickoff_local_hour", "Kickoff hour at the venue (0-24; empty when TBD)", SOURCES, 2014),
        FeatureColumn("ctx_night_game", f"1 if kickoff is at or after {NIGHT_HOUR:.0f}:00 local time", SOURCES, 2014),
        FeatureColumn("ctx_away_travel_km", "Distance the away team travels to the venue (km)", SOURCES, 2014),
        FeatureColumn("ctx_travel_km_diff", "Away team's travel minus home team's travel (km); at neutral sites "
                      "positive means the venue is closer to the home-listed team", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        venues = venue_table(view)
        homes = team_homes(view, venues)
        gv = game_venues(games, venues, homes).drop_duplicates("game_id")
        g = games[["game_id", "start_time_utc", "home_id", "away_id"]].reset_index(drop=True).merge(
            gv, on="game_id", how="left")
        off = utc_offset_hours(g["tz"], g["start_time_utc"])
        tbd = _unknown_time(games.drop_duplicates("game_id").reset_index(drop=True))
        hour = np.where(tbd, np.nan, local_hour(g["start_time_utc"], off))
        loc = homes.set_index("team_id")
        dist = {}
        for side in ("home_id", "away_id"):
            ids = g[side].astype("float64")
            dist[side] = haversine_km(ids.map(loc["lat"]), ids.map(loc["lon"]), g["lat"], g["lon"])
        return pd.DataFrame({
            "game_id": g["game_id"],
            "ctx_venue_elev_m": g["elev"].astype("float64"),
            "ctx_venue_dome": g["dome"].astype("float64"),
            "ctx_venue_grass": g["grass"].astype("float64"),
            "ctx_kickoff_local_hour": hour,
            "ctx_night_game": np.where(np.isnan(hour), np.nan, (hour >= NIGHT_HOUR).astype("float64")),
            "ctx_away_travel_km": dist["away_id"],
            "ctx_travel_km_diff": dist["away_id"] - dist["home_id"],
        }).drop_duplicates("game_id")
