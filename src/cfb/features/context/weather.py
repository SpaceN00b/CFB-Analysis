"""Game weather from CFBD (`weather`). Played games hold observed values, upcoming games
forecasts. Stored for phase 2 (totals); units as CFBD reports them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup

SOURCES = ("weather",)


class Weather(FeatureGroup):
    name = "ctx_weather"
    prefix = "ctx_"
    level = "game"
    columns = [
        FeatureColumn("ctx_wx_indoors", "1 if the game is played indoors", SOURCES, 2014),
        FeatureColumn("ctx_wx_temp_f", "Temperature at kickoff (deg F)", SOURCES, 2014),
        FeatureColumn("ctx_wx_wind_mph", "Wind speed (mph)", SOURCES, 2014),
        FeatureColumn("ctx_wx_precip_in", "Precipitation (inches)", SOURCES, 2014),
        FeatureColumn("ctx_wx_snow_in", "Snowfall (inches)", SOURCES, 2014),
        FeatureColumn("ctx_wx_humidity", "Relative humidity (%)", SOURCES, 2014),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        w = view.table("weather", columns=["id", "gameIndoors", "temperature", "windSpeed", "precipitation",
                                           "snowfall", "humidity"])
        w = w.drop_duplicates("id", keep="last")
        g = games[["game_id"]].drop_duplicates().merge(w, left_on="game_id", right_on="id", how="left")
        num = lambda c: pd.to_numeric(g[c], errors="coerce").astype("float64").to_numpy()
        indoors = g["gameIndoors"]
        return pd.DataFrame({
            "game_id": g["game_id"].to_numpy(),
            "ctx_wx_indoors": np.where(indoors.isna(), np.nan, indoors.fillna(False).astype(bool).astype("float64")),
            "ctx_wx_temp_f": num("temperature"),
            "ctx_wx_wind_mph": num("windSpeed"),
            "ctx_wx_precip_in": num("precipitation"),
            "ctx_wx_snow_in": num("snowfall"),
            "ctx_wx_humidity": num("humidity"),
        })
