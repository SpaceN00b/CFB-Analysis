"""Transfer portal entries from a team since its last game (a proxy for bowl opt-outs).

Only `transferDate` and `origin` are point-in-time. The portal data starts on
2020-08-01 (entries for season 2021); before that the values are unknown, not zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from ._util import empty_output, _at, history_with_targets

PORTAL_START = pd.Timestamp("2020-08-01T00:00:00Z")
RATED_STARS = 3
SOURCES = ("transfer_portal", "teams", "schedule")


class Portal(FeatureGroup):
    name = "ctx_portal"
    prefix = "ctx_"
    level = "team"
    columns = [
        FeatureColumn("ctx_portal_out_since_last", "Players from the team who entered the portal after the "
                      "kickoff of its previous game and by the cutoff", SOURCES, 2020),
        FeatureColumn("ctx_portal_out_rated_since_last", f"Same, players rated {RATED_STARS} stars or more",
                      SOURCES, 2020),
        FeatureColumn("ctx_portal_out_since_regular", "Players from the team who entered the portal after its "
                      "last regular-season game and by the cutoff (postseason only)", SOURCES, 2020),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        if not view.has("transfer_portal"):  # absent from this warehouse: unknown
            return empty_output(self, games)
        tp = view.table("transfer_portal", columns=["origin", "transferDate", "stars"])
        teams = view.table("teams", columns=["season", "team_id", "team"])
        names = teams.sort_values("season").drop_duplicates("team", keep="last").set_index("team")["team_id"]
        tp = tp.assign(team_id=tp["origin"].map(names),
                       when=pd.to_datetime(tp["transferDate"], utc=True, errors="coerce", format="ISO8601"),
                       rated=pd.to_numeric(tp["stars"], errors="coerce").ge(RATED_STARS).astype("float64"))
        tp = tp.dropna(subset=["team_id", "when"])
        tp["team_id"] = tp["team_id"].astype("int64")
        tp = tp[tp["when"] <= view.cutoff]

        rows = history_with_targets(view, games, seasons=games["season"].unique())
        tgt = rows[rows["_target"]].reset_index(drop=True)
        tgt["_k"] = np.arange(len(tgt))
        pair = tgt[["_k", "team_id", "season", "start_time_utc"]].merge(
            rows[["team_id", "season", "season_type", "start_time_utc"]].rename(
                columns={"start_time_utc": "t", "season_type": "st"}),
            on=["team_id", "season"], how="left")
        before = pair[pair["t"] < pair["start_time_utc"]]
        prev = _at(before.groupby("_k")["t"].max(), tgt["_k"])
        last_reg = _at(before[before["st"] == "regular"].groupby("_k")["t"].max(), tgt["_k"])

        def count(since: pd.Series, rated: bool) -> np.ndarray:
            known = since.notna() & (since >= PORTAL_START)
            base = pd.DataFrame({"_k": tgt["_k"], "team_id": tgt["team_id"], "since": since})[known]
            hits = base.merge(tp[["team_id", "when", "rated"]], on="team_id", how="inner")
            hits = hits[hits["when"] > hits["since"]]
            n = hits.groupby("_k")["rated"].sum() if rated else hits.groupby("_k").size()
            return np.where(known, _at(n, tgt["_k"]).fillna(0.0), np.nan).astype("float64")

        post = (tgt["season_type"] != "regular").to_numpy()
        out = pd.DataFrame({
            "game_id": tgt["game_id"],
            "team_id": tgt["team_id"],
            "ctx_portal_out_since_last": count(prev, False),
            "ctx_portal_out_rated_since_last": count(prev, True),
            "ctx_portal_out_since_regular": np.where(post, count(last_reg, False), np.nan),
        })
        return out.drop_duplicates(["game_id", "team_id"])
