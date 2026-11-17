"""Power rating: opponent-adjusted points ratings with the preseason projection as prior."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import points as P
from . import prior as PR
from .units import TeamIndex

# Chosen by scripts/strength_check.py on 2015-2025 (never 2026).
POWER = P.PointsParams(lam=4.0, half_life=16.0, lam_mu=40.0, lam_hfa=400.0)
# Residual SD of one team's points in one game around the model (2015-2025), used for
# the rating's standard error.
POINTS_SD = 11.7
# Logistic scale: P(win) = 1 / (1 + exp(-(net gap + home edge) / WIN_SCALE)).
WIN_SCALE = 8.5


@dataclass
class PowerState:
    idx: TeamIndex
    fit: P.PointsFit
    off0: np.ndarray | None
    def0: np.ndarray | None
    comp: pd.DataFrame
    obs: pd.DataFrame  # this season's visible team-game observations

    def lookup(self, team_ids) -> pd.DataFrame:
        """off, def, net (points, neutral field, against an average FBS team), games, prior values."""
        team_ids = np.asarray(team_ids, dtype=np.int64)
        pos = self.idx.index(team_ids)
        f = self.fit
        off = f.mu + f.off[pos]
        deff = f.mu + f.deff[pos]
        games = f.games[pos]
        own = np.isin(team_ids, self.idx.fbs_ids)
        # Lower-division teams share the pooled rating; their own game count is used.
        if (~own).any() and len(self.obs):
            counts = self.obs.groupby("offense_id").size()
            games = np.where(own, games, counts.reindex(team_ids).fillna(0.0).to_numpy())
        out = pd.DataFrame({"team_id": team_ids, "off": off, "def": deff, "net": off - deff, "games": games})
        if self.off0 is not None:
            out["prior_off"] = f.mu + self.off0[pos]
            out["prior_def"] = f.mu + self.def0[pos]
        else:
            out["prior_off"] = out["prior_def"] = np.nan
        out["prior_net"] = out["prior_off"] - out["prior_def"]
        return out


def power_state(view, params: P.PointsParams = POWER, version: int = 1) -> PowerState:
    """Power ratings at the view's cutoff. `version` 2 takes the player-based preseason
    projection (prior.COMPONENTS_V2) as the prior; everything else is the same."""
    season = view.season
    teams = view.table("teams", seasons=season)
    idx = TeamIndex.from_teams(teams)
    tg = view.table("team_game", columns=P.TG_COLUMNS, seasons=season)
    obs = P.observations(tg)
    comp = PR.components(view, season) if version == 1 else PR.components_v2(view, season)
    prev, _ = PR.previous_final(view, season)
    off0 = def0 = None
    mu0, hfa0 = None, 0.0
    if prev is not None:
        o, d = PR.projection(comp.reindex(idx.fbs_ids), version)
        off0 = np.append(o, comp["pool_off"].iloc[0] if len(comp) else 0.0)
        def0 = np.append(d, comp["pool_def"].iloc[0] if len(comp) else 0.0)
        mu0, hfa0 = prev.mu, prev.hfa
    fit = P.fit(obs, idx, int(view.cutoff.value), params, off0, def0, mu0, hfa0)
    return PowerState(idx, fit, off0, def0, comp, obs)


def win_prob(gap: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(gap, dtype=np.float64) / WIN_SCALE))
