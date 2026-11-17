"""Opponent-adjusted unit ratings (the `unit_ratings` table).

Per team, cutoff and unit (off/def x rush/pass/all) three ratings on the scale of the
placeholder: EPA per play, success rate, EPA per successful play. Method:

1. Scrimmage plays outside garbage time (the shared `pbp` flags), summed per game,
   offense and defense.
2. A ridge fit per unit and metric (see `ridge.fit`): offense value + defense value
   + league mean + home field, weighted by plays and by recency (half-life in weeks).
   Teams below FBS in the season are pooled into one team.
3. The prior pulled toward is last season's final fit of the same model, shrunk
   toward the league mean by `rho`. With no games played a team sits at its prior.

Everything is built from rows visible in the view: the cutoff season's games so far
and the whole previous season.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core.pbp import unit_sums
from . import ridge

UNITS = ("off_rush", "off_pass", "def_rush", "def_pass", "off_all", "def_all")
COLUMNS = ["team_id", "cutoff_utc", "unit", "epa_adj", "success_adj", "explosive_adj", "n_plays"]
PBP_COLUMNS = ["game_id", "offense_id", "defense_id", "is_scrimmage", "is_rush", "is_pass", "ppa",
               "success", "garbage_time"]
POOL = -1  # team id used for the pooled lower-division team
DAY_NS = 86_400 * 10**9


@dataclass(frozen=True)
class MetricParams:
    lam: float        # prior strength in plays (successes for explosiveness)
    half_life: float  # weeks; inf = no decay
    rho: float        # share of last season's deviation kept in the prior


@dataclass(frozen=True)
class UnitParams:
    # Chosen by scripts/strength_tune_units.py on 2015-2025 (docs/strength.md).
    epa: MetricParams = MetricParams(lam=300.0, half_life=np.inf, rho=0.8)
    success: MetricParams = MetricParams(lam=300.0, half_life=np.inf, rho=0.8)
    explosive: MetricParams = MetricParams(lam=300.0, half_life=np.inf, rho=0.7)
    lam_mu: float = 3000.0
    lam_hfa: float = 20000.0
    metrics: tuple = field(default=("epa", "success", "explosive"))


DEFAULT = UnitParams()

# metric -> (numerator column, weight column) in the per-game sums
_METRIC_SOURCES = {"epa": ("epa_sum", "plays"), "success": ("successes", "plays"),
                   "explosive": ("expl_sum", "successes")}


def observations(pbp: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Per (game, offense, defense): additive sums of non-garbage scrimmage plays,
    with season, kickoff time and home sign h (+1 offense at home, -1 defense at home,
    0 neutral)."""
    plays = pbp[~pbp["garbage_time"].to_numpy()]
    sums = unit_sums(plays)
    if sums.empty:
        cols = ["game_id", "offense_id", "defense_id", "season", "start_ns", "h"]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols + list(sums.columns[3:])})
    sched = schedule[["game_id", "season", "start_time_utc", "neutral", "home_id"]]
    out = sums.merge(sched, on="game_id", how="inner")
    home = (out["offense_id"] == out["home_id"]).to_numpy()
    neutral = out["neutral"].fillna(False).astype(bool).to_numpy()
    out["h"] = np.where(neutral, 0.0, np.where(home, 1.0, -1.0))
    out["start_ns"] = out["start_time_utc"].astype("int64")
    return out.drop(columns=["start_time_utc", "neutral", "home_id"])


@dataclass
class TeamIndex:
    """FBS teams of one season get their own index; everyone else shares the last one."""
    fbs_ids: np.ndarray

    @classmethod
    def from_teams(cls, teams: pd.DataFrame) -> "TeamIndex":
        ids = teams.loc[teams["fbs"].fillna(False).astype(bool), "team_id"].drop_duplicates()
        return cls(np.sort(ids.to_numpy(dtype=np.int64)))

    @property
    def n(self) -> int:
        return len(self.fbs_ids) + 1

    def index(self, team_ids) -> np.ndarray:
        team_ids = np.asarray(team_ids, dtype=np.int64)
        if not len(self.fbs_ids):
            return np.zeros(len(team_ids), dtype=np.int64)
        pos_c = np.minimum(np.searchsorted(self.fbs_ids, team_ids), len(self.fbs_ids) - 1)
        hit = self.fbs_ids[pos_c] == team_ids
        return np.where(hit, pos_c, len(self.fbs_ids))


def _decay(start_ns: np.ndarray, now_ns: int, half_life: float) -> np.ndarray:
    if not np.isfinite(half_life):
        return np.ones(len(start_ns))
    age_weeks = np.maximum(now_ns - start_ns, 0) / (7 * DAY_NS)
    return 0.5 ** (age_weeks / half_life)


@dataclass
class SideFit:
    """Fitted offense and defense values for one unit kind (all, rush, pass) and metric."""
    mu: float
    hfa: float
    off: np.ndarray
    deff: np.ndarray


def fit_metric(obs: pd.DataFrame, idx: TeamIndex, kind: str, metric: str, p: MetricParams, now_ns: int,
               prior: SideFit | None, prior_idx: TeamIndex | None, lam_mu: float, lam_hfa: float) -> SideFit:
    num_col, w_col = _METRIC_SOURCES[metric]
    num = obs[f"{kind}_{num_col}"].to_numpy(dtype=np.float64)
    w = obs[f"{kind}_{w_col}"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(w > 0, num / w, np.nan)
    w = w * _decay(obs["start_ns"].to_numpy(dtype=np.int64), now_ns, p.half_life)
    o = idx.index(obs["offense_id"].to_numpy())
    d = idx.index(obs["defense_id"].to_numpy())
    off0 = def0 = None
    mu0, hfa0, lmu = None, 0.0, 0.0
    if prior is not None and np.isfinite(prior.mu):
        off0, def0 = carry_prior(prior, prior_idx, idx, p.rho)
        mu0, hfa0, lmu = prior.mu, prior.hfa, lam_mu
    f = ridge.fit(idx.n, o, d, obs["h"].to_numpy(dtype=np.float64), y, w, p.lam, off0, def0,
                  mu0=mu0, lam_mu=lmu, hfa0=hfa0, lam_hfa=lam_hfa)
    return SideFit(f.mu, f.hfa, f.off, f.deff)


def carry_prior(prior: SideFit, prior_idx: TeamIndex, idx: TeamIndex, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Last season's deviations moved onto this season's team index, times rho.
    A team that was not FBS last season starts from last season's pooled value."""
    ids = np.append(idx.fbs_ids, POOL)
    src = prior_idx.index(ids)
    src[-1] = prior_idx.n - 1
    return rho * prior.off[src], rho * prior.deff[src]


def season_fits(obs: pd.DataFrame, idx: TeamIndex, now_ns: int, params: UnitParams,
                prior: dict | None = None, prior_idx: TeamIndex | None = None) -> dict:
    """{(kind, metric): SideFit} for kind in all, rush, pass."""
    out = {}
    for kind in ("all", "rush", "pass"):
        for metric in params.metrics:
            mp = getattr(params, metric)
            pf = prior.get((kind, metric)) if prior else None
            out[(kind, metric)] = fit_metric(obs, idx, kind, metric, mp, now_ns, pf, prior_idx,
                                             params.lam_mu, params.lam_hfa)
    return out


def rate(obs_cur: pd.DataFrame, obs_prev: pd.DataFrame | None, teams_cur: pd.DataFrame,
         teams_prev: pd.DataFrame | None, now_ns: int, params: UnitParams = DEFAULT):
    """Fits for the current season at `now_ns`, with last season's final fit as prior.

    Returns (fits, idx). `obs_prev` holds the whole previous season (or None)."""
    idx = TeamIndex.from_teams(teams_cur)
    prior = prior_idx = None
    if obs_prev is not None and len(obs_prev) and teams_prev is not None:
        prior_idx = TeamIndex.from_teams(teams_prev)
        prev_now = int(obs_prev["start_ns"].max()) + DAY_NS
        prior = season_fits(obs_prev, prior_idx, prev_now, params)
    return season_fits(obs_cur, idx, now_ns, params, prior, prior_idx), idx


def ratings_frame(fits: dict, idx: TeamIndex, teams_cur: pd.DataFrame, obs_cur: pd.DataFrame,
                  cutoff: pd.Timestamp) -> pd.DataFrame:
    """The unit_ratings contract frame for every team of the season."""
    team_ids = np.sort(teams_cur["team_id"].drop_duplicates().to_numpy(dtype=np.int64))
    pos = idx.index(team_ids)
    parts = []
    for unit in UNITS:
        side, kind = unit.split("_")
        vals = {}
        for metric in ("epa", "success", "explosive"):
            f = fits.get((kind, metric))
            if f is None:
                vals[metric] = np.full(len(team_ids), np.nan)
                continue
            dev = f.off if side == "off" else f.deff
            vals[metric] = f.mu + dev[pos]
        key = "offense_id" if side == "off" else "defense_id"
        counts = obs_cur.groupby(key)[f"{kind}_plays"].sum() if len(obs_cur) else pd.Series(dtype="float64")
        n_plays = counts.reindex(team_ids).fillna(0.0).to_numpy()
        parts.append(pd.DataFrame({
            "team_id": team_ids, "unit": unit, "epa_adj": vals["epa"], "success_adj": vals["success"],
            "explosive_adj": vals["explosive"], "n_plays": n_plays.astype("int64"),
        }))
    out = pd.concat(parts, ignore_index=True)
    out["cutoff_utc"] = cutoff
    out["cutoff_utc"] = pd.to_datetime(out["cutoff_utc"], utc=True).dt.as_unit("ns")
    return out[COLUMNS]


def view_inputs(view):
    """Observations and team lists for the view's season and the one before."""
    season = view.season
    sched = view.schedule()
    pbp = view.table("pbp", columns=PBP_COLUMNS + ["season"], seasons=[season - 1, season])
    obs = observations(pbp.drop(columns="season"), sched)
    teams = view.table("teams", seasons=[season - 1, season])
    obs_cur = obs[obs["season"] == season].reset_index(drop=True)
    obs_prev = obs[obs["season"] == season - 1].reset_index(drop=True)
    teams_cur = teams[teams["season"] == season]
    teams_prev = teams[teams["season"] == season - 1]
    return obs_cur, (obs_prev if len(obs_prev) else None), teams_cur, (teams_prev if len(teams_prev) else None)


def unit_ratings(view, params: UnitParams = DEFAULT) -> pd.DataFrame:
    obs_cur, obs_prev, teams_cur, teams_prev = view_inputs(view)
    fits, idx = rate(obs_cur, obs_prev, teams_cur, teams_prev, int(view.cutoff.value), params)
    return ratings_frame(fits, idx, teams_cur, obs_cur, view.cutoff)
