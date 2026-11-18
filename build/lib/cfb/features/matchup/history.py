"""matchup_history: how a team has done against styles like this opponent's.

For every past game of the team in the current and previous season against an FBS
opponent, a residual in EPA per play (scrimmage plays, garbage time removed):

    defense   expected allowed - allowed       expected = team's unit rating + opponent's
    offense   gained - expected                           unit rating - league average,
                                                          all from unit_ratings (off_all,
                                                          def_all) at that game's cutoff

so the expectation is point in time and the opponent adjustment is W3's. Opponents are
put in thirds of the FBS teams of that season by style: offenses by neutral run rate
(pass-heavy, balanced, run-heavy) and seconds per play (fast, medium, slow); defenses by
EPA allowed on rushes minus passes, each against the league (run-stout, even,
pass-stout) and by havoc created (low, mid, high). The current season's styles use the
same prior-season blend as the style profile.

Each value is the mean residual against that kind of opponent shrunk toward zero:
sum / (n + HISTORY_SHRINK). Positive is good for the row's team. A team with no rated
past games against FBS opponents gets empty values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup
from .stats import PRIOR_GAMES, blended_profiles, fbs_teams, game_stats, season_profiles

# Pseudo-games of zero residual added to every class. Chosen on seasons through 2025
# (docs/matchup.md).
HISTORY_SHRINK = 150.0

FAMILIES = {
    # family: (side whose residual is used, opponent profile column, low label, mid, high label)
    "run": ("def", "off_run_rate_neutral", "pass_heavy", "balanced", "run_heavy"),
    "tempo": ("def", "off_sec_per_play", "fast", "medium", "slow"),
    "front": ("off", "front_tilt", "run_stout", "even", "pass_stout"),
    "havoc": ("off", "def_havoc", "low_havoc", "mid_havoc", "high_havoc"),
}

_SOURCES = ("pbp", "team_game")
_GENERIC = [
    ("def", "run", "run_heavy", "the team's defense against run-heavy offenses", _SOURCES),
    ("def", "run", "pass_heavy", "the team's defense against pass-heavy offenses", _SOURCES),
    ("def", "tempo", "fast", "the team's defense against fast offenses (fewest seconds per play)",
     _SOURCES + ("drives",)),
    ("off", "front", "run_stout", "the team's offense against defenses relatively strong against the run",
     _SOURCES),
    ("off", "front", "pass_stout", "the team's offense against defenses relatively strong against the pass",
     _SOURCES),
    ("off", "havoc", "high_havoc", "the team's offense against high-havoc defenses",
     _SOURCES + ("stats_game_havoc",)),
]
_APPLIED = [
    ("mat_hist_def_vs_opp_run_class", "def", "run",
     "Defense residual against offenses in the same run-rate third as this opponent", _SOURCES),
    ("mat_hist_def_vs_opp_tempo_class", "def", "tempo",
     "Defense residual against offenses in the same tempo third as this opponent", _SOURCES + ("drives",)),
    ("mat_hist_off_vs_opp_front_class", "off", "front",
     "Offense residual against defenses in the same run/pass-strength third as this opponent", _SOURCES),
    ("mat_hist_off_vs_opp_havoc_class", "off", "havoc",
     "Offense residual against defenses in the same havoc third as this opponent", _SOURCES + ("stats_game_havoc",)),
]


def _columns() -> list[FeatureColumn]:
    cols = []
    for side, fam, label, what, sources in _GENERIC:
        cols.append(FeatureColumn(
            f"mat_hist_{side}_vs_{label}",
            f"Shrunk mean EPA/play residual of {what}, this and last season; positive favors the team",
            sources, 2014))
    for side, fam, label, what, sources in _GENERIC:
        cols.append(FeatureColumn(f"mat_hist_n_{side}_vs_{label}",
                                  f"Games behind mat_hist_{side}_vs_{label}", sources, 2014))
    for name, side, fam, what, sources in _APPLIED:
        cols.append(FeatureColumn(name, what + " (shrunk, EPA/play; positive favors the team)", sources, 2014))
    return cols


def _with_tilt(prof: pd.DataFrame, fbs) -> pd.DataFrame:
    prof = prof.copy()
    sub = prof.reindex(fbs)
    prof["front_tilt"] = ((prof["def_rush_epa"] - sub["def_rush_epa"].mean())
                          - (prof["def_pass_epa"] - sub["def_pass_epa"].mean()))
    return prof


def _classes(prof: pd.DataFrame, fbs) -> pd.DataFrame:
    """Third (0 low, 1 mid, 2 high) of each family per team, from FBS thresholds."""
    out = pd.DataFrame(index=prof.index)
    sub = prof.reindex(fbs)
    for fam, (_, col, *_labels) in FAMILIES.items():
        vals = sub[col].dropna()
        x = prof[col].to_numpy(dtype=float)
        if len(vals) < 3:
            out[fam] = np.nan
            continue
        lo, hi = np.quantile(vals.to_numpy(dtype=float), [1 / 3, 2 / 3])
        cls = np.where(x <= lo, 0.0, np.where(x >= hi, 2.0, 1.0))
        out[fam] = np.where(np.isfinite(x), cls, np.nan)
    return out


def game_residuals(view, stats: pd.DataFrame) -> pd.DataFrame:
    """Per visible team-game against an FBS opponent: res_off and res_def in EPA per play,
    against the expectation from unit_ratings at that game's cutoff. Empty when either
    team had no rated plays at the time."""
    cols = ["season", "game_id", "team_id", "opp_id", "res_off", "res_def"]
    g = stats[stats["opp_fbs"].fillna(False).astype(bool).to_numpy()]
    if g.empty:
        return pd.DataFrame(columns=cols)
    sched = view.schedule()[["game_id", "cutoff_utc"]]
    g = g.merge(sched, on="game_id", how="inner")
    ur = view.table("unit_ratings", columns=["team_id", "cutoff_utc", "unit", "epa_adj", "n_plays"])
    ur = ur[ur["unit"].isin(["off_all", "def_all"]).to_numpy() & ur["cutoff_utc"].isin(g["cutoff_utc"].unique()).to_numpy()
            & (ur["n_plays"] > 0).to_numpy()]
    wide = ur.pivot(index=["cutoff_utc", "team_id"], columns="unit", values="epa_adj")
    wide = wide.reindex(columns=["off_all", "def_all"]).astype(float)
    # league average per cutoff over the FBS teams of that season
    season_of = g.drop_duplicates("cutoff_utc").set_index("cutoff_utc")["season"]
    teams = view.table("teams", columns=["season", "team_id", "fbs"])
    fbs = set(map(tuple, teams.loc[teams["fbs"].fillna(False).astype(bool), ["season", "team_id"]].to_numpy()))
    flat = wide.reset_index()
    flat["season"] = season_of.reindex(flat["cutoff_utc"]).to_numpy()
    is_fbs = np.array([(s, t) in fbs for s, t in zip(flat["season"], flat["team_id"])], dtype=bool)
    lg = flat[is_fbs].groupby("cutoff_utc")["off_all"].mean()

    def rating(team_col: str, unit: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([g["cutoff_utc"], g[team_col]])
        return wide[unit].reindex(idx).to_numpy(dtype=float)

    league = lg.reindex(g["cutoff_utc"]).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        off_g = np.where(g["off_scrim"] > 0, g["off_epa"] / g["off_scrim"], np.nan)
        def_g = np.where(g["def_scrim"] > 0, g["def_epa"] / g["def_scrim"], np.nan)
    out = g[["season", "game_id", "team_id", "opp_id"]].copy()
    out["res_off"] = off_g - (rating("team_id", "off_all") + rating("opp_id", "def_all") - league)
    out["res_def"] = (rating("team_id", "def_all") + rating("opp_id", "off_all") - league) - def_g
    return out.reset_index(drop=True)


class MatchupHistory(FeatureGroup):
    name = "matchup_history"
    prefix = "mat_"
    level = "team"
    mode = "per_cutoff"
    columns = _columns()

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        season = view.season
        stats = game_stats(view, [season - 1, season])
        rows = pd.concat([
            pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["home_id"].to_numpy(),
                          "opp_id": games["away_id"].to_numpy()}),
            pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["away_id"].to_numpy(),
                          "opp_id": games["home_id"].to_numpy()}),
        ], ignore_index=True).drop_duplicates(["game_id", "team_id"])

        # season style profiles and opponent classes for the previous and current season
        per_season = {}
        for s in (season - 1, season):
            fbs = fbs_teams(view, s)
            if s == season:
                prof = blended_profiles(stats, s, PRIOR_GAMES)
            else:
                prof = season_profiles(stats[stats["season"] == s])
                prof = prof.droplevel("season") if not prof.empty else prof
            if prof.empty:
                continue
            per_season[s] = _classes(_with_tilt(prof, fbs), fbs)

        resid_parts = []
        res = game_residuals(view, stats)
        for s_, cls in per_season.items():
            part = res[res["season"] == s_].copy()
            for fam in FAMILIES:
                part[fam] = cls[fam].reindex(part["opp_id"]).to_numpy(dtype=float)
            resid_parts.append(part)

        out = rows[["game_id", "team_id"]].copy()
        n_rows = len(rows)
        resid = pd.concat(resid_parts, ignore_index=True) if resid_parts else None
        if resid is not None:
            resid = resid[np.isfinite(resid["res_off"]) | np.isfinite(resid["res_def"])]
            resid = resid if len(resid) else None
        has_history = (np.isin(rows["team_id"].to_numpy(), resid["team_id"].unique())
                       if resid is not None else np.zeros(n_rows, dtype=bool))

        def shrunk(fam: str, side: str):
            """Per team and class: (shrunk mean, n) frames indexed by team_id, columns 0, 1, 2."""
            if resid is None:
                return None, None
            r = resid[np.isfinite(resid[fam]) & np.isfinite(resid[f"res_{side}"])]
            if r.empty:
                return None, None
            agg = r.groupby(["team_id", fam])[f"res_{side}"].agg(["sum", "size"]).unstack(fam)
            sums = agg["sum"].reindex(columns=[0.0, 1.0, 2.0]).fillna(0.0)
            ns = agg["size"].reindex(columns=[0.0, 1.0, 2.0]).fillna(0.0)
            return sums / (ns + HISTORY_SHRINK), ns

        cache = {(fam, FAMILIES[fam][0]): shrunk(fam, FAMILIES[fam][0]) for fam in FAMILIES}

        def pick(fam: str, side: str, cls_idx) -> tuple[np.ndarray, np.ndarray]:
            mean, n = cache[(fam, side)]
            if mean is None:
                return np.full(n_rows, np.nan), np.full(n_rows, np.nan)
            m = mean.reindex(rows["team_id"]).fillna(0.0).to_numpy()
            c = n.reindex(rows["team_id"]).fillna(0.0).to_numpy()
            idx = np.asarray(cls_idx, dtype=float)
            ok = np.isfinite(idx)
            j = np.where(ok, idx, 0).astype(int)
            val = np.where(ok, m[np.arange(n_rows), j], np.nan)
            cnt = np.where(ok, c[np.arange(n_rows), j], np.nan)
            return np.where(has_history, val, np.nan), np.where(has_history, cnt, np.nan)

        for side, fam, label, *_ in _GENERIC:
            labels = FAMILIES[fam][2:]
            val, cnt = pick(fam, side, np.full(n_rows, float(labels.index(label))))
            out[f"mat_hist_{side}_vs_{label}"] = val
            out[f"mat_hist_n_{side}_vs_{label}"] = cnt

        cur = per_season.get(season)
        for name, side, fam, *_ in _APPLIED:
            opp_cls = (cur[fam].reindex(rows["opp_id"]).to_numpy(dtype=float) if cur is not None
                       else np.full(n_rows, np.nan))
            out[name] = pick(fam, side, opp_cls)[0]
        return out[["game_id", "team_id"] + self.column_names]
