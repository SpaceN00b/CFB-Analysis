"""unit_ratings: the contract, and a placeholder that fills it.

Contract (one row per team, cutoff and unit):
    team_id, cutoff_utc, unit, epa_adj, success_adj, explosive_adj, n_plays
    unit is one of off_rush, off_pass, def_rush, def_pass, off_all, def_all

Scale: EPA per play, success rate (0 to 1), and EPA per successful play. Defensive
units hold what the defense allowed, so lower is better for a defense. `n_plays` is
the number of plays behind the row; a team with no plays yet has n_plays 0 and empty
ratings.

The placeholder is the unadjusted season-to-date average from team_game. W3 replaces
it with opponent-adjusted ratings by defining `unit_ratings(view)` in
`cfb.features.strength`; the build driver picks that up automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNITS = ("off_rush", "off_pass", "def_rush", "def_pass", "off_all", "def_all")
UNIT_RATINGS_COLUMNS = ["team_id", "cutoff_utc", "unit", "epa_adj", "success_adj", "explosive_adj", "n_plays"]
_RATING_COLUMNS = ["epa_adj", "success_adj", "explosive_adj"]


def placeholder_unit_ratings(view) -> pd.DataFrame:
    """Unadjusted averages over the cutoff season's games visible in `view`."""
    teams = view.table("teams", seasons=view.season)["team_id"].drop_duplicates().sort_values().to_numpy()
    cols = [f"pbp_{side}_{u}_{s}" for side in ("off", "def") for u in ("all", "rush", "pass")
            for s in ("plays", "epa_sum", "successes", "expl_sum")]
    tg = view.table("team_game", columns=["team_id"] + cols, seasons=view.season)
    sums = tg.groupby("team_id")[cols].sum(min_count=1).reindex(teams)
    parts = []
    for unit in UNITS:
        side, kind = unit.split("_")
        pre = f"pbp_{side}_{kind}_"
        plays = sums[pre + "plays"].fillna(0.0).to_numpy()
        succ = sums[pre + "successes"].fillna(0.0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            epa = np.where(plays > 0, sums[pre + "epa_sum"].to_numpy() / plays, np.nan)
            success = np.where(plays > 0, succ / plays, np.nan)
            explosive = np.where(succ > 0, sums[pre + "expl_sum"].to_numpy() / succ, np.nan)
        parts.append(pd.DataFrame({
            "team_id": teams, "unit": unit, "epa_adj": epa, "success_adj": success,
            "explosive_adj": explosive, "n_plays": plays.astype("int64"),
        }))
    out = pd.concat(parts, ignore_index=True)
    out["cutoff_utc"] = view.cutoff
    out["cutoff_utc"] = out["cutoff_utc"].dt.as_unit("ns")
    return out[UNIT_RATINGS_COLUMNS]


def validate_unit_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError unless `frame` meets the contract. Returns it with canonical dtypes."""
    missing = [c for c in UNIT_RATINGS_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"unit_ratings is missing columns: {missing}")
    out = frame[UNIT_RATINGS_COLUMNS].copy()
    bad = sorted(set(out["unit"].unique()) - set(UNITS))
    if bad:
        raise ValueError(f"unit_ratings has unknown units: {bad}")
    if not isinstance(out["cutoff_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("unit_ratings.cutoff_utc must be timezone-aware UTC")
    if out.duplicated(["team_id", "cutoff_utc", "unit"]).any():
        raise ValueError("unit_ratings has duplicate (team_id, cutoff_utc, unit) rows")
    out["team_id"] = out["team_id"].astype("int64")
    out["unit"] = out["unit"].astype("str")
    out["cutoff_utc"] = out["cutoff_utc"].dt.tz_convert("UTC").dt.as_unit("ns")
    for col in _RATING_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    if out["n_plays"].isna().any() or (out["n_plays"] < 0).any():
        raise ValueError("unit_ratings.n_plays must be a non-negative count")
    out["n_plays"] = out["n_plays"].astype("int64")
    return out


def resolve_builder():
    """The unit ratings builder to use: W3's if it exists, else the placeholder.

    Returns (callable(view) -> DataFrame, source label).
    """
    try:
        from importlib import import_module

        strength = import_module("cfb.features.strength")
    except ModuleNotFoundError as exc:
        if exc.name != "cfb.features.strength":
            raise
        return placeholder_unit_ratings, "placeholder"
    builder = getattr(strength, "unit_ratings", None)
    if callable(builder):
        return builder, "cfb.features.strength.unit_ratings"
    return placeholder_unit_ratings, "placeholder"


def build_unit_ratings(warehouse, cutoffs=None, builder=None, progress=None) -> pd.DataFrame:
    """Run the builder at every cutoff (default: every cutoff in game_index)."""
    builder = builder or resolve_builder()[0]
    cutoffs = list(warehouse.cutoffs() if cutoffs is None else cutoffs)
    parts = []
    for i, cutoff in enumerate(cutoffs):
        parts.append(validate_unit_ratings(builder(warehouse.view(cutoff))))
        if progress:
            progress(i + 1, len(cutoffs))
    if not parts:
        empty = pd.DataFrame({c: [] for c in UNIT_RATINGS_COLUMNS})
        empty["cutoff_utc"] = pd.to_datetime(empty["cutoff_utc"], utc=True)
        return empty
    return pd.concat(parts, ignore_index=True)
