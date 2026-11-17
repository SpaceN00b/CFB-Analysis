"""Feature group interface, catalog, runner and leak test.

A feature group turns point-in-time data into model inputs:

    from cfb.features.base import FeatureColumn, FeatureGroup

    class Rest(FeatureGroup):
        name = "rest"
        prefix = "ctx_"
        level = "team"                       # rows keyed by game_id, team_id
        columns = [
            FeatureColumn("ctx_rest_days", "Days since the team's previous game",
                          sources=("schedule",), first_season=2014),
        ]

        def build(self, view, games):
            ...
            return frame                     # game_id, team_id, ctx_rest_days

    GROUPS = [Rest()]                        # in cfb/features/<area>/__init__.py

Rules the runner enforces:
    * `build(view, games)` gets an AsOfView and the games to describe (schedule columns
      only, never results). In the default mode every game in `games` has
      cutoff_utc == view.cutoff.
    * The result is keyed by game_id, team_id (level "team", one row per side) or
      game_id (level "game"). Rows may be missing; the runner fills them with empty values.
    * Every other column starts with the group's prefix and has a catalog entry.
      Values are numeric or boolean. Unknown stays empty (NaN); never fill from later data.
    * `build` must be a pure function of its arguments: no state kept between calls.

Modes:
    per_cutoff (default)   one call per distinct cutoff. Leak-safe by construction as
                           long as the code reads data only through the view.
    batch                  one call with a view at the last cutoff and all games. Faster
                           for running totals, but the code itself must make sure each
                           row uses only data before that game's cutoff_utc. The leak
                           test is what guards this mode.
"""
from __future__ import annotations

import copy
import inspect
import json
import pkgutil
import re
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..core.asof import Warehouse
from ..core.leak import LeakError, diff_frames, lint_paths, pick_cutoffs, scramble_after
from ..core.policies import POLICIES
from ..core.timeutil import as_utc

PREFIXES = {"strength": "str_", "matchup": "mat_", "coaching": "coach_", "context": "ctx_", "market": "mkt_"}
ALLOWED_PREFIXES = tuple(PREFIXES.values())
LEVEL_KEYS = {"team": ["game_id", "team_id"], "game": ["game_id"]}
MODES = ("per_cutoff", "batch")
# Sources a catalog line may name besides table names.
EXTRA_SOURCES = ("schedule",)


@dataclass(frozen=True)
class FeatureColumn:
    """One catalog line: what the column is, where it comes from, since when it has data."""

    name: str
    description: str
    sources: tuple[str, ...]
    first_season: int


class FeatureGroup:
    name: str = ""
    prefix: str = ""
    level: str = "team"
    mode: str = "per_cutoff"
    columns: Sequence[FeatureColumn] = ()

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def keys(self) -> list[str]:
        return list(LEVEL_KEYS[self.level])

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def catalog(self) -> dict:
        return {
            "name": self.name, "prefix": self.prefix, "level": self.level, "mode": self.mode,
            "columns": [{**asdict(c), "sources": list(c.sources)} for c in self.columns],
        }


class FeatureError(ValueError):
    """A group or its output breaks the feature contract."""


def validate_group(group: FeatureGroup) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", group.name or ""):
        raise FeatureError(f"group name {group.name!r} must be lower snake case")
    who = f"group '{group.name}'"
    if group.prefix not in ALLOWED_PREFIXES:
        raise FeatureError(f"{who}: prefix {group.prefix!r} is not one of {ALLOWED_PREFIXES}")
    if group.level not in LEVEL_KEYS:
        raise FeatureError(f"{who}: level must be 'team' or 'game'")
    if group.mode not in MODES:
        raise FeatureError(f"{who}: mode must be one of {MODES}")
    if not group.columns:
        raise FeatureError(f"{who}: no columns in its catalog")
    seen = set()
    for col in group.columns:
        if not isinstance(col, FeatureColumn):
            raise FeatureError(f"{who}: catalog entries must be FeatureColumn")
        if not col.name.startswith(group.prefix) or col.name == group.prefix:
            raise FeatureError(f"{who}: column '{col.name}' must start with '{group.prefix}'")
        if col.name in seen:
            raise FeatureError(f"{who}: column '{col.name}' is listed twice")
        seen.add(col.name)
        if not col.description.strip():
            raise FeatureError(f"{who}: column '{col.name}' has no description")
        if not col.sources:
            raise FeatureError(f"{who}: column '{col.name}' names no sources")
        unknown = [s for s in col.sources if s not in POLICIES and s not in EXTRA_SOURCES]
        if unknown:
            raise FeatureError(f"{who}: column '{col.name}' names unknown sources {unknown}")
        if not isinstance(col.first_season, int):
            raise FeatureError(f"{who}: column '{col.name}' needs an integer first_season")


def expected_keys(group: FeatureGroup, games: pd.DataFrame) -> pd.DataFrame:
    """The key rows a complete output has, in the order of `games`."""
    if group.level == "game":
        return games[["game_id"]].reset_index(drop=True)
    home = pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["home_id"].to_numpy(), "_o": 0})
    away = pd.DataFrame({"game_id": games["game_id"].to_numpy(), "team_id": games["away_id"].to_numpy(), "_o": 1})
    both = pd.concat([home, away], ignore_index=True)
    both["_g"] = np.tile(np.arange(len(games)), 2)
    return both.sort_values(["_g", "_o"])[["game_id", "team_id"]].reset_index(drop=True)


def validate_output(group: FeatureGroup, out: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Check a build result against the contract and return it completed to one row per key."""
    who = f"group '{group.name}'"
    if not isinstance(out, pd.DataFrame):
        raise FeatureError(f"{who}: build must return a DataFrame")
    keys = group.keys
    missing = [k for k in keys if k not in out.columns]
    if missing:
        raise FeatureError(f"{who}: output lacks key columns {missing}")
    declared = group.column_names
    extra = [c for c in out.columns if c not in keys and c not in declared]
    if extra:
        raise FeatureError(f"{who}: output has columns missing from the catalog: {extra}")
    absent = [c for c in declared if c not in out.columns]
    if absent:
        raise FeatureError(f"{who}: catalog columns missing from the output: {absent}")
    if out.duplicated(keys).any():
        raise FeatureError(f"{who}: duplicate rows for {keys}")
    for col in declared:
        dtype = out[col].dtype
        if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            raise FeatureError(f"{who}: column '{col}' has dtype {dtype}; features must be numeric or boolean")
    want = expected_keys(group, games)
    merged = want.merge(out[keys + declared], on=keys, how="left", indicator=True)
    stray = len(out) - int((merged["_merge"] == "both").sum())
    if stray:
        raise FeatureError(f"{who}: {stray} output rows are not among the requested games")
    return merged.drop(columns="_merge")


def run_group(group: FeatureGroup, warehouse: Warehouse, games: pd.DataFrame | None = None,
              progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Build a group for `games` (default: every game) and return one row per key."""
    validate_group(group)
    games = warehouse.schedule() if games is None else games
    if games.empty:
        return pd.DataFrame({c: [] for c in group.keys + group.column_names})
    if group.mode == "batch":
        view = warehouse.view(games["cutoff_utc"].max())
        return validate_output(group, group.build(view, games.reset_index(drop=True)), games)
    parts = []
    slates = list(games.groupby("cutoff_utc", sort=True))
    for i, (cutoff, slate) in enumerate(slates):
        slate = slate.reset_index(drop=True)
        parts.append(validate_output(group, group.build(warehouse.view(cutoff), slate), slate))
        if progress:
            progress(i + 1, len(slates))
    return pd.concat(parts, ignore_index=True)


# ---- discovery and catalog ---------------------------------------------------------

def discover_groups() -> list[FeatureGroup]:
    """Every group listed as GROUPS in a subpackage of cfb.features."""
    package = import_module("cfb.features")
    groups: list[FeatureGroup] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if not info.ispkg:
            continue
        module = import_module(f"cfb.features.{info.name}")
        for group in getattr(module, "GROUPS", ()):
            validate_group(group)
            groups.append(group)
    names = [g.name for g in groups]
    if len(set(names)) != len(names):
        raise FeatureError(f"duplicate group names: {sorted(n for n in set(names) if names.count(n) > 1)}")
    cols = [c for g in groups for c in g.column_names]
    if len(set(cols)) != len(cols):
        raise FeatureError(f"column defined by two groups: {sorted(c for c in set(cols) if cols.count(c) > 1)}")
    return groups


def catalog(groups: Iterable[FeatureGroup]) -> list[dict]:
    return [g.catalog() for g in groups]


def catalog_table(groups: Iterable[FeatureGroup]) -> pd.DataFrame:
    """One line per column: group, column, description, sources, first season."""
    rows = [
        {"group": g.name, "level": g.level, "column": c.name, "description": c.description,
         "sources": ", ".join(c.sources), "first_season": c.first_season}
        for g in groups for c in g.columns
    ]
    return pd.DataFrame(rows, columns=["group", "level", "column", "description", "sources", "first_season"])


def write_catalog(groups: Iterable[FeatureGroup], path: str | Path) -> None:
    payload = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "groups": catalog(groups)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---- leak test -----------------------------------------------------------------------

def group_source_dir(group: FeatureGroup) -> Path | None:
    """The package directory of a group defined under cfb.features.<area>, else None."""
    module = type(group).__module__
    if not module.startswith("cfb.features.") or module in ("cfb.features.base", "cfb.features.build"):
        return None
    area = module.split(".")[2]
    return Path(inspect.getfile(import_module(f"cfb.features.{area}"))).parent


def _games_at(games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return games[games["cutoff_utc"] == cutoff].reset_index(drop=True)


def check_group_leak(group: FeatureGroup, warehouse: Warehouse, cutoffs: Iterable | None = None,
                     n: int = 3, seed: int = 0, lint: bool = True) -> list[str]:
    """Problems found by the leak test for one group. Empty when clean.

    For each cutoff the group is built for that cutoff's games on real data and on data
    scrambled after the cutoff. A per-cutoff group is first run, on a fresh copy, at the
    last cutoff of the same season and of the data, so anything it remembers between
    calls shows up as a change.
    Sources under cfb.features.<area>/ are also linted for data access that bypasses the view.
    """
    validate_group(group)
    problems: list[str] = []
    src = group_source_dir(group) if lint else None
    if src is not None:
        problems += lint_paths([src])
    cutoffs = [as_utc(c) for c in (pick_cutoffs(warehouse, n) if cutoffs is None else cutoffs)]
    games = warehouse.schedule()
    last = games["cutoff_utc"].max()

    def batch(wh: Warehouse) -> pd.DataFrame:
        return run_group(copy.deepcopy(group), wh, games)

    def build(wh: Warehouse, cutoff: pd.Timestamp, full: pd.DataFrame | None = None) -> pd.DataFrame:
        if group.mode == "batch":
            out = batch(wh) if full is None else full
            ids = set(_games_at(games, cutoff)["game_id"])
            return out[out["game_id"].isin(ids)].reset_index(drop=True)
        fresh = copy.deepcopy(group)
        slate = _games_at(games, cutoff)
        # Prime the copy with later slates first: the end of the same season and the
        # end of the data. Anything it remembers from those calls would be a leak.
        season_last = games[games["season"].isin(slate["season"].unique())]["cutoff_utc"].max()
        for later in sorted({season_last, last} - {cutoff}):
            run_group(fresh, wh, _games_at(games, later))
        return run_group(fresh, wh, slate)

    real_full = batch(warehouse) if group.mode == "batch" else None
    for cutoff in cutoffs:
        if _games_at(games, cutoff).empty:
            problems.append(f"no games have cutoff {cutoff.isoformat()}")
            continue
        real = build(warehouse, cutoff, real_full)
        other = build(scramble_after(warehouse, cutoff, seed), cutoff)
        problems += diff_frames(real, other, group.keys, label=f"{group.name} at {cutoff.isoformat()}")
    return problems


def assert_group_no_leak(group: FeatureGroup, warehouse: Warehouse, cutoffs: Iterable | None = None,
                         n: int = 3, seed: int = 0, lint: bool = True) -> None:
    """Raise LeakError if the group fails the leak test. Every feature thread runs this."""
    problems = check_group_leak(group, warehouse, cutoffs, n, seed, lint)
    if problems:
        raise LeakError(f"group '{group.name}' failed the leak test:\n  " + "\n  ".join(problems))


__all__ = [
    "ALLOWED_PREFIXES", "PREFIXES", "FeatureColumn", "FeatureError", "FeatureGroup",
    "assert_group_no_leak", "catalog", "catalog_table", "check_group_leak", "discover_groups",
    "expected_keys", "run_group", "validate_group", "validate_output", "write_catalog",
]
