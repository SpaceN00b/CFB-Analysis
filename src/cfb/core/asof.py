"""Warehouse and AsOfView: point-in-time access to every table.

    wh = Warehouse("data")
    view = wh.view("2025-10-07T07:00:00Z")      # or a game's cutoff_utc
    view.table("team_game")                      # only games already played at the cutoff
    view.table("lines")                          # past games and the slate at this cutoff
    view.prior_seasons("ratings_sp")             # finals of seasons that ended before it
    view.schedule()                              # who plays whom, no results

Feature code receives a view and nothing else. Each table has an access policy
(cfb.core.policies); a table without one cannot be read. Returned frames share memory
with the warehouse cache (pandas copy-on-write), so slicing is cheap and callers may
modify what they get without affecting anyone else.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..store import Store
from .game_index import REGULAR, RESULT_COLUMNS, season_bounds
from .policies import (
    DATED, GAME, POLICIES, PREGAME, PRESEASON, SEASON_FINAL, STATIC, WEEKLY, TablePolicy,
)
from .timeutil import NS_MAX, NS_MIN, RESULT_LAG, as_utc, ns

CORE_TABLES = ("game_index", "team_game", "pbp", "teams", "weeks", "game_pregame")
DERIVED_TABLES = ("unit_ratings",)


class UnknownTableError(KeyError):
    """The table has no access policy, so AsOfView will not serve it."""


class BannedTableError(PermissionError):
    """Same-season values of this table contain results after the cutoff."""


class MissingTableError(FileNotFoundError):
    """The table has a policy but no data in this warehouse."""


@dataclass
class Prepared:
    frame: pd.DataFrame  # sorted by availability, RangeIndex
    avail: np.ndarray  # int64 epoch ns per row, ascending


class Warehouse:
    """Loads tables once, orders each by the time its rows become known, and hands out views.

    Tables come from `frames` (in memory) or from disk under `root`:
    core/<name>.parquet, derived/<name>.parquet, then tables/<name>/*.parquet (raw).
    """

    def __init__(self, root: str | Path | None = "data", *, frames: dict[str, pd.DataFrame] | None = None,
                 policies: dict[str, TablePolicy] | None = None, result_lag: pd.Timedelta = RESULT_LAG):
        self.root = Path(root) if root is not None else None
        self.policies = dict(POLICIES if policies is None else policies)
        self.result_lag = pd.Timedelta(result_lag)
        self._frames: dict[str, pd.DataFrame] = dict(frames or {})
        self._prepared: dict[str, Prepared] = {}
        self._meta: dict | None = None

    # ---- loading ---------------------------------------------------------------
    def policy(self, name: str) -> TablePolicy:
        try:
            return self.policies[name]
        except KeyError:
            raise UnknownTableError(
                f"table '{name}' has no access policy. Known tables: {', '.join(sorted(self.policies))}. "
                "Policies live in cfb/core/policies.py."
            ) from None

    def _path(self, name: str) -> Path | None:
        if self.root is None:
            return None
        for sub in ("core", "derived"):
            path = self.root / sub / f"{name}.parquet"
            if path.exists():
                return path
        return None

    def exists(self, name: str) -> bool:
        if name in self._frames or self._path(name) is not None:
            return True
        return self.root is not None and any((self.root / "tables" / name).glob("*.parquet"))

    def _read(self, name: str) -> pd.DataFrame:
        if name in self._frames:
            return self._frames[name]
        path = self._path(name)
        if path is not None:
            return pd.read_parquet(path)
        if self.root is not None:
            try:
                return Store(self.root).load(name)
            except FileNotFoundError:
                pass
        hint = " Run `python -m cfb.core.build` first." if name in CORE_TABLES else ""
        raise MissingTableError(f"no data for table '{name}' under {self.root}.{hint}")

    def put(self, name: str, frame: pd.DataFrame, persist: bool = False) -> None:
        """Add or replace a table in memory; with persist, also write derived/<name>.parquet."""
        self.policy(name)
        self._frames[name] = frame
        self._prepared.pop(name, None)
        if name in ("game_index", "weeks"):
            self._meta = None
            self._prepared.clear()
        if persist:
            if self.root is None:
                raise ValueError("an in-memory warehouse cannot persist tables")
            out = self.root / "derived" / f"{name}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out, index=False)

    # ---- time metadata ---------------------------------------------------------
    @property
    def meta(self) -> dict:
        if self._meta is None:
            gi = self._read("game_index").sort_values(["start_time_utc", "game_id"]).reset_index(drop=True)
            weeks = self._read("weeks")
            bounds = season_bounds(weeks)
            start_ns, cutoff_ns = ns(gi["start_time_utc"]), ns(gi["cutoff_utc"])
            wk = pd.MultiIndex.from_frame(weeks[["season", "season_type", "week"]])
            self._meta = {
                "game_index": gi,
                "weeks": weeks,
                "game_avail": pd.Series(start_ns + self.result_lag.value, index=gi["game_id"].to_numpy()),
                "game_cutoff": pd.Series(cutoff_ns, index=gi["game_id"].to_numpy()),
                "avail_sorted": np.sort(start_ns + self.result_lag.value),
                "season_start": pd.Series(ns(bounds["start_utc"]), index=bounds.index.to_numpy()),
                "season_end": pd.Series(ns(bounds["end_utc"]), index=bounds.index.to_numpy()),
                "week_start": pd.Series(ns(weeks["start_utc"]), index=wk),
                "week_end": pd.Series(ns(weeks["end_utc"]), index=wk),
            }
        return self._meta

    @property
    def game_index(self) -> pd.DataFrame:
        """The full game_index including results. For drivers and evaluation, not feature code."""
        return self.meta["game_index"]

    def schedule(self, seasons=None) -> pd.DataFrame:
        """Every game without its result columns, in kickoff order."""
        gi = self.game_index
        if seasons is not None:
            gi = gi[gi["season"].isin(list(seasons))]
        return gi[[c for c in gi.columns if c not in RESULT_COLUMNS]].reset_index(drop=True)

    def cutoffs(self, seasons=None) -> list[pd.Timestamp]:
        return sorted(pd.unique(self.schedule(seasons)["cutoff_utc"]))

    def season_of(self, cutoff) -> int:
        """The latest season whose first calendar week has started at the cutoff."""
        starts = self.meta["season_start"]
        k = int(np.searchsorted(starts.to_numpy(), as_utc(cutoff).value, side="right"))
        return int(starts.index[k - 1]) if k else int(starts.index[0]) - 1

    def _season_ns(self, seasons: pd.Series, which: str) -> np.ndarray:
        """Season start or end per row. Seasons before the data are always visible,
        seasons after it (or unknown) never."""
        table = self.meta[which]
        s = pd.to_numeric(seasons, errors="coerce")
        idx = pd.Index(table.index).get_indexer(s.fillna(-1).astype("int64"))
        res = np.full(len(s), NS_MAX, dtype="int64")
        res[idx >= 0] = table.to_numpy()[idx[idx >= 0]]
        res[(s < table.index.min()).to_numpy()] = NS_MIN
        return res

    @staticmethod
    def _plus_one(values: np.ndarray) -> np.ndarray:
        """Turn "strictly after x" into "at or after x + 1 ns", leaving the sentinels alone."""
        return np.where((values == NS_MIN) | (values == NS_MAX), values, values + 1)

    def _availability(self, name: str, policy: TablePolicy, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(epoch ns at which each row becomes visible, mask of rows to keep).

        Integer arithmetic throughout: epoch nanoseconds do not survive float64.
        """
        n = len(raw)
        keep = np.ones(n, dtype=bool)
        if policy.kind == STATIC:
            return np.full(n, NS_MIN, dtype="int64"), keep
        if policy.kind in (GAME, PREGAME):
            table = self.meta["game_avail" if policy.kind == GAME else "game_cutoff"]
            ids = pd.to_numeric(raw[policy.game_col], errors="coerce").fillna(-1).astype("int64")
            idx = pd.Index(table.index).get_indexer(ids)
            keep = idx >= 0  # rows of games that are not in game_index cannot be dated
            out = np.full(n, NS_MAX, dtype="int64")
            out[keep] = table.to_numpy()[idx[keep]]
            return out, keep
        if policy.kind == PRESEASON:
            return self._season_ns(raw[policy.season_col], "season_start"), keep
        if policy.kind == SEASON_FINAL:
            return self._plus_one(self._season_ns(raw[policy.season_col], "season_end")), keep
        if policy.kind == DATED:
            return ns(raw[policy.date_col]), keep
        if policy.kind == WEEKLY:
            if policy.season_type_col and policy.season_type_col in raw.columns:
                stype = raw[policy.season_type_col].astype("object").reset_index(drop=True)
            else:
                stype = pd.Series(REGULAR, index=range(n), dtype="object")
            keys = pd.MultiIndex.from_arrays([
                pd.to_numeric(raw[policy.season_col], errors="coerce").fillna(-1).astype("int64").to_numpy(),
                stype.to_numpy(),
                pd.to_numeric(raw[policy.week_col], errors="coerce").fillna(-1).astype("int64").to_numpy(),
            ])
            idx = self.meta["week_start"].index.get_indexer(keys)
            post = stype.map(lambda t: policy.weekly.get(t, "post") == "post").to_numpy(dtype=bool)
            # A week the calendar does not know waits for the end of its season.
            out = self._plus_one(self._season_ns(raw[policy.season_col], "season_end"))
            hit = idx >= 0
            start = self.meta["week_start"].to_numpy()[idx[hit]]
            end = self.meta["week_end"].to_numpy()[idx[hit]] + 1
            out[hit] = np.where(post[hit], end, start)
            return out, keep
        raise ValueError(policy.kind)

    def prepared(self, name: str) -> Prepared:
        if name not in self._prepared:
            policy = self.policy(name)
            raw = self._read(name)
            hidden = policy.hidden(raw.columns)
            if hidden:
                raw = raw.drop(columns=hidden)
            if raw.empty:
                self._prepared[name] = Prepared(raw.reset_index(drop=True), np.empty(0, dtype="int64"))
                return self._prepared[name]
            avail, keep = self._availability(name, policy, raw)
            avail = avail[keep]
            order = np.argsort(avail, kind="stable")
            if keep.all() and _is_sorted(order):
                frame = raw.reset_index(drop=True)
            else:
                frame = raw[keep].iloc[order].reset_index(drop=True)
            self._prepared[name] = Prepared(frame, avail[order])
        return self._prepared[name]

    # ---- views -----------------------------------------------------------------
    def view(self, cutoff, season: int | None = None) -> "AsOfView":
        return AsOfView(self, cutoff, season)

    def outcomes(self) -> pd.DataFrame:
        """Completed games with their scores, for scoring predictions. Not for features."""
        gi = self.game_index
        return gi[gi["completed"]].reset_index(drop=True)


def _is_sorted(order: np.ndarray) -> bool:
    return bool((order == np.arange(len(order))).all())


class AsOfView:
    """What was known at `cutoff`. The only data access feature code may use."""

    def __init__(self, warehouse: Warehouse, cutoff, season: int | None = None):
        self._wh = warehouse
        self.cutoff = as_utc(cutoff)
        self._cutoff_ns = int(self.cutoff.value)
        # `season` overrides the calendar only for off-season use (predicting an
        # opener in July): preseason tables of that season become visible.
        self.season = int(season) if season is not None else warehouse.season_of(self.cutoff)

    def __repr__(self) -> str:
        return f"AsOfView(cutoff={self.cutoff.isoformat()}, season={self.season})"

    @property
    def data_key(self) -> int:
        """Number of games whose results are visible. Equal keys mean identical
        result tables, so expensive work can be cached on it."""
        return int(np.searchsorted(self._wh.meta["avail_sorted"], self._cutoff_ns, side="right"))

    def has(self, name: str) -> bool:
        return name in self._wh.policies and self._wh.exists(name)

    def _limit(self, policy: TablePolicy) -> int:
        if policy.kind == PRESEASON:
            starts = self._wh.meta["season_start"]
            if self.season in starts.index:
                return max(self._cutoff_ns, int(starts.loc[self.season]))
            if self.season > starts.index.max():
                return NS_MAX - 1
        return self._cutoff_ns

    def _slice(self, name: str, policy: TablePolicy, columns, seasons) -> pd.DataFrame:
        prep = self._wh.prepared(name)
        k = int(np.searchsorted(prep.avail, self._limit(policy), side="right"))
        out = prep.frame.iloc[:k]
        if seasons is not None:
            col = policy.season_col or ("season" if "season" in out.columns else None)
            if col is None:
                raise ValueError(f"table '{name}' has no season column to filter on")
            wanted = [seasons] if np.isscalar(seasons) else list(seasons)
            out = out[out[col].isin(wanted)]
        if policy.transform is not None:
            out = policy.transform(out, self)
        if columns is not None:
            out = out[list(columns)]
        return out

    def table(self, name: str, columns=None, seasons=None) -> pd.DataFrame:
        """Rows of `name` known at the cutoff, oldest first.

        `seasons` (an int or a list) and `columns` narrow the result. Raises
        BannedTableError for as-of-pull season tables (use prior_seasons) and
        UnknownTableError for a table without a policy.
        """
        policy = self._wh.policy(name)
        if policy.kind == SEASON_FINAL:
            raise BannedTableError(
                f"'{name}' holds season values as of the pull date; a season's own row contains results "
                f"after this cutoff. Use view.prior_seasons('{name}') for seasons that had ended."
            )
        return self._slice(name, policy, columns, seasons)

    def prior_seasons(self, name: str, columns=None, seasons=None) -> pd.DataFrame:
        """Final values of an as-of-pull season table, for seasons that ended before the cutoff."""
        policy = self._wh.policy(name)
        if policy.kind != SEASON_FINAL:
            raise ValueError(f"'{name}' is not a season-final table; use view.table('{name}')")
        return self._slice(name, policy, columns, seasons)

    def latest(self, name: str, by, columns=None) -> pd.DataFrame:
        """The most recent visible row per `by` key, e.g. latest('unit_ratings', ['team_id', 'unit'])."""
        frame = self.table(name)
        out = frame.drop_duplicates(list(by) if not isinstance(by, str) else [by], keep="last")
        return out[list(columns)] if columns is not None else out

    def schedule(self, future: bool = False) -> pd.DataFrame:
        """Games without results: every game whose cutoff is at or before this view's
        (the past and the slate being predicted).

        future=True adds later regular-season games of the cutoff season. Postseason
        games and December conference games from week 14 on are left out because
        conference title games and bowls are set by results. The future schedule is
        as of the pull date: games moved or added during a season (2020 above all)
        appear as finally played.
        """
        gi = self._wh.game_index
        cutoff = self._wh.meta["game_cutoff"].reindex(gi["game_id"]).to_numpy()
        mask = cutoff <= self._cutoff_ns
        if future:
            local_month = (gi["start_time_utc"] - pd.Timedelta(hours=8)).dt.month
            title_like = gi["conference_game"] & (gi["week"] >= 14) & (local_month == 12)
            mask = mask | ((gi["season"] == self.season) & (gi["season_type"] == REGULAR) & ~title_like).to_numpy()
        out = gi[mask]
        return out[[c for c in out.columns if c not in RESULT_COLUMNS]].reset_index(drop=True)


__all__ = [
    "AsOfView", "Warehouse", "Prepared", "BannedTableError", "MissingTableError",
    "UnknownTableError", "CORE_TABLES", "DERIVED_TABLES",
]
