"""Leak test: scramble everything after a cutoff, rebuild, and nothing built for that
cutoff may change.

    from cfb.core.leak import scramble_after, check_leak
    scrambled = scramble_after(warehouse, cutoff)     # same keys and schedule, different values

`scramble_after` returns a warehouse in which every row that is not visible at the
cutoff has its values shuffled among those rows and perturbed. Identifiers and schedule
facts stay, so joins still work. What counts as "after the cutoff" follows each table's
access policy, which makes it wider than game results: later seasons of preseason
tables, later dated rows, and same-season rows of as-of-pull tables are scrambled too.

Feature groups use the wrapper in cfb.features.base (assert_group_no_leak), which adds
call-order checks and a source lint. This module is the engine and has no knowledge of
feature groups.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .asof import Prepared, Warehouse
from .policies import DEFAULT_KEYS, TablePolicy
from .timeutil import as_utc


class LeakError(AssertionError):
    """A value built for a cutoff changed when later data was scrambled."""


def _protected(policy: TablePolicy) -> set[str]:
    cols = set(DEFAULT_KEYS) | set(policy.keys)
    for col in (policy.game_col, policy.season_col, policy.date_col, policy.week_col, policy.season_type_col):
        if col:
            cols.add(col)
    return cols


def scramble_frame(frame: pd.DataFrame, protect: Iterable[str], rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle every unprotected column across rows and perturb numbers, keeping dtypes."""
    out = frame.copy()
    n = len(out)
    if n == 0:
        return out
    protect = set(protect)
    for col in out.columns:
        if col in protect:
            continue
        shuffled = out[col].iloc[rng.permutation(n)]
        shuffled.index = out.index
        dtype = shuffled.dtype
        if pd.api.types.is_bool_dtype(dtype) and not isinstance(dtype, pd.BooleanDtype):
            shuffled = pd.Series(rng.integers(0, 2, n).astype(bool), index=out.index)
        elif pd.api.types.is_integer_dtype(dtype):
            shuffled = (shuffled + rng.integers(1, 9, n)).astype(dtype)
        elif pd.api.types.is_float_dtype(dtype):
            shuffled = (shuffled * rng.uniform(0.5, 1.5, n) + rng.normal(0.0, 1.0, n)).astype(dtype)
        out[col] = shuffled
    return out


class ScrambledWarehouse(Warehouse):
    """A warehouse whose rows not visible at `cutoff` are scrambled. Built lazily per table."""

    def __init__(self, parent: Warehouse, cutoff, seed: int = 0):
        super().__init__(parent.root, policies=parent.policies, result_lag=parent.result_lag)
        self._parent = parent
        self.scramble_cutoff = as_utc(cutoff)
        self._seed = seed
        self.scrambled_rows: dict[str, int] = {}

    @property
    def meta(self) -> dict:
        return self._parent.meta  # keys and schedule are never scrambled

    def exists(self, name: str) -> bool:
        return self._parent.exists(name)

    def _read(self, name: str) -> pd.DataFrame:
        return self._parent._read(name)

    def put(self, name, frame, persist=False):
        raise TypeError("a scrambled warehouse is read-only")

    def prepared(self, name: str) -> Prepared:
        if name not in self._prepared:
            base = self._parent.prepared(name)
            k = int(np.searchsorted(base.avail, self.scramble_cutoff.value, side="right"))
            self.scrambled_rows[name] = len(base.frame) - k
            if k >= len(base.frame):
                self._prepared[name] = base
            else:
                # Seeded per table so the result does not depend on access order.
                rng = np.random.default_rng([self._seed, *name.encode()])
                tail = scramble_frame(base.frame.iloc[k:], _protected(self.policy(name)), rng)
                frame = pd.concat([base.frame.iloc[:k], tail], ignore_index=True)
                self._prepared[name] = Prepared(frame, base.avail)
        return self._prepared[name]


def scramble_after(warehouse: Warehouse, cutoff, seed: int = 0) -> ScrambledWarehouse:
    return ScrambledWarehouse(warehouse, cutoff, seed)


def diff_frames(real: pd.DataFrame, other: pd.DataFrame, keys: list[str], rtol: float = 1e-9,
                atol: float = 1e-12, label: str = "") -> list[str]:
    """Differences between two builds, as readable lines. Empty when they agree."""
    where = f" [{label}]" if label else ""
    if set(real.columns) != set(other.columns):
        return [f"columns differ{where}: {sorted(set(real.columns) ^ set(other.columns))}"]
    a = real.sort_values(keys).reset_index(drop=True)
    b = other[real.columns].sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not a[keys].equals(b[keys]):
        return [f"row keys differ{where}: {len(a)} rows vs {len(b)}"]
    problems = []
    for col in a.columns:
        if col in keys:
            continue
        x, y = a[col], b[col]
        both_na = (x.isna() & y.isna()).to_numpy()
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(dtype="float64", na_value=np.nan), y.to_numpy(dtype="float64", na_value=np.nan)
            same = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = (x.astype("object") == y.astype("object")).fillna(False).to_numpy(dtype=bool) | both_na
        if not same.all():
            i = int(np.flatnonzero(~same)[0])
            key = ", ".join(f"{k}={a[k].iloc[i]}" for k in keys)
            problems.append(
                f"{col}{where}: {int((~same).sum())} of {len(a)} rows changed "
                f"(first: {key}: {x.iloc[i]!r} -> {y.iloc[i]!r})"
            )
    return problems


def pick_cutoffs(warehouse: Warehouse, n: int = 3) -> list[pd.Timestamp]:
    """Cutoffs spread over the data, each with games before and after it."""
    cutoffs = warehouse.cutoffs()
    if len(cutoffs) < 3:
        raise ValueError("need at least three cutoffs to leak-test")
    inner = cutoffs[1:-1]
    picks = sorted({inner[int(round(q * (len(inner) - 1)))] for q in np.linspace(0.2, 0.9, n)})
    return [as_utc(c) for c in picks]


def check_leak(build: Callable[[Warehouse, pd.Timestamp], pd.DataFrame], warehouse: Warehouse,
               cutoffs: Iterable, keys: list[str], seed: int = 0, rtol: float = 1e-9) -> list[str]:
    """Run `build(warehouse, cutoff)` on real and scrambled data for each cutoff.

    `build` must return what is computed for that cutoff, keyed by `keys`. Returns the
    differences found (empty when leak-free).
    """
    problems: list[str] = []
    for cutoff in cutoffs:
        cutoff = as_utc(cutoff)
        scrambled = scramble_after(warehouse, cutoff, seed)
        real = build(warehouse, cutoff)
        other = build(scrambled, cutoff)
        problems += diff_frames(real, other, keys, rtol=rtol, label=f"cutoff {cutoff.isoformat()}")
    return problems


def assert_no_leak(build, warehouse: Warehouse, cutoffs: Iterable, keys: list[str], seed: int = 0) -> None:
    problems = check_leak(build, warehouse, cutoffs, keys, seed)
    if problems:
        raise LeakError("values changed when data after the cutoff was scrambled:\n  " + "\n  ".join(problems))


# ---- source lint -------------------------------------------------------------------
# The scramble test cannot see code that reads files itself, so feature sources are also
# checked for anything that reaches data outside the view.

FORBIDDEN_MODULES = (
    "cfb.store", "cfb.client", "cfb.ingest", "cfb.core.build", "cfb.core.leak",
    "requests", "urllib", "http", "glob", "pyarrow", "sqlite3", "duckdb", "polars", "pickle",
)
FORBIDDEN_RELATIVE = ("store", "client", "ingest", "core.build", "core.leak")
FORBIDDEN_NAMES = ("Warehouse", "Store", "ScrambledWarehouse")
FORBIDDEN_CALLS = (
    "read_parquet", "read_csv", "read_json", "read_pickle", "read_feather", "read_table",
    "read_excel", "read_sql", "read_orc", "read_hdf",
)
FORBIDDEN_ATTRIBUTES = ("_wh", "_parent", "outcomes")


def lint_source(source: str, filename: str = "<source>") -> list[str]:
    """Reasons this source could bypass AsOfView. Empty when clean."""
    problems = []

    def flag(node, what):
        problems.append(f"{filename}:{getattr(node, 'lineno', 0)}: {what}")

    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == m or alias.name.startswith(m + ".") for m in FORBIDDEN_MODULES):
                    flag(node, f"imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and any(mod == m or mod.startswith(m + ".") for m in FORBIDDEN_MODULES):
                flag(node, f"imports from {mod}")
            if node.level > 0 and any(mod == m or mod.endswith("." + m) for m in FORBIDDEN_RELATIVE):
                flag(node, f"imports from {'.' * node.level}{mod}")
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES or (not mod and alias.name in ("store", "client", "ingest")):
                    flag(node, f"imports {alias.name}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "open":
                flag(node, "calls open()")
            elif isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_CALLS:
                flag(node, f"calls {fn.attr}()")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            flag(node, f"touches .{node.attr}")
    return problems


def lint_paths(paths: Iterable[str | Path]) -> list[str]:
    """Lint every .py file under the given files or directories."""
    problems = []
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            problems += lint_source(file.read_text(encoding="utf-8"), str(file))
    return problems
