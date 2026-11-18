"""Build derived tables and every registered feature group.

Usage:
    python -m cfb.features.build                       # unit_ratings, then all groups
    python -m cfb.features.build --seasons 2024-2026   # features for some seasons only
    python -m cfb.features.build --groups rest,travel
    python -m cfb.features.build --list                # print the feature catalog
    python -m cfb.features.build --tables              # print table access policies
    python -m cfb.features.build --leak-test           # run the leak test on the groups

Reads data/core (built by `python -m cfb.core.build`, run automatically when missing).
Writes data/derived/unit_ratings.parquet, data/features/<group>.parquet and
data/features/catalog.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .. import settings
from ..core import unit_ratings as ur
from ..core.asof import CORE_TABLES, Warehouse
from ..core.build import build_core
from ..core.policies import describe
from .base import FeatureGroup, catalog_table, check_group_leak, discover_groups, run_group, write_catalog


def parse_seasons(text: str | None) -> list[int] | None:
    """'2024-2026' or '2019,2021' -> list of seasons. None means all."""
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def open_warehouse(data_dir: str | Path, rebuild_core: bool = False) -> Warehouse:
    core = Path(data_dir) / "core"
    if rebuild_core or not all((core / f"{t}.parquet").exists() for t in CORE_TABLES):
        print("building core tables")
        build_core(data_dir)
    return Warehouse(data_dir)


def _progress(label: str):
    start = time.time()

    def report(done: int, total: int) -> None:
        if done == total or done % 100 == 0:
            print(f"  {label}: {done}/{total} cutoffs, {time.time() - start:.0f}s", flush=True)

    return report


def build_derived(warehouse: Warehouse, persist: bool = True) -> pd.DataFrame:
    """unit_ratings at every cutoff. Always covers all seasons: feature groups may
    look back across seasons."""
    builder, source = ur.resolve_builder()
    print(f"unit_ratings from {source}")
    frame = ur.build_unit_ratings(warehouse, builder=builder, progress=_progress("unit_ratings"))
    warehouse.put("unit_ratings", frame, persist=persist)
    return frame


def build_features(warehouse: Warehouse, groups: list[FeatureGroup], seasons: list[int] | None = None,
                   write: bool = True) -> dict[str, pd.DataFrame]:
    """Build `groups` for the games of `seasons` (None: every season) and write
    data/features/<group>.parquet. With `seasons`, rows of other seasons already in the file
    are kept; the returned frames hold the built games only."""
    games = warehouse.schedule(seasons)
    out_dir = (warehouse.root / "features") if warehouse.root is not None else None
    built = {}
    for group in groups:
        print(f"group {group.name} ({group.level}, {group.mode}, {len(group.columns)} columns)")
        frame = run_group(group, warehouse, games, progress=_progress(group.name))
        built[group.name] = frame
        if write and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{group.name}.parquet"
            if seasons is not None and path.exists():
                # Some seasons only: keep the stored rows of every other game, so a weekly
                # run that rebuilds the current season keeps the training seasons' features.
                kept = pd.read_parquet(path)
                kept = kept[~kept["game_id"].isin(games["game_id"])]
                if len(kept):
                    frame = pd.concat([kept, frame], ignore_index=True)
            frame.to_parquet(path, index=False)
    if write and out_dir is not None and groups:
        write_catalog(groups, out_dir / "catalog.json")
    return built


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--seasons", default="", help="e.g. 2024-2026 or 2019,2021 (default: all)")
    ap.add_argument("--groups", default="", help="comma-separated group names (default: all)")
    ap.add_argument("--list", action="store_true", help="print the feature catalog and exit")
    ap.add_argument("--tables", action="store_true", help="print table access policies and exit")
    ap.add_argument("--leak-test", action="store_true", help="leak-test the selected groups and exit")
    ap.add_argument("--leak-cutoffs", type=int, default=3)
    ap.add_argument("--rebuild-core", action="store_true")
    ap.add_argument("--skip-unit-ratings", action="store_true", help="reuse data/derived/unit_ratings.parquet")
    args = ap.parse_args(argv)

    if args.tables:
        print(describe().to_string(index=False))
        return 0
    groups = discover_groups()
    wanted = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = sorted(set(wanted) - {g.name for g in groups})
    if unknown:
        ap.error(f"unknown groups: {', '.join(unknown)}")
    if wanted:
        groups = [g for g in groups if g.name in wanted]
    if args.list:
        table = catalog_table(groups)
        print(table.to_string(index=False) if len(table) else "no feature groups registered")
        return 0

    warehouse = open_warehouse(args.data_dir, args.rebuild_core)
    if not (args.skip_unit_ratings and warehouse.exists("unit_ratings")):
        build_derived(warehouse)

    if args.leak_test:
        failed = 0
        for group in groups:
            problems = check_group_leak(group, warehouse, n=args.leak_cutoffs)
            print(f"{group.name}: {'ok' if not problems else 'LEAK'}")
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            failed += bool(problems)
        if not groups:
            print("no feature groups registered")
        return 1 if failed else 0

    if not groups:
        print("no feature groups registered; derived tables are up to date")
        return 0
    build_features(warehouse, groups, parse_seasons(args.seasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
