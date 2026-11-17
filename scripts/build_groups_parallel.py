"""Build feature groups with one process per season (same output as cfb.features.build).

    python scripts/build_groups_parallel.py str_qb [--workers 6] [--seasons 2014-2026]

Needs data/derived/unit_ratings.parquet (python -m cfb.features.build builds it).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from cfb.core.asof import Warehouse
from cfb.features.base import discover_groups, run_group, write_catalog
from cfb.features.build import parse_seasons


def _one(args):
    data_dir, name, season = args
    wh = Warehouse(data_dir)
    group = next(g for g in discover_groups() if g.name == name)
    return run_group(group, wh, wh.schedule([season]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("groups")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seasons", default="")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    seasons = parse_seasons(args.seasons) or sorted(wh.game_index["season"].unique().tolist())
    out_dir = Path(args.data_dir) / "features"
    for name in args.groups.split(","):
        with ProcessPoolExecutor(args.workers) as ex:
            parts = list(ex.map(_one, [(args.data_dir, name, int(s)) for s in seasons]))
        frame = pd.concat(parts, ignore_index=True)
        path = out_dir / f"{name}.parquet"
        if args.seasons and path.exists():
            kept = pd.read_parquet(path)
            frame = pd.concat([kept[~kept["game_id"].isin(frame["game_id"])], frame], ignore_index=True)
        frame.to_parquet(path, index=False)
        print(f"{name}: {len(frame):,} rows -> {path}")
    write_catalog(discover_groups(), out_dir / "catalog.json")


if __name__ == "__main__":
    main()
