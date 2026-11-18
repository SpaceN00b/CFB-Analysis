"""Build the core tables from the raw store.

Usage:
    python -m cfb.core.build                 # data/tables -> data/core
    python -m cfb.core.build --data-dir data --check

Writes data/core/{weeks,game_index,teams,game_pregame,pbp,team_game}.parquet.
Takes about 15 seconds on the 2014 to 2026 snapshot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .. import settings
from ..store import Store
from .game_index import build_game_index, build_game_pregame, build_teams, build_weeks
from .pbp import build_pbp, load_raw_plays
from .team_game import build_team_game


def build_core(data_dir: str | Path = "data", write: bool = True) -> dict[str, pd.DataFrame]:
    """Build every core table. Returns them by name; with write=True also saves them."""
    store = Store(data_dir)
    weeks = build_weeks(store)
    game_index = build_game_index(store, weeks)
    pbp = build_pbp(load_raw_plays(store), game_index)
    tables = {
        "weeks": weeks,
        "game_index": game_index,
        "teams": build_teams(store),
        "game_pregame": build_game_pregame(store),
        "pbp": pbp,
        "team_game": build_team_game(store, game_index, pbp),
    }
    if write:
        out = Path(data_dir) / "core"
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            tmp = out / f"{name}.parquet.tmp"
            frame.to_parquet(tmp, index=False)
            tmp.replace(out / f"{name}.parquet")
    return tables


def summarize(tables: dict[str, pd.DataFrame]) -> dict:
    gi, tg, pbp = tables["game_index"], tables["team_game"], tables["pbp"]
    done = gi[gi["completed"]]
    fbs2 = done["home_fbs"] & done["away_fbs"]
    return {
        "games": len(gi),
        "completed": len(done),
        "completed_fbs_vs_fbs": int(fbs2.sum()),
        "seasons": f"{gi['season'].min()}-{gi['season'].max()}",
        "cutoffs": int(gi["cutoff_utc"].nunique()),
        "team_game_rows": len(tg),
        "team_game_columns": tg.shape[1],
        "plays": len(pbp),
        "games_with_plays": int(pbp["game_id"].nunique()) if len(pbp) else 0,
    }


def check(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Structural checks. Returns a list of problems (empty when clean)."""
    gi, tg = tables["game_index"], tables["team_game"]
    problems = []
    if gi["game_id"].duplicated().any():
        problems.append("game_index: duplicate game_id")
    if (gi["cutoff_utc"] > gi["start_time_utc"]).any():
        problems.append("game_index: cutoff after kickoff")
    if (gi["completed"] & (gi["home_points"].isna() | gi["away_points"].isna())).any():
        problems.append("game_index: completed game without points")
    if len(tg) != 2 * len(gi):
        problems.append(f"team_game: {len(tg)} rows for {len(gi)} games")
    if tg.duplicated(["game_id", "team_id"]).any():
        problems.append("team_game: duplicate (game_id, team_id)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--check", action="store_true", help="build in memory and report, without writing")
    args = ap.parse_args(argv)
    tables = build_core(args.data_dir, write=not args.check)
    for key, value in summarize(tables).items():
        print(f"{key:24s} {value}")
    problems = check(tables)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
