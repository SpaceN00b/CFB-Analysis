"""Kneel-removal experiment: rebuild the core tables in a separate data folder with likely kneels
dropped from the plays, so every play-based feature (unit ratings, str_unit_*) leaves them out.

    python scripts/kneel_experiment.py --src data --out data_k1 --yards -6,-1 --clock 60
    python -m cfb.features.build --data-dir data_k1 --groups str_power,str_power2,str_program
    python -m cfb.predict --data-dir data_k1 --model model_v2 walk-forward --seasons 2026

A play is dropped when CFBD types it "Rush", it gained between the two --yards values (inclusive),
it started with under --clock seconds on the game clock in the 2nd or 4th quarter, and the
offense was ahead. `--yards none` drops nothing (the baseline folder, built with the same code).
The source folder is not changed: tables/ and raw/ are copied, features/ is copied so groups
that are not rebuilt (str_elo) are there, and core/ is written fresh.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from cfb.core.game_index import build_game_index, build_game_pregame, build_teams, build_weeks
from cfb.core.pbp import build_pbp, load_raw_plays
from cfb.core.team_game import build_team_game
from cfb.store import Store


def kneel_mask(raw: pd.DataFrame, yards: tuple[int, int], clock: int) -> pd.Series:
    secs = raw["clock.minutes"].astype(float) * 60 + raw["clock.seconds"].astype(float)
    lead = raw["offenseScore"].astype(float) - raw["defenseScore"].astype(float)
    gained = pd.to_numeric(raw["yardsGained"], errors="coerce")
    return ((raw["playType"] == "Rush") & gained.between(yards[0], yards[1])
            & (secs < clock) & raw["period"].isin([2, 4]) & (lead > 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--yards", default="-6,-1", help="lo,hi inclusive, or 'none' for the baseline")
    ap.add_argument("--clock", type=int, default=60, help="game-clock seconds left, strictly under")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "raw", "features"):
        if not (out / sub).exists():
            shutil.copytree(src / sub, out / sub)
    store = Store(out)
    weeks = build_weeks(store)
    game_index = build_game_index(store, weeks)
    raw = load_raw_plays(store)
    if a.yards != "none":
        lo, hi = (int(v) for v in a.yards.split(","))
        mask = kneel_mask(raw, (lo, hi), a.clock)
        by_season = raw.loc[mask, "season"].value_counts().sort_index().to_dict()
        print(f"dropping {int(mask.sum())} plays as kneels: {by_season}")
        raw = raw[~mask].reset_index(drop=True)
    pbp = build_pbp(raw, game_index)
    tables = {"weeks": weeks, "game_index": game_index, "teams": build_teams(store),
              "game_pregame": build_game_pregame(store), "pbp": pbp,
              "team_game": build_team_game(store, game_index, pbp)}
    core = out / "core"
    core.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(core / f"{name}.parquet", index=False)
    print(f"core tables written to {core} ({len(pbp)} plays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
