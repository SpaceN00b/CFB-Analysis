"""Predictions from the frozen W8 model (`cfb.model.config`).

    python -m cfb.predict season-forward            # 2018-2025, refit once per season, writes data/predictions/model_v1/
    python -m cfb.predict walk-forward --seasons 2026   # refit at every cutoff (the 2026 test run)
    python -m cfb.predict matchup "Ohio State" "Michigan" [--neutral] [--cutoff 2026-11-22] [--qb-out "Michigan"]

Training and slate features come from the stored feature files (data/features/, each game
built at its own cutoff by `python -m cfb.features.build`). A game that is in no stored file
(an on-demand matchup) is built live through the view at its cutoff, with the same groups.

For W9 (`run_week`): `produce(warehouse, games)` returns the walk_forward frame
(prediction schema + season, season_type, week) for `games`, refitting at every cutoff.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .core.asof import Warehouse
from .core.schemas import reasons_json
from .core.timeutil import as_utc
from .eval.splits import walk_forward, write_weekly
from .features.base import discover_groups
from .model import config as C
from .model import points_config as P
from .model.matrix import game_matrix, groups_by_name, live_features, stored_features

N_REASONS = 5


@dataclass
class Fitted:
    model: object
    n_train: int
    cutoff: pd.Timestamp
    margin_model: object = None
    total_model: object = None


class Predictor:
    """Holds the stored features once, so repeated fits at many cutoffs stay fast."""

    def __init__(self, warehouse: Warehouse, data_dir: str | Path | None = None,
                 groups: list[str] | None = None, features: dict[str, pd.DataFrame] | None = None):
        """`groups` and `features` (frames as the feature build writes them) override the
        configured groups and the stored files; tests use them."""
        self.wh = warehouse
        self.data_dir = Path(data_dir) if data_dir is not None else warehouse.root
        self.group_names = groups
        names = list(groups or C.MODEL_GROUPS)
        if groups is None:
            for t in ("margin", "total"):
                names += [g for g in P.matrix_groups(t) if g not in names]
        self.groups = groups_by_name(names)
        self._stored = features
        self._matrix = None
        self._descr = {c.name: c.description for g in discover_groups() for c in g.columns}

    # ---- features ----
    def stored(self) -> dict[str, pd.DataFrame]:
        if self._stored is None:
            self._stored = stored_features(self.data_dir, self.groups)
        return self._stored

    def matrix_all(self) -> pd.DataFrame:
        """Stored-feature matrix of every scheduled game (no results), indexed by game_id."""
        if self._matrix is None:
            sched = self.wh.schedule()
            self._matrix = game_matrix(sched, self.stored(), self.groups).set_index("game_id", drop=False)
        return self._matrix

    def matrix_for(self, view, games: pd.DataFrame) -> pd.DataFrame:
        games = games.reset_index(drop=True)
        M = self.matrix_all()
        known = games["game_id"].isin(M.index)
        parts = []
        if known.any():
            parts.append(M.loc[games.loc[known, "game_id"].to_numpy()].reset_index(drop=True))
        if (~known).any():
            g = games[~known].reset_index(drop=True)
            parts.append(game_matrix(g, live_features(view, g, self.groups), self.groups))
        return pd.concat(parts, ignore_index=True)

    # ---- walk_forward callables ----
    def fit(self, view, train: pd.DataFrame) -> Fitted:
        t = train[(train["season"] >= C.TRAIN_START)]
        margin = (t["home_points"] - t["away_points"]).to_numpy(dtype="float64")
        keep = margin != 0
        t, margin = t[keep], margin[keep]
        X = self.matrix_all().loc[t["game_id"].to_numpy()].reset_index(drop=True)
        model = C.make_model(self.group_names).fit(X, (margin > 0).astype(float), margin)
        # Points models (phase 2): every completed game from P.TRAIN_START, ties included.
        tp = train[train["season"] >= P.TRAIN_START]
        Xp = self.matrix_all().loc[tp["game_id"].to_numpy()].reset_index(drop=True)
        hp, ap = tp["home_points"].to_numpy(dtype="float64"), tp["away_points"].to_numpy(dtype="float64")
        points = {}
        for target, y in (("margin", hp - ap), ("total", hp + ap)):
            points[target] = P.make_model(target, self.group_names).fit(Xp, P.clip_target(target, y))
        return Fitted(model, len(X), view.cutoff, points["margin"], points["total"])

    def predict(self, fitted: Fitted, view, slate: pd.DataFrame) -> pd.DataFrame:
        X = self.matrix_for(view, slate)
        p = fitted.model.predict_proba(X)
        contrib = fitted.model.contributions(X)
        reasons = [self._reasons(contrib.iloc[i], X.iloc[i]) for i in range(len(X))]
        out = pd.DataFrame({"game_id": X["game_id"].to_numpy(), "p_home_win": p, "reasons": reasons})
        # Phase 2: spread and total from the separately selected points models
        # (cfb.model.points_config). p_home_win stays the frozen model_v1.
        if fitted.margin_model is not None:
            out["pred_margin"] = fitted.margin_model.predict(X)
        if fitted.total_model is not None:
            out["pred_total"] = np.maximum(fitted.total_model.predict(X), 0.0)
        return out

    def _reasons(self, contrib: pd.Series, row: pd.Series) -> str:
        top = contrib[contrib.abs() > 0.01].abs().sort_values(ascending=False).index[:N_REASONS]
        items = []
        for name in top:
            if name.startswith("missing:"):
                items.append({"text": f"No data yet for {name[8:]}", "feature": name,
                              "effect": round(float(contrib[name]), 4)})
                continue
            base = name[2:] if name[:2] in ("d_", "h_", "a_", "s_") else name
            descr = _label(self._descr.get(base, base))
            if name.startswith("d_"):
                h, a = row.get(f"h_{base}"), row.get(f"a_{base}")
                text = f"{descr}: home {_fmt(h)}, away {_fmt(a)}"
            else:
                text = f"{descr}: {_fmt(row.get(name))}"
            v = row.get(name)
            items.append({"text": text, "feature": name, "effect": round(float(contrib[name]), 4),
                          "value": None if v is None or pd.isna(v) else round(float(v), 4)})
        return reasons_json(items)


def _label(description: str) -> str:
    """Short label from a catalog description: the text before the first ':' or ' ('."""
    cut = len(description)
    for sep in (":", " (", ";"):
        k = description.find(sep)
        if 0 < k < cut:
            cut = k
    return description[:cut].strip()


def _fmt(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}"


def run(warehouse: Warehouse, seasons, refit: str, games: pd.DataFrame | None = None,
        progress=None) -> pd.DataFrame:
    pr = Predictor(warehouse)
    return walk_forward(warehouse, pr.fit, pr.predict, C.MODEL_ID, seasons=seasons, games=games,
                        refit=refit, progress=progress)


def produce(warehouse: Warehouse, games: pd.DataFrame) -> pd.DataFrame:
    """Entry point for cfb.run_week: predictions for `games`, refit at every cutoff."""
    seasons = sorted(games["season"].unique().tolist())
    return run(warehouse, seasons, "cutoff", games=games)


def matchup(warehouse: Warehouse, home: str, away: str, neutral: bool = False, cutoff=None,
            qb_out: dict | None = None) -> dict:
    """On-demand prediction for two teams (names or ids) as of `cutoff` (default: now).
    `qb_out`: {team: "out" | replacement name}, the manual override of cfb.model.qb_out."""
    cutoff = as_utc(cutoff) if cutoff is not None else pd.Timestamp.now(tz="UTC")
    view = warehouse.view(cutoff)
    teams = view.table("teams", seasons=view.season)
    def tid(x):
        if str(x).isdigit():
            return int(x)
        hit = teams[teams["team"].str.lower() == str(x).lower()]
        if hit.empty:
            raise SystemExit(f"unknown team {x!r} in season {view.season}")
        return int(hit["team_id"].iloc[0])
    h, a = tid(home), tid(away)
    fbs = dict(zip(teams["team_id"], teams["fbs"]))
    name = dict(zip(teams["team_id"], teams["team"]))
    game = pd.DataFrame([{
        "game_id": -1, "season": view.season, "season_type": "regular", "week": 0,
        "start_time_utc": cutoff, "cutoff_utc": cutoff, "neutral": bool(neutral),
        "conference_game": False, "venue_id": pd.NA, "home_id": h, "away_id": a,
        "home_fbs": bool(fbs.get(h, False)), "away_fbs": bool(fbs.get(a, False)),
        "home_team": name.get(h, str(home)), "away_team": name.get(a, str(away)), "start_time_tbd": False,
    }])
    pr = Predictor(warehouse)
    from .eval.splits import visible_results
    fitted = pr.fit(view, visible_results(warehouse, cutoff))
    pred = pr.predict(fitted, view, game)
    adj = {}
    if qb_out:
        from .model import qb_out as QO
        adj = QO.resolve(view, qb_out)
        pred = QO.apply(pred, game, adj, logit_per_point=float(getattr(fitted.model, "calib", (0, QO.LOGIT_PER_POINT))[1]))
    out = pred.iloc[0]
    return {"home": name.get(h), "away": name.get(a), "neutral": bool(neutral), "cutoff_utc": str(cutoff),
            "p_home_win": float(out["p_home_win"]), "reasons": json.loads(out["reasons"]),
            **({"qb_out": list(adj.values())} if adj else {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.predict", description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("season-forward", help="2018-2025, refit once per season")
    s.add_argument("--seasons", default="2018-2025")
    w = sub.add_parser("walk-forward", help="refit at every cutoff (the 2026 test run)")
    w.add_argument("--seasons", required=True)
    m = sub.add_parser("matchup")
    m.add_argument("home")
    m.add_argument("away")
    m.add_argument("--neutral", action="store_true")
    m.add_argument("--cutoff", default=None)
    m.add_argument("--qb-out", action="append", default=[], metavar="TEAM[=REPLACEMENT]",
                   help="manual override: the team's expected starter is out (repeatable)")
    args = ap.parse_args(argv)
    wh = Warehouse(args.data_dir)
    if args.cmd == "matchup":
        qbo = dict((x.split("=", 1) + ["out"])[:2] for x in args.qb_out)
        print(json.dumps(matchup(wh, args.home, args.away, args.neutral, args.cutoff, qbo), indent=2, default=str))
        return 0
    from .features.build import parse_seasons
    seasons = parse_seasons(args.seasons)
    refit = "season" if args.cmd == "season-forward" else "cutoff"
    def prog(i, n):
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n} cutoffs", flush=True)
    preds = run(wh, seasons, refit, progress=prog)
    # keep other seasons' files: write only the seasons predicted here
    paths = write_weekly(preds, args.data_dir, replace=False)
    print(f"{len(preds)} predictions for {C.MODEL_ID}, {len(paths)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
