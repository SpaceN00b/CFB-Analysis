"""Data access for the local app: which data dir to read, which reports exist, and the
bundle.json sections as tables. No Streamlit here, so it is testable without the UI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import settings

SEASON_TYPE_ORDER = {"regular": 0, "postseason": 1}


def newest_report_key(data_dir: Path) -> tuple:
    reports = list_reports(data_dir)
    return report_sort_key(reports[0][0]) if reports else (0, 0, 0)


def resolve_data_dir() -> Path:
    """Which data the pages read: the local engine's data dir (settings.data_dir()) or the copy
    synced from the cloud (Data > Sync now), whichever has the newer report week; the local data
    wins a tie, because the engine ran on this PC. Without a synced copy: settings.data_dir()."""
    local = settings.data_dir()
    try:
        from ..warehouse_sync import current_data_dir
        synced = current_data_dir(settings.warehouse_dir())
    except Exception:
        synced = None
    if synced is None:
        return local
    return local if newest_report_key(local) >= newest_report_key(Path(synced)) else Path(synced)


def data_version(data_dir: Path) -> dict | None:
    """The manifest the synced copy came from, if the folder holds one."""
    for name in ("manifest.json", "MANIFEST.json"):
        p = Path(data_dir) / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def report_sort_key(key: str) -> tuple:
    """`2026_regular_12` -> (2026, 0, 12); unknown shapes sort first."""
    parts = key.split("_")
    try:
        return int(parts[0]), SEASON_TYPE_ORDER.get(parts[1], 9), int(parts[2])
    except (IndexError, ValueError):
        return (0, 0, 0)


def list_reports(data_dir: Path) -> list[tuple[str, Path]]:
    """(key, folder) of every report with a bundle.json, newest first."""
    root = Path(data_dir) / "reports"
    if not root.exists():
        return []
    found = [(p.name, p) for p in root.iterdir() if p.is_dir() and (p / "bundle.json").exists()]
    found = [f for f in found if report_sort_key(f[0]) != (0, 0, 0)]
    return sorted(found, key=lambda f: report_sort_key(f[0]), reverse=True)


def load_bundle(folder: Path) -> dict:
    return json.loads((Path(folder) / "bundle.json").read_text(encoding="utf-8"))


def model_label(bundle: dict, model_id: str) -> str:
    for m in bundle.get("models", []):
        if m.get("id") == model_id:
            return m.get("label", model_id)
    return model_id


def games_frame(bundle: dict, games: list | None = None) -> pd.DataFrame:
    """One row per game of the week (or of `games`), for the slate table."""
    primary = bundle["meta"]["primary"]
    rows = []
    for g in games if games is not None else bundle.get("games", []):
        home, away = g["home"], g["away"]
        market = g.get("market") or {}
        ats = g.get("ats") or {}
        ou = g.get("ou") or {}
        pick = g.get("pick") or {}
        res = g.get("result") or {}
        rows.append({
            "game_id": g["game_id"],
            "kickoff": g.get("kickoff_local"),
            "kickoff_utc": g.get("kickoff_utc"),
            "away": _team_label(away),
            "home": _team_label(home),
            "neutral": bool(g.get("neutral")),
            "fbs_vs_fbs": bool(g.get("fbs_vs_fbs")),
            "conferences": sorted({c for c in (home.get("conference"), away.get("conference")) if c}),
            "pick": pick.get("team"),
            "p_pick": pick.get("p"),
            "p_home": (g.get("p") or {}).get(primary),
            "market_p_home": market.get("p_home"),
            "edge": g.get("edge"),
            "model_spread": ats.get("model_line"),
            "market_spread": market.get("spread"),
            "ats_pick": ats.get("team"),
            "ats_edge": ats.get("edge"),
            "pred_total": g.get("pred_total"),
            "market_total": market.get("total"),
            "ou_pick": ou.get("side"),
            "ou_edge": ou.get("edge"),
            "score": (f"{away['name']} {res['away_points']} - {home['name']} {res['home_points']}"
                      if res.get("home_points") is not None else None),
            "pick_correct": res.get("correct"),
        })
    return pd.DataFrame(rows)


def _team_label(team: dict) -> str:
    rank = team.get("rank")
    return f"#{rank} {team['name']}" if rank else team["name"]


def game_by_id(bundle: dict, game_id: int) -> dict | None:
    for g in bundle.get("games", []) + (bundle.get("previous") or {}).get("games", []):
        if g["game_id"] == game_id:
            return g
    return None


def own_metrics_frame(bundle: dict) -> pd.DataFrame:
    own = (bundle.get("season") or {}).get("own") or {}
    rows = [{"model": v.get("label", k), "model_id": k, **{m: v.get(m) for m in ("n", "log_loss", "brier", "accuracy")}}
            for k, v in own.items()]
    return pd.DataFrame(rows)


def weekly_frame(bundle: dict, metric: str = "cumulative") -> pd.DataFrame:
    """Week x model table of season-to-date (`cumulative`) or weekly `log_loss` / `accuracy`."""
    rows = []
    for w in (bundle.get("season") or {}).get("weekly", []):
        vals = w.get(metric) or {}
        rows.append({"week": w["week"], **{model_label(bundle, k): v for k, v in vals.items()}})
    return pd.DataFrame(rows).set_index("week") if rows else pd.DataFrame()


def calibration_frame(bundle: dict, model_id: str) -> pd.DataFrame:
    return pd.DataFrame(((bundle.get("season") or {}).get("calibration") or {}).get(model_id, []))


def rankings_frame(bundle: dict) -> pd.DataFrame:
    rk = bundle.get("rankings") or {}
    rows = []
    for t in rk.get("teams", []):
        polls = t.get("polls") or {}
        rows.append({"rank": t.get("rank"), "move": t.get("move"), "team": t.get("team"),
                     "conference": t.get("conference"), "record": t.get("record"), "rating": t.get("rating"),
                     "off": t.get("off"), "def": t.get("def"), "sor_rank": t.get("sor_rank"),
                     "AP": polls.get("ap"), "Coaches": polls.get("coaches"), "CFP": polls.get("cfp")})
    return pd.DataFrame(rows)


def teams_list(bundle: dict) -> list[tuple[int, str]]:
    """(team_id, name) of the bundle's teams, by name."""
    teams = bundle.get("teams") or {}
    return sorted(((int(k), v["name"]) for k, v in teams.items()), key=lambda t: t[1])


def team_games_frame(bundle: dict, team_id: int) -> pd.DataFrame:
    team = (bundle.get("teams") or {}).get(str(team_id))
    if not team:
        return pd.DataFrame()
    rows = []
    for g in team.get("games", []):
        r = g.get("result") or {}
        rows.append({"week": g.get("week"), "date": g.get("date"), "opponent": g.get("opp"), "site": g.get("site"),
                     "p_win": g.get("p_win"), "result": (f"{r['outcome']} {r['pts']}-{r['opp_pts']}" if r else None)})
    return pd.DataFrame(rows)
