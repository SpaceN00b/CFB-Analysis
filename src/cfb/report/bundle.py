"""The data behind the weekly report and the dashboard, as one JSON-ready dict.

    bundle = build_bundle(warehouse, slate, predictions, config)

`predictions` maps model_id to that model's predictions for the season through the target
week (the frame `walk_forward` returns). Everything the report and the dashboard show
comes from the bundle, so both say the same thing, and `bundle.json` is the record of what
a run predicted.

Scoring uses only results visible at `slate.asof`; the headline scope is FBS vs FBS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..core.schemas import parse_reasons, read_metrics
from ..eval.bootstrap import paired_bootstrap
from ..eval.metrics import calibration, score_block, scored_games
from ..eval.points import block as points_block
from ..features.market import MarketLines
from .config import ReportConfig
from .rankings import build_rankings
from .slate import Slate, known_results

BUNDLE_VERSION = 1
AP_POLL = "AP Top 25"
COACHES_POLL = "Coaches Poll"
TOP_REASONS = 3

MODEL_LABELS = {
    "model_v1": "Model v1",
    "model_v2": "Model v2 (player priors)",
    "baseline_close": "Closing spread",
    "baseline_moneyline": "Moneyline (no vig)",
    "baseline_elo": "CFBD Elo + home",
    "baseline_home": "Home team",
}
MODEL_NOTES = {
    "model_v1": "Win probability from the frozen W8 model (power rating and Elo gaps); spread and total from the "
                "phase 2 points models. Refit each week on results visible at the cutoff.",
    "baseline_close": "Spread converted to a probability. For games not yet played this is the line at the last data pull.",
    "baseline_moneyline": "Moneylines with the vig removed (2021 on). Upcoming games: line at the last data pull.",
    "baseline_elo": "Logistic fit on CFBD pregame Elo gap and home field.",
    "baseline_home": "Same probability for every home team; 0.5 at neutral sites.",
}

# Team-level feature columns shown in team profiles and the matchup view, when the feature
# files exist. (column, label, higher_is_better, format). Missing columns are skipped.
PROFILE_FEATURES = [
    ("str_pwr_net", "Power rating, net (pts vs avg FBS, neutral)", True, "+.1f"),
    ("str_pwr_off", "Offense (pts scored vs avg defense)", True, ".1f"),
    ("str_pwr_def", "Defense (pts allowed vs avg offense)", False, ".1f"),
    ("str_elo", "In-house Elo", True, ".0f"),
    ("str_sos_played", "Schedule strength so far (opp. net rating)", True, "+.1f"),
    ("str_sor", "Strength of record (wins vs a 25th-best team)", True, "+.2f"),
    ("str_form_pts3", "Form: last 3 games vs rating (pts)", True, "+.1f"),
    ("mat_off_run_rate_neutral", "Run rate, neutral situations", None, ".0%"),
    ("mat_off_plays", "Offensive plays per game", None, ".1f"),
    ("mat_off_expl_rate", "Explosive plays per play (offense)", True, ".1%"),
    ("mat_def_havoc", "Havoc created per play (defense)", True, ".1%"),
    ("coach_tenure_seasons", "Head coach season at school", None, ".0f"),
    ("coach_fourth_go_oe", "Coach 4th-down go rate above expected", None, "+.1%"),
    ("coach_run_oe", "Coach run rate above expected", None, "+.1%"),
    ("ctx_qb_leader_share", "Last game's QB: share of season attempts", None, ".0%"),
]
# Game-specific columns for this week's games, shown in the matchup view for each team.
MATCHUP_FEATURES = [
    ("str_elo_win_prob", "In-house Elo win probability", True, ".0%"),
    ("mat_off_edge_rush_epa", "Rush offense vs their rush defense (EPA/play edge)", True, "+.3f"),
    ("mat_off_edge_pass_epa", "Pass offense vs their pass defense (EPA/play edge)", True, "+.3f"),
    ("mat_off_edge_mix_epa", "Offense vs their defense, play-mix weighted (EPA/play)", True, "+.3f"),
    ("coach_flex_shift", "Expected run-rate shift vs this defense", None, "+.1%"),
    ("ctx_rest_days", "Days of rest", None, ".0f"),
    ("ctx_travel_km", "Travel (km)", None, ".0f"),
]


# ---- small helpers ------------------------------------------------------------------------

def _f(x, digits: int = 4):
    """JSON-safe float (None for missing)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _local(ts, tz: ZoneInfo, tbd: bool = False) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts).tz_convert(tz)
    # No "%-d" / "%-I": glibc-only, Windows strftime rejects them.
    day = f"{t:%a %b} {t.day}"
    if tbd:
        return f"{day}, time TBD"
    hour = t.hour % 12 or 12
    return f"{day}, {hour}:{t:%M %p} " + (t.tzname() or "")


def _month_day(t: pd.Timestamp) -> str:
    return f"{t:%b} {t.day}"


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _block(y, p) -> dict:
    b = score_block(np.asarray(y, float), np.asarray(p, float), with_calibration=False)
    return {k: _f(v) if k != "n" else v for k, v in b.items()}


# ---- inputs -------------------------------------------------------------------------------

def _team_table(warehouse, slate: Slate) -> pd.DataFrame:
    view = warehouse.view(slate.asof)
    teams = view.table("teams", seasons=slate.season)
    return teams.drop_duplicates("team_id", keep="last").set_index("team_id")


def _ranks(warehouse, slate: Slate) -> dict[str, dict[int, int]]:
    """Latest AP and Coaches ranks visible at the slate's moment."""
    view = warehouse.view(slate.asof)
    if not view.has("rankings"):
        return {}
    r = view.table("rankings", seasons=slate.season)
    out = {}
    for key, poll in (("ap", AP_POLL), ("coaches", COACHES_POLL)):
        p = r[r["poll"] == poll]
        if p.empty:
            continue
        order = p.assign(_t=(p["seasonType"] == "postseason").astype(int))
        last = order.sort_values(["_t", "week"]).iloc[-1]
        latest = p[(p["seasonType"] == last["seasonType"]) & (p["week"] == last["week"])]
        out[key] = {int(t): int(k) for t, k in zip(latest["teamId"], latest["rank"]) if pd.notna(t)}
    return out


def _market(warehouse, slate: Slate, games: pd.DataFrame) -> pd.DataFrame:
    """Home-oriented line values (mkt_ columns) for the given games, as the lines table
    holds them now: closing lines for played games, the line at the last pull otherwise."""
    if games.empty:
        return pd.DataFrame(columns=["game_id"])
    view = warehouse.view(max(slate.asof, games["cutoff_utc"].max()))
    if not view.has("lines"):
        return pd.DataFrame(columns=["game_id"])
    return MarketLines().build(view, games[["game_id"]])


def _elo(warehouse, slate: Slate, games: pd.DataFrame) -> pd.DataFrame:
    view = warehouse.view(max(slate.asof, games["cutoff_utc"].max()) if len(games) else slate.asof)
    if not view.has("game_pregame"):
        return pd.DataFrame(columns=["game_id", "home_pregame_elo", "away_pregame_elo"])
    gp = view.table("game_pregame")
    return gp[gp["game_id"].isin(games["game_id"])]


def _records(outcomes: pd.DataFrame, season: int, asof: pd.Timestamp) -> dict[int, dict]:
    """W-L (all games and conference games) from results visible at `asof`."""
    done = known_results(outcomes, asof)
    done = done[done["season"] == season]
    rec: dict[int, dict] = {}
    for _, g in done.iterrows():
        if g["home_points"] == g["away_points"]:
            continue
        home_won = g["home_points"] > g["away_points"]
        for team, won in ((g["home_id"], home_won), (g["away_id"], not home_won)):
            r = rec.setdefault(int(team), {"w": 0, "l": 0, "cw": 0, "cl": 0})
            r["w" if won else "l"] += 1
            if g["conference_game"]:
                r["cw" if won else "cl"] += 1
    return rec


def _record_text(r: dict | None) -> str:
    if not r:
        return "0-0"
    return f"{r['w']}-{r['l']}"


def _top_reasons(text) -> list[dict]:
    if not isinstance(text, str):
        return []
    try:
        items = parse_reasons(text)
    except ValueError:
        return []
    return [{"text": i["text"], "effect": _f(i.get("effect")), "feature": i.get("feature"),
             "value": _f(i.get("value"))} for i in items[:TOP_REASONS]]


def _pred_lookup(predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {m: p.drop_duplicates("game_id", keep="last").set_index("game_id") for m, p in predictions.items()
            if p is not None and len(p)}


# ---- games ----------------------------------------------------------------------------------

def _game_rows(games: pd.DataFrame, slate: Slate, config: ReportConfig, preds: dict[str, pd.DataFrame],
               market: pd.DataFrame, elo: pd.DataFrame, teams: pd.DataFrame, ranks: dict, records: dict,
               outcomes: pd.DataFrame, tz: ZoneInfo, with_results: bool) -> list[dict]:
    mk = market.set_index("game_id") if len(market) else pd.DataFrame()
    el = elo.set_index("game_id") if len(elo) else pd.DataFrame()
    res = outcomes.set_index("game_id") if with_results else pd.DataFrame()
    primary = preds.get(config.primary)
    rows = []
    for _, g in games.iterrows():
        gid = int(g["game_id"])
        side = {}
        for s in ("home", "away"):
            tid = int(g[f"{s}_id"])
            conf = teams["conference"].get(tid) if tid in teams.index else None
            side[s] = {
                "id": tid, "name": str(g[f"{s}_team"]), "conference": None if pd.isna(conf) else str(conf),
                "fbs": bool(g[f"{s}_fbs"]), "rank": ranks.get("ap", {}).get(tid),
                "record": _record_text(records.get(tid)),
                "elo": _f(el.loc[gid, f"{s}_pregame_elo"], 0) if gid in el.index else None,
            }
        p = {m: (_f(df.loc[gid, "p_home_win"]) if gid in df.index else None) for m, df in preds.items()}
        row = {
            "game_id": gid, "season": int(g["season"]), "season_type": str(g["season_type"]), "week": int(g["week"]),
            "kickoff_utc": _iso(g["start_time_utc"]),
            "kickoff_local": _local(g["start_time_utc"], tz, bool(g.get("start_time_tbd", False))),
            "neutral": bool(g["neutral"]), "conference_game": bool(g["conference_game"]),
            "fbs_vs_fbs": bool(g["home_fbs"] and g["away_fbs"]),
            "home": side["home"], "away": side["away"], "p": p,
            "pick": None, "reasons": [], "market": None, "edge": None,
            "pred_margin": None, "pred_total": None, "ats": None, "ou": None, "result": None,
        }
        if primary is not None and gid in primary.index:
            ph = float(primary.loc[gid, "p_home_win"])
            pick = "home" if ph >= 0.5 else "away"
            row["pick"] = {"side": pick, "team": side[pick]["name"], "p": _f(ph if pick == "home" else 1 - ph)}
            row["reasons"] = _top_reasons(primary.loc[gid, "reasons"])
            row["pred_margin"] = _f(primary.loc[gid, "pred_margin"], 2)
            row["pred_total"] = _f(primary.loc[gid, "pred_total"], 2)
        if gid in mk.index:
            m = mk.loc[gid]
            p_mkt, source = None, None
            for mm in config.market_models:
                if p.get(mm) is not None:
                    p_mkt, source = p[mm], mm
                    break
            row["market"] = {
                "spread": _f(m.get("mkt_spread_close"), 1), "total": _f(m.get("mkt_total_close"), 1),
                "spread_open": _f(m.get("mkt_spread_open"), 1),
                "ml_home": _f(m.get("mkt_ml_home"), 0), "ml_away": _f(m.get("mkt_ml_away"), 0),
                "p_home": p_mkt, "source": source,
            }
            if p_mkt is not None and p.get(config.primary) is not None and config.primary != source:
                row["edge"] = _f(p[config.primary] - p_mkt)
        row["ats"], row["ou"] = _points_picks(row, side)
        if with_results and gid in res.index and pd.notna(res.loc[gid, "home_points"]):
            hp, ap = res.loc[gid, "home_points"], res.loc[gid, "away_points"]
            y = None if hp == ap else float(hp > ap)
            loss = {}
            for mm, pv in p.items():
                if pv is not None and y is not None:
                    q = min(max(pv, 1e-6), 1 - 1e-6)
                    loss[mm] = _f(-(y * math.log(q) + (1 - y) * math.log(1 - q)))
            row["result"] = {
                "home_points": int(hp), "away_points": int(ap), "home_won": None if y is None else bool(y),
                "winner": None if y is None else (side["home"]["name"] if y else side["away"]["name"]),
                "correct": None if (y is None or row["pick"] is None) else (row["pick"]["side"] == ("home" if y else "away")),
                "log_loss": loss,
            }
            if row["market"] and row["market"]["spread"] is not None:
                row["result"]["home_cover_margin"] = _f(hp - ap + row["market"]["spread"], 1)
            if row["market"] and row["market"]["total"] is not None:
                row["result"]["total_vs_line"] = _f(hp + ap - row["market"]["total"], 1)
            cover, over = row["result"].get("home_cover_margin"), row["result"].get("total_vs_line")
            if row["ats"] is not None and cover is not None:
                row["result"]["ats_correct"] = None if cover == 0 else ((cover > 0) == (row["ats"]["side"] == "home"))
            if row["ou"] is not None and over is not None:
                row["result"]["ou_correct"] = None if over == 0 else ((over > 0) == (row["ou"]["side"] == "over"))
        rows.append(row)
    return rows


def _points_picks(row: dict, side: dict) -> tuple[dict | None, dict | None]:
    """The primary model's side of the spread and the total, from pred_margin/pred_total
    against the market line. None without a prediction, a line, or with a zero edge."""
    m = row.get("market") or {}
    ats = ou = None
    if row.get("pred_margin") is not None and m.get("spread") is not None:
        e = row["pred_margin"] + m["spread"]     # > 0: the home team covers
        if e != 0:
            s = "home" if e > 0 else "away"
            line = m["spread"] if s == "home" else -m["spread"]
            ats = {"side": s, "team": side[s]["name"], "line": _f(line, 1), "edge": _f(abs(e), 1),
                   "model_line": _f(-row["pred_margin"], 1)}
    if row.get("pred_total") is not None and m.get("total") is not None:
        e = row["pred_total"] - m["total"]
        if e != 0:
            ou = {"side": "over" if e > 0 else "under", "line": m["total"], "edge": _f(abs(e), 1)}
    return ats, ou


def _points_record(rows: list[dict]) -> dict:
    """Won-lost-push of the primary model's spread and total sides over completed games."""
    out = {}
    for key, res, cmp in (("ats", "home_cover_margin", "ats_correct"), ("ou", "total_vs_line", "ou_correct")):
        w = l = pu = 0
        for r in rows:
            if not r["fbs_vs_fbs"] or r.get(key) is None or r["result"].get(res) is None:
                continue
            c = r["result"].get(cmp)
            if c is None:
                pu += 1
            elif c:
                w += 1
            else:
                l += 1
        out[key] = {"won": w, "lost": l, "push": pu}
    return out


def _season_points(predictions: dict[str, pd.DataFrame], outcomes: pd.DataFrame, slate: Slate,
                   config: ReportConfig, warehouse) -> dict | None:
    """Season-to-date spread and total scores of the primary model against the closing lines."""
    p = predictions.get(config.primary)
    if p is None or p.empty or p["pred_margin"].notna().sum() == 0:
        return None
    known = known_results(outcomes, slate.asof)
    known = known[(known["season"] == slate.season) & known["home_fbs"].astype(bool) & known["away_fbs"].astype(bool)]
    p = p.drop_duplicates("game_id", keep="last")
    g = p[["game_id", "pred_margin", "pred_total"]].merge(known, on="game_id", how="inner")
    if g.empty:
        return {"scope": "FBS vs FBS, results visible at the report's cutoff", "margin": None, "total": None}
    lines = _market(warehouse, slate, g)
    g = g.merge(lines[[c for c in ("game_id", "mkt_spread_close", "mkt_total_close") if c in lines.columns]],
                on="game_id", how="left")
    g["margin"] = (g["home_points"] - g["away_points"]).astype(float)
    g["total"] = (g["home_points"] + g["away_points"]).astype(float)
    return {"scope": "FBS vs FBS, results visible at the report's cutoff",
            "margin": points_block(g, "margin"), "total": points_block(g, "total")}


# ---- season metrics -----------------------------------------------------------------------

def _season_metrics(predictions: dict[str, pd.DataFrame], outcomes: pd.DataFrame, slate: Slate,
                    config: ReportConfig) -> dict:
    known = known_results(outcomes, slate.asof)
    known = known[known["season"] == slate.season]
    scored = {}
    for m, p in predictions.items():
        if p is None or p.empty:
            continue
        g = scored_games(p.drop_duplicates("game_id", keep="last"), known)
        scored[m] = g[g["fbs_vs_fbs"]]
    own = {m: {"label": model_label(m), **_block(g["y"], g["p_home_win"])} for m, g in scored.items()}

    cal = {m: calibration(g["y"].to_numpy(float), g["p_home_win"].to_numpy(float))
           for m, g in scored.items() if len(g)}

    # Common games: the primary model and every comparison model that has predictions.
    common_models = [m for m in [config.primary, *config.comparison] if m in scored and len(scored[m])]
    # Keep the common set from shrinking to the moneyline's coverage: drop models that would
    # remove more than 15% of the primary model's games, and list them separately.
    base = scored.get(config.primary)
    kept = []
    if base is not None and len(base):
        ids = set(base["game_id"])
        for m in common_models:
            inter = ids & set(scored[m]["game_id"])
            if m == config.primary or len(inter) >= 0.85 * len(base):
                kept.append(m)
                ids = inter
        frame = base[base["game_id"].isin(ids)][["game_id", "season_type", "week", "start_time_utc", "y"]]
        for m in kept:
            frame = frame.merge(scored[m][["game_id", "p_home_win"]].rename(columns={"p_home_win": m}),
                                on="game_id", how="inner")
    else:
        frame = pd.DataFrame(columns=["game_id", "season_type", "week", "y"])
    common = {
        "models": kept, "n": int(len(frame)),
        "blocks": {m: _block(frame["y"], frame[m]) for m in kept} if len(frame) else {},
    }
    weekly = []
    if len(frame):
        frame = frame.sort_values("start_time_utc")
        cum = {m: 0.0 for m in kept}
        n_cum = 0
        for (stype, wk), d in frame.groupby(["season_type", "week"], sort=False):
            n_cum += len(d)
            row = {"season_type": str(stype), "week": int(wk), "n": int(len(d)), "log_loss": {}, "cumulative": {},
                   "accuracy": {}}
            for m in kept:
                b = _block(d["y"], d[m])
                cum[m] += b["log_loss"] * len(d)
                row["log_loss"][m] = b["log_loss"]
                row["accuracy"][m] = b["accuracy"]
                row["cumulative"][m] = _f(cum[m] / n_cum)
            weekly.append(row)

    comparisons = []
    if base is not None and len(base):
        for m in config.comparison:
            if m not in scored or not len(scored[m]):
                continue
            g = base[["game_id", "y", "p_home_win"]].merge(
                scored[m][["game_id", "p_home_win"]].rename(columns={"p_home_win": "p_b"}), on="game_id")
            if len(g) < 2:
                continue
            r = paired_bootstrap(g["y"], g["p_home_win"], g["p_b"], "log_loss", n_boot=2000, seed=0)
            acc = paired_bootstrap(g["y"], g["p_home_win"], g["p_b"], "accuracy", n_boot=200, seed=0)
            comparisons.append({
                "model": m, "label": model_label(m), "n": r["n"], "primary": _f(r["a"]), "other": _f(r["b"]),
                "diff": _f(r["diff"]), "se": _f(r["se"]), "ci_lo": _f(r["ci_lo"]), "ci_hi": _f(r["ci_hi"]),
                "p_primary_better": _f(r["p_a_better"], 3), "accuracy_primary": _f(acc["a"]),
                "accuracy_other": _f(acc["b"]),
            })
    return {"scope": "FBS vs FBS, results visible at the report's cutoff", "own": own, "common": common,
            "weekly": weekly, "comparisons": comparisons, "calibration": cal}


def _validation(data_dir: Path | None, models: list[str]) -> dict:
    """Overall 2018-2025 numbers from data/eval/<model_id>.json, where they exist."""
    out = {}
    if data_dir is None:
        return out
    for m in models:
        try:
            doc = read_metrics(data_dir, m)
        except (FileNotFoundError, ValueError):
            continue
        seasons = [r["season"] for r in doc.get("by_season", [])]
        if any(s >= 2026 for s in seasons):
            continue  # not a validation document
        o = doc["overall"]
        out[m] = {"scope": doc["scope"], "seasons": f"{min(seasons)}-{max(seasons)}" if seasons else "",
                  "n": o["n"], "log_loss": o["log_loss"], "brier": o["brier"], "accuracy": o["accuracy"]}
    return out


# ---- teams ----------------------------------------------------------------------------------

def _read_features(features_dir: Path, wanted: set[str], game_ids) -> pd.DataFrame | None:
    """Team-level rows (game_id, team_id, columns) of the wanted columns for the given games."""
    import pyarrow.parquet as pq

    merged = None
    ids = pd.Index(game_ids)
    for path in sorted(Path(features_dir).glob("*.parquet")):
        try:
            cols = set(pq.ParquetFile(path).schema.names)
        except Exception:
            continue
        take = sorted(wanted & cols)
        if not take or "team_id" not in cols:
            continue
        f = pd.read_parquet(path, columns=["game_id", "team_id", *take])
        f = f[f["game_id"].isin(ids)]
        merged = f if merged is None else merged.merge(f, on=["game_id", "team_id"], how="outer")
    return merged


def _feature_values(features_dir: Path | None, slate: Slate, schedule: pd.DataFrame,
                    fbs_ids: set[int]) -> tuple[dict, dict, dict]:
    """Profile columns: latest row per team at or before the slate's last cutoff, with the
    percentile among FBS teams. Matchup columns: this week's rows per (game, team).
    Returns (profile values by team, matchup values by game and team, catalog)."""
    empty = ({}, {}, {"profile": [], "matchup": []})
    if features_dir is None or not Path(features_dir).exists():
        return empty
    sched = schedule[(schedule["season"] == slate.season) & (schedule["cutoff_utc"] <= slate.last_cutoff)]
    wanted = {c for c, *_ in PROFILE_FEATURES} | {c for c, *_ in MATCHUP_FEATURES}
    merged = _read_features(features_dir, wanted, sched["game_id"])
    if merged is None or merged.empty:
        return empty
    merged = merged.merge(sched[["game_id", "start_time_utc"]], on="game_id", how="inner")
    catalog = {"profile": [], "matchup": []}
    profile: dict = {}
    latest = merged.sort_values("start_time_utc").groupby("team_id").last()
    fbs_latest = latest[latest.index.isin(list(fbs_ids))]
    for col, label, hib, fmt in PROFILE_FEATURES:
        if col not in fbs_latest.columns or fbs_latest[col].notna().sum() == 0:
            continue
        s = fbs_latest[col].astype(float)
        catalog["profile"].append({"column": col, "label": label, "higher_is_better": hib, "format": fmt})
        pct = s.rank(pct=True)
        for tid, v in s.items():
            if pd.notna(v):
                profile.setdefault(str(int(tid)), {})[col] = {"value": _f(v), "pct": _f(pct[tid], 3)}
    week = merged[merged["game_id"].isin(slate.games["game_id"])]
    matchup: dict = {}
    for col, label, hib, fmt in MATCHUP_FEATURES:
        if col not in week.columns or week[col].notna().sum() == 0:
            continue
        catalog["matchup"].append({"column": col, "label": label, "higher_is_better": hib, "format": fmt})
        for gid, tid, v in zip(week["game_id"], week["team_id"], week[col]):
            if pd.notna(v):
                matchup.setdefault(str(int(gid)), {}).setdefault(str(int(tid)), {})[col] = _f(v)
    return profile, matchup, catalog


def _team_profiles(slate: Slate, schedule: pd.DataFrame, outcomes: pd.DataFrame, preds: dict, config: ReportConfig,
                   teams: pd.DataFrame, ranks: dict, records: dict, feature_values: dict, tz: ZoneInfo) -> dict:
    season = schedule[(schedule["season"] == slate.season) & (schedule["cutoff_utc"] <= slate.last_cutoff)]
    known = known_results(outcomes, slate.asof).set_index("game_id")
    primary = preds.get(config.primary)
    fbs_ids = set(season.loc[season["home_fbs"], "home_id"]) | set(season.loc[season["away_fbs"], "away_id"])
    out = {}
    for tid in sorted(int(t) for t in fbs_ids):
        mine = season[(season["home_id"] == tid) | (season["away_id"] == tid)].sort_values("start_time_utc")
        games = []
        for _, g in mine.iterrows():
            gid = int(g["game_id"])
            is_home = int(g["home_id"]) == tid
            opp = "away" if is_home else "home"
            p_home = float(primary.loc[gid, "p_home_win"]) if primary is not None and gid in primary.index else None
            item = {
                "game_id": gid, "week": int(g["week"]), "season_type": str(g["season_type"]),
                "date": _month_day(pd.Timestamp(g["start_time_utc"]).tz_convert(tz)),
                "opp": str(g[f"{opp}_team"]), "opp_id": int(g[f"{opp}_id"]),
                "site": "neutral" if g["neutral"] else ("home" if is_home else "away"),
                "p_win": _f(p_home if is_home else (None if p_home is None else 1 - p_home)),
                "result": None,
            }
            if gid in known.index and pd.notna(known.loc[gid, "home_points"]):
                pts = int(known.loc[gid, "home_points" if is_home else "away_points"])
                opp_pts = int(known.loc[gid, "away_points" if is_home else "home_points"])
                item["result"] = {"pts": pts, "opp_pts": opp_pts,
                                  "outcome": "W" if pts > opp_pts else ("L" if pts < opp_pts else "T")}
            games.append(item)
        rec = records.get(tid)
        conf = teams["conference"].get(tid) if tid in teams.index else None
        out[str(tid)] = {
            "id": tid, "name": str(teams["team"].get(tid)) if tid in teams.index else str(tid),
            "conference": None if pd.isna(conf) else str(conf),
            "record": _record_text(rec),
            "conf_record": f"{rec['cw']}-{rec['cl']}" if rec else "0-0",
            "rank_ap": ranks.get("ap", {}).get(tid), "rank_coaches": ranks.get("coaches", {}).get(tid),
            "games": games, "features": feature_values.get(str(tid), {}),
        }
    return out


# ---- rankings -------------------------------------------------------------------------------

def _rankings(warehouse, slate: Slate, config: ReportConfig, notes: list[str]) -> dict | None:
    """Model top N vs the polls at the report's cutoff. Display only; built after the
    predictions, never an input to them. A failure leaves the section out with a note."""
    if not config.rankings_source:
        return None
    try:
        return build_rankings(warehouse, slate.asof, config.rankings_source, config.rankings_top)
    except Exception as exc:  # the rankings should never stop the report
        notes.append(f"Model rankings could not be built: {exc}")
        return None


# ---- the bundle ---------------------------------------------------------------------------

def build_bundle(warehouse, slate: Slate, predictions: dict[str, pd.DataFrame], config: ReportConfig,
                 data_dir: str | Path | None = None, created_utc: pd.Timestamp | None = None,
                 lines_asof: pd.Timestamp | None = None, run_notes: list[str] | None = None) -> dict:
    tz = ZoneInfo(config.timezone)
    created = created_utc if created_utc is not None else pd.Timestamp.now(tz="UTC")
    outcomes = warehouse.outcomes()
    schedule = warehouse.schedule([slate.season])
    preds = _pred_lookup(predictions)
    teams = _team_table(warehouse, slate)
    ranks = _ranks(warehouse, slate)
    records = _records(outcomes, slate.season, slate.asof)
    both = pd.concat([slate.games, slate.previous], ignore_index=True)
    market = _market(warehouse, slate, both)
    elo = _elo(warehouse, slate, both)

    this_week = _game_rows(slate.games, slate, config, preds, market, elo, teams, ranks, records, outcomes, tz,
                           with_results=False)
    prev_records = _records(outcomes, slate.season, slate.previous["cutoff_utc"].min()) if len(slate.previous) else {}
    last_week = _game_rows(slate.previous, slate, config, preds, market, elo, teams, ranks, prev_records, outcomes,
                           tz, with_results=True)
    last_week = [g for g in last_week if g["result"] is not None]

    def summary(rows):
        out = {}
        for m in config.models:
            pairs = [(float(r["result"]["home_won"]), r["p"][m]) for r in rows
                     if r["fbs_vs_fbs"] and r["result"]["home_won"] is not None and r["p"].get(m) is not None]
            if pairs:
                y, p = zip(*pairs)
                out[m] = {"label": model_label(m), **_block(y, p),
                          "correct": int(sum((pp >= 0.5) == (yy == 1) for yy, pp in pairs))}
        return out

    data_path = Path(data_dir) if data_dir is not None else None
    in_scope = schedule[schedule["cutoff_utc"] <= slate.last_cutoff]
    fbs_ids = {int(t) for t in in_scope.loc[in_scope["home_fbs"], "home_id"]} | \
        {int(t) for t in in_scope.loc[in_scope["away_fbs"], "away_id"]}
    feature_values, matchup_values, feature_catalog = _feature_values(
        data_path / "features" if data_path else None, slate, schedule, fbs_ids)
    for g in this_week:
        g["matchup"] = matchup_values.get(str(g["game_id"]), {})
    notes = list(run_notes or [])
    rankings = _rankings(warehouse, slate, config, notes)
    primary_preds = predictions.get(config.primary)
    phase2 = bool(primary_preds is not None and len(primary_preds) and primary_preds["pred_margin"].notna().any())
    return {
        "bundle_version": BUNDLE_VERSION,
        "meta": {
            "season": slate.season, "season_type": slate.season_type, "week": slate.week, "label": slate.label,
            "key": slate.key, "asof_utc": _iso(slate.asof), "asof_local": _local(slate.asof, tz),
            "created_utc": _iso(created), "created_local": _local(created, tz),
            "data_through_utc": _iso(known_results(outcomes, created)["start_time_utc"].max()),
            "lines_asof_utc": _iso(lines_asof), "lines_asof_local": _local(lines_asof, tz) if lines_asof is not None else None,
            "primary": config.primary, "primary_label": model_label(config.primary),
            "comparison": config.comparison, "timezone": config.timezone, "phase2": phase2,
            "previous_label": slate.previous_label, "notes": notes,
        },
        "models": [{"id": m, "label": model_label(m), "note": MODEL_NOTES.get(m, ""),
                    "role": "primary" if m == config.primary else "baseline",
                    "n_predictions": int(len(predictions[m])) if predictions.get(m) is not None else 0}
                   for m in config.models],
        "games": this_week,
        "previous": {"label": slate.previous_label, "games": last_week, "summary": summary(last_week),
                     "points_record": _points_record(last_week) if phase2 else None},
        "season": _season_metrics(predictions, outcomes, slate, config),
        "season_points": _season_points(predictions, outcomes, slate, config, warehouse) if phase2 else None,
        "validation": _validation(data_path, config.models),
        "teams": _team_profiles(slate, schedule, outcomes, preds, config, teams, ranks, records, feature_values, tz),
        "feature_catalog": feature_catalog,
        "rankings": rankings,
    }


def write_bundle(bundle: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return p


__all__ = ["BUNDLE_VERSION", "PROFILE_FEATURES", "build_bundle", "model_label", "write_bundle"]
