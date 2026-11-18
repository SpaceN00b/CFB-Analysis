"""Model rankings: the top FBS teams by a team-strength rating, side by side with the AP,
Coaches and playoff committee (CFP) top 25.

    rk = build_rankings(warehouse, asof, source="power", top=50)

The rankings are for reading only. Nothing here is a feature group (feature discovery scans
`cfb.features` only), nothing is written to `data/features` or `data/predictions`, and the
weekly run builds them after every model has predicted, so they cannot change a prediction.

Rating sources. A source is a function `rate(view) -> DataFrame` with columns `team_id` and
`rating` (higher is better) and, optionally, `off`, `def`, `se`, `games`. It sees only the
AsOfView at the ranking's moment, like feature code, so a ranking as of a week's cutoff uses
the same information as that week's predictions. Built in:

- `power`: the W3 power rating, net points against an average FBS team on a neutral field
  (`cfb.features.strength.power`). Default.
- `elo`: the in-house Elo (`cfb.features.strength.elo`).

Any other source is given as "module:function"; the trained model can be plugged in that way
(for example, a function that rates each team by its expected margin or win probability
against an average FBS team on a neutral field). The function may set `label` and `unit`
attributes for display.

Polls: the latest AP, Coaches and committee rankings visible at the moment (the committee
ranking is published on Tuesday evenings, so a Monday run shows the previous week's). Each
poll is labeled with its own week.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .. import settings

POLLS = (("ap", "AP Top 25", "AP"), ("coaches", "Coaches Poll", "Coaches"),
         ("cfp", "Playoff Committee Rankings", "CFP"))
POLL_SIZE = 25
DEFAULT_TOP = 50
DISAGREEMENTS = 5
# Strength of record compares a team's wins with what the 25th-best FBS team would expect
# against the same schedule (as in the str_schedule feature group).
SOR_REFERENCE = 25

RateFn = Callable[[object], pd.DataFrame]


@dataclass(frozen=True)
class RatingSource:
    id: str
    label: str
    unit: str
    note: str
    rate: RateFn


# ---- built-in sources -------------------------------------------------------------------------

def _rate_power(view) -> pd.DataFrame:
    from ..features.strength.power import POINTS_SD, POWER, power_state

    st = power_state(view)
    ids = _fbs_ids(view)
    p = st.lookup(ids)
    se = POINTS_SD * np.sqrt(2.0 / (POWER.lam + p["games"].to_numpy()))
    return pd.DataFrame({"team_id": ids, "rating": p["net"].to_numpy(), "off": p["off"].to_numpy(),
                         "def": p["def"].to_numpy(), "se": se, "games": p["games"].to_numpy()})


def _rate_elo(view) -> pd.DataFrame:
    from ..features.strength import elo as E

    teams = view.table("teams", seasons=view.season)
    r = E.ratings_at(view, view.season, teams)
    ids = _fbs_ids(view)
    return pd.DataFrame({"team_id": ids, "rating": r.reindex(ids).to_numpy(dtype=np.float64)})


BUILTIN_SOURCES = {
    "power": RatingSource(
        "power", "Power rating", "pts",
        "Net points per game against an average FBS team on a neutral field: opponent-adjusted ridge on "
        "points this season, with the preseason projection as prior (W3 str_power).", _rate_power),
    "elo": RatingSource(
        "elo", "In-house Elo", "Elo",
        "In-house Elo from all visible games (margin-of-victory multiplier, season regression).", _rate_elo),
}


def resolve_source(spec: str | RatingSource | None) -> RatingSource:
    """A built-in source by id, a RatingSource, or "module:function"."""
    if isinstance(spec, RatingSource):
        return spec
    spec = spec or "power"
    if spec in BUILTIN_SOURCES:
        return BUILTIN_SOURCES[spec]
    if ":" not in spec:
        raise ValueError(f"unknown rating source {spec!r}: use {', '.join(BUILTIN_SOURCES)} or 'module:function'")
    module, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(module), name)
    return RatingSource(spec, str(getattr(fn, "label", spec)), str(getattr(fn, "unit", "")),
                        str(getattr(fn, "note", "")), fn)


# ---- helpers --------------------------------------------------------------------------------

def _f(x, digits: int = 2):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _i(x):
    return None if x is None or pd.isna(x) else int(x)


def _iso(ts) -> str | None:
    return None if ts is None else pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _fbs_ids(view) -> np.ndarray:
    teams = view.table("teams", seasons=view.season)
    teams = teams.drop_duplicates("team_id", keep="last")
    return np.sort(teams.loc[teams["fbs"].astype(bool), "team_id"].to_numpy(dtype=np.int64))


def _ranked(rating: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted best first with a 1-based `rank`; teams without a rating go last, unranked."""
    r = rating.copy()
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    r = r.sort_values(["rating", "team_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    r["rank"] = np.where(r["rating"].notna(), np.arange(1, len(r) + 1), np.nan)
    return r


def _records(view) -> pd.DataFrame:
    """Wins and losses this season from results visible in the view."""
    tg = view.table("team_game", columns=["team_id", "win", "points", "opp_points"], seasons=view.season)
    tg = tg[tg["points"].notna() & tg["opp_points"].notna() & (tg["points"] != tg["opp_points"])]
    if tg.empty:
        return pd.DataFrame(columns=["wins", "losses"])
    w = tg["points"] > tg["opp_points"]
    return pd.DataFrame({"wins": w.groupby(tg["team_id"]).sum(), "losses": (~w).groupby(tg["team_id"]).sum()})


def strength_of_record(view, ids) -> pd.Series:
    """Wins minus the wins the 25th-best FBS team (by power rating) would expect against the
    same opponents and sites. Same definition as `str_sor` in cfb.features.strength.groups;
    tests/test_rankings.py checks that they agree."""
    from ..features.strength.power import power_state, win_prob

    st = power_state(view)
    played = view.table("team_game", columns=["team_id", "opp_id", "is_home", "neutral", "win", "points"],
                        seasons=view.season)
    played = played[played["team_id"].isin(ids) & played["points"].notna()]
    fbs_net = np.sort(st.fit.off[:-1] - st.fit.deff[:-1])[::-1]
    if not len(fbs_net):
        return pd.Series(np.nan, index=pd.Index(ids, name="team_id"))
    ref = fbs_net[SOR_REFERENCE - 1] if len(fbs_net) >= SOR_REFERENCE else fbs_net[-1]
    if played.empty:
        return pd.Series(0.0, index=pd.Index(ids, name="team_id"))
    hfa2 = 2.0 * st.fit.hfa if np.isfinite(st.fit.hfa) else 0.0
    opp_net = st.lookup(played["opp_id"])["net"].to_numpy()
    h = np.where(played["neutral"].fillna(False), 0.0, np.where(played["is_home"], 1.0, -1.0))
    exp_ref = win_prob(ref - opp_net + h * hfa2)
    sor = (played["win"].astype("float64").to_numpy() - exp_ref)
    return pd.Series(sor, index=played["team_id"].to_numpy()).groupby(level=0).sum().reindex(ids).fillna(0.0)


def latest_polls(view) -> dict[str, dict]:
    """The latest visible AP, Coaches and committee top 25 of the view's season:
    {key: {"name", "short", "season_type", "week", "ranks": {team_id: rank}, "points": {...}}}."""
    if not view.has("rankings"):
        return {}
    r = view.table("rankings", seasons=view.season)
    out = {}
    for key, poll, short in POLLS:
        p = r[(r["poll"] == poll) & r["teamId"].notna() & (r["rank"] <= POLL_SIZE)]
        if p.empty:
            continue
        order = p.assign(_t=(p["seasonType"] == "postseason").astype(int))
        last = order.sort_values(["_t", "week"]).iloc[-1]
        latest = p[(p["seasonType"] == last["seasonType"]) & (p["week"] == last["week"])]
        out[key] = {
            "name": poll, "short": short, "season_type": str(last["seasonType"]), "week": int(last["week"]),
            "ranks": {int(t): int(k) for t, k in zip(latest["teamId"], latest["rank"])},
            "points": {int(t): _i(v) for t, v in zip(latest["teamId"], latest["points"])},
        }
    return out


def _spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _compare(poll: dict, model_rank: dict[int, int], names: dict[int, str]) -> dict:
    """How the model's order agrees with one poll's top 25."""
    ranks = poll["ranks"]
    teams = [t for t in ranks if model_rank.get(t) is not None]
    poll_r = [ranks[t] for t in teams]
    model_r = [model_rank[t] for t in teams]
    model_top = {t for t, k in model_rank.items() if k is not None and k <= POLL_SIZE}
    diffs = [m - p for m, p in zip(model_r, poll_r)]
    lower = sorted(({"team_id": t, "team": names.get(t, str(t)), "poll_rank": ranks[t], "model_rank": model_rank[t],
                     "diff": model_rank[t] - ranks[t]} for t in teams), key=lambda d: -d["diff"])
    higher = sorted(({"team_id": t, "team": names.get(t, str(t)), "poll_rank": ranks.get(t),
                      "model_rank": model_rank[t]} for t in model_top if t not in ranks),
                    key=lambda d: d["model_rank"])
    return {
        "poll": poll["short"], "week": poll["week"], "season_type": poll["season_type"], "n": len(teams),
        "overlap": len(model_top & set(ranks)), "spearman": _f(_spearman(poll_r, model_r), 3),
        "mean_abs_diff": _f(np.mean(np.abs(diffs)) if diffs else None, 1),
        "model_lower": [d for d in lower if d["diff"] > 0][:DISAGREEMENTS],
        "model_only": higher,
    }


# ---- the ranking ----------------------------------------------------------------------------

def build_rankings(warehouse, asof, source: str | RatingSource | None = "power", top: int = DEFAULT_TOP,
                   previous_asof=None) -> dict:
    """JSON-ready rankings as of `asof` (normally the report's cutoff).

    Rows: the model's top `top` FBS teams, then any team in a poll's top 25 that the model
    ranks lower (`in_top` false), so every poll team can be compared. `previous_asof`
    (default: one week earlier, same season only) gives each team's rank movement."""
    src = resolve_source(source)
    asof = pd.Timestamp(asof)
    view = warehouse.view(asof)
    rating = src.rate(view)
    missing = {"team_id", "rating"} - set(rating.columns)
    if missing:
        raise ValueError(f"rating source {src.id} returned no {sorted(missing)} column")
    fbs = set(_fbs_ids(view).tolist())
    rating = rating[rating["team_id"].isin(fbs)]
    ranked = _ranked(rating)
    model_rank = {int(t): _i(k) for t, k in zip(ranked["team_id"], ranked["rank"])}

    prev_rank: dict[int, int] = {}
    prev_ts = pd.Timestamp(previous_asof) if previous_asof is not None else asof - pd.Timedelta(days=7)
    pview = warehouse.view(prev_ts)
    if pview.season == view.season and pview.data_key != view.data_key:
        pr = _ranked(src.rate(pview))
        prev_rank = {int(t): _i(k) for t, k in zip(pr["team_id"], pr["rank"])}

    teams = view.table("teams", seasons=view.season).drop_duplicates("team_id", keep="last").set_index("team_id")
    names = {int(t): str(n) for t, n in teams["team"].items()}
    records = _records(view)
    sor = strength_of_record(view, ranked["team_id"].to_numpy())
    sor_rank = sor.rank(ascending=False, method="min")
    polls = latest_polls(view)

    poll_teams = {t for p in polls.values() for t in p["ranks"]}
    show = ranked[(ranked["rank"] <= top) | ranked["team_id"].isin(poll_teams)]
    rows = []
    for _, r in show.iterrows():
        tid = int(r["team_id"])
        rec = records.loc[tid] if tid in records.index else None
        conf = teams["conference"].get(tid) if tid in teams.index else None
        rank = _i(r["rank"])
        rows.append({
            "rank": rank, "prev_rank": prev_rank.get(tid),
            "move": None if prev_rank.get(tid) is None or rank is None else prev_rank[tid] - rank,
            "in_top": rank is not None and rank <= top,
            "team_id": tid, "team": names.get(tid, str(tid)), "conference": None if pd.isna(conf) else str(conf),
            "record": f"{int(rec['wins'])}-{int(rec['losses'])}" if rec is not None else "0-0",
            "rating": _f(r["rating"]), "off": _f(r.get("off")), "def": _f(r.get("def")), "se": _f(r.get("se")),
            "sor": _f(sor.get(tid)), "sor_rank": _i(sor_rank.get(tid)),
            "polls": {k: p["ranks"].get(tid) for k, p in polls.items()},
        })
    return {
        "source": {"id": src.id, "label": src.label, "unit": src.unit, "note": src.note,
                   "columns": [c for c in ("off", "def", "se") if c in rating.columns]},
        "asof_utc": _iso(asof), "previous_asof_utc": _iso(prev_ts) if prev_rank else None,
        "season": int(view.season), "top": int(top), "n_fbs": int(len(fbs)),
        "polls": [{"key": k, "name": p["name"], "short": p["short"], "season_type": p["season_type"],
                   "week": p["week"]} for k, p in polls.items()],
        "teams": rows,
        "comparisons": {k: _compare(p, model_rank, names) for k, p in polls.items()},
    }


# ---- text -------------------------------------------------------------------------------------

def _poll_label(p: dict) -> str:
    return f"{p['short']} ({'final' if p['season_type'] == 'postseason' else 'week ' + str(p['week'])})"


def render_markdown_section(rk: dict, heading: str = "##") -> list[str]:
    """The rankings as Markdown lines (a section of the weekly report, or the CLI output)."""
    src = rk["source"]
    out = [f"{heading} Model rankings vs polls", ""]
    polls = rk["polls"]
    out.append(f"Model top {rk['top']} of {rk['n_fbs']} FBS teams by **{src['label']}**, from results visible at "
               "the report's cutoff. Rankings are for comparison only and are not a model input. "
               + (f"Polls: {', '.join(_poll_label(p) for p in polls)}." if polls else "No polls visible yet."))
    if src.get("note"):
        out.append("")
        out.append(f"{src['label']}: {src['note']}")
    out.append("")
    comps = rk["comparisons"]
    if comps:
        rows = []
        for p in polls:
            c = comps[p["key"]]
            rows.append([_poll_label(p), f"{c['overlap']}/{POLL_SIZE}",
                         "" if c["spearman"] is None else f"{c['spearman']:.2f}",
                         "" if c["mean_abs_diff"] is None else f"{c['mean_abs_diff']:.1f}"])
        out += _table(["Poll", "Same top 25", "Rank correlation", "Mean rank gap"], rows, "lrrr")
        out.append("")
        out.append("Same top 25: poll teams that are also in the model's top 25. Rank correlation: Spearman "
                   "between poll rank and model rank for the poll's 25 teams. Mean rank gap: average of "
                   "|model rank - poll rank| for those teams.")
        out.append("")
        ap_key = "ap" if "ap" in comps else polls[0]["key"]
        c = comps[ap_key]
        if c["model_lower"]:
            out.append(f"Ranked by the {c['poll']} poll, lower in the model: " + "; ".join(
                f"{d['team']} ({c['poll']} {d['poll_rank']}, model {d['model_rank']})" for d in c["model_lower"]) + ".")
            out.append("")
        if c["model_only"]:
            out.append(f"In the model's top 25, not in the {c['poll']} top 25: " + "; ".join(
                f"{d['team']} (model {d['model_rank']})" for d in c["model_only"]) + ".")
            out.append("")
    unit = f" ({src['unit']})" if src.get("unit") else ""
    header = ["Rank", "Move", "Team", "Conf", "Record", src["label"] + unit]
    extra = [c for c in ("off", "def") if c in src["columns"]]
    header += [{"off": "Off", "def": "Def"}[c] for c in extra]
    header += ["SOR rank"] + [p["short"] for p in polls]
    rfmt = {"pts": "+.1f", "Elo": ".0f"}.get(src.get("unit"), ".2f")
    rows = []
    for t in rk["teams"]:
        mv = t["move"]
        rows.append([
            t["rank"] if t["in_top"] else f"({t['rank']})" if t["rank"] else "",
            "" if mv is None else ("=" if mv == 0 else f"{mv:+d}"),
            t["team"], t["conference"] or "", t["record"], _num(t["rating"], rfmt),
            *[_num(t[c], ".1f") for c in extra],
            t["sor_rank"] or "", *[t["polls"].get(p["key"]) or "" for p in polls],
        ])
    out += _table(header, rows, "rrlll" + "r" * (len(header) - 5))
    out.append("")
    out.append(f"Rows in parentheses: teams in a poll's top 25 outside the model's top {rk['top']}. Move: places "
               "gained since the previous week's ranking. SOR rank: strength of record (wins minus a 25th-best "
               "team's expected wins against the same schedule) among FBS teams.")
    out.append("")
    return out


def _num(x, fmt: str = "+.1f") -> str:
    return "" if x is None else format(x, fmt)


def _table(header, rows, align):
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


# ---- command line -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cfb.rankings",
                                 description="Model top N vs AP, Coaches and CFP top 25, printed as Markdown.")
    ap.add_argument("--data-dir", default=str(settings.data_dir()))
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--season-type", choices=("regular", "postseason"))
    ap.add_argument("--now", help="pick the week containing this UTC time (default: now)")
    ap.add_argument("--source", default=None, help="power (default from the config), elo, or module:function")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.week is not None and args.season is None:
        ap.error("--week needs --season")

    from ..core.asof import Warehouse
    from .config import load_config
    from .slate import resolve_slate

    cfg = load_config(args.config)
    wh = Warehouse(args.data_dir)
    slate = resolve_slate(wh, args.season, args.season_type, args.week, now=args.now)
    rk = build_rankings(wh, slate.asof, source=args.source or cfg.rankings_source, top=args.top or cfg.rankings_top)
    print(f"# {slate.label}\n")
    print("\n".join(render_markdown_section(rk)))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["BUILTIN_SOURCES", "RatingSource", "build_rankings", "latest_polls", "render_markdown_section",
           "resolve_source", "strength_of_record"]
