"""The weekly report as Markdown, rendered from the bundle (see bundle.py)."""
from __future__ import annotations

from .bundle import model_label

EDGE_LIST = 10
MISSES = 5


def _pct(p, digits: int = 0) -> str:
    return "" if p is None else f"{100 * p:.{digits}f}%"


def _num(x, fmt: str = ".4f") -> str:
    return "" if x is None else format(x, fmt)


def _pp(x) -> str:
    """Probability difference in percentage points."""
    return "" if x is None else f"{100 * x:+.1f}"


def _team(side: dict) -> str:
    rank = f"#{side['rank']} " if side.get("rank") else ""
    return f"{rank}{side['name']}"


def _matchup(g: dict) -> str:
    sep = " vs " if g["neutral"] else " @ "
    return f"{_team(g['away'])}{sep}{_team(g['home'])}"


def _spread(g: dict) -> str:
    m = g.get("market") or {}
    s = m.get("spread")
    if s is None:
        return ""
    if s == 0:
        return "PK"
    fav = g["home"]["name"] if s < 0 else g["away"]["name"]
    return f"{fav} -{abs(s):g}"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    align = align or "l" * len(header)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


def _pick_p_market(g: dict) -> float | None:
    """Market probability for the primary model's pick."""
    m = g.get("market") or {}
    if m.get("p_home") is None or not g.get("pick"):
        return None
    return m["p_home"] if g["pick"]["side"] == "home" else 1 - m["p_home"]


def _pick_edge(g: dict) -> float | None:
    if g.get("edge") is None or not g.get("pick"):
        return None
    return g["edge"] if g["pick"]["side"] == "home" else -g["edge"]


def render_markdown(bundle: dict) -> str:
    meta = bundle["meta"]
    primary = meta["primary"]
    plabel = meta["primary_label"]
    out: list[str] = [f"# Weekly report: {meta['label']}", ""]
    facts = [
        f"Predictions as of **{meta['asof_local']}** (cutoff: nothing later is used).",
        f"Model: **{plabel}** (`{primary}`). Compared with: "
        + ", ".join(f"{model_label(m)}" for m in meta["comparison"]) + ".",
        f"Report written {meta['created_local']}.",
    ]
    if meta.get("lines_asof_local"):
        facts.append(f"Lines for games not yet played are as of the data pull at {meta['lines_asof_local']}, "
                     "not closing lines.")
    out += [f"- {f}" for f in facts]
    for note in meta.get("notes", []):
        out.append(f"- Note: {note}")
    out.append("")

    # ---- season to date
    season = bundle["season"]
    common = season["common"]
    out += ["## Season to date", ""]
    if common["n"]:
        out.append(f"FBS vs FBS games that every listed model predicted (n = {common['n']}). "
                   "Lower log loss and Brier are better.")
        out.append("")
        rows = []
        for m in common["models"]:
            b = common["blocks"][m]
            rows.append([model_label(m) + (" (model)" if m == primary else ""), b["n"], _num(b["log_loss"]),
                         _num(b["brier"]), _pct(b["accuracy"], 1)])
        out += _table(["Model", "Games", "Log loss", "Brier", "Accuracy"], rows, "lrrrr")
        out.append("")
    else:
        out += ["No completed games with predictions yet this season.", ""]
    if season["comparisons"]:
        out.append(f"{plabel} against each comparison on the games both predicted (paired bootstrap; "
                   "a negative difference means the model is better):")
        out.append("")
        rows = []
        for c in season["comparisons"]:
            rows.append([c["label"], c["n"], _num(c["primary"]), _num(c["other"]), f"{c['diff']:+.4f}",
                         _num(c["se"]), f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]",
                         _pct(c["p_primary_better"], 0)])
        out += _table(["Versus", "Games", "Model", "Other", "Difference", "SE", "95% interval", "P(model better)"],
                      rows, "lrrrrrrr")
        out.append("")
    if bundle.get("validation"):
        out.append("Validation, season-forward (from `data/eval`):")
        out.append("")
        rows = [[model_label(m), v["seasons"], v["n"], _num(v["log_loss"]), _num(v["brier"]), _pct(v["accuracy"], 1)]
                for m, v in bundle["validation"].items()]
        out += _table(["Model", "Seasons", "Games", "Log loss", "Brier", "Accuracy"], rows, "llrrrr")
        out.append("")

    # ---- last week
    prev = bundle["previous"]
    out += [f"## Last week: {prev['label']}", ""]
    if prev["games"]:
        rows = []
        for m, s in prev["summary"].items():
            rows.append([model_label(m) + (" (model)" if m == primary else ""), s["n"], f"{s['correct']}/{s['n']}",
                         _num(s["log_loss"]), _num(s["brier"])])
        out.append("FBS vs FBS games:")
        out.append("")
        out += _table(["Model", "Games", "Correct", "Log loss", "Brier"], rows, "lrrrr")
        out.append("")
        scored = [g for g in prev["games"] if g["result"]["log_loss"].get(primary) is not None]
        misses = sorted(scored, key=lambda g: -g["result"]["log_loss"][primary])[:MISSES]
        if misses:
            out.append(f"Biggest misses for {plabel}:")
            out.append("")
            for g in misses:
                r = g["result"]
                out.append(f"- {_matchup(g)}: {r['winner']} won {max(r['home_points'], r['away_points'])}-"
                           f"{min(r['home_points'], r['away_points'])}; model had {g['pick']['team']} at "
                           f"{_pct(g['pick']['p'])}.")
            out.append("")
        rows = []
        for g in prev["games"]:
            r = g["result"]
            score = f"{g['away']['name']} {r['away_points']}, {g['home']['name']} {r['home_points']}"
            pick = f"{g['pick']['team']} {_pct(g['pick']['p'])}" if g.get("pick") else ""
            mark = "" if r["correct"] is None else ("yes" if r["correct"] else "no")
            rows.append([score, pick, mark, _pct(_pick_p_market(g)), _spread(g),
                         _num(r["log_loss"].get(primary), ".3f")])
        out.append("<details><summary>All results</summary>")
        out.append("")
        out += _table(["Final", "Pick", "Right", "Market on pick", "Closing spread", "Log loss"], rows, "lllrlr")
        out += ["", "</details>", ""]
    else:
        out += ["No completed games.", ""]

    # ---- this week
    games = bundle["games"]
    out += [f"## This week: {meta['label']}", ""]
    if not games:
        out += ["No games scheduled.", ""]
        return "\n".join(out).rstrip() + "\n"
    with_pick = [g for g in games if g.get("pick")]
    out.append(f"{len(games)} games, {len(with_pick)} with a prediction from {plabel}. Edge = model probability "
               "for its pick minus the market's, in percentage points.")
    out.append("")
    edges = sorted([g for g in with_pick if _pick_edge(g) is not None], key=lambda g: -abs(_pick_edge(g)))
    if edges:
        out.append("Largest differences from the market:")
        out.append("")
        rows = [[_matchup(g), g["pick"]["team"], _pct(g["pick"]["p"]), _pct(_pick_p_market(g)), _pp(_pick_edge(g)),
                 _spread(g)] for g in edges[:EDGE_LIST]]
        out += _table(["Game", "Pick", "Model", "Market", "Edge (pts)", "Spread"], rows, "llrrrl")
        out.append("")
    rows = []
    for g in games:
        pick = g.get("pick")
        reason = g["reasons"][0]["text"] if g.get("reasons") else ""
        rows.append([g["kickoff_local"] or "", _matchup(g), pick["team"] if pick else "", _pct(pick["p"]) if pick else "",
                     _pct(_pick_p_market(g)), _pp(_pick_edge(g)), _spread(g), reason])
    out.append("All games (kickoff in " + meta["timezone"].split("/")[-1].replace("_", " ") + " time):")
    out.append("")
    out += _table(["Kickoff", "Game", "Pick", "Model", "Market", "Edge", "Spread", "Top reason"], rows, "lllrrrll")
    out.append("")
    out += ["### Reasons per pick", ""]
    for g in with_pick:
        parts = [f"{r['text']}" + (f" ({r['effect']:+.2f})" if r.get("effect") is not None else "")
                 for r in g["reasons"]]
        out.append(f"- **{_matchup(g)}**: {g['pick']['team']} {_pct(g['pick']['p'])}. "
                   + ("; ".join(parts) if parts else "No reasons given."))
    out.append("")
    out.append("Reason effects are in log-odds toward the home team.")
    out.append("")

    if meta.get("phase2"):
        out += ["## Spread and total", ""]
        rows = []
        for g in with_pick:
            m = g.get("market") or {}
            margin, total = g.get("pred_margin"), g.get("pred_total")
            ats = None if (margin is None or m.get("spread") is None) else margin + m["spread"]
            ou = None if (total is None or m.get("total") is None) else total - m["total"]
            rows.append([_matchup(g), _num(margin, "+.1f"), _spread(g), _num(ats, "+.1f"), _num(total, ".1f"),
                         _num(m.get("total"), ".1f"), _num(ou, "+.1f")])
        out += _table(["Game", "Model margin (home)", "Spread", "Model vs spread", "Model total", "Line total",
                       "Model vs total"], rows, "lrlrrrr")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_markdown"]
