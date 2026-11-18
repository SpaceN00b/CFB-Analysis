"""The weekly report as Markdown, rendered from the bundle (see bundle.py)."""
from __future__ import annotations

from .bundle import model_label
from .rankings import render_markdown_section

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


def _wlp(r: dict) -> str:
    return f"{r['won']}-{r['lost']}-{r['push']}"


def _mark(c) -> str:
    return "push" if c is None else ("yes" if c else "no")


def _model_line(g: dict) -> str:
    """The model's predicted margin written as a spread for its favourite."""
    m = g.get("pred_margin")
    if m is None:
        return ""
    if round(m, 1) == 0:
        return "PK"
    fav = g["home"]["name"] if m > 0 else g["away"]["name"]
    return f"{fav} -{abs(m):.1f}"


def _ats_text(g: dict) -> str:
    a = g.get("ats")
    if not a:
        return ""
    return f"{a['team']} {a['line']:+g}" if a["line"] != 0 else f"{a['team']} PK"


def _ou_text(g: dict) -> str:
    o = g.get("ou")
    return "" if not o else f"{o['side'].capitalize()} {o['line']:g}"


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

    sp = bundle.get("season_points")
    if sp and (sp.get("margin") or {}).get("n"):
        out.append(f"Spread and total, {plabel} against the closing line ({sp['scope']}). Hit rate = share of "
                   "games where the model's side of the line won, pushes left out; 52.4% breaks even at -110.")
        out.append("")
        rows = []
        for key, name in (("margin", "Spread (home margin)"), ("total", "Total points")):
            b = sp.get(key) or {}
            if not b.get("n"):
                continue
            hits = {h["min_edge"]: h for h in b["vs_line"]}
            rows.append([name, b["n"], _num(b["mae"], ".2f"), _num(b["line_mae"], ".2f"), _num(b["bias"], "+.2f")]
                        + [f"{_pct(hits[t]['hit_rate'], 1)} ({hits[t]['n']})" for t in (0.0, 3.0, 7.0)])
        out += _table(["Target", "Games", "Model MAE", "Line MAE", "Bias", "Hit rate, all", "Edge 3+", "Edge 7+"],
                      rows, "lrrrrrrr")
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
        rec = prev.get("points_record")
        if rec:
            out.append(f"{plabel} against the closing line: spread {_wlp(rec['ats'])}, total {_wlp(rec['ou'])} "
                       "(won-lost-push).")
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
            row = [score, pick, mark, _pct(_pick_p_market(g)), _spread(g), _num(r["log_loss"].get(primary), ".3f")]
            if meta.get("phase2"):
                row += [_ats_text(g), _mark(r.get("ats_correct")) if g.get("ats") else "",
                        _ou_text(g), _mark(r.get("ou_correct")) if g.get("ou") else ""]
            rows.append(row)
        out.append("<details><summary>All results</summary>")
        out.append("")
        head = ["Final", "Pick", "Right", "Market on pick", "Closing spread", "Log loss"]
        align = "lllrlr"
        if meta.get("phase2"):
            head += ["Spread pick", "Covered", "Total pick", "Right"]
            align += "llll"
        out += _table(head, rows, align)
        out += ["", "</details>", ""]
    else:
        out += ["No completed games.", ""]

    # ---- this week
    games = bundle["games"]
    out += [f"## This week: {meta['label']}", ""]
    if not games:
        out += ["No games scheduled.", ""]
        if bundle.get("rankings"):
            out += render_markdown_section(bundle["rankings"])
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
        out.append(f"{plabel}'s line against the market line. Model line = predicted margin as a spread for the "
                   "favourite; edge = points between the model and the market. Model lines are set at the week's "
                   "cutoff; market lines for games not yet played are from the last data pull.")
        out.append("")
        picks = [g for g in with_pick if g.get("ats") or g.get("ou")]
        top = sorted([g for g in picks if g.get("ats")], key=lambda g: -g["ats"]["edge"])[:EDGE_LIST]
        if top:
            out.append("Largest spread differences:")
            out.append("")
            out += _table(["Game", "Model line", "Market", "Pick", "Edge (pts)"],
                          [[_matchup(g), _model_line(g), _spread(g), _ats_text(g), _num(g["ats"]["edge"], ".1f")]
                           for g in top], "lllll")
            out.append("")
        top = sorted([g for g in picks if g.get("ou")], key=lambda g: -g["ou"]["edge"])[:EDGE_LIST]
        if top:
            out.append("Largest total differences:")
            out.append("")
            out += _table(["Game", "Model total", "Market total", "Pick", "Edge (pts)"],
                          [[_matchup(g), _num(g.get("pred_total"), ".1f"), _num(g["ou"]["line"], "g"), _ou_text(g),
                            _num(g["ou"]["edge"], ".1f")] for g in top], "lrrlr")
            out.append("")
        rows = []
        for g in with_pick:
            m = g.get("market") or {}
            rows.append([_matchup(g), _model_line(g), _spread(g), _ats_text(g),
                         _num(g["ats"]["edge"], ".1f") if g.get("ats") else "",
                         _num(g.get("pred_total"), ".1f"), _num(m.get("total"), "g"), _ou_text(g),
                         _num(g["ou"]["edge"], ".1f") if g.get("ou") else ""])
        out.append("<details><summary>All games</summary>")
        out.append("")
        out += _table(["Game", "Model line", "Market", "Spread pick", "Edge", "Model total", "Market total",
                       "Total pick", "Edge"], rows, "llllrrrlr")
        out += ["", "</details>", ""]
    if bundle.get("rankings"):
        out += render_markdown_section(bundle["rankings"])
    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_markdown"]
