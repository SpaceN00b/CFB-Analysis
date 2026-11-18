"""A short plain-text summary of the weekly report, for the weekly run's chat post.

`python -m cfb.run_week` prints it last and writes it as summary.md next to report.md, so the
session that runs the week can post it without reading report.md (about 50 KB) or
bundle.json (about 600 KB).
"""
from __future__ import annotations

from .markdown import _matchup, _pct, _pick_edge, _pick_p_market, _pp, _spread, _wlp

TOP_EDGES = 3


def render_summary(bundle: dict, top: int = TOP_EDGES) -> str:
    meta = bundle["meta"]
    primary = meta["primary"]
    out = [f"{meta['label']}: {meta['primary_label']} (`{primary}`), as of {meta['asof_local']}."]

    own = bundle["season"]["own"]
    if primary in own:
        parts = [f"{meta['primary_label']} {own[primary]['log_loss']:.4f} (n {own[primary]['n']})"]
        for m in ("baseline_close", "baseline_elo"):
            if m in own:
                parts.append(f"{own[m]['label']} {own[m]['log_loss']:.4f} (n {own[m]['n']})")
        out.append("Season log loss, FBS vs FBS (lower is better): " + "; ".join(parts) + ".")
    else:
        out.append("Season log loss: no completed games with predictions yet.")

    prev = bundle.get("previous") or {}
    s = (prev.get("summary") or {}).get(primary)
    if s:
        line = f"Last week ({prev['label']}): {s['correct']}/{s['n']} correct, log loss {s['log_loss']:.4f}"
        rec = prev.get("points_record") or {}
        if rec.get("ats"):
            line += f"; spread {_wlp(rec['ats'])}"
        if rec.get("ou"):
            line += f", total {_wlp(rec['ou'])}"
        out.append(line + ".")
    else:
        out.append("Last week: no scored games.")

    games = bundle.get("games") or []
    if not games:
        out.append("No games this week.")
        return "\n".join(out) + "\n"
    edges = sorted([g for g in games if g.get("pick") and _pick_edge(g) is not None],
                   key=lambda g: -abs(_pick_edge(g)))[:top]
    out.append(f"{len(games)} games. Largest edges vs the market (model minus market probability for the pick, percentage points):")
    for g in edges:
        out.append(f"- {_matchup(g)}: {g['pick']['team']} {_pct(g['pick']['p'])} vs market "
                   f"{_pct(_pick_p_market(g))} ({_pp(_pick_edge(g))}), line {_spread(g) or 'n/a'}")
    for note in meta.get("notes", []):
        out.append(f"Note: {note}")
    return "\n".join(out) + "\n"
