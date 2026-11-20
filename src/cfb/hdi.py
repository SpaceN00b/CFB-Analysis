"""Conference availability (injury) reports from HD Intelligence (app.hdintelligence.com).

The SEC, ACC, Big 12 and Big Ten report pages embed HDI's app, which reads two public JSON
endpoints (no sign-in; POST {sport, organization, conference}):

    /api/get-publish-public   the current report of every conference game this week, keyed by HDI
                              game id: ReportType (Initial, Update 1, Update 2, Game Day),
                              publishDate + postedTime and kickoff (footer date + time) in the
                              conference's time zone, and every roster row "POS #N First Last" with
                              its status (Out, Out - (1st Half), Doubtful, Questionable, Probable,
                              Available, Exempt)
    /api/get-archive-public   the season so far, one row per listed player and game with a column
                              per report. The current week's later columns read '-' until HDI
                              fills them, so the current week is taken from the publish endpoint.

Conference games only; Notre Dame and non-conference games are not covered. No "out for season"
status. Each pull saves both endpoints (team logos stripped) under one folder; `merged_archive`
overlays the current reports on the archive rows so cfb.lab.availability.hdi_listings reads them.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE = "https://app.hdintelligence.com/api"
CONFERENCES = ("SEC", "ACC", "B12", "B10")
CONF_NAME = {"SEC": "SEC", "ACC": "ACC", "B12": "Big 12", "B10": "Big Ten"}
REPORTS = ("Initial", "Update 1", "Update 2", "Game Day")
TZ = {"ET": "America/New_York", "CT": "America/Chicago", "MT": "America/Denver", "PT": "America/Los_Angeles"}
NOT_LISTED = {"Available", "Exempt", "", None}
NAME_RE = re.compile(r"^\s*(\S+)\s+#(\d+)\s+(.+?)\s*$")


def _strip(o):
    """Drop embedded images (data: URIs), which are most of the publish payload."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not (isinstance(v, str) and v.startswith("data:"))}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def fetch(endpoint: str, conf: str, session=None, tries: int = 4, timeout: int = 90):
    import requests

    s = session or requests.Session()
    body = {"sport": "Football", "organization": conf, "conference": conf}
    err = None
    for i in range(tries):
        try:
            r = s.post(f"{BASE}/{endpoint}", json=body, timeout=timeout)
            if r.status_code == 200:
                return _strip(r.json())
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # network errors: retry with backoff
            err = repr(exc)
        time.sleep(2 ** i)
    raise RuntimeError(f"HDI {endpoint} {conf}: {err}")


def pull(dest: Path, confs=CONFERENCES) -> dict:
    """Save pub_<C>.json.gz and arch_<C>.json.gz for each conference under dest."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    out = {}
    for c in confs:
        for kind, ep in (("pub", "get-publish-public"), ("arch", "get-archive-public")):
            data = fetch(ep, c, s)
            with gzip.open(dest / f"{kind}_{c}.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(data, fh)
            out[(kind, c)] = data
    return out


def load(d: Path, kind: str, conf: str):
    p = d / f"{kind}_{conf}.json.gz"
    return json.load(gzip.open(p, "rt", encoding="utf-8")) if p.exists() else None


def _local(day: str, hhmm: str | None, tz: str | None) -> pd.Timestamp | None:
    if not day:
        return None
    try:
        t = datetime.strptime(f"{day} {hhmm or '00:00:00'}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return pd.Timestamp(t.replace(tzinfo=ZoneInfo(TZ.get(tz or "ET", "America/New_York")))).tz_convert("UTC")


def parse_name(s: str) -> tuple[str | None, str | None, str]:
    m = NAME_RE.match(str(s))
    return (m[1], m[2], m[3]) if m else (None, None, str(s).strip())


def reports(pub: dict | None, conf: str) -> pd.DataFrame:
    """One row per HDI game: report type, posted time, kickoff (UTC), teams."""
    rows = []
    if not isinstance(pub, dict):
        return pd.DataFrame(rows)
    for gid, g in pub.items():
        if not isinstance(g, dict) or "games" not in g:
            continue
        tz = g.get("conferenceTimeZone")
        f = g.get("footer") or {}
        teams = [t.get("teamName") for t in g.get("games") or []]
        rows.append({"conference": CONF_NAME.get(conf, conf), "hdi_game_id": str(gid),
                     "report_type": g.get("ReportType"), "tz": tz,
                     "posted_utc": _local(g.get("publishDate"), g.get("postedTime"), tz),
                     "game_date": f.get("date"), "kickoff_utc": _local(f.get("date"), f.get("time"), tz),
                     "location": f.get("location"), "teams": " vs ".join(t for t in teams if t)})
    return pd.DataFrame(rows)


def publish_rows(pub: dict | None, conf: str) -> pd.DataFrame:
    """Every roster row of the current reports, with its game and report."""
    rows = []
    if not isinstance(pub, dict):
        return pd.DataFrame(rows)
    for gid, g in pub.items():
        if not isinstance(g, dict) or "games" not in g:
            continue
        tz = g.get("conferenceTimeZone")
        f = g.get("footer") or {}
        sides = g.get("games") or []
        names = [t.get("teamName") for t in sides]
        for i, t in enumerate(sides):
            opp = names[1 - i] if len(names) == 2 else None
            for r in t.get("rows") or []:
                pos, num, player = parse_name(r.get("name"))
                rows.append({"conference": CONF_NAME.get(conf, conf), "hdi_game_id": str(gid),
                             "report_type": g.get("ReportType"),
                             "posted_utc": _local(g.get("publishDate"), g.get("postedTime"), tz),
                             "game_date": f.get("date"), "kickoff_utc": _local(f.get("date"), f.get("time"), tz),
                             "team": t.get("teamName"), "opponent": opp, "home": i == 1,
                             "player": player, "pos": pos, "number": num,
                             "status": (r.get("status") or "").strip(), "exempt_status": r.get("exemptStatus")})
    return pd.DataFrame(rows)


def _week_key(day: str) -> str:
    return datetime.strptime(day, "%Y-%m-%d").strftime("%a, %d %b %Y 00:00:00 GMT")


def merged_archive(arch: dict | None, pub: dict | None, conf: str) -> dict:
    """The archive with each current report written into its column: listed players get their
    status, archive rows of that game whose player the report now shows Available get
    'Available', and players listed only in the current report get a new row."""
    rows = [dict(r) for r in ((arch or {}).get("data") or [])]
    p = publish_rows(pub, conf)
    if p.empty:
        return {"data": rows}
    idx = {(r.get("Team"), r.get("Week"), str(r.get("Player")).strip()): r for r in rows}
    wknum = {r.get("Week"): r.get("WeekNum") for r in rows}
    for x in p.itertuples(index=False):
        if x.report_type not in REPORTS or not x.game_date:
            continue
        wk = _week_key(x.game_date)
        r = idx.get((x.team, wk, x.player))
        if x.status in NOT_LISTED:
            if r is not None:  # listed earlier this week, now cleared
                r[x.report_type] = "Available" if x.status == "Available" else x.status
            continue
        if r is None:
            r = {"Team": x.team, "TeamDisplay": x.team, "Player": x.player, "Number": f"#{x.number}" if x.number else None,
                 "Opponent": ("vs. " if x.home else "at ") + str(x.opponent), "Week": wk, "WeekNum": wknum.get(wk),
                 **{c: "-" for c in REPORTS}}
            rows.append(r)
            idx[(x.team, wk, x.player)] = r
        r[x.report_type] = x.status
    return {"data": rows}
