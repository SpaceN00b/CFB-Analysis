"""Scheduled injury report pulls (Josiah 2026-11-20): conference availability reports from HDI
into the R2 warehouse and the week's player-availability package (test model, PR 35/36).

    python scripts/injury_pull.py run --kind scheduled        # Wed/Thu/Fri 10:30 PM ET
    python scripts/injury_pull.py run --kind kickoff --slot 2026-11-21T17:00:00Z
    python scripts/injury_pull.py run --kind manual [--week N] [--no-publish] [--no-package]

Each run:
  1. pulls the current reports and the season archive of the SEC, ACC, Big 12 and Big Ten
     (cfb.hdi) into /mnt/project-files/injury-reports/hdi_pulls/<UTC stamp>/, with merged/ = the
     archive with the current reports written in (what the impact scripts read);
  2. prepares ./data for the week: the cached base (pull-base/base_<season>_week<N>.tar) or, the
     first time in a week, the LATEST snapshot + a CFBD refresh when last week's results are
     missing (scratch copy of the asof archive) + features + depth charts + the impact inputs;
  3. writes tables/availability_reports/<stamp>.parquet for every pull so far and publishes only
     that table onto the bucket's current warehouse manifest;
  4. re-runs the depth-chart (PR 35) and defensive (PR 37) impact scores and this week's picks with
     the newest CFBDepth list (injury-reports/cfbdepth_<date>/, Josiah downloads it by hand) and this
     pull's reports (the pull wins where it is newer), then publishes the availability package
     (scripts/availability_export.py --publish -> extras/availability/);
  5. scheduled runs plan the kickoff pulls: games whose report lists a high-impact player (value
     over replacement >= THRESH points per game, offense or defense) who is not known to be out for
     the season, one pull 90 minutes before each kickoff time (plan.json);
  6. writes summary.md (printed at the end).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cfb import hdi
from cfb.lab import availability as A

PF = Path("/mnt/project-files")
IR = PF / "injury-reports"
PULLS = IR / "hdi_pulls"
BASES = IR / "pull-base"
LEGACY = IR / "hdi_raw_2026-11-18"          # 2025 season reports (status rates) and the 11-18 archive
CHECKS = IR / "checks_2026-11-19" / "injury_checks_2026-11-19.csv"
DEPTH = PF / "depth-charts"
DATA = Path("data")
EXP = DATA / "exploration"
THRESH = 1.0                                 # points per game over the replacement = high impact
LISTED = ("Out", "Out - (1st Half)", "Doubtful", "Questionable", "Probable")
ET = "America/New_York"
SCRIPTS = Path(__file__).resolve().parent
BRANCH = "claude/project-thread-cssahn"
KICKOFF_PROMPT = """Kickoff injury pull for the "College Football Analysis Algo" project, 90 minutes before the \
{kickoff_et} kickoffs. This is a fresh session; everything you need is below.

1. In the CFB-Analysis repo (SpaceN00b/CFB-Analysis): `git fetch origin {branch} && git checkout --detach FETCH_HEAD`, \
then `pip install -q -e '.[dev]'`.
2. Start `python scripts/injury_pull.py run --kind kickoff --slot {kickoff_utc} 2>&1 | tail -40` in the background \
(run_in_background; about 8 minutes) and wait for it to finish.
3. Post one short message to Josiah: the report lines (conference, report type, posted time), the "Kickoff slot" \
lines (each listed high-impact player's latest status), the changes since the last pull, and the warehouse and \
availability package lines. If the run failed, post the failing step and its last error lines instead.

Rules: keep context small (do not read files under data/ or the raw pulls). Push nothing, open no pull requests, \
publish no app release, do not touch /mnt/project-files/cfbd-data. Resolve the thread after posting."""


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def et(ts) -> str:
    return pd.Timestamp(ts).tz_convert(ET).strftime("%a %b %-d %-I:%M %p ET")


def sh(args: list[str], env: dict | None = None, tail: int = 0) -> str:
    log("$ " + " ".join(args))
    r = subprocess.run(args, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"failed ({r.returncode}): {' '.join(args)}")
    if tail:
        print("\n".join(out.strip().splitlines()[-tail:]))
    return out


# ---------------------------------------------------------------------------- CFBD

def cfbd():
    from cfb.client import CFBDClient
    return CFBDClient(max_retries=3, timeout=60)


def current_week(client, now: pd.Timestamp) -> dict | None:
    for season in (now.year, now.year - 1):
        for w in client.get("calendar", year=season) or []:
            if pd.Timestamp(w["startDate"]) <= now <= pd.Timestamp(w["endDate"]):
                return {"season": season, "week": int(w["week"]), "season_type": w["seasonType"]}
    return None


def week_games(client, wk: dict) -> pd.DataFrame:
    g = pd.DataFrame(client.get("games", year=wk["season"], week=wk["week"], seasonType=wk["season_type"]) or [])
    if g.empty:
        return g
    g["kickoff_utc"] = pd.to_datetime(g["startDate"], utc=True)
    return g[["id", "kickoff_utc", "startTimeTBD", "completed", "homeId", "homeTeam", "awayId", "awayTeam",
              "homeConference", "awayConference"]].rename(columns={"id": "game_id"})


# ---------------------------------------------------------------------------- 1. pull

def do_pull(now: pd.Timestamp) -> Path:
    d = PULLS / now.strftime("%Y%m%dT%H%M%SZ")
    raw = hdi.pull(d)
    m = d / "merged"
    m.mkdir(exist_ok=True)
    for c in hdi.CONFERENCES:
        with open(m / f"arch_{c}.json", "w") as fh:
            json.dump(hdi.merged_archive(raw[("arch", c)], raw[("pub", c)], c), fh)
        subprocess.run(["gzip", "-f", str(m / f"arch_{c}.json")], check=True)
        if (LEGACY / f"leg_{c}.json.gz").exists():
            shutil.copy(LEGACY / f"leg_{c}.json.gz", m / f"leg_{c}.json.gz")
    (d / "pulled_utc").write_text(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return d


def pull_rows(d: Path) -> pd.DataFrame:
    parts = [hdi.publish_rows(hdi.load(d, "pub", c), c) for c in hdi.CONFERENCES]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["conference", "hdi_game_id", "report_type", "posted_utc", "game_date",
                                     "kickoff_utc", "team", "opponent", "home", "player", "pos", "number",
                                     "status", "exempt_status"])
    return pd.concat(parts, ignore_index=True)


def pull_reports(d: Path) -> pd.DataFrame:
    parts = [hdi.reports(hdi.load(d, "pub", c), c) for c in hdi.CONFERENCES]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------- 2. data

def base_path(wk: dict) -> Path:
    return BASES / f"base_{wk['season']}_week{wk['week']}.tar"


def results_missing(wk: dict, now: pd.Timestamp) -> int:
    g = pd.read_parquet(DATA / "tables" / "games" / f"{wk['season']}_regular.parquet",
                        columns=["week", "startDate", "completed"])
    g = g[(g["week"] < wk["week"]) & (pd.to_datetime(g["startDate"], utc=True) < now - pd.Timedelta(hours=6))]
    return int((~g["completed"].fillna(False).astype(bool)).sum())


def prepare_data(wk: dict, now: pd.Timestamp) -> list[str]:
    notes = []
    tag = f"{wk['season']}_week{wk['week']}"
    if (DATA / ".pull_base").exists() and (DATA / ".pull_base").read_text().strip() == tag:
        return notes
    bp = base_path(wk)
    if bp.exists():
        log(f"restoring {bp}")
        with tarfile.open(bp) as t:
            t.extractall(".", filter="data")
        return notes
    log(f"building the week base {bp.name} (first pull of the week, ~15-25 min)")
    from cfb import run_week as RW
    RW.ensure_data(DATA, RW.default_snapshot())
    missing = results_missing(wk, now)
    args = [sys.executable, "-m", "cfb.run_week", "--season", str(wk["season"]), "--week", str(wk["week"]),
            "--season-type", "regular", "--no-warehouse", "--no-write-predictions"]
    if missing:
        arch = Path(os.environ.get("TMPDIR", "/tmp")) / "asof_scratch"
        if arch.exists():
            shutil.rmtree(arch)
        shutil.copytree(PF / "cfbd-data" / "asof", arch)  # the shared archive is never written
        args += ["--refresh", "--asof-archive", str(arch)]
        notes.append(f"Base: CFBD refresh of {wk['season']} ({missing} earlier games lacked results).")
    else:
        args += ["--no-refresh"]
    sh(args, tail=4)
    dc = DEPTH / (DEPTH / "LATEST").read_text().strip()
    with tarfile.open(dc) as t:
        t.extractall(".", filter="data")
    sh([sys.executable, str(SCRIPTS / "impact_base.py")], tail=2)
    sh([sys.executable, str(SCRIPTS / "injury_select.py"), "--build"], tail=2)
    (DATA / ".pull_base").write_text(tag)
    BASES.mkdir(parents=True, exist_ok=True)
    tmp = bp.with_suffix(".tar.part")
    with tarfile.open(tmp, "w") as t:
        for top in sorted(DATA.iterdir()):
            if top.name not in ("raw", "reports"):
                t.add(top)
    tmp.replace(bp)
    for old in BASES.glob(f"base_{wk['season']}_week*.tar"):
        if old != bp:
            old.unlink()
    notes.append(f"Base {bp.name} built ({bp.stat().st_size / 1e6:.0f} MB).")
    return notes


# ---------------------------------------------------------------------------- 3. warehouse table

def season_ending(teams: pd.DataFrame, list_csv: Path | None) -> set[int]:
    out = set()
    if list_csv is not None and list_csv.exists():
        d = pd.read_csv(list_csv)
        d = d[d["Status"].astype(str).str.lower().isin(["out for season", "ofs"])]
        if len(d):
            r, tid = A._roster_keys(DATA, 2026, teams)
            d["team_id"] = d["Team"].map(tid)
            ids = A._match(d[d["team_id"].notna()].rename(columns={"Name": "name"}), r)
            out |= {int(x) for x in ids.dropna()}
    if CHECKS.exists():
        c = pd.read_csv(CHECKS, dtype=str)
        c = c[c["expected_return"].astype(str).str.strip().isin(["season", "2027"])]
        out |= {int(float(x)) for x in pd.to_numeric(c["athlete_id"], errors="coerce").dropna()}
    return out


def match_ids(rows: pd.DataFrame, teams: pd.DataFrame, season: int) -> pd.DataFrame:
    if rows.empty:
        return rows.assign(team_id=pd.Series(dtype="float64"), athlete_id=pd.Series(dtype="float64"))
    r, tid = A._roster_keys(DATA, season, teams)
    rows = rows.copy()
    rows["team_cfbd"] = rows["team"].map(lambda t: A.HDI_TEAM_FIX.get(t, str(t).removesuffix(" St.") + " State"
                                                                       if str(t).endswith(" St.") else t))
    rows["team_id"] = rows["team_cfbd"].map(tid)
    ok = rows["team_id"].notna()
    rows["athlete_id"] = float("nan")
    rows.loc[ok, "athlete_id"] = A._match(rows[ok].rename(columns={"player": "name"}), r, jersey_col="number")
    return rows


def warehouse_table(teams: pd.DataFrame, season: int, publish: bool) -> str:
    dest = DATA / "tables" / "availability_reports"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(PULLS.glob("2*")):
        if not (d / "pulled_utc").exists():
            continue
        rows = pull_rows(d)
        rows = rows[rows["status"].isin(LISTED)]
        rows = match_ids(rows, teams, season).drop(columns="team_cfbd", errors="ignore")
        rows.insert(0, "pulled_utc", pd.Timestamp((d / "pulled_utc").read_text().strip()))
        rows.insert(1, "season", season)
        rows["posted_utc"] = pd.to_datetime(rows["posted_utc"], utc=True)
        rows["kickoff_utc"] = pd.to_datetime(rows["kickoff_utc"], utc=True)
        rows.to_parquet(dest / f"{d.name}.parquet", index=False)
        n += 1
    if not publish:
        return f"Warehouse: {n} pull partitions written locally (not published)."
    from cfb import warehouse_sync as WS
    r = WS.publish(DATA, WS.open_store(WS.r2_settings()), only=("tables/availability_reports",),
                   commit=WS.git_commit(), log=lambda m: None)
    return (f"Warehouse: published {r.version} (tables/availability_reports, {n} pulls; "
            f"{r.uploaded_files} new files).")


# ---------------------------------------------------------------------------- 4. package

def newest_list() -> tuple[Path | None, str | None]:
    best = None
    for p in glob.glob(str(IR / "cfbdepth_20*" / "cfbdepth-player-updates.csv")):
        day = Path(p).parent.name.removeprefix("cfbdepth_")
        if best is None or day > best[1]:
            best = (Path(p), day)
    return best or (None, None)


def package(wk: dict, pull: Path, list_csv: Path, list_day: str, publish: bool) -> str:
    N = str(wk["week"])
    env = {"CFB_HDI_DIR": str(pull / "merged"), "CFB_HDI_PULLED_UTC": (pull / "pulled_utc").read_text().strip(),
           "CFB_CFBDEPTH_LIST": str(list_csv), "CFB_CFBDEPTH_DOWNLOAD": list_day}
    py = sys.executable
    sh([py, str(SCRIPTS / "depth_impact_select.py"), "--week", N, "--out", str(EXP / "depth_impact")], env)
    sh([py, str(SCRIPTS / "defense_impact_select.py"), "--week", N, "--out", str(EXP / "defense_impact")], env)
    sh([py, str(SCRIPTS / "depth_impact_week.py"), "--week", N, "--out", str(EXP / "depth_impact_week")], env)
    newer = [d for d in PULLS.glob("2*") if d.name > pull.name and (d / "pulled_utc").exists()]
    if publish and newer:
        publish = False  # a later pull finished its fetch first: let it publish
    pulled = pd.Timestamp(env["CFB_HDI_PULLED_UTC"])
    args = [py, str(SCRIPTS / "availability_export.py"), "--week", N,
            "--listings", f"CFBDepth list {list_day} + conference reports (HDI) pulled {et(pulled)}"]
    out = sh(args + (["--publish"] if publish else []), env)
    line = next((x for x in out.splitlines() if x.startswith("published")), None)
    return (f"Availability package {wk['season']}_regular_{N}: " + (line or ("not published (a newer pull exists)"
            if newer else "written, not published")) + ".")


# ---------------------------------------------------------------------------- 5. plan

def flagged(pull: Path, wk: dict, teams: pd.DataFrame, games: pd.DataFrame, list_csv: Path | None) -> pd.DataFrame:
    """Listed players on this pull's reports with impact >= THRESH, not known out for the season."""
    rows = pull_rows(pull)
    rows = match_ids(rows[rows["status"].isin(LISTED)], teams, wk["season"])
    N = wk["week"]
    off = pd.read_csv(EXP / "depth_impact" / f"impact_scores_week{N}_2026.csv")
    dfn = pd.read_csv(EXP / "defense_impact" / f"def_impact_scores_week{N}_2026.csv")
    sc = pd.concat([off.assign(score=off["vor"])[["athlete_id", "grp", "score"]],
                    dfn.assign(score=dfn[["vor_fill", "vor_own"]].max(axis=1))[["athlete_id", "grp", "score"]]])
    sc = sc.sort_values("score").drop_duplicates("athlete_id", keep="last")
    x = rows.merge(sc, on="athlete_id", how="inner")
    ofs = season_ending(teams, list_csv)
    x["season_ending"] = x["athlete_id"].astype("int64").isin(ofs)
    x = x[(x["score"] >= THRESH) & ~x["season_ending"]]
    if x.empty or games.empty:
        return x.assign(game_id=pd.Series(dtype="int64"))
    side = pd.concat([games.rename(columns={"homeId": "team_id"})[["game_id", "team_id"]],
                      games.rename(columns={"awayId": "team_id"})[["game_id", "team_id"]]])
    x = x.merge(side, on="team_id", how="left")
    return x.merge(games[["game_id", "kickoff_utc", "startTimeTBD", "completed", "homeTeam", "awayTeam"]]
                   .rename(columns={"kickoff_utc": "kickoff"}), on="game_id", how="left")


def plan(pull: Path, fl: pd.DataFrame, now: pd.Timestamp) -> dict:
    slots, skipped = {}, []
    for gid, g in fl.dropna(subset=["game_id"]).groupby("game_id"):
        r = g.iloc[0]
        who = [f"{p.player} ({p.team} {p.pos}, {p.status}, {p.score:.1f})" for p in g.sort_values("score", ascending=False).itertuples()]
        game = {"game_id": int(gid), "matchup": f"{r.awayTeam} at {r.homeTeam}", "players": who}
        pull_at = r.kickoff - pd.Timedelta(minutes=90)
        if bool(r.completed) or bool(r.startTimeTBD) or pull_at < now + pd.Timedelta(minutes=10):
            skipped.append({**game, "why": "completed" if r.completed else "kickoff TBD" if r.startTimeTBD else "too late"})
            continue
        k = r.kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        s = slots.setdefault(k, {"kickoff_utc": k, "kickoff_et": et(r.kickoff),
                                 "pull_at_utc": pull_at.strftime("%Y-%m-%dT%H:%M:%SZ"), "pull_at_et": et(pull_at),
                                 "name": f"Injury pull before {r.kickoff.tz_convert(ET):%a %b %-d %-I:%M %p} ET kickoffs",
                                 "prompt": KICKOFF_PROMPT.format(kickoff_et=et(r.kickoff), kickoff_utc=k, branch=BRANCH),
                                 "games": []})
        s["games"].append(game)
    p = {"created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "pull": pull.name, "threshold": THRESH,
         "slots": sorted(slots.values(), key=lambda s: s["kickoff_utc"]), "skipped": skipped}
    (pull / "plan.json").write_text(json.dumps(p, indent=1))
    return p


# ---------------------------------------------------------------------------- 6. summary

def changes(pull: Path) -> tuple[str | None, pd.DataFrame]:
    prev = sorted(d for d in PULLS.glob("2*") if d.name < pull.name and (d / "pulled_utc").exists())
    if not prev:
        return None, pd.DataFrame()
    key = ["team", "player", "game_date"]
    a = pull_rows(prev[-1])
    b = pull_rows(pull)
    a = a[a["status"].isin(LISTED)][key + ["status"]]
    b = b[key + ["status"]]
    m = b.merge(a, on=key, how="outer", suffixes=("", "_prev"))
    m = m[m["game_date"].isin(set(b["game_date"]))]
    m["status"] = m["status"].fillna("not on report")
    m["status_prev"] = m["status_prev"].fillna("not listed")
    m = m[(m["status"] != m["status_prev"]) & ~((m["status_prev"] == "not listed") & ~m["status"].isin(LISTED))]
    return prev[-1].name, m


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=("scheduled", "kickoff", "manual"), default="manual")
    r.add_argument("--slot", help="kickoff time (UTC ISO) of a kickoff pull")
    r.add_argument("--week", type=int, help="regular-season week (default: the CFBD calendar week now)")
    r.add_argument("--no-publish", action="store_true", help="write locally, publish nothing to R2")
    r.add_argument("--no-package", action="store_true", help="pull + warehouse table only")
    a = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    publish = not a.no_publish
    lines = []

    pull = do_pull(now)
    rep = pull_reports(pull)
    rows = pull_rows(pull)
    lst = rows[rows["status"].isin(LISTED)]
    lines.append(f"# Injury pull {et(now)} ({a.kind})")
    lines.append(f"Raw: {pull}")
    if rep.empty:
        lines.append("Reports: none posted (HDI returned no current reports).")
    else:
        for conf, g in rep.groupby("conference"):
            kinds = ", ".join(f"{k} {n}" for k, n in g["report_type"].value_counts().items())
            lines.append(f"- {conf}: {len(g)} games ({kinds}); latest posted {et(g['posted_utc'].max())}")
        lines.append("Listed: " + ", ".join(f"{k} {v}" for k, v in lst["status"].value_counts().items()))

    client = cfbd()
    wk = current_week(client, now)
    if a.week:
        wk = {"season": now.year, "week": a.week, "season_type": "regular"}
    if wk is None or wk["season_type"] != "regular" or rep.empty:
        lines.append(f"Week: {wk}; no current regular-season reports: raw pull saved, warehouse table, "
                     "package and plan skipped.")
        return finish(pull, lines)
    lines.append(f"Week: {wk['season']} regular week {wk['week']}")

    lines += prepare_data(wk, now)
    teams = pd.read_parquet(DATA / "core" / "teams.parquet")
    lines.append(warehouse_table(teams, wk["season"], publish))
    list_csv, list_day = newest_list()
    lines.append(f"CFBDepth list: {list_csv} (downloaded {list_day})")
    if not a.no_package:
        lines.append(package(wk, pull, list_csv, list_day, publish))

    prev, ch = changes(pull)
    if prev:
        lines.append(f"Changes since pull {prev}: {len(ch)} " + (
            "(" + ", ".join(f"{k} {v}" for k, v in ch["status"].value_counts().items()) + ")" if len(ch) else ""))

    if a.no_package:
        return finish(pull, lines)
    games = week_games(client, wk)
    fl = flagged(pull, wk, teams, games, list_csv)
    if len(ch) and len(fl):
        hi = ch.merge(fl[["team", "player", "score"]].drop_duplicates(), on=["team", "player"])
        for x in hi.sort_values("score", ascending=False).head(15).itertuples():
            lines.append(f"  - {x.player} ({x.team}, {x.score:.1f} pts): {x.status_prev} -> {x.status}")
    if a.kind == "kickoff" and a.slot:
        k = pd.Timestamp(a.slot)
        s = fl[fl["kickoff"] == k]
        lines.append(f"Kickoff slot {et(k)}: {s['game_id'].nunique()} games with listed high-impact players")
        for gid, g in s.groupby("game_id"):
            lines.append(f"- {g.iloc[0].awayTeam} at {g.iloc[0].homeTeam}: " + "; ".join(
                f"{p.player} {p.status}" for p in g.sort_values("score", ascending=False).itertuples()))
    if a.kind in ("scheduled", "manual"):
        p = plan(pull, fl, now)
        lines.append(f"High-impact listed players (>= {THRESH:.1f} pts/game, not out for season): {len(fl)} "
                     f"in {fl['game_id'].nunique()} games")
        lines.append(f"Kickoff pulls to schedule ({len(p['slots'])}; plan {pull / 'plan.json'}):")
        for s in p["slots"]:
            lines.append(f"- SLOT kickoff_utc={s['kickoff_utc']} run_at_utc={s['pull_at_utc']} name=\"{s['name']}\" "
                         f"games: " + "; ".join(g["matchup"] for g in s["games"]))
        if p["skipped"]:
            lines.append(f"Not scheduled: {len(p['skipped'])} games (" + ", ".join(
                f"{g['matchup']}: {g['why']}" for g in p["skipped"][:8]) + ")")
    return finish(pull, lines)


def finish(pull: Path, lines: list[str]) -> int:
    text = "\n".join(lines) + "\n"
    (pull / "summary.md").write_text(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
