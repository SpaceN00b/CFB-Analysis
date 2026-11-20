"""Player availability and injury features per team-game (test bench, 2026-11-19).

Builds, for every FBS team-game 2014-2026, what was known before kickoff about who would be
missing and what those players were worth, so a test model can adjust model_v2 for injuries.

Pieces (all from the team's earlier games of the same season, so they are pregame):

* Player roles: share of team non-garbage dropbacks (QB), carries (RB/FB), targets (WR/TE,
  CFBD /plays/stats Target + Reception rows) and tackles (box-score TOT, DL/LB/DB), averaged
  over the games the player appeared in (box-score line or /plays/stats row), at least 2.
* QB value over replacement: (starter EPA/dropback - backup EPA/dropback) x team dropbacks per
  game, both EPA rates shrunk (K dropbacks) toward league means for starters and backups. The
  starter is the QB with the most season dropbacks; the backup is the next QB by dropbacks who is
  not expected out (p_out < 0.5), else a league-average backup.
* Probability out (`p_out`), best source first:
    1. availability listing for the game: conference reports (HDI archive, 2025-2026 conference
       games, last pregame status) and the CFBDepth list (official-report entries, current status
       for the week after its download, and the per-player check windows: games missed after the
       injury date / first listing);
    2. box-score streak: missed the last k team games -> P(absent) by position group and k, from
       earlier seasons.
  Status -> probability comes from earlier games where both a listing and a box score exist.
* Baked-in share: how much of the season the player already missed (games missed since his first
  appearance / team games since then). Ratings already contain those games, so a player who has
  been out for weeks costs less than a new absence.

Team-game columns (one row per game_id, team_id) are written by `team_features`; the test model in
`scripts/injury_select.py` uses home-minus-away differences.

The start-of-season depth chart (Josiah, to be added later) slots in through `--preseason-chart`
in the script: it marks preseason starters so a starter who never played (hurt in camp) is known.
"""
from __future__ import annotations

import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PASSER = ("Completion", "Incompletion", "Sack Taken", "Interception Thrown")
TARGET = ("Reception", "Target")
POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
       "OL": "OL", "OT": "OL", "G": "OL", "C": "OL", "OG": "OL", "IOL": "OL",
       "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
       "LB": "LB", "ILB": "LB", "OLB": "LB",
       "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB"}
DEF = ("DL", "LB", "DB")
MIN_GAMES = 2
K_QB = 120.0                  # dropbacks of shrinkage for QB EPA rates
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
HDI_TEAM_FIX = {"Miami (FL)": "Miami", "Southern California": "USC", "Pitt": "Pittsburgh"}
REPORT_RE = re.compile(r"Official report (\d{4}-\d{2}-\d{2}): ([A-Za-z ]+?)\. Game (\d{4}-\d{2}-\d{2})")
STATUS_NORM = {"out": "Out", "out for season": "Out", "ofs": "Out", "doubtful": "Doubtful",
               "questionable": "Questionable", "probable": "Probable", "available": "Available",
               "game-time decision": "Questionable", "gtd": "Questionable"}


# ---------------------------------------------------------------------------- inputs

def _parts(data: Path, table: str, seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        for f in sorted(glob.glob(str(data / "tables" / table / f"{s}_*.parquet"))):
            t = pd.read_parquet(f)
            if "gameId" in t.columns and len(t):
                out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def player_games(data: Path, seasons) -> pd.DataFrame:
    """One row per (game_id, team_id, athlete_id) who appeared: non-garbage dropbacks / carries /
    targets with EPA sums, box-score tackles, and season position group."""
    seasons = list(seasons)
    pbp = pd.read_parquet(data / "core" / "pbp.parquet", columns=[
        "play_id", "game_id", "season", "offense_id", "defense_id", "garbage_time", "ppa"])
    pbp = pbp[pbp["season"].isin(seasons)]
    teams = pd.read_parquet(data / "core" / "teams.parquet")
    tid = {(s, t): i for s, t, i in zip(teams["season"], teams["team"], teams["team_id"])}

    st = _parts(data, "plays_stats", seasons)
    st = st[st["athleteId"].notna()]
    st["athlete_id"] = pd.to_numeric(st["athleteId"], errors="coerce")
    st = st[st["athlete_id"].notna() & (st["athlete_id"] > 0)]
    st["athlete_id"] = st["athlete_id"].astype("int64")
    st["play_id"] = st["playId"].astype(str)
    st["team_id"] = [tid.get((s, t)) for s, t in zip(st["season"], st["team"])]
    st = st.drop_duplicates(["play_id", "athlete_id", "statType"])
    pres_st = st[["gameId", "team_id", "athlete_id"]].dropna().rename(columns={"gameId": "game_id"})

    off = st[st["statType"].isin(PASSER + ("Rush",) + TARGET)].merge(
        pbp[["play_id", "game_id", "offense_id", "garbage_time", "ppa"]], on="play_id", how="inner")
    off = off[~off["garbage_time"].fillna(False).astype(bool)]
    off["kind"] = np.select([off["statType"].isin(PASSER), off["statType"].eq("Rush")], ["drop", "carry"], "tgt")
    # one credit per play and kind (a reception and its target row are the same target)
    off = off.drop_duplicates(["play_id", "athlete_id", "kind"])
    off["ppa"] = off["ppa"].fillna(0.0)
    agg = off.pivot_table(index=["game_id", "offense_id", "athlete_id"], columns="kind",
                          values="ppa", aggfunc=["size", "sum"], fill_value=0.0)
    agg.columns = [f"{k}" if a == "size" else f"{k}_epa" for a, k in agg.columns]
    agg = agg.reset_index().rename(columns={"offense_id": "team_id"})

    gp = _parts(data, "games_players", seasons)
    gp = gp[gp["athleteId"].notna() & ~gp["athleteId"].astype(str).str.startswith("-")]
    gp["athlete_id"] = pd.to_numeric(gp["athleteId"], errors="coerce")
    gp = gp[gp["athlete_id"].notna() & (gp["athlete_id"] > 0)]
    gp["athlete_id"] = gp["athlete_id"].astype("int64")
    gp["team_id"] = [tid.get((s, t)) for s, t in zip(gp["season"], gp["team"])]
    gp["game_id"] = pd.to_numeric(gp["gameId"], errors="coerce").astype("int64")
    tk = gp[(gp["category"] == "defensive") & (gp["statType"] == "TOT")].copy()
    tk["tkl"] = pd.to_numeric(tk["stat"], errors="coerce").fillna(0.0)
    tk = tk.groupby(["game_id", "team_id", "athlete_id"], as_index=False)["tkl"].sum()

    pres = pd.concat([pres_st, gp[["game_id", "team_id", "athlete_id"]]]).dropna().drop_duplicates()
    pres["team_id"] = pres["team_id"].astype("int64")
    pres["game_id"] = pres["game_id"].astype("int64")
    u = pres.merge(agg, on=["game_id", "team_id", "athlete_id"], how="left").merge(
        tk, on=["game_id", "team_id", "athlete_id"], how="left")
    for c in ("drop", "carry", "tgt", "drop_epa", "carry_epa", "tgt_epa", "tkl"):
        if c not in u:
            u[c] = 0.0
        u[c] = u[c].fillna(0.0)
    gi = pd.read_parquet(data / "core" / "game_index.parquet", columns=["game_id", "season", "start_time_utc"])
    u = u.merge(gi, on="game_id")
    pos = []
    for s in seasons:
        f = data / "tables" / "roster" / f"{s}.parquet"
        if f.exists():
            r = pd.read_parquet(f, columns=["id", "position"])
            r["athlete_id"] = pd.to_numeric(r["id"], errors="coerce")
            r["season"] = s
            pos.append(r.dropna(subset=["athlete_id"]).drop_duplicates("athlete_id"))
    pos = pd.concat(pos)
    pos["athlete_id"] = pos["athlete_id"].astype("int64")
    pos["grp"] = pos["position"].map(POS).fillna("other")
    u = u.merge(pos[["season", "athlete_id", "grp"]], on=["season", "athlete_id"], how="left")
    u["grp"] = u["grp"].fillna("other")
    # a passer listed elsewhere on the roster is still a QB for this purpose when he throws a lot
    u.loc[(u["drop"] >= 10) & (u["grp"] != "QB"), "grp"] = "QB"
    return u


def team_schedule(data: Path, seasons, upcoming_until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Completed games as (season, game_id, team_id, start_time_utc), one row per team; with
    `upcoming_until` (UTC), also the not-yet-played games that start before it (forward use)."""
    gi = pd.read_parquet(data / "core" / "game_index.parquet")
    done = gi["completed"].fillna(False).astype(bool)
    if upcoming_until is not None:
        done = done | (gi["start_time_utc"] < upcoming_until)
    gi = gi[gi["season"].isin(list(seasons)) & done]
    h = gi[["season", "game_id", "home_id", "start_time_utc"]].rename(columns={"home_id": "team_id"})
    a = gi[["season", "game_id", "away_id", "start_time_utc"]].rename(columns={"away_id": "team_id"})
    return pd.concat([h, a]).sort_values(["team_id", "start_time_utc"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- roles

def roles(u: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (team-game, player with >= MIN_GAMES earlier appearances this season)."""
    tot = u.groupby(["game_id", "team_id"])[["drop", "carry", "tgt", "tkl"]].sum()
    rows = []
    for (season, team), g in sched.groupby(["season", "team_id"], sort=False):
        games = g.sort_values("start_time_utc")["game_id"].to_numpy()
        n = len(games)
        if n < 3:
            continue
        gidx = {gid: i for i, gid in enumerate(games)}
        pu = u[(u["team_id"] == team) & u["game_id"].isin(gidx)]
        if pu.empty:
            continue
        ath = pu["athlete_id"].unique()
        aidx = {a: i for i, a in enumerate(ath)}
        m = len(ath)
        P = np.zeros((m, n))
        V = {c: np.zeros((m, n)) for c in ("drop", "carry", "tgt", "tkl", "drop_epa")}
        S = {c: np.zeros((m, n)) for c in ("drop", "carry", "tgt", "tkl")}
        ri = pu["athlete_id"].map(aidx).to_numpy()
        ci = pu["game_id"].map(gidx).to_numpy()
        P[ri, ci] = 1.0
        t = tot.reindex(list(zip(pu["game_id"], pu["team_id"])))
        for c in V:
            V[c][ri, ci] = pu[c].to_numpy()
        for c in S:
            den = t[c].to_numpy()
            S[c][ri, ci] = np.where(den > 0, pu[c].to_numpy() / np.where(den > 0, den, 1), 0.0)
        grp = pu.drop_duplicates("athlete_id").set_index("athlete_id")["grp"].reindex(ath).to_numpy()
        team_drop = np.array([tot["drop"].get((gid, team), 0.0) for gid in games])
        # cumulative before game k (exclusive)
        cP = np.cumsum(P, axis=1) - P
        cS = {c: np.cumsum(S[c], axis=1) - S[c] for c in S}
        cV = {c: np.cumsum(V[c], axis=1) - V[c] for c in V}
        first = np.where(P.any(axis=1), P.argmax(axis=1), n)
        # streak of absences ending at k-1
        streak = np.zeros((m, n))
        for k in range(1, n):
            streak[:, k] = np.where(P[:, k - 1] == 0, streak[:, k - 1] + 1, 0)
        for k in range(2, n):
            gpk = cP[:, k]
            ok = gpk >= MIN_GAMES
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            since = k - first[idx]
            d = pd.DataFrame({
                "season": season, "game_id": games[k], "team_id": team, "k": k, "athlete_id": ath[idx],
                "grp": grp[idx], "gp": gpk[idx], "team_gp": k, "since_first": since,
                "missed_frac": np.where(since > 0, (since - gpk[idx]) / np.maximum(since, 1), 0.0),
                "streak": streak[idx, k], "absent": 1.0 - P[idx, k],
                "sh_drop": cS["drop"][idx, k] / gpk[idx], "sh_carry": cS["carry"][idx, k] / gpk[idx],
                "sh_tgt": cS["tgt"][idx, k] / gpk[idx], "sh_tkl": cS["tkl"][idx, k] / gpk[idx],
                "n_drop": cV["drop"][idx, k], "epa_drop": cV["drop_epa"][idx, k],
                "team_dpg": team_drop[:k].mean(),
            })
            d = d[(d[["sh_drop", "sh_carry", "sh_tgt"]].max(axis=1) >= 0.03) | (d["sh_tkl"] >= 0.03)]
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------- listings

def _norm(name) -> str:
    toks = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " ").replace(".", "")).split()
    return " ".join(t for t in toks if t not in SUFFIXES)


def _roster_keys(data: Path, season: int, teams: pd.DataFrame):
    r = pd.read_parquet(data / "tables" / "roster" / f"{season}.parquet")
    r["athlete_id"] = pd.to_numeric(r["id"], errors="coerce")
    tid = dict(zip(teams.loc[teams["season"] == season, "team"], teams.loc[teams["season"] == season, "team_id"]))
    r["team_id"] = r["team"].map(tid)
    r["full"] = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).map(_norm)
    r["key2"] = r["full"].str.split().map(lambda t: f"{t[0][:1]} {t[-1]}" if t else "")
    r["last"] = r["lastName"].astype(str).map(_norm).str.split().str[-1]
    r["jersey"] = pd.to_numeric(r["jersey"], errors="coerce")
    return r, tid


def _match(df, r, team_col="team_id", name_col="name", jersey_col=None):
    k1 = r.drop_duplicates(["team_id", "full"], keep=False).set_index(["team_id", "full"])["athlete_id"]
    k2 = r.drop_duplicates(["team_id", "key2"], keep=False).set_index(["team_id", "key2"])["athlete_id"]
    full = df[name_col].map(_norm)
    key2 = full.str.split().map(lambda t: f"{t[0][:1]} {t[-1]}" if t else "")
    out = pd.Series([k1.get((t, n)) for t, n in zip(df[team_col], full)], index=df.index, dtype="float64")
    if jersey_col is not None:
        k3 = r.dropna(subset=["jersey"]).drop_duplicates(["team_id", "jersey", "last"], keep=False)
        k3 = k3.set_index(["team_id", "jersey", "last"])["athlete_id"]
        last = full.str.split().str[-1]
        j = pd.to_numeric(df[jersey_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        out = out.fillna(pd.Series([k3.get((t, jj, ll)) for t, jj, ll in zip(df[team_col], j, last)],
                                   index=df.index, dtype="float64"))
    out = out.fillna(pd.Series([k2.get((t, n)) for t, n in zip(df[team_col], key2)], index=df.index, dtype="float64"))
    return out


def _game_on(sched_team: pd.DataFrame, team_id, day: pd.Timestamp, tol_days=3):
    """game_id of team's game within tol_days of `day` (ET date), else None."""
    g = sched_team.get(team_id)
    if g is None:
        return None
    dd = np.abs((g["et_date"] - day).dt.days.to_numpy())
    i = int(dd.argmin())
    return int(g["game_id"].iloc[i]) if dd[i] <= tol_days else None


def _sched_index(sched: pd.DataFrame):
    s = sched.copy()
    s["et_date"] = s["start_time_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return {t: g.sort_values("et_date").reset_index(drop=True) for t, g in s.groupby("team_id")}


def hdi_listings(data: Path, sched: pd.DataFrame, teams: pd.DataFrame, hdi_dir: Path) -> pd.DataFrame:
    """Conference availability reports (2025 legacy + 2026 archive): last pregame status and the
    initial-report status per (game_id, team_id, athlete_id)."""
    out = []
    si = _sched_index(sched)
    for kind, season, cols in (("leg", 2025, ("Gameday", "Friday", "Thursday", "Initial")),
                               ("arch", 2026, ("Game Day", "Update 2", "Update 1", "Initial"))):
        rows = []
        for f in glob.glob(str(hdi_dir / f"{kind}_*.json.gz")):
            rows += json.load(gzip.open(f)).get("data", [])
        if not rows:
            continue
        h = pd.DataFrame(rows)
        for c in cols:
            if c not in h:
                h[c] = None

        def last(r):  # '-' = that report not filled in (yet)
            for c in cols:
                v = r.get(c)
                if isinstance(v, str) and v.strip() not in ("", "-"):
                    return v.strip()
            return None

        def first(r):
            for c in reversed(cols):
                v = r.get(c)
                if isinstance(v, str) and v.strip() not in ("", "-"):
                    return v.strip()
            return None
        h["status"] = h.apply(last, axis=1)
        h["status_initial"] = h.apply(first, axis=1)
        h["team"] = h["Team"].map(lambda t: HDI_TEAM_FIX.get(t, re.sub(r" St\.$", " State", str(t))))
        r, tid = _roster_keys(data, season, teams)
        h["team_id"] = h["team"].map(tid)
        h = h[h["team_id"].notna()]
        h["athlete_id"] = _match(h.rename(columns={"Player": "name"}), r, jersey_col="Number")
        h = h[h["athlete_id"].notna()]
        h["day"] = pd.to_datetime(h["Week"], format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce")
        h["game_id"] = [_game_on(si, t, d) for t, d in zip(h["team_id"], h["day"])]
        h = h[h["game_id"].notna()]
        h["source"] = f"hdi{season}"
        out.append(h[["game_id", "team_id", "athlete_id", "status", "status_initial", "source"]])
    if not out:
        return pd.DataFrame(columns=["game_id", "team_id", "athlete_id", "status", "status_initial", "source"])
    o = pd.concat(out, ignore_index=True)
    for c in ("status", "status_initial"):
        o[c] = o[c].map(lambda v: STATUS_NORM.get(str(v).strip().lower(), str(v)) if v is not None else None)
    o["game_id"] = o["game_id"].astype("int64")
    o["team_id"] = o["team_id"].astype("int64")
    o["athlete_id"] = o["athlete_id"].astype("int64")
    return o.drop_duplicates(["game_id", "team_id", "athlete_id"])


def cfbdepth_listings(data: Path, sched: pd.DataFrame, teams: pd.DataFrame, list_csv: Path,
                      checks: pd.DataFrame | None, download: pd.Timestamp) -> pd.DataFrame:
    """CFBDepth list (downloaded `download`, a naive ET date): official-report entries (one game
    each), the current status for games from its Last Update date to the download, and, from the
    per-player checks, every game missed after the injury date / first listing / Last Update of an
    Out listing (status Out)."""
    d = pd.read_csv(list_csv)
    d.insert(0, "row_id", range(1, len(d) + 1))
    r, tid = _roster_keys(data, 2026, teams)
    d["team_id"] = d["Team"].map(tid)
    d = d[d["team_id"].notna()]
    d["athlete_id"] = _match(d.rename(columns={"Name": "name"}), r)
    d = d[d["athlete_id"].notna()]
    si = _sched_index(sched)
    d["lu"] = pd.to_datetime(d["Last Update"], format="%m/%d/%Y", errors="coerce")
    rows = []
    for x in d.itertuples(index=False):
        # current status holds from its Last Update date to the download
        st_now = STATUS_NORM.get(str(x.Status).strip().lower())
        g = si.get(x.team_id)
        if st_now is not None and g is not None and pd.notna(x.lu):
            for gid in g.loc[(g["et_date"] >= x.lu) & (g["et_date"] <= download), "game_id"]:
                rows.append((int(gid), x.team_id, x.athlete_id, st_now, "cfbdepth_current", x.row_id))
            # forward use: the team's next game after the download (only if it is in `sched`)
            nxt = g.loc[g["et_date"] > download, "game_id"]
            if len(nxt):
                rows.append((int(nxt.iloc[0]), x.team_id, x.athlete_id, st_now, "cfbdepth_next", x.row_id))
        for _rep, st, game in REPORT_RE.findall(str(x.Update)):
            gid = _game_on(si, x.team_id, pd.Timestamp(game), tol_days=1)
            if gid is not None:
                rows.append((gid, x.team_id, x.athlete_id, STATUS_NORM.get(st.strip().lower(), st.strip().title()),
                             "cfbdepth_report", x.row_id))
    if checks is not None and len(checks):
        c = checks.merge(d[["row_id", "team_id", "athlete_id", "lu", "Status"]], on="row_id", how="inner",
                         suffixes=("_chk", ""))
        for x in c.itertuples(index=False):
            dates = [s.strip() for s in str(x.out_game_dates).split(";") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.strip())]
            if not dates:
                continue
            known = None
            inj = str(x.injury_date).strip()
            if inj == "preseason":
                known = pd.Timestamp("2026-08-01")
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", inj):
                known = pd.Timestamp(inj) + pd.Timedelta(days=1)
            fl = str(x.first_listed_out).strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fl):
                known = pd.Timestamp(fl) if known is None else min(known, pd.Timestamp(fl))
            if str(x.Status).startswith("Out") and pd.notna(x.lu):
                known = x.lu if known is None else min(known, x.lu)
            if known is None:
                continue
            for day in dates:
                dd = pd.Timestamp(day)
                if dd < known:
                    continue
                gid = _game_on(si, x.team_id, dd, tol_days=1)
                if gid is not None:
                    rows.append((gid, x.team_id, x.athlete_id, "Out", "check_window", x.row_id))
    o = pd.DataFrame(rows, columns=["game_id", "team_id", "athlete_id", "status", "source", "row_id"])
    if o.empty:
        return o
    o["status_initial"] = o["status"]
    for col in ("game_id", "team_id", "athlete_id"):
        o[col] = o[col].astype("int64")
    # an official-report entry beats a reconstructed window for the same game
    o["pri"] = o["source"].map({"cfbdepth_report": 0, "cfbdepth_current": 1, "cfbdepth_next": 1, "check_window": 2})
    return o.sort_values("pri").drop_duplicates(["game_id", "team_id", "athlete_id"]).drop(columns="pri")


# ---------------------------------------------------------------------------- probabilities

def status_rates(ro: pd.DataFrame, lst: pd.DataFrame, col="status") -> dict:
    """P(absent | listed status) from listed role players (QB/RB/WR/TE, reliable presence)."""
    m = ro[ro["grp"].isin(["QB", "RB", "WR", "TE"])].merge(lst[["game_id", "team_id", "athlete_id", col]],
                                                           on=["game_id", "team_id", "athlete_id"])
    t = m.groupby(col)["absent"].agg(["mean", "size"])
    return {k: (float(v["mean"]), int(v["size"])) for k, v in t.iterrows()}


def streak_rates(ro: pd.DataFrame, upto_season: int) -> pd.DataFrame:
    """P(absent | position group, streak bucket) from seasons before `upto_season`."""
    r = ro[ro["season"] < upto_season]
    b = r.assign(sb=np.minimum(r["streak"], 3).astype(int), g=r["grp"].where(r["grp"].isin(["QB", "RB", "WR", "TE"]), "DEF"))
    return b.groupby(["g", "sb"])["absent"].mean()


def streak_p_out(ro: pd.DataFrame) -> pd.DataFrame:
    """p_out per role row from the box-score streak: P(absent | group, missed last k games) measured
    on earlier seasons (streak 0 -> 0, so only a known signal moves it). Adds `src`."""
    out = ro.copy()
    out["g"] = out["grp"].where(out["grp"].isin(["QB", "RB", "WR", "TE"]), "DEF")
    out["sb"] = np.minimum(out["streak"], 3).astype(int)
    p = np.zeros(len(out))
    for s in out["season"].unique():
        sr = streak_rates(ro, s)
        msk = (out["season"] == s).to_numpy()
        keys = list(zip(out.loc[msk, "g"], out.loc[msk, "sb"]))
        p[msk] = [sr.get(k, 0.0) if k[1] > 0 else 0.0 for k in keys]
    out["p_out"] = p
    out["src"] = np.where(out["sb"] > 0, "streak", "none")
    return out


def overlay_listings(po: pd.DataFrame, lst: pd.DataFrame, rates: dict, status_col: str = "status") -> pd.DataFrame:
    """Replace p_out with P(absent | listed status) where a listing exists for that player and game."""
    out = po.copy()
    if lst is None or not len(lst):
        return out
    m = out[["game_id", "team_id", "athlete_id"]].reset_index().merge(
        lst[["game_id", "team_id", "athlete_id", status_col, "source"]], on=["game_id", "team_id", "athlete_id"])
    m["pl"] = m[status_col].map({k: v[0] for k, v in rates.items()})
    m = m[m["pl"].notna()]
    out.loc[m["index"], "p_out"] = m["pl"].to_numpy()
    out.loc[m["index"], "src"] = m["source"].to_numpy()
    return out


# ---------------------------------------------------------------------------- team features

def qb_league_means(ro: pd.DataFrame, before_season: int) -> tuple[float, float]:
    """League EPA/dropback of season starters (top QB by dropbacks) and of other QBs."""
    r = ro[(ro["season"] < before_season) & (ro["grp"] == "QB")]
    last = r.sort_values("k").groupby(["season", "team_id", "athlete_id"]).tail(1)
    top = last.sort_values("n_drop").groupby(["season", "team_id"]).tail(1)
    rest = last.drop(top.index)
    mu_s = top["epa_drop"].sum() / max(top["n_drop"].sum(), 1)
    mu_b = rest["epa_drop"].sum() / max(rest["n_drop"].sum(), 1)
    return float(mu_s), float(mu_b)


def team_features(ro: pd.DataFrame, mu: dict) -> pd.DataFrame:
    """Per (game_id, team_id) injury columns from role rows with p_out.

    mu: season -> (starter mean, backup mean) EPA/dropback for shrinkage."""
    r = ro.copy()
    r["oracle"] = r["absent"]
    out = []
    for (season, game, team), g in r.groupby(["season", "game_id", "team_id"], sort=False):
        mu_s, mu_b = mu[season]
        row = {"season": season, "game_id": game, "team_id": team}
        qb = g[g["grp"] == "QB"].sort_values("n_drop", ascending=False)
        if len(qb) and qb["n_drop"].iloc[0] >= 20:
            s = qb.iloc[0]
            es = (s["epa_drop"] + K_QB * mu_s) / (s["n_drop"] + K_QB)
            # replacement = next QB by dropbacks who is not expected out (p_out < 0.5); none -> league backup
            rest = qb.iloc[1:]
            rest = rest[rest["p_out"] < 0.5]
            if len(rest):
                b = rest.iloc[0]
                eb = (b["epa_drop"] + K_QB * mu_b) / (b["n_drop"] + K_QB)
            else:
                eb = mu_b
            vor = max(es - eb, 0.0) * s["team_dpg"]
            row.update(qb_vor=vor, qb_out=vor * s["p_out"], qb_baked=vor * s["missed_frac"],
                       qb_out_orc=vor * s["oracle"], qb_p=s["p_out"], qb_src=s["src"],
                       qb_share_out=s["sh_drop"] * s["p_out"])
        else:
            row.update(qb_vor=0.0, qb_out=0.0, qb_baked=0.0, qb_out_orc=0.0, qb_p=0.0, qb_src="none", qb_share_out=0.0)
        for grp, col in (("RB", "sh_carry"), ("WR", "sh_tgt"), ("TE", "sh_tgt")):
            x = g[g["grp"] == grp]
            row[f"{grp.lower()}_out"] = float((x[col] * x["p_out"]).sum())
            row[f"{grp.lower()}_baked"] = float((x[col] * x["missed_frac"]).sum())
            row[f"{grp.lower()}_out_orc"] = float((x[col] * x["oracle"]).sum())
        d = g[g["grp"].isin(DEF)].nlargest(11, "sh_tkl")
        row["def_out"] = float((d["sh_tkl"] * d["p_out"]).sum())
        row["def_n_out"] = float(d["p_out"].sum())
        row["def_out_orc"] = float((d["sh_tkl"] * d["oracle"]).sum())
        key = pd.concat([qb.head(1), g[(g["grp"] == "RB") & (g["sh_carry"] >= 0.30)],
                         g[g["grp"].isin(["WR", "TE"]) & (g["sh_tgt"] >= 0.15)]])
        row["key_n_out"] = float(key["p_out"].sum())
        row["key_multi"] = float((key["p_out"] >= 0.5).sum() >= 2)
        row["key_n_out_orc"] = float(key["oracle"].sum())
        row["key_multi_orc"] = float(key["oracle"].sum() >= 2)
        row["listed_n"] = int((g["src"].astype(str).str.startswith(("hdi", "cfbdepth", "check"))).sum())
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- CFBDepth position exports

UNIT_SNAPS = {   # export -> (snap columns, starters per team for the reference)
    "ol": (("Snaps LT", "Snaps LG", "Snaps C", "Snaps RG", "Snaps RT"), 5),
    "dl": (("Snaps DL", "Snaps OffBall"), 4),
    "lb": (("Snaps OffBall", "Snaps DL", "Snaps FS"), 3),
    "db": (("Snaps CB", "Snaps Slot", "Snaps FS"), 4),
}


def cfbdepth_unit_values(export_dir: Path, data: Path, teams: pd.DataFrame, season: int = 2026) -> pd.DataFrame:
    """Starter share for OL / DL / LB / DB from the CFBDepth season exports (snaps through the
    download): snaps per game played / mean of the team's top-N players at the unit, capped at 1.
    Season totals as of the download, so for earlier weeks this is a snapshot, not point in time."""
    info = pd.read_csv(export_dir / "cfbdepth-general-information.csv")
    abbr = dict(zip(info["Abbrev"], info["School Name"]))
    r, tid = _roster_keys(data, season, teams)
    out = []
    for unit, (cols, n_top) in UNIT_SNAPS.items():
        adv = pd.read_csv(export_dir / f"cfbdepth-{unit}-advanced.csv")
        st = pd.read_csv(export_dir / f"cfbdepth-{unit}-stats.csv")[["Name", "School", "Games"]]
        rt = pd.read_csv(export_dir / f"cfbdepth-{unit}-ratings.csv")[["Name", "School", "Overall"]]
        d = adv.merge(st, on=["Name", "School"], how="left").merge(rt, on=["Name", "School"], how="left")
        d["snaps"] = sum(pd.to_numeric(d[c], errors="coerce").fillna(0.0) for c in cols if c in d)
        d["games"] = pd.to_numeric(d["Games"], errors="coerce")
        d = d[(d["snaps"] > 0) & (d["games"] > 0)]
        d["spg"] = d["snaps"] / d["games"]
        ref = d.groupby("School")["spg"].apply(lambda s: s.nlargest(n_top).mean())
        d["starter"] = (d["spg"] / d["School"].map(ref)).clip(upper=1.0)
        d["team"] = d["School"].map(abbr)
        d["team_id"] = d["team"].map(tid)
        d = d[d["team_id"].notna()]
        d["athlete_id"] = _match(d.rename(columns={"Name": "name"}), r)
        d["unit"] = unit.upper()
        d["rating"] = pd.to_numeric(d["Overall"], errors="coerce")
        out.append(d[["team_id", "athlete_id", "unit", "starter", "spg", "rating", "Name", "School"]])
    o = pd.concat(out, ignore_index=True)
    o = o[o["athlete_id"].notna()]
    o["athlete_id"] = o["athlete_id"].astype("int64")
    o["team_id"] = o["team_id"].astype("int64")
    # a player in two exports (e.g. LB and DL) keeps the larger starter share
    return o.sort_values("starter", ascending=False).drop_duplicates(["team_id", "athlete_id"])


def unit_out(listings: pd.DataFrame, values: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Per (game_id, team_id): sum of starter share x P(out) for listed OL and defensive players."""
    m = listings.merge(values, on=["team_id", "athlete_id"], how="inner")
    m["p"] = m["status"].map({k: v[0] for k, v in rates.items()}).fillna(0.0)
    m["v"] = m["p"] * m["starter"]
    m["vr"] = m["v"] * m["rating"].fillna(m["rating"].median()) / 15.0
    m["grp"] = np.where(m["unit"] == "OL", "ol", "dfs")
    w = m.pivot_table(index=["game_id", "team_id"], columns="grp", values=["v", "vr"], aggfunc="sum", fill_value=0.0)
    w.columns = [f"{g}_out" if a == "v" else f"{g}_out_r" for a, g in w.columns]
    for c in ("ol_out", "dfs_out", "ol_out_r", "dfs_out_r"):
        if c not in w:
            w[c] = 0.0
    return w.reset_index()
