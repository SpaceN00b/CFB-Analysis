"""Mark players available or out and recompute the week's picks (test model, PR 34 / PR 35).

The test model is model_v2 plus the depth-chart injury term (`cfb.lab.depth_impact`, the week
rule Josiah set 2026-11-19): each full-strength starter (most touches so far this season) who is
out costs his value over the next man up on this week's chart, times P(out); this week's chart is
the current team chart less every player listed out. Home minus away cost moves model_v2's margin
through its calibration slope.

The cloud builds a week package (`scripts/depth_impact_select.py` writes the inputs,
`scripts/availability_export.py` checks and packs them): this season's usage of the week's teams,
last season's stats, slot labels, rosters, offensive charts, the week's listings, model_v2's
linear predictor per game. On the PC, a team's cost is `depth_impact.team_costs` run on that team
alone with the user's overrides applied to P(out) and to the chart, so with no overrides it gives
the cloud's numbers (the export checks this for every team).

Defense (PR 37, optional tables `DEF_TABLES` and meta["def"]): each defensive regular (DL 4, LB 3,
DB 5 by involvement so far this season) who is out costs his disruption value (sacks, TFL, INT, PD,
hurries in points) over the next man up on this week's defensive chart, or over the league fill-in
rate for a team without a chart (`cfb.lab.defense_impact`). It tested at no measurable gain
(docs/defense-impact.md), so the Picks tab can leave it out. A package without the defense tables
gives the offense-only model.

Bucket keys: extras/availability/LATEST.json -> {"key", "object", "sha256", "created_utc"};
extras/availability/<key>.zip. Local: <user data>/availability/<key>.zip (downloaded copy) and
<user data>/availability/overrides.json (the user's settings, per week).
No Streamlit here, so it is testable without the UI.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import settings

PREFIX = "extras/availability/"
LATEST = PREFIX + "LATEST.json"
SCHEMA = 1
TABLES = ("u", "prev", "sched", "lab", "rost", "chart", "listings", "off_chart", "teams", "names", "games",
          "cloud_scores")
STATES = ("listing", "available", "out")
GROUPS = ("QB", "RB", "WR", "TE")
COST_COLS = ["qb_cost", "rb_cost", "wr_cost", "te_cost"]
DEF_TABLES = ("dg", "dprev", "def_chart", "def_scores")
DEF_GROUPS = ("DL", "LB", "DB")
DEF_COST_CHART = ["dl_cost_own", "lb_cost_own", "db_cost_own"]   # charted team: next man up on the chart
DEF_COST_FILL = ["dl_cost", "lb_cost", "db_cost"]                # no chart: league fill-in rate
P4 = {"ACC", "Big Ten", "Big 12", "SEC"}


def local_dir() -> Path:
    env = os.environ.get("CFB_AVAILABILITY_DIR")
    return Path(env) if env else settings.user_data_root() / "availability"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ---- package -----------------------------------------------------------------------------------

@dataclass
class WeekPackage:
    meta: dict
    tables: dict[str, pd.DataFrame]

    @property
    def key(self) -> str:
        return self.meta["key"]

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("meta.json", json.dumps(self.meta, indent=1, sort_keys=True))
            for name, df in self.tables.items():
                b = io.BytesIO()
                df.to_parquet(b, index=False)
                z.writestr(f"{name}.parquet", b.getvalue())
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "WeekPackage":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            meta = json.loads(z.read("meta.json"))
            if meta.get("schema") != SCHEMA:
                raise ValueError(f"unsupported availability package schema {meta.get('schema')}")
            tables = {n[:-8]: pd.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith(".parquet")}
        missing = [t for t in TABLES if t not in tables]
        if missing:
            raise ValueError(f"availability package is missing {missing}")
        return cls(meta, tables)

    @classmethod
    def load(cls, path: Path) -> "WeekPackage":
        return cls.from_bytes(Path(path).read_bytes())


def local_packages(folder: Path | None = None) -> list[Path]:
    """Downloaded packages, newest week first."""
    from .data import report_sort_key

    d = folder or local_dir()
    if not d.exists():
        return []
    return sorted(d.glob("*.zip"), key=lambda p: report_sort_key(p.stem), reverse=True)


def fetch_latest(store, folder: Path | None = None, log=print) -> Path | None:
    """Download the newest package from the bucket when the local copy differs. Returns its path,
    or None when the bucket has none."""
    raw = store.get(LATEST)
    if raw is None:
        log("no player-availability package in the bucket yet")
        return None
    info = json.loads(raw)
    d = folder or local_dir()
    dest = d / f"{info['key']}.zip"
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == info.get("sha256"):
        log(f"player availability {info['key']}: up to date")
        return dest
    data = store.get(info["object"])
    if data is None:
        raise RuntimeError(f"missing object {info['object']}")
    if info.get("sha256") and hashlib.sha256(data).hexdigest() != info["sha256"]:
        raise RuntimeError(f"checksum mismatch for {info['object']}")
    WeekPackage.from_bytes(data)  # readable before it replaces the local copy
    d.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    log(f"player availability {info['key']}: downloaded ({len(data) / 1e6:.1f} MB)")
    return dest


def publish(store, pkg: WeekPackage) -> dict:
    """Upload a package and point LATEST at it (cloud)."""
    data = pkg.to_bytes()
    obj = f"{PREFIX}{pkg.key}.zip"
    info = {"key": pkg.key, "object": obj, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "created_utc": pkg.meta.get("created_utc")}
    store.put_bytes(obj, data, content_type="application/zip")
    store.put_bytes(LATEST, json.dumps(info).encode("utf-8"))
    return info


# ---- the user's settings -------------------------------------------------------------------------

class Overrides:
    """{week key: {team_id: {athlete_id: {"state": "available" | "out", "name", "set_utc"}}}} in a
    local JSON file. "listing" (follow the injury list) is the absence of an entry."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else local_dir() / "overrides.json"

    def _read(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    def _write(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def week(self, key: str) -> dict[int, dict[int, str]]:
        """{team_id: {athlete_id: state}} for one week."""
        out = {}
        for t, players in (self._read().get(key) or {}).items():
            st = {int(a): v["state"] for a, v in players.items() if v.get("state") in ("available", "out")}
            if st:
                out[int(t)] = st
        return out

    def set(self, key: str, team_id: int, athlete_id: int, state: str, name: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        d = self._read()
        wk = d.setdefault(key, {})
        team = wk.setdefault(str(int(team_id)), {})
        if state == "listing":
            team.pop(str(int(athlete_id)), None)
            if not team:
                wk.pop(str(int(team_id)), None)
        else:
            team[str(int(athlete_id))] = {"state": state, "name": name,
                                          "set_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if not wk:
            d.pop(key, None)
        self._write(d)

    def set_team(self, key: str, team_id: int, states: dict[int, str], names: dict[int, str] | None = None) -> None:
        """Replace one team's settings for the week ("listing" entries are dropped)."""
        d = self._read()
        wk = d.setdefault(key, {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        names = names or {}
        team = {str(int(a)): {"state": s, "name": names.get(a), "set_utc": now}
                for a, s in states.items() if s in ("available", "out")}
        if team:
            wk[str(int(team_id))] = team
        else:
            wk.pop(str(int(team_id)), None)
        if not wk:
            d.pop(key, None)
        self._write(d)

    def clear(self, key: str) -> None:
        d = self._read()
        if d.pop(key, None) is not None:
            self._write(d)


# ---- the model -----------------------------------------------------------------------------------

@dataclass
class TeamResult:
    team_id: int
    game_id: int | None
    cost: float             # offense + defense (offense only for a package without defense)
    starters: pd.DataFrame  # grp, athlete_id, player, p_out, touches_pg, rate, replacement, repl_rate, vor, cost
    cost_off: float = 0.0
    cost_def: float = 0.0


class WeekModel:
    def __init__(self, pkg: WeekPackage):
        from cfb.lab import depth_impact as D

        self.D = D
        self.pkg = pkg
        self.meta = pkg.meta
        self.season = int(pkg.meta["season"])
        T = pkg.tables
        self.u, self.prev, self.lab, self.rost = T["u"], T["prev"], T["lab"], T["rost"]
        self.sched = T["sched"]
        self.chart = T["chart"]
        self.teams = T["teams"]
        self.games = T["games"].copy()
        self.mu = {self.season: {g: tuple(v) for g, v in pkg.meta["mu"].items()}}
        t = self.teams[self.teams["season"] == self.season]
        self.team_name = dict(zip(t["team_id"].astype(int), t["team"]))
        self.team_conf = dict(zip(t["team_id"].astype(int), t.get("conference", pd.Series(index=t.index, dtype=object))))
        nm = T["names"].dropna(subset=["athlete_id"])
        self.names = dict(zip(nm["athlete_id"].astype("int64"), nm["name"]))
        lst = T["listings"]
        self.listing = {(int(r.team_id), int(r.athlete_id)): (float(r.p), r.status) for r in lst.itertuples(index=False)}
        oc = T["off_chart"]
        self.off_chart = {}
        for r in oc.itertuples(index=False):
            self.off_chart.setdefault(int(r.team_id), set()).add(int(r.athlete_id))
        self.week_game = {}
        for g in self.games.itertuples(index=False):
            self.week_game[int(g.home_id)] = int(g.game_id)
            self.week_game[int(g.away_id)] = int(g.game_id)
        cs = T["cloud_scores"]
        self.cloud_cost = cs.groupby("team_id")["cost"].sum().to_dict() if "team_id" in cs else {}
        self.has_def = all(t in T for t in DEF_TABLES) and "def" in pkg.meta
        self.cloud_def: dict = {}
        if self.has_def:
            from cfb.lab import defense_impact as F

            self.F = F
            self.dg, self.dprev, self.def_chart = T["dg"], T["dprev"], T["def_chart"]
            self.mu_d = pkg.meta["def"]["mu"]
            self.K_d = pkg.meta["def"]["K"]
            ds = T["def_scores"]
            self.cloud_def = ds.groupby("team_id")["cost_chart"].sum().to_dict() if len(ds) else {}
            self.def_charted = set(self.def_chart["team"])
        self._cache: dict = {}

    @property
    def key(self) -> str:
        return self.pkg.key

    def week_teams(self) -> list[tuple[int, str]]:
        return sorted(((t, self.team_name.get(t, str(t))) for t in self.week_game), key=lambda x: x[1])

    # one team ------------------------------------------------------------------------------------
    def team(self, team_id: int, overrides: dict[int, str] | None = None) -> TeamResult:
        team_id = int(team_id)
        ov = {int(a): s for a, s in (overrides or {}).items() if s in ("available", "out")}
        ck = (team_id, tuple(sorted(ov.items())))
        if ck in self._cache:
            return self._cache[ck]
        D = self.D
        gid = self.week_game.get(team_id)
        if gid is None:
            return TeamResult(team_id, None, 0.0, pd.DataFrame())
        p_out = {(gid, team_id, a): p for (t, a), (p, _s) in self.listing.items() if t == team_id}
        off = set(self.off_chart.get(team_id, set()))
        for a, s in ov.items():
            p_out[(gid, team_id, a)] = 1.0 if s == "out" else 0.0
            if s == "out":
                off.add(a)
            else:
                off.discard(a)
        name = self.team_name.get(team_id)
        ch = self.chart[self.chart["team"] == name]
        repl = None
        if len(ch):
            out_ids = pd.DataFrame({"team_id": team_id, "athlete_id": sorted(off), "status": "out"},
                                   columns=["team_id", "athlete_id", "status"])
            wkc = D.week_chart(ch, out_ids, self.teams)
            _, repl = D.chart_slots(wkc[wkc["depth"].notna()], self.teams)
        sched = self.sched[self.sched["team_id"] == team_id]
        rows, det = D.team_costs(self.u, sched, self.lab, self.rost, self.mu, p_out, False, [self.season],
                                 record_all=True, starter_by="total", chart_repl=repl or None, prev=self.prev)
        r = rows[rows["game_id"] == gid] if len(rows) else rows
        cost = float(r[COST_COLS].sum(axis=1).iloc[0]) if len(r) else 0.0
        st = det[det["game_id"] == gid].copy() if len(det) else det
        cols = ["grp", "align", "athlete_id", "player", "p_out", "touches_pg", "rate", "repl_id", "replacement",
                "repl_rate", "vor", "cost", "season_long"]
        if len(st):
            st["player"] = st["athlete_id"].map(self.names)
            st["replacement"] = st["repl_id"].map(self.names)
            st["cost"] = st["vor"] * st["p_out"]
            st = st[cols]
        cost_def, sd = (self._defense(team_id, gid, name, p_out, off) if self.has_def
                        else (0.0, pd.DataFrame(columns=cols)))
        if len(sd):
            st = pd.concat([st, sd[cols]], ignore_index=True) if len(st) else sd[cols]
        res = TeamResult(team_id, gid, cost + cost_def, st, cost_off=cost, cost_def=cost_def)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[ck] = res
        return res

    def _defense(self, team_id: int, gid: int, name: str, p_out: dict, off: set) -> tuple[float, pd.DataFrame]:
        """Defensive regulars of one team for the week game: cost and one row per regular."""
        F, D = self.F, self.D
        dch = self.def_chart[self.def_chart["team"] == name]
        repl = None
        if len(dch):
            out_ids = pd.DataFrame({"team_id": team_id, "athlete_id": sorted(off), "status": "out"},
                                   columns=["team_id", "athlete_id", "status"])
            repl = F.chart_defense(D.week_chart(dch, out_ids, self.teams), self.teams) or None
        sched = self.sched[self.sched["team_id"] == team_id]
        rows, det = F.team_costs(self.dg, sched, self.mu_d, self.K_d, p_out, False, [self.season],
                                 record_all=True, chart_repl=repl, prev=self.dprev)
        r = rows[rows["game_id"] == gid] if len(rows) else rows
        cc = DEF_COST_CHART if repl else DEF_COST_FILL
        cost = float(r[cc].sum(axis=1).iloc[0]) if len(r) else 0.0
        d = det[det["game_id"] == gid].copy() if len(det) else det
        if len(d):
            al = (dch.dropna(subset=["athlete_id"]).sort_values("depth")
                  .assign(athlete_id=lambda x: x["athlete_id"].astype("int64")).drop_duplicates("athlete_id")
                  .set_index("athlete_id")["role"]) if len(dch) else pd.Series(dtype=object)
            d["align"] = d["athlete_id"].map(al)
            d["player"] = d["athlete_id"].map(self.names)
            d["replacement"] = d["repl_id"].map(self.names)
            d["touches_pg"] = d["inv_pg"]
            d["vor"] = d["vor_own"] if repl else d["vor_fill"]
            if not repl:
                d["repl_rate"] = [self.mu_d[g_]["fill"] for g_ in d["grp"]]
            d["cost"] = d["vor"] * d["p_out"]
            d["season_long"] = False
        return cost, d

    def team_cost(self, team_id: int, overrides: dict[int, str] | None = None, defense: bool = True) -> float:
        """Injury cost of a team this week (points). defense=False leaves out the defensive term."""
        ov = {a: s for a, s in (overrides or {}).items() if s in ("available", "out")}
        t = int(team_id)
        if not ov and t in self.cloud_cost and (not self.has_def or t in self.cloud_def or not defense):
            return float(self.cloud_cost[t]) + (float(self.cloud_def.get(t, 0.0)) if defense else 0.0)
        r = self.team(team_id, ov)
        return r.cost_off + (r.cost_def if defense else 0.0)

    # the slate -----------------------------------------------------------------------------------
    def picks(self, overrides: dict[int, dict[int, str]] | None = None, defense: bool = True) -> pd.DataFrame:
        """One row per game: model_v2, the test model on the listings ("p_dc"), and with the user's
        settings ("p_mine"); defense=False leaves out the defensive term in both."""
        ov = overrides or {}
        g = self.games.copy()
        hl = [self.team_cost(h, None, defense) for h in g["home_id"]]
        al = [self.team_cost(a, None, defense) for a in g["away_id"]]
        g["p_dc"] = _sigmoid(g["lin"] + g["calib_s"] * -(np.asarray(hl) - np.asarray(al)))
        g["home_cost_mine"] = [self.team_cost(h, ov.get(int(h)), defense) for h in g["home_id"]]
        g["away_cost_mine"] = [self.team_cost(a, ov.get(int(a)), defense) for a in g["away_id"]]
        g["adj_mine"] = -(g["home_cost_mine"] - g["away_cost_mine"])
        g["p_mine"] = _sigmoid(g["lin"] + g["calib_s"] * g["adj_mine"])
        g["changed"] = [bool(ov.get(int(h)) or ov.get(int(a))) for h, a in zip(g["home_id"], g["away_id"])]
        g["pick"] = np.where(g["p_mine"] >= 0.5, g["home_team"], g["away_team"])
        g["p_pick"] = np.where(g["p_mine"] >= 0.5, g["p_mine"], 1 - g["p_mine"])
        g["flip_vs_listed"] = (g["p_mine"] >= 0.5) != (g["p_dc"] >= 0.5)
        g["flip_vs_v2"] = (g["p_mine"] >= 0.5) != (g["p_v2"] >= 0.5)
        return g

    # one team's players --------------------------------------------------------------------------
    def players(self, team_id: int, overrides: dict[int, str] | None = None) -> pd.DataFrame:
        """QB / RB / WR / TE (and DL / LB / DB when the package has defense) of a team: season usage,
        chart spot, listing, your setting, cost if out. Defense "touches" = involvement (tackles + PD
        + hurries + INT)."""
        team_id = int(team_id)
        ov = {int(a): s for a, s in (overrides or {}).items()}
        u = self.u[self.u["team_id"] == team_id]
        use = u.groupby("athlete_id").agg(games=("game_id", "nunique"), drop=("drop", "sum"), carry=("carry", "sum"),
                                          tgt=("tgt", "sum"))
        lab = self.lab[self.lab["season"] == self.season].drop_duplicates("athlete_id").set_index("athlete_id")
        name = self.team_name.get(team_id)
        ch = self.chart[(self.chart["team"] == name) & self.chart["role"].isin(["QB", "RB", "X", "Z", "SLOT", "TE"])
                        & self.chart["athlete_id"].notna()].copy()
        ch["athlete_id"] = ch["athlete_id"].astype("int64")
        ch = ch.sort_values("depth")
        spot = ch.drop_duplicates("athlete_id").set_index("athlete_id")
        listed = {a: v for (t, a), v in self.listing.items() if t == team_id}
        ids = set(use.index.astype("int64")) | set(spot.index) | set(listed) | set(ov)
        dgrp, duse, dspot = {}, pd.DataFrame(), pd.DataFrame()
        if self.has_def:
            dt = self.dg[self.dg["team_id"] == team_id]
            duse = dt.groupby("athlete_id").agg(games=("game_id", "nunique"), inv=("inv", "sum"))
            dgrp = dt.drop_duplicates("athlete_id").set_index("athlete_id")["grp"].to_dict()
            dc = self.def_chart[(self.def_chart["team"] == name) & self.def_chart["athlete_id"].notna()].copy()
            dc["athlete_id"] = dc["athlete_id"].astype("int64")
            dspot = dc.sort_values("depth").drop_duplicates("athlete_id").set_index("athlete_id")
            for a, r_ in dspot.iterrows():
                dgrp.setdefault(a, self.F.CHART_GRP.get(r_["role"]))
            ids |= set(duse.index.astype("int64")) | set(dspot.index)
        res = self.team(team_id, ov)
        base = self.team(team_id, {})
        st = res.starters.set_index("athlete_id") if len(res.starters) else pd.DataFrame()
        st0 = base.starters.set_index("athlete_id") if len(base.starters) else pd.DataFrame()
        rows = []
        for a in ids:
            grp = lab["grp"].get(a) if a in lab.index else None
            if grp is None and a in spot.index:
                grp = self.D.ALIGN_GRP.get(spot.at[a, "role"])
            is_def = grp not in GROUPS and dgrp.get(a) in DEF_GROUPS
            if is_def:
                grp = dgrp[a]
            elif grp not in GROUPS:
                continue
            p, status = listed.get(a, (0.0, None))
            s = st if a in st.index else st0
            if is_def:
                sp = dspot if a in dspot.index else None
                chart_txt = f"{sp.at[a, 'position']} {int(sp.at[a, 'depth'])}" if sp is not None else None
                games = int(duse["games"].get(a, 0)) if a in duse.index else 0
                touches = int(round(duse["inv"].get(a, 0))) if a in duse.index else 0
                slot = sp.at[a, "role"] if sp is not None else grp
            else:
                touch = {"QB": "drop", "RB": "carry", "WR": "tgt", "TE": "tgt"}[grp]
                chart_txt = f"{spot.at[a, 'position']} {int(spot.at[a, 'depth'])}" if a in spot.index else None
                games = int(use["games"].get(a, 0)) if a in use.index else 0
                touches = int(use[touch].get(a, 0)) if a in use.index else 0
                slot = lab["align"].get(a) if a in lab.index else None
            rows.append({
                "athlete_id": int(a), "player": self.names.get(a, str(a)), "grp": grp, "slot": slot,
                "chart": chart_txt, "games": games, "touches": touches,
                "listing": status or "", "p_listed": p,
                "setting": {"available": "Available", "out": "Out"}.get(ov.get(a), "Listing"),
                "starter": a in st.index or a in st0.index,
                "value_pts": float(s.at[a, "vor"]) if a in s.index else None,
                "p_out_now": float(st.at[a, "p_out"]) if a in st.index else (1.0 if ov.get(a) == "out" else
                                                                          0.0 if ov.get(a) == "available" else p),
                "cost_now": float(st.at[a, "cost"]) if a in st.index else 0.0,
                "next_up": st.at[a, "replacement"] if a in st.index else None,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["g_order"] = df["grp"].map({g: i for i, g in enumerate(GROUPS + DEF_GROUPS)})
        return df.sort_values(["g_order", "starter", "touches"], ascending=[True, False, False]).drop(columns="g_order")


def load_model(path: Path) -> WeekModel:
    return WeekModel(WeekPackage.load(path))
