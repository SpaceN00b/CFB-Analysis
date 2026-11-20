"""Players page engine (cfb.ui.availability) and the Cloudflare connection helpers, on a synthetic
two-team week: the per-team recompute equals the cloud's all-teams run, overrides move the cost
the right way, packages and settings round-trip."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cfb.lab import depth_impact as D
from cfb.ui import availability as AV

S = 2026
TEAMS = {1: "Alpha", 2: "Beta"}
WEEK_GAME = 900
# athlete -> (team, align, per-game drop, carry, tgt, epa per touch)
PLAYERS = {
    11: (1, "QB", 30, 3, 0, 0.25), 12: (1, "QB", 4, 0, 0, 0.05), 13: (1, "QB", 2, 0, 0, -0.10),
    14: (1, "RB", 0, 15, 2, 0.10), 15: (1, "RB", 0, 5, 1, 0.00),
    16: (1, "X", 0, 0, 8, 0.45), 17: (1, "Z", 0, 0, 7, 0.40), 18: (1, "SLOT", 0, 0, 6, 0.35),
    19: (1, "X", 0, 0, 2, 0.10), 20: (1, "TE", 0, 0, 3, 0.30),
    21: (2, "QB", 28, 2, 0, 0.15), 22: (2, "QB", 5, 0, 0, 0.00),
    23: (2, "RB", 0, 18, 2, 0.05), 24: (2, "X", 0, 0, 9, 0.30), 25: (2, "Z", 0, 0, 6, 0.25),
    26: (2, "SLOT", 0, 0, 5, 0.20), 27: (2, "TE", 0, 0, 3, 0.20), 28: (2, "Z", 0, 0, 2, 0.05),
}
CHART_POS = {"QB": "QB", "RB": "RB", "X": "WR-X", "Z": "WR-Z", "SLOT": "WR-SL", "TE": "TE"}


def build_inputs():
    games = {S: [100, 101, 102, 103], S - 1: [50, 51, 52, 53]}
    u_rows, sched = [], []
    for season, gids in games.items():
        for i, gid in enumerate(gids):
            for t in TEAMS:
                sched.append((season, gid, t, pd.Timestamp(f"{season}-09-0{i + 1}", tz="UTC")))
        for a, (t, al, dr, ca, tg, e) in PLAYERS.items():
            for gid in gids:
                u_rows.append((season, gid, t, a, dr, ca, tg, dr * e, ca * e, tg * e))
    for t in TEAMS:
        sched.append((S, WEEK_GAME, t, pd.Timestamp(f"{S}-11-21", tz="UTC")))
    u = pd.DataFrame(u_rows, columns=["season", "game_id", "team_id", "athlete_id", "drop", "carry", "tgt",
                                      "drop_epa", "carry_epa", "tgt_epa"])
    sched = pd.DataFrame(sched, columns=["season", "game_id", "team_id", "start_time_utc"])
    lab = pd.DataFrame([(s, a, al, D.ALIGN_GRP[al]) for s in games for a, (t, al, *_r) in PLAYERS.items()],
                       columns=["season", "athlete_id", "align", "grp"])
    rost = pd.DataFrame([(s, t, a, D.ALIGN_GRP[al]) for s in games for a, (t, al, *_r) in PLAYERS.items()],
                        columns=["season", "team_id", "athlete_id", "rgrp"])
    teams = pd.DataFrame([(S, n, t, "SEC") for t, n in TEAMS.items()], columns=["season", "team", "team_id", "conference"])
    chart = []
    for a, (t, al, *_r) in PLAYERS.items():
        depth = 1 + sum(1 for b, (tb, alb, *_x) in PLAYERS.items() if tb == t and alb == al and b < a)
        chart.append((TEAMS[t], "offense", CHART_POS[al], float(depth), f"P{a}", float(a), al))
    chart = pd.DataFrame(chart, columns=["team", "unit", "position", "depth", "player", "athlete_id", "role"])
    mu = {S: {"QB": (0.15, 0.0), "RB": (0.05, 0.0), "WR": (0.25, 0.1), "TE": (0.2, 0.1)}}
    prev = D.prev_season_stats(u, lab)
    prev = prev[prev["season"] == S]
    # listings: Alpha's QB1 out, Beta's X questionable
    listings = pd.DataFrame([(WEEK_GAME, 1, 11, 0.998, "Out", True), (WEEK_GAME, 2, 24, 0.57, "Questionable", True)],
                            columns=["game_id", "team_id", "athlete_id", "p", "status", "off_chart"])
    return dict(u=u[u["season"] == S], prev=prev, sched=sched[sched["season"] == S], lab=lab[lab["season"] == S],
                rost=rost[rost["season"] == S], chart=chart, listings=listings,
                off_chart=listings[["team_id", "athlete_id", "status"]], teams=teams,
                names=pd.DataFrame({"athlete_id": list(PLAYERS), "name": [f"Player {a}" for a in PLAYERS]})), mu


def cloud_costs(T, mu):
    """The cloud week run: all teams at once, chart = current chart less every listed-out player."""
    pf = {(r.game_id, r.team_id, r.athlete_id): r.p for r in T["listings"].itertuples()}
    wkc = D.week_chart(T["chart"], T["off_chart"], T["teams"])
    _, repl = D.chart_slots(wkc[wkc["depth"].notna()], T["teams"])
    rows, det = D.team_costs(T["u"], T["sched"], T["lab"], T["rost"], mu, pf, False, [S], record_all=True,
                             starter_by="total", chart_repl=repl, prev=T["prev"])
    r = rows[rows["game_id"] == WEEK_GAME].set_index("team_id")
    return r[AV.COST_COLS].sum(axis=1).to_dict(), det[det["game_id"] == WEEK_GAME]


@pytest.fixture()
def pkg():
    T, mu = build_inputs()
    cost, det = cloud_costs(T, mu)
    lin, s = 0.3, 0.12
    adj = -(cost[1] - cost[2])
    T["games"] = pd.DataFrame([{
        "game_id": WEEK_GAME, "start_time_utc": "2026-11-21 17:00:00+00:00", "home_team": "Alpha", "away_team": "Beta",
        "home_id": 1, "away_id": 2, "neutral": False, "pred": 2.5, "lin": lin, "calib_s": s,
        "p_v2": 1 / (1 + np.exp(-lin)), "p_dc": 1 / (1 + np.exp(-(lin + s * adj))), "home_cost": cost[1],
        "away_cost": cost[2], "home_out": "", "away_out": "", "p4": True}])
    T["cloud_scores"] = pd.DataFrame({"team_id": list(cost), "cost": list(cost.values())})
    meta = {"schema": AV.SCHEMA, "key": "2026_regular_12", "season": S, "week": 12, "label": "test",
            "created_utc": "2026-11-19T00:00:00Z", "mu": {g: list(v) for g, v in mu[S].items()},
            "sources": {"listings": "x", "charts": "y"}}
    return AV.WeekPackage.from_bytes(AV.WeekPackage(meta, T).to_bytes())


def test_recompute_matches_cloud_run(pkg):
    m = AV.WeekModel(pkg)
    cloud = pkg.tables["cloud_scores"].set_index("team_id")["cost"]
    for t in TEAMS:
        assert m.team(t, {}).cost == pytest.approx(float(cloud[t]), abs=1e-12)
    assert m.team(1, {}).cost > 0  # QB1 out costs his value over QB2
    p = m.picks({})
    assert p["p_mine"].iloc[0] == pytest.approx(p["p_dc"].iloc[0], abs=1e-12)


def test_overrides_move_the_cost(pkg):
    m = AV.WeekModel(pkg)
    base = m.team(1, {}).cost
    assert m.team(1, {11: "available"}).cost == pytest.approx(0.0, abs=1e-12)
    # QB2 out as well: QB1's replacement is the worse QB3
    both = m.team(1, {12: "out"})
    assert both.cost > base
    qb = both.starters[both.starters["grp"] == "QB"].iloc[0]
    assert int(qb["repl_id"]) == 13
    # Beta: questionable X marked out costs more, marked available costs nothing
    b = m.team(2, {}).cost
    assert m.team(2, {24: "out"}).cost > b > m.team(2, {24: "available"}).cost == pytest.approx(0.0, abs=1e-12)
    p = m.picks({1: {11: "available"}})
    assert p["p_mine"].iloc[0] > p["p_dc"].iloc[0]  # home team gets its QB back
    assert bool(p["changed"].iloc[0])


def test_players_table(pkg):
    m = AV.WeekModel(pkg)
    df = m.players(1, {12: "out"})
    assert list(df["grp"].unique()) == ["QB", "RB", "WR", "TE"]
    row = df.set_index("athlete_id").loc[11]
    assert row["listing"] == "Out" and row["setting"] == "Listing" and bool(row["starter"])
    assert df.set_index("athlete_id").loc[12, "setting"] == "Out"


# ---- defense (PR 37) -------------------------------------------------------------------------

# athlete -> (team, group, chart label or None, per-game involvement, per-game disruption points)
DEFENDERS = {
    101: (1, "DL", "DE", 3, 1.8), 102: (1, "DL", "DT", 3, 0.9), 103: (1, "DL", "DT", 3, 0.8),
    104: (1, "DL", "DE", 3, 0.7), 105: (1, "DL", "DE", 1, 0.2),
    106: (1, "LB", "LB", 7, 1.2), 107: (1, "LB", "LB", 6, 1.0), 108: (1, "LB", "LB", 5, 0.9),
    109: (1, "LB", "LB", 1, 0.1),
    **{a: (1, "DB", "CB", 4, 1.0) for a in range(110, 115)}, 115: (1, "DB", "CB", 1, 0.1),
    **{a: (2, g, None, n, v) for a, (g, n, v) in {
        201: ("DL", 3, 1.0), 202: ("DL", 3, 1.0), 203: ("DL", 3, 1.0), 204: ("DL", 3, 1.0), 205: ("DL", 1, 0.3),
        206: ("LB", 6, 1.1), 207: ("LB", 6, 1.1), 208: ("LB", 6, 1.1), 209: ("LB", 1, 0.2),
        210: ("DB", 5, 1.5), 211: ("DB", 4, 1.0), 212: ("DB", 4, 1.0), 213: ("DB", 4, 1.0), 214: ("DB", 4, 1.0),
        215: ("DB", 1, 0.1)}.items()},
}
MU_D = {g: {"start": 1.0, "fill": 0.6} for g in ("DL", "LB", "DB")}
K_D = {"DL": 6.0, "LB": 10.0, "DB": 24.0}


@pytest.fixture()
def pkg_def(pkg):
    from cfb.lab import defense_impact as F

    T = dict(pkg.tables)
    rows = []
    for season, gids in {S: [100, 101, 102, 103], S - 1: [50, 51, 52, 53]}.items():
        for a, (t, g, _lab, n, v) in DEFENDERS.items():
            for gid in gids:
                rows.append((season, gid, t, a, g, float(n), float(v)))
    dg_all = pd.DataFrame(rows, columns=["season", "game_id", "team_id", "athlete_id", "grp", "inv", "dv"])
    dprev = F.prev_season(dg_all)
    dprev = dprev[dprev["season"] == S]
    dg = dg_all[dg_all["season"] == S]
    def_chart = []
    for a, (t, g, lab, *_r) in DEFENDERS.items():
        if lab:
            depth = 1 + sum(1 for b, (tb, _g, lb, *_x) in DEFENDERS.items() if tb == t and lb == lab and b < a)
            role = {"DE": "EDGE", "DT": "IDL", "LB": "LB", "CB": "CB"}[lab]
            def_chart.append((TEAMS[t], "defense", lab, float(depth), f"D{a}", float(a), role))
    def_chart = pd.DataFrame(def_chart, columns=["team", "unit", "position", "depth", "player", "athlete_id", "role"])
    # Alpha's best edge rusher is listed out
    extra = pd.DataFrame([(WEEK_GAME, 1, 101, 0.998, "Out", True)], columns=T["listings"].columns)
    T["listings"] = pd.concat([T["listings"], extra], ignore_index=True)
    T["off_chart"] = T["listings"][["team_id", "athlete_id", "status"]]
    pf = {(r.game_id, r.team_id, r.athlete_id): r.p for r in T["listings"].itertuples()}
    repl = F.chart_defense(D.week_chart(def_chart, T["off_chart"], T["teams"]), T["teams"])
    _, det = F.team_costs(dg, T["sched"], MU_D, K_D, pf, False, [S], record_all=True, chart_repl=repl, prev=dprev)
    sc = det[det["game_id"] == WEEK_GAME].copy()
    sc["charted_team"] = sc["team_id"].isin(set(repl))
    sc["cost_chart"] = np.where(sc["charted_team"], sc["vor_own"], sc["vor_fill"]) * sc["p_out"]
    T.update(dg=dg, dprev=dprev, def_chart=def_chart, def_scores=sc)
    meta = dict(pkg.meta, **{"def": {"mu": MU_D, "K": K_D}})
    return AV.WeekPackage.from_bytes(AV.WeekPackage(meta, T).to_bytes())


def test_defense_recompute_matches_cloud(pkg_def):
    m = AV.WeekModel(pkg_def)
    assert m.has_def
    cloud = pkg_def.tables["def_scores"].groupby("team_id")["cost_chart"].sum()
    for t in TEAMS:
        assert m.team(t, {}).cost_def == pytest.approx(float(cloud[t]), abs=1e-12)
    a = m.team(1, {})
    assert a.cost_def > 0 and a.cost == pytest.approx(a.cost_off + a.cost_def)
    dl = a.starters.set_index("athlete_id").loc[101]
    assert dl["grp"] == "DL" and int(dl["repl_id"]) == 105  # next man up at DE (104 is a healthy regular)
    p = m.picks({})
    assert p["p_mine"].iloc[0] == pytest.approx(p["p_dc"].iloc[0], abs=1e-12)
    p_off = m.picks({}, defense=False)
    assert p_off["p_dc"].iloc[0] == pytest.approx(float(pkg_def.tables["games"]["p_dc"].iloc[0]), abs=1e-12)
    assert p["p_dc"].iloc[0] < p_off["p_dc"].iloc[0]  # home team's defense is short a starter


def test_defense_overrides(pkg_def):
    m = AV.WeekModel(pkg_def)
    assert m.team(1, {101: "available"}).cost_def == pytest.approx(0.0, abs=1e-12)
    # Beta has no defensive chart: its out DB costs his value over the league fill-in rate
    b = m.team(2, {210: "out"})
    row = b.starters.set_index("athlete_id").loc[210]
    assert b.cost_def > 0 and row["repl_rate"] == pytest.approx(MU_D["DB"]["fill"])
    assert m.team_cost(2, {210: "out"}, defense=False) == pytest.approx(m.team(2, {}).cost_off)


def test_players_table_with_defense(pkg_def):
    m = AV.WeekModel(pkg_def)
    df = m.players(1, {})
    assert list(df["grp"].unique()) == ["QB", "RB", "WR", "TE", "DL", "LB", "DB"]
    r = df.set_index("athlete_id").loc[101]
    assert r["listing"] == "Out" and bool(r["starter"]) and r["cost_now"] > 0 and r["chart"] == "DE 1"
    assert set(m.players(2, {})["grp"]) >= {"DL", "LB", "DB"}


def test_overrides_file_round_trip(tmp_path):
    o = AV.Overrides(tmp_path / "ov.json")
    o.set("2026_regular_12", 1, 11, "available", "Player 11")
    o.set("2026_regular_12", 2, 24, "out")
    assert o.week("2026_regular_12") == {1: {11: "available"}, 2: {24: "out"}}
    assert o.week("2026_regular_13") == {}
    o.set_team("2026_regular_12", 1, {11: "listing", 12: "out"})
    assert o.week("2026_regular_12") == {1: {12: "out"}, 2: {24: "out"}}
    o.set("2026_regular_12", 2, 24, "listing")
    o.clear("2026_regular_12")
    assert json.loads((tmp_path / "ov.json").read_text()) == {}


def test_publish_and_fetch(pkg, tmp_path):
    from cfb.warehouse_sync import LocalStore

    store = LocalStore(tmp_path / "bucket")
    info = AV.publish(store, pkg)
    assert info["object"] == "extras/availability/2026_regular_12.zip"
    lines = []
    dest = AV.fetch_latest(store, tmp_path / "local", log=lines.append)
    assert dest.exists() and AV.WeekPackage.load(dest).key == "2026_regular_12"
    assert AV.fetch_latest(store, tmp_path / "local", log=lines.append) == dest
    assert "up to date" in lines[-1]
    assert AV.local_packages(tmp_path / "local") == [dest]
    assert AV.fetch_latest(LocalStore(tmp_path / "empty"), tmp_path / "x", log=lines.append) is None


# ---- Cloudflare connection --------------------------------------------------------------------

def test_connection_save_status_and_forget(monkeypatch, tmp_path):
    from cfb.ui import connection as C

    acc = "0123456789abcdef0123456789abcdef"
    assert C.parse_account_id(f"https://{acc}.r2.cloudflarestorage.com") == acc
    assert C.parse_account_id(acc.upper()) == acc
    assert C.parse_account_id("nope") is None
    assert not C.status().configured
    with pytest.raises(ValueError):
        C.save("not an id", "k", "s")
    saved = C.save(f"https://{acc}.r2.cloudflarestorage.com/cfb-warehouse", "kid", "sec", "")
    assert set(saved) == {"r2_account_id", "r2_access_key_id", "r2_secret_access_key"}
    st = C.status()
    assert st.configured and st.where() == "the encrypted keys file"
    C.save("", "", "new-secret")  # blank fields keep the saved values
    from cfb import settings
    assert settings.secret("r2_access_key_id") == "kid" and settings.secret("r2_secret_access_key") == "new-secret"
    C.forget()
    assert not C.status().configured


def test_connection_check_reads_the_manifest(monkeypatch, tmp_path):
    from cfb.ui import connection as C

    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "manifest.json").write_text(json.dumps({"version": "v1"}))
    monkeypatch.setenv("CFB_WAREHOUSE_LOCAL", str(bucket))
    C.save("0123456789abcdef0123456789abcdef", "kid", "sec")
    assert C.check() == "v1"


def test_prefs_and_autostart(monkeypatch, tmp_path):
    from cfb.ui import connection as C
    from cfb.ui import sync_job

    monkeypatch.setenv("CFB_APP_PREFS", str(tmp_path / "app.json"))
    assert C.prefs()["auto_sync"] is True
    started = []
    monkeypatch.setitem(sync_job._AUTO, "done", False)
    assert sync_job.autostart(start=lambda run: started.append(run) or True) is False  # no keys saved
    C.save("0123456789abcdef0123456789abcdef", "kid", "sec")
    monkeypatch.setitem(sync_job._AUTO, "done", False)
    assert sync_job.autostart(start=lambda run: started.append(run) or True) is True
    assert sync_job.autostart(start=lambda run: started.append(run) or True) is False  # once per process
    C.set_pref("auto_sync", False)
    monkeypatch.setitem(sync_job._AUTO, "done", False)
    assert sync_job.autostart(start=lambda run: started.append(run) or True) is False
    assert started == [sync_job.run_sync]


# ---- the page --------------------------------------------------------------------------------

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.mark.parametrize("which", ["pkg", "pkg_def"])
def test_players_page_renders(which, request, monkeypatch, tmp_path):
    pkg = request.getfixturevalue(which)
    monkeypatch.setenv("CFB_AVAILABILITY_DIR", str(tmp_path / "av"))
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_WAREHOUSE_DIR", str(tmp_path / "warehouse"))
    (tmp_path / "av").mkdir()
    (tmp_path / "av" / "2026_regular_12.zip").write_bytes(pkg.to_bytes())
    AV.Overrides(tmp_path / "av" / "overrides.json").set("2026_regular_12", 1, 11, "available", "Player 11")

    def script():
        from cfb.ui import pages
        pages.page_players()

    at = st_testing.AppTest.from_function(script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Players and injuries"
    assert len(at.dataframe) >= 2  # picks table and your changes


def test_players_page_without_package(monkeypatch, tmp_path):
    monkeypatch.setenv("CFB_AVAILABILITY_DIR", str(tmp_path / "none"))

    def script():
        from cfb.ui import pages
        pages.page_players()

    at = st_testing.AppTest.from_function(script, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.info
