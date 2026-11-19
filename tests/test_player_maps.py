"""Player maps test bench (cfb.lab.player_maps): location labels, defender events, the role model
and the clean-incompletion model on small synthetic inputs."""
import math

import numpy as np
import pandas as pd

from cfb.lab import player_maps as pm


def test_side_lean_is_measured_against_the_league_share():
    assert pm.side_lean(30, 10, center=0.5) == "offense's left (defense's right)"
    assert pm.side_lean(10, 30, center=0.5) == "offense's right (defense's left)"
    assert pm.side_lean(20, 20, center=0.5) is None
    assert pm.side_lean(12, 18, center=0.25) == "offense's left (defense's right)"  # 40% left vs 25% league
    assert pm.side_lean(5, 1, center=0.5) is None  # too few plays
    assert pm.side_lean(30, 10, defense=False) == "offense's left"


def test_primary_location_depth_and_lane():
    base = {"rush_rate": 0.05, "depth_med": 1.0, "left": 10, "middle": 30, "right": 10}
    assert pm.primary_location(pd.Series(base), 0.45, 0.5) == "At the line (0-2 yds), between the tackles"
    deep = dict(base, depth_med=8.0, left=30, middle=5, right=10)
    assert pm.primary_location(pd.Series(deep), 0.45, 0.5) == (
        "Intermediate (7-12 yds), outside, leans offense's left (defense's right)")
    rush = dict(base, rush_rate=0.5, left=0, middle=0, right=0)
    assert pm.primary_location(pd.Series(rush), 0.45, 0.5) == "Backfield (pass rush), side unknown"


def _bundle():
    plays = pd.DataFrame({
        "play_id": ["1", "2", "3", "4", "5"], "game_id": [10] * 5, "season": [2026] * 5, "week": [1] * 5,
        "offense_id": [1] * 5, "defense_id": [2] * 5,
        "action": ["rush", "pass", "pass", "sack", "pass"],
        "run_dir": ["left", pd.NA, pd.NA, pd.NA, pd.NA],
        "pass_dir": [pd.NA, "right", "middle", pd.NA, "left"],
        "completed": [pd.NA, True, False, pd.NA, False],
        "text_yards": [3, 12, 0, -7, 0], "air_yards": [pd.NA, 8, 25, pd.NA, 6],
        "n_tacklers": [1, 2, pd.NA, pd.NA, pd.NA], "garbage_time": [False] * 5,
    })
    players = pd.DataFrame({
        "play_id": ["1", "2", "2", "3", "4", "5"],
        "role": ["tackler", "tackler", "tackler", "breakup", "sacker", "hurry"],
        "athlete_id": pd.array([7, 7, 8, 8, 9, 9], dtype="Int64"), "team_id": [2] * 6,
        "side": ["defense"] * 6,
    })
    return {"plays": plays, "players": players}


def test_defense_events_place_each_credit():
    e = pm.defense_events(_bundle()).set_index(["play_id", "athlete_id"])
    assert e.loc[("1", 7), ["kind", "lat", "depth", "zone_depth"]].tolist() == ["run_tackle", "left", 3.0, "Short 3-7"]
    assert e.loc[("2", 8), ["kind", "lat", "depth"]].tolist() == ["pass_tackle", "right", 12.0]
    assert e.loc[("3", 8), ["kind", "lat", "depth", "zone_depth"]].tolist() == ["breakup", "middle", 25.0, "Deep 16+"]
    assert e.loc[("4", 9), ["kind", "depth", "zone_depth"]].tolist() == ["sack", -7.0, "Backfield"]
    assert e.loc[("5", 9), "kind"] == "hurry" and pd.isna(e.loc[("5", 9), "lat"])


def test_defender_profiles_counts():
    prof = pm.defender_profiles(pm.defense_events(_bundle())).set_index("athlete_id")
    assert prof.loc[7, "n_events"] == 2 and prof.loc[7, "f_run"] == 0.5
    assert prof.loc[8, "cov_rate"] == 0.5 and prof.loc[8, "shared"] == 1.0
    assert prof.loc[9, "rush_rate"] == 1.0


def test_softmax_separates_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2)), rng.normal([2, -2], 0.5, (50, 2))])
    y = np.repeat([0, 1, 2], 50)
    W = pm.softmax_fit(X, y, 3, iters=800)
    assert (pm.softmax_predict(W, X).argmax(axis=1) == y).mean() > 0.95


def test_binom_sf():
    assert math.isclose(pm.binom_sf(0, 5, 0.3), 1.0)
    assert math.isclose(pm.binom_sf(5, 5, 0.5), 1 / 32)
    assert math.isclose(pm.binom_sf(2, 3, 0.5), 0.5)


def test_clean_inc_model_puts_a_shared_excess_on_the_qb():
    """One QB misses on every receiver; one receiver misses with every QB."""
    rng = np.random.default_rng(1)
    rows = []
    for qb, qb_bad in ((1, 1.0), (2, 0.0), (3, 0.0)):
        for rec, rec_bad in ((11, 0.0), (12, 0.0), (13, 1.0), (14, 0.0)):
            for _ in range(250):
                air = int(rng.integers(-2, 30))
                eta = -1.6 + 0.04 * air + qb_bad + rec_bad
                rows.append({"qb_id": qb, "rec_id": rec, "defense_id": int(rng.integers(100, 110)),
                             "air": air, "pass_dir": rng.choice(list(pm.DIRS)), "down": int(rng.integers(1, 4)),
                             "clean_inc": float(rng.random() < 1 / (1 + math.exp(-eta)))})
    t = pd.DataFrame(rows)
    t["air_bin"] = pd.cut(t["air"], pm.AIR_BINS, labels=False, right=True)
    fit = pm.fit_clean_inc(t)
    qb = fit["qb"].set_index("id")["coef"]
    rec = fit["rec"].set_index("id")["coef"]
    assert qb[1] > qb[2] + 0.5 and qb[1] > qb[3] + 0.5
    assert rec[13] > max(rec[11], rec[12], rec[14]) + 0.5
    assert abs(rec[11] - rec[12]) < 0.4


def test_chart_role_maps_team_labels():
    from cfb.lab.player_maps import chart_role
    assert chart_role("LDE", "DL", three_man=False) == ("EDGE", "L")
    assert chart_role("DE", "DL", three_man=True) == ("IDL", None)
    assert chart_role("JACK", "EDGE", three_man=True) == ("EDGE", None)
    assert chart_role("RCB", "DB", three_man=False) == ("CB", "R")
    assert chart_role("STAR", "DB", three_man=False) == ("NB", None)
    assert chart_role("SS", "DB", three_man=False) == ("S", None)
    assert chart_role("WLB", "LB", three_man=False) == ("LB", None)
    assert chart_role("MONEY", "LB", three_man=False) == ("NB", None)


def test_depth_chart_roles_keeps_highest_listing(tmp_path):
    from cfb.lab.player_maps import depth_chart_roles
    d = tmp_path / "tables" / "depth_charts"
    d.mkdir(parents=True)
    ts = pd.Timestamp("2026-11-19T04:00:00Z")
    pd.DataFrame({
        "season": 2026, "asof_utc": ts, "team": "A",
        "unit": ["defense", "defense", "defense", "offense", "special_teams"],
        "position": ["DE", "JACK", "NB", "WR-SL", "KR"], "pos_group": ["DL", "EDGE", "DB", "WR", "ST"],
        "depth": [1, 1, 2, 1, 1], "athlete_id": [1, 2, 3, 4, 3],
    }).to_parquet(d / "2026-11-19_A.parquet")
    c = depth_chart_roles(tmp_path).set_index("athlete_id")
    assert c.loc[1, "chart_role"] == "IDL"       # three-man front: the chart lists a JACK
    assert c.loc[2, "chart_role"] == "EDGE"
    assert c.loc[3, "chart_pos"] == "NB" and c.loc[3, "chart_role"] == "NB"
    assert c.loc[4, "chart_align"] == "SLOT"
    assert depth_chart_roles(tmp_path / "none").empty
