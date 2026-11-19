"""cfb.lab.depth_impact: slot labels, value over the slot replacement, adjusted chart."""
import numpy as np
import pandas as pd

from cfb.lab import depth_impact as D

TEAM = 7


def _games(season, n=5):
    t0 = pd.Timestamp(f"{season}-09-01", tz="UTC")
    return pd.DataFrame({"season": season, "game_id": [season * 100 + k for k in range(n)], "team_id": TEAM,
                         "start_time_utc": [t0 + pd.Timedelta(days=7 * k) for k in range(n)]})


def _u(season, plays):
    """plays: {athlete: [(game k, drops, drop_epa, tgts, tgt_epa), ...]}"""
    rows = []
    for a, gl in plays.items():
        for k, dr, de, tg, te in gl:
            rows.append({"game_id": season * 100 + k, "team_id": TEAM, "athlete_id": a, "carry": 0.0, "drop": dr,
                         "tgt": tg, "carry_epa": 0.0, "drop_epa": de, "tgt_epa": te, "tkl": 0.0, "season": season,
                         "start_time_utc": pd.NaT, "grp": "QB" if dr else "WR"})
    return pd.DataFrame(rows)


MU = {2026: {"QB": (0.2, 0.05), "RB": (0.1, 0.05), "WR": (0.6, 0.6), "TE": (0.6, 0.6)}}


def _setup():
    # QB 1 starts games 0-3 (30 dropbacks, +0.3 EPA each), QB 2 backs up; WRs 10 (X) and 11 (X backup)
    plays = {1: [(k, 30, 9.0, 0, 0.0) for k in range(4)],
             2: [(k, 3, 0.0, 0, 0.0) for k in range(4)] + [(4, 30, 0.0, 0, 0.0)],
             10: [(k, 0, 0.0, 8, 8.0) for k in range(5)],
             11: [(k, 0, 0.0, 2, 0.0) for k in range(5)],
             12: [(k, 0, 0.0, 6, 3.6) for k in range(5)],
             13: [(k, 0, 0.0, 5, 3.0) for k in range(5)]}
    u = _u(2026, plays)
    lab = pd.DataFrame({"season": 2026, "athlete_id": [1, 2, 10, 11, 12, 13],
                        "align": ["QB", "QB", "X", "X", "SLOT", "Z"], "grp": ["QB", "QB", "WR", "WR", "WR", "WR"]})
    rost = pd.DataFrame({"season": 2026, "team_id": TEAM, "athlete_id": [1, 2, 10, 11, 12, 13], "rgrp": "QB"})
    return u, lab, rost


def test_qb_out_costs_value_over_backup():
    u, lab, rost = _setup()
    sched = _games(2026)
    tf, det = D.team_costs(u, sched, lab, rost, MU, None, True, [2026], groups=("QB",))
    tf = tf.set_index("game_id")
    # game 4: QB 1 did not play (realized) -> cost = (rate1 - rate2) x 30 dropbacks per game
    r1 = (4 * 9.0 + D.K["QB"] * 0.2) / (120 + D.K["QB"])
    r2 = (0.0 + D.K["QB"] * 0.05) / (12 + D.K["QB"])
    assert np.isclose(tf.loc[202604, "qb_cost"], (r1 - r2) * 30.0)
    assert tf.loc[202603, "qb_cost"] == 0.0
    # baked-in discount: no games missed yet -> no discount
    assert np.isclose(tf.loc[202604, "qb_cost_d"], tf.loc[202604, "qb_cost"])
    d = det[det["game_id"] == 202604].iloc[0]
    assert d["athlete_id"] == 1 and d["repl_id"] == 2


def test_listing_mode_uses_p_out_and_wr_spot_replacement():
    u, lab, rost = _setup()
    sched = _games(2026)
    p = {(202603, TEAM, 10): 1.0}
    tf, det = D.team_costs(u, sched, lab, rost, MU, p, False, [2026], groups=("WR",))
    tf = tf.set_index("game_id")
    assert tf.loc[202602, "wr_cost"] == 0.0
    d = det[det["game_id"] == 202603].iloc[0]
    # X starter out -> the X backup (11) fills in, not a starter at another spot
    assert d["athlete_id"] == 10 and d["repl_id"] == 11
    r10 = (3 * 8.0 + D.K["WR"] * 0.6) / (24 + D.K["WR"])
    r11 = (0.0 + D.K["WR"] * 0.6) / (6 + D.K["WR"])
    assert np.isclose(tf.loc[202603, "wr_cost"], (r10 - r11) * 8.0)


def test_starter_rule_per_game_keeps_out_starter():
    # QB 1 plays games 0-1 (30 each), then is out; QB 2 plays 2-5 (25 each)
    plays = {1: [(0, 30, 9.0, 0, 0.0), (1, 30, 9.0, 0, 0.0)],
             2: [(k, 25, 2.5, 0, 0.0) for k in range(2, 6)] + [(0, 2, 0.0, 0, 0.0), (1, 2, 0.0, 0, 0.0)]}
    u = _u(2026, plays)
    lab = pd.DataFrame({"season": 2026, "athlete_id": [1, 2], "align": "QB", "grp": "QB"})
    rost = pd.DataFrame({"season": 2026, "team_id": TEAM, "athlete_id": [1, 2], "rgrp": "QB"})
    sched = _games(2026, 6)
    pg, _ = D.team_costs(u, sched, lab, rost, MU, None, True, [2026], groups=("QB",))
    tot, _ = D.team_costs(u, sched, lab, rost, MU, None, True, [2026], groups=("QB",), starter_by="total")
    pg, tot = pg.set_index("game_id"), tot.set_index("game_id")
    # game 5: QB 2 has more dropbacks so far (79 vs 60) -> the total rule makes him the starter
    assert pg.loc[202605, "qb_cost"] > 0 and tot.loc[202605, "qb_cost"] == 0.0
    # discount after 3 missed games of 5: 1 - 3 / (4 + 5)
    assert np.isclose(pg.loc[202605, "qb_cost_d"], pg.loc[202605, "qb_cost"] * (1 - 3 / 9))


def test_slot_labels_chart_only_when_touch_type_matches():
    u = _u(2025, {1: [(0, 0, 0.0, 12, 5.0)], 2: [(0, 0, 0.0, 12, 5.0)]})
    rost = pd.DataFrame({"season": 2025, "team_id": TEAM, "athlete_id": [1, 2], "rgrp": ["WR", "WR"]})
    chart = pd.DataFrame({"athlete_id": [1, 2], "chart_team": "X", "align": ["SLOT", "RB"], "chart_depth": 1})
    lab = D.slot_labels(u, rost, chart, pd.DataFrame(columns=["athlete_id", "align"])).set_index("athlete_id")
    assert lab.loc[1, "align"] == "SLOT"          # chart spot, same touch type
    assert lab.loc[2, "align"] == "WR"            # chart says RB but he had targets: keep the roster group


def test_adjusted_chart_puts_out_starter_back():
    chart = pd.DataFrame({
        "team": "T", "unit": "defense", "position": ["LCB", "LCB", "RCB", "RCB"], "pos_group": "DB",
        "depth": [1, 2, 1, 2], "athlete_id": [101, 102, 103, 104], "player": ["a", "b", "c", "d"],
        "role": "CB"})
    teams = pd.DataFrame({"season": [2026], "team": ["T"], "team_id": [TEAM]})
    exports = pd.DataFrame({"team_id": TEAM, "athlete_id": [101, 102, 103, 104, 105],
                            "starter": [1.0, 0.3, 0.6, 0.2, 0.9], "spot": "CB"})
    listed = pd.DataFrame({"team_id": [TEAM], "athlete_id": [105], "status": ["Out"], "name": ["e"]})
    adj = D.adjusted_chart(chart, listed, exports, pd.DataFrame(columns=["team_id", "athlete_id", "grp", "align"]),
                           teams)
    rcb = adj[adj["position"] == "RCB"].sort_values("depth")
    # 105 (share 0.9) is on the bench; he takes the corner spot whose starter has the lower share (RCB, 0.6)
    assert list(rcb["athlete_id"]) == [105, 103, 104]
    assert bool(rcb.iloc[0]["inserted"]) and rcb.iloc[0]["status"] == "Out"
    assert list(adj[adj["position"] == "LCB"].sort_values("depth")["athlete_id"]) == [101, 102]
