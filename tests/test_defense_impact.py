"""cfb.lab.defense_impact: disruption points, regulars, value over the fill-in / next man up."""
import numpy as np
import pandas as pd

from cfb.lab import defense_impact as F

TEAM = 7
EV = {"sack": 1.5, "tfl": 1.0, "int": 5.0, "pbu": 1.0, "hurry": 0.5}
MU = {g: {"start": 1.0, "fill": 0.5} for g in F.DGROUPS}
K = {g: 4.0 for g in F.DGROUPS}


def _games(season=2026, n=5):
    t0 = pd.Timestamp(f"{season}-09-01", tz="UTC")
    return pd.DataFrame({"season": season, "game_id": [season * 100 + k for k in range(n)], "team_id": TEAM,
                         "start_time_utc": [t0 + pd.Timedelta(days=7 * k) for k in range(n)]})


def _dg(rows, season=2026):
    """rows: {athlete: (grp, [(game k, tot, sacks), ...])}"""
    out = []
    for a, (g, gl) in rows.items():
        for k, tk, sk in gl:
            out.append({"season": season, "game_id": season * 100 + k, "team_id": TEAM, "athlete_id": a, "grp": g,
                        "start_time_utc": pd.NaT, "tot": float(tk), "sacks": float(sk), "tfl": float(sk), "pd": 0.0,
                        "hur": 0.0, "int": 0.0})
    d = pd.DataFrame(out)
    d["inv"] = d["tot"] + d["pd"] + d["hur"] + d["int"]
    d["dv"] = F.disruption(d, EV)
    return d


def test_disruption_counts_sacks_once():
    d = pd.DataFrame({"sacks": [1.0], "tfl": [2.0], "int": [1.0], "pd": [1.0], "hur": [2.0]})
    # 1 sack (1.5) + 1 non-sack TFL (1.0) + INT 5 + PBU 1 + 2 hurries (1.0)
    assert np.isclose(F.disruption(d, EV).iloc[0], 9.5)


def _setup():
    # DL 1 regular with a sack per game in games 0-3, out in game 4; DL 5 is the backup
    rows = {1: ("DL", [(k, 4, 1) for k in range(4)]),
            2: ("DL", [(k, 3, 0) for k in range(5)]), 3: ("DL", [(k, 3, 0) for k in range(5)]),
            4: ("DL", [(k, 3, 0) for k in range(5)]), 5: ("DL", [(k, 1, 0) for k in range(5)])}
    return _dg(rows), _games()


def test_regular_out_costs_value_over_fill():
    dg, sched = _setup()
    tf, det = F.team_costs(dg, sched, MU, K, None, True, [2026])
    last = tf[tf["game_id"] == 202604].iloc[0]
    # rate = (4 games x 1.5 + K x 1.0) / (4 + K) = 1.25; fill 0.5 -> 0.75
    assert np.isclose(last["dl_cost"], 0.75)
    d = det[det["game_id"] == 202604]
    assert list(d["athlete_id"]) == [1] and d["repl_id"].iloc[0] == 5
    # own: backup 0 disruption -> (0 + K x 0.5) / (4 + K) = 0.25 -> 1.0
    assert np.isclose(last["dl_cost_own"], 1.0)
    assert tf.loc[tf["game_id"] < 202604, "dl_cost"].eq(0).all()


def test_listed_and_chart_next_man_up():
    dg, sched = _setup()
    dg = dg[dg["game_id"] < 202604]
    sched = sched.copy()
    p_out = {(202604, TEAM, 1): 1.0}
    chart = {TEAM: {"DE": {"role": "EDGE", "ids": [1, 9]}, "DT": {"role": "IDL", "ids": [2, 3, 5]}}}
    tf, det = F.team_costs(dg, sched, MU, K, p_out, False, [2026], chart_repl=chart)
    d = det[det["game_id"] == 202604].iloc[0]
    # next man up at his label is 9, who has not played: fill rate
    assert d["repl_id"] == 9 and np.isclose(d["repl_rate"], 0.5)
    assert np.isclose(tf.loc[tf["game_id"] == 202604, "dl_cost_own"].iloc[0], 0.75)


def test_streak_rule_reads_isolated_blank_as_played():
    rows = {1: ("DB", [(k, 6, 0) for k in (0, 1, 3, 4)])}
    rows.update({a: ("DB", [(k, 3, 0) for k in range(5)]) for a in (2, 3, 4, 5, 6)})
    dg = _dg(rows)
    tf, _ = F.team_costs(dg, _games(), MU, K, None, True, [2026], streak=False)
    tfs, _ = F.team_costs(dg, _games(), MU, K, None, True, [2026], streak=True)
    assert tf.loc[tf["game_id"] == 202602, "db_n_out"].iloc[0] == 1.0
    assert tfs.loc[tfs["game_id"] == 202602, "db_n_out"].iloc[0] == 0.0
