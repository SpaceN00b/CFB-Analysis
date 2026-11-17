"""Quarterback group str_qb (P1): hand-checked values on synthetic data, leak tests."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.synthetic import synthetic_warehouse
from cfb.features.base import assert_group_no_leak, run_group, validate_group
from cfb.features.strength import qb as Q
from cfb.features.strength.qb import QBQuality

GOOD, BAD = 0.5, -0.1


@pytest.fixture(scope="module")
def qbwh(tmp_path_factory):
    """Each team has QBs <team>-1 (good) and <team>-2 (poor). Team 1 switches to its
    poor QB from 2022 week 3 (regular season)."""
    wh = synthetic_warehouse(tmp_path_factory.mktemp("qb"))
    gi = wh.game_index
    succ, ppa = [], []
    for g in gi[gi["completed"]].itertuples():
        for team, tid, opp in ((g.home_team, g.home_id, g.away_team), (g.away_team, g.away_id, g.home_team)):
            starter = 2 if tid == 1 and (g.season, g.week) >= (2022, 3) and g.season_type == "regular" else 1
            for k, plays in ((starter, 30), (3 - starter, 2)):
                key = {"season": g.season, "seasonType": g.season_type, "season_type": g.season_type,
                       "week": g.week, "id": f"{tid}-{k}", "team": team, "opponent": opp}
                succ.append({**key, "gameId": g.game_id, "position": "QB", "name": f"{team} {k}",
                             "passing.plays": plays, "passing.successes": plays // 2,
                             "rushing.plays": 0, "rushing.successes": 0})
                ppa.append({**key, "position": "QB", "name": f"{team} {k}",
                            "averagePPA.pass": GOOD if k == 1 else BAD, "averagePPA.rush": None})
    wh.put("player_success_games", pd.DataFrame(succ))
    wh.put("ppa_players_games", pd.DataFrame(ppa))
    teams = pd.concat([gi[["home_id", "home_team"]].set_axis(["id", "team"], axis=1),
                       gi[["away_id", "away_team"]].set_axis(["id", "team"], axis=1)]).drop_duplicates()
    roster = []
    for s in sorted(gi["season"].unique()):
        for t in teams.itertuples():
            for k in (1, 2):
                roster.append({"id": f"{t.id}-{k}", "team": t.team, "position": "QB", "year": k + 1,
                               "season": s, "recruitIds": None})
    wh.put("roster", pd.DataFrame(roster))
    wh.put("recruiting_players", pd.DataFrame({"id": ["r1"], "athleteId": ["1-1"], "rating": [0.95],
                                               "season": [int(gi["season"].min()) - 1]}))
    return wh


def _season(out, wh, team, season):
    sched = wh.schedule()[["game_id", "season", "season_type", "week"]]
    o = out.merge(sched, on="game_id")
    return o[(o["team_id"] == team) & (o["season"] == season) & (o["season_type"] == "regular")].sort_values("week")


def test_group_is_valid():
    validate_group(QBQuality())


def test_starter_change_shows_in_diff(qbwh):
    out = run_group(QBQuality(), qbwh)
    t1 = _season(out, qbwh, 1, 2022).set_index("week")
    # Week 1: no game yet this season, starter from the roster (most passes last season).
    assert t1.loc[1, "str_qb_from_roster"] == 1
    # Weeks 2 and 3: last game led by the good QB; the rating's QBs are the same.
    assert t1.loc[3, "str_qb_from_roster"] == 0
    assert abs(t1.loc[3, "str_qb_diff"]) < 0.05
    # Week 4: last game led by the poor QB, rating built mostly on the good one.
    assert t1.loc[4, "str_qb_diff"] < -0.1
    assert t1.loc[4, "str_qb_q"] < t1.loc[3, "str_qb_q"]
    assert t1.loc[4, "str_qb_share"] == pytest.approx((2 + 2 + 30) / (3 * 32))
    t2 = _season(out, qbwh, 2, 2022).set_index("week")
    assert (t2["str_qb_diff"].abs() < 0.05).all()


def test_quality_is_shrunk_towards_the_prior(qbwh):
    view = qbwh.view(qbwh.cutoffs()[-1])
    st, q = Q.qb_state(view)
    qi = q.set_index("athlete")
    good, bad = qi.loc["2-1"], qi.loc["2-2"]
    assert bad["q"] < good["q"]
    for row, mean in ((good, GOOD), (bad, BAD)):  # between the observed mean and the prior
        assert min(mean, row["prior"]) < row["q"] < max(mean, row["prior"])
    # The recruiting rating raises 1-1's prior above that of an unrated player.
    pri = Q.player_priors(view, view.season, ["1-1", "2-1"]).set_index("athlete")["prior"]
    assert pri["1-1"] > pri["2-1"]


def test_override_replaces_the_starter(qbwh):
    view = qbwh.view(qbwh.cutoffs()[-1])
    st, _ = Q.qb_state(view)
    alt, _ = Q.qb_state(view, starters={2: "2-2"})
    assert alt.at[2, "athlete"] == "2-2" and alt.at[2, "q"] < st.at[2, "q"]
    assert alt.at[3, "athlete"] == st.at[3, "athlete"]


def test_without_player_tables_values_are_empty(synthetic):
    out = run_group(QBQuality(), synthetic)
    assert len(out) and out[QBQuality().column_names].isna().all().all()


def test_no_leak_on_synthetic(qbwh):
    assert_group_no_leak(QBQuality(), qbwh, n=3)


def test_no_leak_on_real_data(snapshot):
    assert_group_no_leak(QBQuality(), snapshot, n=3)


# ---- manual QB-out override (cfb.model.qb_out) ------------------------------------------

def test_qb_out_adjustment_and_apply(qbwh):
    from cfb.model import qb_out as QO
    view = qbwh.view(qbwh.cutoffs()[-1])
    adj = QO.resolve(view, {"2": "out"})
    a = adj[2]
    assert a["starter"] == "2-1" and a["replacement"] == "2-2"
    assert QO.MIN_POINTS <= a["points"] < QO.A  # poorer replacement: below the intercept
    sched = qbwh.schedule()
    g = sched[(sched["home_id"] == 2) | (sched["away_id"] == 2)].iloc[[-1]]
    preds = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "p_home_win": [0.5], "pred_margin": [0.0],
                          "reasons": ["[]"]})
    out = QO.apply(preds, g, adj)
    home = int(g["home_id"].iat[0]) == 2
    assert (out["p_home_win"].iat[0] < 0.5) == home
    assert out["pred_margin"].iat[0] == pytest.approx(a["points"] if home else -a["points"])
    assert "QB out (manual)" in out["reasons"].iat[0]
    # Untouched when no adjusted team plays.
    assert QO.apply(preds, g, {}).equals(preds)


def test_qb_out_in_run_config(tmp_path):
    from cfb.report.config import load_config
    p = tmp_path / "config.toml"
    p.write_text('[models]\nprimary = "model_v1"\n[qb_out]\n"Ohio State" = "out"\nTexas = "Arch Manning"\n')
    cfg = load_config(p)
    assert cfg.qb_out == {"Ohio State": "out", "Texas": "Arch Manning"}
