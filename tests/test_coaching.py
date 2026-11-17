"""Coaching features (W5): coach-to-game map, ATS, style metrics, leak tests."""
import json

import numpy as np
import pandas as pd
import pytest

from cfb.core import Warehouse
from cfb.features.base import assert_group_no_leak, run_group
from cfb.features.coaching import GROUPS, CoachStyle, CoachTenure
from cfb.features.coaching import metrics as M
from cfb.features.coaching.mapping import coach_game_map, tenures

INTERIM_HIRE = "2022-09-13"  # between team 1's week 2 and week 3 games of the synthetic 2022 season


def _tenure(tid, coach, team, start, end, hire, interim=False):
    return {"id": tid, "hireDate": hire, "startYear": start, "endYear": end, "effectiveStart": None,
            "effectiveEnd": None, "isInterim": interim, "active": end is None, "seasons": 1,
            "attributionComplete": True, "coach.id": coach, "coach.firstName": "C", "coach.lastName": str(coach),
            "team.id": team, "team.school": f"Team {team}", "record.games": 99, "record.wins": 99,
            "record.losses": 0, "record.ties": 0, "record.winPercentage": 1.0}


def _coaching_warehouse(synthetic) -> Warehouse:
    """The synthetic warehouse plus coach tenures with two changes:
    team 1 replaces coach 501 by interim 900 during 2022; team 2 hires coach 902 for 2023."""
    rows = []
    for team in range(1, 9):
        end = 2022 if team in (1, 2) else None
        rows.append(_tenure(100 + team, 500 + team, team, 2019, end, "2018-12-01"))
    rows.append(_tenure(200, 900, 1, 2022, None, INTERIM_HIRE, interim=False))
    rows.append(_tenure(201, 902, 2, 2023, None, "2022-12-05"))
    tenure = pd.DataFrame(rows)
    tenure["endYear"] = tenure["endYear"].astype("object")

    # Season game counts consistent with the tenures, as CFBD's /coaches has them.
    played = synthetic.view(synthetic.cutoffs()[-1]).table("team_game", columns=["team_id", "season", "start_time_utc"])
    coach_rows = []
    for season in (2021, 2022):
        for team in range(1, 9):
            n = int(((played["team_id"] == team) & (played["season"] == season)).sum())
            if season == 2022 and team == 1:
                g = played[(played["team_id"] == 1) & (played["season"] == 2022)]
                before = int((g["start_time_utc"] < pd.Timestamp(INTERIM_HIRE, tz="UTC")).sum())
                parts = [(501, before), (900, n - before)]
            else:
                parts = [(500 + team, n)]
            for coach, games in parts:
                coach_rows.append({"id": coach, "firstName": "C", "lastName": str(coach), "hireDate": "2018-12-01",
                                   "seasons": json.dumps([{"teamId": team, "year": season, "games": games}]),
                                   "season": season})
    wh = Warehouse(synthetic.root)
    wh.put("coach_tenures", tenure)
    wh.put("coaches", pd.DataFrame(coach_rows))
    return wh


@pytest.fixture(scope="module")
def coaching(synthetic):
    return _coaching_warehouse(synthetic)


def _team1_2022(wh):
    s = wh.schedule([2022])
    return s[((s["home_id"] == 1) | (s["away_id"] == 1))].reset_index(drop=True)


# ---- coach-to-game map -----------------------------------------------------------------

def test_map_in_season_change_by_date(coaching):
    g = _team1_2022(coaching)
    wk4 = g[(g["season_type"] == "regular") & (g["week"] == 4)].iloc[0]
    view = coaching.view(wk4["cutoff_utc"])
    m = coach_game_map(view, g[g["game_id"] == wk4["game_id"]])
    m = m[(m["team_id"] == 1) & (m["season"] == 2022)].set_index("game_id")
    by_week = g.set_index("game_id")["week"]
    assert m.loc[g.loc[g["week"] == 2, "game_id"].iloc[0], "coach_id"] == 501
    assert m.loc[g.loc[(g["week"] == 3) & (g["season_type"] == "regular"), "game_id"].iloc[0], "coach_id"] == 900
    row = m.loc[wk4["game_id"]]
    assert row["coach_id"] == 900 and row["interim"] == 1.0 and row["tenure_seasons"] == 0.0
    assert set(by_week.index) >= set(m.index)


def test_map_hides_a_coach_before_the_hire(coaching):
    g = _team1_2022(coaching)
    wk3 = g[(g["season_type"] == "regular") & (g["week"] == 3)].iloc[0]
    view = coaching.view(wk3["cutoff_utc"])  # cutoff before the interim hire date
    assert 900 not in set(tenures(view)["coach_id"])
    m = coach_game_map(view, g[g["game_id"] == wk3["game_id"]])
    assert m[(m["game_id"] == wk3["game_id"]) & (m["team_id"] == 1)]["coach_id"].iloc[0] == 501


def test_map_new_season_and_counts(coaching):
    s = coaching.schedule([2023])
    first = s[s["week"] == 2].reset_index(drop=True)
    view = coaching.view(first["cutoff_utc"].iloc[0])
    m = coach_game_map(view, first)
    now = m[m["game_id"].isin(first["game_id"])].set_index("team_id")
    if 2 in now.index:
        assert now.loc[2, "coach_id"] == 902 and now.loc[2, "tenure_seasons"] == 1.0 and now.loc[2, "interim"] == 0.0
    if 3 in now.index:
        assert now.loc[3, "coach_id"] == 503 and now.loc[3, "tenure_seasons"] == 5.0
    # 2022 is finished: its games follow the /coaches counts (501 then 900).
    past = m[(m["team_id"] == 1) & (m["season"] == 2022)].sort_values("start_time_utc")
    played = set(view.table("team_game", columns=["game_id"])["game_id"])
    past = past[past["game_id"].isin(played)]
    assert list(past["coach_id"].iloc[:2]) == [501, 501] and set(past["coach_id"].iloc[2:]) == {900}
    assert set(past["interim"].iloc[2:]) == {1.0}
    # FCS teams have no tenure: unknown, not zero.
    fcs = m[m["team_id"] > 100]
    assert fcs["coach_id"].isna().all() and fcs["interim"].isna().all()


def test_tenure_first_full_season_joins_back_to_back_rows(coaching):
    rows = pd.DataFrame([
        _tenure(1, 7, 3, 2015, 2015, "2015-10-10", interim=True),  # in-season debut
        _tenure(2, 7, 3, 2016, None, "2015-12-01"),                 # then permanent
        _tenure(3, 8, 4, 2010, 2012, None),
        _tenure(4, 8, 4, 2016, None, "2015-12-03"),                 # gap: a new spell
    ])
    rows["endYear"] = rows["endYear"].astype("object")
    wh = Warehouse(coaching.root)
    wh.put("coach_tenures", rows)
    t = tenures(wh.view(coaching.cutoffs()[-1])).set_index("tenure_id")
    assert t.loc[1, "in_season"] and t.loc[1, "first_full"] == 2016 and t.loc[2, "first_full"] == 2016
    assert t.loc[4, "first_full"] == 2016 and t.loc[3, "first_full"] == 2010


# ---- lines and ATS -----------------------------------------------------------------------

def test_closing_spreads_provider_priority():
    lines = pd.DataFrame({"id": [1, 2, 3], "lines": [
        json.dumps([{"provider": "Bovada", "spread": -3.0}, {"provider": "consensus", "spread": -2.5}]),
        json.dumps([{"provider": "teamrankings", "spread": 7.0}, {"provider": "DraftKings", "spread": None},
                    {"provider": "ESPN Bet", "spread": 6.5}]),
        json.dumps([{"provider": "numberfire", "spread": 1.0}]),
    ]})
    out = M.closing_spreads(lines).set_index("game_id")["spread"]
    assert out.to_dict() == {1: -2.5, 2: 6.5}


def test_ats_results_sign_and_pushes():
    hist = pd.DataFrame({"game_id": [1, 1, 2, 2, 3], "team_id": [10, 20, 10, 20, 10],
                         "is_home": [True, False, True, False, True], "margin": [7.0, -7.0, 3.0, -3.0, 1.0]})
    spreads = pd.DataFrame({"game_id": [1, 2, 3], "spread": [-10.0, -3.0, 2.0]})
    out = M.ats_results(hist, spreads).set_index(["game_id", "team_id"])["cover"]
    # Game 1: home favoured by 10, won by 7 -> away covers. Game 2: push. Game 3: home +2 and won.
    assert out.to_dict() == {(1, 10): 0.0, (1, 20): 1.0, (3, 10): 1.0}


def test_coach_ats_matches_a_hand_count(coaching):
    s = coaching.schedule([2023])
    slate = s[s["week"] == 3].reset_index(drop=True)
    view = coaching.view(slate["cutoff_utc"].iloc[0])
    out = CoachTenure().build(view, slate).set_index("team_id")
    team = 3 if 3 in out.index else out.index[out.index < 100][0]
    coach = 500 + team
    m = coach_game_map(view)
    tg = view.table("team_game", columns=["game_id", "team_id", "is_home", "margin"])
    hist = tg.merge(m, on=["game_id", "team_id"])
    hist = hist[hist["coach_id"] == coach]
    spreads = M.closing_spreads(view.table("lines", columns=["id", "lines"]))
    covers = M.ats_results(hist, spreads)["cover"]
    expect = 0.5 + (covers - 0.5).sum() / (len(covers) + M.ATS_PRIOR_GAMES)
    assert out.loc[team, "coach_ats_career"] == pytest.approx(expect)
    assert out.loc[team, "coach_ats_career_n"] == len(covers) and out.loc[team, "coach_games_career"] == len(hist)
    assert out.loc[team, "coach_first_year"] == 0.0


# ---- style metrics -----------------------------------------------------------------------

def test_expectation_uses_own_and_previous_season_only():
    rng = np.random.default_rng(1)
    n = 4000
    frame = pd.DataFrame({"game_id": np.arange(n), "season": np.repeat([2018, 2019, 2020, 2021], n // 4),
                          "offense_id": 1, "coarse": 1, "cell": rng.integers(0, 3, n)})
    frame["y"] = (frame["season"] >= 2020).astype(float)  # run rate jumps from 0 to 1 in 2020
    out = M.add_expectation(frame)
    e = out.groupby("season")["exp"].mean()
    assert e[2019] == pytest.approx(0.0) and e[2021] == pytest.approx(1.0)
    assert 0.3 < e[2020] < 0.7  # 2019 and 2020 pooled
    assert out["resid"].abs().max() <= 1.0


def test_shrunk_rate_weights_and_prior():
    plays = pd.DataFrame({"coach_id": [1, 1, 2], "season": [2020, 2021, 2021], "resid": [1.0, 1.0, -1.0]})
    out = M.shrunk_rate(plays, ["coach_id"], 2021, 1.0, 1.0).set_index("coach_id")
    assert out.loc[1, "value"] == pytest.approx(1.5 / 2.5) and out.loc[1, "n"] == 2
    assert out.loc[2, "value"] == pytest.approx(-0.5)


def test_flex_slope_recovers_direction():
    rng = np.random.default_rng(3)
    rows = []
    for coach, slope in ((1, 0.4), (2, -0.4)):
        for season in (2020, 2021):
            for _ in range(60):
                x = rng.normal(0, 0.2)
                rows.append({"coach_id": coach, "season": season, "n": 200, "x": x,
                             "r": 0.05 * (coach == 1) + slope * x + rng.normal(0, 0.03)})
    slopes, pooled = M.flex_slopes(pd.DataFrame(rows))
    s = slopes.set_index("coach_id")
    assert abs(pooled) < 0.1
    assert s.loc[1, "raw_slope"] > 0.3 and s.loc[2, "raw_slope"] < -0.3
    assert s.loc[1, "slope"] > pooled > s.loc[2, "slope"]  # shrunk, but on the right side
    assert s.loc[1, "games"] == 120


def test_defensive_gap_is_centred_per_cutoff():
    t = pd.Timestamp("2022-09-01", tz="UTC")
    ur = pd.DataFrame({"team_id": [1, 1, 2, 2, 3, 3], "cutoff_utc": t, "unit": ["def_rush", "def_pass"] * 3,
                       "epa_adj": [0.2, 0.0, 0.0, 0.2, 0.1, 0.1], "n_plays": [100, 100, 100, 100, 10, 100]})
    gaps = M.defensive_gaps(ur).set_index("team_id")["gap"]
    assert gaps.to_dict() == pytest.approx({1: 0.2, 2: -0.2})  # team 3 has too few rush plays


# ---- groups --------------------------------------------------------------------------------

def test_groups_registered():
    assert [g.name for g in GROUPS] == ["coach_tenure", "coach_style"]


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_group_runs_and_does_not_leak_on_synthetic(coaching, group):
    out = run_group(group, coaching, coaching.schedule([2022, 2023]))
    fbs = out[out["team_id"] < 100]
    always = [c for c in group.column_names if c != "coach_flex_shift"]  # needs a rated opponent
    assert fbs[always].notna().all().all()
    assert out[out["team_id"] > 100][group.column_names].isna().all().all()
    cutoffs = [c for c in coaching.cutoffs([2022, 2023])][2::4]
    assert_group_no_leak(group, coaching, cutoffs=cutoffs)


def test_style_values_are_sane_on_synthetic(coaching):
    s = coaching.schedule([2023])
    slate = s[s["week"] == 3].reset_index(drop=True)
    out = CoachStyle().build(coaching.view(slate["cutoff_utc"].iloc[0]), slate)
    fbs = out[out["team_id"] < 100]
    assert (fbs["coach_run_n"] > 0).all() and fbs["coach_run_oe"].abs().max() < 0.5
    assert (fbs["coach_flex_games"] >= 0).all()


# ---- real data (skipped in CI) -----------------------------------------------------------------

SNAPSHOT_CUTOFFS = [
    "2016-10-31T07:00:00Z",  # in-season, older data
    "2022-11-07T08:00:00Z",  # after several 2022 interim hires
    "2026-11-03T08:00:00Z",  # current season
]


def _nearest_cutoffs(wh):
    cutoffs = pd.Series(wh.cutoffs())
    picks = []
    for c in SNAPSHOT_CUTOFFS:
        t = pd.Timestamp(c)
        picks.append(cutoffs.iloc[(cutoffs - t).abs().argmin()])
    post = wh.schedule([2025])
    picks.append(post[post["season_type"] == "postseason"]["cutoff_utc"].sort_values().iloc[5])
    return sorted(set(picks))


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g.name)
def test_snapshot_leak(snapshot, group):
    assert_group_no_leak(group, snapshot, cutoffs=_nearest_cutoffs(snapshot))


def test_snapshot_values(snapshot):
    games = snapshot.schedule([2025])
    games = games[(games["season_type"] == "regular") & (games["week"] == 10)]
    tenure = run_group(CoachTenure(), snapshot, games)
    style = run_group(CoachStyle(), snapshot, games)
    both = tenure.merge(style, on=["game_id", "team_id"])
    fbs = pd.concat([games[games["home_fbs"]][["game_id", "home_id"]].rename(columns={"home_id": "team_id"}),
                     games[games["away_fbs"]][["game_id", "away_id"]].rename(columns={"away_id": "team_id"})])
    f = both.merge(fbs, on=["game_id", "team_id"])
    assert f["coach_tenure_seasons"].notna().all()
    assert f["coach_ats_career"].between(0.4, 0.6).all()
    assert f["coach_fourth_go_oe"].between(-0.2, 0.2).all() and f["coach_run_oe"].between(-0.25, 0.25).all()
    assert (f["coach_run_n"] > 0).mean() > 0.9
