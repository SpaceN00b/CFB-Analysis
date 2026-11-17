"""game_index, team_game and pbp."""
import numpy as np
import pandas as pd
import pytest

from cfb.core.build import build_core, check, summarize
from cfb.core.game_index import GAME_INDEX_COLUMNS, GAME_INDEX_EXTRA, compute_cutoff
from cfb.core.pbp import PBP_COLUMNS, build_pbp, classify, unit_sums
from cfb.core.team_game import KEY_COLUMNS, PBP_COLUMNS as TG_PBP, _clock, _pair, parse_box, snake
from cfb.core.timeutil import RESULT_LAG


def test_game_index_contract_columns(synthetic):
    gi = synthetic.game_index
    assert list(gi.columns) == GAME_INDEX_COLUMNS + GAME_INDEX_EXTRA
    assert gi["game_id"].is_unique
    assert str(gi["start_time_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(gi["cutoff_utc"].dtype) == "datetime64[ns, UTC]"
    assert (gi["cutoff_utc"] <= gi["start_time_utc"]).all()
    assert gi.loc[~gi["completed"], ["home_points", "away_points"]].isna().all().all()
    assert gi.loc[gi["completed"], ["home_points", "away_points"]].notna().all().all()


def test_cutoff_is_week_start_or_24h_before_a_bowl(synthetic):
    gi = synthetic.game_index
    weeks = synthetic.meta["weeks"].rename(columns={"start_utc": "week_start"})
    reg = gi[gi["season_type"] == "regular"].merge(weeks, on=["season", "season_type", "week"])
    assert (reg["cutoff_utc"] == reg["week_start"]).all()
    post = gi[gi["season_type"] == "postseason"]
    assert len(post) and (post["start_time_utc"] - post["cutoff_utc"] == pd.Timedelta(hours=24)).all()
    # every regular-season game of a week shares one cutoff; each bowl has its own
    assert reg.groupby(["season", "week"])["cutoff_utc"].nunique().eq(1).all()
    assert post["cutoff_utc"].is_unique


def test_cutoff_never_trails_kickoff():
    start = pd.Series(pd.to_datetime(["2025-09-06T16:00Z", "2025-09-06T16:00Z"], utc=True))
    week_start = pd.Series(pd.to_datetime(["2025-09-01T07:00Z", "2025-09-08T07:00Z"], utc=True))
    cutoff = compute_cutoff(start, pd.Series(["regular", "regular"]), week_start)
    assert cutoff[0] == week_start[0]
    assert cutoff[1] == start[1] - pd.Timedelta(hours=24)  # calendar week starts after kickoff


def test_team_game_two_rows_per_game(synthetic):
    gi, tg = synthetic.game_index, synthetic._read("team_game")
    assert list(tg.columns[:4]) == KEY_COLUMNS
    assert len(tg) == 2 * len(gi)
    assert not tg.duplicated(["game_id", "team_id"]).any()
    assert tg.groupby("game_id")["is_home"].sum().eq(1).all()
    home = tg[tg["is_home"]].set_index("game_id")
    g = gi.set_index("game_id")
    assert (home["team_id"] == g["home_id"].reindex(home.index)).all()
    assert (home["opp_id"] == g["away_id"].reindex(home.index)).all()
    played = home[g["completed"].reindex(home.index).to_numpy()]
    assert (played["points"] == g["home_points"].reindex(played.index)).all()
    assert ((played["margin"] > 0) == (played["win"] == 1)).all()
    unplayed = tg[tg["points"].isna()]
    assert len(unplayed) and unplayed["win"].isna().all() and unplayed[TG_PBP].isna().all().all()
    for family in ("box_", "adv_off_", "adv_def_", "ppa_off_", "pbp_off_", "pbp_def_"):
        assert any(c.startswith(family) for c in tg.columns), family


def test_team_game_offense_mirrors_opponent_defense(synthetic):
    tg = synthetic._read("team_game").dropna(subset=["points"])
    opp = tg.rename(columns={"team_id": "opp_id", "opp_id": "team_id"})
    m = tg.merge(opp, on=["game_id", "team_id", "opp_id"], suffixes=("", "_opp"))
    assert len(m) == len(tg)
    for stat in ("plays", "epa_sum", "successes", "expl_sum"):
        assert np.allclose(m[f"pbp_off_all_{stat}"], m[f"pbp_def_all_{stat}_opp"])
    assert np.allclose(m["pbp_off_all_plays"], m["pbp_off_rush_plays"] + m["pbp_off_pass_plays"])


def test_box_score_parsing():
    assert _pair("5-13") == (5.0, 13.0)
    assert _pair("3--5") == (3.0, -5.0)
    assert all(np.isnan(_pair("n/a")))
    assert _clock("33:41") == 33 * 60 + 41 and np.isnan(_clock("bad"))
    assert snake("yardsPerRushAttempt") == "yards_per_rush_attempt"
    frame = pd.DataFrame({"id": [1], "teams": ['[{"teamId": 7, "stats": [{"category": "thirdDownEff", "stat": "5-13"},'
                                                ' {"category": "totalYards", "stat": "401"},'
                                                ' {"category": "possessionTime", "stat": "30:15"}]}]']})
    box = parse_box(frame).iloc[0]
    assert (box["box_third_down_conv"], box["box_third_down_att"]) == (5.0, 13.0)
    assert box["box_total_yards"] == 401.0 and box["box_possession_seconds"] == 1815.0


def test_play_classification():
    types = pd.Series(["Rush", "Pass Incompletion", "Sack", "Punt", "Fumble Recovery (Opponent)",
                       "Fumble Recovery (Own)", "Field Goal Good", "Kickoff", "Penalty", "Timeout", "Interception"])
    text = pd.Series(["run for 3", "pass incomplete", "sacked", "punt", "QB sacked, fumbled", "run for 2, fumbled",
                      "FG good", "kickoff", "penalty", "timeout", "pass intercepted"])
    cls = classify(types, text)
    assert list(cls["play_class"]) == ["rush", "pass", "pass", "punt", "pass", "rush", "field_goal", "kickoff",
                                       "penalty", "other", "pass"]
    assert not (cls["is_rush"] & cls["is_pass"]).any()


def _plays(rows, game_id=1):
    base = {"gameId": game_id, "driveId": "d1", "driveNumber": 1, "offense": "A", "defense": "B", "home": "A",
            "away": "B", "period": 1, "yardsToGoal": 50, "scoring": False, "clock.minutes": 10, "clock.seconds": 0,
            "season": 2024, "season_type": "regular", "week": 1, "offenseScore": 0, "defenseScore": 0}
    return pd.DataFrame([{**base, "id": str(i), "playNumber": i, **row} for i, row in enumerate(rows)])


def _one_game_index():
    return pd.DataFrame({"game_id": [1], "home_id": [10], "away_id": [20], "home_team": ["A"], "away_team": ["B"],
                         "start_time_utc": pd.to_datetime(["2024-09-07T16:00Z"], utc=True)})


def test_pbp_success_passing_down_and_garbage_time():
    rows = [
        {"playType": "Rush", "down": 1, "distance": 10, "yardsGained": 5, "ppa": 0.2, "playText": "run"},  # 50% on 1st
        {"playType": "Rush", "down": 1, "distance": 10, "yardsGained": 4, "ppa": -0.1, "playText": "run"},
        {"playType": "Pass Reception", "down": 2, "distance": 10, "yardsGained": 7, "ppa": 0.5, "playText": "pass"},
        {"playType": "Pass Reception", "down": 3, "distance": 5, "yardsGained": 4, "ppa": -0.4, "playText": "pass"},
        {"playType": "Pass Interception Return", "down": 1, "distance": 10, "yardsGained": 30, "ppa": -3.0, "playText": "pass"},
        {"playType": "Rushing Touchdown", "down": 4, "distance": 9, "yardsGained": 2, "ppa": 3.0, "playText": "run"},
        {"playType": "Punt", "down": 4, "distance": 9, "yardsGained": 40, "ppa": None, "playText": "punt"},
        {"playType": "Rush", "down": 2, "distance": 3, "yardsGained": 1, "ppa": 0.0, "playText": "run",
         "period": 4, "offenseScore": 45, "defenseScore": 20},
    ]
    pbp = build_pbp(_plays(rows), _one_game_index())
    assert list(pbp.columns) == PBP_COLUMNS
    assert (pbp["offense_id"] == 10).all() and (pbp["defense_id"] == 20).all()
    assert pbp["success"].tolist()[:6] == [True, False, True, False, False, True]
    assert pd.isna(pbp["success"].iloc[6]) and not pbp["is_scrimmage"].iloc[6]
    assert pbp["passing_down"].tolist() == [False, False, True, True, False, True, True, False]
    assert pbp["turnover"].tolist()[4] and pbp["garbage_time"].tolist() == [False] * 7 + [True]
    sums = unit_sums(pbp).iloc[0]
    assert (sums["all_plays"], sums["rush_plays"], sums["pass_plays"]) == (7, 4, 3)
    assert sums["all_successes"] == 3 and sums["all_expl_sum"] == pytest.approx(0.2 + 0.5 + 3.0)
    assert sums["all_epa_sum"] == pytest.approx(0.2 - 0.1 + 0.5 - 0.4 - 3.0 + 3.0 + 0.0)


def test_pbp_drops_plays_of_unknown_games_and_teams():
    rows = [{"playType": "Rush", "down": 1, "distance": 10, "yardsGained": 5, "ppa": 0.2, "playText": "run"}]
    stray = _plays(rows, game_id=99)
    odd = _plays(rows).assign(offense="Someone Else")
    assert build_pbp(pd.concat([stray, odd]), _one_game_index()).empty


def test_build_core_is_reproducible_and_clean(synthetic):
    tables = build_core(synthetic.root, write=False)
    assert check(tables) == []
    assert summarize(tables)["team_game_rows"] == 2 * len(tables["game_index"])
    for name in ("game_index", "team_game", "pbp", "teams", "weeks", "game_pregame"):
        pd.testing.assert_frame_equal(tables[name], pd.read_parquet(synthetic.root / "core" / f"{name}.parquet"))


def test_synthetic_last_season_is_in_progress(synthetic):
    gi = synthetic.game_index
    last = gi[gi["season"] == gi["season"].max()]
    assert last["completed"].any() and (~last["completed"]).any()
    assert RESULT_LAG == pd.Timedelta(hours=5)


# ---- against the real snapshot (skipped in CI) ---------------------------------------

def test_snapshot_game_counts(snapshot):
    """Seasons through 2025 are final, so their counts are fixed."""
    gi = snapshot.game_index
    done = gi[(gi["season"] <= 2025) & gi["completed"]]
    assert len(done) == 10371
    assert int((done["home_fbs"] & done["away_fbs"]).sum()) == 9085
    assert gi["game_id"].is_unique and (gi["cutoff_utc"] <= gi["start_time_utc"]).all()


def test_snapshot_matches_raw_games(snapshot):
    from cfb.store import Store

    raw = Store(snapshot.root).load("games").drop_duplicates("id")
    gi = snapshot.game_index
    assert len(gi) == len(raw)
    assert int(gi["completed"].sum()) == int(raw["completed"].sum())
    fbs2 = (raw["homeClassification"] == "fbs") & (raw["awayClassification"] == "fbs") & raw["completed"]
    assert int((gi["completed"] & gi["home_fbs"] & gi["away_fbs"]).sum()) == int(fbs2.sum())
    tg = snapshot._read("team_game")
    assert len(tg) == 2 * len(gi)


def test_snapshot_pbp_agrees_with_cfbd_advanced_stats(snapshot):
    """Play-derived success rate and EPA track CFBD's own per-game numbers."""
    tg = snapshot._read("team_game")
    ok = tg["adv_off_plays"].notna() & (tg["pbp_off_all_plays"] > 20)
    d = tg[ok]
    assert np.corrcoef(d["adv_off_success_rate"], d["pbp_off_all_successes"] / d["pbp_off_all_plays"])[0, 1] > 0.98
    assert np.corrcoef(d["adv_off_ppa"], d["pbp_off_all_epa_sum"] / d["pbp_off_all_plays"])[0, 1] > 0.98
    assert (d["adv_off_plays"] - d["pbp_off_all_plays"]).abs().mean() < 1.5
