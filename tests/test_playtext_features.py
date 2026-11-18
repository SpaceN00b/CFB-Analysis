"""Feature groups from the detailed play text: str_pass_ci, matchup_run_dir, matchup_pass_zone."""
import numpy as np
import pandas as pd
import pytest

from cfb.features.base import assert_group_no_leak
from cfb.features.matchup.playtext import PassZone, RunDirection, _zone_profiles, pass_zone
from cfb.features.strength.playtext import AIR_BINS, PassAccuracy, fit_clean_inc, flag

GROUPS = [PassAccuracy(), RunDirection(), PassZone()]


def test_flag_handles_booleans_numbers_and_missing():
    assert flag(pd.Series([True, False, pd.NA], dtype="boolean")).tolist() == [True, False, False]
    assert flag(pd.Series([1.0, 0.0, np.nan])).tolist() == [True, False, False]
    assert flag(pd.Series([True, None, False], dtype="object")).tolist() == [True, False, False]


def test_zone_profiles_shrink_and_fit_sign():
    """An offense that runs left against a defense weak to the left gets a positive fit."""
    rows = []
    rng = np.random.default_rng(0)
    for off in range(1, 9):
        for _ in range(200):
            d = rng.choice(["left", "middle", "right"], p=[0.6, 0.2, 0.2] if off == 1 else [1 / 3] * 3)
            de = int(rng.integers(11, 19))
            p = 0.6 if (de == 11 and d == "left") else 0.4
            rows.append({"offense_id": off, "defense_id": de, "run_dir": d, "success": rng.random() < p})
    plays = pd.DataFrame(rows)
    prof = _zone_profiles(plays, plays["run_dir"])
    share = prof["off_share"]
    assert np.allclose(share.sum(axis=1), 1.0)
    assert share.loc[1, "left"] > 0.5 > share.loc[2, "left"]
    assert prof["def_adj"].loc[11, "left"] > 0.1
    fit_vs_11 = (share.loc[1].to_numpy() - prof["fbs_share"].to_numpy()) @ prof["def_adj"].loc[11].to_numpy()
    assert fit_vs_11 > 0


def test_pass_zone_labels():
    p = pd.DataFrame({"air_yards": [-2, 5, 12, 30, None], "pass_dir": ["left", "middle", "right", "left", "left"]})
    assert pass_zone(p).tolist()[:4] == ["0_left", "1_middle", "2_right", "3_left"]
    assert pd.isna(pass_zone(p).iloc[4])


def test_fit_clean_inc_recovers_a_bad_passer():
    rng = np.random.default_rng(2)
    rows = []
    for qb, bad in ((1, 1.0), (2, 0.0)):
        for rec in (11, 12, 13):
            for _ in range(300):
                air = int(rng.integers(-2, 30))
                eta = -1.6 + 0.04 * air + bad
                rows.append({"qb_id": qb, "rec_id": rec, "defense_id": int(rng.integers(100, 105)), "air": air,
                             "pass_dir": rng.choice(["left", "middle", "right"]), "down": 1,
                             "clean_inc": float(rng.random() < 1 / (1 + np.exp(-eta)))})
    t = pd.DataFrame(rows)
    t["air_bin"] = pd.cut(t["air"], AIR_BINS, labels=False, right=True)
    qb = fit_clean_inc(t)["qb"].set_index("id")["coef"]
    assert qb[1] - qb[2] > 0.6


def _detailed_cutoffs(wh):
    gi = wh.game_index
    picks = []
    for season, week, stype in ((2025, 12, "regular"), (2025, 1, "postseason"), (2026, 5, "regular"),
                                (2026, 10, "regular")):
        g = gi[(gi["season"] == season) & (gi["week"] == week) & (gi["season_type"] == stype)]
        if len(g):
            picks.append(g["cutoff_utc"].iloc[0])
    return picks


@pytest.mark.parametrize("group", GROUPS, ids=[g.name for g in GROUPS])
def test_no_leak_on_real_data(snapshot, group):
    if not snapshot.exists("play_detail"):
        pytest.skip("no play_detail table (run python -m cfb.core.playtext)")
    assert_group_no_leak(group, snapshot, cutoffs=_detailed_cutoffs(snapshot))
