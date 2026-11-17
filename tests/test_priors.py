"""P2 early-season priors: player components, the v2 groups and model_v2's configuration."""
import numpy as np
import pandas as pd
import pytest

from cfb.features.base import assert_group_no_leak, discover_groups, run_group
from cfb.features.strength import players as PL
from cfb.features.strength import prior as PR
from cfb.features.strength.groups_v2 import GROUPS_V2, Power2, Prior2, Program
from cfb.model import config as C


class StubView:
    """The few view calls players.components makes, over hand-made frames."""

    def __init__(self, frames, season):
        self.frames, self.season = frames, season

    def has(self, name):
        return name in self.frames

    def _get(self, name, columns=None, seasons=None):
        f = self.frames[name]
        if seasons is not None:
            f = f[f["season"].isin(np.atleast_1d(seasons))]
        return f[list(columns)] if columns is not None else f

    table = _get
    prior_seasons = _get


def _frames():
    teams = pd.DataFrame({"season": [2024, 2024], "team_id": [1, 2], "team": ["A", "B"]})
    roster = pd.DataFrame({"season": 2024, "id": ["10", "11", "12", "13"], "team": ["A", "A", "A", "B"]})
    # 10 stays at A, 11 transfers B -> A, 12 is a freshman, 13 stays at B; 14 left A.
    ppa = pd.DataFrame({"season": 2023, "id": ["10", "11", "13", "14"], "team": ["A", "B", "B", "A"],
                        "totalPPA.all": [100.0, 50.0, 30.0, 20.0]})
    gp = pd.DataFrame({
        "season": 2023, "team": ["A", "A", "B", "B", "A"],
        "category": ["defensive", "defensive", "defensive", "defensive", "interceptions"],
        "statType": ["TOT", "TOT", "TOT", "TOT", "INT"],
        "athleteId": [10, 14, 11, 13, 10], "stat": ["30", "10", "20", "20", "2"],
    })
    portal = pd.DataFrame({"season": 2024, "origin": ["B"], "destination": ["A"], "stars": [4.0],
                           "transferDate": ["2024-01-05T00:00:00Z"]})
    return {"teams": teams, "roster": roster, "ppa_players_season": ppa, "games_players": gp,
            "transfer_portal": portal}


def test_player_components_split_returning_incoming_and_lost():
    out = PL.components(StubView(_frames(), 2024), 2024, np.array([1, 2]))
    a, b = out.loc[1], out.loc[2]
    assert a["pl_ret_ppa"] == pytest.approx(1.0)     # player 10, per 100 PPA
    assert a["pl_in_ppa"] == pytest.approx(0.5)      # transfer 11 keeps his B production
    assert a["pl_lost_ppa"] == pytest.approx(0.2)    # player 14 left
    assert b["pl_ret_ppa"] == pytest.approx(0.3) and b["pl_lost_ppa"] == pytest.approx(0.5)
    assert a["pl_ret_def_tkl"] == pytest.approx(0.75)  # 30 of A's 40 tackles
    assert a["pl_in_def_tkl"] == pytest.approx(0.5)    # 20 of B's 40
    assert a["pl_ret_def_hav"] == pytest.approx(1.0)   # the only havoc play (an interception)
    assert a["pl_portal_in4"] == 1 and b["pl_portal_out4"] == 1 and a["pl_portal_out3"] == 0


def test_player_components_respect_asof_for_portal_rows():
    out = PL.components(StubView(_frames(), 2024), 2024, np.array([1, 2]), asof=pd.Timestamp("2024-01-01", tz="UTC"))
    assert out.loc[1, "pl_portal_in4"] == 0


def test_player_components_are_unknown_without_tables():
    f = {"teams": _frames()["teams"]}
    out = PL.components(StubView(f, 2024), 2024, np.array([1, 2]))
    assert out[["pl_ret_ppa", "pl_in_def_tkl"]].isna().all().all()
    assert (out["pl_portal_in3"].isna()).all()  # a portal season without portal data stays unknown
    old = PL.components(StubView({"teams": f["teams"].assign(season=2019)}, 2019), 2019, np.array([1, 2]))
    assert (old["pl_portal_in3"] == 0).all()     # before the portal: no transfers counted


def test_v2_projection_has_a_coefficient_per_component():
    assert set(PR.COEF_OFF_V2) == set(PR.COEF_DEF_V2) == {"one", *PR.COMPONENTS_V2}
    comp = pd.DataFrame(0.0, index=[1, 2], columns=PR.COMPONENTS_V2 + ["pool_off", "pool_def"])
    comp.loc[1, "pl_ret_ppa_c"] = 1.0
    off, _ = PR.projection(comp, version=2)
    assert off[0] - off[1] == pytest.approx(PR.COEF_OFF_V2["pl_ret_ppa_c"])


def test_v2_groups_are_registered_and_model_v2_columns_exist():
    groups = {g.name: g for g in discover_groups()}
    assert {"str_power2", "str_prior2", "str_program"} <= set(groups)
    cols = {f"d_{c}" for n in C.MODEL_V2_GROUPS for c in groups[n].column_names} | {"site", "fcs_game"}
    assert set(C.MODEL_V2_COLUMNS) <= cols
    assert C.MODEL_ID == "model_v1" and C.MODEL_GROUPS == ["str_power", "str_elo"]  # v1 untouched


def test_v2_groups_run_and_do_not_leak(synthetic):
    for g in (Power2(), Prior2()):
        out = run_group(g, synthetic)
        assert len(out)
        assert_group_no_leak(g, synthetic, n=2)
    p = run_group(Power2(), synthetic)
    assert p["str_pwr2_net"].notna().any()


def test_program_residual_is_point_in_time(synthetic):
    out = run_group(Program(), synthetic)
    assert np.isfinite(out["str_prog_resid"]).all() and (out["str_prog_games"] >= 0).all()
    first = synthetic.cutoffs()[0]
    sched = synthetic.schedule()
    first_ids = sched.loc[sched["cutoff_utc"] == first, "game_id"]
    assert (out[out["game_id"].isin(first_ids)]["str_prog_games"] == 0).all()
    assert_group_no_leak(Program(), synthetic, n=2)


def test_snapshot_v2_groups_do_not_leak(snapshot):
    cutoffs = [c for c in snapshot.cutoffs() if c.year == 2024][6:7] + [c for c in snapshot.cutoffs() if c.year == 2025][-1:]
    for group in GROUPS_V2:
        assert_group_no_leak(group, snapshot, cutoffs)
