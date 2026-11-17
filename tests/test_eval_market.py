"""Market feature group (cfb.features.market)."""
import json

import numpy as np
import pandas as pd
import pytest

from cfb.features.base import assert_group_no_leak, run_group
from cfb.features.market import MarketLines, devig, implied_probability, parse_lines


def _lines(entries_by_game: dict[int, list[dict]]) -> pd.DataFrame:
    return pd.DataFrame({"id": list(entries_by_game), "lines": [json.dumps(v) for v in entries_by_game.values()]})


def test_implied_probability_and_devig():
    assert implied_probability([-200, 150, 50, np.nan]) == pytest.approx([2 / 3, 0.4, np.nan, np.nan], nan_ok=True)
    p, over = devig([-110], [-110])
    assert p[0] == pytest.approx(0.5) and over[0] == pytest.approx(2 * 110 / 210 - 1)


def test_parse_lines_follows_provider_priority():
    frame = _lines({
        1: [
            {"provider": "teamrankings", "spread": -3.0, "spreadOpen": None},
            {"provider": "DraftKings", "spread": -6.5, "spreadOpen": -5.5, "overUnder": 50, "overUnderOpen": 48,
             "homeMoneyline": -250, "awayMoneyline": 200},
            {"provider": "consensus", "spread": -7.0, "spreadOpen": None, "overUnder": 51, "overUnderOpen": None,
             "homeMoneyline": None, "awayMoneyline": None},
            {"provider": "Bovada", "spread": -7.5, "spreadOpen": -4.0, "overUnder": None, "overUnderOpen": 47,
             "homeMoneyline": -300, "awayMoneyline": None},
        ],
        2: [],
        3: [{"provider": "numberfire", "spread": 3.5}],
    })
    out = parse_lines(frame).set_index("game_id")
    one = out.loc[1]
    assert one["spread"] == -7.0  # consensus first
    assert one["spread_open"] == -4.0  # first provider with an open: Bovada
    assert one["spread_move"] == pytest.approx(-3.5)  # Bovada has both close and open
    assert one["total"] == 51 and one["total_open"] == 47 and one["total_move"] == 2  # DraftKings has both
    assert (one["ml_home"], one["ml_away"]) == (-250, 200)  # Bovada lacks the away line
    assert one["p_home_ml"] == pytest.approx((250 / 350) / (250 / 350 + 100 / 300))
    assert one["n_providers"] == 4 and one["spread_range"] == pytest.approx(4.5)
    assert out.loc[2].drop("n_providers").isna().all() and out.loc[2, "n_providers"] == 0
    assert out.loc[3, "spread"] == 3.5  # another provider when none of the four has a spread


def test_market_group_on_synthetic(synthetic):
    out = run_group(MarketLines(), synthetic)
    assert len(out) == len(synthetic.schedule())
    assert out["mkt_spread_close"].notna().mean() > 0.9
    assert out["mkt_p_home_ml"].dropna().between(0, 1).all()
    assert_group_no_leak(MarketLines(), synthetic)


def test_market_group_without_lines_is_empty(synthetic):
    class NoLines:
        def has(self, name):
            return False

    games = synthetic.schedule().head(3)
    out = MarketLines().build(NoLines(), games)
    assert out.empty and list(out.columns) == ["game_id"] + MarketLines().column_names


def test_market_group_does_not_leak_on_real_data(snapshot):
    assert_group_no_leak(MarketLines(), snapshot, n=3)


def test_market_group_on_real_data(snapshot):
    games = snapshot.schedule([2025])
    out = run_group(MarketLines(), snapshot, games)
    fbs = games[games["home_fbs"] & games["away_fbs"]]["game_id"]
    have = out[out["game_id"].isin(fbs)]
    assert have["mkt_spread_close"].notna().mean() > 0.95
    assert have["mkt_p_home_ml"].notna().mean() > 0.8
