"""cfb.lab.availability: p_out overlay, streak rates, QB value over replacement."""
import numpy as np
import pandas as pd

from cfb.lab import availability as A


def _roles():
    # two seasons of one team: season 2025 sets the streak rates, 2026 is scored
    rows = []
    for season in (2025, 2026):
        for k in range(2, 6):
            for aid, grp, drops, epa, absent, streak in (
                    (1, "QB", 200, 40.0, 0.0 if season == 2025 else float(k >= 4), 0 if k < 5 else 1),
                    (2, "QB", 30, 0.0, 0.0, 0),
                    (3, "WR", 0, 0.0, float(k == 3), 0)):
                rows.append({"season": season, "game_id": season * 10 + k, "team_id": 7, "k": k, "athlete_id": aid,
                             "grp": grp, "gp": k, "team_gp": k, "since_first": k, "missed_frac": 0.0,
                             "streak": streak, "absent": absent, "sh_drop": 0.9 if aid == 1 else 0.1,
                             "sh_carry": 0.0, "sh_tgt": 0.3 if aid == 3 else 0.0, "sh_tkl": 0.0,
                             "n_drop": drops, "epa_drop": epa, "team_dpg": 30.0})
    return pd.DataFrame(rows)


def test_overlay_listing_beats_streak():
    ro = _roles()
    po = A.streak_p_out(ro)
    lst = pd.DataFrame([{"game_id": 20264, "team_id": 7, "athlete_id": 1, "status": "Out", "source": "hdi2026"}])
    po = A.overlay_listings(po, lst, {"Out": (0.99, 10)})
    hit = po[(po["game_id"] == 20264) & (po["athlete_id"] == 1)].iloc[0]
    assert hit["p_out"] == 0.99 and hit["src"] == "hdi2026"
    # a player with no streak and no listing is not moved
    assert po.loc[(po["game_id"] == 20262) & (po["athlete_id"] == 1), "p_out"].iloc[0] == 0.0


def test_qb_value_uses_available_backup():
    ro = _roles()
    po = A.streak_p_out(ro).assign(p_out=0.0)
    mu = {2025: (0.10, 0.05), 2026: (0.10, 0.05)}
    po.loc[(po["game_id"] == 20264) & (po["athlete_id"] == 1), "p_out"] = 1.0
    f = A.team_features(po, mu).set_index("game_id")
    es = (40.0 + A.K_QB * 0.10) / (200 + A.K_QB)
    eb = (0.0 + A.K_QB * 0.05) / (30 + A.K_QB)
    assert np.isclose(f.loc[20264, "qb_vor"], (es - eb) * 30.0)
    assert np.isclose(f.loc[20264, "qb_out"], f.loc[20264, "qb_vor"])
    assert f.loc[20263, "qb_out"] == 0.0
    # backup also out -> league backup mean replaces him
    po.loc[(po["game_id"] == 20264) & (po["athlete_id"] == 2), "p_out"] = 1.0
    f2 = A.team_features(po, mu).set_index("game_id")
    assert np.isclose(f2.loc[20264, "qb_vor"], (es - 0.05) * 30.0)
    assert not np.isclose(f2.loc[20264, "qb_vor"], f.loc[20264, "qb_vor"])
