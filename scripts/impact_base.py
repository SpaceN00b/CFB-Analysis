"""model_v2 margins for the player-impact study (diagnostic; writes data/exploration/impact_base.parquet).

Season-forward 2018-2025 (fit 2015..S-1, score S) and 2026 walk-forward (fit 2015..2025, score
2026 completed games), with the frozen model_v2 columns. One row per scored game with the
margin, predicted margin, the calibrated linear term and the calibration slope.

    python scripts/impact_base.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_select  # noqa: E402

from cfb.model import config as C  # noqa: E402
from cfb.model.models import MarginModel  # noqa: E402

OUT = Path("data/exploration/impact_base.parquet")
SEASONS = list(range(2018, 2027))


def main():
    model_select.LAST_SEASON = 2026          # this study scores 2026 as a walk-forward hold-out
    X, _ = model_select.load()
    cols = list(C.MODEL_V2_COLUMNS)
    rows = []
    for s in SEASONS:
        tr = X[(X["season"] >= C.TRAIN_START) & (X["season"] < s) & X["y"].notna()]
        te = X[(X["season"] == s) & X["in_eval"] & X["y"].notna()]
        m = MarginModel(cols, lam=C.LAM).fit(tr, tr["y"].to_numpy(), tr["margin"].to_numpy())
        rows.append(pd.DataFrame({
            "game_id": te["game_id"].to_numpy(), "season": s, "season_type": te["season_type"].to_numpy(),
            "week": te["week"].to_numpy(), "home_id": te["home_id"].to_numpy(), "away_id": te["away_id"].to_numpy(),
            "margin": te["margin"].to_numpy(), "y": te["y"].to_numpy(), "pred": m.predict_margin(te),
            "lin": m.linear(te), "calib_s": m.calib[1]}))
        print(s, len(te), flush=True)
    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    from cfb.model.models import log_loss, sigmoid
    for lab, d in (("2018-2025", out[out.season <= 2025]), ("2026", out[out.season == 2026])):
        print(lab, len(d), round(log_loss(d["y"], sigmoid(d["lin"])), 4))


if __name__ == "__main__":
    main()
