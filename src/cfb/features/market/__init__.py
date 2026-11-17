"""Market features: betting lines for the game, home-oriented. Prefix `mkt_`.

Not an input of the primary model (approved default). The group exists for the
baselines in `cfb.eval` and for a market-aware variant after phase 1.

`lines` holds closing lines, which are set at kickoff, later than the week-start cutoff
the table is served at. Treat every column here as kickoff-time information.

Provider priority for each value: consensus, Bovada, DraftKings, ESPN Bet, then any other
provider in the order CFBD lists them. Each value comes from the first provider that has
it, except the moves and the moneyline pair, which come from the first provider that has
both of their parts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..base import FeatureColumn, FeatureGroup

PROVIDER_PRIORITY = ("consensus", "Bovada", "DraftKings", "ESPN Bet")
_RANK = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}

LINE_COLUMNS = [
    "game_id", "spread", "spread_open", "spread_move", "total", "total_open", "total_move",
    "ml_home", "ml_away", "p_home_ml", "overround", "n_providers", "spread_range",
]


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def implied_probability(ml) -> np.ndarray:
    """Win probability implied by American odds, vig included. Odds between -100 and 100 are invalid."""
    ml = np.asarray(ml, dtype="float64")
    out = np.full(ml.shape, np.nan)
    neg, pos = ml <= -100, ml >= 100
    out[neg] = -ml[neg] / (-ml[neg] + 100.0)
    out[pos] = 100.0 / (ml[pos] + 100.0)
    return out


def devig(ml_home, ml_away) -> tuple[np.ndarray, np.ndarray]:
    """(home win probability with the vig removed proportionally, overround)."""
    ph, pa = implied_probability(ml_home), implied_probability(ml_away)
    total = ph + pa
    return ph / total, total - 1.0


def _first(entries: list[dict], *keys: str) -> tuple[float, ...]:
    for e in entries:
        vals = tuple(_num(e.get(k)) for k in keys)
        if not any(np.isnan(v) for v in vals):
            return vals
    return tuple(np.nan for _ in keys)


def parse_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per game from the raw `lines` table (column `lines` holds JSON). Columns LINE_COLUMNS."""
    rows = []
    for gid, text in zip(lines["id"].to_numpy(), lines["lines"].to_numpy()):
        try:
            entries = json.loads(text) if isinstance(text, str) else []
        except json.JSONDecodeError:
            entries = []
        entries = [e for e in entries if isinstance(e, dict)]
        # stable sort: the four named providers first, the rest in CFBD's order
        entries.sort(key=lambda e: _RANK.get(e.get("provider"), len(_RANK)))
        spreads = [v for v in (_num(e.get("spread")) for e in entries) if not np.isnan(v)]
        (spread,) = _first(entries, "spread")
        (spread_open,) = _first(entries, "spreadOpen")
        sc, so = _first(entries, "spread", "spreadOpen")
        (total,) = _first(entries, "overUnder")
        (total_open,) = _first(entries, "overUnderOpen")
        tc, to = _first(entries, "overUnder", "overUnderOpen")
        mh, ma = _first(entries, "homeMoneyline", "awayMoneyline")
        rows.append((int(gid), spread, spread_open, sc - so, total, total_open, tc - to, mh, ma,
                     len(spreads), (max(spreads) - min(spreads)) if spreads else np.nan))
    out = pd.DataFrame(rows, columns=[
        "game_id", "spread", "spread_open", "spread_move", "total", "total_open", "total_move",
        "ml_home", "ml_away", "n_providers", "spread_range",
    ])
    out["game_id"] = out["game_id"].astype("int64")
    p, over = devig(out["ml_home"].to_numpy(), out["ml_away"].to_numpy())
    out["p_home_ml"], out["overround"] = p, over
    out["n_providers"] = out["n_providers"].astype("float64")
    out = out.drop_duplicates("game_id", keep="last")
    return out[LINE_COLUMNS].reset_index(drop=True)


def _col(name: str, description: str, first_season: int = 2014) -> FeatureColumn:
    return FeatureColumn(name, description, ("lines",), first_season)


class MarketLines(FeatureGroup):
    """Closing and opening lines for the game, from the home team's side."""

    name = "market_lines"
    prefix = "mkt_"
    level = "game"
    mode = "per_cutoff"
    columns = [
        _col("mkt_spread_close", "Closing spread for the home team (negative: home favoured). Set at kickoff"),
        _col("mkt_spread_open", "Opening spread for the home team", 2021),
        _col("mkt_spread_move", "Closing minus opening spread, same provider (negative: moved toward home)", 2021),
        _col("mkt_total_close", "Closing over/under"),
        _col("mkt_total_open", "Opening over/under", 2021),
        _col("mkt_total_move", "Closing minus opening over/under, same provider", 2021),
        _col("mkt_ml_home", "Closing home moneyline, American odds", 2021),
        _col("mkt_ml_away", "Closing away moneyline, American odds, same provider as mkt_ml_home", 2021),
        _col("mkt_p_home_ml", "Home win probability implied by the moneylines, vig removed proportionally", 2021),
        _col("mkt_overround", "Sum of both implied moneyline probabilities minus 1 (the vig)", 2021),
        _col("mkt_n_providers", "Number of providers with a closing spread"),
        _col("mkt_spread_range", "Highest minus lowest closing spread across providers"),
    ]

    def build(self, view, games: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in self.column_names})
        empty.insert(0, "game_id", pd.Series(dtype="int64"))
        if not view.has("lines"):
            return empty
        raw = view.table("lines", columns=["id", "lines"])
        raw = raw[raw["id"].isin(games["game_id"])]
        if raw.empty:
            return empty
        parsed = parse_lines(raw)
        out = parsed.rename(columns={
            "spread": "mkt_spread_close", "spread_open": "mkt_spread_open", "spread_move": "mkt_spread_move",
            "total": "mkt_total_close", "total_open": "mkt_total_open", "total_move": "mkt_total_move",
            "ml_home": "mkt_ml_home", "ml_away": "mkt_ml_away", "p_home_ml": "mkt_p_home_ml",
            "overround": "mkt_overround", "n_providers": "mkt_n_providers", "spread_range": "mkt_spread_range",
        })
        return out[["game_id"] + self.column_names]


GROUPS = [MarketLines()]

__all__ = ["GROUPS", "LINE_COLUMNS", "MarketLines", "PROVIDER_PRIORITY", "devig", "implied_probability",
           "parse_lines"]
