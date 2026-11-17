"""The four baselines a model has to beat, written as prediction files.

    baseline_close      closing spread -> probability: logistic without intercept on minus the
                        home spread, fitted on earlier seasons
    baseline_moneyline  closing moneylines with the vig removed proportionally (2021 on); no fit
    baseline_elo        CFBD pregame Elo (`game_pregame`) plus home field: logistic without
                        intercept on [home, Elo gap / 100], fitted on earlier seasons
    baseline_home       logistic without intercept on the home indicator: one probability for
                        every non-neutral game, 0.5 at neutral sites

Fitting (season-forward, refit="season" in the walk-forward runner): completed FBS vs FBS
games of earlier seasons that have the baseline's inputs. Predictions: every game of the
season with the inputs, FBS vs FCS included, except that baseline_elo leaves out FBS vs FCS
games from 2022 on, where CFBD's pregame Elo is not a pregame value (docs/data-coverage.md).

Closing lines are set at kickoff, so the two market baselines carry `cutoff_utc` = kickoff:
they are a benchmark, not something the model could have known at the week-start cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..core.asof import Warehouse
from ..features.market import MarketLines
from .splits import DEFAULT_SEASONS, TEST_SEASON, walk_forward

BASELINES = ("baseline_close", "baseline_moneyline", "baseline_elo", "baseline_home")
DEFAULT_BASELINE_SEASONS = tuple(DEFAULT_SEASONS) + (TEST_SEASON,)
ELO_FCS_UNRELIABLE_FROM = 2022
_MARKET = MarketLines()


def logit_fit(y, X, iterations: int = 50, ridge: float = 1e-8) -> np.ndarray:
    """Logistic regression by Newton's method, no intercept unless X has a column of ones."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    b = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        w = p * (1 - p)
        step = np.linalg.solve((X.T * w) @ X + ridge * np.eye(X.shape[1]), X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ---- inputs, read through the view ----------------------------------------------------------

def _home(games: pd.DataFrame) -> pd.Series:
    return (~games["neutral"].astype(bool)).astype("float64")


def inputs_home(view, games: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"game_id": games["game_id"].to_numpy(), "home": _home(games).to_numpy()})


def inputs_elo(view, games: pd.DataFrame) -> pd.DataFrame:
    elo = view.table("game_pregame")
    g = games[["game_id", "season", "neutral", "home_fbs", "away_fbs"]].merge(elo, on="game_id", how="inner")
    fcs_bad = ~(g["home_fbs"].astype(bool) & g["away_fbs"].astype(bool)) & (g["season"] >= ELO_FCS_UNRELIABLE_FROM)
    g = g[~fcs_bad]
    return pd.DataFrame({
        "game_id": g["game_id"].to_numpy(), "home": _home(g).to_numpy(),
        "elo_gap": ((g["home_pregame_elo"] - g["away_pregame_elo"]) / 100.0).to_numpy(),
    })


def inputs_market(view, games: pd.DataFrame) -> pd.DataFrame:
    m = _MARKET.build(view, games[["game_id"]])
    return m.merge(games[["game_id", "start_time_utc"]], on="game_id", how="inner")


# ---- one logistic baseline ---------------------------------------------------------------

@dataclass(frozen=True)
class LogisticBaseline:
    model_id: str
    inputs: Callable[[object, pd.DataFrame], pd.DataFrame]
    columns: tuple[str, ...]
    at_kickoff: bool = False

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model_id == "baseline_close":
            frame = frame.assign(neg_spread=-frame["mkt_spread_close"])
        return frame.dropna(subset=list(self.columns))

    def fit(self, view, train: pd.DataFrame) -> np.ndarray:
        fbs = train[train["home_fbs"].astype(bool) & train["away_fbs"].astype(bool)]
        fbs = fbs[fbs["home_points"] != fbs["away_points"]]
        x = self.design(self.inputs(view, fbs))
        g = fbs[["game_id", "home_points", "away_points"]].merge(x, on="game_id", how="inner")
        if g.empty:
            raise ValueError(f"{self.model_id}: no training games at {view.cutoff}")
        y = (g["home_points"] > g["away_points"]).to_numpy(dtype=float)
        return logit_fit(y, g[list(self.columns)].to_numpy(dtype=float))

    def predict(self, coef: np.ndarray, view, slate: pd.DataFrame) -> pd.DataFrame:
        x = self.design(self.inputs(view, slate))
        if x.empty:
            return x
        X = x[list(self.columns)].to_numpy(dtype=float)
        out = pd.DataFrame({"game_id": x["game_id"].to_numpy(), "p_home_win": sigmoid(X @ coef)})
        out["reasons"] = [self.reasons(row, coef) for row in X]
        if self.at_kickoff:
            out["cutoff_utc"] = x["start_time_utc"].array
        return out

    def reasons(self, row: np.ndarray, coef: np.ndarray) -> list[dict]:
        items = []
        for name, value, b in zip(self.columns, row, coef):
            effect = round(float(value * b), 4)
            if name == "neg_spread":
                text = f"Closing spread {(-value):+g} for the home team"
                items.append({"text": text, "feature": "mkt_spread_close", "value": float(-value), "effect": effect})
            elif name == "elo_gap":
                items.append({"text": f"Pregame Elo gap {value * 100:+.0f} (home minus away)",
                              "feature": "elo_gap", "value": round(float(value * 100), 1), "effect": effect})
            elif name == "home" and value:
                items.append({"text": "Home field", "feature": "home", "value": 1.0, "effect": effect})
        return sorted(items, key=lambda r: -abs(r["effect"]))


CLOSE = LogisticBaseline("baseline_close", inputs_market, ("neg_spread",), at_kickoff=True)
ELO = LogisticBaseline("baseline_elo", inputs_elo, ("home", "elo_gap"))
HOME = LogisticBaseline("baseline_home", inputs_home, ("home",))


def _moneyline_fit(view, train):
    return None


def _moneyline_predict(model, view, slate: pd.DataFrame) -> pd.DataFrame:
    m = inputs_market(view, slate).dropna(subset=["mkt_p_home_ml"])
    if m.empty:
        return m
    return pd.DataFrame({
        "game_id": m["game_id"].to_numpy(),
        "p_home_win": m["mkt_p_home_ml"].to_numpy(),
        "cutoff_utc": m["start_time_utc"].array,
        "reasons": [[{"text": f"Moneyline {h:+.0f} / {a:+.0f}, vig removed", "feature": "mkt_p_home_ml",
                      "value": round(float(p), 4)}]
                    for h, a, p in zip(m["mkt_ml_home"], m["mkt_ml_away"], m["mkt_p_home_ml"])],
    })


def run_baseline(warehouse: Warehouse, model_id: str, seasons=DEFAULT_BASELINE_SEASONS, **kwargs) -> pd.DataFrame:
    """Walk-forward predictions of one baseline (refit once per season)."""
    if model_id == "baseline_moneyline":
        return walk_forward(warehouse, _moneyline_fit, _moneyline_predict, model_id, seasons, **kwargs)
    model = {b.model_id: b for b in (CLOSE, ELO, HOME)}[model_id]
    return walk_forward(warehouse, model.fit, model.predict, model_id, seasons, refit="season", **kwargs)


__all__ = [
    "BASELINES", "CLOSE", "DEFAULT_BASELINE_SEASONS", "ELO", "HOME", "LogisticBaseline", "logit_fit",
    "run_baseline", "sigmoid",
]
