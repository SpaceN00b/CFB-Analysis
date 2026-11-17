"""Model families for the home-win probability. numpy only.

- `LogitModel`: ridge-penalised logistic regression on standardised features.
- `MarginModel`: ridge regression on the home margin, turned into a probability by a
  logistic curve fitted on out-of-fold margin predictions (season blocks).

Both share `Prep`: column means and SDs from the training rows, missing values set to the
training mean (0 after scaling), optional missing-value indicators. Training rows can be
weighted by season (`era_half_life`, in seasons, counted back from the newest training
season).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -35, 35)))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@dataclass
class Prep:
    columns: list[str]
    mean: np.ndarray
    sd: np.ndarray
    indicator_cols: list[str]

    @classmethod
    def fit(cls, X: pd.DataFrame, columns: list[str], indicators: float | None = 0.05) -> "Prep":
        A = X[columns].to_numpy(dtype="float64")
        with np.errstate(all="ignore"):
            mean = np.nanmean(A, axis=0)
            sd = np.nanstd(A, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        ind = []
        if indicators is not None:
            miss = np.isnan(A).mean(axis=0)
            ind = [c for c, m in zip(columns, miss) if indicators <= m < 1.0]
        return cls(list(columns), mean, sd, ind)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.columns].to_numpy(dtype="float64")
        Z = (A - self.mean) / self.sd
        Z = np.where(np.isfinite(Z), Z, 0.0)
        if self.indicator_cols:
            M = X[self.indicator_cols].isna().to_numpy(dtype="float64")
            Z = np.hstack([Z, M])
        return Z

    @property
    def names(self) -> list[str]:
        return self.columns + [f"missing:{c}" for c in self.indicator_cols]


def era_weights(seasons: np.ndarray, half_life: float | None) -> np.ndarray:
    seasons = np.asarray(seasons, float)
    if not half_life:
        return np.ones(len(seasons))
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def ridge_logit(Z, y, w, lam: float, iterations: int = 100) -> tuple[float, np.ndarray]:
    """Weighted logistic regression with an unpenalised intercept and L2 penalty lam * sum(w)/n
    on the coefficients (so lam is per observation)."""
    n, k = Z.shape
    X = np.hstack([np.ones((n, 1)), Z])
    b = np.zeros(k + 1)
    pen = np.full(k + 1, lam * w.sum())
    pen[0] = 0.0
    for _ in range(iterations):
        p = sigmoid(X @ b)
        g = X.T @ (w * (y - p)) - pen * b
        H = (X.T * (w * p * (1 - p))) @ X + np.diag(pen) + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(b[0]), b[1:]


def ridge_ls(Z, y, w, lam: float) -> tuple[float, np.ndarray]:
    n, k = Z.shape
    X = np.hstack([np.ones((n, 1)), Z])
    pen = np.full(k + 1, lam * w.sum())
    pen[0] = 0.0
    b = np.linalg.solve((X.T * w) @ X + np.diag(pen) + 1e-9 * np.eye(k + 1), X.T @ (w * y))
    return float(b[0]), b[1:]


@dataclass
class LogitModel:
    columns: list[str]
    lam: float = 0.01
    era_half_life: float | None = None
    indicators: float | None = 0.05
    prep: Prep | None = None
    intercept: float = 0.0
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, X: pd.DataFrame, y_win: np.ndarray, margin: np.ndarray | None = None) -> "LogitModel":
        self.prep = Prep.fit(X, self.columns, self.indicators)
        Z = self.prep.transform(X)
        w = era_weights(X["season"].to_numpy(), self.era_half_life)
        self.intercept, self.coef = ridge_logit(Z, np.asarray(y_win, float), w, self.lam)
        return self

    def linear(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.prep.transform(X) @ self.coef

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.linear(X))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row additive log-odds contribution of each input, toward the home team."""
        Z = self.prep.transform(X)
        return pd.DataFrame(Z * self.coef, columns=self.prep.names, index=X.index)


@dataclass
class MarginModel:
    """Ridge on the home margin; probability = sigmoid(a + s * margin_hat), with (a, s)
    fitted on out-of-fold margins (folds = training seasons) so the curve is not too sharp."""
    columns: list[str]
    lam: float = 0.01
    era_half_life: float | None = None
    indicators: float | None = 0.05
    prep: Prep | None = None
    intercept: float = 0.0
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    calib: tuple[float, float] = (0.0, 0.0)

    def _fit_ls(self, X, margin):
        prep = Prep.fit(X, self.columns, self.indicators)
        w = era_weights(X["season"].to_numpy(), self.era_half_life)
        b0, b = ridge_ls(prep.transform(X), np.asarray(margin, float), w, self.lam)
        return prep, b0, b

    def fit(self, X: pd.DataFrame, y_win: np.ndarray, margin: np.ndarray | None = None) -> "MarginModel":
        margin = np.asarray(margin, float)
        y_win = np.asarray(y_win, float)
        seasons = X["season"].to_numpy()
        oof = np.full(len(X), np.nan)
        uniq = np.unique(seasons)
        if len(uniq) >= 3:
            for s in uniq:
                tr, te = seasons != s, seasons == s
                prep, b0, b = self._fit_ls(X[tr], margin[tr])
                oof[te] = b0 + prep.transform(X[te]) @ b
        self.prep, self.intercept, self.coef = self._fit_ls(X, margin)
        if np.isnan(oof).any():
            oof = self.intercept + self.prep.transform(X) @ self.coef
        w = era_weights(seasons, self.era_half_life)
        a, s = ridge_logit(oof[:, None], y_win, w, 0.0)
        self.calib = (a, float(s[0]))
        return self

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.prep.transform(X) @ self.coef

    def linear(self, X: pd.DataFrame) -> np.ndarray:
        a, s = self.calib
        return a + s * self.predict_margin(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.linear(X))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.prep.transform(X)
        return pd.DataFrame(Z * self.coef * self.calib[1], columns=self.prep.names, index=X.index)


__all__ = ["LogitModel", "MarginModel", "Prep", "era_weights", "log_loss", "ridge_logit", "ridge_ls", "sigmoid"]
