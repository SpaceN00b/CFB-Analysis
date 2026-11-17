"""Opponent-adjusted ridge ratings with a prior.

One model serves every rating in this package (unit EPA, success, explosiveness,
points). Each observation is one offense against one defense in one game:

    y_i = mu + hfa * h_i + off[o_i] + def[d_i] + noise,   weight w_i

with h_i = +1 when the offense is the home team, -1 when the defense is, 0 at a
neutral site. The penalty pulls each team value toward its prior:

    lam * sum_t (off_t - off0_t)^2 + lam * sum_t (def_t - def0_t)^2
    + lam_mu * (mu - mu0)^2 + lam_hfa * (hfa - hfa0)^2

so with no games every team sits at its prior, and with many games the data win.
`lam` is measured in units of the weights (plays for unit ratings, games for points).
Defensive values are what the defense allows: higher is worse for a defense.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeFit:
    mu: float
    hfa: float
    off: np.ndarray        # deviation from mu, one per team index
    deff: np.ndarray       # deviation from mu, one per team index
    weight_off: np.ndarray  # total observation weight per team as offense
    weight_def: np.ndarray


def fit(n_teams: int, o: np.ndarray, d: np.ndarray, h: np.ndarray, y: np.ndarray, w: np.ndarray,
        lam: float, off0: np.ndarray | None = None, def0: np.ndarray | None = None,
        mu0: float | None = None, lam_mu: float = 0.0, hfa0: float = 0.0, lam_hfa: float = 1.0) -> RidgeFit:
    """Solve the penalized least squares problem above.

    `o`, `d` are team indexes in [0, n_teams). Observations with a non-finite `y` or a
    non-positive weight are ignored. When `mu0` is None, mu is left unpenalized; with
    no usable observations the fit is then all NaN.
    """
    o = np.asarray(o, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    o, d, h, y, w = o[ok], d[ok], h[ok], y[ok], w[ok]
    n = n_teams
    off0 = np.zeros(n) if off0 is None else np.nan_to_num(np.asarray(off0, dtype=np.float64))
    def0 = np.zeros(n) if def0 is None else np.nan_to_num(np.asarray(def0, dtype=np.float64))
    wo = np.bincount(o, weights=w, minlength=n)
    wd = np.bincount(d, weights=w, minlength=n)
    if mu0 is None and not len(y):
        nan = np.full(n, np.nan)
        return RidgeFit(np.nan, np.nan, nan, nan.copy(), wo, wd)

    # Work in deviations from the prior so the team penalty is centred on zero.
    r = y - off0[o] - def0[d]
    # Parameter layout: [mu, hfa, off_0..off_{n-1}, def_0..def_{n-1}]
    p = 2 + 2 * n
    A = np.zeros((p, p))
    b = np.zeros(p)
    io, idd = 2 + o, 2 + n + d
    # Diagonal and cross terms of X'WX, accumulated per observation.
    sw, swh, swhh = w.sum(), (w * h).sum(), (w * h * h).sum()
    A[0, 0] += sw
    A[0, 1] += swh
    A[1, 0] += swh
    A[1, 1] += swhh
    wh_o = np.bincount(o, weights=w * h, minlength=n)
    wh_d = np.bincount(d, weights=w * h, minlength=n)
    A[0, 2:2 + n] += wo
    A[2:2 + n, 0] += wo
    A[0, 2 + n:] += wd
    A[2 + n:, 0] += wd
    A[1, 2:2 + n] += wh_o
    A[2:2 + n, 1] += wh_o
    A[1, 2 + n:] += wh_d
    A[2 + n:, 1] += wh_d
    A[np.arange(2, 2 + n), np.arange(2, 2 + n)] += wo
    A[np.arange(2 + n, p), np.arange(2 + n, p)] += wd
    np.add.at(A, (io, idd), w)
    np.add.at(A, (idd, io), w)
    b[0] = (w * r).sum()
    b[1] = (w * h * r).sum()
    b[2:2 + n] = np.bincount(o, weights=w * r, minlength=n)
    b[2 + n:] = np.bincount(d, weights=w * r, minlength=n)
    # Penalties.
    A[np.arange(2, p), np.arange(2, p)] += lam
    if mu0 is not None:
        A[0, 0] += lam_mu
        b[0] += lam_mu * mu0
    else:
        A[0, 0] += 1e-9
    A[1, 1] += lam_hfa
    b[1] += lam_hfa * hfa0
    theta = np.linalg.solve(A, b)
    return RidgeFit(float(theta[0]), float(theta[1]), off0 + theta[2:2 + n], def0 + theta[2 + n:], wo, wd)
