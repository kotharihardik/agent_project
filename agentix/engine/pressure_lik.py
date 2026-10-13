"""Step 2: pressure evidence -> log-likelihood for "leak at junction j" (all 782 at once).

Model:  drop_s = q * h_js + noise,  h_js = g_js - beta_s
  g_js  : nominal signature (pressure drop at sensor s per L/s of leak at j), night/day mix
  beta_s: how much the demand regression already 'explained away' (see preprocess.py)
  q     : unknown leak size; prior from the inflow rise (q_hat, q_var), integrated on a grid
  noise : measurement variance + relative model error (the nominal map is not the real town)
Also returns the no-leak log-likelihood (q = 0) used for P(leak).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .preprocess import Evidence

Q_GRID = np.geomspace(0.2, 15.0, 40)


class SignatureLibrary:
    def __init__(self, path: str):
        z = np.load(path)
        self.junctions = list(z["junctions"])
        self.sizes = list(z["sizes_lps"])
        i3 = self.sizes.index(3.0)
        # per-L/s signatures (3 L/s run is in the middle of our leak range)
        self.g_night = z["dp_night"][i3] / 3.0            # [J_leak, J_obs]
        self.g_day = z["dp_day"][i3] / 3.0
        self.base_p_night = z["base_p_night"]


def pressure_loglik(ev: Evidence, lib: SignatureLibrary, sensor_idx: list[int], rel_model_err: float = 0.25,
                    floor_m: float = 0.01, q_prior_rel: float = 0.4):
    g = ev.night_frac * lib.g_night[:, sensor_idx] + (1 - ev.night_frac) * lib.g_day[:, sensor_idx]   # [J,k]
    h = g - ev.beta_m_per_lps[None, :]
    d = ev.drop_m[None, None, :]                                   # [1,1,k]
    mu = Q_GRID[None, :, None] * h[:, None, :]                      # [J,Q,k]
    var = ev.drop_var[None, None, :] + (rel_model_err * mu) ** 2 + floor_m ** 2
    ll = -0.5 * np.sum((d - mu) ** 2 / var + np.log(2 * np.pi * var), axis=2)   # [J,Q]
    # prior over leak size from the inflow rise (wide, because night demand wobbles a lot)
    sd = np.sqrt(ev.q_var + (q_prior_rel * max(ev.q_hat_lps, 0.5)) ** 2)
    lp = -0.5 * ((Q_GRID - max(ev.q_hat_lps, 0.0)) / sd) ** 2
    lp += np.log(np.gradient(np.log(Q_GRID)) * Q_GRID)               # grid cell widths (log-spaced grid)
    lp -= logsumexp(lp)
    llm = logsumexp(ll + lp[None, :], axis=1)                       # [J] marginal over leak size
    q_map = Q_GRID[np.argmax(ll + lp[None, :], axis=1)]
    var0 = ev.drop_var + floor_m ** 2
    ll0 = float(-0.5 * np.sum(ev.drop_m ** 2 / var0 + np.log(2 * np.pi * var0)))
    # misfit of the best node (chi2 per sensor) -> used by gates / abduction triggers
    jb = int(np.argmax(llm))
    qb = q_map[jb]
    r = ev.drop_m - qb * h[jb]
    vb = ev.drop_var + (rel_model_err * qb * h[jb]) ** 2 + floor_m ** 2
    misfit = float(np.mean(r ** 2 / vb))
    return llm, ll0, q_map, misfit
