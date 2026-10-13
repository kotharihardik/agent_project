"""Step 1 of the engine: turn raw sensor series into leak evidence.

1. Demand normalisation (per sensor, fitted on the quiet baseline days):
       p_s(t) ~ a + b Q(t) + c Q(t)^2 + d H_tank(t) + e Q_pump(t)
   Normal water use moves pressure by much more than a small leak does, so we remove it first (LILA uses the same
   idea with pairwise regressions). Residual r_s(t) = p_s(t) - prediction.
2. Inflow residual: Q(t) minus its time-of-day profile from the baseline days -> leak size evidence.
3. Change point: the onset t0 that best explains a downward pressure shift + upward inflow shift.
4. Evidence: mean pressure drop per sensor after t0 (with an honest variance) and leak size estimate with variance.
A leak adds q to the inflow, so the regression 'explains away' part of its pressure effect; the expected residual
drop is q * (g_j - beta) where beta_s = -dp_s/dQ (see pressure_lik.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Evidence:
    t0_s: float                  # estimated leak onset
    open_s: float
    drop_m: np.ndarray           # [k] mean pressure drop after t0 (positive = pressure went down)
    drop_var: np.ndarray         # [k] variance of drop_m (noise + residual-model error)
    beta_m_per_lps: np.ndarray   # [k] pressure sensitivity to total inflow at night (m per L/s, positive)
    q_hat_lps: float             # leak size estimate from inflow
    q_var: float
    night_frac: float            # share of post-onset samples in night hours (choose signature window)
    shift_score: float           # change-point statistic (how clear the shift is)
    n_post: int
    resid_sigma: np.ndarray      # [k] baseline residual std


def _design(Q, H, Qp):
    return np.column_stack([np.ones_like(Q), Q, Q ** 2, H, Qp])


def _tod_profile(t, x, mask, step_s=900):
    slot = ((t % 86400) // step_s).astype(int)
    prof = np.zeros(86400 // step_s)
    for s in range(len(prof)):
        m = mask & (slot == s)
        prof[s] = x[m].mean() if m.any() else np.nan
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    return prof[slot]


def build_evidence(time_s, pressure, flow, tank_level, open_s, baseline_days: float = 5.0,
                   window: str = "last", window_h: float = 6.0,
                   search_back_h: float = 60.0, min_post_h: float = 2.0, corr_factor: float = 6.0) -> Evidence:
    """pressure [T,k] m; flow [T,3] m3/h (p227, p235, PUMP_1); tank_level [T,1] m.

    window = "last": evidence = the last `window_h` hours before case open (best on dev, see ADR-006);
    window = "changepoint": evidence = everything after an estimated onset (kept for diagnostics).
    """
    t = np.asarray(time_s, dtype=float)
    Q = (flow[:, 0] + flow[:, 1]) / 3.6                    # town inflow, L/s
    Qp = flow[:, 2] / 3.6 if flow.shape[1] > 2 else np.zeros_like(Q)
    H = tank_level[:, 0] if tank_level.size else np.zeros_like(Q)
    base = t < baseline_days * 86400
    X = _design(Q, H, Qp)
    coef, *_ = np.linalg.lstsq(X[base], pressure[base], rcond=None)
    resid = pressure - X @ coef                              # [T,k]
    sig = resid[base].std(axis=0) + 1e-4
    night = ((t % 86400) / 3600 >= 1) & ((t % 86400) / 3600 < 5)
    Qn = Q[base & night].mean() if (base & night).any() else Q[base].mean()
    beta = -(coef[1] + 2 * coef[2] * Qn)                      # m per L/s (positive: more flow, lower pressure)
    eQ = Q - _tod_profile(t, Q, base)
    sigQ = eQ[base].std() + 1e-3

    if window == "last":
        search_back_h, min_post_h = window_h, window_h - 1e-6
    # change point over candidate onsets (hourly grid); with window="last" there is exactly one candidate
    cands = np.arange(max(baseline_days * 86400, open_s - search_back_h * 3600), open_s - min_post_h * 3600 + 1, 3600)
    best, best_t0 = -np.inf, cands[0] if len(cands) else open_s - min_post_h * 3600
    for t0 in cands:
        post = (t >= t0) & (t <= open_s)
        n = post.sum()
        if n < 4:
            continue
        zp = -resid[post].mean(axis=0) / sig                  # positive if pressure dropped
        zq = eQ[post].mean() / sigQ
        stat = n / corr_factor * (np.sum(np.clip(zp, 0, None) ** 2) + max(zq, 0) ** 2)
        if stat > best:
            best, best_t0 = stat, t0
    post = (t >= best_t0) & (t <= open_s)
    n = int(post.sum())
    n_eff = max(n / corr_factor, 1.0)
    drop = -resid[post].mean(axis=0)
    drop_var = sig ** 2 / n_eff
    q_hat = float(eQ[post].mean())
    q_var = float(sigQ ** 2 / n_eff)
    night_frac = float(night[post].mean()) if n else 0.5
    return Evidence(t0_s=float(best_t0), open_s=float(open_s), drop_m=drop, drop_var=drop_var,
                    beta_m_per_lps=beta, q_hat_lps=q_hat, q_var=q_var, night_frac=night_frac,
                    shift_score=float(best), n_post=n, resid_sigma=sig)
