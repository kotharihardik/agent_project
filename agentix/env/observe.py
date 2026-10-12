"""Measurement model for pressure sensors and temporary loggers.

History (see ADR-002 and ADR-005):
- v1 (perturbed truth): large noise (bias 0.09, drift 0.065, white 0.065 m) stood in for model mismatch.
- v2 (official Real truth): the Real model reproduces BattLeDIM SCADA to 0.1 cm, i.e. the official data has NO
  sensor noise (only 0.01 m rounding); the 6 cm SCADA-vs-nominal gaps are pure model mismatch, which the Real truth
  now creates physically. So v2 adds only realistic INSTRUMENT noise on top.

observed = clean + offset_s + drift_s(t) + white(t), rounded to `quant`.
A temporary logger gets a fresh unknown offset (no history at that spot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorNoise:
    bias_sigma: float = 0.05        # installation / calibration offset (constant per sensor)
    drift_sigma: float = 0.02       # slow drift
    drift_tau_days: float = 5.0
    white_sigma: float = 0.03       # reading noise
    quant: float = 0.01             # data resolution (BattLeDIM rounds to 0.01 m)


NOISE_PRESETS = {
    "field": SensorNoise(),                                                     # main experiments
    "battledim_exact": SensorNoise(0.0, 0.0, 5.0, 0.0, 0.01),                  # identical to the official data
    "harsh": SensorNoise(0.10, 0.05, 2.0, 0.06, 0.01),                         # robustness check
    "v1_perturbed": SensorNoise(0.09, 0.065, 2.0, 0.065, 0.01),                # Phase-1 setting (perturbed truth)
}


def drift_series(rng, n_days: int, n_sensors: int, sigma: float, tau_days: float) -> np.ndarray:
    """Daily AR(1) offsets [n_days+1, n_sensors] with stationary std `sigma`."""
    rho = np.exp(-1.0 / tau_days)
    x = np.empty((n_days + 1, n_sensors))
    x[0] = rng.normal(0, sigma, n_sensors)
    for d in range(1, n_days + 1):
        x[d] = rho * x[d - 1] + rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n_sensors)
    return x


def observe(clean: np.ndarray, time_s: np.ndarray, rng, noise: SensorNoise) -> tuple[np.ndarray, dict]:
    """clean: [T, S] true pressures. Returns noisy observations and the hidden noise components."""
    T, S = clean.shape
    n_days = int(np.ceil(time_s[-1] / 86400)) + 1
    bias = rng.normal(0, noise.bias_sigma, S)
    daily = drift_series(rng, n_days, S, noise.drift_sigma, noise.drift_tau_days)
    day_f = time_s / 86400.0
    d0 = np.floor(day_f).astype(int)
    w = (day_f - d0)[:, None]
    drift = (1 - w) * daily[d0] + w * daily[np.minimum(d0 + 1, n_days)]
    obs = clean + bias[None, :] + drift + rng.normal(0, noise.white_sigma, (T, S))
    obs = np.round(obs / noise.quant) * noise.quant
    return obs.astype(np.float32), {"bias": bias, "daily_drift": daily}


def logger_reading(clean_night_mean: float, rng, noise: SensorNoise, n_samples: int = 24) -> float:
    """A temporary logger's night (02-04h) mean: fresh offset + drift + averaged white noise."""
    v = (clean_night_mean + rng.normal(0, noise.bias_sigma) + rng.normal(0, noise.drift_sigma)
         + rng.normal(0, noise.white_sigma / np.sqrt(n_samples)))
    return float(np.round(v / noise.quant) * noise.quant)
