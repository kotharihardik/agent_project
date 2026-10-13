"""Decision-focused value of information (VOI) for candidate investigation actions.

Target quantity: P(success) = P(leak) * P(leak within the dig radius of the best dig spot).
- One-step thresholded VOI (E[max(U_dig after,0)] - max(U_dig now,0)) is exact but MYOPIC: one check rarely lifts
  P(success) over the dig threshold, so it undervalues sequences (pilot: the policy stopped almost at once).
- We therefore rank actions by the expected GAIN in P(success) per unit cost (smooth, decision-focused), spend a
  fixed investigation budget, and dig only when P(success) passes the dig threshold (ADR-006).
Information gain about the whole map is NOT the target: only information about the dig decision counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief import Belief
from .obs_models import (ACOUSTIC_FALSE_POS, ACOUSTIC_LOC_SIGMA_M, ACOUSTIC_P_DETECT, LOGGER_SIGMA_M, STEP_NOISE_LPS,
                         ObsModels)


@dataclass
class Economics:
    benefit_found: float = 3.0      # CU value of finding the leak (repair now instead of later)
    dig_cost: float = 1.5           # CU, matches configs/costs.yaml
    water_cost_per_m3: float = 0.002

    def u_dig(self, p_success: float) -> float:
        return self.benefit_found * p_success - self.dig_cost

    def stop_value(self, b: Belief) -> float:
        _, p = b.best_dig()
        return max(self.u_dig(p), 0.0)

    def delay_cost(self, b: Belief, hours: float) -> float:
        return self.water_cost_per_m3 * b.p_leak * max(b.q_hat, 0.0) * 3.6 * hours


@dataclass
class Candidate:
    kind: str
    args: dict
    cost: float
    delay_h: float
    voi: float = 0.0
    net: float = 0.0


def simulate_outcome(kind, args, j, b: Belief, om: ObsModels, rng):
    """Sample an outcome under hypothesis 'leak at j' (j = -1 means no leak), from the investigator's own model."""
    D = om.D
    if kind == "acoustic":
        c, R = args["c"], args["R"]
        if j >= 0 and D[c, j] <= R and rng.random() < ACOUSTIC_P_DETECT:
            near = np.where(np.abs(D[j] - abs(rng.normal(0, ACOUSTIC_LOC_SIGMA_M))) < 30)[0]
            return {"detected": True, "approx": int(rng.choice(near)) if len(near) else j}
        if rng.random() < ACOUSTIC_FALSE_POS:
            ring = np.where(D[c] <= R)[0]
            return {"detected": True, "approx": int(rng.choice(ring))}
        return {"detected": False, "approx": None}
    if kind == "step":
        inside = j >= 0 and j in set(args["S"].tolist())
        q = max(rng.normal(b.q_hat, b.q_sd), 0.0) if inside else 0.0
        return {"excess": q + rng.normal(0, STEP_NOISE_LPS)}
    if kind == "logger":
        n = args["n"]
        pred = om.lib.base_p_night[n] + args["offset"] - (b.q_hat * om.lib.g_night[j, n] if j >= 0 else 0.0)
        return {"reading": pred + rng.normal(0, LOGGER_SIGMA_M)}
    if kind == "ask":
        P = args["P"]                                     # [J, K+1]
        row = P[j] if j >= 0 else args["P_none"]
        row = row / row.sum()
        return {"answer": int(rng.choice(len(row), p=row))}
    raise ValueError(kind)


def likelihood(kind, args, out, b: Belief, om: ObsModels):
    if kind == "acoustic":
        return om.acoustic(args["c"], args["R"], out["detected"], out["approx"])
    if kind == "step":
        return om.step_test(args["S"], out["excess"], b.q_hat, b.q_sd)
    if kind == "logger":
        return om.logger(args["n"], out["reading"], b.q_hat, args["offset"])
    if kind == "ask":
        P, P0 = args["P"], args["P_none"]
        k = out["answer"]
        return P[:, k], float(P0[k] / P0.sum())
    raise ValueError(kind)


def estimate_voi(cand: Candidate, b: Belief, om: ObsModels, econ: Economics, rng, n_samples: int = 24) -> float:
    now = econ.stop_value(b)
    vals = []
    for _ in range(n_samples):
        j = int(rng.choice(len(b.post), p=b.post)) if rng.random() < b.p_leak else -1
        out = simulate_outcome(cand.kind, cand.args, j, b, om, rng)
        nb = b.copy()
        lik, lik0 = likelihood(cand.kind, cand.args, out, nb, om)
        nb.update(lik, lik0)
        vals.append(econ.stop_value(nb))
    return float(np.mean(vals) - now)


def estimate_gain(cand: Candidate, b: Belief, om: ObsModels, rng, n_samples: int = 24) -> float:
    """Expected increase of P(success) = p_leak * best coverage after observing the action's outcome."""
    now = b.best_dig()[1]
    vals = []
    for _ in range(n_samples):
        j = int(rng.choice(len(b.post), p=b.post)) if rng.random() < b.p_leak else -1
        out = simulate_outcome(cand.kind, cand.args, j, b, om, rng)
        nb = b.copy()
        lik, lik0 = likelihood(cand.kind, cand.args, out, nb, om)
        nb.update(lik, lik0)
        vals.append(nb.best_dig()[1])
    return float(np.mean(vals) - now)
