"""Observation models used by the INVESTIGATOR (inference side) for each action.

These are the investigator's own beliefs about how tools behave - deliberately simple and NOT copied from the
environment's internals (configs/costs.yaml is the env's truth; parameters here are separate and slightly different,
as a real utility would only roughly know its tools' accuracy).
Each function returns (lik_over_junctions, lik_if_no_leak) for an observed outcome, and `outcomes()` lists possible
outcomes with their probabilities under each hypothesis (used by the value-of-information engine).
"""
from __future__ import annotations

import numpy as np

# investigator's assumptions about its tools
ACOUSTIC_P_DETECT = 0.65          # chance a survey hears a leak inside its radius
ACOUSTIC_FALSE_POS = 0.06
ACOUSTIC_LOC_SIGMA_M = 45.0       # how close the reported noise point is to the leak
STEP_NOISE_LPS = 0.45             # flow noise + customer-demand uncertainty in a step test
LOGGER_SIGMA_M = 0.10             # one temporary logger vs the nominal model (offset + model mismatch)


class ObsModels:
    def __init__(self, net, lib, rep_model):
        self.net, self.lib, self.rep = net, lib, rep_model
        self.D = net.pipe_distance
        self.J = len(net.junctions)

    # ---------------- acoustic survey around centre c with radius R
    def acoustic(self, c: int, R: float, detected: bool, approx: int | None):
        inside = self.D[c] <= R
        if not detected:
            lik = np.where(inside, 1 - ACOUSTIC_P_DETECT, 1.0) * (1 - ACOUSTIC_FALSE_POS)
            return lik, 1 - ACOUSTIC_FALSE_POS
        ring = np.where(inside)[0]
        u = 1.0 / max(len(ring), 1)
        if approx is None:
            return np.where(inside, ACOUSTIC_P_DETECT, 0.0) + ACOUSTIC_FALSE_POS * u, ACOUSTIC_FALSE_POS * u
        k = np.exp(-0.5 * (self.D[approx] / ACOUSTIC_LOC_SIGMA_M) ** 2)
        k = k / max(k[inside].sum(), 1e-12)
        lik = np.where(inside, ACOUSTIC_P_DETECT * k, 0.0) + ACOUSTIC_FALSE_POS * u
        return lik, ACOUSTIC_FALSE_POS * u

    # ---------------- step test: isolate node set S at night, measure inflow drop minus expected customer drop
    def step_test(self, S: np.ndarray, excess_lps: float, q_hat: float, q_sd: float):
        inside = np.zeros(self.J, bool)
        inside[S] = True
        sd_in = np.sqrt(STEP_NOISE_LPS ** 2 + q_sd ** 2)
        g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / s
        lik = np.where(inside, g(excess_lps, q_hat, sd_in), g(excess_lps, 0.0, STEP_NOISE_LPS))
        return lik, float(g(excess_lps, 0.0, STEP_NOISE_LPS))

    # ---------------- temporary logger at node n: night mean vs nominal prediction
    def logger(self, n: int, reading: float, q_hat: float, offset_est: float = 0.0):
        pred = self.lib.base_p_night[n] + offset_est - q_hat * self.lib.g_night[:, n]
        g = lambda x, m: np.exp(-0.5 * ((x - m) / LOGGER_SIGMA_M) ** 2)
        return g(reading, pred), float(g(reading, self.lib.base_p_night[n] + offset_est))

    # ---------------- reporter answer to a multiple-choice "where did you see the water?"
    def answer_matrix(self, option_nodes: list[np.ndarray], src: str, eps: float):
        """P(answer = k | leak at j) for k = options..., 'elsewhere/unknown' (last column). Shape [J, K+1]."""
        K = len(option_nodes)
        xy = self.net.xy
        # choice given the reporter's seen node n: nearest option if within 120 m (else elsewhere)
        dmin = np.stack([np.min(np.hypot(*(xy[:, None, :] - xy[o][None, :, :]).transpose(2, 0, 1)), axis=1)
                         if len(o) else np.full(self.J, np.inf) for o in option_nodes], axis=1)      # [J_seen, K]
        choice = np.zeros((self.J, K + 1))
        best = np.argmin(dmin, axis=1)
        close = dmin[np.arange(self.J), best] <= 120
        choice[np.arange(self.J), np.where(close, best, K)] = 0.88
        choice[:, K] += 0.12
        a = {"yes": 0.85, "no": 0.25}.get(src, 0.6)
        seen = a * self.rep.k_near + (1 - a) * self.rep.k_down                    # [J_leak, J_seen]
        related = seen @ choice                                                    # [J_leak, K+1]
        unrelated = self.rep.background @ choice                                   # [K+1]
        return (1 - eps) * related + eps * unrelated[None, :], unrelated
