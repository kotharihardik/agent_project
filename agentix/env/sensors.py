"""Sensor layouts (k = 3/5/10) and an identifiability measure for leak localization by pressure.

Three layout families (same layouts are used for every method):
- random:       seeded random subset of the candidate locations
- engineering:  utility-style: start at the lowest-pressure candidate, then spread out (farthest-point by pipe distance)
- info:         greedy choice that minimises the average ambiguity set size (uses the nominal signature library)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def scaled_residual(sig_true: np.ndarray, sig_all: np.ndarray) -> np.ndarray:
    """min_a ||t - a*s_i||^2 for every candidate i (leak size unknown -> fit a >= 0)."""
    num = sig_all @ sig_true
    den = np.einsum("ij,ij->i", sig_all, sig_all) + 1e-12
    a = np.clip(num / den, 0, None)
    r = sig_true[None, :] - a[:, None] * sig_all
    return np.einsum("ij,ij->i", r, r)


def ambiguity_sets(sig: np.ndarray, sensor_idx: list[int], sigma: float, conf: float = 0.95) -> list[np.ndarray]:
    """For each true leak node j: candidates whose (scaled) sensor signature is within noise of j's.

    sig: [n_leak_nodes, n_junctions] pressure drops for one leak size and window.
    """
    S = sig[:, sensor_idx]
    k = len(sensor_idx)
    thr = chi2.ppf(conf, df=max(k - 1, 1)) * sigma ** 2
    out = []
    for j in range(S.shape[0]):
        if np.linalg.norm(S[j]) < sigma:  # leak invisible at these sensors: everything is ambiguous
            out.append(np.arange(S.shape[0]))
            continue
        out.append(np.where(scaled_residual(S[j], S) <= thr)[0])
    return out


def random_layout(candidates: list[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(candidates, size=k, replace=False).tolist())


def engineering_layout(candidates: list[str], k: int, dist: np.ndarray, cand_idx: list[int],
                       base_pressure: np.ndarray) -> list[str]:
    """Start at the critical (lowest pressure) candidate, then add the candidate farthest (pipe distance) from chosen."""
    chosen = [int(np.argmin(base_pressure[cand_idx]))]
    while len(chosen) < k:
        d = np.min(dist[np.ix_([cand_idx[c] for c in chosen], cand_idx)], axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return sorted(candidates[c] for c in chosen)


def info_layout(candidates: list[str], k: int, sig: np.ndarray, cand_idx: list[int], sigma: float,
                eval_nodes: np.ndarray | None = None) -> list[str]:
    """Greedy: add the candidate that most reduces mean ambiguity-set size over evaluation leak nodes."""
    eval_nodes = eval_nodes if eval_nodes is not None else np.arange(sig.shape[0])
    sub = sig[eval_nodes]
    chosen: list[int] = []
    for _ in range(k):
        best, best_score = None, np.inf
        for c in range(len(candidates)):
            if c in chosen:
                continue
            idx = [cand_idx[x] for x in chosen + [c]]
            sets = ambiguity_sets(sub, idx, sigma)
            score = np.mean([len(s) for s in sets])
            if score < best_score:
                best, best_score = c, score
        chosen.append(best)
    return sorted(candidates[c] for c in chosen)
