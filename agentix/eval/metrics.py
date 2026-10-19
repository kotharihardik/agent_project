"""Metrics for localization and selective dig decisions (RQ1). See docs/04 §8.

Per case (computed in the runner): distance from the dig node to the true leak (pipe metres), hit@100/@300,
confidence, search-plan rank (how many 100 m areas a crew checks before finding the leak).
Aggregates here: rates with bootstrap CIs, risk-coverage curve / AURC, precision at a confidence threshold,
false-dispatch rate on no-leak cases, paired McNemar test between two methods.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def search_rank(scores: np.ndarray, dist_to_leak: np.ndarray, cover: np.ndarray, max_areas: int = 10) -> int:
    """Greedy max-coverage search plan over `scores`; rank (1-based) of the first area whose centre is within
    100 m of the leak; max_areas + 1 if not found."""
    p = np.clip(np.asarray(scores, dtype=float), 0, None)
    if p.sum() <= 0:
        return max_areas + 1
    p = p / p.sum()
    for k in range(1, max_areas + 1):
        c = int(np.argmax(cover @ p))
        if dist_to_leak[c] <= 100:
            return k
        p = p * (1 - cover[c])
        if p.sum() <= 1e-12:
            break
        p = p / p.sum()
    return max_areas + 1


def bootstrap_ci(x, stat=np.mean, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(stat(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def risk_coverage(conf, correct):
    """Sort by confidence (high first); coverage = share acted on, precision = share correct among acted."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


def aurc(conf, correct):
    """Area under the risk(=1-precision)-coverage curve; lower is better."""
    cov, prec = risk_coverage(conf, correct)
    return float(np.mean(1 - prec))


def precision_at_coverage(conf, correct, coverage: float):
    cov, prec = risk_coverage(conf, correct)
    i = min(int(np.ceil(coverage * len(cov))) - 1, len(cov) - 1)
    return float(prec[max(i, 0)])


def mcnemar(a_correct, b_correct):
    """Exact McNemar test on paired binary outcomes. Returns (b_wins, a_wins, p-value)."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic map from raw confidence to empirical hit rate (fit on DEV, apply on TEST)."""

    def fit(self, conf, correct):
        order = np.argsort(conf)
        x, y = np.asarray(conf, float)[order], np.asarray(correct, float)[order]
        blocks = [[v, 1.0, x[i], x[i]] for i, v in enumerate(y)]       # value, weight, x_lo, x_hi
        merged = []
        for b in blocks:
            merged.append(b)
            while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
                v2, w2, lo2, hi2 = merged.pop()
                v1, w1, lo1, hi1 = merged.pop()
                merged.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, lo1, hi2])
        self.x = np.array([(b[2] + b[3]) / 2 for b in merged])
        self.y = np.array([b[0] for b in merged])
        return self

    def __call__(self, conf):
        return np.interp(np.asarray(conf, float), self.x, self.y)
