"""Belief state for active investigation: posterior over 'leak at junction j' + P(leak) + leak-size estimate.

Every new observation is a likelihood vector over junctions (plus a value under 'no leak'); the update is exact
Bayes on the discrete hypothesis set. All maths here is code, never an LLM (guard-rail 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Belief:
    post: np.ndarray                 # P(leak at j | leak), sums to 1
    p_leak: float
    q_hat: float                     # leak size estimate (L/s)
    q_sd: float
    cover: np.ndarray                # [J,J] 1 if pipe distance <= dig radius
    excluded: list = field(default_factory=list)   # dig spots already proven empty

    def copy(self):
        return Belief(self.post.copy(), self.p_leak, self.q_hat, self.q_sd, self.cover, list(self.excluded))

    def update(self, lik: np.ndarray, lik_none: float):
        """Bayes update with P(obs | leak at j) = lik[j] and P(obs | no leak) = lik_none."""
        joint = self.post * lik
        ev_leak = joint.sum()
        if ev_leak <= 0:
            return
        self.post = joint / ev_leak
        num = self.p_leak * ev_leak
        den = num + (1 - self.p_leak) * lik_none
        self.p_leak = float(num / den) if den > 0 else self.p_leak

    def coverage(self) -> np.ndarray:
        return self.cover @ self.post

    def best_dig(self) -> tuple[int, float]:
        cov = self.coverage()
        j = int(np.argmax(cov))
        return j, float(cov[j]) * self.p_leak

    def sample_leaks(self, rng, n: int) -> np.ndarray:
        return rng.choice(len(self.post), size=n, p=self.post)
