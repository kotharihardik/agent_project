"""Evaluation-only access to hidden truth (the ONLY package besides CaseEnv allowed to read hidden files)."""
from __future__ import annotations

import json
import math

import numpy as np

from ..env.network import NetworkInfo


class CaseTruth:
    def __init__(self, bench_dir: str, internal_id: str, net: NetworkInfo):
        self.h = json.load(open(f"{bench_dir}/hidden/{internal_id}.json"))
        self.net = net
        ev = self.h["ground_truth"]["events"]
        self.leak = ev[0] if ev else None
        self.suite = self.h["suite"]
        self.open_s = self.h["open_s"]
        self._dist = None

    @property
    def has_leak(self) -> bool:
        return self.leak is not None

    @property
    def visible(self) -> bool:
        return bool(self.leak and self.leak.get("surfaced"))

    def dist_to_leak_all(self) -> np.ndarray:
        """Pipe distance (m) from every junction to the true leak point."""
        if self._dist is None:
            L, f = self.leak["pipe_length_m"], self.leak["frac"]
            best = np.full(len(self.net.junctions), math.inf)
            for end, extra in ((self.leak["start_node"], f * L), (self.leak["end_node"], (1 - f) * L)):
                j = self.net.index.get(end)
                if j is not None:
                    best = np.minimum(best, self.net.pipe_distance[j] + extra)
            self._dist = best
        return self._dist

    def dist_to_leak(self, node: str) -> float:
        return float(self.dist_to_leak_all()[self.net.index[node]]) if self.has_leak else math.inf

    def public_gold(self) -> dict:
        """Gold reader labels for reports visible at case open (oracle-reader ablation only)."""
        return {r["report_id"]: r["gold"] for r in self.h["reports_all"] if r["received_s"] <= self.open_s}

    def strata(self) -> dict:
        ev = self.leak or {}
        q = ev.get("mean_flow_lps_after_start", 0.0)
        return {"suite": self.suite, "has_leak": self.has_leak, "visible": self.visible,
                "size_bin": ("none" if not self.has_leak else "<1" if q < 1 else "1-3" if q < 3 else "3-6" if q < 6 else ">6"),
                "leak_lps": q, "kind": ev.get("kind")}
