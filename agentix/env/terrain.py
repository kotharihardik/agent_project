"""Where leak water goes after it leaves the pipe: trench travel, surfacing point, overland flow downhill, pooling.

Surface graph = pipe neighbours (streets) + spatial neighbours within `radius_m` (water can cross open ground).
Routing is a random walk that only moves downhill, weighted by slope; it stops at a local low point (pool)
or when the travel budget (which grows with leak flow) is used up.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .network import NetworkInfo


class Terrain:
    def __init__(self, net: NetworkInfo, radius_m: float = 60.0):
        self.net = net
        self.elev = net.elevation
        self.xy = net.xy
        n = len(self.elev)
        nbrs = [set() for _ in range(n)]
        for u, v in net.pipe_graph.edges:
            iu, iv = net.index.get(u), net.index.get(v)
            if iu is not None and iv is not None:
                nbrs[iu].add(iv)
                nbrs[iv].add(iu)
        tree = cKDTree(self.xy)
        for i, js in enumerate(tree.query_ball_point(self.xy, r=radius_m)):
            for j in js:
                if j != i:
                    nbrs[i].add(j)
        self.nbrs = [np.array(sorted(s), dtype=int) for s in nbrs]
        self.is_sink = np.array([not np.any(self.elev[nb] < self.elev[i] - 0.05) if len(nb) else True
                                 for i, nb in enumerate(self.nbrs)])

    def _dist(self, i, j):
        return float(np.hypot(*(self.xy[i] - self.xy[j])))

    def trench_travel(self, start: int, rng: np.random.Generator, mean_m: float = 20.0, cap_m: float = 80.0) -> int:
        """Water often runs along the pipe bedding before it surfaces: move downhill along pipes."""
        budget = min(rng.exponential(mean_m), cap_m)
        cur, travelled = start, 0.0
        pipe_nbrs = lambda i: [self.net.index[w] for w in self.net.pipe_graph.neighbors(self.net.junctions[i])
                               if w in self.net.index]
        while travelled < budget:
            cand = [j for j in pipe_nbrs(cur) if self.elev[j] < self.elev[cur] - 0.05]
            if not cand:
                break
            nxt = min(cand, key=lambda j: self.elev[j])
            d = self._dist(cur, nxt)
            if travelled + d > budget:
                break
            travelled += d
            cur = nxt
        return cur

    def overland_path(self, start: int, rng: np.random.Generator, travel_m: float):
        """Downhill random walk. Returns list of (node_index, cumulative distance m); last node is where water pools."""
        path = [(start, 0.0)]
        cur, travelled, visited = start, 0.0, {start}
        while travelled < travel_m:
            nb = self.nbrs[cur]
            drops = self.elev[cur] - self.elev[nb]
            ok = (drops > 0.05) & np.array([j not in visited for j in nb])
            if not ok.any():
                break
            nb, drops = nb[ok], drops[ok]
            d = np.array([self._dist(cur, j) for j in nb])
            w = drops / d
            nxt = int(rng.choice(nb, p=w / w.sum()))
            step = self._dist(cur, nxt)
            if travelled + step > travel_m:
                break
            travelled += step
            cur = nxt
            visited.add(cur)
            path.append((cur, travelled))
        return path
