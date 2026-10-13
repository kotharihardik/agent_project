"""Step 3: citizen reports -> log-likelihood for "leak at junction j".

Independent of the report GENERATOR (no inverse crime, see ADR-003/ADR-006): different structure and parameters.
  P(report r | leak at j) = (1-eps_r) * sum_n K_r(j,n) w_r(n)  +  eps_r * sum_n B(n) w_r(n)
    w_r(n)  : where the report points (gazetteer resolution of its place words)
    K_r(j,.) : where water from a leak at j is SEEN = mixture of
               near-source kernel (exp(-d/lambda_near))  and
               downhill kernel (deterministic steepest-descent path from j, exp(-path/lambda_down))
               mixed by the reader's is_source_seen (yes: mostly near; no: mostly downhill)
    B(n)     : where unrelated reports come from (people ~ demand)
    eps_r    : probability the report is unrelated/false (base rate, raised for weak or non-visual reports)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..env.network import NetworkInfo
from ..reader.gazetteer_index import GazetteerIndex


class ReportModel:
    def __init__(self, net: NetworkInfo, gaz: GazetteerIndex, lambda_near_m: float = 40.0,
                 lambda_down_m: float = 100.0, max_down_m: float = 700.0, surface_radius_m: float = 60.0,
                 downhill: bool = True):
        self.net, self.gaz = net, gaz
        xy, elev = net.xy, net.elevation
        J = len(xy)
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        self.k_near = np.exp(-d / lambda_near_m)
        self.k_near /= self.k_near.sum(1, keepdims=True)
        # steepest-descent next node on the surface graph (pipe neighbours + neighbours within radius)
        tree = cKDTree(xy)
        nbrs = [set(js) - {i} for i, js in enumerate(tree.query_ball_point(xy, r=surface_radius_m))]
        for u, v in net.pipe_graph.edges:
            iu, iv = net.index.get(u), net.index.get(v)
            if iu is not None and iv is not None:
                nbrs[iu].add(iv)
                nbrs[iv].add(iu)
        nxt = np.full(J, -1)
        for i in range(J):
            cand = [j for j in nbrs[i] if elev[j] < elev[i] - 0.05]
            if cand:
                nxt[i] = max(cand, key=lambda j: (elev[i] - elev[j]) / max(d[i, j], 1.0))
        spread = np.exp(-0.5 * (d / 30.0) ** 2)
        k_down = np.zeros((J, J))
        for j in range(J):
            cur, dist, row = j, 0.0, np.zeros(J)
            row[j] = 1.0
            while nxt[cur] >= 0 and dist < max_down_m:
                dist += d[cur, nxt[cur]]
                cur = nxt[cur]
                row[cur] += np.exp(-dist / lambda_down_m)
            k_down[j] = row @ spread
        self.k_down = k_down / k_down.sum(1, keepdims=True)
        if not downhill:                       # ablation: ignore that water runs downhill
            self.k_down = self.k_near.copy()
        dem = net.base_demand / net.base_demand.mean()
        self.background = (0.3 + dem) / (0.3 + dem).sum()

    def location_weights(self, reader_out: dict) -> np.ndarray | None:
        ws = []
        for m in reader_out["location"]["mentions"]:
            if m["relation"] == "unknown" and reader_out.get("contains_instructions"):
                continue
            w = self.gaz.mention_nodes(m["text"], m["type"])
            if w is not None and w.sum() > 0:
                ws.append(w)
        if not ws:
            return None
        w = ws[0]
        for w2 in ws[1:]:
            # two place words that agree (e.g. a landmark ON a street) -> intersect; if they don't overlap -> either
            cos = float(w @ w2 / np.sqrt((w @ w) * (w2 @ w2) + 1e-300))
            w = w * w2 if cos > 0.05 else 0.5 * (w + w2)
            w = w / w.sum()
        return w

    def report_eps(self, r: dict, base_eps: float, age_h: float | None = None, tau_age_h: float = 24.0) -> float | None:
        """Probability this report is unrelated to the leak; None = ignore the report entirely."""
        if r.get("contains_instructions"):
            return None                                  # quarantined: never used as evidence
        if not r["is_water_supply_issue"]:
            return None
        if r["issue_scope"] == "inside_property":
            return 0.95
        visual = {"water_on_surface", "burst_gushing", "pooling_water", "wet_patch", "road_damage", "sound_of_water"}
        if not (set(r["symptoms"]) & visual):
            return 0.98                                  # tap-pressure complaints carry ~no leak signal here
        eps = base_eps
        if r["reporter_certainty"] == "unsure":
            eps = 1 - (1 - eps) * 0.7
        if age_h is not None:      # recency: a report from long before the case opened is rarely about this leak
            eps = 1 - (1 - eps) * float(np.exp(-max(age_h, 0.0) / tau_age_h))
        return eps

    def loglik(self, reader_outs: list[dict], base_eps: float = 0.6, ages_h: dict | None = None,
               tau_age_h: float = 24.0) -> tuple[np.ndarray, float, list]:
        J = len(self.background)
        ll = np.zeros(J)
        ll0 = 0.0
        used = []
        for r in reader_outs:
            age = (ages_h or {}).get(r["report_id"])
            eps = self.report_eps(r, base_eps, age, tau_age_h)
            if eps is None:
                continue
            w = self.location_weights(r)
            if w is None:
                continue
            src = r["location"]["is_source_seen"]
            a = {"yes": 0.85, "no": 0.25}.get(src, 0.6)
            k = a * (self.k_near @ w) + (1 - a) * (self.k_down @ w)       # P(seen at a place matching w | leak j)
            bg = float(self.background @ w)
            lik = (1 - eps) * k + eps * bg
            ll += np.log(lik + 1e-300)
            ll0 += np.log(bg + 1e-300)
            used.append(r["report_id"])
        return ll, ll0, used
