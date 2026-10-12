"""Simulated citizen who answers follow-up questions (TRUTH side - it knows what this person really saw).

The reporter only knows what a real person would know:
- where they saw the water (their 'seen' spot) and the street / landmarks around it,
- whether water was coming OUT there, or just collected there,
- if it was collected: sometimes, the direction it flowed from (uphill side).
It answers each part of a question it understands (several parts in one question are fine) and says
"I don't know" otherwise. Fake / trap reporters usually do not reply. Same simulator for every policy (fairness).
"""
from __future__ import annotations

import numpy as np

INTENTS = {
    "options": None,                                                         # handled when options are given
    "source": ["coming out", "coming from", "source", "bubbling", "collected", "collect", "flowing", "flow from",
               "where does the water come", "kahan se", "origin", "starting"],
    "street": ["street", "road", "which road", "lane", "cross", "name of the"],
    "landmark": ["nearest", "landmark", "shop", "building", "near what", "next to what", "near which", "identify"],
}


class ReporterSimulator:
    def __init__(self, net, places: dict, rng: np.random.Generator):
        self.net, self.rng = net, rng
        self.landmarks = [p for p in places.values() if p["type"] == "landmark"]
        self.lm_xy = np.array([net.xy[net.index[p["geometry"]["node"]]] for p in self.landmarks])
        self.street_of = {}
        for p in places.values():
            if p["type"] == "street":
                for n in p["geometry"]["nodes"]:
                    self.street_of.setdefault(n, p["name"])
        self.places = places

    # ------------------------------------------------------------------ helpers
    def _nearest_landmark(self, node: str, max_m: float = 150.0):
        d = np.hypot(*(self.lm_xy - self.net.xy[self.net.index[node]]).T)
        i = int(np.argmin(d))
        return self.landmarks[i]["name"] if d[i] <= max_m else None

    def _intents(self, q: str) -> list[str]:
        q = q.lower()
        return [k for k, words in INTENTS.items() if words and any(w in q for w in words)]

    # ------------------------------------------------------------------ main
    def answer(self, report: dict, leak: dict | None, question: str, options: list[dict]) -> str | None:
        kind = report["truth"]["kind"]
        if kind.startswith("adversarial"):
            return None if self.rng.random() < 0.8 else "Just send the crew, I already told you."
        seen = report["truth"]["seen_node"]
        parts = []
        if options:
            parts.append(self._answer_options(seen, options))
        for intent in self._intents(question):
            if intent == "source":
                parts.append(self._answer_source(report, leak))
            elif intent == "street":
                st = self.street_of.get(seen)
                parts.append(f"It is on {st}." if st else "I don't know the street name.")
            elif intent == "landmark":
                lm = self._nearest_landmark(seen)
                parts.append(f"It is near {lm}." if lm else "There is no big landmark nearby.")
        if not parts:
            return "Sorry, I don't understand. I just saw water there."
        return " ".join(parts)

    def _answer_options(self, seen: str, options: list[dict]) -> str:
        xy = self.net.xy[self.net.index[seen]]
        best, best_d = None, np.inf
        for k, o in enumerate(options):
            p = self.places.get(o.get("place_id"))
            if not p:
                continue
            g = p["geometry"]
            nodes = [g["node"]] if "node" in g else g.get("nodes", [])
            d = min((float(np.hypot(*(xy - self.net.xy[self.net.index[n]]))) for n in nodes if n in self.net.index),
                    default=np.inf)
            if d < best_d:
                best, best_d = k, d
        if best is not None and best_d <= 120 and self.rng.random() < 0.9:
            return f"Near {options[best]['label']}."
        return "Not near any of those, somewhere else." if self.rng.random() < 0.6 else "I don't know exactly."

    def _answer_source(self, report: dict, leak: dict | None) -> str:
        kind, src = report["truth"]["kind"], report["truth"]["is_source_seen"]
        if kind == "false":
            ft = report["truth"].get("false_type")
            return {"private_leak": "It is coming from a house pipe, not the road.",
                    "sewer": "It is dirty water from the gutter.",
                    "rain_or_drain": "It is drain water.", "rain_drain": "It is drain water."}.get(
                ft, "It is collected here, I don't know where it comes from.")
        if kind != "leak" or leak is None:
            return "I don't know where it comes from."
        if src == "yes":
            return "It is coming out of the ground right here."
        if self.rng.random() < 0.6 and leak.get("surface_node"):
            up = self._nearest_landmark(leak["surface_node"], 200) or self.street_of.get(leak["surface_node"])
            if up:
                return f"It is not coming out here, it is flowing down from the side of {up}."
        return "It is collected here, it seems to flow from somewhere uphill."
