"""Gazetteer index: find place names in text and turn a place mention into a distribution over network nodes.

Public knowledge only (a utility knows its own streets and landmarks). Used by the rule-based reader (B0) and by
the inference engine to resolve mentions coming from ANY reader (rule-based, fine-tuned, or oracle gold labels).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

_SUFFIX = {"road": "rd", "street": "st", "rd.": "rd", "st.": "st", "lane": "ln"}
_LONG = {"rd": "road", "st": "street", "ln": "lane"}


def norm(s: str) -> str:
    s = s.lower().replace("'", "").replace(",", " ")
    s = re.sub(r"[^a-z0-9ऀ-෿ ]+", " ", s)
    toks = [_SUFFIX.get(t, t) for t in s.split()]
    return " ".join(toks)


class GazetteerIndex:
    def __init__(self, gazetteer: dict, net):
        self.places = {p["place_id"]: p for p in gazetteer["places"]}
        self.net = net
        self.alias_map: dict[str, list[str]] = {}
        for pid, p in self.places.items():
            for a in [p["name"]] + list(p.get("aliases", [])):
                k = norm(a)
                if len(k) >= 3:
                    self.alias_map.setdefault(k, [])
                    if pid not in self.alias_map[k]:
                        self.alias_map[k].append(pid)
        # longest aliases first so "2nd cross canal side" wins over "2nd cross"
        self.alias_keys = sorted(self.alias_map, key=len, reverse=True)
        self._node_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ text -> mentions
    def find_mentions(self, text: str) -> list[tuple[str, int, int, list[str]]]:
        """Return (surface_text, start, end, place_ids) for gazetteer names found in raw text (non-overlapping)."""
        low = text.lower()
        taken = np.zeros(len(low) + 1, dtype=bool)
        out = []
        for key in self.alias_keys:
            words = key.split()
            # a stored "rd" must also match "road" in the text (and st/street, ln/lane)
            alts = [f"(?:{re.escape(w)}|{_LONG[w]})" if w in _LONG else re.escape(w) for w in words]
            pattern = r"\b" + r"[\s,\.']*".join(alts) + r"\.?\b"
            for m in re.finditer(pattern, low):
                if taken[m.start():m.end()].any():
                    continue
                taken[m.start():m.end()] = True
                out.append((text[m.start():m.end()], m.start(), m.end(), list(self.alias_map[key])))
        return sorted(out, key=lambda x: x[1])

    # ------------------------------------------------------------------ mention -> places
    def resolve(self, mention_text: str, mention_type: str | None = None, fuzzy: float = 0.85) -> list[str]:
        k = norm(mention_text)
        if k in self.alias_map:
            ids = self.alias_map[k]
        else:
            best, ids = 0.0, []
            for key in self.alias_keys:
                r = SequenceMatcher(None, k, key).ratio()
                if r > best:
                    best, ids = r, list(self.alias_map[key])
            if best < fuzzy:
                return []
        if mention_type in ("street", "landmark", "locality"):
            typed = [i for i in ids if self.places[i]["type"] == mention_type]
            ids = typed or ids
        return ids

    # ------------------------------------------------------------------ places -> node weights
    def place_nodes(self, place_id: str, landmark_sigma_m: float = 40.0) -> np.ndarray:
        """Normalized weight over junctions for 'somewhere at this place'."""
        if place_id in self._node_cache:
            return self._node_cache[place_id]
        p = self.places[place_id]
        g = p.get("geometry", {})
        w = np.zeros(len(self.net.junctions))
        if "node" in g:      # landmark: soft disc around its node (people say 'near')
            c = self.net.xy[self.net.index[g["node"]]]
            d = np.hypot(*(self.net.xy - c).T)
            w = np.exp(-0.5 * (d / landmark_sigma_m) ** 2)
        else:
            for n in g.get("nodes", []):
                if n in self.net.index:
                    w[self.net.index[n]] = 1.0
        s = w.sum()
        w = w / s if s > 0 else w
        self._node_cache[place_id] = w
        return w

    def mention_nodes(self, mention_text: str, mention_type: str | None = None) -> np.ndarray | None:
        ids = self.resolve(mention_text, mention_type)
        if not ids:
            return None
        w = sum(self.place_nodes(i) for i in ids) / len(ids)   # ambiguous names -> mixture over all matches
        return w

    # ------------------------------------------------------------------ node -> names (public map knowledge)
    def _node_tables(self):
        if hasattr(self, "_street_by_node"):
            return
        self._street_by_node = {}
        for p in self.places.values():
            if p["type"] == "street":
                for n in p.get("geometry", {}).get("nodes", []):
                    self._street_by_node.setdefault(n, p)
        self._lms = [p for p in self.places.values() if p["type"] == "landmark"]
        self._lm_xy = np.array([self.net.xy[self.net.index[p["geometry"]["node"]]] for p in self._lms])

    def node_street(self, node_idx: int):
        """The named street a junction lies on (or None)."""
        self._node_tables()
        return self._street_by_node.get(self.net.junctions[node_idx])

    def node_landmark(self, node_idx: int, max_m: float = 150.0):
        """The closest landmark to a junction within max_m metres (or None)."""
        self._node_tables()
        d = np.hypot(*(self._lm_xy - self.net.xy[node_idx]).T)
        i = int(np.argmin(d))
        return self._lms[i] if d[i] <= max_m else None
