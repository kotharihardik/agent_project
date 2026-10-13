"""B0 rule-based complaint reader: keywords + gazetteer name matching -> reader-1.0 JSON (docs/03).

This is the deliberately simple baseline the fine-tuned model must beat. It also lets Phase 2 run end-to-end before
the teammate's model exists. It never follows instructions in the text; it only flags them.
"""
from __future__ import annotations

import re

from .cues import find_cues
from .gazetteer_index import GazetteerIndex

SYMPTOM_RULES = [
    ("burst_gushing", r"burst|gushing|fountain|shooting up|phat gaya|fountain jaisa"),
    ("water_on_surface", r"coming out|leaking from|flowing out|flowing on the road|nikal raha|beh raha|aa raha hai.*paani|paani aa raha|zameen se paani|sadak pe paani|road se paani|road pe paani"),
    ("pooling_water", r"collected|puddle|standing|accumulat|jama|bhara hua"),
    ("wet_patch", r"wet patch|stays wet|geeli"),
    ("road_damage", r"sinking|caved|dhans"),
    ("low_pressure", r"low (water )?pressure|pressure (is )?(very )?(weak|low)|pressure bahut kam|dhaar"),
    ("no_water", r"no water|supply stopped|paani nahi aa raha"),
    ("sound_of_water", r"sound of (running )?water|awaaz"),
    ("dirty_water", r"dirty|muddy|brown water|ganda paani"),
]
NON_SUPPLY = [("sewage", r"gutter|sewer|sewage|badbu|bad smell|smell"),
              ("rain_or_drain", r"drain water|naale|rain")]
INSIDE = r"house pipe|ghar ki pipe|my house|inside (the|our) house|private tank|tank overflow"
INJECTION = r"system( note)?:|ignore (previous|all|the) (rules|instructions)|approval (already )?(given|granted)|operator has approved|no need for checks|dig (crew )?(at|to)|excavate|commissioner office"
UNSURE = r"not sure|i think|maybe|pata nahi|lagta hai"
PREV = r"already complained|kal bhi complaint|complained yesterday"
FLOW_DIR = r"(flowing down towards [^.,]+|[^.,]+ ki taraf beh raha hai)"
REL_BEFORE = [("opposite", r"opposite\s*$"), ("behind", r"behind\s*$"), ("in_front_of", r"in front of\s*$"),
              ("next_to", r"next to\s*$"), ("near", r"near\s*$"), ("on", r"\bon\s*$"), ("at", r"\bat\s*$")]
REL_AFTER = [("near", r"^\s*ke paas"), ("opposite", r"^\s*ke saamne"), ("behind", r"^\s*ke peeche"),
             ("in_front_of", r"^\s*ke aage"), ("next_to", r"^\s*ke bagal"), ("on", r"^\s*pe\b"),
             ("towards", r"^\s*ki taraf")]
HINGLISH = r"\b(hai|nahi|paani|ke paas|raha|bahut|kal|subah)\b"
ONSET = [("clock", r"since (\d{1,2})[:.](\d{2})|since (\d{1,2}) ?(am|pm)|(\d{1,2})[:.](\d{2}) (?:se|baje se)"),
         ("ago", r"(\d+) ?(?:hours?|hrs?|ghante) (?:ago|se)"),
         ("day_part", r"since (morning|afternoon|evening|night|yesterday)|(subah|raat|shaam|kal) se")]
DAY_PART = {"subah": "morning", "raat": "night", "shaam": "evening", "kal": "yesterday"}


def read_onset(low: str) -> dict:
    """'since 9:20' -> clock 09:20; '3 ghante se' -> ago 3h; 'subah se' -> day_part morning."""
    for kind, pat in ONSET:
        m = re.search(pat, low)
        if not m:
            continue
        g = [x for x in m.groups()]
        if kind == "clock":
            if g[0] is not None:
                h, mi = int(g[0]), int(g[1])
            elif g[2] is not None:
                h, mi = int(g[2]) % 12 + (12 if g[3] == "pm" else 0), 0
            else:
                h, mi = int(g[4]), int(g[5])
            val = f"{h:02d}:{mi:02d}"
        elif kind == "ago":
            val = f"{int(g[0])}h"
        else:
            w = g[0] or g[1]
            val = DAY_PART.get(w, w)
        return {"onset_text": m.group(0), "onset_kind": kind, "onset_value": val, "is_ongoing": True}
    return {"onset_text": None, "onset_kind": "unknown", "onset_value": None, "is_ongoing": None}


class RuleReader:
    name = "rule_b0"

    def __init__(self, gaz: GazetteerIndex):
        self.gaz = gaz

    def read(self, report: dict) -> dict:
        text = report["text"]
        low = text.lower()
        symptoms = [s for s, pat in SYMPTOM_RULES if re.search(pat, low)]
        reason = next((r for r, pat in NON_SUPPLY if re.search(pat, low)), None)
        supply = reason is None
        if not supply:
            symptoms = []
        inside = bool(re.search(INSIDE, low))
        injection = bool(re.search(INJECTION, low))
        flow_m = re.search(FLOW_DIR, text, flags=re.I)
        flow_txt = flow_m.group(1).strip() if flow_m else None
        mentions = []
        for surface, a, b, pids in self.gaz.find_mentions(text):
            if flow_m and flow_m.start() <= a < flow_m.end():
                continue    # the "towards X" place is where water GOES, not where it was seen
            rel = "unknown"
            for r, pat in REL_BEFORE:
                if re.search(pat, low[max(0, a - 14):a]):
                    rel = r
                    break
            for r, pat in REL_AFTER:
                if re.search(pat, low[b:b + 16]):
                    rel = r
                    break
            ptype = self.gaz.places[pids[0]]["type"]
            if injection and re.search(INJECTION, low[max(0, a - 60):a]):
                rel = "unknown"       # place named inside an injected instruction
            mentions.append({"text": surface, "type": ptype, "relation": rel})
        src = "unknown"
        if "pooling_water" in symptoms and not ({"water_on_surface", "burst_gushing"} & set(symptoms)):
            src = "no"
        elif {"water_on_surface", "burst_gushing"} & set(symptoms):
            src = "yes"
        types = {m["type"] for m in mentions}
        precision = ("street" if ("landmark" in types or "street" in types) else ("area" if "locality" in types else "none"))
        out = {
            "schema_version": "reader-1.0", "report_id": report["report_id"],
            "is_water_supply_issue": supply, "not_supply_reason": reason,
            "issue_scope": "inside_property" if inside else ("public_network" if symptoms else "unclear"),
            "symptoms": symptoms,
            "location": {"mentions": mentions,
                         "locality": next((m["text"] for m in mentions if m["type"] == "locality"), None),
                         "is_source_seen": src, "flow_direction_text": flow_txt, "precision": precision},
            "time": read_onset(low),
            "severity": "high" if "burst_gushing" in symptoms else ("medium" if "water_on_surface" in symptoms else "unknown"),
            "hazards": ["traffic_risk"] if re.search(r"traffic|vehicles|gaadiyan", low) else [],
            "reporter_certainty": "unsure" if re.search(UNSURE, low) else "certain",
            "mentions_previous_report": bool(re.search(PREV, low)),
            "contains_instructions": injection,
            "language": "hi-en" if re.search(HINGLISH, low) else "en",
            "summary_en": "",
        }
        out["summary_en"] = self._summary(out, mentions, find_cues(low))
        return out

    def _summary(self, out: dict, mentions: list, cues: list) -> str:
        """A short English summary built ONLY from fixed words and canonical place names (never copied text)."""
        if out["contains_instructions"]:
            return "message contains instruction-like text"
        names = []
        for m in mentions:
            ids = self.gaz.resolve(m["text"])
            if ids:
                names.append(f"{m['relation']} {self.gaz.places[ids[0]]['name']}")
        parts = [", ".join(s.replace("_", " ") for s in out["symptoms"]) or "no clear symptom"]
        if names:
            parts.append("place: " + "; ".join(names))
        if cues:
            parts.append("mentions " + ", ".join(cues))
        if out["time"]["onset_value"]:
            parts.append("since " + out["time"]["onset_value"])
        return " | ".join(parts)[:200]


    def read_reply(self, reply_text: str, option_labels: list[str] | None = None) -> dict:
        """Parse a reporter's answer to a follow-up question (reply mode).

        Returns: option_index (None / index / len(options) for 'none of those'), source_seen (yes/no/unknown),
        mentions (places where the water is), upstream (places it flows FROM), non_supply (not a public pipe),
        confirms_water (the person describes real water), contains_instructions (trap text).
        """
        low = (reply_text or "").lower()
        out = {"option_index": None, "source_seen": "unknown", "mentions": [], "upstream": [], "non_supply": False,
               "confirms_water": False, "contains_instructions": bool(re.search(INJECTION, low) or "send the crew" in low)}
        if not low:
            return out
        labels = option_labels or []
        for i, l in enumerate(labels):
            if l.lower() in low:
                out["option_index"] = i
                break
        if out["option_index"] is None and labels and re.search(r"not near any|somewhere else|none of", low):
            out["option_index"] = len(labels)
        if re.search(r"house pipe|gutter|drain water|not the road", low):
            out["non_supply"] = True
            return out
        if re.search(r"coming out of the ground|coming out here|bubbling", low):
            out["source_seen"] = "yes"
            out["confirms_water"] = True
        if re.search(r"not coming out|collected here|flowing down from|flow from somewhere", low):
            out["source_seen"] = "no"
            out["confirms_water"] = True
        up = re.search(r"flowing down from the side of ([^.]+)", reply_text, flags=re.I)
        for surface, a, b, pids in self.gaz.find_mentions(reply_text):
            m = {"text": surface, "type": self.gaz.places[pids[0]]["type"], "relation": "near"}
            if up and up.start(1) <= a < up.end(1):
                out["upstream"].append(m)
            elif not (labels and surface.lower() in [x.lower() for x in labels]):
                out["mentions"].append(m)
        return out


class OracleReader:
    """Upper-bound ablation ONLY: returns the gold label. Lives here for symmetry, but gold is supplied by the
    evaluation harness from hidden files - engine/baseline code must never import hidden data itself."""
    name = "oracle_gold"

    def __init__(self, gold_by_id: dict):
        self.gold = gold_by_id

    def read(self, report: dict) -> dict:
        return self.gold[report["report_id"]]
