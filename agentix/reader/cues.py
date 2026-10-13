"""Closed vocabularies that are safe to pass from untrusted text to the planner.

The planner (the privileged LLM) never reads citizen text. It only sees words from these fixed lists, canonical
place names from the gazetteer, and times that match a strict pattern. Even a cleverly written message can then
only say "fire brigade" or "hydrant" - it cannot smuggle an instruction through.
"""
from __future__ import annotations

import re

# hints that the water might come from a planned or emergency use, not a leak
CUES = [
    ("fire brigade", r"fire ?brigade|fire truck|fire engine|damkal"),
    ("hydrant", r"hydrant"),
    ("tanker filling", r"tanker"),
    ("flushing", r"flush"),
    ("valve work", r"valve"),
    ("work crew", r"\bcrew\b|workers|repair work|road ?works?|khudai|digging"),
]
TIME_OK = re.compile(r"^(\d{1,2}:\d{2}|\d{1,3}h|morning|afternoon|evening|night|yesterday)$")


def find_cues(text: str) -> list[str]:
    low = (text or "").lower()
    return [name for name, pat in CUES if re.search(pat, low)]


def safe_time(value) -> str | None:
    """Only times like '09:20', '3h', 'morning' get through."""
    v = str(value or "").strip().lower()
    return v if TIME_OK.match(v) else None
