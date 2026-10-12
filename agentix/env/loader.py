"""The ONLY way localization methods should read a case: public data, restricted to one sensor layout.

Never import hidden files in method code. The environment (`CaseEnv`) and scorers are the only hidden-data readers.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class PublicCase:
    case_id: str                 # opaque public id
    internal_dir: str            # path (harness use only; never show to an LLM)
    case_open_s: float
    start_datetime: str
    pattern_start_s: int
    history_days: int
    time_s: np.ndarray           # [T]
    pressure: np.ndarray         # [T, k] for the layout sensors only
    pressure_nodes: list[str]
    flow: np.ndarray             # [T, n_flow] m3/h
    flow_links: list[str]
    tank_level: np.ndarray
    reports: list[dict]          # raw public reports (untrusted text!)


def list_cases(bench_dir: str, prefix: str = "") -> list[str]:
    return sorted(os.path.basename(d) for d in glob.glob(f"{bench_dir}/cases/{prefix}*"))


def load_public_case(bench_dir: str, internal_id: str, layout: list[str]) -> PublicCase:
    d = f"{bench_dir}/cases/{internal_id}"
    case = json.load(open(f"{d}/case.json"))
    obs = np.load(f"{d}/obs.npz")
    nodes = list(obs["pressure_nodes"])
    missing = [n for n in layout if n not in nodes]
    if missing:
        raise ValueError(f"layout sensors not available in this case: {missing}")
    cols = [nodes.index(n) for n in layout]
    return PublicCase(case_id=case["case_id"], internal_dir=d, case_open_s=case["case_open_s"],
                      start_datetime=case["start_datetime"], pattern_start_s=case["pattern_start_s"],
                      history_days=case["history_days"], time_s=obs["time_s"], pressure=obs["pressure"][:, cols],
                      pressure_nodes=list(layout), flow=obs["flow"], flow_links=list(obs["flow_links"]),
                      tank_level=obs["tank_level"], reports=case["reports"])


def load_layouts(bench_dir: str) -> dict[str, list[str]]:
    return json.load(open(f"{bench_dir}/identifiability.json"))["layouts"]
