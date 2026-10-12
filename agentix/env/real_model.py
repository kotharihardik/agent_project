"""Official BattLeDIM truth model (L-TOWN_Real.inp) without loading 177 MB into WNTR each time.

The Real file = the nominal network with slightly different pipe parameters and 108 one-year demand patterns
(5-min steps, 105,120 values each). We parse it ONCE into a compact cache (~45 MB), then build a per-case truth
model by copying the Real pipe parameters and the Real demands for just the case's time window into the nominal
model. See docs/decisions/ADR-005.
"""
from __future__ import annotations

import re

import numpy as np
import wntr

CMH_TO_M3S = 1.0 / 3600.0


def _sections(path: str):
    """Yield (section_name, line) for every non-comment, non-empty data line."""
    sec = None
    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                sec = line.strip("[]").upper()
                continue
            yield sec, line.split(";")[0].split()


def parse_real_inp(path: str) -> dict:
    pipes, demands, patterns, elev, emitters, tanks, valves = {}, [], {}, {}, {}, {}, {}
    for sec, tok in _sections(path):
        if not tok:
            continue
        if sec == "PIPES" and len(tok) >= 6:
            pipes[tok[0]] = (float(tok[3]), float(tok[4]), float(tok[5]))       # length m, diameter mm, roughness
        elif sec == "JUNCTIONS" and len(tok) >= 2:
            elev[tok[0]] = float(tok[1])
        elif sec == "DEMANDS" and len(tok) >= 2:
            demands.append((tok[0], float(tok[1]), tok[2] if len(tok) > 2 else ""))
        elif sec == "PATTERNS":
            patterns.setdefault(tok[0], []).extend(float(x) for x in tok[1:])
        elif sec == "EMITTERS" and len(tok) >= 2:
            emitters[tok[0]] = float(tok[1])
        elif sec == "TANKS" and len(tok) >= 6:
            tanks[tok[0]] = float(tok[5])                                     # diameter m
        elif sec == "VALVES" and len(tok) >= 4:
            valves[tok[0]] = float(tok[3])                                    # diameter mm
    names = sorted(patterns)
    length = max(len(v) for v in patterns.values())
    mat = np.ones((len(names), length), dtype=np.float32)
    for i, n in enumerate(names):
        v = np.asarray(patterns[n], dtype=np.float32)
        mat[i] = np.resize(v, length)          # short patterns (weekly) are tiled to one year
    return {"pipes": pipes, "demands": demands, "pattern_names": names, "patterns": mat, "elevation": elev,
            "emitters": emitters, "tanks": tanks, "valves": valves}


def save_cache(parsed: dict, out_path: str):
    p_ids = sorted(parsed["pipes"])
    pv = np.array([parsed["pipes"][p] for p in p_ids], dtype=np.float64)
    d_j = np.array([d[0] for d in parsed["demands"]])
    d_b = np.array([d[1] for d in parsed["demands"]], dtype=np.float64)
    pidx = {n: i for i, n in enumerate(parsed["pattern_names"])}
    d_p = np.array([pidx.get(d[2], -1) for d in parsed["demands"]], dtype=np.int32)
    np.savez_compressed(out_path, pipe_ids=np.array(p_ids), pipe_length_m=pv[:, 0], pipe_diameter_mm=pv[:, 1],
                        pipe_roughness=pv[:, 2], demand_junction=d_j, demand_base_cmh=d_b, demand_pattern=d_p,
                        pattern_names=np.array(parsed["pattern_names"]), patterns=parsed["patterns"],
                        tank_ids=np.array(sorted(parsed["tanks"])),
                        tank_diameter_m=np.array([parsed["tanks"][t] for t in sorted(parsed["tanks"])]),
                        valve_ids=np.array(sorted(parsed["valves"])),
                        valve_diameter_mm=np.array([parsed["valves"][v] for v in sorted(parsed["valves"])]))


class RealTruth:
    """Builds per-case truth models from the cached official Real model."""

    def __init__(self, cache_path: str):
        z = np.load(cache_path)
        self.pipe = {pid: (float(d), float(r)) for pid, d, r in
                     zip(z["pipe_ids"], z["pipe_diameter_mm"], z["pipe_roughness"])}
        self.d_j, self.d_b, self.d_p = z["demand_junction"], z["demand_base_cmh"], z["demand_pattern"]
        self.pattern_names = list(z["pattern_names"])
        self.patterns = z["patterns"]                       # [P, 105120] one year at 5 min
        self.year_steps = self.patterns.shape[1]
        self.tank = {t: float(d) for t, d in zip(z["tank_ids"], z["tank_diameter_m"])}
        self.valve = {v: float(d) for v, d in zip(z["valve_ids"], z["valve_diameter_mm"])}

    def build(self, nominal_inp: str, start_step: int, n_steps: int, step_s: int = 300):
        """Nominal structure + Real pipe parameters + Real demands for [start_step, start_step + n_steps)."""
        wn = wntr.network.WaterNetworkModel(nominal_inp)
        for pid, (d_mm, rough) in self.pipe.items():
            if pid in wn.pipe_name_list:
                link = wn.get_link(pid)
                link.diameter = d_mm / 1000.0
                link.roughness = rough
        for t, d in self.tank.items():          # tank size and valve sizes also differ (found in calibration)
            wn.get_node(t).diameter = d
        for v, d in self.valve.items():
            wn.get_link(v).diameter = d / 1000.0
        idx = (start_step + np.arange(n_steps)) % self.year_steps
        used = sorted({int(p) for p in self.d_p if p >= 0})
        for p in used:
            wn.add_pattern(f"RP{p}", self.patterns[p, idx].astype(float).tolist())
        for jname in wn.junction_name_list:
            wn.get_node(jname).demand_timeseries_list.clear()
        for j, base, p in zip(self.d_j, self.d_b, self.d_p):
            if base == 0 or j not in wn.junction_name_list:
                continue
            wn.get_node(str(j)).add_demand(float(base) * CMH_TO_M3S, f"RP{int(p)}" if p >= 0 else None)
        wn.options.time.pattern_timestep = step_s
        wn.options.time.pattern_start = 0
        return wn
