"""Hydraulic truth: build a 'true' model (official Real .inp or perturbed nominal), insert leaks, simulate.

Design (see docs/decisions/ADR-001):
- Leaks are inserted like BattLeDIM: split the pipe and put the leak on a new node.
  Abrupt leak  -> EPANET emitter  q = Cd*A*sqrt(2 g h)  (pressure-dependent orifice).
  Incipient    -> demand ramp up to the orifice flow at the local pressure (like BattLeDIM).
- The truth model is never the inference model (no inverse crime).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import wntr

G = 9.81
CD = 0.75


@dataclass
class LeakSpec:
    pipe: str
    frac: float            # position along the pipe (0..1) where the leak node is placed
    diameter_m: float      # orifice diameter
    kind: str              # "abrupt" | "incipient"
    start_s: int           # seconds from simulation start
    ramp_s: int = 0        # incipient: time to reach full size

    def to_dict(self):
        return asdict(self)


def orifice_coeff(diameter_m: float) -> float:
    """Emitter coefficient (m3/s per sqrt(m)) for an orifice of this diameter."""
    area = np.pi * (diameter_m / 2) ** 2
    return CD * area * np.sqrt(2 * G)


def perturb_model(wn, rng: np.random.Generator, rough_pct=0.10, diam_pct=0.05, demand_pct=0.10):
    """Turn the nominal model into a plausible 'real' model (used until/unless the official Real .inp is used).

    README of BattLeDIM: real parameters differ from nominal by <=10 %.
    """
    for name in wn.pipe_name_list:
        p = wn.get_link(name)
        p.roughness *= float(rng.uniform(1 - rough_pct, 1 + rough_pct))
        p.diameter *= float(rng.uniform(1 - diam_pct, 1 + diam_pct))
    for name in wn.junction_name_list:
        for d in wn.get_node(name).demand_timeseries_list:
            d.base_value *= float(rng.normal(1.0, demand_pct))
    return wn


def add_demand_noise(wn, rng: np.random.Generator, n_steps: int, step_s: int, start_step: int = 0,
                     sigma=0.08, rho=0.97):
    """Give every junction its own AR(1)-noisy copy of its demand pattern (real demands are never the nominal profile).

    start_step shifts the weekly nominal pattern so the window can start on any weekday/time.
    """
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        new_list = []
        for k, d in enumerate(list(j.demand_timeseries_list)):
            if d.base_value == 0:
                continue
            base_pat = wn.get_pattern(d.pattern_name).multipliers if d.pattern_name else np.ones(1)
            base_pat = np.asarray(base_pat, dtype=float)
            idx = (start_step + np.arange(n_steps)) % len(base_pat)
            eps = np.empty(n_steps)
            eps[0] = rng.normal(0, sigma)
            shocks = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n_steps)
            for t in range(1, n_steps):
                eps[t] = rho * eps[t - 1] + shocks[t]
            mult = np.clip(base_pat[idx] * (1 + eps), 0, None)
            pname = f"PN_{jname}_{k}"[:31]
            wn.add_pattern(pname, mult.tolist())
            new_list.append((d.base_value, pname, d.category))
        j.demand_timeseries_list.clear()
        for base, pname, cat in new_list:
            j.add_demand(base, pname, cat)
    wn.options.time.pattern_timestep = step_s
    return wn


def insert_leak(wn, leak: LeakSpec, n_steps: int, step_s: int, local_pressure_m: float = 45.0) -> str:
    """Split the pipe and attach the leak to a new node. Returns the leak node name."""
    node = f"{leak.pipe}_leaknode"
    wntr.morph.split_pipe(wn, leak.pipe, f"{leak.pipe}_B", node, split_at_point=leak.frac, return_copy=False)
    j = wn.get_node(node)
    coeff = orifice_coeff(leak.diameter_m)
    if leak.kind == "abrupt":
        # EPANET emitter switched on at start time via a pattern-free trick: emitter from t=0 but leak node is
        # connected through a pipe that is closed until start. Simpler + exact: use emitter and a control on a
        # zero-length "leak pipe" to an emitter node.
        tap = f"{leak.pipe}_leaktap"
        wn.add_junction(tap, base_demand=0.0, elevation=j.elevation, coordinates=j.coordinates)
        wn.add_pipe(f"{leak.pipe}_leakpipe", node, tap, length=1.0, diameter=0.5, roughness=140,
                    initial_status="Closed")
        wn.get_node(tap).emitter_coefficient = coeff
        act = wntr.network.controls.ControlAction(wn.get_link(f"{leak.pipe}_leakpipe"), "status",
                                                  wntr.network.base.LinkStatus.Open)
        cond = wntr.network.controls.SimTimeCondition(wn, "=", leak.start_s)
        wn.add_control(f"leakstart_{leak.pipe}", wntr.network.controls.Control(cond, act))
    else:
        q_full = coeff * np.sqrt(max(local_pressure_m, 1.0))
        t = np.arange(n_steps) * step_s
        ramp = np.clip((t - leak.start_s) / max(leak.ramp_s, step_s), 0, 1)
        ramp[t < leak.start_s] = 0
        pname = f"LEAK_{leak.pipe}"[:31]
        wn.add_pattern(pname, ramp.tolist())
        j.add_demand(q_full, pname, "leak")
    return node


def simulate(wn, prefix_dir: str | None = None):
    prefix_dir = prefix_dir or tempfile.mkdtemp(prefix="agx_")
    prefix = os.path.join(prefix_dir, "sim")
    return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)


def summarize_results(res, junctions: list[str], keep_nodes: list[str], keep_links: list[str],
                      step_s: int, leak_node: str | None, tap_node: str | None):
    """Compress a simulation into what cases need (keeps files small)."""
    p = res.node["pressure"]
    idx = p.index.values
    day = (idx // 86400).astype(int)
    hour = (idx % 86400) / 3600
    night = (hour >= 2) & (hour < 4)
    pj = p[junctions]
    night_mean = pj[night].groupby(day[night]).mean()
    day_min = pj.groupby(day).min()
    out = {
        "time_s": idx.astype(np.int64),
        "sensor_pressure": p[keep_nodes].astype(np.float32),
        "link_flow": (res.link["flowrate"][keep_links] * 3600).astype(np.float32),   # m3/h
        "night_mean_pressure": night_mean.astype(np.float32),                         # [days, junctions]
        "day_min_pressure": day_min.astype(np.float32),
    }
    if leak_node is not None:
        d = res.node["demand"]
        leak = d[tap_node] if tap_node in d.columns else d[leak_node] - 0  # incipient: leak in node demand
        out["leak_flow_lps"] = (leak * 1000).astype(np.float32)
    return out
