"""Leak signature library on the NOMINAL model (the inference side).

For every candidate junction j and leak size s we store the mean pressure change at every junction
during fixed windows (night, day), plus inlet-flow changes. Leak = emitter sized to give s L/s at
the node's no-leak pressure.
"""
from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import wntr

_CFG = {}


def _window_means(res, windows_h, day_index=1):
    p = res.node["pressure"]
    idx = p.index.values
    out = {}
    for name, (h0, h1) in windows_h.items():
        t0, t1 = (day_index * 24 + h0) * 3600, (day_index * 24 + h1) * 3600
        m = (idx >= t0) & (idx < t1)
        out[name] = p[m].mean()
    return out


def _load(cfg):
    wn = wntr.network.WaterNetworkModel(cfg["nominal_inp"])
    wn.options.time.duration = int(cfg["signatures"]["duration_h"] * 3600)
    wn.options.time.report_timestep = 900
    wn.options.time.pattern_start = int(cfg["signatures"]["weekday_offset_days"] * 86400)
    return wn


def _init(cfg):
    _CFG.update(cfg)


def _run_one(args):
    node, q_lps, p_ref = args
    cfg = _CFG
    wn = _load(cfg)
    wn.get_node(node).emitter_coefficient = (q_lps / 1000.0) / np.sqrt(max(p_ref, 1.0))
    d = tempfile.mkdtemp(prefix="agxsig_")
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "s"))
    w = _window_means(res, cfg["signatures"]["windows_h"])
    juncs = list(wn.junction_name_list)
    flows = {}
    for name, (h0, h1) in cfg["signatures"]["windows_h"].items():
        f = res.link["flowrate"][cfg["flow_links"]]
        idx = f.index.values
        m = (idx >= (24 + h0) * 3600) & (idx < (24 + h1) * 3600)
        flows[name] = (f[m].mean() * 3600).values
    return node, q_lps, {k: v[juncs].values for k, v in w.items()}, flows


def build_library(cfg: dict, out_path: str, workers: int = 8):
    wn = _load(cfg)
    junctions = list(wn.junction_name_list)
    d = tempfile.mkdtemp(prefix="agxsig_")
    base = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "base"))
    base_w = _window_means(base, cfg["signatures"]["windows_h"])
    base_flow = {}
    for name, (h0, h1) in cfg["signatures"]["windows_h"].items():
        f = base.link["flowrate"][cfg["flow_links"]]
        idx = f.index.values
        m = (idx >= (24 + h0) * 3600) & (idx < (24 + h1) * 3600)
        base_flow[name] = (f[m].mean() * 3600).values
    p_ref = base_w["night"]
    sizes = cfg["signatures"]["sizes_lps"]
    jobs = [(n, q, float(p_ref[n])) for n in junctions for q in sizes]
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg,)) as ex:
        results = list(ex.map(_run_one, jobs, chunksize=8))
    print(f"signature library: {len(jobs)} simulations in {time.time() - t0:.0f} s")
    for name, (h0, h1) in cfg["signatures"]["windows_h"].items():
        if (24 + h1) * 3600 > cfg["signatures"]["duration_h"] * 3600:
            raise ValueError(f"window {name} ends after the simulation; increase signatures.duration_h")

    jidx = {n: i for i, n in enumerate(junctions)}
    windows = list(cfg["signatures"]["windows_h"])
    nJ, nS = len(junctions), len(sizes)
    dp = {w: np.zeros((nS, nJ, nJ), dtype=np.float32) for w in windows}
    dq = {w: np.zeros((nS, nJ, len(cfg["flow_links"])), dtype=np.float32) for w in windows}
    for node, q, w_means, flows in results:
        s, j = sizes.index(q), jidx[node]
        for w in windows:
            dp[w][s, j, :] = base_w[w][junctions].values - w_means[w]      # positive = pressure drop
            dq[w][s, j, :] = flows[w] - base_flow[w]
    np.savez_compressed(
        out_path, junctions=np.array(junctions), sizes_lps=np.array(sizes), flow_links=np.array(cfg["flow_links"]),
        **{f"dp_{w}": dp[w] for w in windows}, **{f"dq_{w}": dq[w] for w in windows},
        **{f"base_p_{w}": base_w[w][junctions].values.astype(np.float32) for w in windows},
    )
    return out_path
