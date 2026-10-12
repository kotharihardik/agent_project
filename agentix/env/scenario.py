"""Build benchmark cases: truth hydraulics + citizen evidence + noisy public observations.

Public part  (methods may read):  cases/<id>/case.json, cases/<id>/obs.npz
Hidden part  (environment only):  hidden/<id>.json,      hidden/<id>.npz
Cases are layout-agnostic: obs.npz holds all candidate sensor series; loaders expose only the layout's sensors.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import wntr

from .hydraulics import LeakSpec, add_demand_noise, insert_leak, perturb_model, simulate, summarize_results
from .network import NetworkInfo
from .documents import (FAULT_DOC, FIRE_REPORT, FLUSH_DOC, POISON_NOTE, VALVE_DOC, DocWriter, _fmt_time,
                        render)
from .observe import NOISE_PRESETS, SensorNoise, observe
from .real_model import RealTruth
from .reports import ReportGenerator, TownContext

BASE_DATE = dt.datetime(2026, 3, 2)   # v1 (perturbed truth): a Monday; start = BASE_DATE + weekday offset + weeks
REAL_YEAR_START = dt.datetime(2018, 1, 1)  # v2 (official Real truth): its 1-year demand patterns start Mon 2018-01-01


@dataclass
class SuiteSpec:
    name: str
    leak: bool = True
    force_surface: bool | None = None      # None = physics decides; False = silent leak
    adversarial: bool = False
    false_rate_per_day: float | None = None
    history_days: int = 7
    horizon_days: int = 3
    travel_mult: float = 1.0               # >1: water runs far downhill (misleading reports)
    confounder: str | None = None          # S5: flush_doc | flush_fire | valve_doc | fault_doc | fault_undoc


SUITES = {
    "S1_standard": SuiteSpec("S1_standard"),
    "S2_silent": SuiteSpec("S2_silent", force_surface=False),
    "S3_misleading": SuiteSpec("S3_misleading", force_surface=True, travel_mult=3.0),
    "S4_noleak": SuiteSpec("S4_noleak", leak=False, false_rate_per_day=1.5),
    "S7_traps": SuiteSpec("S7_traps", adversarial=True),
    # S5: something that LOOKS like a leak but is explained by text (work orders, logs, a citizen's message)
    "S5_flush_doc": SuiteSpec("S5_flush_doc", leak=False, confounder="flush_doc"),
    "S5_flush_fire": SuiteSpec("S5_flush_fire", leak=False, confounder="flush_fire"),
    "S5_valve_doc": SuiteSpec("S5_valve_doc", leak=False, confounder="valve_doc"),
    "S5_fault_doc": SuiteSpec("S5_fault_doc", leak=False, confounder="fault_doc"),
    "S5_fault_undoc": SuiteSpec("S5_fault_undoc", leak=False, confounder="fault_undoc"),
    "S5_leak_fault_doc": SuiteSpec("S5_leak_fault_doc", leak=True, confounder="fault_doc"),
}


def _sample_leak(net: NetworkInfo, rng, start_s: int):
    pipes = net.pipes()
    pipe = str(rng.choice(pipes))
    frac = float(rng.uniform(0.15, 0.85))
    diam = float(math.exp(rng.uniform(math.log(0.005), math.log(0.025))))
    kind = "abrupt" if rng.random() < 0.6 else "incipient"
    ramp = int(rng.uniform(6, 72) * 3600) // 300 * 300 if kind == "incipient" else 0
    return LeakSpec(pipe=pipe, frac=frac, diameter_m=diam, kind=kind, start_s=start_s, ramp_s=ramp)


def build_case(cfg: dict, net: NetworkInfo, ctx: TownContext, suite: SuiteSpec, index: int, seed: int,
               out_dir: str, noise: SensorNoise | None = None, flow_rel_sigma: float = 0.01,
               real: RealTruth | None = None, base_pressure: np.ndarray | None = None):
    """Build one case. Truth = official Real model (cfg truth.source == 'real', v2) or perturbed nominal (v1)."""
    noise = noise or NOISE_PRESETS[cfg.get("observation_noise", "field")]
    rng = np.random.default_rng(seed)
    step = cfg["step_s"]
    case_id = f"{cfg['network']}-{suite.name}-{index:04d}"
    tp = cfg["truth"]
    use_real = tp.get("source") == "real"
    total_days = suite.history_days + 2 + suite.horizon_days
    duration = total_days * 86400
    n_steps = duration // step + 1
    if use_real:   # random day of the year -> real seasonality; weekday follows the 2018 calendar
        day0 = int(rng.integers(0, 365 - total_days))
        start_dt = REAL_YEAR_START + dt.timedelta(days=day0)
        weekday = day0 % 7
    else:
        day0 = None
        weekday = int(rng.integers(0, 7))
        start_dt = BASE_DATE + dt.timedelta(days=weekday + 7 * int(rng.integers(0, 40)))
    onset_s = suite.history_days * 86400 + int(rng.uniform(0, 24) * 3600) // step * step

    # ---------------- truth hydraulics
    if use_real:
        truth = real.build(cfg["nominal_inp"], start_step=day0 * 288, n_steps=n_steps, step_s=step)
    else:
        truth = wntr.network.WaterNetworkModel(cfg["nominal_inp"])
        perturb_model(truth, rng, **tp["perturb"])
        add_demand_noise(truth, rng, n_steps, step, start_step=weekday * 288, **tp["demand_noise"])
    truth.options.time.duration = duration
    truth.options.time.report_timestep = cfg["report_step_s"]
    leak, leak_node_name = None, None
    if suite.leak:
        leak = _sample_leak(net, rng, onset_s)
        u, v = net.pipe_endpoints(leak.pipe)
        near_end = u if u in net.index else v
        local_p = float(base_pressure[net.index[near_end]]) if base_pressure is not None else 45.0
        leak_node_name = insert_leak(truth, leak, n_steps, step, local_pressure_m=local_p)
    conf = {}
    crng = np.random.default_rng(seed + 4242)              # separate stream: other suites stay identical
    if suite.confounder in ("flush_doc", "flush_fire", "valve_doc"):
        t_e = suite.history_days * 86400 + int(crng.uniform(0, 24) * 3600) // step * step
        dur = int(crng.uniform(3, 6) * 3600) // step * step
        zoneA = [n for n in net.junctions if net.zone_of[n] == "A"]
        if suite.confounder.startswith("flush"):
            h = str(crng.choice(zoneA))
            q_f = float(crng.uniform(4, 9))
            tt = np.arange(n_steps) * step
            pname = "FLUSH_EVT"
            truth.add_pattern(pname, ((tt >= t_e) & (tt < t_e + dur)).astype(float).tolist())
            truth.get_node(h).add_demand(q_f / 1000.0, pname, "flushing")
            conf = {"type": suite.confounder, "node": h, "start_s": t_e, "end_s": t_e + dur, "flow_lps": q_f}
        else:
            cands = [p for p in net.pipes() if net.wn.get_link(p).diameter >= 0.15
                     and net.zone_of.get(net.pipe_endpoints(p)[0]) == "A"]
            import networkx as nx
            for _ in range(50):
                pipe = str(crng.choice(cands))
                u_, v_ = net.pipe_endpoints(pipe)
                g2 = net.pipe_graph.copy()
                g2.remove_edge(u_, v_)
                if nx.is_connected(g2):
                    break
            link = truth.get_link(pipe)
            for t_x, st in ((t_e, wntr.network.base.LinkStatus.Closed), (t_e + dur, wntr.network.base.LinkStatus.Open)):
                act = wntr.network.controls.ControlAction(link, "status", st)
                cnd = wntr.network.controls.SimTimeCondition(truth, "=", t_x)
                truth.add_control(f"valve_{t_x}", wntr.network.controls.Control(cnd, act))
            conf = {"type": "valve_doc", "pipe": pipe, "node": u_, "start_s": t_e, "end_s": t_e + dur}
    tmpd = tempfile.mkdtemp(prefix="agxcase_")
    try:
        res = simulate(truth, tmpd)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    keep_nodes = cfg["pressure_candidates"]
    tap = f"{leak.pipe}_leaktap" if leak and leak.kind == "abrupt" else None
    S = summarize_results(res, net.junctions, keep_nodes, cfg["flow_links"], step,
                          leak_node_name, tap)
    time_s = S["time_s"]
    leak_lps = S["leak_flow_lps"].values if leak else np.zeros(len(time_s))
    tank = res.node["pressure"][cfg["tank_levels"]].astype(np.float32) if cfg.get("tank_levels") else None

    # ---------------- leak location facts (for scoring)
    gt = {"events": [], "confounders": [], "acceptable_dig_radius_m": 100}
    leak_idx, local_p = None, None
    if leak:
        L = float(net.wn.get_link(leak.pipe).length)
        near = u if leak.frac < 0.5 else v
        leak_idx = net.index[near] if near in net.index else net.index[v if near == u else u]
        xy_u, xy_v = np.array(net.wn.get_node(u).coordinates), np.array(net.wn.get_node(v).coordinates)
        nm = S["night_mean_pressure"]
        local_p = float(nm.iloc[: suite.history_days][net.junctions[leak_idx]].mean())
        on = leak_lps > 0.05
        gt["events"].append({
            "type": "leak", "pipe": leak.pipe, "start_node": u, "end_node": v, "frac": leak.frac,
            "pipe_length_m": L, "nearest_node": net.junctions[leak_idx],
            "xy": (xy_u + leak.frac * (xy_v - xy_u)).round(2).tolist(),
            "kind": leak.kind, "diameter_m": leak.diameter_m, "start_s": leak.start_s, "ramp_s": leak.ramp_s,
            "peak_flow_lps": float(leak_lps.max()),
            "mean_flow_lps_after_start": float(leak_lps[on].mean()) if on.any() else 0.0,
        })

    # ---------------- citizen evidence
    gen = ReportGenerator(ctx, rng, start_dt)
    horizon_s = float(time_s[-1])
    reports, trace = [], {}
    if leak:
        reports, trace = gen.leak_reports(leak_idx, leak.kind, time_s, leak_lps, local_p, horizon_s,
                                          force_surface=suite.force_surface, travel_mult=suite.travel_mult)
        gt["events"][0].update({k: v for k, v in trace.items() if k != "wet_nodes"})
        gt["events"][0]["wet_nodes"] = trace.get("wet_nodes", [])
    # service complaints from the truth pressure drop (vs pre-leak days)
    dm = S["day_min_pressure"]
    base = dm.iloc[: suite.history_days].mean()
    days = np.arange(len(dm))
    drop = (base.values[None, :] - dm.values)
    drop[: suite.history_days] = 0
    reports += gen.complaints(np.clip(drop, 0, None), days, onset_s if leak else None, horizon_s)
    reports += gen.false_reports((suite.history_days - 1) * 86400, horizon_s, suite.false_rate_per_day)

    # ---------------- when does the case open? (first leak report, or flow-based detection)
    det_delay = float(rng.uniform(12, 36) * 3600)
    if leak:
        on_t = time_s[np.argmax(leak_lps > (0.05 if leak.kind == "abrupt" else 1.0))] if (leak_lps > 0.05).any() else onset_s
        first_leak_report = min([r["received_s"] for r in reports if r["truth"]["kind"] == "leak"], default=np.inf)
        open_s = min(first_leak_report, on_t + det_delay)
        open_s = max(open_s, onset_s + 3600)
    elif conf:                                   # flushing / valve works: the inflow/pressure alarm opens the case
        open_s = conf["start_s"] + float(crng.uniform(1.0, 2.5)) * 3600
    elif suite.confounder in ("fault_doc", "fault_undoc"):
        t_f = suite.history_days * 86400 + float(crng.uniform(0, 24)) * 3600
        open_s = t_f + float(crng.uniform(2, 5)) * 3600
        conf = {"type": suite.confounder, "start_s": t_f}
    else:
        first_any = min([r["received_s"] for r in reports if r["received_s"] > suite.history_days * 86400],
                        default=np.inf)
        open_s = min(first_any, suite.history_days * 86400 + det_delay)
    open_s = float(min(open_s, (suite.history_days + 2) * 86400))
    if suite.adversarial:
        target = int(rng.integers(len(net.junctions)))
        if leak_idx is not None:
            far = np.where(net.pipe_distance[leak_idx] > 600)[0]
            target = int(rng.choice(far)) if len(far) else target
        reports += gen.adversarial(open_s - 1800, target, n_coordinated=int(rng.integers(0, 4)))
        gt["confounders"].append({"type": "adversarial_reports", "target_node": net.junctions[target]})
    dw = DocWriter(ctx.gazetteer_places, net, np.random.default_rng(seed + 7777))
    if conf.get("type") == "flush_fire":
        lm, st = dw.near_names(conf["node"])
        t_r = conf["start_s"] + float(crng.uniform(0.3, 1.2)) * 3600
        txt = dw.fill(str(crng.choice(FIRE_REPORT)), start_dt + dt.timedelta(seconds=t_r), lm=lm,
                      t0=_fmt_time(start_dt + dt.timedelta(seconds=conf["start_s"]), crng))
        g = gen._gold("R-x", t_r, "en", ["water_on_surface"], [{"text": lm, "type": "landmark", "relation": "near"}],
                      "street", "unknown", None, {"onset_text": None, "onset_kind": "unknown", "onset_value": None})
        g["summary_en"] = f"Fire brigade / tanker taking water from a hydrant near {lm}."
        reports.append({"report_id": "R-x", "received_at": (start_dt + dt.timedelta(seconds=t_r)).isoformat(timespec="minutes"),
                        "received_s": t_r, "channel": "whatsapp", "text": txt, "gold": g,
                        "truth": {"kind": "confounder_report", "seen_node": conf["node"], "is_source_seen": "unknown",
                                  "false_type": None, "place_ids": [], "language": "en"}})
    fault = None
    if suite.confounder in ("fault_doc", "fault_undoc"):
        pool = cfg.get("fault_sensor_pool", keep_nodes)
        fault = {"sensor": str(crng.choice(pool)), "offset_m": -float(crng.uniform(0.25, 0.6)),
                 "start_s": conf.get("start_s", open_s - float(crng.uniform(4, 12)) * 3600)}
        conf = {"type": suite.confounder, **fault}
    if conf:
        gt["confounders"].append(conf)
    # documents: harmless entries for every case + the explaining entry (documented S5) + poison (S7)
    wo, mt = dw.benign(start_dt, total_days, keep_nodes)
    t_doc = lambda x: start_dt + dt.timedelta(seconds=float(x))
    if conf.get("type") == "flush_doc":
        lm, st = dw.near_names(conf["node"])
        wo.append(dw.fill(str(dw.rng.choice(FLUSH_DOC)), t_doc(conf["start_s"]), lm=lm, street=st, q=conf["flow_lps"],
                          t0=_fmt_time(t_doc(conf["start_s"]), dw.rng), t1=_fmt_time(t_doc(conf["end_s"]), dw.rng)))
    if conf.get("type") == "valve_doc":
        lm, st = dw.near_names(conf["node"])
        wo.append(dw.fill(str(dw.rng.choice(VALVE_DOC)), t_doc(conf["start_s"]), lm=lm, street=st,
                          t0=_fmt_time(t_doc(conf["start_s"]), dw.rng), t1=_fmt_time(t_doc(conf["end_s"]), dw.rng)))
    if conf.get("type") == "fault_doc":
        mt.append(dw.fill(str(dw.rng.choice(FAULT_DOC)), t_doc(conf["start_s"]), s=conf["sensor"],
                          t0=_fmt_time(t_doc(conf["start_s"]), dw.rng)))
    if suite.adversarial:
        mt.append(dw.fill(str(dw.rng.choice(POISON_NOTE)), t_doc(open_s - 3600)))
    dw.rng.shuffle(wo)
    dw.rng.shuffle(mt)
    reports.sort(key=lambda r: r["received_s"])
    # IDs by arrival order only, so an ID never reveals how a report was generated (no answer leak)
    for k, r in enumerate(reports, start=1):
        r["report_id"] = f"R-{k:04d}"
        r["gold"]["report_id"] = r["report_id"]

    # ---------------- public observations (noisy, until case open)
    vis = time_s <= open_s
    p_all, noise_parts = observe(S["sensor_pressure"].values, time_s, rng, noise)
    if fault:
        si = keep_nodes.index(fault["sensor"])
        p_all[time_s >= fault["start_s"], si] += fault["offset_m"]
    p_obs = p_all[vis]
    q_full = S["link_flow"].values * (1 + rng.normal(0, flow_rel_sigma, S["link_flow"].values.shape))
    q_obs = q_full[vis]
    public_reports = [{k: r[k] for k in ("report_id", "received_at", "channel", "text")}
                      for r in reports if r["received_s"] <= open_s]

    os.makedirs(f"{out_dir}/cases/{case_id}", exist_ok=True)
    os.makedirs(f"{out_dir}/hidden", exist_ok=True)
    np.savez_compressed(f"{out_dir}/cases/{case_id}/obs.npz", time_s=time_s[vis],
                        pressure=p_obs, pressure_nodes=np.array(keep_nodes),
                        flow=q_obs.astype(np.float32), flow_links=np.array(cfg["flow_links"]),
                        tank_level=(tank.values[vis].astype(np.float32) if tank is not None else np.zeros(0)))
    public_id = "C" + hashlib.sha1(f"agentix-{case_id}-{seed}".encode()).hexdigest()[:10]
    case = {
        "case_id": public_id, "network": cfg["network"], "inference_model": cfg["nominal_inp"],
        "start_datetime": start_dt.isoformat(), "case_open_s": open_s,
        "case_open_at": (start_dt + dt.timedelta(seconds=open_s)).isoformat(timespec="minutes"),
        "history_days": suite.history_days, "horizon_s": horizon_s,
        "pattern_start_s": weekday * 86400,      # where the case starts inside the nominal WEEKLY pattern
        "observations": {"file": "obs.npz", "sensor_noise_note": "pressure (m) at all candidate sensors; "
                         "use only the nodes of your sensor layout", "report_step_s": cfg["report_step_s"]},
        "reports": public_reports,
    }
    with open(f"{out_dir}/cases/{case_id}/case.json", "w") as f:
        json.dump(case, f, indent=1, ensure_ascii=False)
    os.makedirs(f"{out_dir}/cases/{case_id}/docs", exist_ok=True)
    with open(f"{out_dir}/cases/{case_id}/docs/work_orders.md", "w") as f:
        f.write(render(wo, "Work orders (network operations)"))
    with open(f"{out_dir}/cases/{case_id}/docs/maintenance_log.md", "w") as f:
        f.write(render(mt, "Sensor maintenance log"))
    hidden = {"case_id": case_id, "public_id": public_id, "seed": seed, "suite": suite.name,
              "truth_model": "official_real" if use_real else "perturbed_nominal", "real_day_of_year": day0,
              "noise": noise.__dict__,
              "ground_truth": gt, "reports_all": reports, "open_s": open_s, "start_datetime": start_dt.isoformat()}
    with open(f"{out_dir}/hidden/{case_id}.json", "w") as f:
        json.dump(hidden, f, indent=1, ensure_ascii=False, default=float)
    np.savez_compressed(f"{out_dir}/hidden/{case_id}.npz", time_s=time_s,
                        sensor_pressure_clean=S["sensor_pressure"].values,
                        link_flow_clean=S["link_flow"].values,
                        night_mean_pressure=S["night_mean_pressure"].values,
                        day_min_pressure=S["day_min_pressure"].values, leak_lps=leak_lps.astype(np.float32),
                        sensor_bias=noise_parts["bias"], sensor_daily_drift=noise_parts["daily_drift"],
                        pressure_obs_full=p_all, flow_obs_full=q_full.astype(np.float32),
                        tank_level_full=(tank.values.astype(np.float32) if tank is not None else np.zeros(0)),
                        junctions=np.array(net.junctions))
    n_leak_reports = sum(r["truth"]["kind"] == "leak" for r in reports)
    return {"case_id": case_id, "suite": suite.name, "open_h_after_onset": (open_s - onset_s) / 3600,
            "n_reports_total": len(reports), "n_leak_reports": n_leak_reports, "n_public_reports": len(public_reports),
            "surfaced": trace.get("surfaced"), "peak_lps": float(leak_lps.max()),
            "path_len_m": trace.get("path_len_m"), "kind": leak.kind if leak else None}
