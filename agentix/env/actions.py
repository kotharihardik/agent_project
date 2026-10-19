"""The investigation environment: an agent acts on one case; answers come from the HIDDEN truth.

Methods/agents only ever see what `observe()` and the action results return. Every call is logged with time and cost.
Tool errors are returned as {"ok": False, "error": ...} (never raised) so recovery behaviour can be measured.
"""
from __future__ import annotations

import json
import math
import zlib

import numpy as np
import yaml

from .network import NetworkInfo
import datetime as dt

from .observe import SensorNoise, logger_reading
from .operator import Operator
from .reporter_sim import ReporterSimulator


def case_seed(case_id: str, layout_name: str) -> int:
    """A stable random seed per (case, layout): every method meets the same luck (replies, acoustic hits).
    (Python's built-in hash() changes between processes, so it must not be used for this.)"""
    return zlib.crc32(f"{case_id}|{layout_name}".encode()) % 10_000


class CaseEnv:
    def __init__(self, bench_dir: str, case_id: str, net: NetworkInfo, layout: list[str], costs_yaml: str,
                 gazetteer: dict, seed: int = 0, noise: SensorNoise | None = None, crews_per_night: int | None = None,
                 text_variant: str | None = None, citizen: str = "template", crn: bool = False,
                 tool_fail: float = 0.0, acoustic_fp: float | None = None):
        self.net, self.layout = net, list(layout)
        self.case = json.load(open(f"{bench_dir}/cases/{case_id}/case.json"))
        self._hid = json.load(open(f"{bench_dir}/hidden/{case_id}.json"))
        self._arr = np.load(f"{bench_dir}/hidden/{case_id}.npz")
        self.costs = yaml.safe_load(open(costs_yaml))
        # how many field checks (listening, logger, step test) can run in the same night = night crews available
        self.crews = int(crews_per_night or self.costs.get("crews_per_night", 1))
        self._last_night, self._night_used = None, {}
        self.rng = np.random.default_rng(seed)
        # common random numbers (evaluation fairness): with crn=True the luck of each check depends only on WHICH
        # check it is (case, kind, target, how many times it was repeated), not on what was done before, so two
        # methods that run the same check get the same answer
        self.seed, self.crn, self._crn_n = seed, crn, {}
        self.tool_fail = tool_fail   # robustness suite: share of night field checks that fail (night and cost used)
        # robustness suite: noisy streets - the true false-alarm rate of listening is higher than the methods'
        # evidence model assumes (costs.yaml false_positive); None = as configured
        self.acoustic_fp = acoustic_fp
        self.noise = noise or SensorNoise()
        self.now = float(self.case["case_open_s"])
        self.horizon = float(self.case["horizon_s"])
        self.log, self.total_cost, self.loggers = [], 0.0, {}
        self.done, self.outcome = False, None
        self.places = {p["place_id"]: p for p in gazetteer["places"]}
        self._cand = list(self._arr["junctions"])
        self._pnodes = list(np.load(f"{bench_dir}/cases/{case_id}/obs.npz")["pressure_nodes"])
        ev = self._hid["ground_truth"]["events"]
        self._leak = ev[0] if ev else None
        self.text_variant = text_variant
        if text_variant == "realistic":              # reports written the way real people write (same truth)
            from .loader import realistic_texts
            alt = realistic_texts(f"{bench_dir}/cases/{case_id}")
            for r in self._hid["reports_all"]:
                r["text"] = alt.get(r["report_id"], r["text"])
        if citizen == "talking":                     # answers any follow-up question (env/citizen_llm.py)
            from .citizen_llm import TalkingCitizen
            self.reporter = TalkingCitizen(net, self.places, self.rng, realistic=(text_variant == "realistic"))
        else:
            self.reporter = ReporterSimulator(net, self.places, self.rng, realistic=(text_variant == "realistic"))
        self.operator = Operator()
        self.permits: set[str] = set()
        self.notified: list[str] = []
        self.start_dt = dt.datetime.fromisoformat(self.case["start_datetime"])

    def drop_false_reports(self, keep: float) -> set[str]:
        """Experiment setting (harness only): keep each false-alarm / low-pressure message with probability `keep`
        (fixed per message, so every method sees the same messages). Returns the ids removed."""
        if keep >= 1.0:
            return set()
        gone = {r["report_id"] for r in self._hid["reports_all"]
                if r["truth"]["kind"] in ("false", "complaint_background")
                and zlib.crc32(f"keep|{r['report_id']}|{self.case['case_id']}".encode()) % 1000 >= keep * 1000}
        self._hid["reports_all"] = [r for r in self._hid["reports_all"] if r["report_id"] not in gone]
        return gone

    def _r(self, kind: str, key: str) -> np.random.Generator:
        if not self.crn:
            return self.rng
        k = f"{kind}|{key}"
        n = self._crn_n[k] = self._crn_n.get(k, 0) + 1
        return np.random.default_rng(zlib.crc32(f"{self.seed}|{self.case['case_id']}|{k}|{n}".encode()))

    FAIL_REASONS = ["the valve on the block boundary would not close, test aborted",
                    "the listening equipment failed (battery), no result", "heavy traffic noise all night, no result",
                    "the crew was called away to a burst elsewhere", "the meter logger lost its data"]

    def _failed(self, kind: str, key: str, args: dict, cost: float):
        """Robustness suite: did tonight's field check fail? (The night and the cost are used, as in real life.)"""
        if self.tool_fail <= 0:
            return None
        r = np.random.default_rng(zlib.crc32(f"fail|{self.seed}|{self.case['case_id']}|{kind}|{key}|"
                                             f"{self._crn_n.get('fail|' + kind + key, 0)}".encode()))
        self._crn_n["fail|" + kind + key] = self._crn_n.get("fail|" + kind + key, 0) + 1
        if r.random() >= self.tool_fail:
            return None
        reason = self.FAIL_REASONS[int(r.integers(len(self.FAIL_REASONS)))]
        return self._record(kind, args, {"ok": False, "error": f"field check failed: {reason}", "now_s": self.now}, cost)

    def clock(self, t_s: float | None = None) -> dt.datetime:
        """Wall-clock time of the case at t_s seconds (default: now)."""
        return self.start_dt + dt.timedelta(seconds=float(self.now if t_s is None else t_s))

    # ------------------------------------------------------------------ helpers
    def _record(self, action, args, result, cost=0.0):
        self.total_cost += cost
        self.log.append({"t_s": self.now, "action": action, "args": args, "cost": cost, "result": result})
        return result

    def _err(self, action, args, msg):
        return self._record(action, args, {"ok": False, "error": msg})

    def _next_night_end(self) -> float:
        d = math.floor((self.now - 4 * 3600) / 86400) + 1
        return d * 86400 + 5 * 3600, d

    def _night_slot(self):
        """When the next field check can run. Checks run at night. With several crews, a check asked for right after
        a night's results came in runs in that same night (another crew), until all crews of that night are used."""
        if self._last_night is not None and self.now == self._last_night[0] \
                and self._night_used.get(self._last_night[1], 0) < self.crews:
            return self._last_night
        return self._next_night_end()

    def _use_night(self, slot):
        self._last_night = slot
        self._night_used[slot[1]] = self._night_used.get(slot[1], 0) + 1

    def night_left(self) -> bool:
        """Is there still a night slot (field checks run at night) before the case horizon?"""
        return self._night_slot()[0] <= self.horizon

    def leak_distance_m(self, node: str) -> float:
        """Pipe distance from a junction to the true leak point (inf if no leak)."""
        if not self._leak:
            return math.inf
        L, f = self._leak["pipe_length_m"], self._leak["frac"]
        i = self.net.index[node]
        best = math.inf
        for end, extra in ((self._leak["start_node"], f * L), (self._leak["end_node"], (1 - f) * L)):
            j = self.net.index.get(end)
            if j is not None:
                best = min(best, self.net.pipe_distance[i, j] + extra)
        return best

    # ------------------------------------------------------------------ what the agent can see
    def observe(self) -> dict:
        t = self._arr["time_s"]
        vis = t <= self.now
        cols = [self._pnodes.index(n) for n in self.layout]
        reps = [{k: r[k] for k in ("report_id", "received_at", "channel", "text")}
                for r in self._hid["reports_all"] if r["received_s"] <= self.now]
        out = {"now_s": self.now, "time_s": t[vis], "pressure": self._arr["pressure_obs_full"][vis][:, cols],
               "pressure_nodes": self.layout, "reports": reps,
               "loggers": {n: v for n, v in self.loggers.items()}, "total_cost": self.total_cost}
        if "flow_obs_full" in self._arr.files:          # flows + tank level keep coming after case open
            out["flow"] = self._arr["flow_obs_full"][vis]
            out["tank_level"] = self._arr["tank_level_full"][vis][:, None] if self._arr["tank_level_full"].ndim == 1 \
                else self._arr["tank_level_full"][vis]
        return out

    # ------------------------------------------------------------------ approvals, permits, messages
    def request_approval(self, action: str, target, certificate: dict):
        """Ask the duty operator to approve a 'dig' (target = node) or 'step_test' (target = node list)."""
        c = self.clock()
        d = self.operator.review(action, target, certificate, c.hour + c.minute / 60)
        return self._record("request_approval", {"action": action, "target": target if action == "dig" else len(target),
                                                  "certificate": certificate}, {"ok": True, **d})

    def request_permit(self, street: str):
        """Municipal permit for digging on a recently resurfaced road (SOP-6). Takes about 12 hours."""
        if not any(p["type"] == "street" and p["name"].lower() == street.lower() for p in self.places.values()):
            return self._err("request_permit", {"street": street}, f"unknown street '{street}'")
        self.now = min(self.now + 12 * 3600, self.horizon)
        self.permits.add(street.lower())
        return self._record("request_permit", {"street": street}, {"ok": True, "granted": True, "now_s": self.now}, 0.05)

    def notify(self, target: str, message: str):
        """Send a message: 'hospital:<name>', 'reporter:<report_id>', 'supervisor', 'traffic_police'."""
        self.notified.append(target.lower())
        return self._record("notify", {"target": target, "message": message[:300]}, {"ok": True})

    # ------------------------------------------------------------------ actions
    def wait(self, hours: float):
        if hours <= 0:
            return self._err("wait", {"hours": hours}, "hours must be > 0")
        self.now = min(self.now + hours * 3600, self.horizon)
        return self._record("wait", {"hours": hours}, {"ok": True, "now_s": self.now})

    def place_logger(self, node: str):
        args = {"node": node}
        if node not in self.net.index:
            return self._err("place_logger", args, f"unknown node '{node}'")
        if len(self.loggers) >= self.costs["max_loggers_per_case"]:
            return self._err("place_logger", args, "logger limit reached")
        t_end, d = self._night_slot()
        nm = self._arr["night_mean_pressure"]
        if t_end > self.horizon or d >= nm.shape[0]:
            return self._err("place_logger", args, "no night left before horizon")
        self.now = t_end
        self._use_night((t_end, d))
        val = logger_reading(float(nm[d, self.net.index[node]]), self._r("logger", node), self.noise)
        self.loggers[node] = {"night_day": d, "night_mean_m": val}
        return self._record("place_logger", args, {"ok": True, "node": node, "night_mean_pressure_m": val,
                                                    "window": "02:00-04:00", "now_s": self.now},
                            self.costs["place_logger"]["cost"])

    def ask_reporter(self, report_id: str, question: str, options: list[dict] | None = None):
        """Send a follow-up question to the person who sent `report_id`.

        `options` (optional): [{"place_id": ..., "label": ...}] for a multiple-choice "near which of these?".
        The simulated reporter also understands questions about the source (coming out vs collected / flow
        direction), the street name and the nearest landmark - several in one question are fine.
        """
        options = options or []
        args = {"report_id": report_id, "question": question, "options": options}
        rep = next((r for r in self._hid["reports_all"] if r["report_id"] == report_id), None)
        if rep is None or rep["received_s"] > self.now:
            return self._err("ask_reporter", args, f"unknown or not-yet-received report '{report_id}'")
        asked_before = sum(1 for e in self.log if e["action"] == "ask_reporter"
                           and e["args"]["report_id"] == report_id and e["result"].get("ok"))
        if asked_before >= 2:                      # don't pester citizens: at most two questions per report
            return self._err("ask_reporter", args, "this reporter was already asked twice")
        c = self.costs["ask_reporter"]
        rng = self._r("ask", report_id)
        delay = rng.uniform(*c["delay_h"]) * 3600
        self.now = min(self.now + delay, self.horizon)
        if rng.random() > c["response_prob"]:
            return self._record("ask_reporter", args, {"ok": True, "reply_text": None, "note": "no reply"})
        if self.crn:
            self.reporter.rng = rng
        reply = self.reporter.answer(rep, self._leak, question, options)
        return self._record("ask_reporter", args, {"ok": True, "reply_text": reply, "now_s": self.now})

    def acoustic_survey(self, center_node: str, radius_m: float):
        args = {"center_node": center_node, "radius_m": radius_m}
        c = self.costs["acoustic_survey"]
        if center_node not in self.net.index:
            return self._err("acoustic_survey", args, f"unknown node '{center_node}'")
        if radius_m <= 0 or radius_m > c["max_radius_m"]:
            return self._err("acoustic_survey", args, f"radius must be in (0, {c['max_radius_m']}]")
        t_end, d = self._night_slot()
        if t_end > self.horizon:
            return self._err("acoustic_survey", args, "no night left before horizon")
        self.now = t_end
        self._use_night((t_end, d))
        f = self._failed("acoustic_survey", center_node, args, c["cost"])
        if f is not None:
            return f
        d_leak = self.leak_distance_m(center_node)
        rng = self._r("acoustic", center_node)
        q = float(self._leak["mean_flow_lps_after_start"]) if self._leak else 0.0
        p_det = min(c["p_detect_max"], c["p_detect_1lps"] * math.sqrt(max(q, 0.01)))
        fp = c["false_positive"] if self.acoustic_fp is None else self.acoustic_fp
        detected = (d_leak <= radius_m and rng.random() < p_det) or rng.random() < fp
        res = {"ok": True, "detected": bool(detected), "now_s": self.now}
        if detected and d_leak <= radius_m:
            near = self.net.index[self._leak["nearest_node"]]
            offset = abs(rng.normal(0, c["locate_sigma_m"]))
            dist = self.net.pipe_distance[near]
            res["approx_node"] = self.net.junctions[int(np.argmin(np.abs(dist - offset)))]
        elif detected:  # false positive somewhere in the surveyed area
            ring = np.where(self.net.pipe_distance[self.net.index[center_node]] <= radius_m)[0]
            res["approx_node"] = self.net.junctions[int(rng.choice(ring))]
        return self._record("acoustic_survey", args, res, c["cost"])

    def step_test(self, nodes: list[str], approved: bool = False, approval_token: str | None = None):
        args = {"nodes": nodes, "approved": approved, "token": bool(approval_token)}
        c = self.costs["step_test"]
        valid = self.operator.use(approval_token, "step_test", nodes) if approval_token else False
        args["operator_approved"] = valid
        if not (approved or valid):
            return self._err("step_test", args, "approval required for step test (customer disruption)")
        bad = [n for n in nodes if n not in self.net.index]
        if bad or not nodes or len(nodes) > c["max_nodes"]:
            return self._err("step_test", args, f"invalid node set (unknown={bad[:3]}, size={len(nodes)})")
        t_end, d = self._night_slot()
        if t_end > self.horizon:
            return self._err("step_test", args, "no night left before horizon")
        self.now = t_end
        self._use_night((t_end, d))
        f = self._failed("step_test", ",".join(sorted(nodes)), args, c["cost"])
        if f is not None:
            return f
        inside = bool(self._leak and self._leak["nearest_node"] in set(nodes))
        q = float(self._leak["mean_flow_lps_after_start"]) if (self._leak and inside) else 0.0
        night_dem = float(sum(self.net.base_demand[self.net.index[n]] for n in nodes) * 1000 * 0.4)
        measured = night_dem + q + self._r("step", ",".join(sorted(nodes))).normal(0, c["flow_noise_lps"])
        return self._record("step_test", args, {"ok": True, "inflow_drop_lps": round(measured, 2),
                                                 "expected_customer_drop_lps": round(night_dem, 2),
                                                 "now_s": self.now}, c["cost"])

    def request_dig(self, node: str, approved: bool = False, approval_token: str | None = None,
                    schedule: str = "now"):
        """Send a crew to dig at `node`. Needs an operator token (or approved=True for auto-approved non-LLM
        policies, which is recorded). schedule: 'now' | 'after_school_hours' (waits until 16:30 if needed)."""
        valid = self.operator.use(approval_token, "dig", node) if approval_token else False
        args = {"node": node, "approved": approved, "operator_approved": valid, "schedule": schedule}
        c = self.costs["dig"]
        if not (approved or valid):
            return self._err("request_dig", args, "approval required for dig")
        if node not in self.net.index:
            return self._err("request_dig", args, f"unknown node '{node}'")
        if schedule == "after_school_hours":
            ck = self.clock()
            if ck.weekday() < 5 and (7.5 <= ck.hour + ck.minute / 60 < 16.5):
                wait_h = 16.5 - (ck.hour + ck.minute / 60)
                self.now = min(self.now + wait_h * 3600, self.horizon)
        self.now = min(self.now + c["duration_h"] * 3600, self.horizon)
        d = self.leak_distance_m(node)
        found = d <= self.costs["dig_success_radius_m"]
        if found:
            self.done, self.outcome = True, "correct_dig"
        return self._record("request_dig", args, {"ok": True, "leak_found": bool(found), "now_s": self.now},
                            c["cost"])

    def abstain(self, reason: str = ""):
        self.done, self.outcome = True, "abstain"
        return self._record("abstain", {"reason": reason}, {"ok": True})

    # ------------------------------------------------------------------ scoring (evaluation only)
    def score(self) -> dict:
        digs = [e for e in self.log if e["action"] == "request_dig" and e["result"].get("ok")]
        wrong = sum(not e["result"]["leak_found"] for e in digs)
        lost_m3 = 0.0
        if self._leak:
            t = self._arr["time_s"]
            q = self._arr["leak_lps"]
            m = t <= self.now
            lost_m3 = float(np.trapezoid(q[m], t[m]) / 1000)
        return {"outcome": self.outcome or "unfinished", "has_leak": self._leak is not None,
                "wrong_digs": wrong, "n_actions": len(self.log), "action_cost_cu": round(self.total_cost, 3),
                "hours_from_open": (self.now - self.case["case_open_s"]) / 3600, "water_lost_m3": round(lost_m3, 1)}
