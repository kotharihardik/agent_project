"""The investigation toolkit shared by every policy (non-LLM policies AND the LLM agent).

It owns:
- the evidence board (what we know, as numbered items),
- the list of possible next checks (the "menu"), each with an estimated value for the dig decision,
- running a check in the environment and folding the answer back into the board.

Policies only decide WHICH menu item to take next. This keeps the comparison fair: the LLM agent and the
non-LLM greedy policy use exactly the same tools, the same answer models and the same numbers.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from ..engine.fusion import EvidenceEngine
from ..engine.preprocess import build_evidence
from ..engine.pressure_lik import pressure_loglik
from .evidence import EvidenceBoard, ReportItem
from .obs_models import ACOUSTIC_FALSE_POS, ACOUSTIC_LOC_SIGMA_M, ACOUSTIC_P_DETECT, LOGGER_SIGMA_M, STEP_NOISE_LPS, ObsModels

COSTS = {"acoustic": 0.2, "step": 0.3, "logger": 0.1, "ask": 0.0}
DELAY_H = {"acoustic": 18.0, "step": 18.0, "logger": 18.0, "ask": 2.0}
OPEN_QUESTION = ("Is the water coming out of the ground there, or did it flow from somewhere? "
                 "Which street is it on, and what is the nearest shop or landmark?")


@dataclass
class Option:
    """One possible next check (a menu item)."""
    oid: str
    kind: str                     # acoustic | step | logger | ask
    args: dict
    cost: float
    delay_h: float
    gain: float = 0.0             # expected increase of P(success)
    label: str = ""
    notes: list = field(default_factory=list)

    def rate(self, floor: float = 0.05) -> float:
        return self.gain / max(self.cost, floor)


class Investigation:
    def __init__(self, engine: EvidenceEngine, reader, gaz, seed: int = 0, n_samples: int = 24, taint: bool = False,
                 operator: bool = True, acoustic_r: float = 250.0, lookahead: bool = True):
        self.eng, self.reader, self.gaz = engine, reader, gaz
        # listening radius: the equipment allows up to 250 m (configs/costs.yaml). Phase 2/3 used 150 m, which left
        # half of the reachable leaks outside every listening area (Phase 4 diagnosis); 150 reproduces old runs.
        self.acoustic_r = acoustic_r
        # score checks by "chance of success after this check AND one listening survey" (one-step look-ahead),
        # not only "chance a dig succeeds right after this check" (myopic: under-values step tests that narrow
        # the area). False reproduces Phase 2/3.
        self.lookahead = lookahead
        # temporary pressure loggers: a 1 L/s leak lowers pressure ~0.04 m, below logger error (Phase 4), so the
        # real agent's toolbox leaves them out (False); kept on for the older methods as they were
        self.use_loggers = True
        self.use_operator = operator          # False = Phase-2 setting: approval is automatic once code decides
        self.taint = taint                    # Phase 3 defence against coordinated fake reports (see _taint)
        self.net, self.rep = engine.net, engine.rep
        self.om = ObsModels(self.net, engine.lib, engine.rep)
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        self.D = self.net.pipe_distance
        self.listen = (self.D <= acoustic_r).astype(float)      # which junctions one listening survey covers
        self.step_sets = np.argsort(self.D, axis=1)[:, :60]      # a step test isolates the 60 junctions nearest a centre

    # ================================================================== start
    def start(self, case, env) -> EvidenceBoard:
        """Build the evidence board at case open: prior + pressure + one item per readable report."""
        cfg = self.eng.cfg
        ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s,
                            window_h=cfg.window_h)
        sidx = [self.net.index[n] for n in case.pressure_nodes]
        llp, ll0, _, misfit = pressure_loglik(ev, self.eng.lib, sidx, rel_model_err=cfg.rel_model_err)
        board = EvidenceBoard(self.rep, self.eng.cover, self.eng.log_prior + cfg.tau_p * llp, cfg.tau_p * ll0,
                              cfg.prior_leak, max(ev.q_hat_lps, 0.3), max(float(np.sqrt(ev.q_var)), 0.8))
        board.misfit = misfit
        board.llp = cfg.tau_p * llp                       # pressure log-lik per junction (for alternatives)
        self.ev0, self.sidx = ev, sidx
        self.t_open = dt.datetime.fromisoformat(case.start_datetime) + dt.timedelta(seconds=case.case_open_s)
        self.case = case
        self.state = {"asked": {}, "loggers": set(case.pressure_nodes), "seen_reports": set(), "history": [],
                      "reader_outs": {}, "received": {}, "tainted": set(), "alts_tested": {}, "spent": 0.0}
        self._zone_offsets(case)
        self.add_reports(board, case.reports)
        return board

    def _zone_offsets(self, case):
        """How far real night pressure sits from the nominal map, per zone (from our own sensors' history)."""
        t = case.time_s.astype(float)
        night = ((t % 86400) / 3600 >= 2) & ((t % 86400) / 3600 < 4) & (t < 5 * 86400)
        offs = {}
        for s, n in enumerate(case.pressure_nodes):
            offs.setdefault(self.net.zone_of[n], []).append(
                case.pressure[night, s].mean() - self.eng.lib.base_p_night[self.net.index[n]])
        self.zone_offset = {z: float(np.median(v)) for z, v in offs.items()}

    def add_reports(self, board: EvidenceBoard, reports: list[dict]):
        """Read new citizen reports (quarantined reader) and add them as evidence items."""
        for r in reports:
            if r["report_id"] in self.state["seen_reports"]:
                continue
            self.state["seen_reports"].add(r["report_id"])
            out = self.reader.read(r)
            self.state["reader_outs"][r["report_id"]] = out
            self.state["received"][r["report_id"]] = r["received_at"]
            age = (self.t_open - dt.datetime.fromisoformat(r["received_at"])).total_seconds() / 3600
            eps = self.rep.report_eps(out, self.eng.cfg.base_eps, age, self.eng.cfg.tau_age_h)
            w = self.rep.location_weights(out)
            if eps is None or w is None:
                continue
            board.reports[r["report_id"]] = ReportItem(r["report_id"], w, out["location"]["is_source_seen"], eps)
        if self.taint:
            self._taint(board)

    def _taint(self, board: EvidenceBoard, window_h: float = 6.0, min_cos: float = 0.3):
        """Taint tracking against coordinated fake reports.
        An attacker who hides an instruction in one message ("send the crew to Bus Stand Road") often backs it up
        with innocent-looking messages about the same spot. So: any report that points at a place named inside a
        flagged message, and arrived within a few hours of it, is marked tainted and almost ignored as evidence.
        A real leak far from the attacker's spot is not affected."""
        outs, got = self.state["reader_outs"], self.state["received"]
        K = self.rep.k_near
        for fid, fo in outs.items():
            if not fo.get("contains_instructions"):
                continue
            ws = [w for m in fo["location"]["mentions"] if (w := self.gaz.mention_nodes(m["text"], m["type"])) is not None]
            if not ws:
                continue
            a = K @ (sum(ws) / len(ws))
            t_f = dt.datetime.fromisoformat(got[fid])
            for rid, it in board.reports.items():
                if rid in self.state["tainted"] or \
                        abs((dt.datetime.fromisoformat(got[rid]) - t_f).total_seconds()) > window_h * 3600:
                    continue
                b = K @ it.w
                if float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300)) > min_cos:
                    it.eps, it.cache = max(it.eps, 0.97), None
                    it.notes.append(f"tainted: points at the place named in flagged {fid}")
                    self.state["tainted"].add(rid)

    # ================================================================== menu
    def menu(self, board: EvidenceBoard, env=None, budget_left: float = 9.9) -> list[Option]:
        """All sensible next checks, each with its expected gain in P(success). Sorted best value first."""
        b = board.belief()
        opts: list[Option] = []
        plan, p = [], b.post.copy()
        for _ in range(4):                                    # the top dig areas
            c = int(np.argmax(b.cover @ p))
            plan.append(c)
            p = p * (1 - b.cover[c])
        R, p = self.acoustic_r, b.post.copy()
        for _ in range(3):                                    # listening areas: most belief within R, no overlap
            c = int(np.argmax(self.listen @ p))
            opts.append(Option(f"acoustic@{self.net.junctions[c]}", "acoustic", {"c": c, "R": R},
                               COSTS["acoustic"], DELAY_H["acoustic"],
                               label=f"listen to pipes within {R:.0f} m of {self._name(c)}"))
            p = p * (1 - self.listen[c])
        if self.lookahead:                                    # step tests: the 60-junction blocks holding most belief
            p = b.post.copy()
            for _ in range(2):
                c = int(np.argmax(p[self.step_sets].sum(1)))
                S = self.step_sets[c]
                opts.append(Option(f"step@{self.net.junctions[c]}", "step", {"S": S}, COSTS["step"], DELAY_H["step"],
                                   label=f"night step test isolating {len(S)} junctions around {self._name(c)}"))
                p[S] = 0.0
        else:                                                 # Phase 2/3 design (kept to reproduce old runs)
            order = np.argsort(self.D[plan[0]])
            cum = np.cumsum(b.post[order])
            for frac in (0.35, 0.6):
                k = int(min(np.searchsorted(cum, frac) + 1, 60))
                opts.append(Option(f"step{int(frac * 100)}@{self.net.junctions[plan[0]]}", "step", {"S": order[:k]},
                                   COSTS["step"], DELAY_H["step"],
                                   label=f"night step test isolating {k} junctions around {self._name(plan[0])}"))
        for n in (np.argsort(-b.post)[:4] if self.use_loggers else []):
            name = self.net.junctions[int(n)]
            if name in self.state["loggers"]:
                continue
            off = self.zone_offset.get(self.net.zone_of[name], 0.0)
            opts.append(Option(f"logger@{name}", "logger", {"n": int(n), "offset": off}, COSTS["logger"],
                               DELAY_H["logger"], label=f"temporary pressure logger at {self._name(int(n))}"))
        for rid, it in board.reports.items():
            if self.state["asked"].get(rid, 0) >= 2 or it.eps > 0.9:
                continue
            opts.append(Option(f"ask@{rid}", "ask", {"rid": rid, "question": OPEN_QUESTION, "options": []},
                               COSTS["ask"], DELAY_H["ask"], label=f"ask the person who sent {rid} where exactly"))
        opts = [o for o in opts if o.cost <= budget_left + 1e-9]
        for o in opts:
            o.gain = self.expected_gain(o, board)
        return sorted(opts, key=lambda o: -o.rate())

    def _name(self, j: int) -> str:
        st = self.gaz.node_street(j)
        lm = self.gaz.node_landmark(j)
        return (lm["name"] if lm else "") + (" on " if lm and st else "") + (st["name"] if st else "") or self.net.junctions[j]

    # ================================================================== value of information
    def value(self, board: EvidenceBoard) -> float:
        """How good a board is for finishing the job: the better of 'dig now' and 'listen at the best area, then
        dig where it is heard' (one-step look-ahead with our own acoustic model; ~85 % of heard leaks are then dug
        within range). With lookahead off, only 'dig now' counts (the Phase 2/3 myopic score)."""
        b = board.belief()
        p_dig = b.best_dig()[1]
        if not self.lookahead:
            return p_dig
        p_listen = b.p_leak * float(np.max(self.listen @ b.post)) * ACOUSTIC_P_DETECT * 0.85
        return max(p_dig, p_listen)

    def expected_gain(self, opt: Option, board: EvidenceBoard) -> float:
        """Expected increase of the board's value after seeing the result (Monte Carlo over possible results)."""
        b0 = board.belief()
        now = self.value(board)
        vals = []
        for _ in range(self.n_samples):
            j = int(self.rng.choice(len(b0.post), p=b0.post)) if self.rng.random() < b0.p_leak else -1
            nb = board.copy()
            self._apply(opt, self._simulate(opt, j, board), nb)
            vals.append(self.value(nb))
        return float(np.mean(vals) - now)

    def _simulate(self, opt: Option, j: int, board: EvidenceBoard) -> dict:
        """A possible result if the leak were at j (j = -1: no leak) - from OUR model of the tools."""
        rng, a = self.rng, opt.args
        if opt.kind == "acoustic":
            if j >= 0 and self.D[a["c"], j] <= a["R"] and rng.random() < ACOUSTIC_P_DETECT:
                near = np.where(np.abs(self.D[j] - abs(rng.normal(0, ACOUSTIC_LOC_SIGMA_M))) < 30)[0]
                return {"detected": True, "approx": int(rng.choice(near)) if len(near) else j}
            if rng.random() < ACOUSTIC_FALSE_POS:
                return {"detected": True, "approx": int(rng.choice(np.where(self.D[a["c"]] <= a["R"])[0]))}
            return {"detected": False, "approx": None}
        if opt.kind == "step":
            inside = j >= 0 and j in set(a["S"].tolist())
            q = max(rng.normal(board.q_hat, board.q_sd), 0.0) if inside else 0.0
            return {"excess": q + rng.normal(0, STEP_NOISE_LPS)}
        if opt.kind == "logger":
            n = a["n"]
            pred = self.eng.lib.base_p_night[n] + a["offset"] - (board.q_hat * self.eng.lib.g_night[j, n] if j >= 0 else 0)
            return {"reading": pred + rng.normal(0, LOGGER_SIGMA_M)}
        if opt.kind == "ask":
            return {"reply": self._simulate_reply(board.reports[a["rid"]], j, a.get("options", []))}
        raise ValueError(opt.kind)

    def _simulate_reply(self, it: ReportItem, j: int, options: list) -> dict:
        rng = self.rng
        if rng.random() > 0.7:
            return {}                                                      # no reply
        if j < 0 or rng.random() < it.eps:                                  # the report is unrelated to a leak
            if rng.random() < 0.5:
                return {"non_supply": True}
            n = int(rng.choice(len(self.rep.background), p=self.rep.background))
            src = "unknown"
        else:
            a = {"yes": 0.9, "no": 0.15}.get(it.src, 0.6)
            k = a * self.rep.k_near[j] + (1 - a) * self.rep.k_down[j]
            n = int(rng.choice(len(k), p=k / k.sum()))
            src = "yes" if np.hypot(*(self.net.xy[n] - self.net.xy[j])) < 40 else "no"
        rep = {"source_seen": src, "mentions": [], "upstream": [], "confirms_water": src != "unknown"}
        st, lm = self.gaz.node_street(n), self.gaz.node_landmark(n)
        if st:
            rep["mentions"].append({"text": st["name"], "type": "street"})
        if lm:
            rep["mentions"].append({"text": lm["name"], "type": "landmark"})
        if src == "no" and j >= 0 and rng.random() < 0.6:
            up = self.gaz.node_landmark(j, 200)
            if up:
                rep["upstream"].append({"text": up["name"], "type": "landmark"})
        return rep

    def _apply(self, opt: Option, out: dict, board: EvidenceBoard):
        """Fold one result into the evidence board."""
        a = opt.args
        if opt.kind == "acoustic":
            lik, l0 = self.om.acoustic(a["c"], a["R"], out["detected"], out["approx"])
            board.add_check(opt.oid, lik, l0)
        elif opt.kind == "step":
            lik, l0 = self.om.step_test(a["S"], out["excess"], board.q_hat, board.q_sd)
            board.add_check(opt.oid, lik, l0)
        elif opt.kind == "logger":
            lik, l0 = self.om.logger(a["n"], out["reading"], board.q_hat, a["offset"])
            board.add_check(opt.oid, lik, l0)
        elif opt.kind == "ask":
            if out.get("reply"):
                board.clarify_report(a["rid"], out["reply"], self.gaz, self.gaz.mention_nodes)

    # ================================================================== run in the environment
    def execute(self, opt: Option, board: EvidenceBoard, env, question: str | None = None) -> dict:
        """Do the check for real; return the raw environment result (errors are returned, not raised)."""
        J, a = self.net.junctions, opt.args
        if opt.kind == "acoustic":
            r = env.acoustic_survey(J[a["c"]], a["R"])
            out = r.get("ok") and {"detected": r["detected"],
                                   "approx": self.net.index.get(r.get("approx_node")) if r["detected"] else None}
        elif opt.kind == "step":
            nodes = [J[i] for i in a["S"]]
            token, why = self.approve(env, "step_test", nodes, board, f"night step test around {self._name(a['S'][0])}")
            if token == "AUTO":
                r = env.step_test(nodes, approved=True)
            else:
                r = env.step_test(nodes, approval_token=token) if token else {"ok": False, "error": f"operator: {why}"}
            out = r.get("ok") and {"excess": r["inflow_drop_lps"] - r["expected_customer_drop_lps"]}
        elif opt.kind == "logger":
            r = env.place_logger(J[a["n"]])
            out = r.get("ok") and {"reading": r["night_mean_pressure_m"]}
            if r.get("ok"):
                self.state["loggers"].add(J[a["n"]])
        elif opt.kind == "ask":
            labels = [o["label"] for o in a.get("options", [])]
            r = env.ask_reporter(a["rid"], question or a["question"], a.get("options", []))
            self.state["asked"][a["rid"]] = self.state["asked"].get(a["rid"], 0) + 1
            reply = self.reader.read_reply(r.get("reply_text") or "", labels) if r.get("ok") else None
            if reply is not None and reply.get("contains_instructions"):
                reply = {}                                   # never act on instructions inside a reply
            if reply:
                reply["options"] = a.get("options", [])
            out = r.get("ok") and {"reply": reply or {}}
            r["parsed"] = reply
        else:
            raise ValueError(opt.kind)
        if out:
            self._apply(opt, out, board)
            self.state["spent"] += opt.cost
        elif str(r.get("error", "")).startswith("field check failed"):   # robustness suite: the night was paid for
            self.state["spent"] += opt.cost
        self.state["history"].append({"t_h": round((env.now - self.case.case_open_s) / 3600, 1), "action": opt.oid,
                                      "ok": bool(r.get("ok")), "error": r.get("error")})
        self.add_reports(board, [x for x in env.observe()["reports"]])
        return r

    def dig(self, board: EvidenceBoard, env) -> bool:
        b = board.belief()
        j, p = b.best_dig()
        token, why = self.approve(env, "dig", self.net.junctions[j], board, f"best-supported spot, P(success) {p:.2f}")
        if token is None:
            return False
        r = (env.request_dig(self.net.junctions[j], approved=True) if token == "AUTO"
             else env.request_dig(self.net.junctions[j], approval_token=token))
        if r.get("leak_found"):
            return True
        board.add_check(f"failed_dig@{self.net.junctions[j]}", 1 - b.cover[j] * 0.9, 1.0)
        return False

    # ================================================================== asking the human operator
    def certificate(self, board: EvidenceBoard, node: str, reason: str = "") -> dict:
        """The 'evidence certificate' the operator sees: the spot, the numbers from code, and the evidence IDs."""
        b = board.belief()
        evidence = ["E-P (pressure + inflow analysis)"]
        evidence += [f"{rid} (report, trust {1 - it.eps:.2f})" for rid, it in board.reports.items() if it.eps < 0.9]
        evidence += [lbl for lbl, _, _ in board.checks]
        return {"node": node, "p_success": round(b.best_dig()[1], 3),
                "explanations": {k: round(float(v), 3) for k, v in board.explanations().items()},
                "evidence": evidence, "agent_reason": reason[:300]}

    def approve(self, env, action: str, target, board: EvidenceBoard, reason: str = ""):
        """Ask the duty operator. Returns (token or None, operator's reason). Every Phase-3 method goes through this.
        In the Phase-2 setting (operator=False) approval is automatic and recorded as such."""
        if not self.use_operator:
            return "AUTO", "auto-approved (phase 2 setting)"
        node = target if action == "dig" else target[0]
        d = env.request_approval(action, target, self.certificate(board, node, reason))
        return (d["token"] if d.get("approved") else None), d.get("reason")

    # ================================================================== alternative explanations (abduction checks)
    def test_event(self, board: EvidenceBoard, node: str, start_s: float, end_s: float, prior: float, name: str,
                   spread: bool = False):
        """Hypothesis: a temporary water use (flushing, fire brigade, valve works) at `node` between start and end
        caused the anomaly. Hydraulically it looks like a leak at that node, so the pressure data fits it as well as
        a leak there; the difference shows up later: an event ENDS, a leak does not (see recheck)."""
        j = self.net.index[node]
        board.add_alt(name, prior, float(board.llp[j]))
        board.alts[name].update({"kind": "event", "node": node, "j": j, "start_s": start_s, "end_s": end_s,
                                 "spread": spread})
        self.state["alts_tested"][name] = {"kind": "event", "node": node, "start_s": start_s, "end_s": end_s}
        self._onset_check(board, start_s, name)
        return board.explanations()

    def _onset_check(self, board: EvidenceBoard, start_s: float, name: str, memory_h: float = 24.0):
        """Did the extra inflow begin when the event began? Look at the hours just BEFORE the event start.
        - event only: no extra inflow before the start;
        - leak: a leak started at a random time in the last day, so usually it was already there before.
        The answer is a likelihood ratio, like any other check on the board."""
        c = self.case
        if not (5 * 86400 + 2 * 3600 < start_s < c.case_open_s):
            return
        ev = build_evidence(c.time_s, c.pressure, c.flow, c.tank_level, start_s,
                            window_h=min(6.0, (start_s - 5 * 86400) / 3600))
        b, sd_b = ev.q_hat_lps, float(np.sqrt(ev.q_var + 0.5 ** 2))
        pi = float(np.clip(1 - (c.case_open_s - start_s) / (memory_h * 3600), 0.5, 0.95))
        g = lambda x, m, sd: float(np.exp(-0.5 * ((x - m) / sd) ** 2) / sd)
        l_event = g(b, 0.0, sd_b)
        l_leak = pi * g(b, board.q_hat, float(np.hypot(sd_b, board.q_sd))) + (1 - pi) * l_event
        top = max(l_event, l_leak)
        label = f"onset_check:{name}"
        board.add_check(label, np.full(len(board.base), l_leak / top), l_event / top)
        board.alts[name]["onset"] = {"inflow_before_lps": round(b, 2), "inflow_now_lps": round(board.q_hat, 2)}

    def test_sensor_fault(self, board: EvidenceBoard, sensor: str, prior: float, name: str):
        """Hypothesis: sensor `sensor` is faulty and there is no leak. We re-read the pressure evidence WITHOUT that
        sensor: if the other sensors look normal, the fault explains everything."""
        keep = [i for i, n in enumerate(self.case.pressure_nodes) if n != sensor]
        ev = self._sub_evidence(keep)
        _, ll0_wo, _, _ = pressure_loglik(ev, self.eng.lib, [self.sidx[i] for i in keep],
                                          rel_model_err=self.eng.cfg.rel_model_err)
        board.add_alt(name, prior, self.eng.cfg.tau_p * ll0_wo)
        board.alts[name].update({"kind": "sensor_fault", "sensor": sensor})
        self.state["alts_tested"][name] = {"kind": "sensor_fault", "sensor": sensor}
        return board.explanations()

    def exclude_sensor(self, board: EvidenceBoard, sensor: str):
        """Stop trusting one sensor: rebuild the pressure part of the board from the other sensors."""
        keep = [i for i, n in enumerate(self.case.pressure_nodes) if n != sensor]
        ev = self._sub_evidence(keep)
        llp, ll0, _, _ = pressure_loglik(ev, self.eng.lib, [self.sidx[i] for i in keep],
                                         rel_model_err=self.eng.cfg.rel_model_err)
        tp = self.eng.cfg.tau_p
        board.base, board.base_none, board.llp = self.eng.log_prior + tp * llp, tp * ll0, tp * llp
        board.alts.pop(f"sensor_fault:{sensor}", None)
        self.state.setdefault("excluded_sensors", []).append(sensor)
        return board.explanations()

    def _sub_evidence(self, keep: list[int]):
        import copy
        ev = copy.copy(self.ev0)
        ev.drop_m, ev.drop_var = self.ev0.drop_m[keep], self.ev0.drop_var[keep]
        ev.beta_m_per_lps, ev.resid_sigma = self.ev0.beta_m_per_lps[keep], self.ev0.resid_sigma[keep]
        return ev

    def recheck(self, board: EvidenceBoard, env, label: str | None = None):
        """Look at the NEWEST data (after the case opened, and after every suspected event ended): is the anomaly
        still there? A leak persists; a finished event or a no-leak situation shows normal inflow and pressures.

        Only NEW data counts (the hours before the case opened are already on the board), and only the latest
        recheck is kept: a second recheck replaces the first, so the same hours are never counted twice. The
        pressure part is tempered like the main engine (tau_p), because drop estimates from different windows
        differ by more than their noise (model error)."""
        obs = env.observe()
        if "flow" not in obs:
            return {"ok": False, "error": "no new flow data available"}
        keep = [i for i, n in enumerate(self.case.pressure_nodes) if n not in self.state.get("excluded_sensors", [])]
        ended = [a["end_s"] for a in board.alts.values() if a.get("kind") == "event" and a["end_s"] <= env.now]
        since = max([env.now - 6 * 3600, self.case.case_open_s] + [e + 1800 for e in ended])
        window_h = (env.now - since) / 3600
        if window_h < 1.5 - 1e-6:
            return {"ok": False, "wait_h": round(1.5 - window_h + 0.05, 2),
                    "error": f"too early: only {max(window_h, 0):.1f} h of new data; wait about "
                             f"{1.5 - window_h:.1f} more hours"}
        ev2 = build_evidence(obs["time_s"], obs["pressure"][:, keep], obs["flow"], obs["tank_level"], env.now,
                             window_h=window_h)
        sd_q = float(np.sqrt(ev2.q_var + 0.5 ** 2))
        g = lambda x, m, sd: float(-0.5 * ((x - m) / sd) ** 2 - np.log(sd))
        d0 = self.ev0.drop_m[keep]
        var = ev2.drop_var + self.ev0.drop_var[keep] + (0.25 * d0) ** 2 + 0.03 ** 2
        tp = self.eng.cfg.tau_p
        persist = g(ev2.q_hat_lps, board.q_hat, np.sqrt(sd_q ** 2 + board.q_sd ** 2)) + \
            tp * float(np.sum(-0.5 * (ev2.drop_m - d0) ** 2 / var))
        gone = g(ev2.q_hat_lps, 0.0, sd_q) + tp * float(np.sum(-0.5 * ev2.drop_m ** 2 / var))
        # the newest recheck replaces any earlier one (no double counting of the same hours)
        old = [lbl for lbl, _, _ in board.checks if lbl.startswith("recheck")]
        board.checks = [c for c in board.checks if not c[0].startswith("recheck")]
        for a in board.alts.values():
            for lbl in old:
                a["extra"].pop(lbl, None)
        label = label or f"recheck@{round((env.now - self.case.case_open_s) / 3600, 1)}h"
        board.add_check(label, np.full(len(board.base), np.exp(persist - max(persist, gone))),
                        float(np.exp(gone - max(persist, gone))))
        for a in board.alts.values():
            still_running = a.get("kind") == "event" and env.now < a.get("end_s", 0)
            a["extra"][label] = (persist if still_running else gone) - max(persist, gone)
        self.state["history"].append({"t_h": round((env.now - self.case.case_open_s) / 3600, 1), "action": label,
                                      "inflow_excess_lps": round(ev2.q_hat_lps, 2)})
        return {"ok": True, "inflow_excess_lps": round(ev2.q_hat_lps, 2), "explanations": board.explanations()}
