"""Step 4: combine evidence into a posterior map, a dig decision and a ranked search plan.

posterior(j) ∝ prior(j) * L_pressure(j)^tau_p * L_reports(j)^tau_r
  prior(j) ∝ pipe length attached to j (leaks happen per km of pipe)
  tau_p, tau_r: tempering (<1) because the nominal map is wrong -> raw likelihoods are over-confident.
P(leak): compares the best leak explanation with "no leak" (q = 0, all reports unrelated).
Dig spot: the junction whose 100 m pipe-neighbourhood holds the most posterior mass (max-coverage, crew view).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import datetime as dt

import numpy as np
from scipy.special import logsumexp

from ..env.network import NetworkInfo
from .preprocess import Evidence, build_evidence
from .pressure_lik import SignatureLibrary, pressure_loglik
from .report_lik import ReportModel


@dataclass
class EngineConfig:
    use_pressure: bool = True
    use_reports: bool = True
    tau_p: float = 0.3                # tuned on dev with a proper scoring rule (ADR-006): pressure is over-confident
    tau_r: float = 1.0
    base_eps: float = 0.5             # tuned on dev
    tau_age_h: float = 12.0           # report recency scale (6 h was marginally best on dev; 12 h is less benchmark-specific)
    window_h: float = 6.0             # pressure evidence window before case open
    prior_leak: float = 0.8           # P(leak | a case was opened)
    dig_radius_m: float = 90.0        # coverage radius used to choose the dig spot (success counted at 100 m)
    rel_model_err: float = 0.25
    n_search_areas: int = 10


@dataclass
class EngineResult:
    posterior: np.ndarray
    p_leak: float
    dig_node: str
    dig_coverage: float               # P(leak within dig radius of dig_node | leak)
    confidence: float                 # p_leak * dig_coverage
    search_plan: list = field(default_factory=list)   # [(node, cumulative mass)]
    diagnostics: dict = field(default_factory=dict)


class EvidenceEngine:
    def __init__(self, net: NetworkInfo, lib: SignatureLibrary, reports: ReportModel, cfg: EngineConfig | None = None):
        self.net, self.lib, self.rep = net, lib, reports
        self.cfg = cfg or EngineConfig()
        L = np.zeros(len(net.junctions))
        for u, v, data in net.pipe_graph.edges(data=True):
            for n in (u, v):
                if n in net.index:
                    L[net.index[n]] += data["length"] / 2
        self.log_prior = np.log(L / L.sum() + 1e-12)
        self.cover = (net.pipe_distance <= self.cfg.dig_radius_m).astype(np.float32)   # [J,J]

    def infer(self, case, reader_outs: list[dict], evidence: Evidence | None = None) -> EngineResult:
        cfg = self.cfg
        J = len(self.net.junctions)
        sensor_idx = [self.net.index[n] for n in case.pressure_nodes]
        ev = evidence or build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s,
                                        window_h=cfg.window_h)
        t_open = dt.datetime.fromisoformat(case.start_datetime) + dt.timedelta(seconds=case.case_open_s)
        ages = {r["report_id"]: (t_open - dt.datetime.fromisoformat(r["received_at"])).total_seconds() / 3600
                for r in case.reports}
        llp, llp0, q_map, misfit = (pressure_loglik(ev, self.lib, sensor_idx, rel_model_err=cfg.rel_model_err)
                                    if cfg.use_pressure else (np.zeros(J), 0.0, np.zeros(J), 0.0))
        llr, llr0, used = (self.rep.loglik(reader_outs, cfg.base_eps, ages, cfg.tau_age_h) if cfg.use_reports
                           else (np.zeros(J), 0.0, []))
        logp = self.log_prior + cfg.tau_p * llp + cfg.tau_r * llr
        log_ev_leak = logsumexp(logp)
        post = np.exp(logp - log_ev_leak)
        log_ev_none = cfg.tau_p * llp0 + cfg.tau_r * llr0
        lo = np.log(cfg.prior_leak / (1 - cfg.prior_leak)) + log_ev_leak - log_ev_none
        p_leak = float(1 / (1 + np.exp(-np.clip(lo, -50, 50))))
        cov = self.cover @ post
        c = int(np.argmax(cov))
        plan, remaining, cum = [], post.copy(), 0.0
        for _ in range(cfg.n_search_areas):
            cv = self.cover @ remaining
            k = int(np.argmax(cv))
            cum += float(cv[k])
            plan.append((self.net.junctions[k], cum))
            remaining = remaining * (1 - self.cover[k])
        ent = float(-(post * np.log(post + 1e-300)).sum())
        return EngineResult(
            posterior=post, p_leak=p_leak, dig_node=self.net.junctions[c], dig_coverage=float(cov[c]),
            confidence=p_leak * float(cov[c]), search_plan=plan,
            diagnostics={"t0_h_before_open": (ev.open_s - ev.t0_s) / 3600, "q_hat_lps": ev.q_hat_lps,
                         "q_sd_lps": float(np.sqrt(ev.q_var)), "shift_score": ev.shift_score,
                         "pressure_misfit": misfit, "q_map_at_dig": float(q_map[c]) if cfg.use_pressure else None,
                         "reports_used": used, "entropy_nats": ent, "eff_n_nodes": float(np.exp(ent))})
