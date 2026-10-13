"""Evidence board: every piece of evidence is a separate, numbered item; the leak map is recomputed from all items.

Why: when a reporter later clarifies ("it is on 3rd Cross", "it comes from a house pipe") we must CHANGE that
report's evidence, not stack a new guess on top. Keeping items separate makes that exact, and gives the agent a
ledger of evidence IDs it can cite in explanations.

    log P(leak at j)  =  base(j)                    prior + pressure (engine, tempered)
                       + sum over reports  log L_r(j)
                       + sum over checks   log L_c(j)       acoustic, logger, step test, failed dig
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .belief import Belief


@dataclass
class ReportItem:
    rid: str
    w: np.ndarray                 # where the report points (weights over junctions)
    src: str                      # "yes" (water comes out here) / "no" (only collected here) / "unknown"
    eps: float                    # chance the report is unrelated to the leak
    w_up: np.ndarray | None = None  # "flowing from the side of X" -> where the source probably is
    notes: list = field(default_factory=list)
    cache: tuple | None = None    # saved (log-lik, log-lik if no leak); cleared whenever the item changes


class EvidenceBoard:
    def __init__(self, rep_model, cover, base_logp: np.ndarray, base_lognone: float, prior_leak: float,
                 q_hat: float, q_sd: float):
        self.rep, self.cover = rep_model, cover
        self.base, self.base_none = base_logp, base_lognone
        self.prior_leak, self.q_hat, self.q_sd = prior_leak, q_hat, q_sd
        self.reports: dict[str, ReportItem] = {}
        self.checks: list[tuple[str, np.ndarray, float]] = []      # (label, log-lik over j, log-lik if no leak)
        # alternative explanations (not a leak): name -> {"logprior", "base", "extra": {check_label: loglik}}
        self.alts: dict[str, dict] = {}

    # ------------------------------------------------------------------ report likelihood
    def report_loglik(self, it: ReportItem) -> tuple[np.ndarray, float]:
        if it.cache is not None:
            return it.cache
        a = {"yes": 0.9, "no": 0.15}.get(it.src, 0.6)
        related = a * (self.rep.k_near @ it.w) + (1 - a) * (self.rep.k_down @ it.w)
        if it.w_up is not None:                       # source is near the place the water flows FROM
            up = self.rep.k_near @ it.w_up
            related = related * (0.2 + 0.8 * up / max(up.max(), 1e-12))
        bg = float(self.rep.background @ it.w)
        lik = (1 - it.eps) * related + it.eps * bg
        it.cache = (np.log(lik + 1e-300), float(np.log(bg + 1e-300)))
        return it.cache

    # ------------------------------------------------------------------ the map
    def logpost(self):
        lp = self.base.copy()
        lnone = self.base_none
        for it in self.reports.values():
            l, l0 = self.report_loglik(it)
            lp += l
            lnone += l0
        for _, l, l0 in self.checks:
            lp += l
            lnone += l0
        return lp, lnone

    def explanations(self) -> dict:
        """Probability of each explanation: 'leak', 'none' and every alternative (e.g. planned flushing)."""
        lp, lnone = self.logpost()
        ev = logsumexp(lp)
        p_alt_total = sum(np.exp(a["logprior"]) for a in self.alts.values())
        log_w = {"leak": np.log(self.prior_leak) + ev,
                 "none": np.log((1 - self.prior_leak) * max(1 - p_alt_total, 1e-6)) + lnone}
        rep0 = sum(self.report_loglik(it)[1] for it in self.reports.values())
        for name, a in self.alts.items():
            # under an alternative there is no leak: checks behave as 'no leak' unless the check says otherwise for
            # this alternative (a["extra"]). Reports: a temporary event at junction j (flushing, fire brigade) puts
            # water on the street just like a leak at j would, so reports are scored as if the source were at j;
            # for a sensor fault the reports are unrelated.
            chk = sum(a["extra"].get(lbl, l0) for lbl, _, l0 in self.checks)
            if a.get("spread"):
                # a guessed event with no cited source: its place was picked AFTER looking at the data, so it gets
                # no location credit (Occam): pressure + reports are scored over all places, like the leak
                lp0 = self.base + sum(self.report_loglik(it)[0] for it in self.reports.values())
                log_w[name] = a["logprior"] + np.log(1 - self.prior_leak) + logsumexp(lp0) + chk
                continue
            if a.get("j") is not None:
                rep = sum(self.report_loglik(it)[0][a["j"]] for it in self.reports.values())
            else:
                rep = rep0
            log_w[name] = a["logprior"] + np.log(1 - self.prior_leak) + a["base"] + rep + chk
        z = logsumexp(list(log_w.values()))
        return {k: float(np.exp(v - z)) for k, v in log_w.items()}

    def belief(self) -> Belief:
        lp, _ = self.logpost()
        post = np.exp(lp - logsumexp(lp))
        p_leak = self.explanations()["leak"]
        return Belief(post, p_leak, self.q_hat, self.q_sd, self.cover)

    def add_alt(self, name: str, prior: float, base_loglik: float):
        """Register an alternative explanation (prior probability among non-leak causes, data log-likelihood)."""
        self.alts[name] = {"logprior": float(np.log(max(prior, 1e-6))), "base": float(base_loglik), "extra": {}}

    # ------------------------------------------------------------------ edits
    def add_check(self, label: str, lik: np.ndarray, lik_none: float):
        self.checks.append((label, np.log(np.maximum(lik, 1e-300)), float(np.log(max(lik_none, 1e-300)))))

    def clarify_report(self, rid: str, reply: dict, gaz, place_weights_fn):
        """Apply a parsed reply (see RuleReader.read_reply) to report `rid`."""
        it = self.reports.get(rid)
        if it is None:
            return
        it.cache = None
        if reply.get("non_supply"):
            it.eps = 0.99
            it.notes.append("reporter says it is not a public water pipe")
            return
        opts, k = reply.get("options") or [], reply.get("option_index")
        if opts and k is not None:                       # answer to "near which of these places?"
            if k < len(opts):
                w2 = gaz.place_nodes(opts[k]["place_id"])
                cos = float(it.w @ w2 / np.sqrt((it.w @ it.w) * (w2 @ w2) + 1e-300))
                it.w = (it.w * w2) if cos > 0.05 else (0.3 * it.w + 0.7 * w2)
                it.notes.append(f"chose option: {opts[k]['label']}")
            else:                                        # "not near any of those"
                m = np.zeros_like(it.w)
                for o in opts:
                    w2 = gaz.place_nodes(o["place_id"])
                    m = np.maximum(m, w2 / max(w2.max(), 1e-12))
                it.w = it.w * (1 - 0.8 * m)
                it.notes.append("not near any offered option")
            it.w = it.w / it.w.sum()
        if reply.get("source_seen") in ("yes", "no"):
            it.src = reply["source_seen"]
            it.notes.append(f"source_seen={it.src}")
        for m in reply.get("mentions", []):
            w2 = place_weights_fn(m["text"], m["type"])
            if w2 is None:
                continue
            cos = float(it.w @ w2 / np.sqrt((it.w @ it.w) * (w2 @ w2) + 1e-300))
            it.w = (it.w * w2) if cos > 0.05 else (0.3 * it.w + 0.7 * w2)    # reply is usually more exact
            it.w = it.w / it.w.sum()
            it.notes.append(f"place: {_canon(gaz, m['text'])}")
        for m in reply.get("upstream", []):
            w2 = place_weights_fn(m["text"], m["type"])
            if w2 is not None:
                it.w_up = w2 if it.w_up is None else (it.w_up + w2) / 2
                it.notes.append(f"flows from: {_canon(gaz, m['text'])}")
        if reply.get("confirms_water") and it.eps > 0.3:
            it.eps = 0.3                                # a person who answers about real water is more credible

    def copy(self) -> "EvidenceBoard":
        b = EvidenceBoard(self.rep, self.cover, self.base, self.base_none, self.prior_leak, self.q_hat, self.q_sd)
        b.reports = {k: ReportItem(v.rid, v.w.copy(), v.src, v.eps, None if v.w_up is None else v.w_up.copy(),
                                   list(v.notes), v.cache) for k, v in self.reports.items()}
        b.checks = list(self.checks)
        b.alts = {k: {**v, "extra": dict(v["extra"])} for k, v in self.alts.items()}
        for k in ("misfit", "base_pressure_none", "llp"):
            if hasattr(self, k):
                setattr(b, k, getattr(self, k))
        return b


def _canon(gaz, text: str) -> str:
    """Canonical gazetteer name for a place mention (so notes never carry raw citizen text)."""
    ids = gaz.resolve(text)
    return gaz.places[ids[0]]["name"] if ids else "an unknown place"
