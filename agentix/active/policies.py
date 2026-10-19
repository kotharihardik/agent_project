"""Non-LLM investigation policies (Phase 2b, updated in Phase 3 to the evidence board + shared menu).

They all use the same toolkit (`Investigation`): same checks, same answer models, same numbers. They differ only
in how they pick the next check. `GreedyVOI` is the strong non-LLM baseline the LLM agent must beat.
Every dig and step test goes through the simulated duty operator (approval token), exactly like the LLM agent.
"""
from __future__ import annotations

import numpy as np

from .investigation import COSTS, DELAY_H, Investigation, Option


class DigNow:
    """Dig at the best spot from the first look (no checks)."""
    name = "dig_now"

    def run(self, inv: Investigation, case, env):
        board = inv.start(case, env)
        inv.dig(board, env)


class DigIfConfident:
    """Dig only if the first look is already confident enough; otherwise hand over."""
    name = "dig_if_confident"

    def __init__(self, dig_threshold: float = 0.5):
        self.thr = dig_threshold

    def run(self, inv: Investigation, case, env):
        board = inv.start(case, env)
        if board.belief().best_dig()[1] >= self.thr:
            inv.dig(board, env)
        else:
            env.abstain("not confident")


class OneMoreSensor:
    """arXiv 2406.19900 (adapted): put a temporary logger at the current best spot, up to 3 times, then dig."""
    name = "one_more_sensor"

    def run(self, inv: Investigation, case, env, max_loggers: int = 3):
        board = inv.start(case, env)
        for _ in range(max_loggers):
            b = board.belief()
            n = next((int(i) for i in np.argsort(-b.post) if inv.net.junctions[int(i)] not in inv.state["loggers"]), None)
            if n is None:
                break
            off = inv.zone_offset.get(inv.net.zone_of[inv.net.junctions[n]], 0.0)
            r = inv.execute(Option(f"logger@{inv.net.junctions[n]}", "logger", {"n": n, "offset": off},
                                   COSTS["logger"], DELAY_H["logger"]), board, env)
            if not r.get("ok"):
                break
        inv.dig(board, env)


class AcousticSweep:
    """Utility practice: listen at the best areas in turn; dig as soon as something is heard (else best guess)."""
    name = "acoustic_sweep"

    def run(self, inv: Investigation, case, env, max_surveys: int = 3):
        board = inv.start(case, env)
        for _ in range(max_surveys):
            b = board.belief()
            c = int(np.argmax(inv.listen @ b.post))
            r = inv.execute(Option(f"acoustic@{inv.net.junctions[c]}", "acoustic", {"c": c, "R": inv.acoustic_r},
                                   COSTS["acoustic"], DELAY_H["acoustic"]), board, env)
            if not r.get("ok") or r.get("detected"):
                break
        inv.dig(board, env)


class RandomActions:
    """Floor: three random checks from the menu, then dig."""
    name = "random_actions"

    def run(self, inv: Investigation, case, env, n_actions: int = 3):
        board = inv.start(case, env)
        for _ in range(n_actions):
            opts = inv.menu(board)
            if not opts:
                break
            inv.execute(opts[int(inv.rng.integers(len(opts)))], board, env)
        inv.dig(board, env)


class GreedyVOI:
    """Ours, non-LLM: budgeted greedy information gathering for the dig decision (ADR-008).

    Each step: dig if P(success) >= threshold; otherwise take the menu item with the best expected gain per
    crew-unit while the budget lasts; stop and hand over when nothing is worth its cost.
    """
    name = "greedy_voi"

    def __init__(self, budget_cu: float = 1.6, max_steps: int = 10, max_digs: int = 2, dig_threshold: float = 0.5,
                 min_gain_per_cu: float = 0.05):
        self.budget, self.max_steps, self.max_digs = budget_cu, max_steps, max_digs
        self.thr, self.min_rate = dig_threshold, min_gain_per_cu

    def run(self, inv: Investigation, case, env):
        board = inv.start(case, env)
        digs = 0
        for _ in range(self.max_steps):
            p = board.belief().best_dig()[1]
            if p >= self.thr and digs < self.max_digs:
                digs += 1
                if inv.dig(board, env):
                    return
                continue
            opts = inv.menu(board, budget_left=self.budget - inv.state["spent"])
            inv.state["history"].append({"p_success": round(p, 3),
                                         "menu_top": [(o.oid, round(o.gain, 3)) for o in opts[:4]]})
            if not opts or opts[0].rate() < self.min_rate:
                break
            if not inv.execute(opts[0], board, env).get("ok"):
                break
        if not env.done:
            if board.belief().best_dig()[1] >= self.thr and digs < self.max_digs and inv.dig(board, env):
                return
            env.abstain("value of further checks below their cost; confidence below dig threshold")


class Oracle:
    """Ceiling: digs at the true leak."""
    name = "oracle"

    def run(self, inv: Investigation, case, env):
        if env._leak:
            env.request_dig(env._leak["nearest_node"], approved=True)
        else:
            env.abstain("oracle: no leak")


class CoarseToFine:
    """Ours, non-LLM (Phase 4): the search utilities actually use - narrow the area, then pinpoint, then dig.

    Each step:
      1. dig when P(success) >= threshold (after a failed dig the board updates and the search goes on);
      2. ask reporters first (free, quick): every question whose expected gain is worth it;
      3. if the belief is still spread out, STEP-TEST the 60-junction block holding most of it (one crew-night
         checks ~400 m around a spot, and the measured flow tells "leak inside" vs "not inside");
      4. once a block lights up, or the belief is concentrated (the best listening area holds >= `focus` of it),
         LISTEN there (250 m);
      5. stop when no night, budget or useful check is left, and hand over.
    Why: diagnosis on dev showed 64 of 76 missed leaks were never inside any listening area - the problem was
    WHERE to search, not hearing. Step tests cover about twice the ground of a listening survey.
    `next_check` is also the planner tool of the agent (agentix/agent/agent.py) and of the code-only ablation.
    """
    name = "coarse_to_fine"

    def __init__(self, budget_cu: float = 1.6, max_steps: int = 16, max_digs: int = 2, dig_threshold: float = 0.8,
                 focus: float = 0.3, min_ask_gain: float = 0.01, min_block: float = 0.05, reserve: float = 0.4,
                 final_threshold: float = 0.0):
        self.budget, self.max_steps, self.max_digs = budget_cu, max_steps, max_digs
        self.thr, self.focus, self.min_ask, self.min_block = dig_threshold, focus, min_ask_gain, min_block
        self.reserve = reserve        # budget kept back for listening once a step test lights up a block
        # when no check is left, dig if P(success) >= this "last chance" bar (0 = use the normal bar)
        self.final_thr = final_threshold or dig_threshold

    def run(self, inv: Investigation, case, env):
        board = inv.start(case, env)
        digs, memo = 0, {"tested": set(), "heard_at": set()}
        for _ in range(self.max_steps):
            if env.done:
                return
            if board.belief().best_dig()[1] >= self.thr and digs < self.max_digs:
                digs += 1
                if inv.dig(board, env):
                    return
                continue
            left = self.budget - inv.state["spent"]
            asks = [o for o in inv.menu(board, budget_left=left) if o.kind == "ask" and o.gain >= self.min_ask]
            opt = asks[0] if asks else self.next_check(inv, board, left, memo)
            if opt is None or not inv.execute(opt, board, env).get("ok"):
                break
        if not env.done:
            if board.belief().best_dig()[1] >= self.final_thr and digs < self.max_digs and inv.dig(board, env):
                return
            env.abstain("search finished without enough confidence for a dig")

    def next_check(self, inv: Investigation, board, left: float, memo: dict):
        """The next field check of the coarse-to-fine search (a step test or a listening survey), or None."""
        b = board.belief()
        post = b.post / b.post.sum() * b.p_leak
        listen_mass = inv.listen @ post
        listen_mass[list(memo["heard_at"])] = 0.0
        c = int(np.argmax(listen_mass))
        lit = self.lit_block(inv, post, memo["tested"])
        if lit is not None:                                   # a block showed extra flow: listen inside it
            inside = np.zeros(len(post))
            inside[lit] = 1.0
            lm = inv.listen @ (post * inside)
            lm[list(memo["heard_at"])] = 0.0
            if lm.max() > 0:
                c = int(np.argmax(lm))
        listen = Option(f"acoustic@{inv.net.junctions[c]}", "acoustic", {"c": c, "R": inv.acoustic_r},
                        COSTS["acoustic"], DELAY_H["acoustic"],
                        label=f"listen to pipes within {inv.acoustic_r:.0f} m of {inv._name(c)}")
        if (lit is not None or listen_mass[c] >= self.focus * b.p_leak) and left >= COSTS["acoustic"] - 1e-9:
            memo["heard_at"].add(c)
            return listen
        block = post[inv.step_sets].sum(1)
        block[list(memo["tested"])] = 0.0
        s = int(np.argmax(block))
        if left - COSTS["step"] >= self.reserve - 1e-9 and block[s] >= self.min_block * b.p_leak:
            memo["tested"].add(s)
            return Option(f"step@{inv.net.junctions[s]}", "step", {"S": inv.step_sets[s]}, COSTS["step"],
                          DELAY_H["step"], label=f"night step test of the 60 junctions around {inv._name(s)}")
        if left >= COSTS["acoustic"] - 1e-9 and listen_mass[c] > 0:
            memo["heard_at"].add(c)
            return listen
        return None

    @staticmethod
    def lit_block(inv: Investigation, post: np.ndarray, tested: set):
        """The tested block that now holds most of the leak belief (its step test showed extra flow), or None."""
        if not tested or post.sum() <= 0:
            return None
        c = max(tested, key=lambda x: post[inv.step_sets[x]].sum())
        S = inv.step_sets[c]
        return S if post[S].sum() >= 0.5 * post.sum() else None


class FixedRules:
    """The proposal's "Fixed rules, no LLM" baseline (Section 10), exactly as written there:
    "Always place one sensor at the most uncertain spot, then dig the top spot."
    Most uncertain spot = where the top suspect leak spots disagree most about the pressure drop a temporary sensor
    would see (Section 7, step 3): argmax over junctions n of the spread of q_hat * g_night[j, n] over the top-k j.
    """
    name = "fixed_rules"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def run(self, inv: Investigation, case, env):
        board = inv.start(case, env)
        b = board.belief()
        top = np.argsort(-b.post)[: self.top_k]
        pred = max(b.q_hat, 0.5) * inv.eng.lib.g_night[top]              # [k, J] predicted drop per junction
        spread = pred.std(0)
        spread[[inv.net.index[n] for n in inv.state["loggers"] if n in inv.net.index]] = -1.0
        n = int(np.argmax(spread))
        off = inv.zone_offset.get(inv.net.zone_of[inv.net.junctions[n]], 0.0)
        inv.execute(Option(f"logger@{inv.net.junctions[n]}", "logger", {"n": n, "offset": off},
                           COSTS["logger"], DELAY_H["logger"]), board, env)
        inv.dig(board, env)
        if not env.done:
            env.abstain("fixed rules: dig not approved")
