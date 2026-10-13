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
            c = int(np.argmax(b.cover @ b.post))
            r = inv.execute(Option(f"acoustic@{inv.net.junctions[c]}", "acoustic", {"c": c, "R": 150.0},
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
