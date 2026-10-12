# AgentiX — water-leak investigation agent (LMA Group Project 2026)

Few pressure sensors + citizen reports + a physics simulator + a careful LLM investigation agent with a code-enforced safety gate.

## Read first
- `docs/00_INDEX.md` (map of all docs)
- `docs/02_PROJECT_GUIDE_5_PHASES.md` (current phase, guard-rails, status table)
- `docs/decisions/` (decisions already taken; never contradict one silently)
- `docs/CARRY_FORWARD_CHECKLIST.md` (every case/technique found so far, with the phase that must handle it; read at each phase start)

## How to work here
- The phase guide is a compass, not a recipe. At each phase start, check the real system state, think deeply, compare at least 3 design options, run a small pilot if unsure, and record an ADR in `docs/decisions/`.
- The user wants sophisticated, research-grade methods, especially for the reasoning and agentic module (`docs/05_...`). Every LLM role must show measurable value against a matched non-LLM baseline.

## Guard-rails (non-negotiable)
1. Physics and probability maths in code (WNTR/numpy). The LLM never produces numbers that decisions depend on.
2. Raw citizen text, replies and retrieved docs are untrusted. Only the quarantined reader sees raw text. Untrusted data never controls tool calls.
3. `dig`, `step_test` and valve operations need a code gate plus a human approval token.
4. No inverse crime: generate truth with `L-TOWN_Real.inp` (or perturbed models); infer with the nominal model. The report generator must differ from the report likelihood.
5. Matched budgets for all compared agents; the same sensor layouts for all methods; test sets frozen before final runs.
6. Log every agent step (state, belief, VOI table, action, rationale, tool I/O, tokens).

## Current state (update every phase)
- Phase 1 (foundations) built: `agentix/env/*` (truth hydraulics, calibrated sensor noise, gazetteer, terrain, report generator, case builder, `CaseEnv` actions, `loader.py`), `agentix/engine/signatures.py`.
- Phase 2 (engine) built: official truth `agentix/env/real_model.py` (benchmark v2 = `data/benchmark/ltown_v2`), reader `agentix/reader/`, engine `agentix/engine/{preprocess,pressure_lik,report_lik,fusion}.py`, baselines `agentix/baselines/`, active investigation `agentix/active/`, evaluation `agentix/eval/` + `scripts/experiments/`.
- Phase 3 (agent) built: `agentix/agent/` (toolbox with action selector + code gate + operator token + SOP steps in code, case board, LLM investigator with plan-then-execute + abduction + critic, rule brain, ReAct/one-shot baselines, runner), `agentix/llm/client.py`, `agentix/rag/`, `agentix/safety/rules.py`, taint tracking in `agentix/active/investigation.py`, held-out document wording (`docs_heldout/`).
- LLM: free NVIDIA endpoint, `openai/gpt-oss-20b` (key in `.env` as `NVIDIA_API_KEY`, never commit). Free tiers throttle per model (HTTP 429); keep calls per case low and re-run failed runs (`run_agent_eval.py` does it).
- Plain-language findings: `docs/LEARNING_LOG.md`. Technical: `docs/phase1/PHASE1_REPORT.md`, `docs/phase2/PHASE2_REPORT.md`, `docs/phase3/PHASE3_REPORT.md`, ADR-001 to ADR-011.
- Dev/test split is fixed: dev = case index % 10 in {0,1,2}; tune ONLY on dev.
- Python: `.venv/bin/python -I <script>` (the package is installed editable). Use ≤4 worker processes; the laptop has 8.6 GB RAM.
- Key commands:
  - Signatures: `scripts/build/build_signatures.py configs/ltown.yaml data/cache/ltown_signatures.npz`
  - Real truth cache: `scripts/build/prepare_real_model.py data/raw/ltown/L-TOWN_Real.inp data/raw/ltown/L-TOWN.inp data/cache/ltown_real.npz`
  - Benchmark (v2): `scripts/build/build_benchmark.py configs/ltown.yaml data/benchmark/ltown_v2 S1_standard:300 S2_silent:60 S3_misleading:60 S4_noleak:60 S7_traps:50`
  - Experiments: `scripts/experiments/run_static_eval.py`, `run_active_eval.py`, `report_static.py`, `report_active.py`, `make_phase2_figures.py`
  - Agents (Phase 3): `scripts/experiments/run_agent_eval.py --split dev|test --methods ... --out runs/phase3/<name>` (resumable; frozen case list in `<out>/cases.json`), `report_agent.py`, `make_phase3_figures.py`; held-out docs: `scripts/build/build_heldout_docs.py`
  - Seeds: never use Python `hash()` for seeds (it changes per process); use `agentix.env.actions.case_seed`.
  - QA (must pass, in `scripts/qa/`): `check_gold_spans.py`, `audit_leakage.py`, `validate_noise.py`, `check_reproducible.py`, `env_smoke_test.py`, plus `python -m pytest tests`
- Methods must read cases only via `agentix.env.loader.load_public_case` (layout-restricted). Never show internal case IDs or paths to an LLM.

## Ownership
The fine-tuned complaint reader is owned by a teammate. Its contract is in `docs/03_FINETUNE_REPORT_READER.md` (`reader-1.0`, `reply-1.0`, `link-1.0`). Change it only by version bump, and tell the team.
