# AgentiX — Finding water leaks with few sensors, citizen reports and a careful investigation agent

LMA Group Project 2026 (Team AgentiX). Proposal: `AgentiX_Proposal.pdf`.
A control-room operator should get either a **verified** dig recommendation, or an honest *"not sure yet — here is the cheapest next check"*. The inputs are the few pressure sensors a city already has plus messages from citizens.

**Start here:** `docs/00_INDEX.md` (every document) · `docs/LEARNING_LOG.md` (plain-language findings) · `docs/02_PROJECT_GUIDE_5_PHASES.md` (plan and status).

## Repository map

```
agentix/                 Python package (pip install -e .)
  env/                   the simulated town ("truth side", hidden answers live here)
    network.py           network facts: zones, pipe distances, elevations
    real_model.py        official BattLeDIM truth model, time-sliced per case
    hydraulics.py        leak insertion + simulation (EPANET via WNTR)
    observe.py           sensor noise presets (field / battledim_exact / harsh)
    gazetteer.py         street, landmark and locality names on the network
    terrain.py           where leak water flows (downhill)
    reports.py           citizen-report GENERATOR (truth side only)
    scenario.py          builds one benchmark case (public + hidden parts)
    actions.py           CaseEnv: logger / ask reporter / acoustic / step test / dig, with costs
    loader.py            the ONLY way methods read a case (layout-restricted, public data)
  reader/                reading citizen text -> reader-1.0 JSON
    gazetteer_index.py   find place names, resolve them to network nodes
    rule_reader.py       B0 rule-based reader (baseline for the fine-tuned model)
  engine/                inference side: the evidence engine (no LLM)
    signatures.py        leak "fingerprint" library on the nominal model
    preprocess.py        demand-normalised pressure evidence, leak-size estimate
    pressure_lik.py      pressure likelihood for every junction
    report_lik.py        report likelihood (independent of the generator)
    fusion.py            posterior map, P(leak), dig spot, search plan
  active/                investigation loop (Phase 2b): belief, observation models, value of information, policies,
                         evidence board with alternative explanations (events, sensor faults), taint tracking
  agent/                 Phase 3 investigation agent
    toolbox.py           the 12 safe tools (action selector, code gate, operator token, SOP steps in code)
    board_view.py        the "case board" text the LLM reads (numbers from code, no raw citizen text)
    prompts.py           planner + critic prompts and JSON answer shapes
    llm_agent.py         AgentiX LLM investigator (planner loop, hypothesis tests, grounded-veto critic)
    rule_agent.py        matched non-LLM brain on the same toolbox
    llm_baselines.py     ReAct (raw text, no gate) and one-shot LLM baselines
    runner.py            builds a case for any brain and scores it (outcome + safety audit)
    timeparse.py         reads times like "1415 hrs" or "2 PM to 6 PM"
  llm/client.py          one LLM client (NVIDIA / Groq / Gemini / Mistral / Ollama), JSON answers, disk cache
  rag/retriever.py       BM25 search over the SOP + case work orders / maintenance log, with provenance tags
  safety/rules.py        rule book: what the SOP asks before an action; post-run audit of hard + procedure rules
  baselines/             RQ1 baselines (naive, hotspot, sensitivity, LILA-style, Bicik D-S, Mu-style gate)
  eval/                  truth access for scoring, metrics, statistics
configs/                 ltown.yaml (network, truth, noise), costs.yaml (action costs), rules.yaml (SOP rules)
schema/                  reader-1.0 JSON Schema (contract with the fine-tuning teammate)
scripts/
  build/                 prepare_real_model, build_signatures, build_gazetteer, build_benchmark, export_reader_seed,
                         build_heldout_docs (re-worded documents for the generalization test)
  qa/                    must-pass checks: spans, leak audit, noise, reproducibility, env smoke test
  analysis/              calibration and diagnostic analyses
  experiments/           run_static_eval, run_active_eval, report_static/active, make_phase2_figures,
                         run_agent_eval, report_agent, make_phase3_figures (Phase 3)
  pilots/                one-off pilots that informed decisions (kept for traceability)
tests/                   fast unit tests (python -m pytest tests)
docs/                    plans, research, decisions (ADRs), phase reports, plain-language notes
data/                    raw downloads, caches and generated benchmarks (large files are git-ignored)
runs/                    experiment outputs (git-ignored)
```

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e . wntr numpy pandas scipy networkx matplotlib pyyaml pyarrow jsonschema tabulate pytest
# data: BattLeDIM L-Town files into data/raw/ltown/ (Zenodo 4017659)
.venv/bin/python -I scripts/build/prepare_real_model.py data/raw/ltown/L-TOWN_Real.inp data/raw/ltown/L-TOWN.inp data/cache/ltown_real.npz
.venv/bin/python -I scripts/build/build_signatures.py configs/ltown.yaml data/cache/ltown_signatures.npz
.venv/bin/python -I scripts/build/build_benchmark.py configs/ltown.yaml data/benchmark/ltown_v2 S1_standard:300 S2_silent:60 S3_misleading:60 S4_noleak:60 S7_traps:50
.venv/bin/python -I scripts/experiments/run_static_eval.py data/benchmark/ltown_v2 test runs/phase2/static_test.jsonl
.venv/bin/python -I scripts/experiments/run_active_eval.py data/benchmark/ltown_v2 test runs/phase2/active_test.jsonl
```

Phase 3 (LLM agent) needs a free LLM key in a project `.env` file (never commit it), e.g. `NVIDIA_API_KEY=...`
(see ADR-011). Then:

```bash
.venv/bin/python -I scripts/experiments/run_agent_eval.py --split test --per-suite 12 \
    --methods greedy_voi,rule_agent,agentix_llm,react_llm,oneshot_llm --out runs/phase3/test --workers 4
.venv/bin/python -I scripts/experiments/report_agent.py runs/phase3/test/rows.jsonl docs/phase3/results_test.md
```

Guard-rails (physics in code, untrusted text quarantined, gated digs, no inverse crime, matched budgets) are listed in `CLAUDE.md` and `docs/02_PROJECT_GUIDE_5_PHASES.md`.
