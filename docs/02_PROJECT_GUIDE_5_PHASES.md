# 02 — Project Guide in 5 Phases (for future Claude sessions and the team)

> **This is a compass, not a recipe.** It fixes the goal, the guard-rails, the questions to think through and the exit checks for each phase.
> It deliberately does **not** fix the implementation. At the start of every phase, look at the real state of the system and the results so far, think deeply, and choose the best design *then*.
> Companion docs:
> - `01_RESEARCH_FINDINGS.md` — what is known
> - `05_REASONING_AGENT_DESIGN_SPACE.md` — the menu of reasoning and agent techniques
> - `04_DATASETS_AND_EVALUATION.md` — data, suites, metrics
> - `03_FINETUNE_REPORT_READER.md` — the teammate's contract

---

## 0. How to work in each phase (protocol)

1. **Re-read:** the matching section here, the relevant parts of 01/04/05, and every decision record in `docs/decisions/`.
2. **Diagnose the current system:** what exists, what works, what the numbers say. Do not trust this guide over real evidence.
3. **Generate at least 3 design options** for each major decision. For each, write the expected gain, the cost, the risk, and how you would know it worked.
4. **Pilot when unsure:** a small experiment on dev data, a few hours at most, before committing.
5. **Record the decision** as `docs/decisions/ADR-NNN-<topic>.md` (context, options, decision, evidence, date).
6. **Build, then measure against the exit criteria.** Update the status table (§9).
7. **If evidence contradicts this guide, change the guide and record why.**

---

## 1. North star

> A control-room operator gets either a **verified** dig recommendation, or an honest *"not sure yet — here is the cheapest next check"*. The input is only the few sensors the city already has plus citizen reports.

**The headline result we aim for (shape, not numbers):**
- **Figure 1 — risk–coverage curves** at 3, 5 and 10 sensors, built up in five steps:
  1. pressure-only gated (Mu-style);
  2. + physics-informed citizen reports;
  3. + active investigation with a fixed (non-LLM) policy;
  4. + LLM investigation agent;
  5. + oracle (upper bound).

  The story: each layer reaches more cases at the same precision. This is how we "get past the pressure-information limit".
- **Table 1 — RQ2:** correct-dig rate, investigation cost, and unsafe/unapproved actions (target 0, pass^k) for our agent vs ReAct, plain LLM, fixed rules and greedy-VOI. All use the same model, tools and budget.
- **Table 2 — capability ablations:** one row per capability.
- **Figure 2 — equity:** hit rate and time-to-dig by reporting-propensity zone.
- **Analysis — intermediate reasoning:** how belief changes step by step, VOI regret, how often abduction helps, critic saves.

---

## 2. Non-negotiable guard-rails

1. **Physics and probabilities live in code** (WNTR + numpy). The LLM never produces a number that a decision depends on.
2. **Untrusted text never controls tools.** Citizen reports, reporter replies and retrieved documents pass through the quarantined reader (no tools, schema-constrained) or are tagged `untrusted`. Untrusted content may never fill tool arguments, except whitelisted IDs.
3. **Disruptive actions are gated in code and need human approval.** This covers `dig`, `step_test` and valve operations. There is no tool or prompt path around the gate.
4. **No inverse crime.**
   - Truth is generated with `L-TOWN_Real.inp` or perturbed models; inference uses the nominal model.
   - The citizen-report *generator* and the report *likelihood* use different structure or parameters.
   - Sweep misspecification on purpose.
5. **Fair comparisons.**
   - All agent variants get the same LLM, tools, step and token budget, cases and sensor layouts.
   - All localization methods get the same sensor layouts.
   - Baselines get honest prompt effort.
6. **Every claimed component has an ablation, and every capability has a metric** (course requirement).
7. **Freeze before final runs.** Test sets, metrics and thresholds are frozen first. Tune only on dev.
8. **Abstention is a valid result.** Measure it (coverage) and never hide it.
9. **Log every step:** state summary, belief stats, chosen action, EIG/VOI table, LLM rationale, tool input/output, errors, tokens, latency. The course grades analysis of intermediate steps.
10. **Report failures honestly,** with a failure taxonomy and concrete examples.

---

## 3. System concept (layers, not fixed modules)

```
 Citizen text ─────► [Reader: fine-tuned small LLM, NO tools] ─► structured report fields ─┐
 Reporter replies ─► [Reader: reply mode] ─────────────────────► structured answers ───────┤
 SCADA pressure/flow, temporary loggers ────────────────────────────────────────────────────┤
 Records DB (assets, repair history, work orders) ── SQL tools ─────────────────────────────┤
 Utility documents (SOPs, rules, notes) ── RAG (tagged untrusted) ──────────────────────────┤
                                                                                            ▼
            ┌──────────────── Evidence & Inference Engine (code, deterministic) ──────────────┐
            │ hypothesis space: leak@node×size, confounders, 2-leak, report fabrication       │
            │ likelihoods: pressure residuals · flow/MNF · physics-informed report model ·    │
            │              complaint-as-pressure-sensor · logger · acoustic · step test       │
            │ posterior + diagnostics (misfit, entropy, dig areas, conformal radius)          │
            │ action valuation: EIG and decision-focused VOI per cost for each action        │
            └───────────────────────────────┬────────────────────────────────────────────────┘
                                            ▼
            ┌──────────────── Investigation Agent (LLM reasoning + planning) ───────────────────┐
            │ reads belief summary + hypothesis ledger + EIG/VOI table + RAG rules/history      │
            │ abduction: proposes new testable hypothesis classes when the model misfits        │
            │ chooses next action; writes closed-form questions to reporters                    │
            │ falsification critic before any dig; drafts operator explanation with citations   │
            └───────────────────────────────┬──────────────────────────────────────────────────┘
                                            ▼
 [Safety gate (code): contract predicates · allow-list · rate limits · approval token] ─► Operator ─► dig / not yet
```

The layers stay. How each is built is decided phase by phase.

---

## Phase 1 — Foundations: digital twin, scenario + report simulator, benchmark format

**Goal.** Build a reproducible *environment* that:
- generates incident cases with known ground truth;
- answers any agent action (logger reading, reporter reply, step test, acoustic survey, dig outcome) from the **hidden truth**.

**Why first.** Baselines, the agent and the fine-tuning data all depend on it.

**Think through and decide:**
- **Networks for v1.** L-Town first. Also pick 1–2 transfer networks *now* so we don't overfit L-Town (suggested: C-Town + one KY, or Micropolis for realistic streets). See 04 §2.
- **Leak modelling:**
  - WNTR leak model (pressure-dependent orifice) vs added demand;
  - pressure-driven vs demand-driven analysis;
  - leak-size distribution (include *small* leaks);
  - abrupt vs incipient; start time; detection time;
  - leak location drawn uniformly, or weighted by a burst prior (pipe age and material, synthetic if missing).
- **Realism knobs:** demand noise, roughness and diameter errors (the Real model), sensor noise σ (0.05 m standard, 0.15 m stress), sensor faults (offset, drift, stuck), 2-leak cases.
- **Sensor layouts:** k ∈ {3, 5, 10} with ≥3 layouts each (random, engineering-style, information-optimised). Freeze them and share them across all methods.
- **Gazetteer** (streets, landmarks, localities with aliases):
  - how to give L-Town street names: group pipes into "streets" by connectivity and straightness;
  - place landmarks according to demand and centrality as a proxy for footfall;
  - generate **several fictional towns** in the same format so the reader learns to extract rather than memorise. Hold one town out for testing.
- **Physics-informed report generator:**
  - surfacing probability;
  - trench travel;
  - downhill routing on the elevation graph, with pooling at low points;
  - exposure (demand, landmarks, time of day);
  - **reporting propensity by area** (heterogeneous, up to 3×);
  - delays and duplicates;
  - false and unrelated reports (sewer, rain, private leaks);
  - **low-pressure / no-water complaints** derived from simulated nodal pressure;
  - adversarial reports.

  Keep generator parameters in a separate config from inference parameters.
- **Environment API for actions,** with costs and delays from a config file (see 05 §4). Hidden truth answers each action. Reporter replies come from an LLM *user simulator* that knows only what the persona "saw". Validate that it does not leak the truth.
- **Case file format** (04 §6) and **signature library** (precompute per network, leak size and time slot; store compactly; check speed).
- **Hygiene:** no ground-truth leakage into observations (IDs, filenames, ordering).

**Pitfalls:**
- data that is too clean;
- generator and inference sharing identical assumptions;
- slow simulations (vectorise and cache).

**Exit criteria:**
- [ ] ≥300 cases per network per sensor setting, reproducible from seeds.
- [ ] The environment answers every action type.
- [ ] Sanity plots: signature maps, report-displacement histograms, surfacing rates, complaint maps.
- [ ] Data sheet for the benchmark written.
- [ ] **Reader schema v1.0 and gazetteer format frozen with the fine-tune teammate.**

**Sync with fine-tuning.** Hand over the gazetteer files and the report-truth generator so the teammate can turn simulated events into training text. Agree on which generator settings are reserved for *test only*.

---

## Phase 2 — Evidence and inference engine (no LLM) + all RQ1 baselines

**Goal.** Build the best *code-only* fusion of sensors and reports: a calibrated posterior, a decision gate and an action-value engine. Run every RQ1 baseline. This is the foundation the agent must beat or build on.

**Think through and decide:**
- **Pressure likelihood:**
  - residual model (Gaussian with model-error inflation vs correlation/angle score);
  - unknown leak size (profile it with a closed-form least-squares fit, or marginalise over a grid);
  - time windows (night vs day; averaging).
- **Report likelihood**, the core novelty. Candidate structure: a mixture of
  - a *true-signal* part (leak → surface point → downhill kernel → seen at a landmark/street),
  - a *background* part (false or unrelated reports ∝ population),
  - with propensity-scaled detection.

  Also decide:
  - how reader uncertainty enters (gazetteer candidate distributions);
  - how much one report may move the posterior (a robust cap);
  - how to use **negative evidence** (no reports where people *would* report);
  - the **complaint-as-pressure-sensor** likelihood (censored pressure threshold).
- **Hypothesis space.** Start with a single leak. Add Mu-style confounders (sensor fault, demand anomaly, valve state) once the basics work. Two-leak cases come later.
- **Calibration.** Fit temperature or noise inflation on dev; check reliability diagrams (ECE). Model mismatch makes posteriors over-confident, so expect to temper them.
- **Decision rule:**
  - a gate contract (adapt Mu's G1–G6, plus report consistency and rule compliance);
  - **conformal dig radius** from dev cases (target coverage 1−α);
  - a Bayes-risk rule using the cost model.

  Compare simple variants.
- **Action valuation:**
  - EIG vs decision-focused VOI;
  - how to generate candidates (logger nodes, valve sets for step tests, survey segments);
  - one-step vs two-step lookahead (step tests often benefit from bisection);
  - speed target: under 1 s per decision.
- **Baselines** (04 §9):
  - sensitivity matching;
  - LILA (or substitute);
  - report hot-spot;
  - **Bicik-style Dempster–Shafer fusion**;
  - **Mu-style gated pressure-only**;
  - **iterative one-more-sensor**;
  - **greedy VOI/cost fixed policy**;
  - random policy (floor) and oracle (ceiling).

**Pitfalls:**
- over-confident posteriors;
- one report dominating;
- tuning on test;
- making the report model so close to the generator that results are circular (run misspecification sweeps).

**Exit criteria:**
- [ ] RQ1 pilot on dev with bootstrap CIs.
- [ ] Risk–coverage curves for all RQ1 methods.
- [ ] Calibration acceptable.
- [ ] The VOI engine ranks actions fast.
- [ ] Decision record for the report-likelihood design.

**Sync with fine-tuning.** Integrate reader v0 (even if weak) through the real interface. Measure *error propagation*: how much does RQ1 change with gold report fields vs reader output (the "oracle reader" ablation)?

---

## Phase 3 — Investigation agent: reasoning, agentic loop, tools, RAG, safety

**Goal.** An LLM agent that adds **measurable** value on top of the Phase-2 engine, safely. The full menu of options is in `05_REASONING_AGENT_DESIGN_SPACE.md`. Read it fully before starting.

**Think through and decide:**
- **Division of labour** (05 §1): exactly which decisions the LLM owns, and why code can't own them.
- **Architecture,** compared at a matched budget:
  - single planner with structured state;
  - role-separated (planner / abductive analyst / skeptic / compliance);
  - plan-then-execute with CaMeL-style data separation.
- **State shown to the LLM:**
  - a compact belief summary (top dig areas with probabilities, entropy, misfit flags);
  - the VOI table;
  - the hypothesis ledger;
  - pending items.

  Never raw arrays. Summarise every step to avoid context bloat.
- **Tool design:**
  - typed schemas (Pydantic/JSON Schema);
  - informative errors (so the agent can recover);
  - cost and delay returned with every result;
  - idempotent where possible;
  - optionally exposed as an MCP server.
- **Questions to reporters:** prefer *closed-form, multiple-choice* questions whose options are built from the competing hypotheses. They are easier to score (EIG) and easier to parse. Score them BED-LLM style: the LLM predicts answers per hypothesis, and code computes EIG.
- **Abduction:**
  - triggers: posterior-predictive misfit, conflicting evidence, hints in text;
  - the LLM selects and parameterises *testable templates* (sensor fault, hydrant use, planned flushing, valve mis-state, second leak, private leak, non-supply water, fabricated reports);
  - code fits and scores each template with an Occam penalty.
- **Falsification critic** before any dig proposal: "What is the strongest alternative, and what cheap test separates the two?"
- **RAG:**
  - corpus: real manuals + synthetic utility SOPs + incident notes;
  - **hard rules in code, soft and procedural rules via RAG**;
  - hybrid retrieval + rerank; citations; abstain when nothing is relevant;
  - defend against poisoning.
- **Memory:**
  - working memory (the ledger);
  - episodic (past cases);
  - semantic (area and reporter reliability, burst priors).

  Fairness guard: keep propensity separate from reliability.
- **Safety:** the gate, the approval simulator, provenance tags, an AgentSpec-like runtime monitor, and injection handling.
- **Error recovery:** retries with fixed arguments, fallbacks, timeouts. Inject tool faults to test recovery.

**Pitfalls:**
- the LLM re-doing maths;
- hallucinated numbers (force it to cite VOI-table rows);
- endless loops;
- a straw-man ReAct baseline;
- over-asking citizens;
- anchoring on the first report.

**Exit criteria:**
- [ ] The agent runs end to end on all dev suites.
- [ ] 0 unsafe or unapproved actions on dev traps.
- [ ] Tool-call validity ≥ 95 % (target; adjust with evidence).
- [ ] Full step logs.
- [ ] First dev comparison vs ReAct, fixed rules and greedy-VOI.
- [ ] Decision records for architecture and LLM choice.

**Sync with fine-tuning.** Reader v1 in the loop, including **reply mode** (parsing answers to the agent's questions) and the optional place-linking task.

---

## Phase 4 — Full evaluation: networks, suites, baselines, ablations, statistics

**Goal.** Answer RQ1 and RQ2 with fair, pre-registered experiments (04 §§7–12).

**Think through and decide:**
- final case counts per suite (budget-aware);
- the main LLM plus one robustness LLM;
- seeds and k for pass^k;
- which ablations matter for the claims (04 §10);
- the statistics plan (paired tests, Holm correction);
- which LLM-judge uses need human validation.

**Must include:**
- the **real BattLeDIM 2019 leak set** (real SCADA, synthetic reports);
- **cross-network transfer**;
- the **equity suite**;
- the **tool-failure suite**;
- **no-leak controls**;
- the **oracle-reader vs real-reader** comparison;
- a **cost-ratio sensitivity** analysis (does the conclusion hold if digs are 2× or 5× more expensive?).

**Pitfalls:**
- budget blow-up: cache LLM calls; run ablations on a stratified subset;
- changing code mid-run: tag every run with a git hash and config hash;
- peeking at test results.

**Exit criteria:**
- [ ] Every table and figure regenerates from logs with one script.
- [ ] CIs and significance tests reported.
- [ ] A set of failure cases collected for Phase 5.

**Sync with fine-tuning.** Final reader with fairness slices and CPU latency.

---

## Phase 5 — Analysis, failure study, equity audit, operator interface, write-up

**Goal.** Turn results into insight, produce a workshop-quality paper, and build a convincing demo.

**Content:**
- **Intermediate-step analysis:**
  - belief trajectories (mass on the true area per step, entropy);
  - VOI regret of chosen actions;
  - precision and recall of abduction;
  - critic saves and costs;
  - question quality;
  - explanation faithfulness;
  - calibration of the LLM's stated confidence vs the engine posterior vs outcomes.
- **Failure taxonomy,** with 3–5 qualitative case studies:
  - a win;
  - a correct abstention;
  - a fooled-by-downhill case;
  - an injection attempt blocked;
  - a confounder caught by abduction.
- **Equity audit** and its limits.
- **Operator UI (demo):**
  - network map with the posterior heat;
  - report pins (seen point vs inferred source);
  - a "next best check" card with VOI and cost;
  - the evidence certificate;
  - approve / reject buttons.
- **Limitations and future work:** live data, real reports, more networks, multi-leak, voice channel.

**Exit criteria:**
- [ ] Paper draft (problem, related work, method, experiments, results, limitations).
- [ ] Reproducibility package (README, configs, seeds, data card, model card).
- [ ] Demo notebook or video.

---

## 6. Fine-tuning track (runs in parallel; owned by a teammate)

| Milestone | When | What |
|---|---|---|
| M0 Contract freeze | During Phase 1 | Schema `reader-1.0`, gazetteer format, enums, test-only generator settings |
| M1 v0 | During Phase 2 | First data + first tuned model + baseline comparison |
| M2 v1 | Start of Phase 3 | Reply mode + (optional) place linking; eval report |
| M3 Final | Start of Phase 4 | Fairness slices, CPU latency, model and data cards |

**Integration test (every milestone):** `read_report()` on 100 environment reports must return 100 % schema-valid JSON. Field F1 must also be reported against generator truth.

**Schema changes** only by version bump (`reader-1.1`), announced to the whole team.

---

## 7. Cross-cutting engineering

- **Config-driven experiments** (YAML / Hydra): every run records `run_id`, git hash, config hash, seeds, model IDs and API dates.
- **Logs:** step logs as JSONL, one file per case per method. Results tables are built **only** from logs.
- **LLM call cache** keyed by (model, prompt, params), for reproducibility and budget.
- **Cost meter:** tokens, API calls, wall time, and simulated crew cost per case.
- **Decision records** in `docs/decisions/`.
- **Suggested layout** (indicative; change freely):

```
agentix/
  env/          # networks, scenario + report generators, action simulator, case writer
  engine/       # signatures, likelihoods, posterior, gate, VOI, conformal
  reader/       # client for the fine-tuned model (schema validation, fallbacks)
  agent/        # LLM agent(s), prompts, ledger, memory, critic, tools
  rag/          # corpus builder, retriever, citations
  safety/       # gate, approvals simulator, runtime monitor, provenance
  baselines/    # LILA/substitute, Bicik D-S, Mu-style, iterative sensor, ReAct, fixed rules
  eval/         # suites, metrics, stats, figures
  configs/  data/  runs/  docs/
```

---

## 8. Stretch ideas (only after the core works)

- **OSM-Town:** a network generated on real streets of an Indian neighbourhood (OSMnx + WDS-Designer-style rules + DEM elevation). It gives real street names and points of interest for report realism.
- **Learning across cases:** update area propensity and reliability, and pipe burst priors, after each closed case. Evaluate over a "season" of 50 sequential cases.
- **Anytime-valid stopping** with e-values (POPPER-style) for the dig decision.
- **Distil agent trajectories** into a small local agent model (FireAct / AgentTuning style). This would be a second fine-tuning contribution.
- Two simultaneous leaks, incipient leaks, and a voice-complaint channel (ASR).

---

## 9. Status (update as you go)

| Phase | Status | Decision records | Notes |
|---|---|---|---|
| Research | Done 2026-10-08 | — | See 01, 04, 05 |
| 1 Foundations | Done (2026-10-09), except teammate sign-off + official Real model | ADR-001, 002, 003, 004 | Environment, calibrated noise, gazetteer, report generator, 530-case benchmark; see `docs/phase1/PHASE1_REPORT.md` and `docs/LEARNING_LOG.md` |
| 2 Inference engine + RQ1 baselines | Done (2026-10-11) | ADR-005, 006, 007, 008 | Official truth (bench v2), evidence engine, 7 static + 6 active baselines, budgeted VOI policy; see `docs/phase2/PHASE2_REPORT.md` |
| 3 Agent + RAG + safety | Done (2026-10-11) | ADR-009, 010, 011 | Safe toolbox (action selector, code gate, operator token, SOP in code, taint), LLM planner (plan-then-execute, abduction, critic), rule brain, ReAct/one-shot baselines, held-out document wording; test: 0 unsafe vs ReAct 4 trap digs, AgentiX most careful but finds fewer leaks than rules. See `docs/phase3/PHASE3_REPORT.md` |
| 4 Full evaluation | Part 1 done (2026-10-16): bottleneck found and fixed, fresh test | ADR-012 | Coarse-to-fine search, 250 m listening, 80 % dig bar, crews setting, AgentiX v2 (code search + LLM analyst + critic). Fresh 204-case test (3 crews): localization 36.8 → 53.5 %, false dig rate 51.7 → 17.6 %, 0 unsafe; unfamiliar wording: AI 39 % vs rules 11 % (p = 3e-5). Still open: real BattLeDIM leaks, second network, two-leak / tool-failure / equity suites, real teammate reader, pass^k, Holm. See `docs/phase4/PHASE4_REPORT.md` |
| 5 Analysis + write-up | Not started | | |
| Fine-tune track | Not started | | Owner: teammate (see 03) |
