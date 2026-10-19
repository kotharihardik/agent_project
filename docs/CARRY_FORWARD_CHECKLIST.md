# Carry-Forward Checklist — every case and technique we must not forget

> Every finding, edge case and idea discovered so far, with the phase that must handle it.
> **Rule:** at the start of each phase, read this file. Tick items when done, and add new ones as they appear.
> Phase map: **2** = leak-search engine (no LLM) · **3** = reasoning + agent + hypotheses + tools + RAG + safety · **4** = full testing · **5** = analysis + paper.

## Status in Phase 4 (2026-10-16)
- **N1, N5, N6, N7, N14, N17 addressed** by the coarse-to-fine search (ADR-012): step tests now drive the search,
  the policy keeps a listening reserve, and the AI no longer plans field checks (code does), so it cannot give up
  early.
- **N10 partly addressed:** the 80 % dig bar removes most single-false-alarm digs.
- **N16 done:** a fresh 204-case test set with no Phase-3 overlap (120 new S5 cases).
- New: **N19** the inform-reporters audit counts a violation when a case had no reports at all (0.02 per case,
  same for all methods); fix the rule wording in Phase 5. **N20** hidden leaks remain the frontier (~28 % found);
  more budget or sensors are the levers, shown in the sensitivity analysis.

## Status after Phase 3 (2026-10-11)

### Done in Phase 3
| Item | What was done |
|---|---|
| A10 | "Wait" is now a real decision: `wait` + `recheck` (an event stops, a leak does not), onset check, recheck waits by itself when slightly early |
| B1 | Hypothesis ledger on the board, kept up to date by code |
| B2 | Abduction templates: temporary event (flushing / fire brigade / valve works) and sensor fault, tested by physics + Bayes; Occam rule for guessed events; grounded priors (cited note 0.6, report 0.4, guess 0.1) |
| B4 | Critic before every dig, with a grounded veto (may only hold with a cited, testable alternative) |
| B5 | Multiple-choice questions written by the LLM; answers now move the location belief (bug fixed) |
| B6 | Menu with expected gain per cost; the LLM picks; code runs a plan of 1–4 actions and re-asks on news |
| B8 | Night-only checks encoded in tools; the menu hides checks when no night is left |
| B9 | Quarantine + closed vocabularies; instruction stripping in documents; action selector; gate + one-time token; **taint tracking** against coordinated fake reports |
| B10 | Tool errors returned, not raised; repeat guard; LLM retries (429, empty answers); failed runs re-run |
| B11 | Rule book (`configs/rules.yaml`) + SOP corpus with BM25 search; SOP paperwork done by code; post-run audit |
| B14 | Evidence certificate for the operator (evidence IDs, numbers from code) |
| B15 | Matched non-LLM brain on the same toolbox; ablations; held-out document wording |

### Partial or moved
- **B3** (two leaks): no second-leak template yet → Phase 4 (suite S6).
- **B7** (message triage): trust per message + taint; no duplicate merging → Phase 4.
- **B12** (memory across cases) and **A9** (propensity-aware negative evidence) → Phase 4/5 (equity suite S9).
- **B13** (LLM persona reporter) → Phase 4 with the teammate's text work.
- **A13** (AMR channel): still not exposed.

### New items found in Phase 3
| # | Item | Phase |
|---|---|---|
| N8 | Free LLM tiers throttle per model (HTTP 429 after ~1,000 calls). Keep calls per case low (plan-then-execute), cache everything, re-run failed runs | 4 |
| N9 | At 5 sensors a 5–9 L/s hydrant flush moves pressure ~0.05 m, and the inflow estimate's noise (±2.5–3 L/s) is as big as the flush. Recheck is only weakly informative; better inflow modelling (day-of-week, tank) would help | 4 |
| N10 | Acoustic false alarms (6 %) can push P(success) over 0.5 late in a case and cause wrong digs. Consider a second confirming check before a dig that rests on one acoustic hit | 4 |
| N11 | Fake reports WITHOUT any flagged message are not caught by taint. Needs reporter reliability memory / corroboration rules | 4–5 |
| N12 | The smaller model (gpt-oss-20b) rarely forms hypotheses from notes by itself. The code keyword scan + LLM reading together are best. Re-test with a stronger free model when available | 4 |
| N13 | Python `hash()` changes per process: never use it for seeds (fixed with `zlib.crc32`) | all |
| N14 | The LLM gives up earlier than the VOI stopping rule on low-confidence leak cases (fewer leaks found). Idea: show code's stopping advice on the board ("M1 is worth its cost by the VOI rule"); the LLM may override only with a cited reason. Evaluate on FRESH test cases in Phase 4 (the Phase-3 test set has been looked at) | 4 |
| N15 | Any check that re-reads data must never count the same hours twice (recheck bug). Add a generic "evidence ledger" test: repeating any read-only tool must leave the belief unchanged | 4 |
| N17 | When the LLM's 6 planning rounds run out, the fallback abstains even with crew budget left. In 5 of the 9 test leaks the rule brain found and the agent missed, this was the reason. Fix idea: hand the remaining budget to the VOI policy (code finishes the investigation) or give the LLM the same step budget as the rule brain (12 actions). Evaluate on fresh test cases | 4 |
| N18 | A bigger model (nemotron-3-ultra-550b) found 3× more leaks on dev with the same precision. Re-test on fresh test cases when free capacity allows; or use the big model only for the few high-stakes calls (critic, final decision) | 4 |
| N16 | Phase-3 test cases (`runs/phase3/test/cases.json`) were inspected after the first run. Phase 4 must use a disjoint, untouched test sample (plenty left: S1 has 210 test cases, 12 used) | 4 |

## Status after Phase 2 (2026-10-11)

### Done
| Item | What was done |
|---|---|
| A1 | Area-level localization, search-rank metric |
| A2 | Leak size from the inflow rise as a prior on scale |
| A3 | Demand normalisation; real night variability now comes from the official truth |
| A4 | Logger model includes offset uncertainty plus a zone-offset estimate |
| A5 | Downhill kernel in the report likelihood. **No one-shot gain**; it needs questions (B5) |
| A6 | Per-report ε (unrelated probability); quarantine for instructions, non-supply and inside-property reports |
| A7 | Naive baselines in every table |
| A8 | Complaints get ε = 0.98 |
| A11 | Surfacing lowered to 35 %; all results split visible / hidden |
| A12 | Greedy policy digs as soon as P(success) ≥ 0.5 |
| C1 | Official truth loader |
| C2 | Benchmark v2 |

### Partial or moved
- **A9** (propensity-aware negative evidence): not implemented. → Phase 3/4 (equity suite S9).
- **A10** ("wait" as a valued decision): new reports are absorbed, but waiting is not valued. → Phase 3 agent.
- **A13** (AMR channel): not exposed. → Phase 3.

### New items found in Phase 2
| # | Item | Phase |
|---|---|---|
| N1 | Myopic one-step VOI fails. Non-myopic sequences such as "ask, then listen" are an agent opportunity | 3 |
| N2 | Keep confidence calibrated (proper scoring + isotonic) whenever the engine changes | all |
| N3 | Reader quality matters (oracle > rule reader, p = 1e-4). Plug in the teammate's model and measure | 3–4 |
| N4 | Visible misses: downhill > 100 m (28 %), street-only (18 %), ambiguous landmark (15 %), no report yet (10 %). These are the question-strategy targets for the agent | 3 |
| N5 | Step tests are rarely chosen: candidate segments are crude. Build valve-aware segments | 3 |
| N6 | Greedy abstains in 75 % of leak cases. The agent should convert some of these, without raising wrong digs | 3 |
| N7 | Precision–recall frontier (greedy vs sweep). Report both ends; the agent must beat the frontier, not one point | 3–4 |

## A. Physics and evidence facts → how the system must use them

| # | Finding (where it came from) | What the system must do | Phase |
|---|---|---|---|
| A1 | Leak pressure signals are tiny (≈8 cm for 3 L/s), and nearby leaks look 99 % alike (Phase 1 pilot) | Localize to *areas*, not nodes. Report "search length", not only distance | 2 |
| A2 | Leak size is unknown, which makes pressure ambiguity worse (identifiability test) | **Estimate leak size from the night inflow rise** and use it to fix the scale of pressure fingerprints | 2 |
| A3 | Night inflow wobbles 5–7 L/s naturally in real data; small leaks hide in it (real 2018 SCADA) | Average over several nights. Bigger leaks are detected fast, small ones slowly. Raise simulated night variation slightly to match reality | 2 |
| A4 | Temporary sensors have an unknown offset of about ±9 cm (ADR-002) | The logger likelihood must include offset uncertainty. Consider **before/after or two-logger differences**. The agent must know loggers are weak | 2, 3 |
| A5 | Water surfaces away from the leak (trench + downhill); people often report the **pool**, not the source (case S1-0004) | Physics-based report likelihood: "seen point is downhill of the source". Use `is_source_seen` + "flowing towards…" → **reason uphill** | 2 |
| A6 | Most messages at case open are noise (0.5 real out of 3.2) | Score every message's trust; never let one message decide (robust cap). False types: sewer, rain, private, irrigation, mistaken | 2, 3 |
| A7 | "Dig at first report" = 2 % correct | Keep it as the naive baseline in every results table | 2, 4 |
| A8 | Low-pressure complaints carry almost no leak signal in L-Town (pressure valves) | Down-weight this channel for L-Town; keep it generic for other towns; state it as a limitation | 2, 5 |
| A9 | Some areas report less (high / medium / low propensity) | Negative evidence ("no reports here") must depend on how much people report there. Run an equity check | 2, 4 |
| A10 | More reports arrive if we wait, but water is lost | Make **"wait"** a real, valued action (value of information vs water-loss cost) | 2, 3 |
| A11 | Most real leaks are hidden (Thames: >95 % never seen); visible ones are bigger | Lower the simulated surfacing rate (≈1/3); **always split results: visible vs hidden leaks** | 2, 4 |
| A12 | Obvious big bursts are easy for crews | **Fast path:** when evidence is already strong, dig quickly (don't over-investigate); measure over-investigation cost | 3, 4 |
| A13 | Zone C (hill, tank) behaves differently: bigger pressure drops, AMR household meters available | Use the zone structure as a prior; expose AMR data as an extra tool | 2, 3 |

## B. Reasoning, hypotheses and the agent (Phase 3)

| # | Case / technique | Plan direction (decide details in Phase 3) |
|---|---|---|
| B1 | Hypothesis ledger (list of possible explanations with evidence for/against) | Code keeps the numbers; the LLM proposes and explains |
| B2 | "Not a leak" explanations: sensor fault, hydrant/fire use, planned flushing (work orders), valve wrongly closed, private house leak, sewer/rain water, fake reports | **Abduction templates.** The LLM picks which to test when the data doesn't fit; code tests them in the simulator |
| B3 | Two leaks at the same time | A second-leak hypothesis is triggered by misfit |
| B4 | Falsification critic before any dig ("what would prove this wrong? is there a cheap test?") | Separate critic step; count "critic saves" |
| B5 | Asking reporters smart questions | Closed, multiple-choice questions built from the competing areas. The LLM predicts answers per hypothesis; code computes information gain; the reader parses replies |
| B6 | Choosing the next check: logger / acoustic / step test / ask / wait / dig | Value of information per cost. The LLM may override only with a cited reason (night-only work, approvals, school hours…) |
| B7 | Message triage: many messages, conflicting places, duplicates | Trust score per message + physical consistency check |
| B8 | Time rules: step tests and acoustic surveys only at night; reports mostly by day | Encode in tools, rules and the planner |
| B9 | Safety: prompt-injection messages, fake "approval granted" text, coordinated fake reports, poisoned repair notes | The reader acts as quarantine (no tools). Tag untrusted data. Code gate + human approval. Runtime rules monitor |
| B10 | Tool errors (bad node ID, no night left, simulator failure) | Recovery behaviour; measure recovery rate |
| B11 | RAG: dig permits, restricted zones, night-work rules, step-test SOP, work orders, repair history | Hard rules in code, soft rules via RAG. Measure rule violations with vs without RAG |
| B12 | Memory across cases (area reliability, burst-prone streets) | Keep **propensity ≠ reliability** (fairness guard) |
| B13 | A more realistic reporter | Replace the rule-based reporter simulator with an LLM persona simulator (validate it doesn't leak the truth) |
| B14 | Explanations for the operator | Every claim cites evidence IDs; evidence certificate; "not sure yet + next best check" output |
| B15 | Give the LLM jobs code can't do (so it adds *accuracy*, not just words — the lesson from Mu et al. 2026) | Compare against a matched non-LLM policy on every suite |

## C. Data and benchmark work still to do

| # | Item | Phase |
|---|---|---|
| C1 | Loader that time-slices the official `L-TOWN_Real.inp` (177 MB, downloaded); rebuild cases with the official truth | 2 |
| C2 | Benchmark v2: lower surfacing rate, slightly higher night-demand variation, visible/hidden label per case | 2 |
| C3 | New suites: S5 confounders, S6 two leaks, S8 tool failures, S9 equity, S10 real 2019 leaks, S11 transfer (C-Town needs a step-size setting) | 3–4 |
| C4 | Realistic message text from the teammate (LLM personas, Hinglish, typos, voice) + ~200 hand-written messages + real 311-style test messages | Teammate, 2–4 |
| C5 | Share `data/exports/` with the teammate (zip, or add to git) | Now |
| C6 | Teammate signs off on schema `reader-1.0` + gazetteer format | Now |

## D. Testing and honesty (Phase 4–5)

| # | Item |
|---|---|
| D1 | Split every result by visible/hidden leak, leak size, zone and reporting level |
| D2 | Sensitivity sweeps for all our assumptions: surfacing, travel distance, false-report rate, complaint threshold, cost ratios |
| D3 | Re-check the main results on the official Real model |
| D4 | Matched budgets, the same layouts for all methods, pass^k for safety, paired statistics |
| D5 | Correct the proposal sentence: "most new leaks are first found by citizens" → "visible bursts are usually reported first; most leaks never surface" |
| D6 | Case studies for the paper: S1-0004 (pool vs source), an injection blocked, a correct abstention, a hidden leak found by checks, an obvious burst handled fast |
