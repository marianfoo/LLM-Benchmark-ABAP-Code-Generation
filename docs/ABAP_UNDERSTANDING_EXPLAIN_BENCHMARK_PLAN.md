# ABAP Code Understanding and Explain Benchmark Plan

Date: 2026-02-20

## 1. Current repository check (ABAP-1)

From the current project artifacts:

- `data/model_leaderboard.csv`:
  - `sap-abap-1` success `R0 = 10.67%`
  - `sap-abap-1` success `R5 = 19.89%`
  - ranking by `Success_R5_pct`: `9 / 11` models
  - final failed runs are mostly syntax-stage failures (`FinalFail_Stage_Syntax_pct = 85.37%`)
- `python src/llm_generate.py --model sap-abap-1 --mode status` currently shows:
  - `Waiting for LLM round: 14` (0.8%)
  - `Success: 358 / 1800`

Quick content audit of `data/sap-abap-1.json` first responses (heuristic):

- code-like ABAP structure detected in ~99.83% of first responses
- markdown wrapper rate ~0%
- obvious explanation/prose cues are near-zero

Interpretation for your question ("ABAP-1 is bad but mainly described as explain model"):

- ABAP-1 is currently weak in this generation benchmark.
- In this repo setup, underperformance is **not mainly caused by returning explanations instead of code**.
- The dominant failure mode is syntax/compilation robustness, not verbosity/explain-mode output.

## 2. Can we compare explain features of all models vs ABAP-1?

Yes, but not with the current benchmark alone.

Why:

- Current system prompt explicitly says: "Only respond with the code. Do not include any explanations or comments."
- So explain ability is currently suppressed by design.

Conclusion:

- Keep current benchmark as **Code Generation Track**.
- Add a separate **Code Understanding + Explain Track** for fair model comparison (including ABAP-1).

## 3. Reuse audit of existing test cases/assets

## Directly reusable

- `dataset/prompts/*.txt` (180 tasks): reusable as task context.
- `dataset/abap_canonical_solution/*.abap` (180): reusable as behavior reference.
- `dataset/abap_unittests/*.abap` (180): best reusable gold signal for expected behavior and edge cases.
- `data/prompt_classification.csv`: reusable for per-category reporting.
- Existing model logs (`data/<model>.json`): reusable for mining hard prompts and real failure feedback.

## Partially reusable

- SAP test pipeline in `src/abap_test.py` / `src/abap_interaction.py`:
  - reusable for execution-grounded verification
  - but needs a new prompt/eval runner for explanation tasks.

## Not production-ready for this purpose

- `local_test/` scripts are experimental and currently not integrated into the main benchmark flow.
- Existing local result files show 0% pass in stored samples, so treat as prototype code, not a stable eval pipeline.

## 4. Proposed benchmark to test ABAP code understanding

Use tests as backbone (your "maybe use test?" idea is correct).

## Track A: Objective understanding (execution-grounded, recommended first)

1. **Test Outcome Prediction (CRUXEval-style adaptation)**
   - Input: ABAP code + concrete test input.
   - Ask model: expected output or pass/fail result.
   - Score: exact match / accuracy against unit-test expectations.

2. **Assertion Target Prediction**
   - Input: ABAP code + one unit-test method body.
   - Ask model: what value/property is being asserted.
   - Score: exact expected assertion value or normalized semantic label.

3. **Failure Diagnosis from Real Feedback**
   - Input: ABAP code + real SAP feedback message from existing logs.
   - Ask model: root cause + minimal fix location.
   - Score: rubric with required facts (stage, cause, affected symbol/line area).

## Track B: Explain quality (secondary, rubric-based)

1. **Behavior Explanation**
   - Input: ABAP code.
   - Ask model: concise algorithm explanation, edge cases, complexity, and failure risks.

2. **Test-Aware Explanation**
   - Input: ABAP code + selected unit tests.
   - Ask model: explain why each test passes/fails.

3. **Scoring**
   - rubric dimensions:
     - behavioral correctness
     - edge-case coverage
     - test alignment
     - hallucination rate
     - ABAP-specific terminology correctness
   - combine judge scoring with automatic claim checks against unit-test facts.

## 5. Concrete scoring design

Recommended composite:

- `UnderstandingScore = 0.60 * Objective + 0.40 * Explain`
- `Objective` = average of Track A tasks (exact metrics)
- `Explain` = rubric score with hallucination penalty

For ABAP-1 comparison:

- report absolute scores for each model
- report pairwise win-rate vs ABAP-1 on identical items
- use paired bootstrap confidence intervals
- break down by category (`String`, `Array`, `Math`, `Logical`, `ABAP DB`)

## 6. How to build it quickly from existing tests

1. Parse all `dataset/abap_unittests/*.abap`:
   - extract test methods
   - extract input setup and assertion expectations
2. Generate structured eval items (`jsonl`):
   - `item_id`, `prompt_id`, `code`, `test_case`, `gold_answer`, `category`
3. Keep a hidden split (EvalPlus-style):
   - public dev items for prompt tuning
   - hidden evaluation items for leaderboard
4. Run all models with a dedicated "explain/understanding" prompt template.

Estimated starting volume:

- 180 tasks, 1424 test methods in total (about 7.9 test methods per task).
- Enough to create a substantial understanding benchmark without new SAP content authoring.

## 7. What other benchmarks suggest (web research)

## Relevant benchmark ideas to borrow

- **CRUXEval**: code reasoning via output/input prediction; high signal for understanding without relying only on free-text judging.
  - https://github.com/facebookresearch/cruxeval
- **EvalPlus**: strengthen weak benchmark tests and keep hidden tests; reduces overfitting and improves reliability.
  - https://evalplus.github.io/
- **LiveCodeBench**: contamination-aware, continuously updated benchmark with multiple code tasks including execution/test prediction settings.
  - https://arxiv.org/abs/2403.07974
- **RepoQA**: long-context repository understanding and retrieval before reasoning; useful if you later evaluate multi-file ABAP projects.
  - https://arxiv.org/abs/2506.11706
- **SWE-bench / SWE-bench Verified**: realistic issue-level evaluation with human-validated subsets; useful design pattern for higher-stakes ABAP maintenance tasks.
  - https://arxiv.org/abs/2310.06770
  - https://www.swebench.com/
- **CodeXGLUE**: established code understanding task family including code-to-text summarization and related tasks.
  - https://github.com/microsoft/CodeXGLUE

## ABAP-1 product positioning note

SAP states ABAP AI includes explanation-oriented capabilities in initial release ("ABAP code explanation" and optimization) and later coding assist features.

- https://www.sap.com/products/artificial-intelligence/business-ai/abap-ai-models.html

This supports creating a separate explain/understanding track for ABAP-1, instead of evaluating it only with code-generation pass rates.

## 8. Implementation plan for this repo

Phase 1 (fast, objective-first):

1. Add `analysis/build_understanding_items.py`
2. Add `src/understanding_eval.py`
3. Add `analysis/score_understanding.py`
4. Output:
   - `data/understanding_items_dev.jsonl`
   - `data/understanding_items_hidden.jsonl`
   - `data/<model>_understanding_predictions.jsonl`
   - `data/understanding_leaderboard.csv`

Phase 2 (explain-quality extension):

1. Add rubric + judge harness
2. Add claim-checker against parsed test facts
3. Add combined dashboard columns (objective, explain, combined, vs ABAP-1 delta)

## 9. Recommended next actions

1. Finish the 14 pending `sap-abap-1` conversations for clean generation comparability.
2. Implement Phase 1 (objective understanding from existing tests) before rubric-heavy explain scoring.
3. Use ABAP-1 as baseline model in the new track and report head-to-head deltas for every model.
