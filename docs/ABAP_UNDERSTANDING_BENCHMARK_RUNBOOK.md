# ABAP Understanding Benchmark Runbook

Date: 2026-02-26

This document explains how the new ABAP understanding benchmark works, how to run it for OpenAI / Anthropic / ABAP-1, and how to verify and plot results.

## 1. What this benchmark does

The benchmark is objective and auto-scored.

For each task (`dataset/prompts/*.txt`), one composite question is built from:

- task prompt
- canonical ABAP solution (`dataset/abap_canonical_solution/*.abap`)
- ABAP unit test (`dataset/abap_unittests/*_test.abap`)

The model must output strict JSON with these fields:

- `method_name`
- `return_param_name`
- `loop_count`
- `has_select`
- `test_method_count`
- `assert_call_count`

Evaluation is iterative (like current benchmark rounds):

- Round 0: initial answer
- Round 1..5: model gets "incorrect" feedback and retries
- Success at round `r` means all fields are correct at that round

This gives comparable metrics to your current benchmark:

- cumulative success by feedback round (`R0..R5`)
- model leaderboard
- plot similar to `plots/success_by_model_by_feedbackround_in_percent.png`

## 2. New scripts

- Build benchmark items:
  - `analysis/build_understanding_items.py`
- Run LLM understanding benchmark:
  - `src/understanding_eval.py` (sequential mode, all providers)
  - `src/understanding_batch.py` (batch mode for OpenAI/Anthropic, called via `--mode batch`)
- Score model outputs:
  - `analysis/score_understanding.py`
- Generate plots:
  - `analysis/plot_understanding_success_by_round.py`
  - `analysis/plot_understanding_field_accuracy.py`
- Run all scoring + plots in one command:
  - `analysis/build_understanding_assets.py`

## 3. Output files

- `data/understanding_items.jsonl`
- `data/<model>_understanding_predictions.jsonl`
- `data/understanding_results.csv`
- `data/understanding_model_leaderboard.csv`
- `plots/understanding_success_by_model_by_feedbackround_in_percent.png`
- `plots/understanding_field_accuracy_by_model_in_percent.png`

## 4. Prerequisites

From repo root (`/Users/marianzeis/DEV/LLM-Benchmark-ABAP-Code-Generation`):

1. Install deps (`uv sync`) and use venv python (`.venv/bin/python`).
2. `.env` must include provider credentials:
   - OpenAI (`OPENAI_API_KEY`)
   - Anthropic (`ANTHROPIC_API_KEY`)
   - ABAP-1 via SAP AI Core (`AICORE_*` vars)
3. Ensure `src/llms.py` contains the model IDs you want to run.

## 5. Step-by-step process

## Step A: Build understanding items

```bash
.venv/bin/python analysis/build_understanding_items.py
```

Quick check:

```bash
wc -l data/understanding_items.jsonl
# expected: 180
```

## Step B: Smoke test in between (mock, no API calls)

Run smoke for all three target model IDs:

```bash
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --smoke --mock
.venv/bin/python src/understanding_eval.py --model claude-opus-4-5-20251101 --smoke --mock
.venv/bin/python src/understanding_eval.py --model sap-abap-1 --smoke --mock
```

Score + plots (smoke data):

```bash
.venv/bin/python analysis/build_understanding_assets.py
```

If smoke passes, continue to real runs.

## Step C: Real run (OpenAI, Anthropic, ABAP-1)

Recommended start with smaller slice first (cost/risk control):

```bash
# 30 tasks x 3 repetitions first
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --limit-items 30 --repetitions 3
.venv/bin/python src/understanding_eval.py --model claude-opus-4-5-20251101 --limit-items 30 --repetitions 3
.venv/bin/python src/understanding_eval.py --model sap-abap-1 --limit-items 30 --repetitions 3
```

Then full run (current-scope comparable):

```bash
# 180 tasks x 3 repetitions x up to 6 rounds
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --repetitions 3
.venv/bin/python src/understanding_eval.py --model claude-opus-4-5-20251101 --repetitions 3
.venv/bin/python src/understanding_eval.py --model sap-abap-1 --repetitions 3
```

`src/understanding_eval.py` now defaults to `--repetitions 3`, so you can omit the flag if you want.

Resume behavior:

- script is resume-safe by default
- rerun same command after interruption; completed `(item_id, repetition)` pairs are skipped

Status check:

```bash
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode status --repetitions 3
.venv/bin/python src/understanding_eval.py --model claude-opus-4-5-20251101 --mode status --repetitions 3
.venv/bin/python src/understanding_eval.py --model sap-abap-1 --mode status --repetitions 3
```

## Step C-alt: Real run with batch mode (OpenAI, Anthropic – recommended)

Batch mode uses the OpenAI / Anthropic batch APIs (~50% cheaper, higher throughput).
ABAP-1 does not support batch mode – use the sequential run from Step C.

### Option 1: Full automated pipeline (submit → wait → evaluate → next round → …)

```bash
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch --repetitions 10
.venv/bin/python src/understanding_eval.py --model claude-opus-4-5-20251101 --mode batch --repetitions 10
```

The script submits round 0 as a batch, polls for completion, evaluates locally,
then submits round 1 for failed items, and repeats up to `--max-rounds`.
Crash-safe: if interrupted, rerun the same command to resume from the active batch.

### Option 2: Async submit / collect workflow (for long batches)

```bash
# Submit one round and exit
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-submit --repetitions 10

# Check status later
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-status

# Collect completed batch, evaluate, prepare next round
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-collect

# Submit next round
.venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-submit --repetitions 10

# Repeat batch-collect / batch-submit until all rounds done
```

### Batch tracking files

- `data/understanding_openai_batch_tracking.json`
- `data/understanding_anthropic_batch_tracking.json`
- `data/<model>_understanding_batch_state.json` (in-progress items between rounds)

## Step D: Score and build plots

```bash
.venv/bin/python analysis/build_understanding_assets.py
```

Primary plot similar to current benchmark style:

- `plots/understanding_success_by_model_by_feedbackround_in_percent.png`

## 6. How prompt-result verification works

Per run, stored in `data/<model>_understanding_predictions.jsonl`:

- raw model answer per round
- parsed JSON payload
- normalized values used for comparison
- field-level correctness
- overall correctness
- first success round (or unsolved)

Verification logic is deterministic:

- strings normalized (trim/lower/no whitespace noise)
- integer fields parsed numerically
- boolean parsed as `true/false` (`yes/no`, `1/0` accepted)
- run is correct only if all fields are correct in the same round

## 7. Fast troubleshooting

1. Missing item file:
   - run `analysis/build_understanding_items.py` first.
2. Provider auth errors:
   - verify `.env` key names and values.
3. ABAP-1 orchestration errors:
   - verify `AICORE_*` variables and deployment mapping in `docs/ABAP_1_BTP_SETUP.md`.
4. Parsing failures due malformed model JSON:
   - these are handled as incorrect rounds and visible in prediction logs.
5. Mistral batch (`--mode batch`) fails with HTTP 402 on free trial:
   - this is a provider quota limitation for batch jobs.
   - `src/understanding_eval.py` now auto-falls back to sequential run mode for this case.
   - if you need true batch mode, upgrade the Mistral plan.

## 8. Suggested execution order

1. Build items
2. Mock smoke test
3. Small real subset (30x3)
4. Score + inspect plots
5. Full runs (180x3)
6. Final scoring + plots
