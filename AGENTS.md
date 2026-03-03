# AGENTS.md — Project Overview

Benchmarks LLMs for ABAP code generation (180 tasks × 10 repetitions, up to 6 feedback rounds per task).
Paper: [arXiv:2601.15188](https://arxiv.org/abs/2601.15188)

---

## Folder Structure

| Path | Contents |
|---|---|
| `src/` | All runnable Python scripts |
| `dataset/prompts/` | 180 LLM task prompt `.txt` files |
| `dataset/abap_canonical_solution/` | Reference ABAP implementations |
| `dataset/abap_unittests/` | ABAP unit test files |
| `dataset/abap_tables/` | SAP table definitions used by tasks |
| `data/` | All outputs — conversation logs, tiers, batch files, results CSVs |
| `analysis/` | Post-processing: consolidation, scoring, plot generation |
| `plots/` | Generated chart images |
| `webpage/` | Static GitHub Pages site (`index.html`, `app.js`, `styles.css`) |
| `docs/` | Runbooks and setup guides |

---

## Key Source Files (`src/`)

| File | Purpose |
|---|---|
| `llms.py` | **Single registry** of all models (`MODELS_TO_RUN`) and API providers (`API_PROVIDERS`). Edit here to add/change models. |
| `llm_generate.py` | **Main CLI** for LLM generation — first round, next rounds, status, batch completion. |
| `main.py` | One-shot full benchmark runner (generate → test → feedback loop for all rounds). |
| `abap_test.py` | SAP/ADT testing CLI — run syntax check, activation, and unit tests per conversation. |
| `parallel_runner.py` | Multi-worker SAP testing with file-locked queue (faster than `abap_test.py` sequential). |
| `understanding_eval.py` | **Separate benchmark**: ABAP code understanding (auto-scored JSON extraction, no SAP needed). |
| `understanding_batch.py` | Batch API backend for `understanding_eval.py`. |
| `chat_state.py` | Conversation state classification helpers (`NeedsSAPTest`, `WaitingForLLM`, `Success`, etc.). |
| `abap_interaction.py` | Low-level SAP/ADT interaction (compile, activate, unit test). |
| `abap1_orchestration.py` | SAP AI Core orchestration client for the ABAP-1 model. |
| `smoke_test.py` | Tests API connectivity for all configured models. `--model <name>` to filter. |
| `generate_llm_answers_batch_anthropic.py` | Anthropic Batch API implementation. |
| `generate_llm_answers_batch_openai.py` | OpenAI Batch API implementation. |
| `generate_llm_answers_batch_mistral.py` | Mistral Batch API implementation. |
| `generate_llm_answers_batch_google.py` | Google Gemini Batch API implementation (OpenAI-compatible endpoint, separate tracking file). |
| `generate_llm_answers_parallel.py` | Async parallel generation (Groq, SAP AI Core, fallback). |
| `generate_llm_answers_openai_responses.py` | OpenAI `/v1/responses` API (codex/reasoning models). |

---

## Key Data Files (`data/`)

| File | Purpose | Safe to delete? |
|---|---|---|
| `<model>.json` | **Source of truth** — all conversations and state | **No** |
| `<model>_tiers.json` | SAP tier results per round | Yes — regenerated |
| `<model>_retry_state.json` | Retry counters for infra errors | Yes |
| `<model>_abap_test_failures.log` | Failure log (informational) | Yes |
| `<model>_queue.json` | Parallel runner work queue | Yes — recreated |
| `anthropic_batch_tracking.json` | Pending Anthropic batch IDs | Only if no pending batches |
| `openai_batch_tracking.json` | Pending OpenAI batch IDs | Only if no pending batches |
| `google_batch_tracking.json` | Pending Google Gemini batch IDs | Only if no pending batches |
| `<model>_understanding_predictions.jsonl` | Understanding benchmark results | **No** |
| `understanding_items.jsonl` | Built understanding benchmark items | Yes — regenerate with `analysis/build_understanding_items.py` |
| `results.csv` | Consolidated ABAP generation results | Yes — regenerated |

---

## Conversation States

Every `(prompt, repetition)` conversation in `<model>.json` is in one state:

| State | Meaning |
|---|---|
| `NeedsSAPTest` | LLM responded, SAP test not yet run |
| `WaitingForLLM` | SAP feedback recorded, LLM correction not yet generated |
| `InfraRetriable` | Transient ADT error (timeout/500); retry, don't feed to LLM |
| `Success` | Unit tests passed |
| `MaxedOut` | All 6 rounds exhausted |

**Key invariant:** never run `--mode next` while any conversation is still `NeedsSAPTest`.

---

## Adding a New Model

1. Add entry to `MODELS_TO_RUN` in `src/llms.py` with `name`, `provider`, `temperature`, `max_tokens`.
2. Add API key to `.env`.
3. For SAP AI Core / ABAP-1: see `docs/ABAP_1_BTP_SETUP.md`.

Providers: `ANTHROPIC`, `OPENAI`, `OPENAI_DIRECT`, `OPENAI_RESPONSES`, `MISTRAL`, `GROQ`, `SAP_AICORE`, `GOOGLE`.
Batch mode is available for: `ANTHROPIC`, `OPENAI`, `MISTRAL`, `GOOGLE`.

For `GOOGLE` (Gemini): set `GEMINI_API_KEY` in `.env` (obtain at [aistudio.google.com](https://aistudio.google.com)). Uses `generate_llm_answers_batch_google.py` with tracking in `data/google_batch_tracking.json`.

---

## ABAP Generation Benchmark — Workflow

```bash
# 0. Recover any pending batch jobs
.venv/bin/python src/llm_generate.py --model <model> --mode complete-pending

# 1. Generate Round 0 (non-destructive, skips already-done)
.venv/bin/python src/llm_generate.py --model <model> --mode first

# 2. SAP test all untested conversations
.venv/bin/python src/parallel_runner.py --model <model> --workers 4
.venv/bin/python src/abap_test.py --model <model> --mode retry --max-attempts 3

# 3. Generate correction round (blocked if any NeedsSAPTest)
.venv/bin/python src/llm_generate.py --model <model> --mode next

# 4. SAP test again — repeat steps 3–4 up to 5 times (Rounds 1–5)

# Check status at any time (no API calls)
.venv/bin/python src/llm_generate.py --model <model> --mode status
.venv/bin/python src/abap_test.py --model <model> --mode status

# One-shot full run (alternative to manual steps above)
.venv/bin/python src/main.py --model <model>
```

---

## Understanding Benchmark — Workflow

Auto-scored benchmark: model extracts structured facts from ABAP code + unit tests.
No SAP Docker required. Output: `data/<model>_understanding_predictions.jsonl`.

```bash
# Build benchmark items (once)
.venv/bin/python analysis/build_understanding_items.py

# Run — sequential mode (all providers)
.venv/bin/python src/understanding_eval.py --model <model> --repetitions 3

# Run — batch mode (Anthropic/OpenAI/Google, ~50% cheaper)
.venv/bin/python src/understanding_eval.py --model <model> --mode batch --repetitions 3

# Step-by-step batch
.venv/bin/python src/understanding_eval.py --model <model> --mode batch-submit
.venv/bin/python src/understanding_eval.py --model <model> --mode batch-status
.venv/bin/python src/understanding_eval.py --model <model> --mode batch-collect

# Status / smoke test
.venv/bin/python src/understanding_eval.py --model <model> --mode status
.venv/bin/python src/understanding_eval.py --model <model> --smoke --mock

# Score + generate plots
.venv/bin/python analysis/build_understanding_assets.py
```

---

## Generate / Publish Results

```bash
# Regenerate all results, plots, leaderboard, and webpage assets
.venv/bin/python analysis/build_publish_assets.py

# Skip plot regeneration
.venv/bin/python analysis/build_publish_assets.py --skip-plots

# Local website preview
python3 -m http.server --directory webpage 8000
# → http://localhost:8000
```

---

## Important Notes

- `data/<model>.json` is the **single source of truth**. All other `data/` files are derived.
- **Never run `--mode next` before all SAP tests are done** — the script will block if `NeedsSAPTest > 0`.
- **Batch APIs are async.** If interrupted, re-run `--mode complete-pending` to retrieve results before continuing.
- `[INFRA]` prefixed feedback = transient ADT error. Retry via `--mode retry`, never feed to LLM.
- **Canonicalization** (`--canonicalize`) was used for older models (Sonnet); keep it consistent within a full model run.
- **GPT-5 temperature** is locked to `1` by OpenAI. All other models use `0.2`.
- **Never remove existing code comments** — improve them if needed.
- SAP Docker must be running at `localhost:50000` for all ABAP testing steps.
