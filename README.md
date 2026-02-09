# LLM-Benchmark-ABAP-Code-Generation

This repository benchmarks Large Language Models for generating ABAP code,
following the methodology described in **[Benchmarking Large Language Models
for ABAP Code Generation (arXiv:2601.15188)](https://arxiv.org/abs/2601.15188)**.

The benchmark runs **180 tasks x 10 repetitions** per model.  Each task goes
through up to **6 rounds** of interaction (Round 0 = initial code, Rounds 1–5 =
compiler/unit-test feedback corrections).

---

## Prerequisites

### 1. SAP ABAP Developer Trial (Docker)

The SAP/ADT tests require a running SAP ABAP instance accessible at
`http://localhost:50000`. Start the Docker container:

```bash
docker run --stop-timeout 3600 -i --name a4h -h vhcala4hci \
  -p 3200:3200 -p 3300:3300 -p 8443:8443 -p 30213:30213 \
  -p 50000:50000 -p 50001:50001 \
  sapse/abap-cloud-developer-trial:2023
```

Wait until the container is fully started (can take several minutes).
The default credentials are `DEVELOPER` / `ABAPtr2023#00` on client `001`.

### 2. Python environment (>= 3.13)

All commands must be run **from the repository root** using the project's
virtual environment.

**Option A — [uv](https://docs.astral.sh/uv/) (recommended, lockfile-reproducible):**

```bash
uv sync          # creates .venv and installs pinned dependencies from uv.lock
```

**Option B — plain pip:**

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

All subsequent commands use the venv Python:

```bash
.venv/bin/python src/<script>.py ...
```

> **Shortcut:** If you have activated the venv (`source .venv/bin/activate`),
> you can use plain `python` instead.

### 3. API keys

Copy `.env.example` to `.env`, then fill in credentials:

```bash
cp .env.example .env
```

Set API keys in `.env` (or as environment variables):

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...

# SAP AI Core (for ABAP-1 via orchestration)
AICORE_AUTH_URL=...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=...
AICORE_RESOURCE_GROUP=...
AICORE_MODEL_NAME=sap--abap-1
AICORE_MODEL_VERSION=latest
```

The scripts load these via `python-dotenv`. Provider config is in `src/llms.py`.

For ABAP-1/BTP setup details, see [`docs/ABAP_1_BTP_SETUP.md`](docs/ABAP_1_BTP_SETUP.md).

---

## Concepts

### Round definition (paper-aligned)

| Round | What happens |
|-------|-------------|
| **0** | LLM generates initial ABAP code from the prompt. |
| **1–5** | System feeds SAP compiler / unit-test error back to LLM; LLM generates a corrected version. |

A conversation stops early when unit tests pass (`The unit tests were successful.`).

### Conversation states

Every prompt + repetition chat in `data/<model>.json` is in exactly one state:

| State | Meaning | Last message |
|-------|---------|-------------|
| **NeedsSAPTest** | Code exists, SAP feedback not yet recorded | `role: assistant` |
| **WaitingForLLM** | Feedback exists, LLM hasn't responded yet | `role: user` (real error) |
| **InfraRetriable** | Transient ADT error (timeout / 500 / session) | `role: user` starting with `[INFRA]` |
| **Success** | Unit tests passed | contains `The unit tests were successful.` |
| **MaxedOut** | 6 assistant messages reached (Rounds 0–5 exhausted) | `role: user` (final feedback) |

**Key invariant for comparability:** never generate the next LLM round while
any chats are still **NeedsSAPTest**.  The generation scripts enforce this
automatically and refuse to run if untested assistant answers exist.

---

## Quick Start: Add a new model

1. Edit `src/llms.py` — add an entry in `MODELS_TO_RUN` and provider config in
   `API_PROVIDERS`.
2. Set the API key in your `.env` file or environment.
3. For SAP ABAP-1, configure SAP AI Core/BTP orchestration first:
   [`docs/ABAP_1_BTP_SETUP.md`](docs/ABAP_1_BTP_SETUP.md)
4. Follow the workflow below.

---

## End-to-End Workflow (per model)

The benchmark alternates **generate** and **test** steps.  Each step is
**idempotent** and **restartable** — safe to abort and re-run at any point.

```bash
# All commands run from repository root. Use .venv/bin/python if venv is not activated.

# ── Step 0: Complete any pending batch jobs from a previous run ──
python src/llm_generate.py --model <model> --mode complete-pending

# ── Step 1: Generate initial LLM responses (Round 0) ──
#    Non-destructive: skips prompt+rep combos that already have a response.
python src/llm_generate.py --model <model> --mode first

# ── Step 2: Run SAP/ADT tests (syntax check, activation, unit tests) ──
#    Requires SAP Docker container running at localhost:50000.
#    Option A – parallel (recommended, faster):
python src/parallel_runner.py --model <model> --workers 4
#    Option B – sequential:
python src/abap_test.py --model <model> --mode resume

# ── Step 2b: Retry transient infra failures ──
python src/abap_test.py --model <model> --mode retry --max-attempts 3

# ── Step 3: Generate next correction round ──
#    Generates one assistant response for all WaitingForLLM conversations.
#    Refuses to run if any chats still NeedSAPTest.
python src/llm_generate.py --model <model> --mode next

# ── Step 4: Test again ──
python src/parallel_runner.py --model <model> --workers 4
python src/abap_test.py --model <model> --mode retry --max-attempts 3

# ── Repeat Steps 3–4 until done (up to 5 feedback rounds total) ──

# ── Check progress at any time ──
python src/llm_generate.py --model <model> --mode status
# or
python src/abap_test.py --model <model> --mode status
```

### One-shot runner (from scratch)

For a brand-new model where you want to run everything in one go:

```bash
python src/main.py --model <model>
```

This runs the full generate→test→feedback loop automatically.  It is now safe
to re-run: first-round generation is non-destructive, and next-round generation
is blocked if any chats still need SAP testing.

### ABAP-1 quick verification

```bash
# Verify SAP AI Core orchestration connection to ABAP-1
python src/smoke_test_abap1.py

# Run the benchmark pipeline for ABAP-1
python src/llm_generate.py --model sap--abap-1 --mode first
python src/parallel_runner.py --model sap--abap-1 --workers 4
python src/abap_test.py --model sap--abap-1 --mode retry --max-attempts 3
python src/llm_generate.py --model sap--abap-1 --mode next
```

---

## Opus 4.5 catch-up recipe

Opus already has Round 0 data.  To bring it up to Rounds 0–5 (comparable to
Sonnet):

```bash
# 1. Ensure all Round 0 tests have feedback
python src/parallel_runner.py --model claude-opus-4-5-20251101 --workers 4
python src/abap_test.py --model claude-opus-4-5-20251101 --mode retry --max-attempts 3

# 2. Check status – "Needs SAP test" should be 0 before continuing
python src/llm_generate.py --model claude-opus-4-5-20251101 --mode status

# 3. Generate Round 1 corrections
python src/llm_generate.py --model claude-opus-4-5-20251101 --mode next

# 4. Test Round 1
python src/parallel_runner.py --model claude-opus-4-5-20251101 --workers 4
python src/abap_test.py --model claude-opus-4-5-20251101 --mode retry --max-attempts 3

# 5. Repeat steps 3–4 for Rounds 2–5
#    (The script will print [SKIP] when all rounds are exhausted)
```

Then regenerate results and plots (see "Generate results + plots" section below).

---

## Abort / Restart guidance

All state is stored in files under `data/`.  You can safely `Ctrl-C` at any
point and re-run the same command.

| File | Purpose | Safe to delete? |
|------|---------|----------------|
| `data/<model>.json` | Conversation logs (source of truth) | No — this is your data |
| `data/<model>_tiers.json` | SAP tier results per round | Yes — regenerated by SAP testing |
| `data/<model>_retry_state.json` | Retry attempt counts | Yes — resets retry budget |
| `data/<model>_abap_test_failures.log` | Append-only failure log | Yes — informational only |
| `data/<model>_queue.json` | Parallel runner work queue | Yes — recreated on next run |
| `data/<model>_batch.jsonl` | Batch API input | Yes — recreated on next generation |
| `data/anthropic_batch_tracking.json` | Pending Anthropic batches | Only if no pending batches |
| `data/openai_batch_tracking.json` | Pending OpenAI batches | Only if no pending batches |

---

## Comparability checklist

To produce results comparable to the paper and other models:

- Same dataset: `dataset/prompts/` (180 tasks)
- Same repetitions: 10 per task
- Same max feedback rounds: 5 (Rounds 0–5, max 6 assistant messages)
- Same SAP environment: `sapse/abap-cloud-developer-trial:2023` Docker image
- Same validation order: class creation → syntax check → activation → unit tests
- Temperature 0.2 (except GPT-5 which cannot be changed)

---

## Generate publish artifacts (single command)

Run this from the repository root:

```bash
.venv/bin/python analysis/build_publish_assets.py
```

This single command regenerates:

- `data/results.csv` (consolidated benchmark results)
- `data/syntax_errors.json`
- all charts in `plots/`
- `data/model_leaderboard.csv`
- `MODEL_LEADERBOARD.md`
- website data/assets in `webpage/` (`webpage/data/dashboard.json`, copied CSV/JSON, copied plot images)

If you only want to skip plot regeneration:

```bash
.venv/bin/python analysis/build_publish_assets.py --skip-plots
```

## Website + GitHub Pages

- Static website source is in `webpage/` (`index.html`, `styles.css`, `app.js`).
- Deployment workflow is `.github/workflows/deploy-pages.yml`.
- The workflow builds assets and deploys GitHub Pages on every push to `main`.
- The main website tables only include fully evaluated models (`Max rounds tested = 6`).
- First-time setup in GitHub: Settings -> Pages -> Source = `GitHub Actions`.

Local preview:

```bash
python3 -m http.server --directory webpage 8000
```

Then open `http://localhost:8000`.

---

## Important notes for development

- **Never remove existing code comments** — improve them if needed, but don't
  delete them.
- **`data/<model>.json` is the single source of truth** for all conversation
  state.  All other files (`_tiers.json`, `_queue.json`, etc.) are derived and
  can be regenerated.
- **Canonicalization** reformats LLM output to match fixed unit-test contracts.
  It defaults to `false` in `parallel_runner.py` and `true` in `abap_test.py`.
  Older models (Sonnet in the paper) used canonicalization; newer runs
  (GPT 5.2, Opus 4.5) do not.  Keep it consistent within a model's full run.
- **`analysis/consolidate_results.py`** counts feedback rounds by the number of
  `user` feedback messages in each conversation.  Missing feedback = missing
  round in `results.csv` = lower reported success.  This is why ensuring
  complete SAP testing before generating the next round is critical.
- **Batch APIs** (Anthropic, OpenAI) are asynchronous.  If the script is
  interrupted while a batch is pending, use `--mode complete-pending` to
  retrieve results before continuing.
- **The `[INFRA]` prefix** on feedback marks transient ADT errors (timeouts,
  500s, session expiry).  These should be retried via `--mode retry`, never fed
  back into the LLM as "real" compiler errors.
- **Running multiple models in parallel:** LLM generation for different
  providers can run simultaneously (e.g. Anthropic batch + OpenAI batch in
  separate terminals).  SAP testing shares the single Docker container, so run
  one model's SAP tests at a time or keep total `--workers` across all
  terminals reasonable (4–6 total).
- **`--mode status` is free** — it reads the JSON file locally, makes no API
  calls, and can be run at any time (even while SAP tests are running in
  another terminal) to check progress.
- **GPT-5 temperature:** OpenAI restricts GPT-5 models to temperature=1.
  This is set in `src/llms.py` and cannot be changed.  All other models use 0.2.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `command not found: python` | macOS has no system `python` | Use `.venv/bin/python` or `python3` |
| `ModuleNotFoundError: No module named 'pandas'` | Wrong Python interpreter | Use `.venv/bin/python` (the venv has all deps) |
| `ConnectionError` / `500` during SAP tests | SAP Docker container not ready or overloaded | Wait for container startup; reduce `--workers`; retry with `--mode retry` |
| `[BLOCKED] N conversation(s) still need SAP testing` | Tried `--mode next` before finishing SAP tests | Run SAP tests first (`parallel_runner.py` then `--mode retry`) |
| `KeyError: 'Prompt'` in consolidate_results.py | Script run from wrong directory | `cd analysis` first, then run with `../.venv/bin/python` |
| `Error: data file not found` | Wrong `--model` name or data not generated yet | Check exact model name in `src/llms.py`; run `--mode first` |
| Batch job shows `pending` forever | Provider API hasn't finished processing | Use `--mode complete-pending` to poll; check provider dashboard |
| `Missing SAP AI Core environment variables` | `.env` does not include required AICORE values | Fill `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP` |

---

## CLI reference

### `src/llm_generate.py` — LLM response generation

```
python src/llm_generate.py --model <model> --mode <mode>

Modes:
  status           Print conversation state counts (no API calls)
  complete-pending Complete any pending provider batch jobs
  first            Generate initial Round 0 responses (non-destructive)
  next             Generate one correction round (blocked if NeedsSAPTest > 0)
```

### `src/abap_test.py` — SAP/ADT testing

```
python src/abap_test.py --model <model> --mode <mode>

Modes:
  status   Print progress summary
  resume   Run SAP tests on all untested conversations (idempotent)
  retry    Re-test [INFRA] / missing feedback (max 3 attempts, restartable)
```

### `src/parallel_runner.py` — Parallel SAP testing

```
python src/parallel_runner.py --model <model> --workers 5

Uses a file-locked queue for dynamic work distribution.
Safe to restart; finished prompts are preserved.
```

### `src/main.py` — One-shot full benchmark

```
python src/main.py --model <model>

Runs the full generate→test→feedback loop for all rounds.
Safe to re-run (non-destructive generation, guarded next-round).
```

### `src/smoke_test_abap1.py` — ABAP-1 connection test

```
python src/smoke_test_abap1.py
python src/smoke_test_abap1.py --model sap--abap-1 --message "Reply with OK."
```

---

## Project Structure

*   **`src/`**: Core Python scripts.
    *   `llm_generate.py` – LLM generation CLI (first / next / status / complete-pending).
    *   `abap_test.py` – SAP testing CLI (resume / retry / status).
    *   `chat_state.py` – Shared conversation state classification helpers.
    *   `abap_interaction.py` – SAP/ADT interaction logic with exception-safe feedback.
    *   `parallel_runner.py` – Multi-worker SAP testing with file-locked queue.
    *   `main.py` – One-shot full benchmark runner.
    *   `abap1_orchestration.py` – SAP AI Core orchestration client for ABAP-1.
    *   `smoke_test_abap1.py` – Connection smoke test for ABAP-1.
    *   `generate_llm_answers_batch_anthropic.py` – Anthropic Batch API integration.
    *   `generate_llm_answers_batch_openai.py` – OpenAI Batch API integration.
    *   `generate_llm_answers_parallel.py` – Async parallel generation (OpenAI-compatible + ABAP-1 orchestration).
    *   `llms.py` – Model and provider configuration.
*   **`dataset/`**: Benchmark inputs and ABAP resources.
    *   `prompts/` – 180 LLM prompt files.
    *   `abap_canonical_solution/`, `abap_tables/`, `abap_unittests/` – ABAP fixtures.
*   **`data/`**: Outputs and intermediate artifacts.
    *   `<model>.json` – Conversation logs (source of truth).
    *   `<model>_tiers.json` – SAP/ADT tier results per model.
    *   `<model>_retry_state.json` – Retry attempt counts.
    *   `<model>_abap_test_failures.log` – Failure/retry log.
    *   `*_batch.jsonl`, `*_batch_response.jsonl` – Batch API files.
    *   `results.csv` – Consolidated results for plots.
    *   `prompt_classification.csv` – Task category labels.
*   **`analysis/`**: Result consolidation and plotting scripts.
*   **`plots/`**: Generated charts.
