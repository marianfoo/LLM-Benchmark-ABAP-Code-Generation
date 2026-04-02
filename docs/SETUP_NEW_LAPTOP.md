# Setting Up the Benchmark on a New Laptop (Opus 4.6)

This guide covers everything needed to clone the repo, configure credentials, and run the Claude Opus 4.6 generation and understanding benchmarks on a new machine.

---

## 1. Prerequisites

- macOS or Linux
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Clone and install

```bash
git clone https://github.com/marianfoo/LLM-Benchmark-ABAP-Code-Generation.git
cd LLM-Benchmark-ABAP-Code-Generation
uv sync
```

---

## 3. Configure the Anthropic API key

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
ANTHROPIC_API_KEY=<your key>
```

Everything else can be left empty for this run.

---

## 4. Set the remote SAP system credentials

Add the three SAP connection values to your `.env` file:

```bash
SAP_HOST=http://<REPLACE_URL>
SAP_USERNAME=<REPLACE_USERNAME>
SAP_PASSWORD=<REPLACE_PASSWORD>
```

The remote SAP URL, username, and password will be provided to you separately.

---

## 5. Smoke test

Verify the Anthropic API key and model name work before starting:

```bash
.venv/bin/python src/smoke_test.py --model claude-opus-4-6
```

Expected output: `✅ All models verified successfully!`

---

## 6. Run the ABAP code generation benchmark

This single command runs the full generate → SAP test → feedback loop (Rounds 0–5):

```bash
.venv/bin/python src/main.py --model claude-opus-4-6
```

It is safe to re-run if interrupted — it resumes from where it left off.

---

## 7. Run the understanding benchmark

```bash
.venv/bin/python src/understanding_eval.py --model claude-opus-4-6 --repetitions 3
```

Output is written to `data/claude-opus-4-6_understanding_predictions.jsonl`.

---

## 8. Continuing a run started on another machine

Copy the relevant files from the original machine to the same paths on the new one.

For Opus 4.6, the files that matter are:

| File | Why |
|---|---|
| `data/claude-opus-4-6.json` | Source of truth — all conversation state and results |
| `data/claude-opus-4-6_understanding_predictions.jsonl` | Understanding benchmark results |
| `data/anthropic_batch_tracking.json` | Only needed if there are pending Anthropic batch jobs |

After copying, re-run the same commands from steps 6 and 7 — they will pick up where the other machine left off.
