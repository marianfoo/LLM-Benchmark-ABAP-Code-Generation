# Model leaderboard (ABAP code generation benchmark)

Generated on **2026-04-07** from the raw benchmark logs in `data/*.json` (180 tasks × 10 repetitions per model, up to 6 feedback rounds).

This file is meant as a *developer-friendly* starting point for publishing results in a sortable website table.

## What the original paper reports (quick mapping)

The paper (2601.15188v1) primarily compares models via:

- **Table 1**: cumulative success (%) by feedback round (R0–R5).
- **Table 2**: success (%) by task focus category.
- **Table 3**: error-stage distribution (class creation vs syntax vs unit test), noting SAP’s fixed validation order.

The tables below reproduce the same *style* of metrics, and add a few commonly-used, practical ones (like pass@k).

## Recommended sortable columns (practical)

- **Success R5**: overall probability a single run succeeds after up to 5 feedback iterations (higher is better).
- **Success R0**: “first-try” success (higher is better).
- **AUC (R0–R5)**: summarizes both early and final success (higher is better).
- **Median feedbacks to success**: how many feedback iterations a successful run typically needs (lower is better).
- **R0 reaches unit tests**: how often the model produces code that compiles/activates and reaches unit execution immediately (higher is better).
- **pass@5 (final)**: common in code-gen benchmarks; probability a task is solved at least once if you can do 5 independent tries (higher is better).

## What’s in `data/model_leaderboard.csv`

The CSV contains a wider set of columns for website ingestion. Column groups:

- **Success curve**: `Success_R0_pct` … `Success_R5_pct`, plus `AUC_Success_pct`.
- **Uncertainty** (run-level Wilson 95% CI): `Success_R0_ci95_low/high_pct`, `Success_R5_ci95_low/high_pct`.
- **Feedback efficiency**: `Mean_Success_Round`, `Median_Success_Round`.
- **Cost proxies** (LLM calls): `Mean_LLM_Calls_Per_Run`, `Calls_Per_Success`, `Mean/Median_Calls_To_Success`.
- **Benchmark completeness**: `Max_LLM_Calls_Per_Run` (should be 6 for a fully-run model), `Pct_Runs_Reaching_R5_Test`.
- **Retry friendliness**: `PassAt5_Final_pct`, prompt-level consistency columns like `Prompts_Solved_Any_pct` and `Prompts_Solved_All_pct`.
- **Failure stage**: Round-0 stage breakdown (`R0_Stage_*`) and final failure stage breakdown (`FinalFail_Stage_*`).
- **Category performance**: `Success_R5_<CategoryKey>_pct` with corresponding `Runs_<CategoryKey>` sample sizes.

## Leaderboard (developer-friendly summary)

| Model | Success R5 | Success R0 | AUC (R0–R5) | Median feedbacks to success | R0 reaches unit tests | pass@5 (final) | Prompts solved (≥1/10) | Prompts solved (10/10) | Max rounds tested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 88.33% | 44.33% | 76.77% | 0.0 | 46.56% | 90.31% | 91.11% | 85.00% | 6 |
| gemini-3-flash-preview | 84.44% | 10.44% | 63.35% | 1.0 | 11.56% | 87.11% | 87.22% | 77.22% | 6 |
| GPT-5.3 Codex | 84.39% | 18.56% | 62.45% | 1.0 | 21.11% | 88.27% | 89.44% | 76.67% | 6 |
| claude-opus-4-6 | 84.17% | 36.94% | 69.43% | 1.0 | 44.56% | 89.39% | 90.56% | 76.11% | 6 |
| Claude Opus 4.5 (2025-11-01) | 78.72% | 31.61% | 65.03% | 1.0 | 37.78% | 80.90% | 81.11% | 76.11% | 6 |
| deepseek-reasoner | 78.22% | 30.39% | 62.12% | 1.0 | 41.67% | 87.28% | 89.44% | 54.44% | 6 |
| GPT-5 (2025-08-07) | 77.11% | 19.28% | 56.60% | 1.0 | 22.39% | 84.70% | 86.11% | 59.44% | 6 |
| Claude Sonnet 4 (2025-05-14) | 74.67% | 24.11% | 54.34% | 1.0 | 29.06% | 82.83% | 84.44% | 63.89% | 6 |
| GPT-5.2 | 64.00% | 16.33% | 44.74% | 1.0 | 20.00% | 76.80% | 78.89% | 41.11% | 6 |
| Mistral Large 3 | 51.33% | 9.22% | 33.20% | 2.0 | 20.00% | 67.41% | 71.11% | 27.22% | 6 |
| GPT-OSS 120B | 46.17% | 1.44% | 27.44% | 2.0 | 2.28% | 67.05% | 74.44% | 20.56% | 6 |
| GPT-OSS 20B | 29.33% | 3.11% | 19.44% | 2.0 | 4.22% | 48.36% | 56.11% | 8.89% | 6 |
| Qwen3 Coder | 20.94% | 11.72% | 17.70% | 0.0 | 21.67% | 30.61% | 33.89% | 10.00% | 6 |
| Llama 3.3 70B Instruct | 20.83% | 0.00% | 8.93% | 3.0 | 0.00% | 29.51% | 32.78% | 10.00% | 6 |
| sap-abap-1 | 19.89% | 10.67% | 16.56% | 0.0 | 22.67% | 35.23% | 41.67% | 5.00% | 6 |
| Qwen2.5 Coder 32B Instruct | 13.00% | 6.00% | 10.29% | 1.0 | 9.17% | 17.65% | 19.44% | 6.67% | 6 |
| claude-haiku-4-5-20251001 | 5.72% | 2.06% | 3.80% | 2.0 | 2.50% | 13.03% | 16.11% | 0.00% | 6 |
| Codestral 22B | 0.00% | 0.00% | 0.00% | — | 0.00% | 0.00% | 0.00% | 0.00% | 6 |
| glm-5 | 0.00% | 0.00% | 0.00% | — | 0.00% | 0.00% | 0.00% | 0.00% | 0 |

## Cumulative success by feedback round (paper Table 1 style)

| Model | R0 | R1 | R2 | R3 | R4 | R5 |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 44.33% | 71.72% | 82.61% | 86.17% | 87.44% | 88.33% |
| gemini-3-flash-preview | 10.44% | 47.22% | 73.11% | 81.22% | 83.67% | 84.44% |
| GPT-5.3 Codex | 18.56% | 44.17% | 66.72% | 78.39% | 82.50% | 84.39% |
| claude-opus-4-6 | 36.94% | 59.83% | 73.78% | 79.44% | 82.39% | 84.17% |
| Claude Opus 4.5 (2025-11-01) | 31.61% | 56.67% | 70.17% | 75.67% | 77.33% | 78.72% |
| deepseek-reasoner | 30.39% | 52.06% | 64.56% | 71.83% | 75.67% | 78.22% |
| GPT-5 (2025-08-07) | 19.28% | 41.83% | 58.67% | 68.56% | 74.17% | 77.11% |
| Claude Sonnet 4 (2025-05-14) | 24.11% | 41.50% | 52.78% | 63.17% | 69.83% | 74.67% |
| GPT-5.2 | 16.33% | 33.00% | 44.56% | 52.06% | 58.50% | 64.00% |
| Mistral Large 3 | 9.22% | 20.89% | 31.72% | 40.17% | 45.89% | 51.33% |
| GPT-OSS 120B | 1.44% | 13.61% | 26.33% | 35.28% | 41.78% | 46.17% |
| GPT-OSS 20B | 3.11% | 12.94% | 20.06% | 24.39% | 26.78% | 29.33% |
| Qwen3 Coder | 11.72% | 16.22% | 17.89% | 19.28% | 20.17% | 20.94% |
| Llama 3.3 70B Instruct | 0.00% | 0.00% | 3.67% | 12.17% | 16.89% | 20.83% |
| sap-abap-1 | 10.67% | 14.83% | 16.78% | 18.17% | 19.00% | 19.89% |
| Qwen2.5 Coder 32B Instruct | 6.00% | 8.50% | 10.06% | 11.72% | 12.44% | 13.00% |
| claude-haiku-4-5-20251001 | 2.06% | 2.33% | 3.22% | 4.39% | 5.06% | 5.72% |
| Codestral 22B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| glm-5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Success by task category (paper Table 2 style, final outcome)

| Model | String Handling | List or Array Operation | Mathematical Calculation | Logical Condition | ABAP Database Operation |
| --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 95.39% | 82.59% | 89.89% | 88.56% | 31.25% |
| gemini-3-flash-preview | 93.03% | 77.50% | 85.27% | 85.20% | 31.25% |
| GPT-5.3 Codex | 92.76% | 76.85% | 83.66% | 85.12% | 25.00% |
| claude-opus-4-6 | 89.21% | 80.46% | 84.41% | 87.04% | 28.12% |
| Claude Opus 4.5 (2025-11-01) | 88.95% | 68.70% | 78.92% | 80.40% | 25.00% |
| deepseek-reasoner | 84.21% | 71.48% | 78.71% | 80.32% | 23.12% |
| GPT-5 (2025-08-07) | 81.84% | 68.61% | 76.88% | 77.52% | 68.12% |
| Claude Sonnet 4 (2025-05-14) | 81.58% | 63.98% | 71.94% | 76.80% | 66.25% |
| GPT-5.2 | 75.26% | 51.02% | 63.12% | 65.92% | 16.25% |
| Mistral Large 3 | 49.74% | 47.13% | 55.05% | 52.48% | 13.75% |
| GPT-OSS 120B | 45.66% | 32.50% | 53.66% | 45.20% | 58.13% |
| GPT-OSS 20B | 26.18% | 18.43% | 39.46% | 26.80% | 24.38% |
| Qwen3 Coder | 19.08% | 10.46% | 24.41% | 18.80% | 21.88% |
| Llama 3.3 70B Instruct | 18.82% | 8.52% | 26.99% | 18.24% | 26.25% |
| sap-abap-1 | 15.13% | 11.94% | 27.20% | 17.92% | 13.12% |
| Qwen2.5 Coder 32B Instruct | 11.45% | 6.48% | 17.96% | 11.44% | 14.37% |
| claude-haiku-4-5-20251001 | 5.79% | 4.17% | 5.27% | 5.52% | 10.62% |
| Codestral 22B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| glm-5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Notes / caveats

- This table does **not** include cost/latency/token-usage; add those later if you want a “best value” ranking.
- “pass@k” is widely used in code-generation literature (e.g., HumanEval-style benchmarks). Here it reflects *re-running the benchmark pipeline* k times.
- If **Max rounds tested < 6**, that model has not been evaluated through all 5 feedback iterations (not directly comparable to fully-run models).
