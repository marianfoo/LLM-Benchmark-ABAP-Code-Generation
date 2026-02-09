from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


DISPLAY_NAME_BY_MODEL: dict[str, str] = {
    "claude-opus-4-5-20251101": "Claude Opus 4.5 (2025-11-01)",
    "claude-sonnet-4-20250514": "Claude Sonnet 4 (2025-05-14)",
    "gpt-5-2025-08-07": "GPT-5 (2025-08-07)",
    "gpt-5.2": "GPT-5.2",
    "gpt-oss_120b": "GPT-OSS 120B",
    "gpt-oss_20b": "GPT-OSS 20B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "qwen2.5-coder-32b-instruct": "Qwen2.5 Coder 32B Instruct",
    "qwen3-coder": "Qwen3 Coder",
    "codestral-22b": "Codestral 22B",
}


CATEGORY_COLUMNS: list[str] = [
    "String Handling",
    "List or Array Operation",
    "Mathematical Calculation",
    "Logical Condition",
    "ABAP Database Operation",
]

CATEGORY_KEYS: dict[str, str] = {
    "String Handling": "StringHandling",
    "List or Array Operation": "ListOrArrayOperation",
    "Mathematical Calculation": "MathematicalCalculation",
    "Logical Condition": "LogicalCondition",
    "ABAP Database Operation": "ABAPDatabaseOperation",
}


def _is_model_conversation_json(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    first_key = next(iter(payload))
    return isinstance(first_key, str) and first_key.endswith(".txt")


def _iter_model_json_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.json")):
        name = path.name
        if name in {
            "results.json",
            "anthropic_batch_tracking.json",
            "openai_batch_tracking.json",
            "syntax_errors.json",
        }:
            continue
        if name.endswith("_tiers.json") or name.endswith("_retry_state.json"):
            continue
        yield path


def _normalize_feedback_status(content: str) -> str:
    text = (content or "").strip()
    lower = text.lower()

    if lower.startswith("[infra]"):
        return "infra error"

    if "unit tests were successful" in lower:
        return "unit tests were successful"

    if "unit test failed" in lower:
        return "unit test failed"

    if "unittest syntax check failed" in lower:
        return "unittest syntax check failed"

    if "syntax check failed" in lower:
        return "syntax check failed"

    if "activation failed" in lower:
        return "activation failed"

    if "source code could not be set" in lower:
        return "source code could not be set"

    if "class could not be created" in lower:
        return "class could not be created"

    if "class name not found" in lower:
        return "class name not found"

    if "there should only be the one public method" in lower:
        return "public method rule violated"

    if lower.startswith("code structure issue:") or "code structure issue" in lower:
        return "code structure issue"

    if "abaplint" in lower:
        return "abaplint failure"

    return "other error"


def _stage_for_status(status: str | None) -> str | None:
    if status is None or status == "":
        return None
    if status == "infra error":
        return "infra"
    if status == "unit tests were successful":
        return "success"
    if status == "unit test failed":
        return "unit"
    if status in {"syntax check failed", "unittest syntax check failed", "activation failed"}:
        return "syntax"
    if status in {
        "class could not be created",
        "source code could not be set",
        "class name not found",
        "public method rule violated",
        "code structure issue",
        "abaplint failure",
    }:
        return "class"
    return "other"


def _wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Standard pass@k estimator used in code-generation benchmarks (sampling
    without replacement), given n samples and c correct samples for a task.
    """
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _load_prompt_categories(classification_path: Path) -> dict[str, dict[str, bool]]:
    if not classification_path.exists():
        raise FileNotFoundError(f"Missing classification file: {classification_path}")

    class_df = pd.read_csv(classification_path, sep=";", encoding="utf-8-sig")
    if "HumanEval/Nr" in class_df.columns:
        class_df["Prompt_ID"] = class_df["HumanEval/Nr"].astype(str)
    elif "Nr" in class_df.columns:
        class_df["Prompt_ID"] = class_df["Nr"].astype(str)
    else:
        raise ValueError(
            "prompt_classification.csv missing expected id column ('HumanEval/Nr' or 'Nr')"
        )

    rename_map = {
        "List & Array Operations": "List or Array Operation",
        "Mathematical Calculations": "Mathematical Calculation",
        "Logical Checks": "Logical Condition",
        "Database Operations (ABAP)": "ABAP Database Operation",
    }
    class_df = class_df.rename(columns=rename_map)

    categories: dict[str, dict[str, bool]] = {}
    for _, row in class_df.iterrows():
        prompt_id = str(row["Prompt_ID"])
        categories[prompt_id] = {}
        for cat in CATEGORY_COLUMNS:
            val = row.get(cat, "")
            categories[prompt_id][cat] = str(val).strip().upper() == "X"
    return categories


@dataclass(frozen=True)
class RunRecord:
    model: str
    prompt: str
    repetition: int
    success_round: int | None
    rounds_tested: int | None
    feedback_by_round: tuple[str | None, ...]  # length 6


def _parse_conversation_to_record(
    model: str, prompt: str, repetition: int, conversation: list[dict[str, Any]]
) -> RunRecord:
    feedback_by_round: list[str | None] = [None] * 6

    assistant_round = -1
    awaiting_feedback = False

    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            assistant_round += 1
            awaiting_feedback = True
            continue

        if role != "user" or not awaiting_feedback or assistant_round < 0:
            continue

        status = _normalize_feedback_status(msg.get("content", ""))
        if status == "infra error":
            continue  # retryable; does not consume a feedback iteration

        if 0 <= assistant_round <= 5:
            feedback_by_round[assistant_round] = status
        awaiting_feedback = False

    success_round: int | None = None
    for r, status in enumerate(feedback_by_round):
        if status == "unit tests were successful":
            success_round = r
            break

    last_tested: int | None = None
    for r in range(5, -1, -1):
        if feedback_by_round[r] is not None:
            last_tested = r
            break
    rounds_tested = (last_tested + 1) if last_tested is not None else None

    return RunRecord(
        model=model,
        prompt=prompt,
        repetition=repetition,
        success_round=success_round,
        rounds_tested=rounds_tested,
        feedback_by_round=tuple(feedback_by_round),
    )


def _load_runs_from_json_logs(
    data_dir: Path, prompt_categories: dict[str, dict[str, bool]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_path in _iter_model_json_files(data_dir):
        model_name = model_path.stem
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not _is_model_conversation_json(payload):
            continue

        for prompt_file, reps in payload.items():
            prompt_id = str(prompt_file).removesuffix(".txt")
            for rep_idx, conversation in enumerate(reps):
                if not isinstance(conversation, list):
                    continue
                record = _parse_conversation_to_record(
                    model=model_name,
                    prompt=prompt_id,
                    repetition=rep_idx,
                    conversation=conversation,
                )
                row: dict[str, Any] = {
                    "Model": record.model,
                    "Model_Display": DISPLAY_NAME_BY_MODEL.get(
                        record.model, record.model
                    ),
                    "Prompt": record.prompt,
                    "Repetition": record.repetition,
                    "Success": record.success_round is not None,
                    "Success_Round": record.success_round,
                    "Rounds_Tested": record.rounds_tested,
                }
                for r in range(6):
                    row[f"Feedback_Round_{r}"] = record.feedback_by_round[r]
                    row[f"Stage_Round_{r}"] = _stage_for_status(record.feedback_by_round[r])

                cats = prompt_categories.get(prompt_id, {})
                for cat in CATEGORY_COLUMNS:
                    row[cat] = bool(cats.get(cat, False))

                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No model conversation JSONs found under data/. Expected files like data/<model>.json."
        )

    # Stable sort for deterministic outputs
    df = df.sort_values(["Model", "Prompt", "Repetition"]).reset_index(drop=True)
    return df


def _format_md_table(headers: list[str], rows: list[list[str]]) -> str:
    def esc(val: str) -> str:
        return val.replace("|", "\\|")

    out_lines: list[str] = []
    out_lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
    out_lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out_lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(out_lines)


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def _build_model_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    feedback_cols = [f"Feedback_Round_{r}" for r in range(6)]

    def success_by_r(series: pd.Series, r: int) -> pd.Series:
        return series.notna() & (series <= r)

    summaries: list[dict[str, Any]] = []

    for model, mdf in df_runs.groupby("Model", sort=True):
        n_runs = len(mdf)
        n_prompts = mdf["Prompt"].nunique()
        display = mdf["Model_Display"].iloc[0] if "Model_Display" in mdf.columns else model

        succ_round = mdf["Success_Round"]

        success_rates: dict[int, float] = {}
        for r in range(6):
            success_rates[r] = float(success_by_r(succ_round, r).mean()) * 100.0

        auc_success = sum(success_rates.values()) / 6.0

        successes = int(mdf["Success"].sum())
        r0_successes = int(success_by_r(succ_round, 0).sum())
        r5_successes = int(success_by_r(succ_round, 5).sum())

        r0_ci_low, r0_ci_high = _wilson_ci_95(r0_successes, n_runs)
        r5_ci_low, r5_ci_high = _wilson_ci_95(r5_successes, n_runs)

        succ_only = succ_round.dropna()
        mean_succ_round = float(succ_only.mean()) if not succ_only.empty else float("nan")
        median_succ_round = float(succ_only.median()) if not succ_only.empty else float("nan")

        # LLM call proxies
        rounds_tested = mdf["Rounds_Tested"].dropna()
        mean_calls_per_run = float(rounds_tested.mean()) if not rounds_tested.empty else float("nan")
        max_rounds_tested = int(rounds_tested.max()) if not rounds_tested.empty else 0
        pct_round5_tested = (
            float((rounds_tested == 6).mean()) * 100.0 if not rounds_tested.empty else float("nan")
        )
        calls_per_success = (float(rounds_tested.sum()) / successes) if successes > 0 else float("inf")
        mean_calls_to_success = float((succ_only + 1).mean()) if not succ_only.empty else float("nan")
        median_calls_to_success = float((succ_only + 1).median()) if not succ_only.empty else float("nan")

        # Prompt-level (n=10 reps each)
        prompt_success_counts = mdf.groupby("Prompt")["Success"].sum()
        prompts_solved_any = float((prompt_success_counts > 0).mean()) * 100.0
        prompts_solved_all = float((prompt_success_counts == 10).mean()) * 100.0
        prompts_unsolved = float((prompt_success_counts == 0).mean()) * 100.0
        median_prompt_success_rate = float((prompt_success_counts / 10.0).median()) * 100.0

        pass5_final = float(prompt_success_counts.apply(lambda c: _pass_at_k(10, int(c), 5)).mean()) * 100.0

        # Round-0 stages
        r0_stage = mdf["Stage_Round_0"].astype("object")
        r0_stage_success = float((r0_stage == "success").mean()) * 100.0
        r0_stage_unit = float((r0_stage == "unit").mean()) * 100.0
        r0_stage_syntax = float((r0_stage == "syntax").mean()) * 100.0
        r0_stage_class = float((r0_stage == "class").mean()) * 100.0
        r0_stage_other = float((r0_stage == "other").mean()) * 100.0
        r0_stage_missing = float(r0_stage.isna().mean()) * 100.0
        r0_reaches_unit_tests = r0_stage_success + r0_stage_unit

        # Final failure stage (among failed runs only)
        failed = mdf[~mdf["Success"]]
        last_stage: pd.Series
        if failed.empty:
            last_stage = pd.Series(dtype="object")
        else:
            last_stage = (
                failed[feedback_cols]
                .apply(lambda row: _stage_for_status(next((v for v in reversed(row.tolist()) if pd.notna(v)), None)), axis=1)
                .astype("object")
            )
        final_fail_class = float((last_stage == "class").mean()) * 100.0 if not last_stage.empty else 0.0
        final_fail_syntax = float((last_stage == "syntax").mean()) * 100.0 if not last_stage.empty else 0.0
        final_fail_unit = float((last_stage == "unit").mean()) * 100.0 if not last_stage.empty else 0.0
        final_fail_other = float((last_stage == "other").mean()) * 100.0 if not last_stage.empty else 0.0
        final_fail_missing = float(last_stage.isna().mean()) * 100.0 if not last_stage.empty else 0.0

        summary: dict[str, Any] = {
            "Model": model,
            "Model_Display": display,
            "Runs": n_runs,
            "Prompts": n_prompts,
            "Success_R0_pct": success_rates[0],
            "Success_R1_pct": success_rates[1],
            "Success_R2_pct": success_rates[2],
            "Success_R3_pct": success_rates[3],
            "Success_R4_pct": success_rates[4],
            "Success_R5_pct": success_rates[5],
            "AUC_Success_pct": auc_success,
            "Success_R0_ci95_low_pct": r0_ci_low * 100.0,
            "Success_R0_ci95_high_pct": r0_ci_high * 100.0,
            "Success_R5_ci95_low_pct": r5_ci_low * 100.0,
            "Success_R5_ci95_high_pct": r5_ci_high * 100.0,
            "Mean_Success_Round": mean_succ_round,
            "Median_Success_Round": median_succ_round,
            "Mean_LLM_Calls_Per_Run": mean_calls_per_run,
            "Max_LLM_Calls_Per_Run": max_rounds_tested,
            "Pct_Runs_Reaching_R5_Test": pct_round5_tested,
            "Calls_Per_Success": calls_per_success,
            "Mean_Calls_To_Success": mean_calls_to_success,
            "Median_Calls_To_Success": median_calls_to_success,
            "PassAt5_Final_pct": pass5_final,
            "Prompts_Solved_Any_pct": prompts_solved_any,
            "Prompts_Solved_All_pct": prompts_solved_all,
            "Prompts_Unsolved_pct": prompts_unsolved,
            "Median_Prompt_SuccessRate_pct": median_prompt_success_rate,
            "R0_Reaches_UnitTests_pct": r0_reaches_unit_tests,
            "R0_Stage_Success_pct": r0_stage_success,
            "R0_Stage_UnitTestFail_pct": r0_stage_unit,
            "R0_Stage_SyntaxFail_pct": r0_stage_syntax,
            "R0_Stage_ClassFail_pct": r0_stage_class,
            "R0_Stage_OtherFail_pct": r0_stage_other,
            "R0_Stage_Missing_pct": r0_stage_missing,
            "FinalFail_Stage_Class_pct": final_fail_class,
            "FinalFail_Stage_Syntax_pct": final_fail_syntax,
            "FinalFail_Stage_Unit_pct": final_fail_unit,
            "FinalFail_Stage_Other_pct": final_fail_other,
            "FinalFail_Stage_Missing_pct": final_fail_missing,
        }

        # Category success rates (final, pass@1)
        for cat in CATEGORY_COLUMNS:
            subset = mdf[mdf[cat] == True]  # noqa: E712
            cat_key = CATEGORY_KEYS.get(cat, cat.replace(" ", ""))
            summary[f"Success_R5_{cat_key}_pct"] = (
                float(subset["Success"].mean()) * 100.0 if not subset.empty else float("nan")
            )
            summary[f"Runs_{cat_key}"] = int(len(subset))

        summaries.append(summary)

    out = pd.DataFrame(summaries)
    out = out.sort_values(["Success_R5_pct", "AUC_Success_pct", "Success_R0_pct"], ascending=False).reset_index(drop=True)
    return out


def _write_markdown_report(df_summary: pd.DataFrame, output_md: Path) -> None:
    today = date.today().isoformat()

    # Table: developer-friendly summary
    summary_headers = [
        "Model",
        "Success R5",
        "Success R0",
        "AUC (R0–R5)",
        "Median feedbacks to success",
        "R0 reaches unit tests",
        "pass@5 (final)",
        "Prompts solved (≥1/10)",
        "Prompts solved (10/10)",
        "Max rounds tested",
    ]
    summary_rows: list[list[str]] = []
    for _, row in df_summary.iterrows():
        summary_rows.append(
            [
                str(row["Model_Display"]),
                _pct(float(row["Success_R5_pct"])),
                _pct(float(row["Success_R0_pct"])),
                _pct(float(row["AUC_Success_pct"])),
                _num(float(row["Median_Success_Round"]), digits=1),
                _pct(float(row["R0_Reaches_UnitTests_pct"])),
                _pct(float(row["PassAt5_Final_pct"])),
                _pct(float(row["Prompts_Solved_Any_pct"])),
                _pct(float(row["Prompts_Solved_All_pct"])),
                str(int(row["Max_LLM_Calls_Per_Run"])) if pd.notna(row["Max_LLM_Calls_Per_Run"]) else "—",
            ]
        )

    # Table: success by feedback round (paper Table 1 style)
    round_headers = ["Model", "R0", "R1", "R2", "R3", "R4", "R5"]
    round_rows: list[list[str]] = []
    for _, row in df_summary.iterrows():
        round_rows.append(
            [
                str(row["Model_Display"]),
                _pct(float(row["Success_R0_pct"])),
                _pct(float(row["Success_R1_pct"])),
                _pct(float(row["Success_R2_pct"])),
                _pct(float(row["Success_R3_pct"])),
                _pct(float(row["Success_R4_pct"])),
                _pct(float(row["Success_R5_pct"])),
            ]
        )

    # Table: category success (paper Table 2 style, final only)
    cat_headers = ["Model"] + CATEGORY_COLUMNS
    cat_rows: list[list[str]] = []
    for _, row in df_summary.iterrows():
        def _cat_val(cat: str) -> float:
            cat_key = CATEGORY_KEYS.get(cat, cat.replace(" ", ""))
            return float(row[f"Success_R5_{cat_key}_pct"])

        cat_rows.append(
            [str(row["Model_Display"])]
            + [_pct(_cat_val(cat)) for cat in CATEGORY_COLUMNS]
        )

    body = f"""# Model leaderboard (ABAP code generation benchmark)

Generated on **{today}** from the raw benchmark logs in `data/*.json` (180 tasks × 10 repetitions per model, up to 6 feedback rounds).

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

{_format_md_table(summary_headers, summary_rows)}

## Cumulative success by feedback round (paper Table 1 style)

{_format_md_table(round_headers, round_rows)}

## Success by task category (paper Table 2 style, final outcome)

{_format_md_table(cat_headers, cat_rows)}

## Notes / caveats

- This table does **not** include cost/latency/token-usage; add those later if you want a “best value” ranking.
- “pass@k” is widely used in code-generation literature (e.g., HumanEval-style benchmarks). Here it reflects *re-running the benchmark pipeline* k times.
- If **Max rounds tested < 6**, that model has not been evaluated through all 5 feedback iterations (not directly comparable to fully-run models).
"""

    output_md.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a model leaderboard (Markdown + CSV) from raw benchmark logs. "
            "Does not change prompts/tests; only computes derived metrics."
        )
    )
    parser.add_argument(
        "--output-csv",
        default=str(DATA_DIR / "model_leaderboard.csv"),
        help="Where to write the CSV summary.",
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "MODEL_LEADERBOARD.md"),
        help="Where to write the Markdown report.",
    )
    args = parser.parse_args()

    prompt_categories = _load_prompt_categories(DATA_DIR / "prompt_classification.csv")
    df_runs = _load_runs_from_json_logs(DATA_DIR, prompt_categories)
    df_summary = _build_model_summary(df_runs)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv(output_csv, index=False)
    _write_markdown_report(df_summary, output_md)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
