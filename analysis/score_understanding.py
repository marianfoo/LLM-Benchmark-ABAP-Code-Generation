#!/usr/bin/env python3
"""Score ABAP understanding benchmark predictions.

Reads:
    data/*_understanding_predictions.jsonl

Writes:
    data/understanding_results.csv
    data/understanding_model_leaderboard.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CATEGORY_COLUMNS = [
    "String Handling",
    "List or Array Operation",
    "Mathematical Calculation",
    "Logical Condition",
    "ABAP Database Operation",
]

FIELD_KEYS = [
    "method_name",
    "return_param_name",
    "loop_count",
    "has_select",
    "test_method_count",
    "assert_call_count",
]


def _iter_prediction_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*_understanding_predictions.jsonl"))


def _wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return False


def _build_run_rows(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for pred_file in _iter_prediction_files(data_dir):
        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                model = str(rec.get("model", "")).strip()
                prompt = str(rec.get("prompt_id", "")).strip()
                repetition = int(rec.get("repetition", 0))
                success_round = rec.get("success_round", None)
                success = success_round is not None
                rounds = rec.get("rounds", []) if isinstance(rec.get("rounds"), list) else []
                categories = rec.get("categories", []) if isinstance(rec.get("categories"), list) else []

                by_round: dict[int, dict[str, Any]] = {}
                for r in rounds:
                    if not isinstance(r, dict):
                        continue
                    idx = r.get("round")
                    if isinstance(idx, int):
                        by_round[idx] = r

                row: dict[str, Any] = {
                    "Model": model,
                    "Prompt": prompt,
                    "Repetition": repetition,
                    "Success": success,
                    "Success_Round": success_round,
                }

                for r in range(6):
                    rr = by_round.get(r)
                    if rr is None:
                        row[f"Feedback_Round_{r}"] = None
                    else:
                        row[f"Feedback_Round_{r}"] = "correct" if _safe_bool(rr.get("overall_correct")) else "incorrect"

                # Field-level final correctness uses the success round if available;
                # otherwise, the last attempted round.
                final_round_dict: dict[str, Any] | None = None
                if success and isinstance(success_round, int):
                    final_round_dict = by_round.get(success_round)
                if final_round_dict is None and by_round:
                    final_round_dict = by_round[max(by_round.keys())]

                final_field_correct = {}
                if isinstance(final_round_dict, dict):
                    maybe_fields = final_round_dict.get("field_correct", {})
                    if isinstance(maybe_fields, dict):
                        final_field_correct = maybe_fields

                for field in FIELD_KEYS:
                    row[f"Field_{field}_final_correct"] = _safe_bool(final_field_correct.get(field, False))

                category_set = set(str(c) for c in categories)
                for cat in CATEGORY_COLUMNS:
                    row[cat] = cat in category_set

                rows.append(row)

    if not rows:
        raise RuntimeError("No understanding prediction rows found. Run src/understanding_eval.py first.")

    df = pd.DataFrame(rows)

    # Deterministic sort
    def _prompt_sort_key(x: str) -> tuple[int, int]:
        if str(x).startswith("erp_"):
            try:
                return (1, int(str(x).split("_", 1)[1]))
            except (ValueError, IndexError):
                return (1, 999999)
        try:
            return (0, int(str(x)))
        except ValueError:
            return (0, 999999)

    df["_prompt_sort_k"] = df["Prompt"].map(_prompt_sort_key)
    df = df.sort_values(["Model", "_prompt_sort_k", "Repetition"]).drop(columns=["_prompt_sort_k"]).reset_index(drop=True)
    return df


def _build_model_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []

    for model, mdf in df_runs.groupby("Model", sort=True):
        n = len(mdf)
        succ_round = mdf["Success_Round"]

        success_by_round_pct: dict[int, float] = {}
        for r in range(6):
            success_by_round_pct[r] = float(((succ_round.notna()) & (succ_round <= r)).mean()) * 100.0

        auc = sum(success_by_round_pct.values()) / 6.0

        r0_successes = int(((succ_round.notna()) & (succ_round <= 0)).sum())
        r5_successes = int(((succ_round.notna()) & (succ_round <= 5)).sum())

        r0_low, r0_high = _wilson_ci_95(r0_successes, n)
        r5_low, r5_high = _wilson_ci_95(r5_successes, n)

        summary: dict[str, Any] = {
            "Model": model,
            "Runs": n,
            "Prompts": int(mdf["Prompt"].nunique()),
            "Success_R0_pct": success_by_round_pct[0],
            "Success_R1_pct": success_by_round_pct[1],
            "Success_R2_pct": success_by_round_pct[2],
            "Success_R3_pct": success_by_round_pct[3],
            "Success_R4_pct": success_by_round_pct[4],
            "Success_R5_pct": success_by_round_pct[5],
            "AUC_Success_pct": auc,
            "Success_R0_ci95_low_pct": r0_low * 100.0,
            "Success_R0_ci95_high_pct": r0_high * 100.0,
            "Success_R5_ci95_low_pct": r5_low * 100.0,
            "Success_R5_ci95_high_pct": r5_high * 100.0,
        }

        for field in FIELD_KEYS:
            col = f"Field_{field}_final_correct"
            summary[f"Field_{field}_final_acc_pct"] = float(mdf[col].mean()) * 100.0

        # Category-level final success
        for cat in CATEGORY_COLUMNS:
            subset = mdf[mdf[cat] == True]  # noqa: E712
            key = cat.replace(" ", "").replace("/", "Or")
            summary[f"Success_R5_{key}_pct"] = float(subset["Success"].mean()) * 100.0 if not subset.empty else float("nan")
            summary[f"Runs_{key}"] = int(len(subset))

        summaries.append(summary)

    out = pd.DataFrame(summaries)
    out = out.sort_values(["Success_R5_pct", "AUC_Success_pct", "Success_R0_pct"], ascending=False).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score ABAP understanding benchmark predictions and create summary tables.")
    parser.add_argument(
        "--output-results",
        default=str(DATA_DIR / "understanding_results.csv"),
        help="Output CSV for run-level results.",
    )
    parser.add_argument(
        "--output-leaderboard",
        default=str(DATA_DIR / "understanding_model_leaderboard.csv"),
        help="Output CSV for model summary.",
    )
    args = parser.parse_args()

    df_runs = _build_run_rows(DATA_DIR)
    df_summary = _build_model_summary(df_runs)

    out_results = Path(args.output_results)
    out_leaderboard = Path(args.output_leaderboard)

    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_leaderboard.parent.mkdir(parents=True, exist_ok=True)

    df_runs.to_csv(out_results, index=False, sep=";")
    df_summary.to_csv(out_leaderboard, index=False)

    print(f"Wrote {out_results} ({len(df_runs)} rows)")
    print(f"Wrote {out_leaderboard} ({len(df_summary)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
