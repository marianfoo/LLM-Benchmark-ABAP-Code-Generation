from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
DATA_DIR = REPO_ROOT / "data"
PLOTS_DIR = REPO_ROOT / "plots"
WEB_DIR = REPO_ROOT / "webpage"
WEB_DATA_DIR = WEB_DIR / "data"
WEB_PLOTS_DIR = WEB_DIR / "assets" / "plots"

CORE_SCRIPTS = [
    "consolidate_results.py",
    "generate_syntax_errors.py",
    # Understanding benchmark scoring (produces understanding_model_leaderboard.csv)
    "score_understanding.py",
]

PLOT_SCRIPTS = [
    "plot_success_by_round.py",
    "plot_success_by_category.py",
    "success_distribution.py",
    "plot_syntax_error_patterns.py",
    "plot_error_types_distribution.py",
    "plot_failed_tasks_intersection.py",
    "km_analysis.py",
    "km_analysis_abap.py",
    # Understanding benchmark plots
    "plot_understanding_success_by_round.py",
    "plot_understanding_field_accuracy.py",
]

POST_SCRIPTS = [
    "generate_leaderboard.py",
]

PLOT_METADATA = [
    {
        "file": "success_by_model_by_feedbackround_in_percent.png",
        "title": "Cumulative Success By Feedback Round",
        "description": (
            "Shows cumulative success at each feedback round (R0 to R5). "
            "A steep early increase means the model can quickly turn SAP compiler and unit-test feedback into fixes. "
            "Higher values at R5 indicate stronger final reliability."
        ),
    },
    {
        "file": "success_by_model_by_task_category_in_percent.png",
        "title": "Success By Task Category",
        "description": (
            "Compares final success by benchmark task type, including ABAP Database Operations. "
            "Use this to see whether a model is consistently strong across categories or only performs well on general-purpose tasks. "
            "This helps identify models with better ABAP-specific robustness."
        ),
    },
    {
        "file": "understanding_success_by_model_by_feedbackround_in_percent.png",
        "title": "Understanding Benchmark: Cumulative Success By Feedback Round",
        "description": (
            "Cumulative share of (item, repetition) pairs answered correctly across feedback rounds R0\u20135. "
            "Each model answered 180 structured questions about ABAP code and unit tests. "
            "Claude Opus 4.5 leads, GPT-5.2 trails \u2014 the reverse of the code generation benchmark."
        ),
    },
]

# Models to exclude from all dashboard tables (e.g. incomplete or deprecated runs)
EXCLUDED_MODELS = {"Codestral 22B"}

# Models excluded specifically from the understanding benchmark table and plots
# (e.g. runs that are not yet fully validated)
UNDERSTANDING_EXCLUDED_MODELS = {"gemini-3.1-pro-preview"}


MAIN_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text", "default_sort": "none"},
    # % of all 1800 runs (180 tasks × 10 reps) that passed after up to 5 feedback rounds
    {"key": "Success_R5_pct", "label": "Code Gen (after R5)", "type": "percent", "default_sort": "desc"},
    # % of all 540 understanding runs (180 tasks × 3 reps) that passed after up to 5 feedback rounds
    {"key": "Understanding_R5_pct", "label": "Understanding (after R5)", "type": "percent", "default_sort": "desc"},
    # % of runs that passed on the very first attempt (no feedback given)
    {"key": "Success_R0_pct", "label": "Code Gen (1st attempt)", "type": "percent", "default_sort": "desc"},
    # Area under the cumulative-success curve across rounds R0–R5; higher = improves faster
    {"key": "AUC_Success_pct", "label": "AUC R0–R5 (improves fast?)", "type": "percent", "default_sort": "desc"},
    {
        "key": "R0_Reaches_UnitTests_pct",
        # % of R0 runs where code compiled & activated (reached the unit-test stage), regardless of test outcome
        "label": "R0 Code Compiles (%)",
        "type": "percent",
        "default_sort": "desc",
    },
    # pass@5: given 10 independent runs, probability at least 1 of any 5 drawn passes (standard HumanEval metric)
    {"key": "PassAt5_Final_pct", "label": "pass@5", "type": "percent", "default_sort": "desc"},
    # % of the 180 prompts where at least 1 of the 10 runs eventually succeeded
    {"key": "Prompts_Solved_Any_pct", "label": "Tasks Solved (≥1/10 runs)", "type": "percent", "default_sort": "desc"},
    # % of the 180 prompts where all 10 runs succeeded (maximum consistency)
    {"key": "Prompts_Solved_All_pct", "label": "Tasks Solved (10/10 runs)", "type": "percent", "default_sort": "desc"},
]

ROUND_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text", "default_sort": "none"},
    # Cumulative % of runs passing after each feedback round (R0 = first attempt, R5 = after 5 corrections)
    {"key": "Success_R0_pct", "label": "R0 (1st attempt)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R1_pct", "label": "R1 (+1 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R2_pct", "label": "R2 (+2 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R3_pct", "label": "R3 (+3 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R4_pct", "label": "R4 (+4 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R5_pct", "label": "R5 (+5 feedback)", "type": "percent", "default_sort": "desc"},
]

CATEGORY_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text", "default_sort": "none"},
    # Final success rate (after R5) per task category
    {"key": "Success_R5_StringHandling_pct", "label": "String Handling", "type": "percent", "default_sort": "desc"},
    {
        "key": "Success_R5_ListOrArrayOperation_pct",
        "label": "List / Array Operation",
        "type": "percent",
        "default_sort": "desc",
    },
    {
        "key": "Success_R5_MathematicalCalculation_pct",
        "label": "Mathematical Calculation",
        "type": "percent",
        "default_sort": "desc",
    },
    {"key": "Success_R5_LogicalCondition_pct", "label": "Logical Condition", "type": "percent", "default_sort": "desc"},
    {
        "key": "Success_R5_ABAPDatabaseOperation_pct",
        "label": "ABAP Database Operation",
        "type": "percent",
        "default_sort": "desc",
    },
]


def _run_script(script_name: str, env: dict[str, str]) -> None:
    cmd = [sys.executable, script_name]
    print(f"[RUN] {' '.join(cmd)} (cwd={ANALYSIS_DIR})")
    subprocess.run(cmd, cwd=ANALYSIS_DIR, env=env, check=True)


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _records_for_columns(df: pd.DataFrame, columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [col["key"] for col in columns]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = {key: _safe_value(row.get(key)) for key in keys}
        records.append(record)
    return records


def _copy_plot_files() -> list[str]:
    WEB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    for plot in PLOT_METADATA:
        filename = plot["file"]
        src = PLOTS_DIR / filename
        dst = WEB_PLOTS_DIR / filename
        if not src.exists():
            missing.append(filename)
            continue
        shutil.copy2(src, dst)
    return missing


def _copy_data_files() -> list[str]:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = ["results.csv", "model_leaderboard.csv", "syntax_errors.json"]
    missing: list[str] = []
    for name in required:
        src = DATA_DIR / name
        dst = WEB_DATA_DIR / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dst)
    return missing


def _merge_understanding_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Understanding_R5_pct from understanding_model_leaderboard.csv if available."""
    understanding_path = DATA_DIR / "understanding_model_leaderboard.csv"
    if not understanding_path.exists():
        print(f"[INFO] No understanding leaderboard found at {understanding_path}, skipping merge.")
        return df

    try:
        udf = pd.read_csv(understanding_path)
        # Remove models not yet validated for the understanding benchmark
        udf = udf[~udf["Model"].isin(UNDERSTANDING_EXCLUDED_MODELS)]
        # Rename Success_R5_pct to Understanding_R5_pct and keep only what we need
        udf = udf[["Model", "Success_R5_pct"]].rename(columns={"Success_R5_pct": "Understanding_R5_pct"})
        df = df.merge(udf, on="Model", how="left")
        # Round to 2 decimal places to match existing dashboard format
        df["Understanding_R5_pct"] = df["Understanding_R5_pct"].round(2)
        merged = int(df["Understanding_R5_pct"].notna().sum())
        print(f"[OK] Merged understanding scores for {merged} model(s).")
    except Exception as exc:
        print(f"[WARN] Could not merge understanding scores: {exc}")

    return df


UNDERSTANDING_TABLE_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text", "default_sort": "none"},
    # Area under the cumulative-success curve across rounds R0–R5; higher = improves faster with feedback
    {"key": "AUC_Success_pct", "label": "AUC R0\u2013R5 (improves fast?)", "type": "percent", "default_sort": "desc"},
    # Cumulative % of understanding runs passing after each feedback round (R0 = first attempt, R5 = after 5 corrections)
    {"key": "Success_R0_pct", "label": "R0 (1st attempt)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R1_pct", "label": "R1 (+1 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R2_pct", "label": "R2 (+2 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R3_pct", "label": "R3 (+3 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R4_pct", "label": "R4 (+4 feedback)", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R5_pct", "label": "R5 (+5 feedback)", "type": "percent", "default_sort": "desc"},
]

# Human-readable display names for understanding benchmark models
UNDERSTANDING_MODEL_DISPLAY = {
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "mistral-large-2512": "Mistral Large 2512",
    "sap-abap-1": "SAP ABAP-1",
    "gpt-5.2": "GPT-5.2",
}


def _build_understanding_section() -> dict[str, Any]:
    """Build the understanding benchmark section from understanding_model_leaderboard.csv."""
    path = DATA_DIR / "understanding_model_leaderboard.csv"
    if not path.exists():
        print(f"[INFO] No understanding leaderboard at {path}, skipping section.")
        return {"table": {"columns": UNDERSTANDING_TABLE_COLUMNS, "rows": []}, "plot": None}

    try:
        udf = pd.read_csv(path)
        # Remove models that are not yet validated for the understanding benchmark
        udf = udf[~udf["Model"].isin(UNDERSTANDING_EXCLUDED_MODELS)].reset_index(drop=True)
        udf = udf.sort_values("Success_R5_pct", ascending=False).reset_index(drop=True)

        rows = []
        for _, row in udf.iterrows():
            model_key = str(row.get("Model", ""))
            display = UNDERSTANDING_MODEL_DISPLAY.get(model_key, model_key)
            rows.append({
                "Model_Display": display,
                "Runs": int(row["Runs"]) if not pd.isna(row.get("Runs")) else None,
                "Success_R0_pct": round(float(row["Success_R0_pct"]), 2) if not pd.isna(row.get("Success_R0_pct")) else None,
                "Success_R1_pct": round(float(row["Success_R1_pct"]), 2) if not pd.isna(row.get("Success_R1_pct")) else None,
                "Success_R2_pct": round(float(row["Success_R2_pct"]), 2) if not pd.isna(row.get("Success_R2_pct")) else None,
                "Success_R3_pct": round(float(row["Success_R3_pct"]), 2) if not pd.isna(row.get("Success_R3_pct")) else None,
                "Success_R4_pct": round(float(row["Success_R4_pct"]), 2) if not pd.isna(row.get("Success_R4_pct")) else None,
                "Success_R5_pct": round(float(row["Success_R5_pct"]), 2) if not pd.isna(row.get("Success_R5_pct")) else None,
                "AUC_Success_pct": round(float(row["AUC_Success_pct"]), 2) if not pd.isna(row.get("AUC_Success_pct")) else None,
            })

        print(f"[OK] Built understanding section with {len(rows)} model(s).")
    except Exception as exc:
        print(f"[WARN] Could not build understanding section: {exc}")
        rows = []

    plot_file = "understanding_success_by_model_by_feedbackround_in_percent.png"
    return {
        "table": {"columns": UNDERSTANDING_TABLE_COLUMNS, "rows": rows},
        "plot": {
            "file": plot_file,
            "title": "Understanding Benchmark: Cumulative Success By Feedback Round",
            "description": (
                "Cumulative share of (item, repetition) pairs answered correctly across feedback rounds R0\u20135. "
                "Each model answered 180 structured questions about ABAP code and unit tests."
            ),
        },
    }


def _build_dashboard_json() -> Path:
    leaderboard_path = DATA_DIR / "model_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Missing {leaderboard_path}. Run generate_leaderboard.py first."
        )

    df = pd.read_csv(leaderboard_path)
    df = _merge_understanding_scores(df)
    # Remove explicitly excluded models (e.g. Codestral 22B)
    if "Model_Display" in df.columns:
        df = df[~df["Model_Display"].isin(EXCLUDED_MODELS)].reset_index(drop=True)
    df = df.sort_values(
        ["Success_R5_pct", "AUC_Success_pct", "Success_R0_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df_full = df[df["Max_LLM_Calls_Per_Run"] >= 6].copy().reset_index(drop=True)
    if df_full.empty:
        df_full = df.copy()

    top_final = (
        df_full.sort_values("Success_R5_pct", ascending=False)
        .head(3)["Model_Display"]
        .tolist()
    )
    top_r0 = (
        df_full.sort_values("Success_R0_pct", ascending=False)
        .head(3)["Model_Display"]
        .tolist()
    )
    fully_evaluated = int((df["Max_LLM_Calls_Per_Run"] >= 6).sum())

    understanding_section = _build_understanding_section()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper_url": "https://arxiv.org/abs/2601.15188",
        "main_table": {
            "columns": MAIN_COLUMNS,
            "rows": _records_for_columns(df_full, MAIN_COLUMNS),
        },
        "round_table": {
            "columns": ROUND_COLUMNS,
            "rows": _records_for_columns(df_full, ROUND_COLUMNS),
        },
        "category_table": {
            "columns": CATEGORY_COLUMNS,
            "rows": _records_for_columns(df_full, CATEGORY_COLUMNS),
        },
        "plots": PLOT_METADATA,
        "understanding": understanding_section,
        "summary": {
            "models_count": int(df_full["Model"].nunique()),
            "total_models_count": int(df["Model"].nunique()),
            "fully_evaluated_models_count": fully_evaluated,
            "top_by_final_success": top_final,
            "top_by_first_try_success": top_r0,
        },
    }

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WEB_DATA_DIR / "dashboard.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_pipeline(skip_plots: bool) -> None:
    mpl_cache = REPO_ROOT / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_cache)
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_parts = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    for script in CORE_SCRIPTS:
        _run_script(script, env)

    if not skip_plots:
        for script in PLOT_SCRIPTS:
            _run_script(script, env)

    for script in POST_SCRIPTS:
        _run_script(script, env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Single-command build for benchmark publishing assets: "
            "results.csv, plots, leaderboard files, and webpage data/assets."
        )
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot regeneration (still builds leaderboard + webpage data).",
    )
    args = parser.parse_args()

    run_pipeline(skip_plots=args.skip_plots)

    missing_data = _copy_data_files()
    missing_plots = _copy_plot_files()
    dashboard_json = _build_dashboard_json()

    print(f"[OK] Wrote {dashboard_json}")
    if missing_data:
        print(f"[WARN] Missing data files not copied to webpage/data: {', '.join(missing_data)}")
    if missing_plots:
        print(f"[WARN] Missing plot files not copied to webpage/assets/plots: {', '.join(missing_plots)}")

    print("[OK] Publish assets ready.")
    print(f"      Web root: {WEB_DIR}")
    print(f"      Leaderboard CSV: {DATA_DIR / 'model_leaderboard.csv'}")
    print(f"      Leaderboard Markdown: {REPO_ROOT / 'MODEL_LEADERBOARD.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
