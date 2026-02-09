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
        "file": "km_survival_curve_all_models.png",
        "title": "Kaplan-Meier Survival Curves (All Tasks)",
        "description": (
            "Survival here means the share of runs that are still unsolved after each round. "
            "A faster downward curve indicates faster convergence to passing solutions. "
            "Lower survival at the same round is better."
        ),
    },
    {
        "file": "km_survival_curve_abap.png",
        "title": "Kaplan-Meier Survival Curves (ABAP DB Tasks)",
        "description": (
            "The same survival analysis, restricted to ABAP Database Operation tasks only. "
            "This isolates performance on SAP/ABAP-specific problems and highlights which models stay reliable in this harder subset."
        ),
    },
    {
        "file": "error_categories_by_model.png",
        "title": "Response Error Stage Distribution",
        "description": (
            "Breaks down where failures occur in the SAP validation pipeline: class creation, syntax checks, or unit tests. "
            "Interpret this in order: a low unit-test failure rate is only meaningful if the model reliably passes creation and syntax stages first."
        ),
    },
    {
        "file": "syntax_error_types_by_model.png",
        "title": "Syntax Error Category Distribution",
        "description": (
            "Splits syntax failures into lexical/token, structural, type/conversion, OOP, and declaration errors. "
            "This shows whether a model mostly fails on basic ABAP syntax mechanics or on deeper typing and object-oriented constraints."
        ),
    },
    {
        "file": "failed_tasks_by_category.png",
        "title": "Tasks Failed By All Models",
        "description": (
            "Shows the percentage of tasks in each category that no evaluated model could solve. "
            "Higher values indicate benchmark blind spots and hard task regions where model capabilities are still weak."
        ),
    },
    {
        "file": "success_distribution.png",
        "title": "Task-Level Success Distribution",
        "description": (
            "Displays the distribution of per-task success rates, not just a single average score. "
            "Use it to distinguish stable models from polarized ones that solve some tasks almost always and others almost never."
        ),
    },
]


MAIN_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text", "default_sort": "none"},
    {"key": "Success_R5_pct", "label": "Success R5", "type": "percent", "default_sort": "desc"},
    {"key": "Success_R0_pct", "label": "Success R0", "type": "percent", "default_sort": "desc"},
    {"key": "AUC_Success_pct", "label": "AUC (R0-R5)", "type": "percent", "default_sort": "desc"},
    {
        "key": "Median_Success_Round",
        "label": "Median Feedbacks To Success",
        "type": "number",
        "decimals": 1,
        "default_sort": "asc",
    },
    {
        "key": "R0_Reaches_UnitTests_pct",
        "label": "R0 Reaches Unit Tests",
        "type": "percent",
        "default_sort": "desc",
    },
    {"key": "PassAt5_Final_pct", "label": "pass@5 (Final)", "type": "percent", "default_sort": "desc"},
    {"key": "Prompts_Solved_Any_pct", "label": "Prompts Solved >=1/10", "type": "percent", "default_sort": "desc"},
    {"key": "Prompts_Solved_All_pct", "label": "Prompts Solved 10/10", "type": "percent", "default_sort": "desc"},
    {"key": "Max_LLM_Calls_Per_Run", "label": "Max Rounds Tested", "type": "number", "default_sort": "desc"},
]

ROUND_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text"},
    {"key": "Success_R0_pct", "label": "R0", "type": "percent"},
    {"key": "Success_R1_pct", "label": "R1", "type": "percent"},
    {"key": "Success_R2_pct", "label": "R2", "type": "percent"},
    {"key": "Success_R3_pct", "label": "R3", "type": "percent"},
    {"key": "Success_R4_pct", "label": "R4", "type": "percent"},
    {"key": "Success_R5_pct", "label": "R5", "type": "percent"},
]

CATEGORY_COLUMNS = [
    {"key": "Model_Display", "label": "Model", "type": "text"},
    {"key": "Success_R5_StringHandling_pct", "label": "String Handling", "type": "percent"},
    {
        "key": "Success_R5_ListOrArrayOperation_pct",
        "label": "List Or Array Operation",
        "type": "percent",
    },
    {
        "key": "Success_R5_MathematicalCalculation_pct",
        "label": "Mathematical Calculation",
        "type": "percent",
    },
    {"key": "Success_R5_LogicalCondition_pct", "label": "Logical Condition", "type": "percent"},
    {
        "key": "Success_R5_ABAPDatabaseOperation_pct",
        "label": "ABAP Database Operation",
        "type": "percent",
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


def _build_dashboard_json() -> Path:
    leaderboard_path = DATA_DIR / "model_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Missing {leaderboard_path}. Run generate_leaderboard.py first."
        )

    df = pd.read_csv(leaderboard_path)
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
