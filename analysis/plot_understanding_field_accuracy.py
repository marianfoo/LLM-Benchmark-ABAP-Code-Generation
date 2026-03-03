#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_style import COLORS, apply_plot_style


apply_plot_style()

FIELD_COLUMNS = [
    ("Field_method_name_final_acc_pct", "method_name"),
    ("Field_return_param_name_final_acc_pct", "return_param_name"),
    ("Field_loop_count_final_acc_pct", "loop_count"),
    ("Field_has_select_final_acc_pct", "has_select"),
    ("Field_test_method_count_final_acc_pct", "test_method_count"),
    ("Field_assert_call_count_final_acc_pct", "assert_call_count"),
]


def main():
    df = pd.read_csv("../data/understanding_model_leaderboard.csv")
    if df.empty:
        raise RuntimeError("understanding_model_leaderboard.csv is empty")

    # Exclude models not yet validated for the understanding benchmark
    EXCLUDED = {"gemini-3.1-pro-preview"}
    df = df[~df["Model"].isin(EXCLUDED)].reset_index(drop=True)

    models = df["Model"].tolist()
    fields = [label for _, label in FIELD_COLUMNS]

    x = np.arange(len(models))
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (col, label) in enumerate(FIELD_COLUMNS):
        vals = [float(v) for v in df[col].tolist()]
        positions = x + (i - (len(FIELD_COLUMNS) - 1) / 2) * bar_width
        ax.bar(
            positions,
            vals,
            width=bar_width,
            label=label,
            color=COLORS[i % len(COLORS)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("Final Field Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_title("ABAP Understanding Field Accuracy by Model (Final Round)")
    ax.legend(title="Extracted Field", loc="best", ncols=2)

    plt.tight_layout()
    out_file = "../plots/understanding_field_accuracy_by_model_in_percent.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    main()
