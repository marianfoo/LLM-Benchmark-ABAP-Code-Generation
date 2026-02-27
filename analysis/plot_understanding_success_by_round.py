#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_style import COLORS, apply_plot_style


apply_plot_style()


def get_model_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("understanding_model_leaderboard.csv is empty")

    # Keep leaderboard order (already sorted by score)
    models = df["Model"].tolist()

    model_data = {}
    sample_counts = {}
    for _, row in df.iterrows():
        model = row["Model"]
        sample_counts[model] = int(row.get("Runs", 0))
        model_data[model] = [
            float(row.get(f"Success_R{r}_pct", 0.0)) for r in range(6)
        ]

    return models, model_data, sample_counts


def _ci95_margin_pct(pct: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, pct / 100.0))
    se = np.sqrt((p * (1.0 - p)) / n)
    return float(1.96 * se * 100.0)


def plot_success_by_round(models, model_data, sample_counts):
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(12, 7))

    round_labels = [
        "No Feedback",
        "1 Feedback",
        "2 Feedbacks",
        "3 Feedbacks",
        "4 Feedbacks",
        "5 Feedbacks",
    ]

    for i in rounds:
        values = [model_data[m][i] for m in models]
        errs = [_ci95_margin_pct(model_data[m][i], sample_counts.get(m, 0)) for m in models]
        positions = x + (i - 2.5) * bar_width

        ax.bar(
            positions,
            values,
            width=bar_width,
            label=round_labels[i],
            color=COLORS[i % len(COLORS)],
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("Cumulative Successful Understanding Runs")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_title("Cumulative Understanding Success by Feedback Round in Percent\n(with 95% Confidence Intervals)")
    ax.legend(title="Feedback Rounds", loc="best")

    plt.tight_layout()
    out_file = "../plots/understanding_success_by_model_by_feedbackround_in_percent.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


def print_table(models, model_data):
    print("\nUnderstanding Success Rates by Feedback Round:")
    print(
        f"{'Model':<35} | {'No Feedback':<12} | {'1 Feedback':<12} | {'2 Feedbacks':<12} | {'3 Feedbacks':<12} | {'4 Feedbacks':<12} | {'5 Feedbacks':<12}"
    )
    print("-" * 160)
    for model in models:
        values = model_data[model]
        row = f"{model:<35}"
        for val in values:
            row += f" | {val:11.2f}%"
        print(row)


def main():
    models, model_data, sample_counts = get_model_data("../data/understanding_model_leaderboard.csv")
    plot_success_by_round(models, model_data, sample_counts)
    print_table(models, model_data)


if __name__ == "__main__":
    main()
