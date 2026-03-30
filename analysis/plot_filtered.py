"""Generate filtered versions of the feedback-round and category plots, excluding
specific low-tier models to keep the charts readable for presentation purposes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Dict, List
from plot_style import COLORS, apply_plot_style

apply_plot_style()

# Models removed in addition to the standing codestral-22b / glm-5 exclusions
EXTRA_EXCLUDED = {
    "qwen2.5-coder-32b-instruct",
    "llama-3.3-70b-instruct",
    "qwen3-coder",
    "gpt-oss_120b",
    "gpt-oss_20b",
    "claude-haiku-4-5-20251001",
    "gpt-5.2",
}

ALL_EXCLUDED = {"codestral-22b", "glm-5"} | EXTRA_EXCLUDED


# ---------------------------------------------------------------------------
# Plot 1 – Cumulative success by feedback round
# ---------------------------------------------------------------------------

def get_model_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    models = [m for m in df['Model'].unique() if m.lower() not in {e.lower() for e in ALL_EXCLUDED}]

    model_data = {}
    sample_counts = {}

    for model in models:
        model_df = df[df['Model'] == model]
        total_samples = len(model_df)
        sample_counts[model] = total_samples

        cumulative_success = []

        for r in range(6):
            success_count = 0
            for _, row in model_df.iterrows():
                is_success = False
                duration = 0

                if str(row['Success']) == 'True':
                    is_success = True

                rounds_content = [row[f'Feedback_Round_{i}'] for i in range(6)]
                found_string = False
                for idx, content in enumerate(rounds_content):
                    if pd.isna(content) or content == '':
                        break
                    duration = idx
                    if 'unit tests were successful' in str(content):
                        found_string = True
                        break

                if found_string:
                    is_success = True

                if is_success and duration <= r:
                    success_count += 1

            cumulative_success.append((success_count / total_samples) * 100)

        model_data[model] = cumulative_success

    # Sort best → worst by R5 cumulative success
    models = sorted(models, key=lambda m: model_data[m][5], reverse=True)
    model_data = {m: model_data[m] for m in models}
    sample_counts = {m: sample_counts[m] for m in models}

    return model_data, sample_counts


def visualize_success_by_llm(model_data: Dict[str, List[float]], sample_counts: Dict[str, int] = None, output_file: str = '../plots/success_by_model_by_feedbackround_filtered.png'):
    models = list(model_data.keys())
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(12, 7))

    feedback_labels = [
        "No Feedback",
        "1 Feedback",
        "2 Feedbacks",
        "3 Feedbacks",
        "4 Feedbacks",
        "5 Feedbacks",
    ]

    for i in rounds:
        round_values = [model_data[model][i] for model in models]

        errors = []
        for model in models:
            if sample_counts and model in sample_counts:
                p = model_data[model][i] / 100.0
                n = sample_counts[model]
                se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
                errors.append(1.96 * se * 100)
            else:
                errors.append(0)

        bar_positions = x + (i - 2.5) * bar_width
        ax.bar(
            bar_positions,
            round_values,
            width=bar_width,
            label=feedback_labels[i],
            color=COLORS[i % len(COLORS)],
            yerr=errors if sample_counts else None,
            capsize=3,
            error_kw={'elinewidth': 1, 'alpha': 0.7}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Cumulative Successful Runs")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_title("Cumulative Successful Code Generations by Feedback Round in Percent\n(with 95% Confidence Intervals)")
    ax.legend(title="Feedback Rounds", loc='best')
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


# ---------------------------------------------------------------------------
# Plot 2 – Success by task category
# ---------------------------------------------------------------------------

def get_category_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    models = [m for m in df['Model'].unique() if m.lower() not in {e.lower() for e in ALL_EXCLUDED}]

    categories = [
        'String Handling',
        'List or Array Operation',
        'Mathematical Calculation',
        'Logical Condition',
        'ABAP Database Operation'
    ]

    data = {model: {} for model in models}
    counts = {model: {} for model in models}

    for model in models:
        for category in categories:
            if category in df.columns:
                subset = df[(df['Model'] == model) & (df[category].astype(str) == 'True')]
            else:
                subset = pd.DataFrame()

            total = len(subset)
            counts[model][category] = total

            if total == 0:
                data[model][category] = 0.0
                continue

            success_count = 0
            for _, row in subset.iterrows():
                is_success = False

                if str(row['Success']) == 'True':
                    is_success = True
                else:
                    rounds_content = [row[f'Feedback_Round_{i}'] for i in range(6)]
                    for content in rounds_content:
                        if pd.isna(content) or content == '':
                            continue
                        if 'unit tests were successful' in str(content):
                            is_success = True
                            break

                if is_success:
                    success_count += 1

            data[model][category] = (success_count / total) * 100

    # Sort best → worst by average success across all categories
    models = sorted(models, key=lambda m: sum(data[m].values()) / len(categories), reverse=True)
    data = {m: data[m] for m in models}
    counts = {m: counts[m] for m in models}

    return data, counts


def visualize_prompt_classification_success(data: Dict[str, Dict[str, float]], counts: Dict[str, Dict[str, int]] = None, output_file: str = '../plots/success_by_model_by_task_category_filtered.png'):
    models = list(data.keys())
    categories = list(data[models[0]].keys())

    num_categories = len(categories)
    bar_width = 0.13
    x = np.arange(len(models))

    plt.figure(figsize=(15, 8))

    for i, category in enumerate(categories):
        values = [data[model][category] for model in models]

        errors = []
        for model in models:
            if counts and category in counts[model]:
                n = counts[model][category]
                p = data[model][category] / 100.0
                se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
                errors.append(1.96 * se * 100)
            else:
                errors.append(0)

        plt.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=category,
            color=COLORS[i % len(COLORS)],
            yerr=errors if counts else None,
            capsize=3,
            error_kw={'elinewidth': 1, 'alpha': 0.7}
        )

    plt.xticks(x + bar_width * (num_categories - 1) / 2, models, rotation=15, ha="right")
    plt.ylabel("Success Rate")
    plt.title("Model Performance by Task Category (with 95% CIs)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
    plt.legend(bbox_to_anchor=(0.5, 1.08), loc='lower center', ncol=3)
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_round, counts_round = get_model_data_from_csv('../data/results.csv')
    if data_round:
        visualize_success_by_llm(data_round, counts_round)

    data_cat, counts_cat = get_category_data_from_csv('../data/results.csv')
    if data_cat:
        visualize_prompt_classification_success(data_cat, counts_cat)
