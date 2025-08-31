import json
import os
import re
from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from abap_errors import Error, ERRORS_BY_CATEGORY
from generate_llm_answers import REPETITIONS
import llms

MODELS_TO_RUN = [llm["name"].replace(":", "_") for llm in llms.MODELS_TO_RUN]
NUM_PROMPTS = len(os.listdir("prompts/"))

RESULTS_FILE = "analyse_results/results.json"
SYNTAX_ERROR_FILE = "analyse_results/syntax_errors.json"


def categorize_errors() -> Dict[str, Dict[str, List[str]]]:
    results = {
        model: {
            "The syntax check failed": [],
            "The class could not be created": [],
            "The unit tests were successful.": [],
            "The unit test failed": [],
            "The unittest syntax check failed": [],
            "The source code could not be set": [],
            "There should only be the one public method.": [],
            "Class name not found.": [],
            "other": [],
        }
        for model in MODELS_TO_RUN
    }

    for model_name in MODELS_TO_RUN:
        filename = f"{model_name}.json"
        model_results = results[model_name]

        with open(filename, "r", encoding="utf-8") as file:
            prompt_files = json.load(file)
            for prompt_file, chats in prompt_files.items():
                for chat in chats:
                    for message in chat[2:]:
                        if message["role"] == "user":
                            message_text = message["content"]
                            added = False
                            for key in model_results:
                                if message_text.startswith(key):
                                    model_results[key].append(message_text)
                                    added = True

                            if not added:
                                model_results["other"].append(message_text)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(results, indent=4, ensure_ascii=False))

    return results


def analyse_syntax_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:

    syntax_errors = {}
    for model_name in MODELS_TO_RUN:
        runs = [
            eval(m[m.index("\n") + 1 :])
            for m in results[model_name]["The syntax check failed"]
        ]

        errors = [e for msgs in runs for e in msgs if e["type"] == "E"]
        warnings = [w for msgs in runs for w in msgs if w["type"] == "W"]

        print(
            f"For {model_name} there are {len(errors)} errors and {len(warnings)} warnings."
        )

        model_syntax_errors = {}
        for error_category_name, error_category in ERRORS_BY_CATEGORY.items():
            model_syntax_errors[error_category_name] = {}
            for error_type in error_category:
                model_syntax_errors[error_category_name][error_type.name] = []

        for error in errors:
            found = False
            for error_category_name, error_category in ERRORS_BY_CATEGORY.items():
                for error_type in error_category:
                    if (
                        re.search(error_type.regex, error["short_text"], re.IGNORECASE)
                        != None
                    ):
                        model_syntax_errors[error_category_name][
                            error_type.name
                        ].append(error["short_text"])
                        found = True
                        break
                    if found:
                        break
        syntax_errors[model_name] = model_syntax_errors

    with open(SYNTAX_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(syntax_errors, indent=4, ensure_ascii=False))

    return syntax_errors


def visualize_syntax_errors(data: Dict[str, Dict[str, Dict[str, List[str]]]]):
    counts = []
    for model, categories in data.items():
        for category, errors in categories.items():
            total_entries = sum(len(v) for v in errors.values())
            counts.append(
                {"Model": model, "Category": category, "Count": total_entries}
            )

    df = pd.DataFrame(counts)
    pivot_df = df.pivot(index="Model", columns="Category", values="Count").fillna(0)
    ax = pivot_df.plot(kind="bar", figsize=(10, 6))

    plt.title("Error Counts per Category per Model")
    plt.ylabel("Count")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.legend(title="Category")
    plt.tight_layout()
    plt.show()


def analyse_success_by_round() -> Dict[str, List[int]]:
    successes = {}
    for model_name in MODELS_TO_RUN:
        filename = f"{model_name}.json"
        successes[model_name] = [0, 0, 0, 0, 0, 0]

        with open(filename, "r", encoding="utf-8") as file:
            prompt_files = json.load(file)
            for prompt_file, chats in prompt_files.items():
                for chat in chats:
                    user_responses = [msg for msg in chat if msg["role"] == "user"][1:]
                    for msg_num, msg in enumerate(user_responses):
                        if msg["content"] == "The unit tests were successful.":
                            successes[model_name][msg_num] += 1

    cumulative_data = {
        model: list(np.cumsum(values)) for model, values in successes.items()
    }
    return cumulative_data


def visualize_success_by_round(success_data: Dict[str, List[int]]):
    models = list(success_data.keys())
    num_models = len(models)
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(rounds))

    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][
        :num_models
    ]

    for idx, model in enumerate(models):
        model_data = success_data[model]
        bar_positions = x + (idx - (num_models - 1) / 2) * bar_width
        ax.bar(
            bar_positions,
            model_data,
            width=bar_width,
            label=model,
            color=model_colors[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.set_xlabel("Rounds of Feedback")
    ax.set_ylabel("Cumulative Successful Prompts")
    ax.set_title("Cumulative Successful Code Generations by Feedback Round")
    ax.legend(title="Model")

    plt.tight_layout()
    plt.show()


def visualize_success_by_llm(success_data: Dict[str, List[int]]):
    models = list(success_data.keys())
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))

    feedback_labels = [
        "No Feedback",
        "1 Feedback",
        "2 Feedbacks",
        "3 Feedbacks",
        "4 Feedbacks",
        "5 Feedbacks",
    ]
    custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in rounds:
        round_values = [
            (success_data[model][i] / NUM_PROMPTS * REPETITIONS * 100)
            for model in models
        ]
        bar_positions = x + (i - 2.5) * bar_width
        ax.bar(
            bar_positions,
            round_values,
            width=bar_width,
            label=feedback_labels[i],
            color=custom_colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Cumulative Successful Runs")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Cumulative Successful Code Generations by Feedback Round in Percent")
    ax.legend(title="Feedback Rounds")
    plt.tight_layout()
    plt.show()


def prompt_classification() -> Dict[str, Dict[str, float]]:
    df = pd.read_csv("analyse_results/prompt_classification.csv", sep=";")
    categories: Dict[str, List[int]] = {}
    for classification in df.columns[3:]:
        categories[classification] = list(df[df[classification] == "X"]["HumanEval/Nr"])

    successes = {}
    for model_name in MODELS_TO_RUN:
        successes[model_name] = {}
        for category, category_prompts in categories.items():
            filename = f"{model_name}.json"
            successes[model_name][category] = 0

            with open(filename, "r", encoding="utf-8") as file:
                prompt_files = json.load(file)
                for prompt_file, chats in prompt_files.items():
                    if prompt_file[:-4] in category_prompts:
                        for chat in chats:
                            user_responses = [
                                msg for msg in chat if msg["role"] == "user"
                            ][1:]
                            for msg_num, msg in enumerate(user_responses):
                                if msg["content"] == "The unit tests were successful.":
                                    successes[model_name][category] += 1

    relative_successrate = successes
    for model_name in MODELS_TO_RUN:
        for category, category_prompts in categories.items():
            current_value = relative_successrate[model_name][category]
            relative_successrate[model_name][category] = round(current_value / (
                len(category_prompts) * REPETITIONS
            ), 4)

    return relative_successrate

def visualize_prompt_classification_success(data: Dict[str, Dict[str, float]]):
    models = list(data.keys())
    categories = list(next(iter(data.values())).keys())
    num_models = len(models)
    num_categories = len(categories)

    bar_width = 0.13
    x = np.arange(num_models)
    plt.figure(figsize=(15,6))
    for i, category in enumerate(categories):
        values = [data[model][category] for model in models]
        plt.bar(x + i*bar_width, values, width=bar_width, label=category)
    plt.xticks(x + bar_width*(num_categories-1)/2, models, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Model Performance by Task Category')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) 
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    error_categories = categorize_errors()
    syntax_errors = analyse_syntax_error(error_categories)
    visualize_syntax_errors(syntax_errors)
    successes = analyse_success_by_round()
    visualize_success_by_round(successes)
    visualize_success_by_llm(successes)
    relative_prompt_success = prompt_classification()
    visualize_prompt_classification_success(relative_prompt_success)
