import json
import os
import re
from typing import Dict, List
from matplotlib import pyplot as plt
import matplotlib
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
CLASS_CREATION_ERROR_FILE = "analyse_results/class_creation_errors.json"
UNITTEST_RESULT_ERROR_FILE = "analyse_results/unittest_result_errors.json"
UNITTEST_SYNTAX_ERROR_FILE = "analyse_results/unittest_syntax_errors.json"
SOURCE_CODE_ERROR_FILE = "analyse_results/source_code_errors.json"
PUBLIC_METHOD_ERROR_FILE = "analyse_results/public_method_errors.json"
CLASS_NAME_NOT_FOUND_FILE = "analyse_results/class_name_errors.json"
UNIT_TEST_SUCCESS_FILE = "analyse_results/successful.json"


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

matplotlib.use("Qt5Agg")


def categorize_errors() -> Dict[str, Dict[str, List[str]]]:
    results = {
        model: {
            "The syntax check failed": [],
            "The class could not be created": [],
            "The unit tests were successful.": [],
            "The unit test failed": [],
            "The unittest syntax check failed": [],
            "The source code could not be set": [],
            "There should only be the one public method": [],
            "Class name not found": [],
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
        runs = [eval(m[m.index("\n") + 1 :]) for m in results[model_name]["The syntax check failed"]]

        errors = [e for msgs in runs for e in msgs if e["type"] == "E"]
        warnings = [w for msgs in runs for w in msgs if w["type"] == "W"]

        print(f"For {model_name} there are {len(errors)} errors and {len(warnings)} warnings.")

        model_syntax_errors = {}
        for error_category_name, error_category in ERRORS_BY_CATEGORY.items():
            model_syntax_errors[error_category_name] = {}
            for error_type in error_category:
                model_syntax_errors[error_category_name][error_type.name] = []

        for error in errors:
            found = False
            for error_category_name, error_category in ERRORS_BY_CATEGORY.items():
                for error_type in error_category:
                    if re.search(error_type.regex, error["short_text"], re.IGNORECASE) != None:
                        model_syntax_errors[error_category_name][error_type.name].append(error["short_text"])
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
            counts.append({"Model": model, "Category": category, "Count": total_entries})

    df = pd.DataFrame(counts)

    df["Model"] = pd.Categorical(df["Model"], categories=df["Model"].unique(), ordered=True)
    df["Category"] = pd.Categorical(df["Category"], categories=df["Category"].unique(), ordered=True)

    pivot_df = df.pivot(index="Model", columns="Category", values="Count").fillna(0)
    relative_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    print(relative_df)

    ax = relative_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Relative Error Distribution per Category per Model")
    plt.ylabel("Proportion of Errors")
    plt.xlabel("Model")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(rotation=45)
    plt.legend(title="Category")
    plt.tight_layout()
    plt.show()


def analyse_class_creation_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[Dict[str, str]]]:
    creation_errors = {}

    for model_name in MODELS_TO_RUN:
        raw_errors = results[model_name]["The class could not be created"]
        parsed_errors = []
        for e in raw_errors:
            try:
                error_dict = eval(e[e.index("\n") + 1 :])
                parsed_errors.append(error_dict)
            except Exception as ex:
                print(f"Failed to parse error for {model_name}: {e}")
                print(ex)

        print(f"For {model_name} there are {len(parsed_errors)} class creation errors.")

        creation_errors[model_name] = parsed_errors

    with open(CLASS_CREATION_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(creation_errors, indent=4, ensure_ascii=False))

    return creation_errors


def analyse_unittest_result_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[List[Dict[str, str]]]]:
    unit_test_errors = {}
    for model_name in MODELS_TO_RUN:
        raw_errors = results[model_name]["The unit test failed"]
        parsed_errors = []

        for e in raw_errors:
            e = re.sub(r"<Element 'stack' at .*?>", "''", e)
            try:
                error_list = eval(e[e.index("\n") + 1 : e.rindex("\n")])
                parsed_errors.append(error_list)
            except Exception as ex:
                print(f"Failed to parse unit test error for {model_name}: {e}")
                print(ex)

        print(f"For {model_name} there are {sum(len(f) for f in parsed_errors)} unit test failures.")
        unit_test_errors[model_name] = parsed_errors

    with open(UNITTEST_RESULT_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(unit_test_errors, indent=4, ensure_ascii=False, default=str))

    return unit_test_errors


def analyse_unittest_syntax_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[List[Dict[str, str]]]]:
    unittest_syntax_errors: Dict[str, List[List[Dict[str, str]]]] = {}

    for model_name in MODELS_TO_RUN:
        raw_errors = results[model_name]["The unittest syntax check failed"]
        parsed_errors: List[List[Dict[str, str]]] = []

        for e in raw_errors:
            try:
                error_list = eval(e[e.index("\n") + 1 : e.rindex("\n")])
                parsed_errors.append(error_list)
            except Exception as ex:
                print(f"Failed to parse unittest syntax error for {model_name}: {e}")
                print(ex)

        total = sum(len(f) for f in parsed_errors)
        print(f"For {model_name} there are {total} unittest syntax errors.")
        unittest_syntax_errors[model_name] = parsed_errors

    with open(UNITTEST_SYNTAX_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(unittest_syntax_errors, indent=4, ensure_ascii=False, default=str))

    return unittest_syntax_errors


def analyse_source_code_set_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    source_code_errors: Dict[str, List[str]] = {}

    for model_name in MODELS_TO_RUN:
        errors = results[model_name].get("The source code could not be set", [])
        print(f"For {model_name} there are {len(errors)} 'source code could not be set' errors.")
        source_code_errors[model_name] = errors

    with open(SOURCE_CODE_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(source_code_errors, indent=4, ensure_ascii=False))

    return source_code_errors


def analyse_public_method_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:

    public_method_errors: Dict[str, List[str]] = {}

    for model_name in MODELS_TO_RUN:
        errors = results[model_name].get("There should only be the one public method", [])
        print(f"For {model_name} there are {len(errors)} 'public method' errors.")
        public_method_errors[model_name] = errors

    with open(PUBLIC_METHOD_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(public_method_errors, indent=4, ensure_ascii=False))

    return public_method_errors


def analyse_class_name_not_found_error(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    class_name_errors: Dict[str, List[str]] = {}

    for model_name in MODELS_TO_RUN:
        errors = results[model_name].get("Class name not found", [])
        print(f"For {model_name} there are {len(errors)} 'class name not found' errors.")
        class_name_errors[model_name] = errors

    with open(CLASS_NAME_NOT_FOUND_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(class_name_errors, indent=4, ensure_ascii=False))

    return class_name_errors


def analyse_unit_test_success(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:

    unit_test_success: Dict[str, List[str]] = {}

    for model_name in MODELS_TO_RUN:
        successes = results[model_name].get("The unit tests were successful.", [])
        print(f"For {model_name} there are {len(successes)} successful unit test messages.")
        unit_test_success[model_name] = successes

    with open(UNIT_TEST_SUCCESS_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(unit_test_success, indent=4, ensure_ascii=False))

    return unit_test_success


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

    cumulative_data = {model: list(np.cumsum(values)) for model, values in successes.items()}
    return cumulative_data


def summarize_all_errors(
    syntax_errors: Dict[str, dict],
    class_creation_errors: Dict[str, list],
    unit_test_errors: Dict[str, list],
    unittest_syntax_errors: Dict[str, list],
    source_code_errors: Dict[str, list],
    public_method_errors: Dict[str, list],
    class_name_errors: Dict[str, list],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}

    all_models = set(
        list(syntax_errors.keys())
        + list(class_creation_errors.keys())
        + list(unit_test_errors.keys())
        + list(unittest_syntax_errors.keys())
        + list(source_code_errors.keys())
        + list(public_method_errors.keys())
        + list(class_name_errors.keys())
        # + list(unit_test_success.keys())
    )

    for model in all_models:
        summary[model] = {
            "syntax_errors": sum(
                len(errors) for category in syntax_errors.get(model, {}).values() for errors in category.values()
            ),
            "class_creation_errors": len(class_creation_errors.get(model, [])),
            "unit_test_failures": sum(len(lst) for lst in unit_test_errors.get(model, [])),
            "unittest_syntax_errors": sum(len(lst) for lst in unittest_syntax_errors.get(model, [])),
            "source_code_errors": len(source_code_errors.get(model, [])),
            "public_method_errors": len(public_method_errors.get(model, [])),
            "class_name_errors": len(class_name_errors.get(model, [])),
            # "unit_test_successes": len(unit_test_success.get(model, [])),
        }

    return summary


def sum_summary(summary: dict) -> dict:
    total_per_model = {}

    for model, counts in summary.items():
        total = 0
        for key, value in counts.items():
            total += value
        total_per_model[model] = total

    return total_per_model


def visualize_success_by_round(model_data: Dict[str, List[float]]):
    models = list(model_data.keys())
    num_models = len(models)
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(rounds))

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, model in enumerate(models):

        bar_positions = x + (idx - (num_models - 1) / 2) * bar_width
        ax.bar(
            bar_positions,
            height=model_data[model],
            width=bar_width,
            label=model,
            color=COLORS[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.set_xlabel("Rounds of Feedback")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel("Cumulative Successful Prompts")
    ax.set_title("Cumulative Successful Code Generations by Feedback Round")
    ax.legend(title="Model")

    plt.tight_layout()
    plt.show()


def visualize_success_by_llm(model_data: Dict[str, List[float]]):
    models = list(model_data.keys())
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

    for i in rounds:
        round_values = [(model_data[model][i]) for model in models]
        bar_positions = x + (i - 2.5) * bar_width
        ax.bar(
            bar_positions,
            round_values,
            width=bar_width,
            label=feedback_labels[i],
            color=COLORS[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Cumulative Successful Runs")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
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
                            user_responses = [msg for msg in chat if msg["role"] == "user"][1:]
                            for msg_num, msg in enumerate(user_responses):
                                if msg["content"] == "The unit tests were successful.":
                                    successes[model_name][category] += 1

    relative_successrate = successes
    for model_name in MODELS_TO_RUN:
        for category, category_prompts in categories.items():
            current_value = relative_successrate[model_name][category]
            relative_successrate[model_name][category] = round(current_value / (len(category_prompts) * REPETITIONS), 4)

    return relative_successrate


def visualize_prompt_classification_success(data: Dict[str, Dict[str, float]]):
    models = list(data.keys())
    categories = list(next(iter(data.values())).keys())
    num_models = len(models)
    num_categories = len(categories)

    bar_width = 0.13
    x = np.arange(num_models)
    plt.figure(figsize=(15, 6))
    for i, category in enumerate(categories):
        values = [data[model][category] for model in models]
        plt.bar(x + i * bar_width, values, width=bar_width, label=category)
    plt.xticks(x + bar_width * (num_categories - 1) / 2, models, rotation=45, ha="right")
    plt.ylabel("Success Rate")
    plt.title("Model Performance by Task Category")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.tight_layout()
    plt.show()


def filter_error_type(text: str) -> str:
    text = text.strip().lower()
    match = re.match(
        r".*(syntax check failed|class could not be created|unit tests were successful|unit test failed|unittest syntax check failed|source code could not be set|there should only be the one public method|class name not found)(.*)",
        text,
        re.DOTALL,
    )

    if match and match.group(1) in [
        "source code could not be set",
        "unittest syntax check failed",
        "there should only be the one public method",
        "class name not found",
    ]:
        return "syntax check failed"

    if match:
        return match.group(1)

    print(text)
    return "other"


def create_csv():
    import csv

    header = [
        "Model",
        "Prompt",
        "Repetition",
        "Feedback_Round_0",
        "Feedback_Round_1",
        "Feedback_Round_2",
        "Feedback_Round_3",
        "Feedback_Round_4",
        "Feedback_Round_5",
    ]
    rows = []

    for model_name in MODELS_TO_RUN:
        filename = f"{model_name}.json"

        with open(filename, "r", encoding="utf-8") as file:
            prompt_files = json.load(file)
            for prompt_file, chats in prompt_files.items():
                for repetition, chat in enumerate(chats):
                    user_responses = [msg for msg in chat if msg["role"] == "user"][1:]
                    user_responses = [filter_error_type(m["content"]) for m in user_responses]

                    rows.append([model_name, prompt_file[:-4], repetition, *user_responses])

    with open("analyse_results/prompt_success_by_round.csv", "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=";")
        csvwriter.writerow(header)
        csvwriter.writerows(rows)

    print("CSV file 'analyse_results/prompt_success_by_round.csv' created successfully.")


def visualize_combined_summary_errors(summary: dict):
    combined_summary = {}
    for model, errors in summary.items():
        combined = {
            "class_creation": errors.get("class_creation_errors", 0) + errors.get("class_name_errors", 0),
            "syntax_errors": errors.get("syntax_errors", 0) + errors.get("source_code_errors", 0) + errors.get("public_method_errors", 0),
            "unit_test_errors": errors.get("unit_test_failures", 0) + errors.get("unittest_syntax_errors", 0),
        }
        total_errors = sum(combined.values())
        if total_errors > 0:
            combined_summary[model] = {k: v / total_errors for k, v in combined.items()}
        else:
            combined_summary[model] = {k: 0 for k in combined.keys()}

    df = pd.DataFrame(combined_summary).T  # rows = models, columns = error types
    df = df[["class_creation", "syntax_errors", "unit_test_errors"]]  # ensure column order

    model_order = [
        "llama-3.3-70b-instruct",
        "qwen2.5-coder-32b-instruct",
        "codestral-22b",
        "gpt-oss_20b",
        "gpt-oss_120b",
        "qwen3-coder",
        "gpt-5-2025-08-07",
        "claude-sonnet-4-20250514"
    ]
    df = df.reindex(model_order)

    error_types = df.columns.tolist()
    models = df.index.tolist()
    num_models = len(models)
    num_error_types = len(error_types)
    bar_width = 0.2
    x = np.arange(num_models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, etype in enumerate(error_types):
        error_data = df[etype].values
        bar_positions = x + (idx - (num_error_types - 1) / 2) * bar_width
        ax.bar(
            bar_positions,
            error_data,
            width=bar_width,
            label=etype.replace("_", " ").title(),
            color=COLORS[idx % len(COLORS)]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Models")
    ax.set_ylabel("Percentage of Errors")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Relative Distribution of Errors by Model")
    ax.legend(title="Error Type")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    error_categories = categorize_errors()
    syntax_errors = analyse_syntax_error(error_categories)
    class_creation_errors = analyse_class_creation_error(error_categories)
    unit_test_errors = analyse_unittest_result_error(error_categories)
    unittest_syntax_errors = analyse_unittest_syntax_error(error_categories)
    source_code_errors = analyse_source_code_set_error(error_categories)
    public_method_errors = analyse_public_method_error(error_categories)
    class_name_errors = analyse_class_name_not_found_error(error_categories)
    unit_test_success = analyse_unit_test_success(error_categories)

    summary = summarize_all_errors(
        syntax_errors,
        class_creation_errors,
        unit_test_errors,
        unittest_syntax_errors,
        source_code_errors,
        public_method_errors,
        class_name_errors,
    )

    print(summary)
    visualize_combined_summary_errors(summary)

    total_counts = sum_summary(summary)
    print("----")
    visualize_syntax_errors(syntax_errors)

    successes = analyse_success_by_round()
    successes = {key: [int(x)/1600*100 for x in value] for key, value in successes.items()}
    print(successes)
    visualize_success_by_round(successes)
    visualize_success_by_llm(successes)
    create_csv()

    relative_prompt_success = prompt_classification()
    print(relative_prompt_success)
    visualize_prompt_classification_success(relative_prompt_success)
