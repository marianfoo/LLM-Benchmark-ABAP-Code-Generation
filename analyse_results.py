import json
import re
from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

MODELS_TO_RUN = [
    "llama-3.3-70b-instruct",
    "qwen2.5-coder-32b-instruct",
    "codestral-22b",
    "qwen3-coder",
    "gpt-oss_20b", 
    "gpt-oss_120b" 
]

RESULTS_FILE = "analyse_results/results.json"
SYNTAX_ERROR_FILE = "analyse_results/syntax_errors.json"


class Error:
    def __init__(self, name: str, regex: str, entries: List[str] = []) -> None:
        self.name: str = name
        self.regex: str = regex
        self.entries: List[str] = entries


SYNTAX_ERROR_TYPES = [
    Error("Unable to interpret x.", r'Unable to interpret ".*"\.'),
    Error("The statement x is unexpected.", r"The statement .* is unexpected"),
    Error("type x is unknown.", r"type \".*\" is unknown"),
    Error("Field x is unknown.", r"Field \".*\" is unknown."),
    Error("x is not an internal table.", r"\".*\" is not an internal table."),
    Error("x cannot be modified.", r"\".*\" cannot be modified."),
    Error("x is not a constant.", r"\".*\" is not a constant."),
    Error(
        "x is missing between y and z.",
        r"\".*\" is missing between \".*\" and \".*\".",
    ),
    Error("Class statement PUBLIC is missing.", r"The addition \"PUBLIC\""),
    Error("A x parameter must be fully typed.", r"A .* parameter must be fully typed."),
    Error("The x does not have a structure.", r"The .* does not have a structure"),
    Error("The statement ended unexpectedly.", r"The statement .* ended unexpectedly."),
    Error(
        "Multiline literal are not allowed.",
        r"Literals across more than one line are not allowed.",
    ),
    Error(
        "Invalid line break in string template.",
        r"Invalid line break in string template.",
    ),
    Error(
        "Invalid expression limiter in string template.",
        r"Invalid expression limiter .* in string template.",
    ),
    Error(
        "The name is longer than the allowed 30 characters.",
        r"The name \".*\" is longer than the allowed 30 characters.",
    ),
    Error(
        "Method is unknown or PROTECTED or PRIVATE.",
        r"Method \".*\" is unknown or PROTECTED or PRIVATE.",
    ),
    Error("x is not valid.", r".* is not valid."),
    Error("x expected, not y.", r".* expected, not .*"),
    Error(
        "Returning parameters must be passed as value.",
        r"RETURNING parameters must be specified to be passed as VALUE().",
    ),
    Error(
        "Text literal is too long.",
        r"The text literal .* is longer than 255 characters. Check whether it ends correctly.",
    ),
    Error(
        "The method does not have a returning parameter.",
        r"The method .* does not have a RETURNING parameter",
    ),
    Error(
        "The generic types cannot be specified in returning parameters.",
        r"The generic types .* cannot be specified in RETURNING parameters.",
    ),
    Error("Method does not exist.", r"Method .* does not exist."),
    Error(
        "x must be a character-like data object.",
        r"must be a character-like data object \(data type C, N, D, T, or STRING\)\.",
    ),
    Error(
        "Type definition can only be specified once.",
        r"The addition \"... TYPE type\" can only be specified once.",
    ),
    Error("x expected after y", r".* expected after .*"),
    Error("Other", r".*"),
]


def categorize_errors():
    results = {
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

    for model_name in MODELS_TO_RUN:
        filename = f"{model_name}.json"

        with open(filename, "r", encoding="utf-8") as file:
            prompt_files = json.load(file)
            for prompt_file, chats in prompt_files.items():
                for chat in chats:
                    last_message = chat[-1]
                    if last_message["role"] == "user":
                        last_message = last_message["content"]
                        added = False
                        for key in results:
                            if last_message.startswith(key):
                                results[key].append(last_message)
                                added = True

                        if not added:
                            results["other"].append(last_message)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(results, indent=4, ensure_ascii=False))

    return results


def analyse_syntax_error(results: Dict):
    runs = [eval(m[m.index("\n") + 1 :]) for m in results["The syntax check failed"]]

    errors = [e for msgs in runs for e in msgs if e["type"] == "E"]
    warnings = [w for msgs in runs for w in msgs if w["type"] == "W"]


    print(f"There are {len(errors)} errors")
    print(f"There are {len(warnings)} warnings")

    syntax_errors = {}
    for error_type in SYNTAX_ERROR_TYPES:
        syntax_errors[error_type.name] = []

    for error in errors:
        for error_type in SYNTAX_ERROR_TYPES:
            if re.search(error_type.regex, error["short_text"], re.IGNORECASE) != None:
                syntax_errors[error_type.name].append(error["short_text"])
                break

    with open(SYNTAX_ERROR_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(syntax_errors, indent=4, ensure_ascii=False))

    return runs

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

def visualize_success_by_round(success_data: Dict[str, List[int]]):

    models = list(success_data.keys())
    num_models = len(models)
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(rounds))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:num_models]

    for idx, model in enumerate(models):
        model_data = success_data[model]
        bar_positions = x + (idx - (num_models - 1) / 2) * bar_width
        ax.bar(bar_positions, model_data, width=bar_width, label=model, color=model_colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels([ str(r) for r in rounds ])
    ax.set_xlabel('Rounds of Feedback')
    ax.set_ylabel('Cumulative Successful Prompts')
    ax.set_title('Cumulative Successful Code Generations by Feedback Round')
    ax.legend(title='Model')

    plt.tight_layout()
    plt.show()
   
   
   
def visualize_success_by_llm(success_data: Dict[str, List[int]]):
    models = list(success_data.keys())
    rounds = list(range(6))
    bar_width = 0.13
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))

    feedback_labels = ['No Feedback', '1 Feedback', '2 Feedbacks', '3 Feedbacks', '4 Feedbacks', '5 Feedbacks']
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in rounds:
        round_values = [(success_data[model][i]/1800 * 100) for model in models]
        bar_positions = x + (i - 2.5) * bar_width
        ax.bar(bar_positions, round_values, width=bar_width, label=feedback_labels[i], color=custom_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Cumulative Successful Runs')
    ax.yaxis.set_major_formatter(PercentFormatter()) 
    ax.set_title('Cumulative Successful Code Generations by Feedback Round in Percent')
    ax.legend(title='Feedback Rounds')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    results = categorize_errors()
    analyse_syntax_error(results)
    successes = analyse_success_by_round()
    visualize_success_by_round(successes)
    visualize_success_by_llm(successes)
    
