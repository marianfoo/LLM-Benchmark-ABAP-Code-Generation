import json
import re
from typing import Dict, List

MODELS_TO_RUN = [
    "llama-3.3-70b-instruct",
    "qwen2.5-coder-32b-instruct",
    "codestral-22b",
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
        "other": [],
    }

    for model_name in MODELS_TO_RUN:
        filename = f"{model_name}.json"

        with open(filename, "r") as file:
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

    with open(RESULTS_FILE, "w") as file:
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

    with open(SYNTAX_ERROR_FILE, "w") as file:
        file.write(json.dumps(syntax_errors, indent=4, ensure_ascii=False))

    return runs

if __name__ == "__main__":
    results = categorize_errors()
    analyse_syntax_error(results)
