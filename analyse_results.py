import json

MODELS_TO_RUN = [
    "llama-3.3-70b-instruct",
    "qwen2.5-coder-32b-instruct",
    "codestral-22b",
]

RESULTS_FILE = "analyse_results/results.json"


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


if __name__ == "__main__":
    results = categorize_errors()
