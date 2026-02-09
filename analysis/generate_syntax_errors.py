import json
import os
import glob
import sys
import re
import ast

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from abap_errors import ERRORS_BY_CATEGORY

def generate_syntax_errors():
    root_dir = ".."
    output_file = "../syntax_errors.json"
    
    if os.path.basename(os.getcwd()) == "analysis":
        root_dir = ".."
        data_dir = os.path.join(root_dir, "data")
        output_file = "../data/syntax_errors.json"
    else:
        root_dir = "."
        data_dir = os.path.join(root_dir, "data")
        output_file = "data/syntax_errors.json"

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    results = {}

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        if filename in [
            "package.json",
            "tsconfig.json",
            "results.json",
            "syntax_errors.json",
            "openai_batch_tracking.json",
            "anthropic_batch_tracking.json",
        ]:
            continue
        if filename.endswith("_tiers.json"):
            continue
        if filename.endswith("_retry_state.json"):
            continue
        if filename.endswith("_queue.json"):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {filename}: Invalid JSON")
            continue
            
        if not content or not isinstance(content, dict):
            continue

        first_key = next(iter(content))
        if not str(first_key).endswith(".txt"):
            continue
        
        model_name = filename.replace(".json", "")
        print(f"Processing {model_name}...")
        
        results[model_name] = {category: {error.regex: [] for error in errors} 
                               for category, errors in ERRORS_BY_CATEGORY.items()}
        results[model_name]["Uncategorized"] = {"Other": []}

        for prompt_file, repetitions in content.items():
            if not isinstance(repetitions, list):
                continue
            for conversation in repetitions:
                if not isinstance(conversation, list):
                    continue
                for message in conversation:
                    if not isinstance(message, dict):
                        continue
                    if message.get("role") == "user":
                        content_txt = message.get("content", "")
                        if "The syntax check failed with the following errors:" in content_txt:
                            try:
                                error_list_str = content_txt.split("The syntax check failed with the following errors:\n")[-1]
                                error_list = ast.literal_eval(error_list_str)
                                
                                for error_item in error_list:
                                    if isinstance(error_item, dict) and 'short_text' in error_item:
                                        short_text = error_item['short_text']
                                        
                                        matched = False
                                        for category, errors in ERRORS_BY_CATEGORY.items():
                                            for error_def in errors:
                                                if re.match(error_def.regex, short_text):
                                                    results[model_name][category][error_def.regex].append(short_text)
                                                    matched = True
                                                    break
                                            if matched:
                                                break
                                        
                                        if not matched:
                                            results[model_name]["Uncategorized"]["Other"].append(short_text)
                                            
                            except (SyntaxError, ValueError, IndexError) as e:
                                pass

    final_results = {}
    for model, categories in results.items():
        final_results[model] = {}
        for category, patterns in categories.items():
            non_empty_patterns = {p: v for p, v in patterns.items() if v}
            if non_empty_patterns:
                final_results[model][category] = non_empty_patterns

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    generate_syntax_errors()
