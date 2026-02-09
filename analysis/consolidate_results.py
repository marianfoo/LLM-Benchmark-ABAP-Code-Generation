import json
import os
import glob
import pandas as pd

def consolidate_results():
    root_dir = ".."
    data_dir = os.path.join(root_dir, "data")

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    data = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        if filename in [
            "package.json",
            "tsconfig.json",
            "results.json",
            "openai_batch_tracking.json",
            "anthropic_batch_tracking.json",
            "syntax_errors.json",
        ]:
            continue
        if filename.endswith("_tiers.json"):
            continue
        if filename.endswith("_retry_state.json"):
            continue
        if filename.endswith("_queue.json"):
            continue
            
        model_name = filename.replace(".json", "")
        
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

        print(f"Processing {model_name}...")

        for prompt_file, repetitions in content.items():
            if not str(prompt_file).endswith(".txt"):
                continue
            if not isinstance(repetitions, list):
                continue
            prompt_id = prompt_file.replace(".txt", "")
            
            for rep_idx, conversation in enumerate(repetitions):
                if not isinstance(conversation, list):
                    continue
                row = {
                    "Model": model_name,
                    "Prompt": prompt_id,
                    "Repetition": rep_idx,
                    "Success": "False",
                    "Feedback_Round_0": None,
                    "Feedback_Round_1": None,
                    "Feedback_Round_2": None,
                    "Feedback_Round_3": None,
                    "Feedback_Round_4": None,
                    "Feedback_Round_5": None
                }
                
                feedback_round = 0
                
                for message in conversation:
                    if not isinstance(message, dict):
                        continue
                    if message.get("role") == "user":
                        content_txt = str(message.get("content", ""))
                        
                        is_feedback = False
                        feedback_status = None
                        
                        if "The syntax check failed" in content_txt:
                            feedback_status = "syntax check failed"
                            is_feedback = True
                        elif "The unit tests were successful" in content_txt:
                            feedback_status = "unit tests were successful"
                            is_feedback = True
                            row["Success"] = "True"
                        elif "The unit test failed" in content_txt:
                            feedback_status = "unit test failed"
                            is_feedback = True
                        elif "The class could not be created" in content_txt:
                             feedback_status = "class could not be created"
                             is_feedback = True
                        elif content_txt.startswith("[INFRA]"):
                            # Transient/infrastructure ADT error (retriable)
                            feedback_status = "infra error"
                            is_feedback = True
                    
                        if is_feedback:
                            if feedback_round <= 5:
                                row[f"Feedback_Round_{feedback_round}"] = feedback_status
                            feedback_round += 1
                data.append(row)
    df = pd.DataFrame(data)
    
    classification_path = os.path.join(data_dir, "prompt_classification.csv")
    print(f"Checking for classification at: {os.path.abspath(classification_path)}")
    if os.path.exists(classification_path):
        print(f"Loading classification from {classification_path}")
        try:
            class_df = pd.read_csv(classification_path, sep=';')
            if 'HumanEval/Nr' in class_df.columns:
                class_df['Prompt_ID'] = class_df['HumanEval/Nr'].astype(str)
            elif 'Nr' in class_df.columns:
                 class_df['Prompt_ID'] = class_df['Nr'].astype(str)
            
            rename_map = {
                'List & Array Operations': 'List or Array Operation',
                'Mathematical Calculations': 'Mathematical Calculation',
                'Logical Checks': 'Logical Condition',
                'Database Operations (ABAP)': 'ABAP Database Operation'
            }
            class_df = class_df.rename(columns=rename_map)
            
            class_df = class_df.replace('X', 'True')

            cols_to_keep = ['Prompt_ID', 'String Handling', 'List or Array Operation', 'Mathematical Calculation', 'Logical Condition', 'ABAP Database Operation']
            cols_to_keep = [c for c in cols_to_keep if c in class_df.columns]
            
            class_df = class_df[cols_to_keep]
            
            df = df.merge(class_df, left_on='Prompt', right_on='Prompt_ID', how='left')
            df = df.drop(columns=['Prompt_ID'])
            print("Merged prompt classification data.")
            
        except Exception as e:
            print(f"Failed to load/merge classification data: {e}")
    else:
        print(f"Warning: {classification_path} not found. Skipping classification merge.")

    try:
        df['Prompt_Int'] = df['Prompt'].astype(int)
        df = df.sort_values(by=['Model', 'Prompt_Int', 'Repetition'])
        df = df.drop(columns=['Prompt_Int'])
    except:
        df = df.sort_values(by=['Model', 'Prompt', 'Repetition'])

    output_path = os.path.join(data_dir, "results.csv")
    df.to_csv(output_path, sep=';', index=False)
    print(f"Successfully created {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    consolidate_results()
