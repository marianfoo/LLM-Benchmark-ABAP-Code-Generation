import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from plot_style import COLORS, apply_plot_style
from matplotlib.ticker import PercentFormatter

apply_plot_style()

def get_failed_task_counts(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    target_models = df['Model'].unique()
    print(f"Models found: {target_models}")
    
    df = df[df['Model'].isin(target_models)]

    categories = [
        'String Handling',
        'List or Array Operation',
        'Mathematical Calculation',
        'Logical Condition',
        'ABAP Database Operation'
    ]
    
    failed_task_ids = {model: {} for model in target_models}
    category_total_counts = {}

    for model in target_models:
        model_df = df[df['Model'] == model]
        
        for category in categories:
            if category not in df.columns:
                failed_task_ids[model][category] = set()
                continue
                
            category_df = model_df[model_df[category].astype(str) == 'True']
            
            if len(category_df) == 0:
                failed_task_ids[model][category] = set()
                continue
            
            tasks = category_df['Prompt'].unique()
            if category not in category_total_counts:
                category_total_counts[category] = len(tasks)
                
            current_category_failed_ids = set()
            
            for task_id in tasks:
                task_attempts = category_df[category_df['Prompt'] == task_id]
                
                task_succeeded_once = False
                
                for _, row in task_attempts.iterrows():
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
                        task_succeeded_once = True
                        break
                
                if not task_succeeded_once:
                    current_category_failed_ids.add(task_id)
            
            failed_task_ids[model][category] = current_category_failed_ids

    return failed_task_ids, category_total_counts

def visualize_failed_tasks_intersection(failed_task_ids: Dict[str, Dict[str, set]], category_total_counts: Dict[str, int]):
    models = list(failed_task_ids.keys())
    first_model = models[0]
    categories = list(failed_task_ids[first_model].keys())
    
    intersection_percentages = {}
    
    for category in categories:
        sets = [failed_task_ids[model][category] for model in models]
        intersection = set.intersection(*sets)
        count = len(intersection)
        total = category_total_counts.get(category, 0)
        
        if total > 0:
            percentage = (count / total) * 100
        else:
            percentage = 0.0
            
        intersection_percentages[category] = percentage
        print(f"Category: {category}, Failed by ALL: {count}/{total} ({percentage:.2f}%)")

    x = np.arange(len(categories))
    values = [intersection_percentages[cat] for cat in categories]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(
        x, 
        values, 
        width=0.5, 
        label=f'Failed by {len(models)} Models',
        color=COLORS[3],
    )
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
        
    plt.xlabel("Task Category")
    plt.ylabel("Percentage of Mutually Failed Tasks")
    plt.title(f"Tasks Failed by ALL {len(models)} Models (All Attempts)")
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    
    plt.xticks(x, categories, rotation=15)
    plt.legend()
    plt.tight_layout()
    
    output_file = '../plots/failed_tasks_by_category.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../data/results.csv')
    failed_task_ids, category_total_counts = get_failed_task_counts(file_path)
    if failed_task_ids:
        visualize_failed_tasks_intersection(failed_task_ids, category_total_counts)
