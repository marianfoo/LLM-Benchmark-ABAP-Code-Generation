import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Dict, List
from plot_style import COLORS, apply_plot_style

apply_plot_style()

def get_category_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    models = df['Model'].unique()


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

    return data, counts

def visualize_prompt_classification_success(data: Dict[str, Dict[str, float]], counts: Dict[str, Dict[str, int]] = None):
    models = list(data.keys())
    first_model = models[0]
    categories = list(data[first_model].keys())
    
    num_models = len(models)
    num_categories = len(categories)

    bar_width = 0.13
    x = np.arange(num_models)
    
    plt.figure(figsize=(15, 8))
    
    for i, category in enumerate(categories):
        values = [data[model][category] for model in models]
        
        errors = []
        for model in models:
            if counts and category in counts[model]:
                n = counts[model][category]
                p = data[model][category] / 100.0
                if n > 0:
                    se = np.sqrt(p * (1 - p) / n)
                    margin = 1.96 * se * 100
                else:
                    margin = 0
                errors.append(margin)
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
    
    output_file = '../plots/success_by_model_by_task_category_in_percent.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    # plt.show()

if __name__ == "__main__":
    data, counts = get_category_data_from_csv('../results.csv')
    if data:
        visualize_prompt_classification_success(data, counts)