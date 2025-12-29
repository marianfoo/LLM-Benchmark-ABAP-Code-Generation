import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plot_style import COLORS, apply_plot_style

apply_plot_style()

def get_success_rates_by_model(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    df['Success_Bool'] = df['Success'].astype(str) == 'True'
    
    if 'Prompt' not in df.columns:
        print("Error: 'Prompt' column not found.")
        return None
        
    target_models = ['gpt-5-2025-08-07', 'claude-sonnet-4-20250514']
    model_data = {}
    
    for model in target_models:
        model_df = df[df['Model'] == model]
        if model_df.empty:
            print(f"Warning: No data found for model {model}")
            continue
            
        task_success = model_df.groupby('Prompt')['Success_Bool'].mean()
        model_data[model] = task_success
    
    return model_data

def plot_comparative_histogram(model_data):
    if not model_data:
        print("No model data to plot.")
        return

    bins = [-0.01, 0.1001, 0.2001, 0.3001, 0.4001, 0.5001, 0.6001, 0.7001, 0.8001, 0.9001, 1.01]
    
    plot_data = {}
    for model, series in model_data.items():
        counts, _ = np.histogram(series, bins=bins)
        total = len(series)
        if total > 0:
            percents = (counts / total) * 100
        else:
            percents = np.zeros(len(counts))
        plot_data[model] = percents
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_width = 0.1
    bar_width = bin_width * 0.4
    
    bin_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    
    labels = ["0-10%", "11-20%", "21-30%", "31-40%", "41-50%", "51-60%", "61-70%", "71-80%", "81-90%", "91-100%"]
    
    models = list(model_data.keys())
    model_colors = [COLORS[2], COLORS[5]] 
    
    for i, model in enumerate(models):
        if i >= len(model_colors):
            break 
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            bin_centers + offset, 
            plot_data[model], 
            width=bar_width, 
            label=model, 
            color=model_colors[i],
            edgecolor='white'
        )
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    height + 0.5, 
                    f'{int(round(height))}%', 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    color='black'
                )
    
    ax.set_xlabel('Success Rate Interval')
    ax.set_ylabel('Percent of Tasks')
    ax.set_title('Distribution of Task Success Rates: GPT-5 vs Claude')
    
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend()
    
    plt.tight_layout()
    output_file = 'plots/success_distribution.png'
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    file_path = 'data/results.csv'
    data = get_success_rates_by_model(file_path)
    if data:
        plot_comparative_histogram(data)
