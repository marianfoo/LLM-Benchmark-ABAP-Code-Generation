import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np
import matplotlib.pyplot as plt
from plot_style import COLORS, LINE_STYLES, apply_plot_style

apply_plot_style()

def calculate_variance(kmf):
    try:
        S_t = kmf.survival_function_.iloc[:, 0]
        cumulative_sq = kmf._cumulative_sq_
        if hasattr(cumulative_sq, 'iloc'):
             cumulative_sq = cumulative_sq.iloc[:, 0] if len(cumulative_sq.shape) > 1 else cumulative_sq
        variance = (S_t ** 2) * cumulative_sq
        return variance
    except Exception as e:
        print(f"Could not calculate variance: {e}")
        return None

def run_analysis():
    file_path = '../data/results.csv'
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    subset_col = 'ABAP Database Operation'
    if subset_col not in df.columns:
         print(f"Column {subset_col} not found!")
         return
         
    print(f"Filtering for {subset_col} == 'True'")
    initial_len = len(df)
    df = df[df[subset_col].astype(str) == 'True']
    print(f"Filtered DataFrame: {len(df)} rows (was {initial_len})")

    required_cols = ['Model', 'Feedback_Round_0', 'Feedback_Round_1', 'Feedback_Round_2', 'Feedback_Round_3', 'Feedback_Round_4', 'Feedback_Round_5', 'Success']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing one of the columns: {required_cols}")
        return

    plt.figure(figsize=(12, 8))
    
    models = df['Model'].unique()

    print(f"Found {len(models)} models: {models}\n")

    for idx, model in enumerate(models):
        print(f"==================================================")
        print(f"Processing Model: {model}")
        print(f"==================================================")
        
        model_df = df[df['Model'] == model].copy()
        
        durations = []
        events = []

        for index, row in model_df.iterrows():
            success_event = False
            
            if str(row['Success']) == 'True':
                 success_event = True
            
            rounds = [row[f'Feedback_Round_{i}'] for i in range(6)]
            
            duration_index = 0
            found_success_string = False
            
            for i, r in enumerate(rounds):
                if pd.isna(r) or r == '':
                    break
                
                duration_index = i
                
                if 'unit tests were successful' in str(r):
                    found_success_string = True
                    break
            
            if found_success_string:
                success_event = True
                
            final_duration = duration_index
            if not success_event:
                 final_duration = 5         
            durations.append(final_duration)
            events.append(1 if success_event else 0)

        km_df = pd.DataFrame({
            'duration': durations,
            'event': events
        })

        print(f"Samples: {len(km_df)} | Events: {km_df['event'].sum()}")

        if len(km_df) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(km_df['duration'], km_df['event'], label=model)

            print("\n--- Survival Table (Head) ---")
            print(kmf.event_table.head())

            print("\n--- Event Probabilities (Survival Function) ---")
            print(kmf.survival_function_)
            
            variance = calculate_variance(kmf)
            if variance is not None:
                print("\n--- Variance (Greenwood's) ---")
                print(variance)
            
            color = COLORS[idx % len(COLORS)]
            linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
            kmf.plot_survival_function(color=color, linestyle=linestyle)
        else:
             print("No samples for this model in subset.")
        print("\n")

    plt.title('Kaplan-Meier Survival Curves (Database_Operations_ABAP == True)')
    plt.xlabel('Feedback Round Index (0-5)')
    plt.ylabel('Survival Probability')
    plt.ylim(0, 1.05)
    plt.xlim(0, 5)
    plt.xticks(range(6))
    plt.tight_layout()
    
    output_file = '../plots/km_survival_curve_abap.png'
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")

if __name__ == "__main__":
    run_analysis()
