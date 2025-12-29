import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

df = pd.read_csv('westenberger_input.csv', delimiter=';')

print("Data loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

def get_survival_time(row):
    feedback_cols = ['Feedback_Round_0', 'Feedback_Round_1', 'Feedback_Round_2', 
                     'Feedback_Round_3', 'Feedback_Round_4', 'Feedback_Round_5']
    
    success = row['Success']
    if pd.notna(success) and str(success).upper() in ['TRUE', '1']:
        return 6, 0
    
    last_round = 0
    for i, col in enumerate(feedback_cols):
        feedback = str(row[col]).strip().lower()
        if feedback and feedback not in ['', 'nan']:
            last_round = i + 1
    
    if last_round > 0:
        return last_round, 1
    else:
        return 1, 1
survival_data = []
for idx, row in df.iterrows():
    time, event = get_survival_time(row)
    survival_data.append({
        'Model': row['Model'],
        'Duration': time,
        'Event': event
    })

survival_df = pd.DataFrame(survival_data)

print("\n" + "="*80)
print("SURVIVAL DATA SUMMARY")
print("="*80)
print(survival_df.groupby('Model').agg({
    'Duration': ['count', 'mean', 'median'],
    'Event': 'sum'
}))

models = survival_df['Model'].unique()
print(f"\n\nFound {len(models)} unique models:")
for model in models:
    print(f"  - {model}")
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

km_results = {}
for i, model in enumerate(models):
    model_data = survival_df[survival_df['Model'] == model]
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=model_data['Duration'], 
            event_observed=model_data['Event'],
            label=model)
    
    km_results[model] = kmf
    
    kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2)
    
    print(f"\n" + "="*80)
    print(f"MODEL: {model}")
    print("="*80)
    print(f"Total observations: {len(model_data)}")
    print(f"Events (failures): {model_data['Event'].sum()}")
    print(f"Censored (success): {len(model_data) - model_data['Event'].sum()}")
    print(f"\nMedian survival time: {kmf.median_survival_time_}")
    
    print("\nSurvival Table:")
    print("-" * 80)
    survival_table = kmf.survival_function_
    confidence_interval = kmf.confidence_interval_survival_function_
    
    event_table = kmf.event_table
    cumulative_variance = []
    cum_var = 0
    
    for idx in event_table.index:
        at_risk = event_table.loc[idx, 'at_risk']
        observed = event_table.loc[idx, 'observed']
        if at_risk > 0 and observed > 0:
            cum_var += observed / (at_risk * (at_risk - observed)) if at_risk > observed else 0
        cumulative_variance.append(cum_var)
    
    variance_series = pd.Series(cumulative_variance, index=event_table.index)
    
    result_table = pd.DataFrame({
        'Survival_Probability': survival_table.iloc[:, 0],
        'Variance': variance_series,
        'Std_Error': np.sqrt(variance_series * survival_table.iloc[:, 0]**2),
        'CI_Lower': confidence_interval.iloc[:, 0],
        'CI_Upper': confidence_interval.iloc[:, 1]
    })
    
    print(result_table.to_string())
    
    print("\n\nEvent Table:")
    print("-" * 80)
    event_table = kmf.event_table
    print(event_table.to_string())

ax.set_xlabel('Feedback Round', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curves by Model\n(Probability of NOT failing by round N)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('kaplan_meier_survival_curves.png', dpi=300, bbox_inches='tight')
print("\n\nSaved plot to: kaplan_meier_survival_curves.png")

if len(models) > 1:
    print("\n" + "="*80)
    print("LOG-RANK TEST (comparing all models)")
    print("="*80)
    results = multivariate_logrank_test(
        survival_df['Duration'], 
        survival_df['Model'], 
        survival_df['Event']
    )
    print(results)
    print(f"\np-value: {results.p_value:.6f}")
    if results.p_value < 0.05:
        print("Result: Significant difference between models (p < 0.05)")
    else:
        print("Result: No significant difference between models (p >= 0.05)")

plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)