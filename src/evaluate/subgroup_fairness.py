import pandas as pd
import numpy as np
import os
import json
import logging

from src.evaluate.metrics import concordance_index

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_subgroup_metrics(df, risk_scores, duration_col='et_min', event_col='events', subgroup_col=None):
    results = {}
    unique_groups = df[subgroup_col].dropna().unique()
    
    for group in unique_groups:
        mask = df[subgroup_col] == group
        group_df = df[mask]
        group_risks = risk_scores[mask]
        
        # Need at least some events and non-events to calculate C-index
        if len(group_df) > 10 and group_df[event_col].sum() > 0 and len(group_df) - group_df[event_col].sum() > 0:
            events = group_df[event_col].astype(bool).values
            times = group_df[duration_col].values
            
            try:
                c_index = concordance_index(times, group_risks, events)
                
                # Calculate simple calibration/prevalence
                event_rate = group_df[event_col].mean()
                mean_risk = np.mean(group_risks)
                
                results[str(group)] = {
                    'N': len(group_df),
                    'Events': int(group_df[event_col].sum()),
                    'Event_Rate': float(event_rate),
                    'C_Index': float(c_index)
                }
            except Exception as e:
                logging.warning(f"Could not calculate metrics for group {group}: {e}")
                
    return results

def run_fairness_analysis():
    # 1. Load data
    prediction_file = "experiments/results/real_test_predictions.csv"
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(
            "Missing real held-out predictions. Run prediction export first; "
            "simulated subgroup metrics are not allowed for submission."
        )
        
    logging.info(f"Loading real predictions from {prediction_file}")
    df = pd.read_csv(prediction_file)
    
    duration_col = 'et_min'
    event_col = 'events'
    risk_col = "risk_score"
    if risk_col not in df.columns:
        raise ValueError(f"Prediction file must contain '{risk_col}'.")
    risk_scores = df[risk_col].values
    
    # 3. Define subgroups
    # Let's create age groups if '透析年龄' exists
    if '透析年龄' in df.columns:
        df['Age_Group'] = pd.cut(df['透析年龄'], bins=[0, 65, 100], labels=['<65', '>=65'])
    else:
        logging.warning("Column '透析年龄' not found. Cannot perform age fairness analysis.")
        
    # Gender if '性别' exists
    if '性别' in df.columns:
        df['Gender_Str'] = df['性别'].map({0: 'Female', 1: 'Male', 2: 'Unknown'})
        # Fill na just in case
        df['Gender_Str'] = df['Gender_Str'].fillna(df['性别'].astype(str))
    
    # Diabetes if '糖尿病' exists (assuming it might be in '传染病' or '高血压诊断' or similar, let's check)
    comorbidity_col = None
    if '糖尿病' in df.columns:
        comorbidity_col = '糖尿病'
    elif '高血压诊断' in df.columns:
        comorbidity_col = '高血压诊断'
        df['Comorbidity'] = df['高血压诊断'].map({0: 'No_HTN', 1: 'Has_HTN'})
        
    
    fairness_results = {}
    
    # 4. Calculate metrics for each subgroup
    logging.info("Calculating subgroup metrics...")
    
    if 'Age_Group' in df.columns:
        fairness_results['Age'] = calculate_subgroup_metrics(df, risk_scores, duration_col, event_col, 'Age_Group')
        
    if 'Gender_Str' in df.columns:
        fairness_results['Gender'] = calculate_subgroup_metrics(df, risk_scores, duration_col, event_col, 'Gender_Str')
        
    if 'Comorbidity' in df.columns:
        fairness_results['Comorbidity (HTN)'] = calculate_subgroup_metrics(df, risk_scores, duration_col, event_col, 'Comorbidity')
        
    # 5. Format and save results
    output_dir = "tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON for easy reading
    json_path = os.path.join(output_dir, "fairness_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(fairness_results, f, indent=4)
        
    # Generate a LaTeX table
    latex_path = os.path.join(output_dir, "table4_fairness.tex")
    generate_latex_table(fairness_results, latex_path)
    
    # Generate CSV
    csv_path = os.path.join(output_dir, "table4_fairness.csv")
    generate_csv_table(fairness_results, csv_path)
    
    logging.info(f"Fairness analysis complete. Results saved to {output_dir}")

def generate_latex_table(results, output_path):
    latex_str = """\\begin{table}[h]
\\centering
\\caption{Subgroup Fairness Analysis (Target Domain: Fuding Hospital)}
\\label{tab:fairness}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Subgroup} & \\textbf{N} & \\textbf{Events} & \\textbf{Event Rate (\\%)} & \\textbf{C-Index (95\\% CI)} \\\\
\\midrule
"""
    for category, groups in results.items():
        latex_str += f"\\multicolumn{{5}}{{-l}}{{\\textbf{{{category}}}}} \\\\\n"
        for group_name, metrics in groups.items():
            # ponytail: rough Wald-style display only; replace with patient bootstrap for submission.
            ci_margin = 1.96 * (0.5 / np.sqrt(metrics['Events'] + 1))
            c_index = metrics['C_Index']
            ci_lower = max(0.5, c_index - ci_margin)
            ci_upper = min(1.0, c_index + ci_margin)
            
            latex_str += f"\\hspace{{1em}} {group_name} & {metrics['N']} & {metrics['Events']} & {metrics['Event_Rate']*100:.1f} & {c_index:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) \\\\\n"
    
    latex_str += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(output_path, 'w') as f:
        f.write(latex_str)

def generate_csv_table(results, output_path):
    rows = []
    for category, groups in results.items():
        for group_name, metrics in groups.items():
            # ponytail: rough Wald-style display only; replace with patient bootstrap for submission.
            ci_margin = 1.96 * (0.5 / np.sqrt(metrics['Events'] + 1))
            c_index = metrics['C_Index']
            ci_lower = max(0.5, c_index - ci_margin)
            ci_upper = min(1.0, c_index + ci_margin)
            
            rows.append({
                'Category': category,
                'Subgroup': group_name,
                'N': metrics['N'],
                'Events': metrics['Events'],
                'Event_Rate': f"{metrics['Event_Rate']*100:.1f}%",
                'C_Index': f"{c_index:.3f}",
                '95% CI': f"{ci_lower:.3f}-{ci_upper:.3f}"
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_fairness_analysis()
