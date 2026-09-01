from pathlib import Path
import json
import pandas as pd

def generate_table2_and_3(results_path: str, output_table2: str, output_table3: str):
    """
    Generate Table 2 (Survival Metrics) and Table 3 (Uno's C-statistic Comparison / Bootstrap P-values)
    Based on the evaluated results.
    """
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        return
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    # Generate Table 2: Survival Metrics
    table2_rows = []
    
    for model_name, metrics in results.get("sweeps", {}).items():
        if "metrics" not in metrics:
            continue
            
        c_index_val = metrics['metrics'].get('C-index', 0)
        c_lower = metrics['metrics'].get('C-index_CI_lower', 0)
        c_upper = metrics['metrics'].get('C-index_CI_upper', 0)
        
        if c_lower > 0 and c_upper > 0:
            c_str = f"{c_index_val:.3f} ({c_lower:.3f}-{c_upper:.3f})"
        else:
            c_str = f"{c_index_val:.3f}"
            
        row = {
            "Model": model_name,
            "C-index (95% CI)": c_str,
        }
        for horizon in (30, 60, 120):
            key = f"AUC_{horizon}m"
            if key in metrics["metrics"]:
                row[f"AUC ({horizon}m)"] = f"{metrics['metrics'][key]:.3f}"
        table2_rows.append(row)
        
    if table2_rows:
        df_table2 = pd.DataFrame(table2_rows)
        Path(output_table2).parent.mkdir(parents=True, exist_ok=True)
        df_table2.to_csv(output_table2.replace(".tex", ".csv"), index=False)
        with open(output_table2, 'w') as f:
            f.write(df_table2.to_latex(index=False, caption="Survival Prediction Performance", label="tab:performance"))
        print(f"Table 2 generated at {output_table2}")
        
    Path(output_table3).parent.mkdir(parents=True, exist_ok=True)
    note = (
        "Pairwise p-values were not generated. "
        "Use paired patient-level bootstrap replicates from real predictions before submission.\n"
    )
    Path(output_table3).write_text(note)
    Path(output_table3.replace(".tex", ".csv")).write_text("note\n" + note)
    print(f"Table 3 placeholder removed; wrote note to {output_table3}")

if __name__ == "__main__":
    generate_table2_and_3(
        "experiments/results/evaluation_results.json",
        "tables/table2_performance.tex",
        "tables/table3_pvalues.tex"
    )
