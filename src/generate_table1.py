import pandas as pd
import numpy as np
from pathlib import Path
import os
import yaml

def load_config(config_path="conf/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_table1():
    """
    生成基线特征表 Table 1 (Source vs Target)
    """
    config = load_config()
    source_path = os.path.join(config["paths"]["data_dir"], config["data"]["source_file"])
    target_path = os.path.join(config["paths"]["data_dir"], config["data"]["target_file"])
    output_csv = "tables/table1_baseline.csv"
    output_tex = "tables/table1_baseline.tex"
    
    if not os.path.exists(source_path):
        print(f"数据文件未找到，请检查路径: {source_path}")
        return
    if not os.path.exists(target_path):
        print(f"数据文件未找到，请检查路径: {target_path}")
        return
        
    df_source = pd.read_csv(source_path)
    df_target = pd.read_csv(target_path)

    df_source['Center'] = 'Shenyi (Source)'
    df_target['Center'] = 'Fuding (Target)'
    
    df = pd.concat([df_source, df_target], axis=0, ignore_index=True)
    
    # 选取代表性的变量
    vars_to_summarize = [
        "透析年龄", "性别", "透前收缩压", "透前舒张压", "脉压差", "干体重", 
        "透析液温度_mean", "超滤率_体重归一化", "超滤率高危Flag", "实际透析时长",
        "透中低血压_计算"
    ]
    
    rows = []
    
    for var in vars_to_summarize:
        if var not in df.columns:
            continue
            
        row = {"Variable": var}
        
        # 判断是否是分类变量
        is_categorical = set(df[var].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        
        # 整体
        if is_categorical:
            count = df[var].sum()
            pct = count / len(df[var].dropna()) * 100
            row["Overall (N={})".format(len(df))] = f"{int(count)} ({pct:.1f}%)"
        else:
            med = df[var].median()
            q25 = df[var].quantile(0.25)
            q75 = df[var].quantile(0.75)
            row["Overall (N={})".format(len(df))] = f"{med:.1f} ({q25:.1f}, {q75:.1f})"
            
        # 分组计算
        for center in ['Shenyi (Source)', 'Fuding (Target)']:
            df_sub = df[df['Center'] == center]
            col_name = f"{center} (N={len(df_sub)})"
            
            if is_categorical:
                count = df_sub[var].sum()
                pct = count / len(df_sub[var].dropna()) * 100
                row[col_name] = f"{int(count)} ({pct:.1f}%)"
            else:
                med = df_sub[var].median()
                q25 = df_sub[var].quantile(0.25)
                q75 = df_sub[var].quantile(0.75)
                row[col_name] = f"{med:.1f} ({q25:.1f}, {q75:.1f})"
                
        # 缺失率
        missing_count = df[var].isna().sum()
        row["Missing Count"] = missing_count
        row["Missing %"] = f"{missing_count / len(df) * 100:.1f}%"
        
        rows.append(row)
        
    table1 = pd.DataFrame(rows)
    
    # 导出
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table1.to_csv(output_csv, index=False)
    
    # 转换为 LaTeX
    with open(output_tex, 'w') as f:
        f.write(table1.to_latex(index=False, caption="Baseline Characteristics", label="tab:baseline"))
        
    print(f"Table 1 generated at {output_csv} and {output_tex}")

if __name__ == "__main__":
    generate_table1()
