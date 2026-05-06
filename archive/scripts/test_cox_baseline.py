import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X_df = pd.DataFrame()
    X_df['age'] = df['透析年龄']
    X_df['gender'] = (df['性别'] == 'M').astype(float)
    X_df['dialysis_age'] = df['透析龄']
    X_df['ufr_ratio'] = df['超滤量MAX'] / df['干体重'].replace(0, np.nan)
    X_df['fluid_overload'] = (df['透前体重'] - df['干体重']) / df['干体重'].replace(0, np.nan)
    X_df['pulse_pressure'] = df['透前收缩压'] - df['透前舒张压']
    X_df['map'] = (df['透前收缩压'] + 2 * df['透前舒张压']) / 3
    
    X_df = X_df.fillna(X_df.median())
    
    # Winsorize
    lower = X_df.quantile(0.01)
    upper = X_df.quantile(0.99)
    X_df = X_df.clip(lower=lower, upper=upper, axis=1)
    
    X_df['event'] = df['透中低血压_计算'].fillna(0).astype(int)
    X_df['time'] = df['et_min'].fillna(0).astype(float)
    
    # filter invalid times
    X_df = X_df[X_df['time'] > 0]
    return X_df

df_s = load_data('data_preprocessing/data/深医_final_data.csv')
df_t = load_data('data_preprocessing/data/福鼎_final_data.csv')

# Subsample for speed
df_s = df_s.sample(20000, random_state=42)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_s, duration_col='time', event_col='event')

# Target
pred_t = cph.predict_partial_hazard(df_t)
c_index_t = concordance_index(df_t['time'], -pred_t, df_t['event'])
print(f"CoxPH Target C-index: {c_index_t:.4f}")

# Source
pred_s = cph.predict_partial_hazard(df_s)
c_index_s = concordance_index(df_s['time'], -pred_s, df_s['event'])
print(f"CoxPH Source C-index: {c_index_s:.4f}")

