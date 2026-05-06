import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X_df = pd.DataFrame()
    X_df['透析年龄'] = df['透析年龄']
    X_df['性别'] = (df['性别'] == 'M').astype(float)
    X_df['透析龄'] = df['透析龄']
    X_df['超滤比'] = df['超滤量MAX'] / df['干体重'].replace(0, np.nan)
    X_df['容量超负荷比'] = (df['透前体重'] - df['干体重']) / df['干体重'].replace(0, np.nan)
    X_df['脉压差'] = df['透前收缩压'] - df['透前舒张压']
    X_df['平均动脉压'] = (df['透前收缩压'] + 2 * df['透前舒张压']) / 3
    
    X = X_df.fillna(X_df.median())
    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    X = np.clip(X, lower, upper)
    
    e = df['透中低血压_计算'].fillna(0).values.astype(int)
    t = df['et_min'].fillna(0).values.astype(float)
    return X, e, t

X_s, e_s, t_s = load_data('data_preprocessing/data/深医_final_data.csv')
X_t, e_t, t_t = load_data('data_preprocessing/data/福鼎_final_data.csv')

# Train LightGBM Survival
train_data = lgb.Dataset(X_s, label=t_s)
train_data.set_weight(e_s) # approximate survival with weights for custom obj or just use cox

params = {
    'objective': 'cox',
    'metric': 'cox',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}
# format for cox: label is time, and we need to pass event. LightGBM cox expects label = time if event=1, else -time
y_train_cox = np.where(e_s == 1, t_s, -t_s)
train_data = lgb.Dataset(X_s, label=y_train_cox)

gbm = lgb.train(params, train_data, num_boost_round=100)

pred_t = gbm.predict(X_t)
c_index = concordance_index(t_t, -pred_t, e_t)
print(f"LightGBM Target C-index: {c_index:.4f}")

# Train on Source, Test on Source
pred_s = gbm.predict(X_s)
c_index_s = concordance_index(t_s, -pred_s, e_s)
print(f"LightGBM Source C-index: {c_index_s:.4f}")

