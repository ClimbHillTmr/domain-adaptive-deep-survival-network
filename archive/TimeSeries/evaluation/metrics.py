import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))

def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def mase(y_true, y_pred, seasonal_period=1):
    y_true = np.array(y_true)
    naive_forecast = y_true[:-seasonal_period]
    naive_shifted = y_true[seasonal_period:]
    q = np.mean(np.abs(naive_shifted - naive_forecast))
    return float(np.mean(np.abs(y_true - y_pred)) / (q + 1e-8))

def theils_u(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    u1 = np.sqrt(np.mean(((y_pred - y_true) ** 2))) / (np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2)) + 1e-8)
    u2 = np.sqrt(np.mean(((y_pred - y_true) ** 2))) / (np.sqrt(np.mean(((y_true[1:] - y_true[:-1]) ** 2))) + 1e-8)
    return float(u1), float(u2)