import numpy as np
import pandas as pd
from typing import Tuple

def build_survival_labels(df: pd.DataFrame, time_col: str, event_rule_col: str) -> Tuple[np.ndarray, np.ndarray]:
    # time_col: 序列时间列表或时间步索引；event_rule_col: 是否低血压事件计算列（按规则）
    durations = []
    events = []
    for _, row in df.iterrows():
        times = row[time_col]
        event_flag = row.get(event_rule_col, 0)
        if isinstance(times, list):
            duration = len(times) if event_flag == 0 else (np.argmax(np.array(times) == times[0]) + 1)
        else:
            duration = row.get('duration', 0)
        durations.append(duration)
        events.append(1 if event_flag == 1 else 0)
    return np.array(durations), np.array(events)