import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any
from src.labels.stage_builder import build_stage_label
from src.data.sessionize import normalize_times_to_minutes, split_sessions, filter_by_missing

def _parse_list(x):
    if isinstance(x, list):
        return [float(v) for v in x if v is not None and str(v) != "NA"]
    if isinstance(x, str):
        import ast
        s = x.strip()
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [float(t) for t in v if t is not None and str(t) != "NA"]
        except Exception:
            return []
    return []

def _parse_times(x):
    lst = _parse_list(x)
    return normalize_times_to_minutes(lst)

def _event_time_minutes(row):
    seq = _parse_list(row.get("透析中收缩压"))
    times = _parse_times(row.get("透中数据记录时间节点"))
    if not seq:
        return None
    first = seq[0]
    for i, v in enumerate(seq):
        if (first - v >= 30) or (v <= 90):
            if times and i < len(times) and times[i] is not None:
                return float(times[i])
            return float(i)
    return None

def _event_time_from_final(row, columns=None):
    event_flag_col = columns.get('event_flag_col', '透中低血压_计算') if columns else '透中低血压_计算'
    drop_time_col = columns.get('drop_time_col', '降幅时间点') if columns else '降幅时间点'
    start_time_col = columns.get('start_time_col', '透析开始时间') if columns else '透析开始时间'
    flag = row.get(event_flag_col)
    try:
        flag_val = int(flag) if flag is not None else 0
    except Exception:
        flag_val = 0
    if flag_val == 0:
        return None
    dt_drop = pd.to_datetime(row.get(drop_time_col), errors='coerce')
    dt_start = pd.to_datetime(row.get(start_time_col), errors='coerce')
    if pd.isna(dt_drop) or pd.isna(dt_start):
        return None
    return float((dt_drop - dt_start).total_seconds() / 60.0)

def _get_float(row, col):
    v = row.get(col)
    try:
        return float(v)
    except Exception:
        return 0.0

def _extract_optional_static(row, names):
    vals = []
    for name in names:
        if name in row and row[name] is not None and str(row[name]) != "":
            vals.append(_get_float(row, name))
    return vals

def _seq_features(seq):
    if not seq:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    arr = np.array(seq, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    slope = float((arr[-1] - arr[0]) / max(len(arr) - 1, 1))
    return [mean, std, mn, mx, slope]

def _rolling_std_mean(seq, w=5):
    arr = np.array(seq or [], dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.size < w:
        return float(np.std(arr))
    s = []
    for i in range(0, arr.size - w + 1):
        s.append(float(np.std(arr[i:i+w])))
    return float(np.mean(s))

def _short_window_slope(seq, w=5):
    arr = np.array(seq or [], dtype=float)
    if arr.size < 2:
        return 0.0
    a = arr[:w] if arr.size >= w else arr
    b = arr[-w:] if arr.size >= w else arr
    return float((np.mean(b) - np.mean(a)) / max(w, 1))

def _min_pos_ratio(seq):
    arr = np.array(seq or [], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.argmin(arr) / max(arr.size - 1, 1))

def _mean_abs_diff(seq):
    arr = np.array(seq or [], dtype=float)
    if arr.size < 2:
        return 0.0
    d = np.diff(arr)
    return float(np.mean(np.abs(d)))

def _nearest_index(times, t):
    if times is None or t is None:
        return None
    idx = None
    best = None
    for i, v in enumerate(times):
        if v is None:
            continue
        dv = abs(float(v) - float(t))
        if best is None or dv < best:
            best = dv
            idx = i
    return idx

def _neighbor_features(seq, times, et, w=5):
    arr = np.array(seq or [], dtype=float)
    if arr.size == 0 or times is None or et is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    k = _nearest_index(times, et)
    if k is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = max(0, k - w)
    b = min(arr.size, k + w + 1)
    pre = arr[a:k] if k > a else arr[a:a+1]
    post = arr[k:b] if b > k else arr[b-1:b]
    pre_mean = float(np.mean(pre))
    pre_std = float(np.std(pre))
    pre_slope = float((pre[-1] - pre[0]) / max(len(pre) - 1, 1))
    post_mean = float(np.mean(post))
    post_std = float(np.std(post))
    post_slope = float((post[-1] - post[0]) / max(len(post) - 1, 1))
    delta_mean = float(post_mean - pre_mean)
    delta_std = float(post_std - pre_std)
    return [pre_mean, pre_std, pre_slope, post_mean, post_std, post_slope, delta_mean, delta_std]

def load_and_build_features(csv_path: str, boundaries=(30, 90), columns: Dict[str, Any] = None,
                            missing_threshold: float = 0.4, gap_threshold: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    base_features_core = [
        "传染病", "抗凝剂类型", "干体重", "瘘管类型", "瘘管位置", "瘘管使用时间",
        "首次透析年龄", "透析方式", "透析龄", "透析龄_天数", "透析年龄", "性别"
    ]
    base_features_history = [
        "历史平均超滤率_mean", "历史平均动脉压_mean", "历史平均干体重",
        "历史平均降幅时间点比值区间", "历史平均降幅时间点差值区间",
        "历史平均静脉压_mean", "历史平均跨膜压_mean", "历史平均实际透析时长",
        "历史平均透析液钙浓度", "历史平均透析液电导率", "历史平均透析液温度_mean",
        "历史平均透析中收缩压_mean", "历史平均透析中舒张压_mean", "历史平均透析中脉搏_mean",
        "历史平均透前收缩压", "历史平均透前舒张压", "历史平均透前呼吸频率",
        "历史平均透前体重", "历史平均透中高血压_计算", "历史平均血流速_mean",
        "历史平均涨幅时间点比值区间", "历史平均涨幅时间点差值区间",
        "history_HBP_rate", "history_LBP_times_0_rate", "history_LBP_times_1_rate",
        "history_LBP_times_2_rate", "history_LBP_times_3_rate", "history_LBP_times_4_rate",
        "history_HBP", "history_LBP_times_0", "history_LBP_times_1",
        "history_LBP_times_2", "history_LBP_times_3", "history_LBP_times_4"
    ]
    base_features_current = [
        "透析液钙浓度", "透析液电导率", "透前呼吸频率", "透前收缩压", "透前舒张压",
        "透前体重", "透前体重-干体重"
    ]
    for _, row in df.iterrows():
        sbp_col = columns.get('sbp_col', '透析中收缩压') if columns else '透析中收缩压'
        map_col = columns.get('map_col', '动脉压') if columns else '动脉压'
        hr_col = columns.get('hr_col', '心率') if columns else '心率'
        uf_col = columns.get('uf_col', '超滤率') if columns else '超滤率'
        time_col = columns.get('time_col', '透中数据记录时间节点') if columns else '透中数据记录时间节点'
        sbp = _parse_list(row.get(sbp_col))
        map_ = _parse_list(row.get(map_col))
        hr = _parse_list(row.get(hr_col))
        uf = _parse_list(row.get(uf_col))
        times = _parse_times(row.get(time_col))
        session_times = split_sessions(times)
        seq_dict = {"sbp": sbp, "map": map_, "hr": hr, "uf": uf}
        if not filter_by_missing(seq_dict, missing_threshold=missing_threshold):
            continue
        et = _event_time_from_final(row, columns=columns)
        if et is None:
            et = _event_time_minutes(row)
        feats = []
        feats += _seq_features(sbp)
        feats += [_rolling_std_mean(sbp, 3), _rolling_std_mean(sbp, 5), _rolling_std_mean(sbp, 7)]
        feats += [_short_window_slope(sbp, 3), _short_window_slope(sbp, 5), _short_window_slope(sbp, 7)]
        feats += [_min_pos_ratio(sbp)]
        feats += _neighbor_features(sbp, times, et, w=5)
        feats += _seq_features(map_)
        feats += [_rolling_std_mean(map_, 3), _rolling_std_mean(map_, 5), _rolling_std_mean(map_, 7)]
        feats += [_short_window_slope(map_, 3), _short_window_slope(map_, 5), _short_window_slope(map_, 7)]
        feats += [_min_pos_ratio(map_)]
        feats += _neighbor_features(map_, times, et, w=5)
        feats += _seq_features(hr)
        feats += [_rolling_std_mean(hr, 3), _rolling_std_mean(hr, 5), _rolling_std_mean(hr, 7)]
        feats += [_short_window_slope(hr, 3), _short_window_slope(hr, 5), _short_window_slope(hr, 7)]
        feats += [_min_pos_ratio(hr)]
        feats += _neighbor_features(hr, times, et, w=5)
        feats += _seq_features(uf)
        feats += [_rolling_std_mean(uf, 3), _rolling_std_mean(uf, 5), _rolling_std_mean(uf, 7)]
        feats += [_short_window_slope(uf, 3), _short_window_slope(uf, 5), _short_window_slope(uf, 7)]
        feats += [_mean_abs_diff(uf)]
        feats += _neighbor_features(uf, times, et, w=5)
        feats += [
            _get_float(row, '透前收缩压'),
            _get_float(row, '透前舒张压'),
            _get_float(row, '透前动脉压'),
            _get_float(row, '透前体重'),
            _get_float(row, '干体重'),
            _get_float(row, '透析龄_天数'),
            _get_float(row, '实际透析时长')
        ]
        feats += _extract_optional_static(row, base_features_core)
        feats += _extract_optional_static(row, base_features_history)
        feats += _extract_optional_static(row, base_features_current)
        features.append(feats)
        labels.append(build_stage_label(et, boundaries=boundaries))
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y

def _pad_or_truncate(seq, L):
    arr = [v for v in (seq or []) if v is not None]
    if len(arr) == 0:
        arr = [0.0]
    if len(arr) >= L:
        return np.array(arr[:L], dtype=float)
    else:
        pad = np.pad(np.array(arr, dtype=float), (0, L - len(arr)), mode='edge')
        return pad

def load_seq_static_survival(csv_path: str, boundaries=(30,90), seq_length=30, sample_limit=None,
                             columns: Dict[str, Any] = None, missing_threshold: float = 0.4, gap_threshold: float = 15.0):
    df = pd.read_csv(csv_path)
    X_seq_list = []
    X_static_list = []
    y_list = []
    durations = []
    events = []
    count = 0
    for _, row in df.iterrows():
        sbp_col = columns.get('sbp_col', '透析中收缩压') if columns else '透析中收缩压'
        map_col = columns.get('map_col', '动脉压') if columns else '动脉压'
        hr_col = columns.get('hr_col', '心率') if columns else '心率'
        uf_col = columns.get('uf_col', '超滤率') if columns else '超滤率'
        time_col = columns.get('time_col', '透中数据记录时间节点') if columns else '透中数据记录时间节点'
        sbp = _parse_list(row.get(sbp_col))
        map_ = _parse_list(row.get(map_col))
        hr = _parse_list(row.get(hr_col))
        uf = _parse_list(row.get(uf_col))
        times = _parse_times(row.get(time_col))
        if not filter_by_missing({"sbp": sbp, "map": map_, "hr": hr, "uf": uf}, missing_threshold=missing_threshold):
            continue
        sbp_pad = _pad_or_truncate(sbp, seq_length)
        map_pad = _pad_or_truncate(map_, seq_length)
        hr_pad = _pad_or_truncate(hr, seq_length)
        uf_pad = _pad_or_truncate(uf, seq_length)
        x_seq = np.stack([sbp_pad, map_pad, hr_pad, uf_pad], axis=-1)
        X_seq_list.append(x_seq)
        # 静态特征：首值与简单统计
        static = [
            sbp_pad[0], map_pad[0], hr_pad[0], uf_pad[0], float(np.mean(sbp_pad)), float(np.std(sbp_pad)),
            _get_float(row, '透前收缩压'),
            _get_float(row, '透前舒张压'),
            _get_float(row, '透前动脉压'),
            _get_float(row, '透前体重'),
            _get_float(row, '干体重'),
            _get_float(row, '透析龄_天数'),
            _get_float(row, '实际透析时长')
        ]
        X_static_list.append(static)
        et = _event_time_from_final(row, columns=columns)
        if et is None:
            et = _event_time_minutes(row)
        y_list.append(build_stage_label(et, boundaries=boundaries))
        # 生存标签
        start_time_col = columns.get('start_time_col', '透析开始时间') if columns else '透析开始时间'
        end_time_col = columns.get('end_time_col', '透析结束时间') if columns else '透析结束时间'
        dt_start = pd.to_datetime(row.get(start_time_col), errors='coerce')
        dt_end = pd.to_datetime(row.get(end_time_col), errors='coerce')
        if not pd.isna(dt_start) and not pd.isna(dt_end):
            last_obs = float((dt_end - dt_start).total_seconds() / 60.0)
        else:
            last_obs = times[-1] if times and times[-1] is not None else float(seq_length)
        durations.append(float(et) if et is not None else float(last_obs))
        events.append(1 if et is not None else 0)
        count += 1
        if sample_limit and count >= sample_limit:
            break
    X_seq = np.array(X_seq_list, dtype=float)
    X_static = np.array(X_static_list, dtype=float)
    y_stage = np.array(y_list, dtype=int)
    durations = np.array(durations, dtype=float)
    events = np.array(events, dtype=int)
    # 对齐长度
    min_len = min(len(X_seq), len(X_static), len(y_stage), len(durations), len(events))
    X_seq = X_seq[:min_len]
    X_static = X_static[:min_len]
    y_stage = y_stage[:min_len]
    durations = durations[:min_len]
    events = events[:min_len]
    return X_seq, X_static, y_stage, durations, events