import numpy as np

def normalize_times_to_minutes(times):
    out = []
    base = None
    for t in times or []:
        if t is None:
            out.append(None)
            continue
        if isinstance(t, (int, float)):
            minute = float(t)
        else:
            s = str(t)
            if ':' in s:
                h, m = s.split(':')[:2]
                minute = int(h) * 60 + int(m)
            else:
                try:
                    minute = float(s)
                except Exception:
                    minute = None
        if base is None:
            base = minute or 0.0
        out.append(minute - base if minute is not None else None)
    return out

def split_sessions(norm_minutes, gap_threshold=15.0):
    sessions = []
    current = []
    prev = None
    for t in norm_minutes:
        if t is None:
            continue
        if prev is None or (t - prev) <= gap_threshold:
            current.append(t)
        else:
            if current:
                sessions.append(current)
            current = [t]
        prev = t
    if current:
        sessions.append(current)
    return sessions

def filter_by_missing(seq_dict, missing_threshold=0.4):
    keys = list(seq_dict.keys())
    valid = True
    for k in keys:
        seq = seq_dict[k] or []
        total = len(seq)
        if total == 0:
            continue
        miss = sum(1 for v in seq if v is None)
        if total > 0 and miss / total > missing_threshold:
            valid = False
            break
    return valid