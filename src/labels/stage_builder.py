from typing import Dict, Any, Sequence

def build_stage_label(event_time_min: float, boundaries: Sequence[float] = (30, 90, 120)) -> int:
    if event_time_min is None:
        return 0
    b = list(boundaries)
    b.sort()
    if len(b) == 2:
        b1, b2 = b
        if event_time_min <= b1:
            return 1
        if event_time_min <= b2:
            return 2
        return 3
    else:
        b1, b2, b3 = b[:3]
        if event_time_min <= b1:
            return 1
        if event_time_min <= b2:
            return 2
        if event_time_min <= b3:
            return 3
        return 3

def to_record(event_time_min, last_observed_min, is_censored):
    return {
        'event_time': event_time_min,
        'stage_label': build_stage_label(event_time_min),
        'is_censored': bool(is_censored),
        'last_observed_time': last_observed_min,
    }