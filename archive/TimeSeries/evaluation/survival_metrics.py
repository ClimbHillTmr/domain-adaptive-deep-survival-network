import numpy as np
from lifelines.utils import concordance_index

def c_index(durations, events, risks):
    return float(concordance_index(durations, -np.array(risks), events))