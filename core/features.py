import numpy as np

def rugosita(s):
    s = np.sort(s)
    diffs = np.diff(s)
    mu = np.mean(diffs)
    return np.std(diffs) / mu if mu != 0 else 0


def delta_h(h, h_prev):
    return h - h_prev
