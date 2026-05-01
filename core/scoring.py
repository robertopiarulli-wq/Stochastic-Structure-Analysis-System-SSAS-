import numpy as np

def score(h, target_h, delta, target_delta):
    return abs(h - target_h) + 10 * abs(delta - target_delta)
