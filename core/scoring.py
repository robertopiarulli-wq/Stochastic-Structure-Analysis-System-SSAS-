import numpy as np

def score(h, target_h, delta, target_delta, w_h=1, w_delta=10):
    err_h = abs(h - target_h)
    err_delta = abs(delta - target_delta)
    
    return (w_h * err_h) + (w_delta * err_delta)
