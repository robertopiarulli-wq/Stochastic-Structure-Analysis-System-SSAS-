import numpy as np

def stability(blocks_freq):
    return 1 / (np.std(blocks_freq) + 1e-9)
