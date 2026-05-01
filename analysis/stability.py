import numpy as np
from collections import Counter

def split_blocks(data, n_blocks=5):
    return np.array_split(data, n_blocks)


def frequency_block(block):
    freq = Counter()
    for s, *_ in block:
        freq.update(s)
    return freq


def stability_index(blocks):
    all_freqs = []

    for block in blocks:
        f = frequency_block(block)
        all_freqs.append([f.get(i, 0) for i in range(1, 91)])

    all_freqs = np.array(all_freqs)

    # deviazione media per numero
    stds = np.std(all_freqs, axis=0)

    # indice globale
    return 1 / (np.mean(stds) + 1e-9)
