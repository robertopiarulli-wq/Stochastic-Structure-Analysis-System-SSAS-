from collections import Counter

def frequency_analysis(ensemble):
    freq = Counter()
    for s in ensemble:
        freq.update(s)
    return freq
