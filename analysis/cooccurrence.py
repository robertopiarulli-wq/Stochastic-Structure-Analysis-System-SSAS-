from collections import Counter
import itertools

def pair_analysis(ensemble):
    pairs = Counter()
    for s in ensemble:
        pairs.update(itertools.combinations(s, 2))
    return pairs
