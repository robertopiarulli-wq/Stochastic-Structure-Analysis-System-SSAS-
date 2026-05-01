from core.generator import generate_ensemble
from core.features import features

def run_simulation(n=100000):
    results = []
    for s in generate_ensemble(n):
        f = features(s)
        results.append((s, f["H"]))
    return results
