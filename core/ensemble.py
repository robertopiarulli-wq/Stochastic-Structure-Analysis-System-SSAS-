import random
from core.features import rugosita, delta_h
from core.scoring import score

def generate_structured_ensemble(
    n,
    target_h,
    target_delta,
    h_last,
    tolerance=0.01
):
    results = []

    for _ in range(n):
        s = sorted(random.sample(range(1, 91), 6))
        h = rugosita(s)
        d = delta_h(h, h_last)

        sc = score(h, target_h, d, target_delta)

        # filtro base (morsa)
        if abs(h - target_h) < tolerance:
            results.append((s, h, d, sc))

    # ordinamento per qualità
    results.sort(key=lambda x: x[3])

    return results
