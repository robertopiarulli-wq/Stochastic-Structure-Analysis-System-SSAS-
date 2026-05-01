import numpy as np

def get_numeri(row):
    return sorted([row['n1'], row['n2'], 
                   row['n3'], row['n4'], 
                   row['n5'], row['n6']])

def calcola_gap(numeri):
    return [numeri[i+1] - numeri[i] for i in range(len(numeri)-1)]

def calcola_entropia(gaps):
    gaps = np.array(gaps, dtype=float)
    totale = gaps.sum()
    if totale == 0:
        return 0.0
    p = gaps / totale
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    return float(H / np.log2(5))

def calcola_spacing_ratio(gaps):
    """
    Wigner-Dyson spacing ratio
    Poisson puro  -> 0.386
    GOE correlato -> 0.536
    """
    if len(gaps) < 2:
        return 0.0
    ratios = []
    for i in range(len(gaps)-1):
        s1, s2 = gaps[i], gaps[i+1]
        if max(s1, s2) == 0:
            continue
        ratios.append(min(s1, s2) / max(s1, s2))
    return float(np.mean(ratios)) if ratios else 0.0

def calcola_decadi(numeri):
    return len(set((n - 1) // 10 for n in numeri))

def calcola_consecutivi(numeri):
    coppie = 0
    max_run = 1
    run = 1
    for i in range(len(numeri)-1):
        if numeri[i+1] - numeri[i] == 1:
            coppie += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return coppie, max_run

def calcola_overlap(set_a, set_b):
    return len(set_a & set_b)

def build_fingerprint(row, sets_storici):
    numeri  = get_numeri(row)
    nums_set = set(numeri)
    gaps    = calcola_gap(numeri)
    gap_arr = np.array(gaps)

    mu  = gap_arr.mean()
    std = gap_arr.std()
    cv  = float(std / mu) if mu != 0 else 0.0
    gap_ratio = float(gap_arr.max() / gap_arr.min()) \
                if gap_arr.min() != 0 else float(gap_arr.max())

    coppie, max_run = calcola_consecutivi(numeri)

    ov1 = calcola_overlap(nums_set, sets_storici[-1]) \
          if len(sets_storici) >= 1 else 0
    ov3 = calcola_overlap(nums_set, sets_storici[-3]) \
          if len(sets_storici) >= 3 else 0
    ov7 = calcola_overlap(nums_set, sets_storici[-7]) \
          if len(sets_storici) >= 7 else 0

    return {
        "estrazione_id":        int(row['id']),
        "data_estrazione":      str(row['data_estrazione']),
        "somma":                int(sum(numeri)),
        "n_pari":               int(sum(1 for n in numeri if n % 2 == 0)),
        "n_dispari":            int(sum(1 for n in numeri if n % 2 != 0)),
        "range_totale":         int(numeri[-1] - numeri[0]),
        "decadi_coperte":       int(calcola_decadi(numeri)),
        "gap_min":              int(gap_arr.min()),
        "gap_max":              int(gap_arr.max()),
        "gap_medio":            round(float(mu), 4),
        "gap_std":              round(float(std), 4),
        "cv_gap":               round(cv, 4),
        "gap_ratio":            round(gap_ratio, 4),
        "n_coppie_consecutive": int(coppie),
        "consecutivi_max":      int(max_run),
        "entropia_gap":         round(calcola_entropia(gaps), 4),
        "spacing_ratio_medio":  round(calcola_spacing_ratio(gaps), 4),
        "overlap_lag1":         int(ov1),
        "overlap_lag3":         int(ov3),
        "overlap_lag7":         int(ov7),
    }, nums_set
