"""
SSAS - Motore Generativo con Filtri a Cascata
Ordine ottimale per massima riduzione con minimo calcolo:

1. Overlap < 4 con storico completo  → -41%
2. Figure gap mai viste               → -30%
3. Triple attive ultime N estrazioni  → -70%
4. Filtri strutturali                 → -10%
"""
from itertools import combinations
import numpy as np
import random

# ── Costanti strutturali ─────────────────────────────────
SOMMA_MIN   = 132
SOMMA_MAX   = 420
RANGE_MIN   = 32
RANGE_MAX   = 88
DECADI_MIN  = 3
DECADI_MAX  = 6
PARI_MIN    = 1
PARI_MAX    = 5
SPACING_MIN = 0.15
SPACING_MAX = 0.65

def carica_storico(client):
    """
    Carica tutte le estrazioni e costruisce:
    - set sestine viste (per overlap check)
    - set figure gap viste
    - set triple attive ultime N estrazioni
    """
    print("  Caricamento storico completo...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .order("data_estrazione", desc=False)\
        .limit(10000)\
        .execute()

    estrazioni = []
    for row in res.data:
        numeri = tuple(sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ]))
        estrazioni.append(numeri)

    # Lista numpy per overlap veloce
    storico_np = np.array(estrazioni)  # shape (7304, 6)

    # Figure gap viste
    figure_viste = set()
    for numeri in estrazioni:
        gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
        figure_viste.add(gaps)

    print(f"  Estrazioni caricate: {len(estrazioni):,}")
    print(f"  Figure gap viste:    {len(figure_viste):,}")

    return storico_np, figure_viste

def carica_triple_attive(client, n_estrazioni=50):
    """
    Triple delle ultime N estrazioni.
    N=50 ≈ 4 mesi → filtro potente.
    """
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .order("data_estrazione", desc=True)\
        .limit(n_estrazioni)\
        .execute()

    triple_attive = set()
    for row in res.data:
        numeri = sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ])
        for t in combinations(numeri, 3):
            triple_attive.add(t)

    print(f"  Triple attive (ultime {n_estrazioni}): "
          f"{len(triple_attive):,}")
    return triple_attive

def carica_mappa_occupazione(client):
    """Z-score per ogni numero 1-90."""
    res = client.table("mappa_occupazione")\
        .select("numero,z_score")\
        .execute()
    return {row['numero']: row['z_score'] or 0.0
            for row in res.data}

# ── Calcoli strutturali O(1) ─────────────────────────────
def calcola_spacing_ratio(numeri):
    gaps = [numeri[i+1] - numeri[i] for i in range(5)]
    ratios = []
    for i in range(len(gaps)-1):
        s1, s2 = gaps[i], gaps[i+1]
        if max(s1, s2) == 0:
            continue
        ratios.append(min(s1, s2) / max(s1, s2))
    return float(np.mean(ratios)) if ratios else 0.0

def calcola_decadi(numeri):
    return len(set((n-1) // 10 for n in numeri))

# ── Filtri a cascata ─────────────────────────────────────
def check_overlap(sestina_set, storico_np):
    """
    FILTRO 1: overlap < 4 con qualsiasi estrazione storica.
    Usa numpy per velocità massima.
    """
    arr = np.array(list(sestina_set))
    # Per ogni estrazione storica conta i numeri in comune
    overlaps = np.sum(np.isin(storico_np, arr), axis=1)
    return int(overlaps.max()) < 4

def check_figura_gap(numeri, figure_viste):
    """FILTRO 2: forma geometrica mai vista."""
    gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
    return gaps not in figure_viste

def check_triple_attive(sestina, triple_attive):
    """FILTRO 3: nessuna tripla nelle attive recenti."""
    for t in combinations(sestina, 3):
        if t in triple_attive:
            return False
    return True

def check_strutturali(numeri, mappa_z):
    """FILTRO 4: parametri strutturali."""
    somma = sum(numeri)
    if not (SOMMA_MIN <= somma <= SOMMA_MAX):
        return False, "somma"

    range_tot = numeri[-1] - numeri[0]
    if not (RANGE_MIN <= range_tot <= RANGE_MAX):
        return False, "range"

    if not (DECADI_MIN <= calcola_decadi(numeri) <= DECADI_MAX):
        return False, "decadi"

    n_pari = sum(1 for n in numeri if n % 2 == 0)
    if not (PARI_MIN <= n_pari <= PARI_MAX):
        return False, "parita"

    sr = calcola_spacing_ratio(numeri)
    if not (SPACING_MIN <= sr <= SPACING_MAX):
        return False, "spacing"

    z_estremi = sum(1 for n in numeri
                    if abs(mappa_z.get(n, 0.0)) > 2.0)
    if z_estremi > 2:
        return False, "densita"

    return True, None

# ── Motore principale ────────────────────────────────────
def ricerca_sistematica(
    storico_np,
    figure_viste,
    triple_attive,
    mappa_z,
    n_campioni  = 5000000,
    max_sestine = 10000,
    seed        = 42
):
    """
    Campionamento casuale con filtri a cascata.
    
    Ordine filtri per massima efficienza:
    1. Overlap < 4     (numpy, velocissimo)
    2. Strutturali     (O(1))
    3. Figure gap      (set lookup O(1))
    4. Triple attive   (set lookup O(20))
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n  Avvio ricerca: {n_campioni:,} campioni")
    print(f"  Max sestine:   {max_sestine:,}")

    sestine_trovate = []
    scarti = {
        "overlap":      0,
        "somma":        0,
        "range":        0,
        "decadi":       0,
        "parita":       0,
        "spacing":      0,
        "densita":      0,
        "figura_gap":   0,
        "triple_attive":0,
    }

    numeri_pool = list(range(1, 91))

    for i in range(n_campioni):
        if len(sestine_trovate) >= max_sestine:
            break

        # Genera sestina casuale
        sestina = sorted(random.sample(numeri_pool, 6))
        sestina_t   = tuple(sestina)
        sestina_set = set(sestina)

        # FILTRO 1: Overlap
        if not check_overlap(sestina_set, storico_np):
            scarti["overlap"] += 1
            continue

        # FILTRO 2: Strutturali
        passa, motivo = check_strutturali(sestina, mappa_z)
        if not passa:
            scarti[motivo] += 1
            continue

        # FILTRO 3: Figure gap
        if not check_figura_gap(sestina, figure_viste):
            scarti["figura_gap"] += 1
            continue

        # FILTRO 4: Triple attive
        if not check_triple_attive(sestina_t, triple_attive):
            scarti["triple_attive"] += 1
            continue

        sestine_trovate.append(sestina)

        if len(sestine_trovate) % 1000 == 0:
            print(f"  Trovate {len(sestine_trovate):,} "
                  f"su {i+1:,} campioni...")

    # Report
    print(f"\n  === Risultati ===")
    print(f"  Campioni testati:  {min(i+1, n_campioni):,}")
    print(f"  Sestine trovate:   {len(sestine_trovate):,}")
    print(f"  Tasso successo:    "
          f"{len(sestine_trovate)*100/max(i+1,1):.4f}%")
    print(f"\n  Scarti per filtro:")
    tot = sum(scarti.values())
    for k, v in scarti.items():
        if v > 0:
            print(f"    {k:15s}: {v:>10,} "
                  f"({v*100/max(tot,1):.1f}%)")

    return sestine_trovate
