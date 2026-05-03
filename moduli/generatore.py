"""
SSAS - Motore 3: Generatore Sestine
Filtri a cascata su campionamento sistematico.
Usato sia dal flusso base che dal flusso Wyckoff.
"""
from itertools import combinations
import numpy as np
import random

# ── Costanti strutturali dai dati reali ─────────────────
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

# ── Caricamento dati ─────────────────────────────────────
def carica_storico(client):
    print("  Caricamento storico completo...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .limit(10000)\
        .execute()

    estrazioni = []
    for row in res.data:
        numeri = tuple(sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ]))
        estrazioni.append(numeri)

    storico_np   = np.array(estrazioni)
    figure_viste = set()
    for numeri in estrazioni:
        gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
        figure_viste.add(gaps)

    print(f"  Estrazioni caricate: {len(estrazioni):,}")
    print(f"  Figure gap viste:    {len(figure_viste):,}")
    return storico_np, figure_viste

def carica_triple_attive(client, n_estrazioni=50):
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

def check_parita_decade(sestina, vincoli):
    """
    Verifica solo il vincolo parità intermedia della fascia.
    Il vincolo decade è disabilitato: nella fascia target alta
    (360-390) le combinazioni di decade sono strutturalmente
    sbilanciate verso A(61-90) per ragioni matematiche
    e il filtro risulterebbe troppo restrittivo.
    """
    if vincoli is None:
        return True
    n_pari = sum(1 for n in sestina if n % 2 == 0)
    return n_pari == vincoli['n_pari']

# ── Filtri a cascata ─────────────────────────────────────
def check_overlap(sestina_set, storico_np):
    arr      = np.array(list(sestina_set))
    overlaps = np.sum(np.isin(storico_np, arr), axis=1)
    return int(overlaps.max()) < 4

def check_figura_gap(numeri, figure_viste):
    gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
    return gaps not in figure_viste

def check_triple_attive(sestina, triple_attive):
    for t in combinations(sestina, 3):
        if t in triple_attive:
            return False
    return True

def check_strutturali(numeri, mappa_z,
                      somma_min=SOMMA_MIN, somma_max=SOMMA_MAX):
    somma = sum(numeri)
    if not (somma_min <= somma <= somma_max):
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

# ── Motore principale (flusso base) ─────────────────────
def ricerca_sistematica(
    storico_np,
    figure_viste,
    triple_attive,
    mappa_z,
    n_campioni  = 5000000,
    max_sestine = 10000,
    seed        = 42
):
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n  Avvio ricerca base: {n_campioni:,} campioni")
    print(f"  Max sestine:        {max_sestine:,}")

    sestine_trovate = []
    scarti = {
        "overlap":       0,
        "somma":         0,
        "range":         0,
        "decadi":        0,
        "parita":        0,
        "spacing":       0,
        "densita":       0,
        "figura_gap":    0,
        "triple_attive": 0,
    }

    numeri_pool = list(range(1, 91))

    for i in range(n_campioni):
        if len(sestine_trovate) >= max_sestine:
            break

        sestina     = sorted(random.sample(numeri_pool, 6))
        sestina_t   = tuple(sestina)
        sestina_set = set(sestina)

        if not check_overlap(sestina_set, storico_np):
            scarti["overlap"] += 1
            continue

        passa, motivo = check_strutturali(sestina, mappa_z)
        if not passa:
            scarti[motivo] += 1
            continue

        if not check_figura_gap(sestina, figure_viste):
            scarti["figura_gap"] += 1
            continue

        if not check_triple_attive(sestina_t, triple_attive):
            scarti["triple_attive"] += 1
            continue

        sestine_trovate.append(sestina)

        if len(sestine_trovate) % 1000 == 0:
            print(f"  Trovate {len(sestine_trovate):,} "
                  f"su {i+1:,} campioni...")

    print(f"\n  === Risultati Base ===")
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

# ── Motore Wyckoff (flusso pool ristretto) ───────────────
def ricerca_su_pool(
    pool,
    storico_np,
    figure_viste,
    triple_attive,
    mappa_z,
    fascia_min,
    fascia_max,
    vincoli     = None,
    n_campioni  = 2000000,
    max_sestine = 5000,
    seed        = 42
):
    random.seed(seed)
    np.random.seed(seed)

    if len(pool) < 5:
        print("  [Motore3-Wyckoff] Pool troppo piccolo, skip.")
        return []

    print(f"\n  Avvio ricerca Wyckoff guidata dalla somma...")
    print(f"  Pool: {sorted(pool)}")
    print(f"  Fascia somma: {fascia_min}-{fascia_max}")
    print(f"  Max sestine: {max_sestine:,}")
    if vincoli:
        print(f"  Vincolo parità: "
              f"{vincoli['n_pari']}p/{vincoli['n_disp']}d "
              f"({vincoli['pct_pari']}% nella fascia)")

    sestine_trovate = []
    scarti = {
        "no_sesto":      0,
        "parita":        0,
        "overlap":       0,
        "strutturali":   0,
        "figura_gap":    0,
        "triple_attive": 0,
    }

    tentativi = 0

    for _ in range(n_campioni):
        if len(sestine_trovate) >= max_sestine:
            break

        tentativi += 1

        # Campiona 5 numeri dal pool
        cinque = sorted(random.sample(pool, 5))
        somma5 = sum(cinque)

        # Calcola il 6° numero per chiudere la somma target
        target = random.randint(fascia_min, fascia_max)
        sesto  = target - somma5

        # Verifica validità del 6° numero
        if sesto < 1 or sesto > 90:
            scarti["no_sesto"] += 1
            continue
        if sesto in set(cinque):
            scarti["no_sesto"] += 1
            continue

        # Costruisci sestina: 5 dal pool + sesto calcolato
        sestina     = tuple(sorted(cinque + [sesto]))
        sestina_set = set(sestina)
        somma       = sum(sestina)

        # Doppia verifica fascia
        if not (fascia_min <= somma <= fascia_max):
            scarti["no_sesto"] += 1
            continue

        # FILTRO 0: Parità intermedia della fascia
        if not check_parita_decade(sestina, vincoli):
            scarti["parita"] += 1
            continue

        # FILTRO 1: Overlap
        if not check_overlap(sestina_set, storico_np):
            scarti["overlap"] += 1
            continue

        # FILTRO 2: Strutturali
        passa, motivo = check_strutturali(
            list(sestina), mappa_z,
            somma_min=fascia_min,
            somma_max=fascia_max
        )
        if not passa and motivo != "somma":
            scarti["strutturali"] += 1
            continue

        # FILTRO 3: Figure gap
        if not check_figura_gap(list(sestina), figure_viste):
            scarti["figura_gap"] += 1
            continue

        # FILTRO 4: Triple attive
        if not check_triple_attive(sestina, triple_attive):
            scarti["triple_attive"] += 1
            continue

        sestine_trovate.append(list(sestina))

        if len(sestine_trovate) % 500 == 0:
            print(f"  Trovate {len(sestine_trovate):,} "
                  f"su {tentativi:,} tentativi...")

    print(f"\n  === Risultati Wyckoff ===")
    print(f"  Tentativi:        {tentativi:,}")
    print(f"  Sestine trovate:  {len(sestine_trovate):,}")
    print(f"  Tasso successo:   "
          f"{len(sestine_trovate)*100/max(tentativi,1):.4f}%")
    print(f"\n  Scarti per filtro:")
    tot = sum(scarti.values())
    for k, v in scarti.items():
        if v > 0:
            print(f"    {k:15s}: {v:>10,} "
                  f"({v*100/max(tot,1):.1f}%)")

    return sestine_trovate
