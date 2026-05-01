"""
SSAS - Motore Generativo su Quadruple Vergini
Logica deterministica a 3 livelli:
1. Quadrupla vergine (mai apparsa nelle 7304 estrazioni)
2. Quintina vergine (mai apparsa nelle 7304 estrazioni)
3. Sestina vergine (mai apparsa nelle 7304 estrazioni)
+ Filtri strutturali
"""
from itertools import combinations
import numpy as np

# ── Costanti strutturali dai dati reali ─────────────────
SOMMA_MIN    = 132
SOMMA_MAX    = 420
RANGE_MIN    = 32
RANGE_MAX    = 88
DECADI_MIN   = 3
DECADI_MAX   = 6
PARI_MIN     = 1
PARI_MAX     = 5
SPACING_MIN  = 0.15
SPACING_MAX  = 0.65

def carica_sottoinsiemi(client):
    """
    Carica da Supabase tutte le estrazioni e costruisce
    i set di lookup per quadruple, quintuple e sestine.
    """
    print("  Caricamento estrazioni per costruzione lookup...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .limit(10000)\
        .execute()

    quadruple_viste = set()
    quintuple_viste = set()
    sestine_viste   = set()

    for row in res.data:
        numeri = tuple(sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ]))

        # Sestina esatta
        sestine_viste.add(numeri)

        # Tutte le quintuple interne C(6,5)=6
        for q in combinations(numeri, 5):
            quintuple_viste.add(q)

        # Tutte le quadruple interne C(6,4)=15
        for q in combinations(numeri, 4):
            quadruple_viste.add(q)

    print(f"  Sestine viste:   {len(sestine_viste):,}")
    print(f"  Quintuple viste: {len(quintuple_viste):,}")
    print(f"  Quadruple viste: {len(quadruple_viste):,}")
    print(f"  Quadruple totali possibili: 2.555.190")
    print(f"  Quadruple vergini: "
          f"{2555190 - len(quadruple_viste):,} "
          f"({(2555190-len(quadruple_viste))*100/2555190:.2f}%)")

    return quadruple_viste, quintuple_viste, sestine_viste

def carica_figure_gap(client):
    """
    Vettori gap di tutte le estrazioni storiche.
    Filtro anti-ricorrenza forme geometriche.
    """
    print("  Caricamento figure gap storiche...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .limit(10000)\
        .execute()

    figure_viste = set()
    for row in res.data:
        numeri = sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ])
        gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
        figure_viste.add(gaps)

    print(f"  Figure gap viste: {len(figure_viste):,}")
    return figure_viste

def carica_mappa_occupazione(client):
    """Z-score per ogni numero 1-90."""
    res = client.table("mappa_occupazione")\
        .select("numero,z_score")\
        .execute()
    return {row['numero']: row['z_score'] or 0.0
            for row in res.data}

# ── Calcoli strutturali ──────────────────────────────────
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

def passa_filtri_strutturali(sestina, figure_viste, mappa_z):
    """
    Filtri strutturali sulla sestina completa.
    Ritorna (True, None) o (False, motivo)
    """
    numeri = sorted(sestina)

    # Somma
    somma = sum(numeri)
    if not (SOMMA_MIN <= somma <= SOMMA_MAX):
        return False, "somma"

    # Range
    if not (RANGE_MIN <= numeri[-1]-numeri[0] <= RANGE_MAX):
        return False, "range"

    # Decadi
    if not (DECADI_MIN <= calcola_decadi(numeri) <= DECADI_MAX):
        return False, "decadi"

    # Parità
    n_pari = sum(1 for n in numeri if n % 2 == 0)
    if not (PARI_MIN <= n_pari <= PARI_MAX):
        return False, "parita"

    # Spacing ratio Wigner-Dyson
    sr = calcola_spacing_ratio(numeri)
    if not (SPACING_MIN <= sr <= SPACING_MAX):
        return False, "spacing"

    # Densità locale: max 2 numeri con z_score estremo
    z_estremi = sum(1 for n in numeri
                    if abs(mappa_z.get(n, 0.0)) > 2.0)
    if z_estremi > 2:
        return False, "densita"

    # Figure gap: forma geometrica mai vista
    gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
    if gaps in figure_viste:
        return False, "figura_gap"

    return True, None

# ── Motore principale ────────────────────────────────────
def ricerca_sistematica(
    quadruple_viste,
    quintuple_viste,
    sestine_viste,
    figure_viste,
    mappa_z,
    max_quadruple_base = 500,
    max_sestine        = 10000
):
    """
    Ricerca deterministica a 3 livelli:
    1. Quadrupla vergine
    2. Quintina vergine
    3. Sestina vergine
    + Filtri strutturali
    """
    print(f"\n  Avvio ricerca sistematica...")
    print(f"  Max quadruple base: {max_quadruple_base:,}")
    print(f"  Max sestine:        {max_sestine:,}")

    sestine_trovate     = []
    quadruple_esaminate = 0
    quadruple_vergini   = 0

    scarti = {
        "quintina_vista":  0,
        "sestina_vista":   0,
        "somma":           0,
        "range":           0,
        "decadi":          0,
        "parita":          0,
        "spacing":         0,
        "densita":         0,
        "figura_gap":      0,
        "no_quinto":       0,
        "no_sesto":        0,
    }

    stats_quinti = []
    stats_sesti  = []

    # Itera C(90,4) = 2.555.190 quadruple in ordine
    for quad in combinations(range(1, 91), 4):
        if len(sestine_trovate) >= max_sestine:
            break
        if quadruple_vergini >= max_quadruple_base:
            break

        quadruple_esaminate += 1

        # STEP 1: quadrupla deve essere vergine
        if quad in quadruple_viste:
            continue

        quadruple_vergini += 1
        numeri_usati = set(quad)

        # STEP 2: testa 5° numero (1-90)
        # Scarta e se la quintina [a,b,c,d,e] è già vista
        quinti_validi = []
        for e in range(1, 91):
            if e in numeri_usati:
                continue
            quintina = tuple(sorted(quad + (e,)))
            if quintina in quintuple_viste:
                scarti["quintina_vista"] += 1
            else:
                quinti_validi.append(e)

        stats_quinti.append(len(quinti_validi))

        if not quinti_validi:
            scarti["no_quinto"] += 1
            continue

        # STEP 3: testa 6° numero (1-90)
        # Scarta f se la sestina [a,b,c,d,e,f] è già vista
        for e in quinti_validi:
            if len(sestine_trovate) >= max_sestine:
                break

            quintina     = tuple(sorted(quad + (e,)))
            numeri_usati5 = set(quintina)
            sesti_validi  = []

            for f in range(1, 91):
                if f in numeri_usati5:
                    continue
                sestina = tuple(sorted(quintina + (f,)))
                if sestina in sestine_viste:
                    scarti["sestina_vista"] += 1
                else:
                    sesti_validi.append(f)

            stats_sesti.append(len(sesti_validi))

            if not sesti_validi:
                scarti["no_sesto"] += 1
                continue

            # STEP 4: filtri strutturali
            for f in sesti_validi:
                if len(sestine_trovate) >= max_sestine:
                    break

                sestina = tuple(sorted(quintina + (f,)))
                passa, motivo = passa_filtri_strutturali(
                    sestina, figure_viste, mappa_z
                )

                if not passa:
                    scarti[motivo] += 1
                    continue

                sestine_trovate.append(list(sestina))

        if quadruple_vergini % 50 == 0:
            print(f"  Quadruple vergini: {quadruple_vergini:,} "
                  f"| Esaminate: {quadruple_esaminate:,} "
                  f"| Sestine: {len(sestine_trovate):,}")

    # Report finale
    print(f"\n  === Risultati ===")
    print(f"  Quadruple esaminate:    {quadruple_esaminate:,}")
    print(f"  Quadruple vergini:      {quadruple_vergini:,}")
    print(f"  Sestine generate:       {len(sestine_trovate):,}")
    print(f"\n  Scarti per filtro:")
    for k, v in scarti.items():
        if v > 0:
            print(f"    {k:20s}: {v:,}")

    if stats_quinti:
        print(f"\n  5° validi/quadrupla: "
              f"media={np.mean(stats_quinti):.1f} "
              f"min={min(stats_quinti)} "
              f"max={max(stats_quinti)}")
    if stats_sesti:
        print(f"  6° validi/quintupla: "
              f"media={np.mean(stats_sesti):.1f} "
              f"min={min(stats_sesti)} "
              f"max={max(stats_sesti)}")

    return sestine_trovate
