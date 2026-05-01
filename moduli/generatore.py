"""
SSAS - Motore di Ricerca Sistematica
5 filtri integrati:
1. Triple vergini (database completo)
2. Parametri strutturali (somma, range, decadi, parità)
3. Spacing ratio Wigner-Dyson
4. Densità locale mappa occupazione
5. Anti-ricorrenza figure gap
"""
from itertools import combinations
import numpy as np

# ── Costanti strutturali dai dati reali ─────────────────
# Da costanti_sistema calcolate su 7304 estrazioni
SOMMA_MIN       = 132   # percentile 5
SOMMA_MAX       = 420   # percentile 95
RANGE_MIN       = 32    # percentile 5
RANGE_MAX       = 88    # percentile 95
DECADI_MIN      = 3
DECADI_MAX      = 6
PARI_MIN        = 1
PARI_MAX        = 5
SPACING_MIN     = 0.15
SPACING_MAX     = 0.65

def carica_triple_viste(client):
    """Triple storiche su intero database."""
    print("  Caricamento triple storiche (intero database)...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .limit(10000)\
        .execute()

    triple_viste = set()
    for row in res.data:
        numeri = sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ])
        for t in combinations(numeri, 3):
            triple_viste.add(t)

    print(f"  Triple viste:   {len(triple_viste):,}")
    print(f"  Triple vergini: {117480 - len(triple_viste):,}")
    return triple_viste

def carica_figure_gap(client):
    """
    Carica i vettori gap di tutte le estrazioni storiche.
    Filtro 5: anti-ricorrenza figure geometriche.
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
    """
    Carica z_score per ogni numero 1-90.
    Filtro 4: densità locale.
    """
    res = client.table("mappa_occupazione")\
        .select("numero,z_score")\
        .execute()

    mappa = {}
    for row in res.data:
        mappa[row['numero']] = row['z_score'] or 0.0
    return mappa

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
    return len(set((n - 1) // 10 for n in numeri))

def passa_filtri_strutturali(sestina, figure_viste, mappa_z):
    """
    Applica filtri 2-5 sulla sestina completa.
    Ritorna (True, None) se passa tutti
    Ritorna (False, motivo) se scartata
    """
    numeri = sorted(sestina)

    # FILTRO 2a: Somma
    somma = sum(numeri)
    if not (SOMMA_MIN <= somma <= SOMMA_MAX):
        return False, "somma"

    # FILTRO 2b: Range
    range_tot = numeri[-1] - numeri[0]
    if not (RANGE_MIN <= range_tot <= RANGE_MAX):
        return False, "range"

    # FILTRO 2c: Decadi
    decadi = calcola_decadi(numeri)
    if not (DECADI_MIN <= decadi <= DECADI_MAX):
        return False, "decadi"

    # FILTRO 2d: Parità
    n_pari = sum(1 for n in numeri if n % 2 == 0)
    if not (PARI_MIN <= n_pari <= PARI_MAX):
        return False, "parita"

    # FILTRO 3: Spacing ratio Wigner-Dyson
    sr = calcola_spacing_ratio(numeri)
    if not (SPACING_MIN <= sr <= SPACING_MAX):
        return False, "spacing"

    # FILTRO 4: Densità locale (z_score)
    # Scarta sestine con troppi numeri in zone estreme
    z_scores = [mappa_z.get(n, 0.0) for n in numeri]
    z_estremi = sum(1 for z in z_scores if abs(z) > 2.0)
    if z_estremi > 2:
        return False, "densita"

    # FILTRO 5: Figure gap mai viste
    gaps = tuple(numeri[i+1] - numeri[i] for i in range(5))
    if gaps in figure_viste:
        return False, "figura_gap"

    return True, None

# ── Estensione sistematica ───────────────────────────────
def trova_estensioni_valide(base, triple_viste):
    """
    Testa 1-90 come prossimo elemento della base.
    Un numero è valido se tutte le nuove triple
    che forma con i sottoinsiemi da 2 della base
    sono vergini.
    """
    numeri_usati = set(base)
    validi = []

    for n in range(1, 91):
        if n in numeri_usati:
            continue
        virgine = True
        for coppia in combinations(base, 2):
            nuova_t = tuple(sorted(coppia + (n,)))
            if nuova_t in triple_viste:
                virgine = False
                break
        if virgine:
            validi.append(n)

    return validi

# ── Motore principale ────────────────────────────────────
def ricerca_sistematica(
    triple_viste,
    figure_viste,
    mappa_z,
    max_triple_base = 500,
    max_sestine     = 10000
):
    """
    Ricerca deterministica su triple vergini
    con 5 filtri integrati.
    """
    print(f"\n  Avvio ricerca sistematica...")
    print(f"  Max triple base: {max_triple_base:,}")
    print(f"  Max sestine:     {max_sestine:,}")

    sestine_trovate  = []
    triple_esaminate = 0
    triple_vergini   = 0

    # Contatori scarto per filtro
    scarti = {
        "somma": 0, "range": 0, "decadi": 0,
        "parita": 0, "spacing": 0, "densita": 0,
        "figura_gap": 0, "no_quarto": 0,
        "no_quinto": 0, "no_sesto": 0
    }

    stats_quarti = []
    stats_quinti = []
    stats_sesti  = []

    for tripla in combinations(range(1, 91), 3):
        if len(sestine_trovate) >= max_sestine:
            break
        if triple_vergini >= max_triple_base:
            break

        triple_esaminate += 1

        # Tripla base deve essere vergine
        if tripla in triple_viste:
            continue

        triple_vergini += 1

        # STEP 2: trova 4° numero valido
        quarti = trova_estensioni_valide(tripla, triple_viste)
        stats_quarti.append(len(quarti))

        if not quarti:
            scarti["no_quarto"] += 1
            continue

        # STEP 3: per ogni quadrupla valida trova 5°
        for d in quarti:
            if len(sestine_trovate) >= max_sestine:
                break

            quadrupla = tuple(sorted(tripla + (d,)))
            quinti = trova_estensioni_valide(quadrupla, triple_viste)
            stats_quinti.append(len(quinti))

            if not quinti:
                scarti["no_quinto"] += 1
                continue

            # STEP 4: per ogni quintupla valida trova 6°
            for e in quinti:
                if len(sestine_trovate) >= max_sestine:
                    break

                quintupla = tuple(sorted(quadrupla + (e,)))
                sesti = trova_estensioni_valide(quintupla, triple_viste)
                stats_sesti.append(len(sesti))

                if not sesti:
                    scarti["no_sesto"] += 1
                    continue

                # STEP 5: applica filtri strutturali
                for f in sesti:
                    if len(sestine_trovate) >= max_sestine:
                        break

                    sestina = tuple(sorted(quintupla + (f,)))

                    passa, motivo = passa_filtri_strutturali(
                        sestina, figure_viste, mappa_z
                    )

                    if not passa:
                        scarti[motivo] += 1
                        continue

                    sestine_trovate.append(list(sestina))

        if triple_vergini % 50 == 0:
            print(f"  Triple vergini: {triple_vergini:,} "
                  f"| Esaminate: {triple_esaminate:,} "
                  f"| Sestine: {len(sestine_trovate):,}")

    # Report finale
    print(f"\n  === Risultati ===")
    print(f"  Triple esaminate:       {triple_esaminate:,}")
    print(f"  Triple vergini:         {triple_vergini:,}")
    print(f"  Sestine generate:       {len(sestine_trovate):,}")
    print(f"\n  Scarti per filtro:")
    for k, v in scarti.items():
        if v > 0:
            print(f"    {k:20s}: {v:,}")

    if stats_quarti:
        print(f"\n  4° validi/tripla:    "
              f"media={np.mean(stats_quarti):.2f} "
              f"min={min(stats_quarti)} "
              f"max={max(stats_quarti)}")
    if stats_quinti:
        print(f"  5° validi/quadrupla: "
              f"media={np.mean(stats_quinti):.2f} "
              f"min={min(stats_quinti)} "
              f"max={max(stats_quinti)}")
    if stats_sesti:
        print(f"  6° validi/quintupla: "
              f"media={np.mean(stats_sesti):.2f} "
              f"min={min(stats_sesti)} "
              f"max={max(stats_sesti)}")

    return sestine_trovate
