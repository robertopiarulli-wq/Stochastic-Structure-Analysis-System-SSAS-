"""
SSAS - Motore di Ricerca Sistematica su Triple Vergini
Approccio deterministico basato sulle triple mai viste
nel database storico completo (7304 estrazioni).

Selettività reale:
- Triple vergini: 29.12% = 34.215 su 117.480
- Una sestina ha 20 triple interne
- Tutte e 20 devono essere vergini
"""
from itertools import combinations
import numpy as np

def carica_triple_viste(client):
    """
    Carica tutte le triple storiche in un set Python.
    Lookup O(1) per ogni verifica.
    """
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

    print(f"  Triple viste storicamente: {len(triple_viste)}")
    print(f"  Triple vergini:            {117480 - len(triple_viste)}")
    return triple_viste

def carica_triple_attive(client, n_estrazioni=13):
    """
    Triple delle ultime N estrazioni.
    Filtro secondario sul breve periodo.
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

    print(f"  Triple attive (ultimi {n_estrazioni} concorsi): "
          f"{len(triple_attive)}")
    return triple_attive

def trova_quarti_validi(tripla, triple_viste):
    """
    Testa sistematicamente 1-90 come 4° elemento.

    Aggiungendo d a [a,b,c]:
    → 3 nuove triple: (a,b,d) (a,c,d) (b,c,d)
    → tutte e 3 devono essere vergini
    """
    numeri_usati = set(tripla)
    validi = []

    for d in range(1, 91):
        if d in numeri_usati:
            continue
        virgine = True
        for base in combinations(tripla, 2):
            nuova_t = tuple(sorted(base + (d,)))
            if nuova_t in triple_viste:
                virgine = False
                break
        if virgine:
            validi.append(d)

    return validi

def trova_quinti_validi(quadrupla, triple_viste):
    """
    Testa sistematicamente 1-90 come 5° elemento.

    Aggiungendo e a [a,b,c,d]:
    → 6 nuove triple: C([a,b,c,d],2) + e
    → tutte e 6 devono essere vergini
    """
    numeri_usati = set(quadrupla)
    validi = []

    for e in range(1, 91):
        if e in numeri_usati:
            continue
        virgine = True
        for base in combinations(quadrupla, 2):
            nuova_t = tuple(sorted(base + (e,)))
            if nuova_t in triple_viste:
                virgine = False
                break
        if virgine:
            validi.append(e)

    return validi

def trova_sesti_validi(quintupla, triple_viste):
    """
    Testa sistematicamente 1-90 come 6° elemento.

    Aggiungendo f a [a,b,c,d,e]:
    → 10 nuove triple: C([a,b,c,d,e],2) + f
    → tutte e 10 devono essere vergini
    """
    numeri_usati = set(quintupla)
    validi = []

    for f in range(1, 91):
        if f in numeri_usati:
            continue
        virgine = True
        for base in combinations(quintupla, 2):
            nuova_t = tuple(sorted(base + (f,)))
            if nuova_t in triple_viste:
                virgine = False
                break
        if virgine:
            validi.append(f)

    return validi

def ricerca_sistematica(
    triple_viste,
    triple_attive,
    max_triple_base=200,
    max_sestine=10000
):
    """
    Ricerca deterministica su triple vergini.

    1. Itera C(90,3) = 117.480 triple in ordine
    2. Seleziona solo triple vergini (mai viste storicamente)
    3. Per ognuna testa sistematicamente 1-90 come 4°
    4. Per ogni quadrupla valida testa 1-90 come 5°
    5. Per ogni quintupla valida testa 1-90 come 6°
    6. Filtro finale: triple attive ultimi 13 concorsi
    """
    print(f"\n  Ricerca sistematica su triple vergini...")
    print(f"  Max triple base: {max_triple_base}")
    print(f"  Max sestine:     {max_sestine}")

    sestine_trovate     = []
    triple_esaminate    = 0
    triple_vergini      = 0
    scartate_attive     = 0

    stats_quarti  = []
    stats_quinti  = []
    stats_sesti   = []

    for tripla in combinations(range(1, 91), 3):
        if len(sestine_trovate) >= max_sestine:
            break
        if triple_vergini >= max_triple_base:
            break

        triple_esaminate += 1

        # La tripla base deve essere vergine
        if tripla in triple_viste:
            continue

        triple_vergini += 1

        # STEP 2: testa 4° numero (1-90)
        quarti = trova_quarti_validi(tripla, triple_viste)
        stats_quarti.append(len(quarti))

        if not quarti:
            continue

        # STEP 3: per ogni quadrupla valida testa 5°
        for d in quarti:
            if len(sestine_trovate) >= max_sestine:
                break

            quadrupla = tuple(sorted(tripla + (d,)))
            quinti = trova_quinti_validi(quadrupla, triple_viste)
            stats_quinti.append(len(quinti))

            if not quinti:
                continue

            # STEP 4: per ogni quintupla valida testa 6°
            for e in quinti:
                if len(sestine_trovate) >= max_sestine:
                    break

                quintupla = tuple(sorted(quadrupla + (e,)))
                sesti = trova_sesti_validi(quintupla, triple_viste)
                stats_sesti.append(len(sesti))

                if not sesti:
                    continue

                # STEP 5: costruisci sestine e applica filtro
                for f in sesti:
                    if len(sestine_trovate) >= max_sestine:
                        break

                    sestina = tuple(sorted(quintupla + (f,)))

                    # Filtro: nessuna tripla nelle attive recenti
                    passa = True
                    for t in combinations(sestina, 3):
                        if t in triple_attive:
                            passa = False
                            scartate_attive += 1
                            break

                    if passa:
                        sestine_trovate.append(list(sestina))

        if triple_vergini % 10 == 0:
            print(f"  Triple vergini: {triple_vergini:,} "
                  f"| Esaminate: {triple_esaminate:,} "
                  f"| Sestine: {len(sestine_trovate):,}")

    # Statistiche finali
    print(f"\n  === Risultati Ricerca Sistematica ===")
    print(f"  Triple esaminate:          {triple_esaminate:,}")
    print(f"  Triple vergini trovate:    {triple_vergini:,}")
    print(f"  Sestine generate:          {len(sestine_trovate):,}")
    print(f"  Scartate (triple attive):  {scartate_attive:,}")

    if stats_quarti:
        print(f"  4° validi per tripla:      "
              f"media={np.mean(stats_quarti):.1f} "
              f"min={min(stats_quarti)} "
              f"max={max(stats_quarti)}")
    if stats_quinti:
        print(f"  5° validi per quadrupla:   "
              f"media={np.mean(stats_quinti):.1f} "
              f"min={min(stats_quinti)} "
              f"max={max(stats_quinti)}")
    if stats_sesti:
        print(f"  6° validi per quintupla:   "
              f"media={np.mean(stats_sesti):.1f} "
              f"min={min(stats_sesti)} "
              f"max={max(stats_sesti)}")

    return sestine_trovate
