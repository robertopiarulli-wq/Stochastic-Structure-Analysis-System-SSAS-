"""
SSAS - Motore di Ricerca Sistematica
Approccio deterministico: non genera sestine casuali
ma mappa sistematicamente quali numeri possono
estendere ogni quadrupla vergine mantenendo
la purezza strutturale sull'intero database storico.
"""
from itertools import combinations

def carica_quadruple_viste(client):
    """
    Carica tutte le quadruple storiche in un set Python.
    Lookup O(1) per ogni verifica.
    """
    print("  Caricamento quadruple storiche (intero database)...")
    res = client.table("estrazioni")\
        .select("n1,n2,n3,n4,n5,n6")\
        .limit(10000)\
        .execute()

    quadruple_viste = set()
    for row in res.data:
        numeri = sorted([
            row['n1'], row['n2'], row['n3'],
            row['n4'], row['n5'], row['n6']
        ])
        for q in combinations(numeri, 4):
            quadruple_viste.add(q)

    print(f"  Quadruple viste storicamente: {len(quadruple_viste)}")
    return quadruple_viste

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

def trova_quinte_valide(quadrupla, quadruple_viste):
    """
    Testa sistematicamente tutti i numeri 1-90
    come 5° elemento della quadrupla.
    
    Un numero e è valido se tutte le 4 nuove quadruple
    formate con i sottoinsiemi da 3 di [a,b,c,d]+e
    non sono mai apparse nel database storico.
    
    Aggiungendo e a [a,b,c,d]:
    le nuove quadruple sono C([a,b,c,d],3) + e = 4 quadruple
    """
    numeri_usati = set(quadrupla)
    validi = []

    for e in range(1, 91):
        if e in numeri_usati:
            continue

        # Testa le 4 nuove quadruple formate
        virgine = True
        for base in combinations(quadrupla, 3):
            nuova_q = tuple(sorted(base + (e,)))
            if nuova_q in quadruple_viste:
                virgine = False
                break

        if virgine:
            validi.append(e)

    return validi

def trova_seste_valide(quintupla, quadruple_viste):
    """
    Testa sistematicamente tutti i numeri 1-90
    come 6° elemento della quintupla.
    
    Un numero f è valido se tutte le 5 nuove quadruple
    formate con i sottoinsiemi da 3 di [a,b,c,d,e]+f
    non sono mai apparse nel database storico.
    
    Aggiungendo f a [a,b,c,d,e]:
    le nuove quadruple sono C([a,b,c,d,e],3) + f = 10 quadruple
    """
    numeri_usati = set(quintupla)
    validi = []

    for f in range(1, 91):
        if f in numeri_usati:
            continue

        # Testa le 10 nuove quadruple formate
        virgine = True
        for base in combinations(quintupla, 3):
            nuova_q = tuple(sorted(base + (f,)))
            if nuova_q in quadruple_viste:
                virgine = False
                break

        if virgine:
            validi.append(f)

    return validi

def ricerca_sistematica(
    quadruple_viste,
    triple_attive,
    max_quadruple_base=500,
    max_sestine=10000
):
    """
    Ricerca deterministica e sistematica.
    
    1. Trova quadruple vergini iterando C(90,4)
    2. Per ognuna testa sistematicamente 1-90 come 5°
    3. Per ogni quintupla valida testa 1-90 come 6°
    4. Applica filtro triple attive sulla sestina finale
    
    Non usa casualità — il risultato è riproducibile.
    """
    print(f"\n  Ricerca sistematica quadruple vergini...")

    sestine_trovate     = []
    quadruple_esaminate = 0
    quadruple_vergini   = 0
    scartate_triple     = 0

    stats_quinte = []   # quanti 5° validi per quadrupla
    stats_seste  = []   # quanti 6° validi per quintupla

    # Itera C(90,4) = 2.555.190 quadruple in ordine
    for quad in combinations(range(1, 91), 4):
        if len(sestine_trovate) >= max_sestine:
            break

        quadruple_esaminate += 1

        # La quadrupla deve essere vergine
        if quad in quadruple_viste:
            continue

        quadruple_vergini += 1

        # STEP 2: testa sistematicamente tutti i 5°
        quinte_valide = trova_quinte_valide(quad, quadruple_viste)
        stats_quinte.append(len(quinte_valide))

        if not quinte_valide:
            continue

        # STEP 3: per ogni quintupla valida testa i 6°
        for e in quinte_valide:
            if len(sestine_trovate) >= max_sestine:
                break

            quintupla = tuple(sorted(quad + (e,)))
            seste_valide = trova_seste_valide(quintupla, quadruple_viste)
            stats_seste.append(len(seste_valide))

            if not seste_valide:
                continue

            # STEP 4: costruisci sestine e applica filtro triple
            for f in seste_valide:
                if len(sestine_trovate) >= max_sestine:
                    break

                sestina = tuple(sorted(quintupla + (f,)))

                # Filtro secondario: triple attive ultimi 13 concorsi
                passa = True
                for t in combinations(sestina, 3):
                    if t in triple_attive:
                        passa = False
                        scartate_triple += 1
                        break

                if passa:
                    sestine_trovate.append(list(sestina))

        # Log ogni 1000 quadruple esaminate
        if quadruple_esaminate % 1000 == 0:
            print(f"  Esaminate {quadruple_esaminate:,} quadruple "
                  f"| Vergini: {quadruple_vergini:,} "
                  f"| Sestine: {len(sestine_trovate):,}")

        # Ferma dopo max_quadruple_base vergini
        # per non esaurire il timeout GitHub Actions
        if quadruple_vergini >= max_quadruple_base:
            break

    # Statistiche finali
    import numpy as np
    print(f"\n  === Risultati Ricerca Sistematica ===")
    print(f"  Quadruple esaminate:       {quadruple_esaminate:,}")
    print(f"  Quadruple vergini trovate: {quadruple_vergini:,}")
    print(f"  Sestine generate:          {len(sestine_trovate):,}")
    print(f"  Scartate (triple attive):  {scartate_triple:,}")

    if stats_quinte:
        print(f"  5° validi per quadrupla:   "
              f"media={np.mean(stats_quinte):.1f} "
              f"min={min(stats_quinte)} "
              f"max={max(stats_quinte)}")
    if stats_seste:
        print(f"  6° validi per quintupla:   "
              f"media={np.mean(stats_seste):.1f} "
              f"min={min(stats_seste)} "
              f"max={max(stats_seste)}")

    return sestine_trovate
