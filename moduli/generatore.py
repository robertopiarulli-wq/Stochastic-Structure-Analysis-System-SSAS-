"""
Motore generativo basato su quadruple vergini.
Una sestina è valida solo se tutte le sue 15 quadruple
interne non sono mai apparse storicamente.
"""
import random
from itertools import combinations
import numpy as np

def carica_quadruple_viste(client):
    """
    Carica tutte le quadruple storiche in un set Python.
    O(1) lookup per ogni verifica.
    """
    print("  Caricamento quadruple storiche...")
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

    print(f"  Quadruple storiche caricate: {len(quadruple_viste)}")
    return quadruple_viste

def carica_triple_attive(client, n_estrazioni=13):
    """
    Carica le triple delle ultime N estrazioni.
    Filtro secondario.
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

    print(f"  Triple attive (ultimi {n_estrazioni} concorsi): {len(triple_attive)}")
    return triple_attive

def sestina_e_vergine(sestina, quadruple_viste):
    """
    Verifica che tutte le 15 quadruple interne
    alla sestina non siano mai apparse storicamente.
    """
    for q in combinations(sestina, 4):
        if q in quadruple_viste:
            return False
    return True

def sestina_passa_triple(sestina, triple_attive):
    """
    Verifica che nessuna tripla interna
    sia nelle triple attive degli ultimi 30gg.
    """
    for t in combinations(sestina, 3):
        if t in triple_attive:
            return False
    return True

def estendi_con_quinto(quadrupla, quadruple_viste):
    """
    Dato [a,b,c,d] vergine, trova tutti i numeri e
    tali che le 4 nuove quadruple formate siano anch'esse vergini.
    
    Aggiungendo e a [a,b,c,d]:
    → 4 nuove quadruple: C(4,3) combinazioni di [a,b,c,d] + e
    """
    candidati = []
    numeri_usati = set(quadrupla)

    for e in range(1, 91):
        if e in numeri_usati:
            continue
        # Le 4 nuove quadruple formate
        valido = True
        for base in combinations(quadrupla, 3):
            nuova_q = tuple(sorted(base + (e,)))
            if nuova_q in quadruple_viste:
                valido = False
                break
        if valido:
            candidati.append(e)

    return candidati

def estendi_con_sesto(quintupla, quadruple_viste):
    """
    Dato [a,b,c,d,e] con tutte quadruple vergini,
    trova tutti i numeri f tali che le 10 nuove
    quadruple formate siano anch'esse vergini.
    
    Aggiungendo f a [a,b,c,d,e]:
    → 10 nuove quadruple: C(5,3) combinazioni di [a,b,c,d,e] + f
    """
    candidati = []
    numeri_usati = set(quintupla)

    for f in range(1, 91):
        if f in numeri_usati:
            continue
        valido = True
        for base in combinations(quintupla, 3):
            nuova_q = tuple(sorted(base + (f,)))
            if nuova_q in quadruple_viste:
                valido = False
                break
        if valido:
            candidati.append(f)

    return candidati

def genera_sestine(
    quadruple_viste,
    triple_attive,
    n_quadruple_base=5000,
    max_sestine=10000,
    seed=None
):
    """
    Motore generativo principale.
    
    1. Campiona quadruple vergini casuali come base
    2. Estende con 5° numero (verificando virginity)
    3. Estende con 6° numero (verificando virginity)
    4. Applica filtro triple attive
    5. Restituisce sestine candidate ordinate
    """
    if seed:
        random.seed(seed)

    print(f"\n  Generazione sestine da quadruple vergini...")
    print(f"  Base: {n_quadruple_base} quadruple di partenza")

    # Genera quadruple vergini casuali come punto di partenza
    # Invece di iterare tutte le 2.4M, le campionam casualmente
    quadruple_base = set()
    tentativi = 0
    max_tentativi = n_quadruple_base * 20

    while len(quadruple_base) < n_quadruple_base and tentativi < max_tentativi:
        q = tuple(sorted(random.sample(range(1, 91), 4)))
        if q not in quadruple_viste:
            quadruple_base.add(q)
        tentativi += 1

    print(f"  Quadruple vergini trovate: {len(quadruple_base)} "
          f"(su {tentativi} tentativi)")

    sestine_trovate = []
    scartate_triple = 0
    senza_quinto = 0
    senza_sesto = 0

    for idx, quad in enumerate(quadruple_base):
        if len(sestine_trovate) >= max_sestine:
            break

        # Step 1: trova candidati per il 5° numero
        candidati_e = estendi_con_quinto(quad, quadruple_viste)

        if not candidati_e:
            senza_quinto += 1
            continue

        # Step 2: per ogni candidato e, trova il 6° numero
        random.shuffle(candidati_e)
        for e in candidati_e:
            quintupla = tuple(sorted(quad + (e,)))
            candidati_f = estendi_con_sesto(quintupla, quadruple_viste)

            if not candidati_f:
                senza_sesto += 1
                continue

            # Step 3: costruisci sestine e applica filtro triple
            random.shuffle(candidati_f)
            for f in candidati_f:
                sestina = tuple(sorted(quintupla + (f,)))

                # Filtro secondario: triple attive
                if not sestina_passa_triple(sestina, triple_attive):
                    scartate_triple += 1
                    continue

                sestine_trovate.append(list(sestina))

                if len(sestine_trovate) >= max_sestine:
                    break

            if len(sestine_trovate) >= max_sestine:
                break

        if (idx + 1) % 500 == 0:
            print(f"  Processate {idx+1}/{len(quadruple_base)} quadruple "
                  f"→ {len(sestine_trovate)} sestine trovate...")

    print(f"\n  Risultati generazione:")
    print(f"  - Sestine generate:        {len(sestine_trovate)}")
    print(f"  - Scartate (triple):       {scartate_triple}")
    print(f"  - Quadruple senza 5°:      {senza_quinto}")
    print(f"  - Quintuple senza 6°:      {senza_sesto}")

    return sestine_trovate
