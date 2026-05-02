"""
SSAS - Stochastic Structure Analysis System
Popola le tabelle analitiche su Supabase.
"""
import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from supabase import create_client
from moduli.fingerprint import build_fingerprint
from moduli.costanti import calcola_costanti
from moduli.mappa import calcola_mappa
from moduli.generatore import (
    carica_storico,
    carica_triple_attive,
    carica_mappa_occupazione,
    ricerca_sistematica
)

# ── Credenziali ──────────────────────────────────────────
URL = os.environ.get("URL_SUPABASE", "")
KEY = os.environ.get("KEY_SUPABASE", "")

if not URL or not KEY:
    try:
        with open(".env") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                os.environ[k] = v
        URL = os.environ["URL_SUPABASE"]
        KEY = os.environ["KEY_SUPABASE"]
    except Exception:
        print("ERRORE: credenziali non trovate.")
        sys.exit(1)

supabase = create_client(URL, KEY)

# ── Carica estrazioni (tutte) ────────────────────────────
print("Caricamento estrazioni...")

res = supabase.table("estrazioni")\
    .select("id, data_estrazione, n1, n2, n3, n4, n5, n6")\
    .order("data_estrazione", desc=False)\
    .limit(10000)\
    .execute()

df = pd.DataFrame(res.data)
df['data_estrazione'] = pd.to_datetime(df['data_estrazione'])
df = df.sort_values('data_estrazione').reset_index(drop=True)
print(f"Caricate {len(df)} estrazioni totali.")

# ── STEP 1: Fingerprint ──────────────────────────────────
print("\n[1/4] Calcolo fingerprint...")
fingerprints = []
sets_storici = []

for i, row in df.iterrows():
    fp, nums_set = build_fingerprint(row, sets_storici)
    fingerprints.append(fp)
    sets_storici.append(nums_set)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(df)}...")

BATCH = 200
for i in range(0, len(fingerprints), BATCH):
    supabase.table("fingerprint_estrazioni")\
        .upsert(fingerprints[i:i+BATCH], on_conflict="estrazione_id")\
        .execute()
print(f"  Salvati {len(fingerprints)} fingerprint.")

# ── STEP 2: Costanti ─────────────────────────────────────
print("\n[2/4] Calcolo costanti sistema...")
fp_df   = pd.DataFrame(fingerprints)
records = calcola_costanti(fp_df)

for r in records:
    print(f"  {r['nome']:25s} "
          f"media={r['valore_medio']:.4f} "
          f"std={r['std_dev']:.4f}"
          + (f" sigma={r['sigma_da_random']:.2f}"
             if r['sigma_da_random'] else ""))

supabase.table("costanti_sistema")\
    .upsert(records, on_conflict="nome")\
    .execute()
print("  Costanti salvate.")

# ── STEP 3: Mappa occupazione ────────────────────────────
print("\n[3/4] Calcolo mappa occupazione 1-90...")
mappa = calcola_mappa(df)

BATCH = 30
for i in range(0, len(mappa), BATCH):
    supabase.table("mappa_occupazione")\
        .upsert(mappa[i:i+BATCH], on_conflict="numero")\
        .execute()
print("  Mappa completata.")

# ── STEP 4: Ricerca con Filtri a Cascata ─────────────────
print("\n[4/4] Ricerca sistematica con filtri a cascata...")

from moduli.generatore import (
    carica_storico,
    carica_triple_attive,
    carica_mappa_occupazione,
    ricerca_sistematica
)

storico_np, figure_viste = carica_storico(supabase)
triple_attive            = carica_triple_attive(
                               supabase, n_estrazioni=50)
mappa_z                  = carica_mappa_occupazione(supabase)

sestine = ricerca_sistematica(
    storico_np   = storico_np,
    figure_viste = figure_viste,
    triple_attive= triple_attive,
    mappa_z      = mappa_z,
    n_campioni   = 5000000,
    max_sestine  = 10000
)

run_id = int(time.time())
print(f"\n  Salvataggio {len(sestine)} sestine (run_id={run_id})...")

records = []
for s in sestine:
    records.append({
        "n1": s[0], "n2": s[1], "n3": s[2],
        "n4": s[3], "n5": s[4], "n6": s[5],
        "passa_gap":     True,
        "passa_somma":   True,
        "score_armonia": 1.0,
        "run_id":        run_id
    })

BATCH = 200
for i in range(0, len(records), BATCH):
    supabase.table("combinazioni_candidate")\
        .insert(records[i:i+BATCH])\
        .execute()

print(f"  Salvate {len(sestine)} sestine candidate.")
print("\n=== ANALISI COMPLETATA ===")
print("Supabase aggiornato. Lancia dashboard.py su Streamlit.")

# ── STEP 5: Wyckoff ──────────────────────────────────────
print("\n[5/6] Analisi Wyckoff...")

from moduli.wyckoff import esegui_wyckoff
from moduli.compensazione import esegui_compensazione

wyckoff_id, stato, df_zone, df_cicli = esegui_wyckoff(
    df_raw = df,
    client = supabase
)

# ── STEP 6: Compensazione + Sestine Wyckoff ──────────────
print("\n[6/6] Compensazione numerica e generazione sestine...")

pool_numeri = esegui_compensazione(
    df_raw     = df,
    wyckoff_id = wyckoff_id,
    stato      = stato,
    df_zone    = df_zone,
    df_cicli   = df_cicli,
    client     = supabase
)

# Genera sestine dal pool Wyckoff
from moduli.generatore import (
    carica_storico,
    carica_triple_attive,
    carica_mappa_occupazione,
    ricerca_sistematica
)

storico_np, figure_viste = carica_storico(supabase)
triple_attive            = carica_triple_attive(
    supabase, n_estrazioni=50)
mappa_z                  = carica_mappa_occupazione(supabase)

# Adatta ricerca_sistematica per usare pool ristretto
import random

def ricerca_su_pool(pool, storico_np, triple_attive,
                    mappa_z, n_campioni=2000000, max_sestine=5000):
    """Motore 3 sul pool Wyckoff."""
    from moduli.generatore import (
        check_overlap, check_strutturali,
        check_triple_attive, SOMMA_MIN, SOMMA_MAX
    )
    import numpy as np

    # Adatta limiti somma alla fascia Wyckoff
    fascia_min = stato['fascia_min']
    fascia_max = stato['fascia_max']

    trovate = []
    scarti  = {"overlap":0,"strutturali":0,
               "triple":0,"fuori_fascia":0}

    for _ in range(n_campioni):
        if len(trovate) >= max_sestine:
            break
        if len(pool) < 6:
            break

        sestina    = sorted(random.sample(pool, 6))
        sestina_t  = tuple(sestina)
        sestina_set= set(sestina)
        somma      = sum(sestina)

        # Filtro fascia Wyckoff
        if not (fascia_min <= somma <= fascia_max):
            scarti["fuori_fascia"] += 1
            continue

        # Filtro overlap
        if not check_overlap(sestina_set, storico_np):
            scarti["overlap"] += 1
            continue

        # Filtri strutturali
        passa, _ = check_strutturali(sestina, mappa_z)
        if not passa:
            scarti["strutturali"] += 1
            continue

        # Triple attive
        if not check_triple_attive(sestina_t, triple_attive):
            scarti["triple"] += 1
            continue

        trovate.append(sestina)

    print(f"  [Motore3-Wyckoff] Trovate: {len(trovate)}")
    print(f"  [Motore3-Wyckoff] Scarti: {scarti}")
    return trovate

sestine_wyckoff = ricerca_su_pool(
    pool        = pool_numeri,
    storico_np  = storico_np,
    triple_attive=triple_attive,
    mappa_z     = mappa_z,
    n_campioni  = 2000000,
    max_sestine = 5000
)

# Salva sestine Wyckoff con tag specifico
run_id_w = int(time.time()) + 1
records_w = []
for s in sestine_wyckoff:
    records_w.append({
        "n1": s[0], "n2": s[1], "n3": s[2],
        "n4": s[3], "n5": s[4], "n6": s[5],
        "passa_gap":     True,
        "passa_somma":   True,
        "score_armonia": 2.0,  # tag Wyckoff
        "run_id":        run_id_w
    })

BATCH = 200
for i in range(0, len(records_w), BATCH):
    supabase.table("combinazioni_candidate")\
        .insert(records_w[i:i+BATCH]).execute()

print(f"  Salvate {len(sestine_wyckoff)} sestine Wyckoff "
      f"(run_id={run_id_w})")
print("\n=== ANALISI COMPLETATA ===")
