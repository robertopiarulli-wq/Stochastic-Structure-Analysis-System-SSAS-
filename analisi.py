"""
SSAS - Stochastic Structure Analysis System
Flusso: Fingerprint → Costanti → Mappa → Wyckoff
        → Compensazione → Generatore Wyckoff
"""
import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from supabase import create_client
from moduli.fingerprint import build_fingerprint
from moduli.costanti    import calcola_costanti
from moduli.mappa       import calcola_mappa
from moduli.wyckoff     import esegui_wyckoff
from moduli.compensazione import esegui_compensazione
from moduli.generatore  import (
    carica_storico,
    carica_triple_attive,
    carica_mappa_occupazione,
    ricerca_su_pool
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

# ── Carica estrazioni ────────────────────────────────────
print("Caricamento estrazioni...")
res = supabase.table("estrazioni")\
    .select("id,data_estrazione,n1,n2,n3,n4,n5,n6")\
    .order("data_estrazione", desc=False)\
    .limit(10000)\
    .execute()

df = pd.DataFrame(res.data)
df['data_estrazione'] = pd.to_datetime(df['data_estrazione'])
df = df.sort_values('data_estrazione').reset_index(drop=True)
print(f"Caricate {len(df)} estrazioni totali.")

# ── STEP 1: Fingerprint ──────────────────────────────────
print("\n[1/5] Calcolo fingerprint...")
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
        .upsert(fingerprints[i:i+BATCH],
                on_conflict="estrazione_id")\
        .execute()
print(f"  Salvati {len(fingerprints)} fingerprint.")

# ── STEP 2: Costanti ─────────────────────────────────────
print("\n[2/5] Calcolo costanti sistema...")
fp_df   = pd.DataFrame(fingerprints)
records = calcola_costanti(fp_df)

for r in records:
    print(f"  {r['nome']:25s} "
          f"media={r['valore_medio']:.4f} "
          f"std={r['std_dev']:.4f}"
          + (f" sigma={r['sigma_da_random']:.2f}"
             if r['sigma_da_random'] else ""))

supabase.table("costanti_sistema")\
    .upsert(records, on_conflict="nome").execute()
print("  Costanti salvate.")

# ── STEP 3: Mappa occupazione ────────────────────────────
print("\n[3/5] Calcolo mappa occupazione 1-90...")
mappa = calcola_mappa(df)

BATCH = 30
for i in range(0, len(mappa), BATCH):
    supabase.table("mappa_occupazione")\
        .upsert(mappa[i:i+BATCH], on_conflict="numero")\
        .execute()
print("  Mappa completata.")

# ── STEP 4: Wyckoff ──────────────────────────────────────
print("\n[4/5] Analisi Wyckoff...")
wyckoff_id, stato, df_zone, df_cicli = esegui_wyckoff(
    df_raw = df,
    client = supabase
)

# ── STEP 5: Compensazione + Generatore ───────────────────
print("\n[5/5] Compensazione e generazione sestine...")

pool_numeri = esegui_compensazione(
    df_raw     = df,
    wyckoff_id = wyckoff_id,
    stato      = stato,
    df_zone    = df_zone,
    df_cicli   = df_cicli,
    client     = supabase
)

# Carica strutture per filtri
storico_np, figure_viste = carica_storico(supabase)
triple_attive            = carica_triple_attive(
    supabase, n_estrazioni=50)
mappa_z                  = carica_mappa_occupazione(supabase)

# Genera sestine dal pool Wyckoff
sestine = ricerca_su_pool(
    pool          = pool_numeri,
    storico_np    = storico_np,
    figure_viste  = figure_viste,
    triple_attive = triple_attive,
    mappa_z       = mappa_z,
    fascia_min    = stato['fascia_min'],
    fascia_max    = stato['fascia_max'],
    n_campioni    = 3000000,
    max_sestine   = 5000
)

# Salva
run_id = int(time.time())
print(f"\n  Salvataggio {len(sestine)} sestine "
      f"(run_id={run_id})...")

records = []
for s in sestine:
    records.append({
        "n1": s[0], "n2": s[1], "n3": s[2],
        "n4": s[3], "n5": s[4], "n6": s[5],
        "passa_gap":     True,
        "passa_somma":   True,
        "score_armonia": 2.0,
        "run_id":        run_id
    })

BATCH = 200
for i in range(0, len(records), BATCH):
    supabase.table("combinazioni_candidate")\
        .insert(records[i:i+BATCH]).execute()

print(f"  Salvate {len(sestine)} sestine candidate.")
print("\n=== ANALISI COMPLETATA ===")

# ── REPORT NUMERI CANDIDATE ──────────────────────────────
if sestine:
    import pandas as pd
    from collections import Counter

    tutti_numeri = [n for s in sestine for n in s]
    freq         = Counter(tutti_numeri)
    n_sestine    = len(sestine)

    print(f"\n  === Composizione {n_sestine} sestine candidate ===")
    print(f"  {'Num':>4} {'Presenze':>9} {'%':>7} {'Saturazione':>12}")
    print(f"  {'-'*38}")

    for numero in sorted(freq.keys()):
        presenze = freq[numero]
        pct      = presenze * 100 / n_sestine
        if pct >= 15:
            sat = "ALTA   🔴"
        elif pct >= 8:
            sat = "MEDIA  🟡"
        else:
            sat = "BASSA  🟢"
        print(f"  {numero:>4} {presenze:>9} {pct:>6.1f}% {sat}")

    # Salva in Supabase
    records_freq = []
    for numero, presenze in freq.items():
        records_freq.append({
            "run_id":    run_id,
            "numero":    numero,
            "presenze":  presenze,
            "pct":       round(presenze * 100 / n_sestine, 2),
        })

    # Crea tabella se non esiste (SQL su Supabase):
    # CREATE TABLE IF NOT EXISTS candidate_frequenze (
    #   id BIGSERIAL PRIMARY KEY,
    #   run_id BIGINT,
    #   numero INTEGER,
    #   presenze INTEGER,
    #   pct FLOAT
    # );
    try:
        supabase.table("candidate_frequenze")\
            .insert(records_freq).execute()
        print(f"\n  Frequenze salvate su Supabase.")
    except Exception as e:
        print(f"\n  (tabella candidate_frequenze non ancora creata: {e})")
