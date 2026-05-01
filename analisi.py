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
    carica_triple_viste,
    carica_figure_gap,
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

# ── STEP 4: Ricerca Sistematica ──────────────────────────
print("\n[4/4] Ricerca sistematica quadruple vergini...")

from moduli.generatore import (
    carica_sottoinsiemi,
    carica_figure_gap,
    carica_mappa_occupazione,
    ricerca_sistematica
)

quadruple_viste, quintuple_viste, sestine_viste = \
    carica_sottoinsiemi(supabase)
figure_viste = carica_figure_gap(supabase)
mappa_z      = carica_mappa_occupazione(supabase)

sestine = ricerca_sistematica(
    quadruple_viste    = quadruple_viste,
    quintuple_viste    = quintuple_viste,
    sestine_viste      = sestine_viste,
    figure_viste       = figure_viste,
    mappa_z            = mappa_z,
    max_quadruple_base = 500,
    max_sestine        = 10000
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
