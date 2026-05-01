"""
SSAS - Stochastic Structure Analysis System
Popola le tabelle analitiche su Supabase.
Eseguire una volta (o quando arrivano nuove estrazioni).
"""
import os
import sys
import pandas as pd
import numpy as np

# Permette import da moduli/ anche se eseguito da root
sys.path.insert(0, os.path.dirname(__file__))

from supabase import create_client
from moduli.fingerprint import build_fingerprint
from moduli.costanti import calcola_costanti
from moduli.mappa import calcola_mappa

# ── Credenziali ──────────────────────────────────────────
URL = os.environ.get("URL_SUPABASE", "")
KEY = os.environ.get("KEY_SUPABASE", "")

if not URL or not KEY:
    # Fallback: leggi da file .env locale
    try:
        with open(".env") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                os.environ[k] = v
        URL = os.environ["URL_SUPABASE"]
        KEY = os.environ["KEY_SUPABASE"]
    except Exception:
        print("ERRORE: credenziali non trovate.")
        print("Crea un file .env con URL_SUPABASE e KEY_SUPABASE")
        sys.exit(1)

supabase = create_client(URL, KEY)

# ── Carica estrazioni ────────────────────────────────────
print("Caricamento estrazioni...")
res = supabase.table("estrazioni")\
    .select("id, data_estrazione, n1, n2, n3, n4, n5, n6")\
    .order("data_estrazione", desc=False)\
    .execute()

df = pd.DataFrame(res.data)
df['data_estrazione'] = pd.to_datetime(df['data_estrazione'])
df = df.sort_values('data_estrazione').reset_index(drop=True)
print(f"Caricate {len(df)} estrazioni.")

# ── STEP 1: Fingerprint ──────────────────────────────────
print("\n[1/3] Calcolo fingerprint...")
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
print("\n[2/3] Calcolo costanti sistema...")
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
print("\n[3/3] Calcolo mappa occupazione 1-90...")
mappa = calcola_mappa(df)

BATCH = 30
for i in range(0, len(mappa), BATCH):
    supabase.table("mappa_occupazione")\
        .upsert(mappa[i:i+BATCH], on_conflict="numero")\
        .execute()
print("  Mappa completata.")

print("\n=== ANALISI COMPLETATA ===")
print("Supabase aggiornato. Lancia dashboard.py su Streamlit.")
