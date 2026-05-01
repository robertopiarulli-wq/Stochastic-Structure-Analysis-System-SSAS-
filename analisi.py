import numpy as np
import pandas as pd
from supabase import create_client
from datetime import datetime
from itertools import combinations
from scipy import stats

URL = "la-tua-url-supabase"
KEY = "la-tua-anon-key"
supabase = create_client(URL, KEY)

# ============================================
# CARICA TUTTE LE ESTRAZIONI
# ============================================
print("Caricamento estrazioni...")
res = supabase.table("estrazioni")\
    .select("id, data_estrazione, n1, n2, n3, n4, n5, n6")\
    .order("data_estrazione", desc=False)\
    .execute()

df = pd.DataFrame(res.data)
df['data_estrazione'] = pd.to_datetime(df['data_estrazione'])
df = df.sort_values('data_estrazione').reset_index(drop=True)
print(f"Caricate {len(df)} estrazioni dal {df['data_estrazione'].min().date()} al {df['data_estrazione'].max().date()}")

# ============================================
# FUNZIONI DI CALCOLO
# ============================================
def get_numeri(row):
    return sorted([row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']])

def calcola_gap(numeri):
    return [numeri[i+1] - numeri[i] for i in range(len(numeri)-1)]

def calcola_entropia(gaps):
    """Entropia di Shannon normalizzata sui gap"""
    gaps = np.array(gaps)
    totale = gaps.sum()
    if totale == 0:
        return 0.0
    p = gaps / totale
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    # Normalizza su log2(5) = max entropia con 5 gap
    return float(H / np.log2(5))

def calcola_spacing_ratio(gaps):
    """
    Spacing ratio medio di Wigner-Dyson
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    Poisson puro -> 0.386
    GOE (correlazioni) -> 0.536
    """
    if len(gaps) < 2:
        return 0.0
    ratios = []
    for i in range(len(gaps)-1):
        s1, s2 = gaps[i], gaps[i+1]
        if max(s1, s2) == 0:
            continue
        ratios.append(min(s1, s2) / max(s1, s2))
    return float(np.mean(ratios)) if ratios else 0.0

def calcola_decadi(numeri):
    """Quante decadi distinte sono coperte (1-10, 11-20, ..., 81-90)"""
    decadi = set()
    for n in numeri:
        decadi.add((n - 1) // 10)
    return len(decadi)

def calcola_consecutivi(numeri):
    """Restituisce (n_coppie_consecutive, max_run_consecutivo)"""
    coppie = 0
    max_run = 1
    run_corrente = 1
    for i in range(len(numeri)-1):
        if numeri[i+1] - numeri[i] == 1:
            coppie += 1
            run_corrente += 1
            max_run = max(max_run, run_corrente)
        else:
            run_corrente = 1
    return coppie, max_run

def calcola_overlap(numeri_set, numeri_prec_set):
    return len(numeri_set & numeri_prec_set)

# ============================================
# STEP 1: FINGERPRINT DI OGNI ESTRAZIONE
# ============================================
print("\nCalcolo fingerprint estrazioni...")

fingerprints = []
sets_storici = []

for i, row in df.iterrows():
    numeri = get_numeri(row)
    nums_set = set(numeri)
    gaps = calcola_gap(numeri)
    gap_arr = np.array(gaps)

    mu = gap_arr.mean()
    std = gap_arr.std()
    cv = float(std / mu) if mu != 0 else 0.0
    gap_ratio = float(gap_arr.max() / gap_arr.min()) if gap_arr.min() != 0 else float(gap_arr.max())

    coppie, max_run = calcola_consecutivi(numeri)

    # Overlap con precedenti
    ov1 = calcola_overlap(nums_set, sets_storici[-1]) if len(sets_storici) >= 1 else 0
    ov3 = calcola_overlap(nums_set, sets_storici[-3]) if len(sets_storici) >= 3 else 0
    ov7 = calcola_overlap(nums_set, sets_storici[-7]) if len(sets_storici) >= 7 else 0

    sets_storici.append(nums_set)

    fp = {
        "estrazione_id":        int(row['id']),
        "data_estrazione":      row['data_estrazione'].date().isoformat(),
        "somma":                int(sum(numeri)),
        "n_pari":               int(sum(1 for n in numeri if n % 2 == 0)),
        "n_dispari":            int(sum(1 for n in numeri if n % 2 != 0)),
        "range_totale":         int(numeri[-1] - numeri[0]),
        "decadi_coperte":       int(calcola_decadi(numeri)),
        "gap_min":              int(gap_arr.min()),
        "gap_max":              int(gap_arr.max()),
        "gap_medio":            float(round(mu, 4)),
        "gap_std":              float(round(std, 4)),
        "cv_gap":               float(round(cv, 4)),
        "gap_ratio":            float(round(gap_ratio, 4)),
        "n_coppie_consecutive": int(coppie),
        "consecutivi_max":      int(max_run),
        "entropia_gap":         float(round(calcola_entropia(gaps), 4)),
        "spacing_ratio_medio":  float(round(calcola_spacing_ratio(gaps), 4)),
        "overlap_lag1":         int(ov1),
        "overlap_lag3":         int(ov3),
        "overlap_lag7":         int(ov7),
    }
    fingerprints.append(fp)

    if (i+1) % 500 == 0:
        print(f"  Processate {i+1}/{len(df)} estrazioni...")

# Inserimento batch fingerprint
print("Salvataggio fingerprint...")
BATCH = 200
for i in range(0, len(fingerprints), BATCH):
    batch = fingerprints[i:i+BATCH]
    supabase.table("fingerprint_estrazioni").upsert(
        batch, on_conflict="estrazione_id"
    ).execute()
print(f"Salvati {len(fingerprints)} fingerprint.")

# ============================================
# STEP 2: CALCOLO COSTANTI DEL SISTEMA
# ============================================
print("\nCalcolo costanti sistema...")

fp_df = pd.DataFrame(fingerprints)

# Valori teorici per estrazioni uniformi i.i.d. su 1-90
# Somma teorica: 6 * 45.5 = 273
# Gap medio teorico: 84/7 = 12 (range medio diviso per 7 slot)
costanti_da_calcolare = [
    {
        "nome": "somma",
        "descrizione": "Somma dei 6 numeri estratti",
        "colonna": "somma",
        "teorico": 6 * 45.5  # = 273
    },
    {
        "nome": "cv_gap",
        "descrizione": "Coefficiente di variazione dei gap (misura disordine interno)",
        "colonna": "cv_gap",
        "teorico": None
    },
    {
        "nome": "spacing_ratio",
        "descrizione": "Spacing ratio Wigner-Dyson (Poisson=0.386, GOE=0.536)",
        "colonna": "spacing_ratio_medio",
        "teorico": 0.386
    },
    {
        "nome": "entropia_gap",
        "descrizione": "Entropia Shannon normalizzata sui gap",
        "colonna": "entropia_gap",
        "teorico": None
    },
    {
        "nome": "range_totale",
        "descrizione": "Range max-min della sestina",
        "colonna": "range_totale",
        "teorico": None
    },
    {
        "nome": "n_pari",
        "descrizione": "Numero di valori pari nella sestina",
        "colonna": "n_pari",
        "teorico": 3.0  # atteso uniforme
    },
    {
        "nome": "decadi_coperte",
        "descrizione": "Quante decadi (1-10, 11-20...) sono presenti",
        "colonna": "decadi_coperte",
        "teorico": None
    },
    {
        "nome": "gap_ratio",
        "descrizione": "Rapporto gap_max/gap_min: asimmetria della distribuzione",
        "colonna": "gap_ratio",
        "teorico": None
    },
    {
        "nome": "overlap_lag1",
        "descrizione": "Numeri in comune con estrazione precedente",
        "colonna": "overlap_lag1",
        "teorico": 6*(6/90)  # atteso ~0.4
    },
    {
        "nome": "n_coppie_consecutive",
        "descrizione": "Coppie di numeri consecutivi nella sestina",
        "colonna": "n_coppie_consecutive",
        "teorico": None
    },
]

costanti_records = []
for c in costanti_da_calcolare:
    col = fp_df[c["colonna"]].dropna()
    n = len(col)

    media = float(col.mean())
    std = float(col.std())
    cv = float(std / media) if media != 0 else None

    teorico = c["teorico"]
    if teorico is not None and std > 0:
        sigma = float((media - teorico) / std)
    else:
        sigma = None

    record = {
        "nome":                     c["nome"],
        "descrizione":              c["descrizione"],
        "valore_medio":             round(media, 4),
        "std_dev":                  round(std, 4),
        "mediana":                  round(float(col.median()), 4),
        "percentile_5":             round(float(col.quantile(0.05)), 4),
        "percentile_10":            round(float(col.quantile(0.10)), 4),
        "percentile_25":            round(float(col.quantile(0.25)), 4),
        "percentile_75":            round(float(col.quantile(0.75)), 4),
        "percentile_90":            round(float(col.quantile(0.90)), 4),
        "percentile_95":            round(float(col.quantile(0.95)), 4),
        "cv_storico":               round(cv, 4) if cv else None,
        "valore_atteso_teorico":    teorico,
        "sigma_da_random":          round(sigma, 3) if sigma else None,
        "n_campioni":               n,
    }
    costanti_records.append(record)
    print(f"  {c['nome']:25s} media={media:.4f} std={std:.4f}" + 
          (f" sigma_vs_random={sigma:.2f}" if sigma else ""))

supabase.table("costanti_sistema").upsert(
    costanti_records, on_conflict="nome"
).execute()
print("Costanti salvate.")

# ============================================
# STEP 3: MAPPA OCCUPAZIONE 1-90
# ============================================
print("\nCalcolo mappa occupazione...")

n_estrazioni = len(df)
freq_attesa = 6.0 / 90.0  # per estrazione

for numero in range(1, 91):
    # Frequenza assoluta
    freq_abs = int(
        df[['n1','n2','n3','n4','n5','n6']]
        .apply(lambda r: numero in r.values, axis=1)
        .sum()
    )
    freq_rel = freq_abs / n_estrazioni

    # Z-score vs atteso
    atteso = n_estrazioni * freq_attesa
    std_binom = np.sqrt(n_estrazioni * freq_attesa * (1 - freq_attesa))
    z = float((freq_abs - atteso) / std_binom)

    # Ritardo: quante estrazioni fa è uscito
    estratto_in = df[['n1','n2','n3','n4','n5','n6']]\
        .apply(lambda r: numero in r.values, axis=1)
    
    ultimo_idx = estratto_in[estratto_in].index.max()
    ritardo = int(n_estrazioni - 1 - ultimo_idx) if pd.notna(ultimo_idx) else n_estrazioni
    ultimo_data = df.loc[ultimo_idx, 'data_estrazione'].date().isoformat() if pd.notna(ultimo_idx) else None

    # Ritardo medio e max
    idxs = estratto_in[estratto_in].index.tolist()
    if len(idxs) > 1:
        ritardi_storici = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
        rit_medio = float(np.mean(ritardi_storici))
        rit_max = int(np.max(ritardi_storici))
    else:
        rit_medio = float(n_estrazioni)
        rit_max = int(n_estrazioni)

    # Densità locale: media freq dei vicini n-2,n-1,n+1,n+2
    vicini = [v for v in [numero-2, numero-1, numero+1, numero+2] if 1 <= v <= 90]
    
    def freq_num(n):
        return df[['n1','n2','n3','n4','n5','n6']]\
            .apply(lambda r: n in r.values, axis=1).sum() / n_estrazioni
    
    densita = float(np.mean([freq_num(v) for v in vicini]))

    supabase.table("mappa_occupazione").update({
        "freq_assoluta":    freq_abs,
        "freq_relativa":    round(freq_rel, 6),
        "z_score":          round(z, 4),
        "ritardo_attuale":  ritardo,
        "ritardo_medio":    round(rit_medio, 2),
        "ritardo_max":      rit_max,
        "ultimo_estratto":  ultimo_data,
        "densita_locale":   round(densita, 6),
        "aggiornato_al":    datetime.now().isoformat(),
    }).eq("numero", numero).execute()

    if numero % 10 == 0:
        print(f"  Numero {numero}/90 completato...")

print("Mappa occupazione completata.")
print("\n=== ANALISI COMPLETATA ===")
print("Esegui la view v_costanti su Supabase per vedere le costanti rilevate.")
