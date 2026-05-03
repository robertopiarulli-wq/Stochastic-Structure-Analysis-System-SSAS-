"""
SSAS - Motore 2: Compensazione Numerica
Data la fascia target da Wyckoff,
analizza le estrazioni storiche IN quella fascia,
scarta i numeri più frequenti (già saturi)
e usa i meno frequenti come pool di compensazione.
"""
import numpy as np
import pandas as pd

N_CICLI_FOCUS = 3
POOL_SIZE     = 30

def estrai_estrazioni_in_fascia(df, fascia_min, fascia_max):
    """Estrazioni storiche con somma nella fascia target."""
    cols = ['n1','n2','n3','n4','n5','n6']
    if 'somma' not in df.columns:
        df = df.copy()
        df['somma'] = df[cols].sum(axis=1)
    mask = ((df['somma'] >= fascia_min) &
            (df['somma'] <= fascia_max))
    return df[mask].copy()

def calcola_frequenze_numeri(df_full, df_fascia,
                              df_cicli, fascia_min, fascia_max):
    """
    Per ogni numero 1-90:
    - freq_storica: frequenza nella fascia su tutto lo storico
    - freq_recente: frequenza nella fascia negli ultimi N cicli
    - delta: freq_recente - freq_storica
    
    I numeri con freq_storica bassa sono i candidati:
    compaiono poco in quella fascia → mancano all'appello
    → per costruzione sono compatibili con la fascia
      perché calcolati SU estrazioni in quella fascia
    """
    cols  = ['n1','n2','n3','n4','n5','n6']
    n_tot = len(df_fascia)

    # Ultimi N cicli da 137
    if not df_cicli.empty:
        idx_min       = int(df_cicli['start_idx'].min())
        df_rec        = df_full[df_full.index >= idx_min].copy()
        if 'somma' not in df_rec.columns:
            df_rec['somma'] = df_rec[cols].sum(axis=1)
        df_rec_fascia = estrai_estrazioni_in_fascia(
            df_rec, fascia_min, fascia_max
        )
    else:
        df_rec_fascia = df_fascia.tail(137 * N_CICLI_FOCUS)

    n_rec = len(df_rec_fascia)

    print(f"  [Compensazione] Estrazioni in fascia (storico): "
          f"{n_tot}")
    print(f"  [Compensazione] Estrazioni in fascia (recenti): "
          f"{n_rec}")

    arr_tot = df_fascia[cols].values     if n_tot > 0 else None
    arr_rec = df_rec_fascia[cols].values if n_rec > 0 else None

    risultati = []
    for numero in range(1, 91):
        freq_s = float(np.sum(arr_tot == numero)) / n_tot \
                 if n_tot > 0 else 0.0
        freq_r = float(np.sum(arr_rec == numero)) / n_rec \
                 if n_rec > 0 else 0.0
        delta  = freq_r - freq_s

        risultati.append({
            'numero':       numero,
            'freq_storica': round(freq_s, 6),
            'freq_recente': round(freq_r, 6),
            'freq_attesa':  round(6.0/90.0, 6),
            'delta':        round(delta, 6),
        })

    return pd.DataFrame(risultati)

def seleziona_pool(df_freq, n_numeri=POOL_SIZE):
    """
    Divide 1-90 in 6 band di 15 numeri.
    Per ogni band identifica i meno frequenti nella fascia.
    Pool misto garantisce compatibilità con la somma target.
    """
    BANDS = [
        (1,  15,  "B1"),
        (16, 30,  "B2"),
        (31, 45,  "B3"),
        (46, 60,  "B4"),
        (61, 75,  "B5"),
        (76, 90,  "B6"),
    ]

    df      = df_freq.copy()
    n_bands = len(BANDS)
    per_band= max(2, n_numeri // n_bands)
    pool    = []

    print(f"  [Compensazione] Analisi per band di 15:")
    for bmin, bmax, label in BANDS:
        df_band = df[
            (df['numero'] >= bmin) &
            (df['numero'] <= bmax) &
            (df['freq_storica'] > 0)   # compatibili con la fascia
        ].sort_values('freq_storica', ascending=True)

        freq_media = df_band['freq_storica'].mean() \
                     if not df_band.empty else 0
        print(f"    {label} [{bmin:2d}-{bmax:2d}] "
              f"presenti={len(df_band)} "
              f"freq_media={freq_media:.4f}")

        top = df_band.head(per_band)
        pool.append(top)

    df_pool = pd.concat(pool).drop_duplicates('numero')

    # Se pool troppo piccolo aggiungi da tutto il resto
    if len(df_pool) < n_numeri:
        df_resto = df[
            (~df['numero'].isin(df_pool['numero'])) &
            (df['freq_storica'] > 0)
        ].sort_values('freq_storica', ascending=True)
        df_pool = pd.concat([df_pool, df_resto])\
                    .drop_duplicates('numero')\
                    .head(n_numeri)

    return df_pool.sort_values('freq_storica').head(n_numeri)

def analizza_struttura_fascia(df_fascia):
    """
    Analizza le estrazioni storiche nella fascia target.
    Trova la distribuzione INTERMEDIA di:
    - parità (n. numeri pari per sestina)
    - decadi B=1-30, M=31-60, A=61-90

    LOGICA INTERMEDIA CORRETTA:
    Non prende la posizione centrale della lista
    ma il valore con frequenza più vicina alla MEDIANA
    delle frequenze → esclude sia i dominanti
    sia i rarissimi, prende il centro della distribuzione.
    """
    from collections import Counter
    cols = ['n1','n2','n3','n4','n5','n6']
    n    = len(df_fascia)
    if n == 0:
        return None

    # ── Parità ───────────────────────────────────────────
    parita_counts_raw = Counter()
    for _, row in df_fascia.iterrows():
        n_pari = sum(1 for c in cols if row[c] % 2 == 0)
        parita_counts_raw[n_pari] += 1

    # Mantieni solo distribuzioni centrali: 2p/4d, 3p/3d, 4p/2d
    # Escludi gli estremi: 0p/6d, 1p/5d, 5p/1d, 6p/0d
    parita_counts = {k: v for k, v in parita_counts_raw.items()
                     if k in [2, 3, 4]}
    if not parita_counts:
        parita_counts = dict(parita_counts_raw)

    parita_sorted = sorted(parita_counts.items(),
                           key=lambda x: x[1], reverse=True)

    # Mediana delle frequenze tra i 3 valori centrali
    freqs_p       = sorted([cnt for _, cnt in parita_sorted])
    mediana_p     = freqs_p[len(freqs_p) // 2]
    n_pari_target = min(parita_sorted,
                        key=lambda x: abs(x[1] - mediana_p))[0]
    pct_pari      = round(parita_counts_raw[n_pari_target]*100/n, 1)

    print(f"  [Struttura] Distribuzione parità (solo 2p-4p):")
    for np_ in sorted(parita_counts_raw.keys()):
        cnt    = parita_counts_raw[np_]
        esclusa = " [esclusa]" if np_ not in [2, 3, 4] else ""
        marker  = " ← INTERMEDIA" if np_ == n_pari_target else ""
        print(f"    {np_}p/{6-np_}d: "
              f"{cnt} ({cnt*100/n:.1f}%){esclusa}{marker}")

    # ── Decadi B=1-30, M=31-60, A=61-90 ─────────────────
    decade_counts = Counter()
    for _, row in df_fascia.iterrows():
        nB = sum(1 for c in cols if row[c] <= 30)
        nM = sum(1 for c in cols if 31 <= row[c] <= 60)
        nA = sum(1 for c in cols if row[c] >= 61)
        decade_counts[(nB, nM, nA)] += 1

    decade_sorted = sorted(decade_counts.items(),
                           key=lambda x: x[1], reverse=True)

    # Mediana delle frequenze
    freqs_d    = sorted([cnt for _, cnt in decade_sorted])
    mediana_d  = freqs_d[len(freqs_d) // 2]
    decade_target = min(decade_sorted,
                        key=lambda x: abs(x[1] - mediana_d))[0]
    pct_decade = round(decade_counts[decade_target]*100/n, 1)

    print(f"  [Struttura] Distribuzione decadi nella fascia "
          f"(B=1-30, M=31-60, A=61-90):")
    for (b, m, a), cnt in decade_sorted[:8]:
        marker = " ← INTERMEDIA" if (b,m,a)==decade_target else ""
        print(f"    B{b}M{m}A{a}: "
              f"{cnt} ({cnt*100/n:.1f}%){marker}")

    vincoli = {
        'n_pari':     n_pari_target,
        'n_disp':     6 - n_pari_target,
        'nB':         decade_target[0],
        'nM':         decade_target[1],
        'nA':         decade_target[2],
        'pct_pari':   pct_pari,
        'pct_decade': pct_decade,
    }
    print(f"  [Struttura] Vincolo applicato: "
          f"{n_pari_target}p/{6-n_pari_target}d "
          f"({pct_pari}%) | "
          f"B{decade_target[0]}M{decade_target[1]}"
          f"A{decade_target[2]} ({pct_decade}%)")
    return vincoli


def esegui_compensazione(df_raw, wyckoff_id, stato,
                          df_zone, df_cicli, client):
    fascia_min = stato['fascia_min']
    fascia_max = stato['fascia_max']

    print(f"\n  [Compensazione] Fascia target: "
          f"{fascia_min}-{fascia_max}")

    cols = ['n1','n2','n3','n4','n5','n6']
    df   = df_raw.copy()
    if 'somma' not in df.columns:
        df['somma'] = df[cols].sum(axis=1)

    df_fascia = estrai_estrazioni_in_fascia(
        df, fascia_min, fascia_max
    )

    # Se fascia troppo stretta allarga
    if len(df_fascia) < 50:
        margine   = 20
        print(f"  [Compensazione] Poche estrazioni, "
              f"allargo fascia di ±{margine}")
        df_fascia = estrai_estrazioni_in_fascia(
            df, fascia_min-margine, fascia_max+margine
        )

    # Calcola frequenze dall'interno della fascia
    df_freq = calcola_frequenze_numeri(
        df, df_fascia, df_cicli, fascia_min, fascia_max
    )

    # Seleziona pool: meno frequenti nella fascia
    pool_df     = seleziona_pool(df_freq, n_numeri=POOL_SIZE)
    pool_numeri = pool_df['numero'].tolist()

    print(f"  [Compensazione] Pool ({len(pool_numeri)} numeri):")
    print(f"    {sorted(pool_numeri)}")
    print(f"  [Compensazione] Top 15 meno frequenti in fascia:")
    for _, r in pool_df.head(15).iterrows():
        print(f"    N.{int(r['numero']):2d}  "
              f"storico={r['freq_storica']:.4f}  "
              f"recente={r['freq_recente']:.4f}  "
              f"delta={r['delta']:+.4f}")

    # Analisi struttura storica della fascia
    print(f"\n  [Struttura] Analisi parità+decade nella fascia...")
    vincoli = analizza_struttura_fascia(df_fascia)

    # Salva su Supabase
    records = []
    for _, r in pool_df.iterrows():
        records.append({
            "wyckoff_id":  wyckoff_id,
            "numero":      int(r['numero']),
            "freq_zona":   float(r['freq_storica']),
            "freq_attesa": float(r['freq_attesa']),
            "delta":       float(r['delta']),
            "incluso":     True,
        })
    if records:
        client.table("pool_compensazione")\
            .insert(records).execute()
        print(f"  [Compensazione] Pool salvato su Supabase")

    return pool_numeri, vincoli
