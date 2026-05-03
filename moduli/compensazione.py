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

def seleziona_universo_tre_fasce(df_freq, n_per_fascia=10):
    """
    Divide 1-90 in 3 fasce di VALORE:
      BASSA valore:  1-30
      MEDIA valore: 31-60
      ALTA valore:  61-90

    Dentro ogni fascia di valore → prende i meno frequenti
    nella zona target (compensazione).

    n_per_fascia = quanti numeri prendere per fascia (default 10)
    → universo totale = 30 numeri (10 per fascia)
    → copre tutto il tabellone garantendo somme alte
    """
    BANDE_VALORE = [
        (1,  30, "BASSA"),
        (31, 60, "MEDIA"),
        (61, 90, "ALTA"),
    ]

    pool_bassa = []
    pool_media = []
    pool_alta  = []
    universo   = []

    print(f"  [Compensazione] Tre fasce di VALORE "
          f"({n_per_fascia} meno freq per fascia):")

    for vmin, vmax, label in BANDE_VALORE:
        df_band = df_freq[
            (df_freq['numero'] >= vmin) &
            (df_freq['numero'] <= vmax) &
            (df_freq['freq_storica'] > 0)
        ].sort_values('freq_storica', ascending=True)

        # Prendi i meno frequenti dentro questa fascia di valore
        top = df_band.head(n_per_fascia)['numero'].tolist()

        freq_media = df_band['freq_storica'].mean() \
                     if not df_band.empty else 0
        print(f"    {label:5s} [{vmin:2d}-{vmax:2d}] "
              f"meno frequenti: {sorted(top)} "
              f"(freq_media_fascia={freq_media:.4f})")

        if label == "BASSA":
            pool_bassa = top
        elif label == "MEDIA":
            pool_media = top
        else:
            pool_alta = top

        universo.extend(top)

    universo = sorted(universo)
    print(f"  [Compensazione] Universo sestine "
          f"({len(universo)} numeri): {universo}")

    return pool_bassa, pool_media, pool_alta, universo

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

    # Seleziona universo: meno frequenti per fascia di valore
    pool_bassa, pool_media, pool_alta, universo = \
        seleziona_universo_tre_fasce(df_freq, n_per_fascia=10)

    pool_numeri = universo  # 30 numeri totali

    # Analisi struttura storica della fascia
    print(f"\n  [Struttura] Analisi parità+decade nella fascia...")
    vincoli = analizza_struttura_fascia(df_fascia)

    # Salva su Supabase
    records = []
    for n in pool_bassa:
        r = df_freq[df_freq['numero']==n].iloc[0]
        records.append({
            "wyckoff_id":  wyckoff_id,
            "numero":      n,
            "freq_zona":   float(r['freq_storica']),
            "freq_attesa": float(r['freq_attesa']),
            "delta":       float(r['delta']),
            "incluso":     True,
        })
    for n in pool_media:
        r = df_freq[df_freq['numero']==n].iloc[0]
        records.append({
            "wyckoff_id":  wyckoff_id,
            "numero":      n,
            "freq_zona":   float(r['freq_storica']),
            "freq_attesa": float(r['freq_attesa']),
            "delta":       float(r['delta']),
            "incluso":     True,
        })
    for n in pool_alta:
        r = df_freq[df_freq['numero']==n].iloc[0]
        records.append({
            "wyckoff_id":  wyckoff_id,
            "numero":      n,
            "freq_zona":   float(r['freq_storica']),
            "freq_attesa": float(r['freq_attesa']),
            "delta":       float(r['delta']),
            "incluso":     True,
        })
    if records:
        client.table("pool_compensazione")\
            .insert(records).execute()
        print(f"  [Compensazione] Universo salvato su Supabase "
              f"({len(universo)} numeri, 10 per fascia di valore)")

    # Aggiorna wyckoff_stato con vincolo parità
    if vincoli:
        try:
            client.table("wyckoff_stato")\
                .update({
                    "vincolo_n_pari":   vincoli['n_pari'],
                    "vincolo_pct_pari": vincoli['pct_pari'],
                })\
                .eq("id", wyckoff_id)\
                .execute()
            print(f"  [Struttura] Vincolo salvato su Supabase")
        except Exception as e:
            print(f"  [Struttura] Vincolo non salvato "
                  f"(colonne mancanti?): {e}")

    return pool_numeri, vincoli
