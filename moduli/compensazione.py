"""
SSAS - Motore 2: Compensazione Numerica
Data la fascia target da Wyckoff,
identifica i numeri sottorappresentati in quella zona
negli ultimi N cicli rispetto allo storico completo.
Il pool copre tutto il tabellone 1-90:
una sestina con somma 260 può avere numeri ovunque.
"""
import numpy as np
import pandas as pd

N_CICLI_FOCUS = 3
POOL_SIZE     = 30

def estrai_estrazioni_in_fascia(df, fascia_min, fascia_max):
    """Estrazioni storiche con somma nella fascia target."""
    cols  = ['n1','n2','n3','n4','n5','n6']
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
    - freq_storica: apparizioni nella fascia su tutto il db
    - freq_recente: apparizioni nella fascia negli ultimi N cicli
    - delta: freq_recente - freq_storica
      negativo = sottorappresentato recentemente = candidato pool
    """
    cols  = ['n1','n2','n3','n4','n5','n6']
    n_tot = len(df_fascia)

    # Ultimi N cicli da 137
    if not df_cicli.empty:
        idx_min = int(df_cicli['start_idx'].min())
        df_rec  = df_full[df_full.index >= idx_min].copy()
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

    # Matrice numpy per velocità
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
    Seleziona i numeri più sottorappresentati nella zona
    negli ultimi cicli rispetto allo storico.
    Delta negativo = mancano all'appello = candidati.
    Tutto il tabellone 1-90 è candidabile.
    """
    df = df_freq.sort_values('delta', ascending=True)
    return df.head(n_numeri)

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
        margine   = 30
        print(f"  [Compensazione] Poche estrazioni, "
              f"allargo fascia di ±{margine}")
        df_fascia = estrai_estrazioni_in_fascia(
            df, fascia_min-margine, fascia_max+margine
        )

    # Calcola frequenze su tutto il tabellone
    df_freq = calcola_frequenze_numeri(
        df, df_fascia, df_cicli, fascia_min, fascia_max
    )

    # Seleziona pool
    pool_df     = seleziona_pool(df_freq, n_numeri=POOL_SIZE)
    pool_numeri = pool_df['numero'].tolist()

    print(f"  [Compensazione] Pool ({len(pool_numeri)} numeri):")
    print(f"    {sorted(pool_numeri)}")
    print(f"  [Compensazione] Top 15 sottorappresentati:")
    for _, r in pool_df.head(15).iterrows():
        print(f"    N.{int(r['numero']):2d}  "
              f"storico={r['freq_storica']:.4f}  "
              f"recente={r['freq_recente']:.4f}  "
              f"delta={r['delta']:+.4f}")

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

    return pool_numeri
