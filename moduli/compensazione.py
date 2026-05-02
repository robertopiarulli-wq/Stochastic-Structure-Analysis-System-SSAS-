"""
SSAS - Motore 2: Compensazione Numerica
Data la fascia target da Wyckoff,
identifica quali numeri sono sottorappresentati
in quella zona negli ultimi N cicli.
Restituisce il pool per il Motore 3.
"""
import numpy as np
import pandas as pd

N_CICLI_FOCUS = 3

def estrai_estrazioni_in_fascia(df_full, fascia_min, fascia_max):
    """
    Restituisce tutte le estrazioni storiche
    la cui somma cade nella fascia target.
    """
    mask = (df_full['somma'] >= fascia_min) & \
           (df_full['somma'] <= fascia_max)
    return df_full[mask].copy()

def calcola_frequenze_numeri(df_fascia, df_cicli_focus):
    """
    Per ogni numero 1-90, calcola:
    - freq_globale: quanto appare nella fascia su tutto lo storico
    - freq_recente: quanto appare negli ultimi N cicli nella fascia
    - delta: sottorappresentazione recente vs storico
    """
    cols = ['n1','n2','n3','n4','n5','n6']
    risultati = []
    n_tot     = len(df_fascia)

    # Estrazioni nella fascia negli ultimi cicli
    if not df_cicli_focus.empty and n_tot > 0:
        idx_min = int(df_cicli_focus['start_idx'].min())
        df_recente_fascia = df_fascia[
            df_fascia.index >= idx_min
        ]
    else:
        df_recente_fascia = df_fascia.tail(137*N_CICLI_FOCUS)

    n_rec = len(df_recente_fascia)

    for numero in range(1, 91):
        # Frequenza globale nella fascia
        freq_glob = int(
            df_fascia[cols].apply(
                lambda r: numero in r.values, axis=1
            ).sum()
        )
        pct_glob = freq_glob / n_tot if n_tot > 0 else 0

        # Frequenza recente nella fascia
        freq_rec = int(
            df_recente_fascia[cols].apply(
                lambda r: numero in r.values, axis=1
            ).sum()
        ) if n_rec > 0 else 0
        pct_rec = freq_rec / n_rec if n_rec > 0 else 0

        # Frequenza attesa teorica
        freq_att = 6.0 / 90.0

        # Delta: negativo = sottorappresentato (candidato compensazione)
        delta = pct_rec - pct_glob

        risultati.append({
            'numero':      numero,
            'freq_globale': round(pct_glob, 6),
            'freq_recente': round(pct_rec, 6),
            'freq_attesa':  round(freq_att, 6),
            'delta':        round(delta, 6),
            'n_fascia_tot': n_tot,
            'n_fascia_rec': n_rec,
        })

    return pd.DataFrame(risultati)

def seleziona_pool(df_freq, fascia_min, fascia_max,
                   n_numeri=30, delta_soglia=0.0):
    """
    Seleziona il pool di numeri per il Motore 3.

    Logica:
    - Priorità ai numeri con delta negativo
      (sottorappresentati nella zona recente)
    - Filtro per compatibilità con la fascia somma:
      esclude numeri troppo alti se la fascia è bassa
      e viceversa
    - Restituisce i top N numeri
    """
    df = df_freq.copy()

    # Stima range valori compatibili con la fascia
    # Per formare una sestina con somma in [fascia_min, fascia_max]
    # i numeri devono essere approssimativamente
    # in [fascia_min/6 - margine, fascia_max/6 + margine]
    centro_fascia = (fascia_min + fascia_max) / 2
    margine       = (fascia_max - fascia_min) * 2

    num_min = max(1,  int(fascia_min/6 - margine/6))
    num_max = min(90, int(fascia_max/6 + margine/6))

    print(f"  [Compensazione] Range numeri compatibili: "
          f"{num_min}-{num_max}")

    # Filtro compatibilità
    df = df[
        (df['numero'] >= num_min) &
        (df['numero'] <= num_max)
    ].copy()

    # Ordina per delta crescente
    # (più negativo = più sottorappresentato = priorità alta)
    df = df.sort_values('delta', ascending=True)

    # Prendi i top N
    pool = df.head(n_numeri)

    return pool

def esegui_compensazione(df_raw, wyckoff_id, stato,
                         df_zone, df_cicli, client):
    """
    Pipeline completa Motore 2.
    Salva pool in pool_compensazione su Supabase.
    Restituisce lista numeri del pool.
    """
    fascia_min = stato['fascia_min']
    fascia_max = stato['fascia_max']

    print(f"\n  [Compensazione] Fascia target: "
          f"{fascia_min}-{fascia_max}")

    # Calcola somma se non presente
    cols = ['n1','n2','n3','n4','n5','n6']
    df   = df_raw.copy()
    if 'somma' not in df.columns:
        df['somma'] = df[cols].sum(axis=1)

    # Estrazioni nella fascia
    df_fascia = estrai_estrazioni_in_fascia(df, fascia_min, fascia_max)
    print(f"  [Compensazione] Estrazioni storiche in fascia: "
          f"{len(df_fascia)}")

    if df_fascia.empty:
        print("  [Compensazione] ATTENZIONE: fascia vuota, "
              "uso fascia allargata")
        margine    = 30
        df_fascia  = estrai_estrazioni_in_fascia(
            df, fascia_min-margine, fascia_max+margine
        )

    # Calcola frequenze
    df_freq = calcola_frequenze_numeri(df_fascia, df_cicli)

    # Seleziona pool
    pool_df = seleziona_pool(
        df_freq, fascia_min, fascia_max, n_numeri=30
    )
    pool_numeri = pool_df['numero'].tolist()

    print(f"  [Compensazione] Pool selezionato ({len(pool_numeri)} numeri):")
    print(f"    {sorted(pool_numeri)}")

    # Dettaglio top 10
    print("  [Compensazione] Top 10 per sottorappresentazione:")
    for _, r in pool_df.head(10).iterrows():
        print(f"    N.{int(r['numero']):2d} "
              f"glob={r['freq_globale']:.4f} "
              f"rec={r['freq_recente']:.4f} "
              f"delta={r['delta']:.4f}")

    # Salva su Supabase
    records = []
    for _, r in pool_df.iterrows():
        records.append({
            "wyckoff_id": wyckoff_id,
            "numero":     int(r['numero']),
            "freq_zona":  float(r['freq_globale']),
            "freq_attesa":float(r['freq_attesa']),
            "delta":      float(r['delta']),
            "incluso":    True,
        })

    if records:
        client.table("pool_compensazione")\
            .insert(records).execute()
        print(f"  [Compensazione] Pool salvato su Supabase")

    return pool_numeri
