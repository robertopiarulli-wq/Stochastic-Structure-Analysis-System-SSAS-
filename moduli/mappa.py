import numpy as np
import pandas as pd
from datetime import datetime

def calcola_mappa(df):
    """
    Calcola occupazione per ogni numero 1-90
    Restituisce lista di dict pronti per upsert
    """
    n_estrazioni = len(df)
    freq_attesa  = 6.0 / 90.0
    cols         = ['n1','n2','n3','n4','n5','n6']
    records      = []

    # Matrice booleana: riga=estrazione, colonna=numero
    # Più veloce del loop su apply
    tutti = df[cols].values  # shape (7304, 6)

    for numero in range(1, 91):
        presente = np.any(tutti == numero, axis=1)  # array bool

        freq_abs = int(presente.sum())
        freq_rel = freq_abs / n_estrazioni

        # Z-score binomiale
        atteso   = n_estrazioni * freq_attesa
        std_bin  = np.sqrt(n_estrazioni * freq_attesa * (1 - freq_attesa))
        z        = float((freq_abs - atteso) / std_bin)

        # Ritardo
        idxs = np.where(presente)[0]
        if len(idxs) > 0:
            ultimo_idx  = int(idxs[-1])
            ritardo     = int(n_estrazioni - 1 - ultimo_idx)
            ultimo_data = str(df.iloc[ultimo_idx]['data_estrazione'])
        else:
            ritardo     = n_estrazioni
            ultimo_data = None

        # Ritardo medio e max storici
        if len(idxs) > 1:
            gaps_rit  = np.diff(idxs)
            rit_medio = float(gaps_rit.mean())
            rit_max   = int(gaps_rit.max())
        else:
            rit_medio = float(n_estrazioni)
            rit_max   = int(n_estrazioni)

        # Densità locale: freq media dei 4 vicini
        vicini = [v for v in [numero-2, numero-1, numero+1, numero+2]
                  if 1 <= v <= 90]
        densita = float(np.mean([
            np.any(tutti == v, axis=1).sum() / n_estrazioni
            for v in vicini
        ]))

        records.append({
            "numero":         numero,
            "freq_assoluta":  freq_abs,
            "freq_relativa":  round(freq_rel, 6),
            "z_score":        round(z, 4),
            "ritardo_attuale":ritardo,
            "ritardo_medio":  round(rit_medio, 2),
            "ritardo_max":    rit_max,
            "ultimo_estratto":ultimo_data,
            "densita_locale": round(densita, 6),
            "aggiornato_al":  datetime.now().isoformat(),
        })

    return records
