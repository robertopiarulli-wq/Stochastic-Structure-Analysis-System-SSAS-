import numpy as np
import pandas as pd

COSTANTI_CONFIG = [
    {
        "nome": "somma",
        "descrizione": "Somma dei 6 numeri estratti",
        "colonna": "somma",
        "teorico": 6 * 45.5
    },
    {
        "nome": "cv_gap",
        "descrizione": "Coefficiente variazione gap (disordine interno)",
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
        "descrizione": "Numero valori pari nella sestina",
        "colonna": "n_pari",
        "teorico": 3.0
    },
    {
        "nome": "decadi_coperte",
        "descrizione": "Quante decadi distinte sono presenti",
        "colonna": "decadi_coperte",
        "teorico": None
    },
    {
        "nome": "gap_ratio",
        "descrizione": "Asimmetria gap_max/gap_min",
        "colonna": "gap_ratio",
        "teorico": None
    },
    {
        "nome": "overlap_lag1",
        "descrizione": "Numeri comuni con estrazione precedente",
        "colonna": "overlap_lag1",
        "teorico": 6 * (6/90)
    },
    {
        "nome": "n_coppie_consecutive",
        "descrizione": "Coppie di numeri consecutivi",
        "colonna": "n_coppie_consecutive",
        "teorico": None
    },
]

def calcola_costanti(fp_df):
    records = []
    for c in COSTANTI_CONFIG:
        col = fp_df[c["colonna"]].dropna()
        n   = len(col)
        media = float(col.mean())
        std   = float(col.std())
        cv    = float(std / media) if media != 0 else None
        teorico = c["teorico"]
        sigma = float((media - teorico) / std) \
                if teorico is not None and std > 0 else None

        records.append({
            "nome":                  c["nome"],
            "descrizione":           c["descrizione"],
            "valore_medio":          round(media, 4),
            "std_dev":               round(std, 4),
            "mediana":               round(float(col.median()), 4),
            "percentile_5":          round(float(col.quantile(0.05)), 4),
            "percentile_10":         round(float(col.quantile(0.10)), 4),
            "percentile_25":         round(float(col.quantile(0.25)), 4),
            "percentile_75":         round(float(col.quantile(0.75)), 4),
            "percentile_90":         round(float(col.quantile(0.90)), 4),
            "percentile_95":         round(float(col.quantile(0.95)), 4),
            "cv_storico":            round(cv, 4) if cv else None,
            "valore_atteso_teorico": teorico,
            "sigma_da_random":       round(sigma, 3) if sigma else None,
            "n_campioni":            n,
        })
    return records
