"""
SSAS - Motore 1: Analisi Wyckoff
Trasforma ogni estrazione in candela OHLC,
calcola indicatori tecnici sulla somma,
identifica zona attuale e fascia target.
"""
import numpy as np
import pandas as pd

# ── Parametri ────────────────────────────────────────────
PERIODO_BB  = 137   # Bollinger: ciclo Parisi
PERIODO_RSI = 14
PERIODO_ADX = 14
N_CICLI_FOCUS = 3  # Ultimi N cicli da 137 per compensazione

def build_ohlc(df):
    """
    Costruisce serie OHLC da ogni estrazione.
    Low=n1, Open=n2, Close=n5, High=n6
    Volume=somma totale
    """
    cols = ['n1','n2','n3','n4','n5','n6']
    df = df.copy()
    df['low']    = df['n1']
    df['open']   = df['n2']
    df['close']  = df['n5']
    df['high']   = df['n6']
    df['volume'] = df[cols].sum(axis=1)
    df['somma']  = df[cols].sum(axis=1)
    df['range_candela'] = df['high'] - df['low']
    df['corpo']  = abs(df['close'] - df['open'])
    return df

def calcola_rsi(series, period=14):
    """RSI classico su serie somme."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcola_bollinger(series, period=137, std_dev=2):
    """Bande di Bollinger su serie somme."""
    sma   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calcola_adx(df, period=14):
    """
    ADX semplificato sulla serie OHLC.
    Misura forza del trend indipendentemente dalla direzione.
    """
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n     = len(close)

    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    ndm  = np.zeros(n)

    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)

        up   = high[i] - high[i-1]
        down = low[i-1] - low[i]
        pdm[i] = up   if up > down and up > 0   else 0
        ndm[i] = down if down > up and down > 0 else 0

    def smooth(arr, p):
        s = np.zeros(len(arr))
        s[p] = arr[1:p+1].sum()
        for i in range(p+1, len(arr)):
            s[i] = s[i-1] - s[i-1]/p + arr[i]
        return s

    atr  = smooth(tr,  period)
    satr = np.where(atr != 0, atr, np.nan)
    pdi  = 100 * smooth(pdm, period) / satr
    ndi  = 100 * smooth(ndm, period) / satr
    dx   = 100 * np.abs(pdi - ndi) / np.where((pdi+ndi)!=0, pdi+ndi, np.nan)

    adx  = np.zeros(n)
    adx[2*period] = np.nanmean(dx[period:2*period+1])
    for i in range(2*period+1, n):
        adx[i] = (adx[i-1]*(period-1) + dx[i]) / period

    return pd.Series(adx, index=df.index)

def identifica_zona(somma, bb_upper, bb_lower, rsi, adx):
    """
    Classifica la zona attuale:
    - saturazione_alta: vicino banda superiore
    - saturazione_bassa: vicino banda inferiore
    - transizione: tra le bande, movimento forte
    - laterale: ADX basso, movimento debole
    """
    if pd.isna(bb_upper) or pd.isna(bb_lower):
        return "indefinita"

    bb_range = bb_upper - bb_lower
    pos_rel  = (somma - bb_lower) / bb_range if bb_range > 0 else 0.5

    if adx > 25:
        if pos_rel > 0.8:
            return "saturazione_alta"
        elif pos_rel < 0.2:
            return "saturazione_bassa"
        else:
            return "transizione"
    else:
        return "laterale"

def identifica_trend(serie_somme, window=10):
    """
    Determina direzione trend recente.
    Guarda pendenza media mobile breve.
    """
    if len(serie_somme) < window:
        return "indefinito"
    recenti = serie_somme.tail(window)
    slope   = np.polyfit(range(window), recenti.values, 1)[0]
    if slope > 2:
        return "markup"
    elif slope < -2:
        return "markdown"
    else:
        return "laterale"

def calcola_zone_saturazione(df_full, bin_size=10):
    """
    Identifica zone di saturazione e transizione
    sull'intero database storico.
    Restituisce DataFrame con frequenza per fascia.
    """
    somme = df_full['somma']
    bins  = range(21, 526, bin_size)
    freq, edges = np.histogram(somme, bins=bins)

    zone = []
    for i in range(len(freq)):
        centro = (edges[i] + edges[i+1]) / 2
        zone.append({
            'fascia_min':  int(edges[i]),
            'fascia_max':  int(edges[i+1]),
            'centro':      centro,
            'frequenza':   int(freq[i]),
        })

    df_zone = pd.DataFrame(zone)
    freq_media = df_zone['frequenza'].mean()
    freq_std   = df_zone['frequenza'].std()

    # Classifica ogni fascia
    def classifica(row):
        if row['frequenza'] > freq_media + freq_std:
            return 'saturazione'
        elif row['frequenza'] < freq_media - freq_std * 0.5:
            return 'transizione'
        else:
            return 'normale'

    df_zone['tipo'] = df_zone.apply(classifica, axis=1)

    # Identifica transizioni tra due saturazioni
    for i in range(1, len(df_zone)-1):
        prev_tipo = df_zone.iloc[i-1]['tipo']
        curr_tipo = df_zone.iloc[i]['tipo']
        next_tipo = df_zone.iloc[i+1]['tipo']
        if (curr_tipo == 'transizione' and
            prev_tipo == 'saturazione' and
            next_tipo == 'saturazione'):
            df_zone.at[i, 'tipo'] = 'transizione_graal'

    return df_zone

def determina_fascia_target(somma_attuale, df_zone, trend):
    """
    Data la posizione attuale e il trend,
    determina la fascia target prossima estrazione.
    """
    # Trova fascia attuale
    fascia_attuale = df_zone[
        (df_zone['fascia_min'] <= somma_attuale) &
        (df_zone['fascia_max'] >  somma_attuale)
    ]

    if fascia_attuale.empty:
        # Fuori range — usa media storica
        centro = df_zone['centro'].mean()
        fascia = df_zone[
            (df_zone['fascia_min'] <= centro) &
            (df_zone['fascia_max'] >  centro)
        ].iloc[0]
        return int(fascia['fascia_min']), int(fascia['fascia_max'])

    idx_att = fascia_attuale.index[0]

    if trend == 'markup':
        # Cerca prossima saturazione verso l'alto
        candidate = df_zone[
            (df_zone.index > idx_att) &
            (df_zone['tipo'].isin(['saturazione','transizione_graal']))
        ]
    elif trend == 'markdown':
        # Cerca prossima saturazione verso il basso
        candidate = df_zone[
            (df_zone.index < idx_att) &
            (df_zone['tipo'].isin(['saturazione','transizione_graal']))
        ].iloc[::-1]
    else:
        # Laterale: resta nella fascia attuale o adiacente
        candidate = df_zone[
            df_zone['tipo'] == 'saturazione'
        ]
        candidate = candidate.iloc[
            (candidate['centro'] - somma_attuale).abs().argsort()
        ]

    if candidate.empty:
        row = df_zone.iloc[idx_att]
    else:
        row = candidate.iloc[0]

    return int(row['fascia_min']), int(row['fascia_max'])

def analizza_cicli(df_full, n_cicli=N_CICLI_FOCUS):
    """
    Analizza gli ultimi N cicli da 137 estrazioni.
    Restituisce caratteristiche per ciclo.
    """
    cicli = []
    n     = len(df_full)

    for i in range(n_cicli):
        start = n - (i+1)*137
        end   = n - i*137
        if start < 0:
            break
        blocco = df_full.iloc[start:end]
        cicli.append({
            'ciclo':          i+1,
            'start_idx':      start,
            'end_idx':        end,
            'somma_media':    round(blocco['somma'].mean(), 2),
            'somma_min':      int(blocco['somma'].min()),
            'somma_max':      int(blocco['somma'].max()),
            'somma_std':      round(blocco['somma'].std(), 2),
            'range_medio':    round(blocco['range_candela'].mean(), 2),
        })

    return pd.DataFrame(cicli)

def esegui_wyckoff(df_raw, client):
    """
    Pipeline completa Motore 1.
    Salva risultato in wyckoff_stato su Supabase.
    Restituisce (stato_dict, df_zone, df_cicli)
    """
    print("  [Wyckoff] Costruzione OHLC...")
    df = build_ohlc(df_raw)
    df = df.sort_values('data_estrazione').reset_index(drop=True)

    # Indicatori sulla serie somme
    print("  [Wyckoff] Calcolo indicatori...")
    somme = df['somma']
    bb_u, bb_m, bb_l = calcola_bollinger(somme, PERIODO_BB)
    rsi               = calcola_rsi(somme, PERIODO_RSI)
    adx               = calcola_adx(df, PERIODO_ADX)

    # Valori correnti (ultima riga)
    ultimo    = df.iloc[-1]
    rsi_att   = float(rsi.iloc[-1])   if not pd.isna(rsi.iloc[-1])   else 50.0
    adx_att   = float(adx.iloc[-1])   if not pd.isna(adx.iloc[-1])   else 0.0
    bbu_att   = float(bb_u.iloc[-1])  if not pd.isna(bb_u.iloc[-1])  else 400.0
    bbl_att   = float(bb_l.iloc[-1])  if not pd.isna(bb_l.iloc[-1])  else 150.0
    somma_att = int(ultimo['somma'])

    trend     = identifica_trend(somme)
    zona_tipo = identifica_zona(somma_att, bbu_att, bbl_att, rsi_att, adx_att)

    print(f"  [Wyckoff] Somma attuale: {somma_att}")
    print(f"  [Wyckoff] Trend:         {trend}")
    print(f"  [Wyckoff] Zona:          {zona_tipo}")
    print(f"  [Wyckoff] RSI:           {rsi_att:.2f}")
    print(f"  [Wyckoff] ADX:           {adx_att:.2f}")
    print(f"  [Wyckoff] BB upper:      {bbu_att:.1f}")
    print(f"  [Wyckoff] BB lower:      {bbl_att:.1f}")

    # Zone saturazione storiche
    print("  [Wyckoff] Mappatura zone storiche...")
    df_zone = calcola_zone_saturazione(df)

    n_saturazione  = int((df_zone['tipo']=='saturazione').sum())
    n_transizione  = int((df_zone['tipo']=='transizione').sum())
    n_graal        = int((df_zone['tipo']=='transizione_graal').sum())
    print(f"  [Wyckoff] Zone saturazione: {n_saturazione}")
    print(f"  [Wyckoff] Zone transizione: {n_transizione}")
    print(f"  [Wyckoff] Zone graal:       {n_graal}")

    # Fascia target
    fascia_min, fascia_max = determina_fascia_target(
        somma_att, df_zone, trend
    )
    print(f"  [Wyckoff] Fascia target: {fascia_min}-{fascia_max}")

    # Analisi cicli
    print("  [Wyckoff] Analisi ultimi cicli...")
    df_cicli = analizza_cicli(df)
    for _, c in df_cicli.iterrows():
        print(f"    Ciclo {int(c['ciclo'])}: "
              f"media={c['somma_media']} "
              f"range=[{c['somma_min']}-{c['somma_max']}]")

    # Salva su Supabase
    stato = {
        "somma_ultima":     somma_att,
        "trend":            trend,
        "rsi_attuale":      round(rsi_att, 4),
        "adx_attuale":      round(adx_att, 4),
        "bb_upper":         round(bbu_att, 2),
        "bb_lower":         round(bbl_att, 2),
        "fascia_min":       fascia_min,
        "fascia_max":       fascia_max,
        "zona_tipo":        zona_tipo,
        "cicli_analizzati": len(df_cicli),
    }

    res = client.table("wyckoff_stato").insert(stato).execute()
    wyckoff_id = res.data[0]['id']
    print(f"  [Wyckoff] Stato salvato (id={wyckoff_id})")

    return wyckoff_id, stato, df_zone, df_cicli
