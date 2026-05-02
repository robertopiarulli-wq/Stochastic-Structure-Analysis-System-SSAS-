"""
SSAS - Motore 1: Analisi Wyckoff
Identifica zone di transizione tra due saturazioni
sull'intero database storico.
"""
import numpy as np
import pandas as pd

PERIODO_BB  = 137
PERIODO_RSI = 14
PERIODO_ADX = 14
N_CICLI_FOCUS = 3
BIN_SIZE = 40  # Ampiezza fascia per istogramma somme

def build_ohlc(df):
    cols = ['n1','n2','n3','n4','n5','n6']
    df = df.copy()
    df['low']    = df['n1']
    df['open']   = df['n2']
    df['close']  = df['n5']
    df['high']   = df['n6']
    df['somma']  = df[cols].sum(axis=1)
    df['range_candela'] = df['high'] - df['low']
    df['corpo']  = abs(df['close'] - df['open'])
    return df

def calcola_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcola_bollinger(series, period=137, std_dev=2):
    sma   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std()
    return sma + std_dev*std, sma, sma - std_dev*std

def calcola_adx(df, period=14):
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n     = len(close)

    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)

    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i]  - close[i-1])
        tr[i] = max(hl, hpc, lpc)
        up    = high[i] - high[i-1]
        down  = low[i-1] - low[i]
        pdm[i] = up   if up > down   and up > 0   else 0
        ndm[i] = down if down > up   and down > 0 else 0

    def smooth(arr, p):
        s = np.zeros(len(arr))
        s[p] = arr[1:p+1].sum()
        for i in range(p+1, len(arr)):
            s[i] = s[i-1] - s[i-1]/p + arr[i]
        return s

    atr  = smooth(tr, period)
    satr = np.where(atr != 0, atr, np.nan)
    pdi  = 100 * smooth(pdm, period) / satr
    ndi  = 100 * smooth(ndm, period) / satr
    dx   = 100 * np.abs(pdi-ndi) / np.where((pdi+ndi)!=0, pdi+ndi, np.nan)

    adx = np.zeros(n)
    adx[2*period] = np.nanmean(dx[period:2*period+1])
    for i in range(2*period+1, n):
        adx[i] = (adx[i-1]*(period-1) + dx[i]) / period

    return pd.Series(adx, index=df.index)

def identifica_trend(serie_somme, window=10):
    if len(serie_somme) < window:
        return "indefinito"
    recenti = serie_somme.tail(window)
    slope   = np.polyfit(range(window), recenti.values, 1)[0]
    if slope > 2:
        return "markup"
    elif slope < -2:
        return "markdown"
    return "laterale"

def calcola_zone_saturazione(df_full, bin_size=BIN_SIZE):
    """
    Identifica zone di saturazione e transizione
    sull'intero database storico.
    Marca le zone 'transizione_graal':
    transizione compresa TRA due saturazioni.
    Esclude zone estreme (vicino a 21 e 525).
    """
    somme = df_full['somma']
    bins  = list(range(21, 526, bin_size))
    freq, edges = np.histogram(somme, bins=bins)

    zone = []
    for i in range(len(freq)):
        centro = (edges[i] + edges[i+1]) / 2
        zone.append({
            'fascia_min': int(edges[i]),
            'fascia_max': int(edges[i+1]),
            'centro':     centro,
            'frequenza':  int(freq[i]),
        })

    df_zone = pd.DataFrame(zone)
    freq_media = df_zone['frequenza'].mean()
    freq_std   = df_zone['frequenza'].std()

    # Classifica fasce
    def classifica(row):
        if row['frequenza'] > freq_media + freq_std * 0.5:
            return 'saturazione'
        elif row['frequenza'] < freq_media - freq_std * 0.3:
            return 'transizione'
        return 'normale'

    df_zone['tipo'] = df_zone.apply(classifica, axis=1)

    # Marca transizioni graal: tra due saturazioni
    # ed esclude zone estreme (prime 3 e ultime 3 fasce)
    n = len(df_zone)
    for i in range(3, n-3):
        curr = df_zone.iloc[i]['tipo']
        if curr != 'transizione':
            continue

        # Cerca saturazione a sinistra
        has_sat_left = False
        for j in range(i-1, max(-1, i-4), -1):
            if df_zone.iloc[j]['tipo'] == 'saturazione':
                has_sat_left = True
                break

        # Cerca saturazione a destra
        has_sat_right = False
        for j in range(i+1, min(n, i+4)):
            if df_zone.iloc[j]['tipo'] == 'saturazione':
                has_sat_right = True
                break

        if has_sat_left and has_sat_right:
            df_zone.at[i, 'tipo'] = 'transizione_graal'

    # Report
    n_sat   = int((df_zone['tipo']=='saturazione').sum())
    n_trans = int((df_zone['tipo']=='transizione').sum())
    n_graal = int((df_zone['tipo']=='transizione_graal').sum())
    print(f"  [Wyckoff] Zone saturazione:      {n_sat}")
    print(f"  [Wyckoff] Zone transizione:      {n_trans}")
    print(f"  [Wyckoff] Zone transizione graal:{n_graal}")

    # Stampa mappa completa
    print("\n  [Wyckoff] Mappa zone storiche:")
    for _, r in df_zone.iterrows():
        if r['tipo'] in ['saturazione','transizione_graal']:
            marker = "🔴" if r['tipo']=='saturazione' else "⭐"
            print(f"    {marker} {int(r['fascia_min']):3d}-"
                  f"{int(r['fascia_max']):3d} "
                  f"freq={int(r['frequenza']):4d} "
                  f"[{r['tipo']}]")

    return df_zone

def determina_fascia_target(somma_attuale, df_zone, trend):
    """
    Data posizione attuale e trend,
    individua la prossima zona graal da raggiungere.
    Priorità: transizione_graal > saturazione > normale
    """
    # Trova fascia attuale
    mask = ((df_zone['fascia_min'] <= somma_attuale) &
            (df_zone['fascia_max'] >  somma_attuale))
    fascia_att = df_zone[mask]

    if fascia_att.empty:
        idx_att = 0
    else:
        idx_att = fascia_att.index[0]

    zona_att = df_zone.iloc[idx_att]['tipo'] \
               if idx_att < len(df_zone) else 'normale'
    print(f"  [Wyckoff] Siamo in zona: {zona_att} "
          f"(somma={somma_attuale})")

    # Se siamo già in una transizione graal
    # la fascia target è quella stessa
    if zona_att == 'transizione_graal':
        r = df_zone.iloc[idx_att]
        print(f"  [Wyckoff] Siamo IN una zona graal!")
        return int(r['fascia_min']), int(r['fascia_max'])

    # Altrimenti cerca la prossima graal nel verso del trend
    if trend == 'markup':
        candidate = df_zone[
            (df_zone.index > idx_att) &
            (df_zone['tipo'] == 'transizione_graal')
        ]
    elif trend == 'markdown':
        candidate = df_zone[
            (df_zone.index < idx_att) &
            (df_zone['tipo'] == 'transizione_graal')
        ].iloc[::-1]
    else:
        # Laterale: graal più vicina in assoluto
        graal = df_zone[df_zone['tipo']=='transizione_graal'].copy()
        graal['dist'] = abs(graal['centro'] - somma_attuale)
        candidate = graal.sort_values('dist')

    if not candidate.empty:
        row = candidate.iloc[0]
        print(f"  [Wyckoff] Prossima zona graal: "
              f"{int(row['fascia_min'])}-{int(row['fascia_max'])}")
        return int(row['fascia_min']), int(row['fascia_max'])

    # Fallback: saturazione più vicina nel verso del trend
    if trend == 'markup':
        candidate = df_zone[
            (df_zone.index > idx_att) &
            (df_zone['tipo'] == 'saturazione')
        ]
    else:
        candidate = df_zone[
            (df_zone.index < idx_att) &
            (df_zone['tipo'] == 'saturazione')
        ].iloc[::-1]

    if not candidate.empty:
        row = candidate.iloc[0]
        return int(row['fascia_min']), int(row['fascia_max'])

    # Ultimo fallback: media storica
    centro = int(df_zone['centro'].mean())
    return centro - 15, centro + 15

def analizza_cicli(df_full, n_cicli=N_CICLI_FOCUS):
    cicli = []
    n     = len(df_full)
    for i in range(n_cicli):
        start = n - (i+1)*137
        end   = n - i*137
        if start < 0:
            break
        blocco = df_full.iloc[start:end]
        cicli.append({
            'ciclo':       i+1,
            'start_idx':   start,
            'end_idx':     end,
            'somma_media': round(float(blocco['somma'].mean()), 2),
            'somma_min':   int(blocco['somma'].min()),
            'somma_max':   int(blocco['somma'].max()),
            'somma_std':   round(float(blocco['somma'].std()), 2),
        })
    return pd.DataFrame(cicli)

def esegui_wyckoff(df_raw, client):
    print("  [Wyckoff] Costruzione OHLC...")
    df = build_ohlc(df_raw)
    df = df.sort_values('data_estrazione').reset_index(drop=True)

    print("  [Wyckoff] Calcolo indicatori...")
    somme       = df['somma']
    bb_u,_,bb_l = calcola_bollinger(somme, PERIODO_BB)
    rsi         = calcola_rsi(somme, PERIODO_RSI)
    adx         = calcola_adx(df, PERIODO_ADX)

    rsi_att  = float(rsi.iloc[-1])  if not pd.isna(rsi.iloc[-1])  else 50.0
    adx_att  = float(adx.iloc[-1])  if not pd.isna(adx.iloc[-1])  else 0.0
    bbu_att  = float(bb_u.iloc[-1]) if not pd.isna(bb_u.iloc[-1]) else 400.0
    bbl_att  = float(bb_l.iloc[-1]) if not pd.isna(bb_l.iloc[-1]) else 150.0
    somma_att= int(df.iloc[-1]['somma'])
    trend    = identifica_trend(somme)

    print(f"  [Wyckoff] Somma attuale: {somma_att}")
    print(f"  [Wyckoff] Trend:         {trend}")
    print(f"  [Wyckoff] RSI:           {rsi_att:.2f}")
    print(f"  [Wyckoff] ADX:           {adx_att:.2f}")
    print(f"  [Wyckoff] BB upper:      {bbu_att:.1f}")
    print(f"  [Wyckoff] BB lower:      {bbl_att:.1f}")

    print("  [Wyckoff] Mappatura zone storiche...")
    df_zone  = calcola_zone_saturazione(df)
    df_cicli = analizza_cicli(df)

    for _, c in df_cicli.iterrows():
        print(f"  [Wyckoff] Ciclo {int(c['ciclo'])}: "
              f"media={c['somma_media']} "
              f"range=[{c['somma_min']}-{c['somma_max']}]")

    fascia_min, fascia_max = determina_fascia_target(
        somma_att, df_zone, trend
    )
    print(f"  [Wyckoff] Fascia target: {fascia_min}-{fascia_max}")

    # Zona tipo attuale
    mask = ((df_zone['fascia_min'] <= somma_att) &
            (df_zone['fascia_max'] >  somma_att))
    zona_tipo = df_zone[mask].iloc[0]['tipo'] \
                if not df_zone[mask].empty else 'indefinita'

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
