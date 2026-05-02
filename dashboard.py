"""
SSAS - Dashboard Streamlit
Analisi strutturale Superenalotto + Sistema Ridotto
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
from supabase import create_client

# ── Configurazione ────────────────────────────────────────
st.set_page_config(
    page_title="SSAS - Superenalotto Analysis",
    page_icon="🎯",
    layout="wide"
)

# ── Connessione Supabase ──────────────────────────────────
@st.cache_resource
def get_client():
    return create_client(
        st.secrets["URL_SUPABASE"],
        st.secrets["KEY_SUPABASE"]
    )

supabase = get_client()

# ── Caricamento dati ──────────────────────────────────────
@st.cache_data(ttl=3600)
def carica_costanti():
    res = supabase.table("costanti_sistema")\
        .select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def carica_mappa():
    res = supabase.table("mappa_occupazione")\
        .select("*").order("numero").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_estrazioni(limit=7304):
    res = supabase.table("estrazioni")\
        .select("*")\
        .order("data_estrazione", desc=False)\
        .limit(limit)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_wyckoff_stato():
    res = supabase.table("wyckoff_stato")\
        .select("*")\
        .order("run_at", desc=True)\
        .limit(1)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_pool(wyckoff_id):
    res = supabase.table("pool_compensazione")\
        .select("*")\
        .eq("wyckoff_id", wyckoff_id)\
        .eq("incluso", True)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_candidate_frequenze(run_id):
    res = supabase.table("candidate_frequenze")\
        .select("*")\
        .eq("run_id", run_id)\
        .order("pct", desc=True)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_candidate(run_id):
    res = supabase.table("combinazioni_candidate")\
        .select("*")\
        .eq("run_id", run_id)\
        .limit(5000)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_run_ids():
    """
    Restituisce solo i run che hanno frequenze calcolate
    → garantisce che la tab Candidate mostri sempre dati
    """
    res = supabase.table("candidate_frequenze")\
        .select("run_id")\
        .execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return []
    return sorted(df['run_id'].unique().tolist(), reverse=True)

# ── Calcoli Wyckoff ───────────────────────────────────────
def calcola_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcola_bollinger(series, period=137, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + std_dev*std, sma, sma - std_dev*std

# ── Sistema Ridotto ───────────────────────────────────────
def genera_sistema_ridotto(numeri, garanzia=5):
    numeri = sorted(numeri)
    if len(numeri) < 6:
        return [], 0

    target_size  = garanzia
    tutti_target = list(combinations(numeri, target_size))
    target_set   = set(tutti_target)
    tutte_sestine = list(combinations(numeri, 6))

    copertura = {}
    for s in tutte_sestine:
        copre = set()
        for t in combinations(s, target_size):
            if t in target_set:
                copre.add(t)
        copertura[s] = copre

    selezionate = []
    non_coperti = set(tutti_target)

    while non_coperti:
        best     = None
        best_cov = set()
        for s, copre in copertura.items():
            if s in selezionate:
                continue
            nuovi = copre & non_coperti
            if len(nuovi) > len(best_cov):
                best     = s
                best_cov = nuovi
        if best is None or len(best_cov) == 0:
            break
        selezionate.append(best)
        non_coperti -= best_cov

    efficienza = round(
        (1 - len(selezionate)/len(tutte_sestine)) * 100, 1
    ) if tutte_sestine else 0

    return selezionate, efficienza

# ── Header ────────────────────────────────────────────────
st.title("🎯 SSAS — Stochastic Structure Analysis System")
st.caption("Analisi strutturale Superenalotto | Wyckoff + Parisi")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Costanti",
    "🗺️ Mappa 1-90",
    "📈 Wyckoff",
    "🎯 Candidate",
    "🔧 Officina",
    "🔢 Estrazioni"
])

# ════════════════════════════════════════════════════════
# TAB 1 — COSTANTI SISTEMA
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Costanti Strutturali del Sistema")
    st.caption("Calcolate su 7304 estrazioni storiche")

    df_cost = carica_costanti()
    if not df_cost.empty:
        c1, c2, c3, c4 = st.columns(4)
        for col, nome, label in [
            (c1, 'spacing_ratio', 'Spacing Ratio (Wigner-Dyson)'),
            (c2, 'somma',        'Somma Media'),
            (c3, 'cv_gap',       'CV Gap (Disordine)'),
            (c4, 'entropia_gap', 'Entropia Gap'),
        ]:
            r = df_cost[df_cost['nome']==nome]
            if not r.empty:
                r = r.iloc[0]
                col.metric(label,
                           f"{r['valore_medio']:.4f}",
                           delta=f"±{r['std_dev']:.4f}",
                           delta_color="off")

        st.divider()
        rows = []
        for _, r in df_cost.iterrows():
            rows.append({
                "Parametro": r['nome'],
                "Media":     round(r['valore_medio'], 4),
                "Std":       round(r['std_dev'], 4),
                "P5":        round(r['percentile_5'], 4),
                "P95":       round(r['percentile_95'], 4),
                "Sigma":     round(r['sigma_da_random'], 3)
                             if r['sigma_da_random'] else "—",
            })
        st.dataframe(pd.DataFrame(rows),
                     hide_index=True,
                     use_container_width=True)

        st.subheader("Spacing Ratio vs sistemi fisici")
        fig = go.Figure()
        fig.add_vline(x=0.386, line_dash="dash",
                      line_color="orange",
                      annotation_text="Poisson 0.386")
        fig.add_vline(x=0.536, line_dash="dash",
                      line_color="green",
                      annotation_text="GOE 0.536")
        sr = df_cost[df_cost['nome']=='spacing_ratio']
        if not sr.empty:
            v = sr.iloc[0]['valore_medio']
            s = sr.iloc[0]['std_dev']
            fig.add_vline(x=v, line_color="red",
                          annotation_text=f"Sistema {v:.4f}")
            fig.add_vrect(x0=v-s, x1=v+s,
                          fillcolor="red", opacity=0.1)
        fig.update_layout(template="plotly_dark", height=180,
                          margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — MAPPA OCCUPAZIONE
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Mappa Occupazione 1-90")
    df_mappa = carica_mappa()

    if not df_mappa.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Top 10 più frequenti**")
            st.dataframe(
                df_mappa.nlargest(10, 'freq_assoluta')[
                    ['numero','freq_assoluta',
                     'freq_relativa','z_score']
                ],
                hide_index=True, use_container_width=True
            )
        with c2:
            st.write("**Top 10 più ritardatari**")
            st.dataframe(
                df_mappa.nlargest(10, 'ritardo_attuale')[
                    ['numero','ritardo_attuale',
                     'ritardo_medio','ultimo_estratto']
                ],
                hide_index=True, use_container_width=True
            )

        st.divider()
        st.subheader("Heatmap Z-score")
        z_vals = df_mappa['z_score'].values
        numeri = df_mappa['numero'].values
        grid_z = np.zeros((9, 10))
        grid_n = np.zeros((9, 10), dtype=int)
        for n, z in zip(numeri, z_vals):
            r = (n-1) // 10
            c = (n-1) % 10
            grid_z[r][c] = z
            grid_n[r][c] = n

        fig = go.Figure(data=go.Heatmap(
            z=grid_z, text=grid_n,
            texttemplate="%{text}",
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="Z-score")
        ))
        fig.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=20,r=20,t=20,b=20),
                          xaxis_showticklabels=False,
                          yaxis_showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rosso=sopra atteso | Blu=sotto atteso")

# ════════════════════════════════════════════════════════
# TAB 3 — WYCKOFF
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analisi Wyckoff — Serie Storica Somme")

    df_est = carica_estrazioni()
    df_wyk = carica_wyckoff_stato()

    if not df_est.empty:
        cols = ['n1','n2','n3','n4','n5','n6']
        df_est['somma'] = df_est[cols].sum(axis=1)
        df_est['data_estrazione'] = pd.to_datetime(
            df_est['data_estrazione'])
        df_est = df_est.sort_values('data_estrazione')\
                       .reset_index(drop=True)

        somme            = df_est['somma']
        bb_u, bb_m, bb_l = calcola_bollinger(somme, 137)
        rsi              = calcola_rsi(somme, 14)

        # Stato corrente
        if not df_wyk.empty:
            w = df_wyk.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Somma attuale", int(w['somma_ultima']))
            c2.metric("Trend", w['trend'].upper())
            c3.metric("RSI", f"{w['rsi_attuale']:.1f}")
            c4.metric("ADX", f"{w['adx_attuale']:.1f}")
            c5.metric("Fascia target",
                      f"{int(w['fascia_min'])}-"
                      f"{int(w['fascia_max'])}")
            st.info(
                f"Zona attuale: **{w['zona_tipo']}** | "
                f"Cicli analizzati: {w['cicli_analizzati']}"
            )

        st.divider()

        tail    = 500
        df_plot = df_est.tail(tail).copy()
        bb_u_p  = bb_u.tail(tail)
        bb_m_p  = bb_m.tail(tail)
        bb_l_p  = bb_l.tail(tail)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=df_plot['somma'],
            mode='lines', name='Somma',
            line=dict(color='white', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_u_p,
            mode='lines', name='BB Upper(137)',
            line=dict(color='red', dash='dash', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_m_p,
            mode='lines', name='BB Media(137)',
            line=dict(color='yellow', dash='dot', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_l_p,
            mode='lines', name='BB Lower(137)',
            line=dict(color='blue', dash='dash', width=1),
            fill='tonexty', fillcolor='rgba(0,0,255,0.05)'
        ))
        if not df_wyk.empty:
            fig.add_hline(y=w['fascia_min'],
                          line_color="gold", line_dash="dash",
                          annotation_text=f"Target min")
            fig.add_hline(y=w['fascia_max'],
                          line_color="gold", line_dash="dash",
                          annotation_text=f"Target max")
        fig.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

        rsi_plot = rsi.tail(tail)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=rsi_plot,
            mode='lines', name='RSI(14)',
            line=dict(color='purple', width=1.5)
        ))
        fig2.add_hline(y=70, line_color="red",
                       line_dash="dash",
                       annotation_text="Iper-comprato 70")
        fig2.add_hline(y=30, line_color="blue",
                       line_dash="dash",
                       annotation_text="Iper-venduto 30")
        fig2.add_hrect(y0=30, y1=70,
                       fillcolor="grey", opacity=0.05)
        fig2.update_layout(template="plotly_dark", height=200,
                           margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Distribuzione storica somme")
        fig3 = px.histogram(df_est, x='somma', nbins=80,
                            color_discrete_sequence=['#636EFA'])
        if not df_wyk.empty:
            fig3.add_vline(x=w['somma_ultima'],
                           line_color="white",
                           annotation_text="Oggi")
            fig3.add_vrect(
                x0=w['fascia_min'], x1=w['fascia_max'],
                fillcolor="gold", opacity=0.15,
                annotation_text="Target"
            )
        fig3.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — SESTINE CANDIDATE
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Sestine Candidate Wyckoff")
    st.caption(
        "Per ogni run vengono mostrati: i numeri più presenti "
        "nelle 5000 sestine candidate (firma del sistema) "
        "e la lista completa scaricabile."
    )

    run_ids = carica_run_ids()

    if not run_ids:
        st.warning(
            "Nessun run con frequenze disponibile. "
            "Esegui analisi.py su GitHub Actions."
        )
    else:
        import datetime
        run_labels = {
            r: datetime.datetime.fromtimestamp(r)\
               .strftime("%d/%m/%Y %H:%M:%S")
            for r in run_ids
        }
        run_sel = st.selectbox(
            "Seleziona run di analisi:",
            options=run_ids,
            format_func=lambda x: run_labels[x]
        )

        df_freq = carica_candidate_frequenze(run_sel)
        df_cand = carica_candidate(run_sel)

        if df_freq.empty:
            st.warning(
                "Frequenze non disponibili per questo run. "
                "Seleziona un run più recente."
            )
        else:
            st.info(
                f"**{len(df_cand):,}** sestine candidate | "
                f"Run del {run_labels[run_sel]}"
            )

            # ── Grafico frequenze numeri ──────────────────
            st.subheader(
                "Frequenza numeri nelle sestine candidate"
            )
            st.caption(
                "Mostra quanto spesso ogni numero (1-90) "
                "appare nelle sestine candidate. "
                "I numeri ALTI sono quelli selezionati dal "
                "pool Wyckoff. I numeri BASSI sono il 6° "
                "calcolato per chiudere la somma target."
            )

            df_freq_sorted = df_freq.sort_values('numero')
            col_colors = df_freq_sorted['pct'].apply(
                lambda p: '#ff4444' if p >= 15
                else '#ffaa00' if p >= 8
                else '#44ff44'
            )

            fig = go.Figure(go.Bar(
                x=df_freq_sorted['numero'],
                y=df_freq_sorted['pct'],
                marker_color=col_colors,
            ))
            fig.add_hline(y=15, line_dash="dash",
                          line_color="red",
                          annotation_text="ALTA ≥15%")
            fig.add_hline(y=8, line_dash="dash",
                          line_color="orange",
                          annotation_text="MEDIA ≥8%")
            fig.update_layout(
                template="plotly_dark", height=320,
                margin=dict(l=20,r=20,t=20,b=20),
                xaxis_title="Numero",
                yaxis_title="% presenze nelle candidate",
                xaxis=dict(dtick=5)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Colonne per saturazione ───────────────────
            st.subheader("Numeri per livello di saturazione")
            c1, c2, c3 = st.columns(3)

            alta = df_freq[df_freq['pct'] >= 15]\
                .sort_values('pct', ascending=False)
            media = df_freq[
                (df_freq['pct'] >= 8) &
                (df_freq['pct'] < 15)
            ].sort_values('pct', ascending=False)
            bassa = df_freq[df_freq['pct'] < 8]\
                .sort_values('pct', ascending=False)\
                .head(15)

            with c1:
                st.write("🔴 **ALTA saturazione** (≥15%)")
                st.caption("Pool Wyckoff dominante")
                for _, r in alta.iterrows():
                    st.write(
                        f"**N.{int(r['numero']):2d}** → "
                        f"{r['pct']:.1f}%"
                    )
            with c2:
                st.write("🟡 **MEDIA saturazione** (8-15%)")
                for _, r in media.iterrows():
                    st.write(
                        f"**N.{int(r['numero']):2d}** → "
                        f"{r['pct']:.1f}%"
                    )
            with c3:
                st.write("🟢 **Pool Wyckoff attivo**")
                df_wyk_s = carica_wyckoff_stato()
                if not df_wyk_s.empty:
                    pool_df = carica_pool(
                        int(df_wyk_s.iloc[0]['id'])
                    )
                    if not pool_df.empty:
                        nums = sorted(
                            pool_df['numero'].tolist()
                        )
                        # Mostra a gruppi di 5
                        for i in range(0, len(nums), 5):
                            st.write(
                                " ".join(
                                    f"**{n}**"
                                    for n in nums[i:i+5]
                                )
                            )

            st.divider()

            # ── Lista sestine ─────────────────────────────
            st.subheader(
                f"Sestine candidate (prime 50 su "
                f"{len(df_cand):,})"
            )
            if not df_cand.empty:
                cols_n = ['n1','n2','n3','n4','n5','n6']
                df_show = df_cand.head(50)[cols_n].copy()
                df_show.columns = ['N1','N2','N3',
                                   'N4','N5','N6']
                df_show['Somma'] = df_show.sum(axis=1)
                df_show['Range'] = (df_show['N6'] -
                                    df_show['N1'])
                df_show.insert(0, '#',
                               range(1, len(df_show)+1))
                st.dataframe(df_show,
                             hide_index=True,
                             use_container_width=True)

                csv = df_cand[cols_n].to_csv(index=False)
                st.download_button(
                    "⬇️ Scarica tutte le candidate (CSV)",
                    csv,
                    f"candidate_{run_sel}.csv",
                    "text/csv"
                )

# ════════════════════════════════════════════════════════
# TAB 5 — OFFICINA (SISTEMA RIDOTTO)
# ════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔧 Officina — Sistema Ridotto")
    st.caption(
        "Inserisci i tuoi numeri e genera un sistema ridotto "
        "con garanzia 4 o 5. "
        "Garanzia 5 = se i 6 vincenti sono tutti nei tuoi numeri, "
        "almeno una schedina ha 5 giusti."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        numeri_input = st.text_input(
            "Inserisci i numeri (separati da virgola o spazio):",
            placeholder="Es: 7 15 22 35 48 63 71 82"
        )
    with c2:
        garanzia = st.radio(
            "Garanzia:",
            options=[4, 5],
            index=1,
            horizontal=True
        )

    applica_filtri = st.checkbox(
        "Applica filtri strutturali (overlap, spacing, ecc.)",
        value=True
    )

    if st.button("🚀 Genera Sistema Ridotto", type="primary"):
        import re
        nums_raw = re.findall(r'\d+', numeri_input)
        numeri   = sorted(set(
            int(n) for n in nums_raw if 1 <= int(n) <= 90
        ))

        if len(numeri) < 6:
            st.error("Inserisci almeno 6 numeri (1-90).")
        elif len(numeri) > 20:
            st.error("Massimo 20 numeri per volta.")
        else:
            n_int  = len(numeri)
            n_full = len(list(combinations(numeri, 6)))
            st.info(
                f"**{n_int} numeri** → "
                f"Sistema integrale: **{n_full:,}** sestine | "
                f"Garanzia **{garanzia}** selezionata"
            )

            with st.spinner("Calcolo sistema ridotto..."):
                sistema, efficienza = genera_sistema_ridotto(
                    numeri, garanzia
                )

            if not sistema:
                st.error("Impossibile generare il sistema.")
            else:
                st.success(
                    f"Sistema ridotto: **{len(sistema)}** "
                    f"sestine (riduzione **{efficienza}%** "
                    f"rispetto al pieno)"
                )

                if applica_filtri:
                    with st.spinner("Applicazione filtri..."):
                        res_st = supabase.table("estrazioni")\
                            .select("n1,n2,n3,n4,n5,n6")\
                            .limit(10000)\
                            .execute()
                        storico = np.array([
                            sorted([r['n1'],r['n2'],r['n3'],
                                    r['n4'],r['n5'],r['n6']])
                            for r in res_st.data
                        ])
                        mappa_z = {
                            r['numero']: r['z_score'] or 0.0
                            for r in supabase.table(
                                "mappa_occupazione"
                            ).select("numero,z_score")\
                             .execute().data
                        }

                        filtrate = []
                        scarti   = 0
                        for s in sistema:
                            s_arr    = np.array(list(s))
                            overlaps = np.sum(
                                np.isin(storico, s_arr),
                                axis=1
                            )
                            if int(overlaps.max()) >= 4:
                                scarti += 1
                                continue
                            filtrate.append(s)

                    sistema = filtrate
                    st.info(
                        f"Dopo filtri: **{len(sistema)}** "
                        f"sestine ({scarti} scartate)"
                    )

                st.subheader(
                    f"Le {len(sistema)} sestine del sistema"
                )
                righe = []
                for i, s in enumerate(sistema):
                    righe.append({
                        "#":     i+1,
                        "N1": s[0], "N2": s[1], "N3": s[2],
                        "N4": s[3], "N5": s[4], "N6": s[5],
                        "Somma": sum(s),
                        "Range": s[-1] - s[0],
                    })

                df_sis = pd.DataFrame(righe)
                st.dataframe(df_sis,
                             hide_index=True,
                             use_container_width=True)

                from collections import Counter
                tutti    = [n for s in sistema for n in s]
                freq_cnt = Counter(tutti)
                freq_df  = pd.DataFrame([
                    {'numero':   k,
                     'presenze': v,
                     'pct':      round(v*100/len(sistema), 1)}
                    for k, v in sorted(freq_cnt.items())
                ])

                st.subheader("Copertura numeri nel sistema")
                fig_off = px.bar(
                    freq_df, x='numero', y='pct',
                    color='pct',
                    color_continuous_scale='RdYlGn',
                    title="% presenze per numero"
                )
                fig_off.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=20,r=20,t=40,b=20)
                )
                st.plotly_chart(fig_off,
                                use_container_width=True)

                csv_sis = df_sis.to_csv(index=False)
                st.download_button(
                    "⬇️ Scarica sistema (CSV)",
                    csv_sis,
                    f"sistema_ridotto_g{garanzia}.csv",
                    "text/csv"
                )

# ════════════════════════════════════════════════════════
# TAB 6 — ULTIME ESTRAZIONI
# ════════════════════════════════════════════════════════
with tab6:
    st.subheader("Ultime Estrazioni")

    n_show = st.slider("Quante estrazioni:", 5, 100, 20)
    res    = supabase.table("estrazioni")\
        .select("*")\
        .order("data_estrazione", desc=True)\
        .limit(n_show)\
        .execute()
    df_ult = pd.DataFrame(res.data)

    if not df_ult.empty:
        cols_n       = ['n1','n2','n3','n4','n5','n6']
        df_ult['somma'] = df_ult[cols_n].sum(axis=1)
        df_ult['range'] = df_ult['n6'] - df_ult['n1']

        st.dataframe(
            df_ult[
                ['data_estrazione'] + cols_n +
                ['jolly','superstar','somma','range']
            ].rename(columns={
                'data_estrazione': 'Data',
                'n1':'N1','n2':'N2','n3':'N3',
                'n4':'N4','n5':'N5','n6':'N6',
                'jolly':'Jolly','superstar':'Superstar'
            }),
            hide_index=True,
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df_ult.iloc[::-1],
                x='data_estrazione', y='somma',
                title="Somma ultime estrazioni",
                color='somma',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=275.8, line_dash="dash",
                          line_color="white",
                          annotation_text="Media storica")
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20,r=20,t=40,b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(
                df_ult.iloc[::-1],
                x='data_estrazione', y='range',
                title="Range ultime estrazioni",
                color='range',
                color_continuous_scale='Blues'
            )
            fig2.add_hline(y=65.3, line_dash="dash",
                           line_color="white",
                           annotation_text="Media storica")
            fig2.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20,r=20,t=40,b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)
