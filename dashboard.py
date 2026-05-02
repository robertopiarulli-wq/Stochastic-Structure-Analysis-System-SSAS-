"""
SSAS - Dashboard Streamlit
Analisi strutturale Superenalotto + Sistema Ridotto
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import re
from itertools import combinations
from collections import Counter
from supabase import create_client

st.set_page_config(
    page_title="SSAS - Superenalotto Analysis",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def get_client():
    return create_client(
        st.secrets["URL_SUPABASE"],
        st.secrets["KEY_SUPABASE"]
    )

supabase = get_client()

@st.cache_data(ttl=3600)
def carica_costanti():
    res = supabase.table("costanti_sistema").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def carica_mappa():
    res = supabase.table("mappa_occupazione").select("*").order("numero").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_estrazioni(limit=7304):
    res = supabase.table("estrazioni").select("*")\
        .order("data_estrazione", desc=False).limit(limit).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_wyckoff_stato():
    res = supabase.table("wyckoff_stato").select("*")\
        .order("run_at", desc=True).limit(1).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_pool(wyckoff_id):
    res = supabase.table("pool_compensazione").select("*")\
        .eq("wyckoff_id", wyckoff_id).eq("incluso", True).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_candidate_frequenze(run_id):
    res = supabase.table("candidate_frequenze").select("*")\
        .eq("run_id", run_id).order("pct", desc=True).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_candidate(run_id):
    res = supabase.table("combinazioni_candidate").select("*")\
        .eq("run_id", run_id).limit(5000).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_run_ids():
    res = supabase.table("candidate_frequenze").select("run_id").execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return []
    return sorted(df['run_id'].unique().tolist(), reverse=True)

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

# ── Covering design su candidate ─────────────────────────
def genera_ridotto_da_candidate(candidate_list, garanzia=5):
    """
    Trova il minimo sottoinsieme delle candidate
    che copre tutte le combinazioni di <garanzia> numeri
    presenti nel pool delle candidate stesse.
    """
    if not candidate_list:
        return [], 0, []

    pool         = sorted(set(n for s in candidate_list for n in s))
    tutti_target = list(combinations(pool, garanzia))
    n_target     = len(tutti_target)

    target_idx        = {t: i for i, t in enumerate(tutti_target)}
    sestina_to_targets = {}
    target_to_sestine  = [[] for _ in range(n_target)]

    for s in candidate_list:
        idxs = []
        for t in combinations(s, garanzia):
            if t in target_idx:
                i = target_idx[t]
                idxs.append(i)
                target_to_sestine[i].append(s)
        sestina_to_targets[s] = idxs

    selezionate = []
    non_coperti = set(range(n_target))
    punteggi    = {
        s: len(set(sestina_to_targets[s]) & non_coperti)
        for s in candidate_list
    }

    while non_coperti:
        if not punteggi:
            break
        best = max(punteggi, key=lambda s: punteggi[s])
        if punteggi[best] == 0:
            break

        selezionate.append(best)
        nuovi_coperti = set(sestina_to_targets[best]) & non_coperti
        non_coperti  -= nuovi_coperti
        punteggi[best] = -1

        da_aggiornare = set()
        for idx in nuovi_coperti:
            for s in target_to_sestine[idx]:
                da_aggiornare.add(s)
        for s in da_aggiornare:
            if punteggi.get(s, -1) >= 0:
                punteggi[s] = len(
                    set(sestina_to_targets[s]) & non_coperti
                )

    efficienza = round(
        (1 - len(selezionate)/len(candidate_list)) * 100, 1
    ) if candidate_list else 0

    return selezionate, efficienza, pool

# ── Sistema Ridotto manuale ───────────────────────────────
def genera_sistema_ridotto(numeri, garanzia=5):
    numeri = sorted(numeri)
    if len(numeri) < 6:
        return [], 0

    tutti_target  = list(combinations(numeri, garanzia))
    tutte_sestine = list(combinations(numeri, 6))
    target_idx    = {t: i for i, t in enumerate(tutti_target)}

    sestina_to_targets = {}
    target_to_sestine  = [[] for _ in range(len(tutti_target))]

    for s in tutte_sestine:
        idxs = []
        for t in combinations(s, garanzia):
            if t in target_idx:
                i = target_idx[t]
                idxs.append(i)
                target_to_sestine[i].append(s)
        sestina_to_targets[s] = idxs

    selezionate = []
    non_coperti = set(range(len(tutti_target)))
    punteggi    = {
        s: len(set(sestina_to_targets[s]) & non_coperti)
        for s in tutte_sestine
    }

    while non_coperti:
        if not punteggi:
            break
        best = max(punteggi, key=lambda s: punteggi[s])
        if punteggi[best] == 0:
            break
        selezionate.append(best)
        nuovi_coperti = set(sestina_to_targets[best]) & non_coperti
        non_coperti  -= nuovi_coperti
        punteggi[best] = -1
        da_aggiornare = set()
        for idx in nuovi_coperti:
            for s in target_to_sestine[idx]:
                da_aggiornare.add(s)
        for s in da_aggiornare:
            if punteggi.get(s, -1) >= 0:
                punteggi[s] = len(
                    set(sestina_to_targets[s]) & non_coperti
                )

    efficienza = round(
        (1 - len(selezionate)/len(tutte_sestine)) * 100, 1
    ) if tutte_sestine else 0

    return selezionate, efficienza

def mostra_sistema(sistema, garanzia, key_prefix):
    righe = []
    for i, s in enumerate(sistema):
        righe.append({
            "#": i+1,
            "N1": s[0], "N2": s[1], "N3": s[2],
            "N4": s[3], "N5": s[4], "N6": s[5],
            "Somma": sum(s), "Range": s[-1]-s[0],
        })
    df_r = pd.DataFrame(righe)
    st.dataframe(df_r, hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Scarica sistema (CSV)",
        df_r.to_csv(index=False),
        f"ridotto_{key_prefix}_g{garanzia}.csv",
        "text/csv",
        key=f"dl_{key_prefix}"
    )

# ── Header ────────────────────────────────────────────────
st.title("🎯 SSAS — Stochastic Structure Analysis System")
st.caption("Analisi strutturale Superenalotto | Wyckoff + Parisi")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Costanti", "🗺️ Mappa 1-90", "📈 Wyckoff",
    "🎯 Candidate", "🔧 Officina", "🔢 Estrazioni"
])

# ════════════════════════════════════════════════════════
# TAB 1 — COSTANTI
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Costanti Strutturali del Sistema")
    df_cost = carica_costanti()
    if not df_cost.empty:
        c1, c2, c3, c4 = st.columns(4)
        for col, nome, label in [
            (c1, 'spacing_ratio', 'Spacing Ratio'),
            (c2, 'somma',        'Somma Media'),
            (c3, 'cv_gap',       'CV Gap'),
            (c4, 'entropia_gap', 'Entropia Gap'),
        ]:
            r = df_cost[df_cost['nome']==nome]
            if not r.empty:
                r = r.iloc[0]
                col.metric(label, f"{r['valore_medio']:.4f}",
                           delta=f"±{r['std_dev']:.4f}",
                           delta_color="off")

        st.divider()
        rows = []
        for _, r in df_cost.iterrows():
            rows.append({
                "Parametro": r['nome'],
                "Media":  round(r['valore_medio'], 4),
                "Std":    round(r['std_dev'], 4),
                "P5":     round(r['percentile_5'], 4),
                "P95":    round(r['percentile_95'], 4),
                "Sigma":  round(r['sigma_da_random'], 3)
                          if r['sigma_da_random'] else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

        st.subheader("Spacing Ratio vs sistemi fisici")
        fig = go.Figure()
        fig.add_vline(x=0.386, line_dash="dash",
                      line_color="orange", annotation_text="Poisson 0.386")
        fig.add_vline(x=0.536, line_dash="dash",
                      line_color="green", annotation_text="GOE 0.536")
        sr = df_cost[df_cost['nome']=='spacing_ratio']
        if not sr.empty:
            v = sr.iloc[0]['valore_medio']
            s = sr.iloc[0]['std_dev']
            fig.add_vline(x=v, line_color="red",
                          annotation_text=f"Sistema {v:.4f}")
            fig.add_vrect(x0=v-s, x1=v+s, fillcolor="red", opacity=0.1)
        fig.update_layout(template="plotly_dark", height=180,
                          margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — MAPPA
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
                    ['numero','freq_assoluta','freq_relativa','z_score']],
                hide_index=True, use_container_width=True)
        with c2:
            st.write("**Top 10 più ritardatari**")
            st.dataframe(
                df_mappa.nlargest(10, 'ritardo_attuale')[
                    ['numero','ritardo_attuale','ritardo_medio','ultimo_estratto']],
                hide_index=True, use_container_width=True)

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
            z=grid_z, text=grid_n, texttemplate="%{text}",
            colorscale="RdBu_r", zmid=0, colorbar=dict(title="Z-score")))
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
        df_est['data_estrazione'] = pd.to_datetime(df_est['data_estrazione'])
        df_est = df_est.sort_values('data_estrazione').reset_index(drop=True)
        somme = df_est['somma']
        bb_u, bb_m, bb_l = calcola_bollinger(somme, 137)
        rsi = calcola_rsi(somme, 14)

        if not df_wyk.empty:
            w = df_wyk.iloc[0]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Somma attuale", int(w['somma_ultima']))
            c2.metric("Trend", w['trend'].upper())
            c3.metric("RSI", f"{w['rsi_attuale']:.1f}")
            c4.metric("ADX", f"{w['adx_attuale']:.1f}")
            c5.metric("Fascia target",
                      f"{int(w['fascia_min'])}-{int(w['fascia_max'])}")
            st.info(f"Zona: **{w['zona_tipo']}** | Cicli: {w['cicli_analizzati']}")

        st.divider()
        tail    = 500
        df_plot = df_est.tail(tail).copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['data_estrazione'],
                                  y=df_plot['somma'], mode='lines',
                                  name='Somma', line=dict(color='white', width=1)))
        fig.add_trace(go.Scatter(x=df_plot['data_estrazione'],
                                  y=bb_u.tail(tail), mode='lines',
                                  name='BB Upper(137)',
                                  line=dict(color='red', dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=df_plot['data_estrazione'],
                                  y=bb_m.tail(tail), mode='lines',
                                  name='BB Media(137)',
                                  line=dict(color='yellow', dash='dot', width=1)))
        fig.add_trace(go.Scatter(x=df_plot['data_estrazione'],
                                  y=bb_l.tail(tail), mode='lines',
                                  name='BB Lower(137)',
                                  line=dict(color='blue', dash='dash', width=1),
                                  fill='tonexty',
                                  fillcolor='rgba(0,0,255,0.05)'))
        if not df_wyk.empty:
            fig.add_hline(y=w['fascia_min'], line_color="gold",
                          line_dash="dash", annotation_text="Target min")
            fig.add_hline(y=w['fascia_max'], line_color="gold",
                          line_dash="dash", annotation_text="Target max")
        fig.update_layout(template="plotly_dark", height=350,
                          margin=dict(l=20,r=20,t=20,b=20),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_plot['data_estrazione'],
                                   y=rsi.tail(tail), mode='lines',
                                   name='RSI(14)',
                                   line=dict(color='purple', width=1.5)))
        fig2.add_hline(y=70, line_color="red", line_dash="dash",
                       annotation_text="Iper-comprato 70")
        fig2.add_hline(y=30, line_color="blue", line_dash="dash",
                       annotation_text="Iper-venduto 30")
        fig2.add_hrect(y0=30, y1=70, fillcolor="grey", opacity=0.05)
        fig2.update_layout(template="plotly_dark", height=200,
                           margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Distribuzione storica somme")
        fig3 = px.histogram(df_est, x='somma', nbins=80,
                            color_discrete_sequence=['#636EFA'])
        if not df_wyk.empty:
            fig3.add_vline(x=w['somma_ultima'], line_color="white",
                           annotation_text="Oggi")
            fig3.add_vrect(x0=w['fascia_min'], x1=w['fascia_max'],
                           fillcolor="gold", opacity=0.15,
                           annotation_text="Target")
        fig3.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — CANDIDATE
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Sestine Candidate Wyckoff")
    run_ids = carica_run_ids()

    if not run_ids:
        st.warning("Nessun run disponibile. Esegui analisi.py.")
    else:
        run_labels = {
            r: datetime.datetime.fromtimestamp(r).strftime("%d/%m/%Y %H:%M:%S")
            for r in run_ids
        }
        run_sel = st.selectbox("Seleziona run:", options=run_ids,
                               format_func=lambda x: run_labels[x])
        df_freq = carica_candidate_frequenze(run_sel)
        df_cand = carica_candidate(run_sel)

        if df_freq.empty:
            st.warning("Frequenze non disponibili.")
        else:
            st.info(f"**{len(df_cand):,}** sestine | Run del {run_labels[run_sel]}")

            st.subheader("Frequenza numeri nelle candidate")
            df_fs = df_freq.sort_values('numero')
            col_colors = df_fs['pct'].apply(
                lambda p: '#ff4444' if p >= 15 else '#ffaa00' if p >= 8 else '#44ff44')
            fig = go.Figure(go.Bar(x=df_fs['numero'], y=df_fs['pct'],
                                   marker_color=col_colors))
            fig.add_hline(y=15, line_dash="dash", line_color="red",
                          annotation_text="ALTA ≥15%")
            fig.add_hline(y=8, line_dash="dash", line_color="orange",
                          annotation_text="MEDIA ≥8%")
            fig.update_layout(template="plotly_dark", height=320,
                              margin=dict(l=20,r=20,t=20,b=20),
                              xaxis_title="Numero", yaxis_title="% presenze",
                              xaxis=dict(dtick=5))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            alta  = df_freq[df_freq['pct'] >= 15].sort_values('pct', ascending=False)
            media = df_freq[(df_freq['pct'] >= 8) & (df_freq['pct'] < 15)]\
                .sort_values('pct', ascending=False)
            with c1:
                st.write("🔴 **ALTA** (≥15%)")
                for _, r in alta.iterrows():
                    st.write(f"**N.{int(r['numero']):2d}** → {r['pct']:.1f}%")
            with c2:
                st.write("🟡 **MEDIA** (8-15%)")
                for _, r in media.iterrows():
                    st.write(f"**N.{int(r['numero']):2d}** → {r['pct']:.1f}%")
            with c3:
                st.write("🟢 **Pool Wyckoff attivo**")
                df_wyk_t = carica_wyckoff_stato()
                if not df_wyk_t.empty:
                    pool_df = carica_pool(int(df_wyk_t.iloc[0]['id']))
                    if not pool_df.empty:
                        nums = sorted(pool_df['numero'].tolist())
                        for i in range(0, len(nums), 5):
                            st.write(" ".join(f"**{n}**" for n in nums[i:i+5]))

            st.divider()
            st.subheader("Prime 50 sestine candidate")
            if not df_cand.empty:
                cols_n  = ['n1','n2','n3','n4','n5','n6']
                df_show = df_cand.head(50)[cols_n].copy()
                df_show.columns = ['N1','N2','N3','N4','N5','N6']
                df_show['Somma'] = df_show.sum(axis=1)
                df_show['Range'] = df_show['N6'] - df_show['N1']
                df_show.insert(0, '#', range(1, len(df_show)+1))
                st.dataframe(df_show, hide_index=True, use_container_width=True)
                st.download_button("⬇️ Scarica tutte (CSV)",
                                   df_cand[cols_n].to_csv(index=False),
                                   f"candidate_{run_sel}.csv", "text/csv")

# ════════════════════════════════════════════════════════
# TAB 5 — OFFICINA
# ════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔧 Officina — Sistema Ridotto")

    df_wyk_off  = carica_wyckoff_stato()
    run_ids_off = carica_run_ids()

    fmin = fmax = None
    if not df_wyk_off.empty:
        w_off = df_wyk_off.iloc[0]
        fmin  = int(w_off['fascia_min'])
        fmax  = int(w_off['fascia_max'])

    # ── SEZIONE A: AUTOMATICA ────────────────────────────
    st.markdown("### 🤖 Sistema Ridotto Automatico")
    st.caption(
        "Parte dalle sestine candidate già filtrate per somma "
        "nel target Wyckoff. Trova il minimo sottoinsieme "
        "con garanzia 4 o 5. Le sestine sono già valide."
    )

    if df_wyk_off.empty or not run_ids_off:
        st.warning("Dati non disponibili. Esegui analisi.py.")
    else:
        st.info(
            f"🎯 Target Wyckoff: **{fmin}-{fmax}** | "
            f"Trend: **{w_off['trend'].upper()}** | "
            f"Zona: **{w_off['zona_tipo']}**"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            run_labels_off = {
                r: datetime.datetime.fromtimestamp(r).strftime("%d/%m/%Y %H:%M")
                for r in run_ids_off
            }
            run_auto = st.selectbox(
                "Run candidate:",
                options=run_ids_off,
                format_func=lambda x: run_labels_off[x],
                key="run_auto"
            )
        with c2:
            n_cand_auto = st.slider(
                "Candidate da usare:",
                min_value=50, max_value=1000,
                value=200, step=50,
                key="n_cand_auto",
                help="Quante sestine in target usare per il ridotto. "
                     "Più sono, più il ridotto è rappresentativo "
                     "ma il calcolo è più lento."
            )
        with c3:
            garanzia_auto = st.radio(
                "Garanzia:", options=[4, 5],
                index=1, horizontal=True,
                key="gar_auto"
            )

        if st.button("🎯 Genera Sistema Automatico",
                     type="primary", key="btn_auto"):

            df_cand_auto = carica_candidate(run_auto)

            if df_cand_auto.empty:
                st.error("Nessuna candidata nel run.")
            else:
                cols_n = ['n1','n2','n3','n4','n5','n6']
                df_cand_auto['somma'] = df_cand_auto[cols_n].sum(axis=1)

                df_in_target = df_cand_auto[
                    (df_cand_auto['somma'] >= fmin) &
                    (df_cand_auto['somma'] <= fmax)
                ]

                n_in = len(df_in_target)
                st.write(
                    f"Candidate nel target {fmin}-{fmax}: "
                    f"**{n_in:,}** su **{len(df_cand_auto):,}**"
                )

                if df_in_target.empty:
                    st.error("Nessuna candidata nel target. Rilancia analisi.py.")
                else:
                    # Converti e limita a N candidate
                    candidate_list = [
                        tuple(sorted([
                            row['n1'], row['n2'], row['n3'],
                            row['n4'], row['n5'], row['n6']
                        ]))
                        for _, row in df_in_target.iterrows()
                    ]
                    candidate_list = candidate_list[:n_cand_auto]

                    st.write(
                        f"Usando le prime **{len(candidate_list)}** "
                        f"candidate in target per il covering design"
                    )

                    pool = sorted(set(n for s in candidate_list for n in s))
                    n_cover = len(list(combinations(pool, garanzia_auto)))

                    st.info(
                        f"Pool: **{len(pool)} numeri** unici | "
                        f"Combinazioni da coprire: **{n_cover:,}** | "
                        f"Garanzia **{garanzia_auto}**"
                    )

                    with st.spinner(
                        f"Covering design su {len(candidate_list)} sestine..."
                    ):
                        sistema, efficienza, pool_out = \
                            genera_ridotto_da_candidate(
                                candidate_list, garanzia_auto
                            )

                    st.success(
                        f"Sistema ridotto: **{len(sistema)}** sestine | "
                        f"Partenza: **{len(candidate_list)}** | "
                        f"Riduzione: **{efficienza}%** | "
                        f"Garanzia: **{garanzia_auto}**"
                    )
                    st.caption(f"Pool numeri: {sorted(pool_out)}")

                    if sistema:
                        st.subheader(
                            f"{len(sistema)} sestine nel target {fmin}-{fmax}"
                        )
                        mostra_sistema(sistema, garanzia_auto, "auto")

                        tutti_a  = [n for s in sistema for n in s]
                        freq_a   = Counter(tutti_a)
                        freq_df_a = pd.DataFrame([
                            {'numero': k, 'presenze': v,
                             'pct': round(v*100/len(sistema), 1)}
                            for k, v in sorted(freq_a.items())
                        ])
                        fig_a = px.bar(freq_df_a, x='numero', y='pct',
                                       color='pct',
                                       color_continuous_scale='RdYlGn',
                                       title="Copertura numeri nel sistema")
                        fig_a.update_layout(template="plotly_dark", height=250,
                                            margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig_a, use_container_width=True)

    st.divider()

    # ── SEZIONE B: MANUALE ───────────────────────────────
    st.markdown("### ✏️ Selezione Manuale")
    st.caption("Inserisci i tuoi numeri e genera il sistema ridotto.")

    c1, c2 = st.columns([3, 1])
    with c1:
        numeri_input = st.text_input(
            "Numeri (separati da virgola o spazio):",
            placeholder="Es: 7 15 22 35 48 63 71 82"
        )
    with c2:
        garanzia_man = st.radio(
            "Garanzia:", options=[4, 5],
            index=1, horizontal=True, key="gar_man"
        )

    filtra_target = st.checkbox(
        "Mostra solo sestine nel target Wyckoff", value=True
    )

    if st.button("🚀 Genera Sistema Manuale",
                 type="primary", key="btn_man"):

        nums_raw = re.findall(r'\d+', numeri_input)
        numeri   = sorted(set(
            int(n) for n in nums_raw if 1 <= int(n) <= 90
        ))

        if len(numeri) < 6:
            st.error("Inserisci almeno 6 numeri (1-90).")
        else:
            if fmin and fmax:
                s_min_m = sum(sorted(numeri)[:6])
                s_max_m = sum(sorted(numeri)[-6:])
                if filtra_target and (s_max_m < fmin or s_min_m > fmax):
                    st.warning(
                        f"⚠️ Somme possibili {s_min_m}-{s_max_m} "
                        f"fuori dal target {fmin}-{fmax}."
                    )

            n_full_m = len(list(combinations(numeri, 6)))
            st.info(
                f"**{len(numeri)} numeri** → "
                f"Integrale: **{n_full_m:,}** sestine | "
                f"Garanzia **{garanzia_man}**"
            )

            with st.spinner("Calcolo sistema ridotto..."):
                sistema_m, efficienza_m = \
                    genera_sistema_ridotto(numeri, garanzia_man)

            if not sistema_m:
                st.error("Impossibile generare il sistema.")
            else:
                if filtra_target and fmin and fmax:
                    sis_show   = [s for s in sistema_m
                                  if fmin <= sum(s) <= fmax]
                    label_filt = f"nel target {fmin}-{fmax}"
                else:
                    sis_show   = sistema_m
                    label_filt = "totali"

                st.success(
                    f"Ridotto: **{len(sistema_m)}** totali | "
                    f"**{len(sis_show)}** {label_filt} | "
                    f"Riduzione **{efficienza_m}%**"
                )

                if sis_show:
                    st.subheader(f"{len(sis_show)} sestine {label_filt}")
                    mostra_sistema(sis_show, garanzia_man, "man")
                else:
                    st.warning("Nessuna sestina nel target. "
                               "Deseleziona filtro o cambia numeri.")

                tutti_m   = [n for s in sistema_m for n in s]
                freq_m    = Counter(tutti_m)
                freq_df_m = pd.DataFrame([
                    {'numero': k, 'presenze': v,
                     'pct': round(v*100/len(sistema_m), 1)}
                    for k, v in sorted(freq_m.items())
                ])
                fig_man = px.bar(freq_df_m, x='numero', y='pct',
                                 color='pct',
                                 color_continuous_scale='RdYlGn',
                                 title="Copertura numeri nel sistema")
                fig_man.update_layout(template="plotly_dark", height=250,
                                      margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_man, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 6 — ULTIME ESTRAZIONI
# ════════════════════════════════════════════════════════
with tab6:
    st.subheader("Ultime Estrazioni")
    n_show = st.slider("Quante estrazioni:", 5, 100, 20)
    res    = supabase.table("estrazioni").select("*")\
        .order("data_estrazione", desc=True).limit(n_show).execute()
    df_ult = pd.DataFrame(res.data)

    if not df_ult.empty:
        cols_n = ['n1','n2','n3','n4','n5','n6']
        df_ult['somma'] = df_ult[cols_n].sum(axis=1)
        df_ult['range'] = df_ult['n6'] - df_ult['n1']

        st.dataframe(
            df_ult[['data_estrazione'] + cols_n +
                   ['jolly','superstar','somma','range']].rename(columns={
                'data_estrazione':'Data','n1':'N1','n2':'N2','n3':'N3',
                'n4':'N4','n5':'N5','n6':'N6',
                'jolly':'Jolly','superstar':'Superstar'}),
            hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_ult.iloc[::-1], x='data_estrazione', y='somma',
                         title="Somma ultime estrazioni",
                         color='somma', color_continuous_scale='RdYlGn')
            fig.add_hline(y=275.8, line_dash="dash", line_color="white",
                          annotation_text="Media storica")
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(df_ult.iloc[::-1], x='data_estrazione', y='range',
                          title="Range ultime estrazioni",
                          color='range', color_continuous_scale='Blues')
            fig2.add_hline(y=65.3, line_dash="dash", line_color="white",
                           annotation_text="Media storica")
            fig2.update_layout(template="plotly_dark", height=300,
                               margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)
