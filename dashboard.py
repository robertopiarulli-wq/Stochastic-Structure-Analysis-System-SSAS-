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
    res = supabase.table("mappa_occupazione")\
        .select("*").order("numero").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_estrazioni(limit=7304):
    res = supabase.table("estrazioni").select("*")\
        .order("data_estrazione", desc=False)\
        .limit(limit).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=60)
def carica_wyckoff_stato():
    res = supabase.table("wyckoff_stato").select("*")\
        .order("run_at", desc=True).limit(1).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=60)
def carica_pool(wyckoff_id):
    res = supabase.table("pool_compensazione").select("*")\
        .eq("wyckoff_id", wyckoff_id)\
        .eq("incluso", True).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=60)
def carica_candidate_frequenze(run_id):
    res = supabase.table("candidate_frequenze").select("*")\
        .eq("run_id", run_id)\
        .order("pct", desc=True).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=60)
def carica_candidate(run_id):
    res = supabase.table("combinazioni_candidate").select("*")\
        .eq("run_id", run_id).limit(50000).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=60)
def carica_run_ids():
    res = supabase.table("candidate_frequenze")\
        .select("run_id").execute()
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

# ── Covering design corretto su candidate ────────────────
def genera_ridotto_da_candidate(candidate_list, garanzia=5):
    """
    Logica corretta del sistema ridotto su sestine
    pre-generate:

    Trova il minimo sottoinsieme S delle candidate tale che:
    per ogni sestina c NON in S, esiste almeno una sestina
    s in S che condivide >= <garanzia> numeri con c.

    Garanzia 5 = ogni sestina non giocata è "coperta"
                 da una giocata che ha 5 numeri uguali
    Garanzia 4 = ogni sestina non giocata è "coperta"
                 da una giocata che ha 4 numeri uguali

    NON calcola C(pool, garanzia) — evita esplosione
    combinatoriale.
    """
    n = len(candidate_list)
    if n == 0:
        return [], 0, []
    if n <= 6:
        pool = sorted(set(x for s in candidate_list for x in s))
        return candidate_list, 0, pool

    # Precalcola overlap tra ogni coppia di sestine
    # copre[i] = set di indici j che i copre
    # (overlap >= garanzia)
    sets = [set(s) for s in candidate_list]
    copre = [set() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                if len(sets[i] & sets[j]) >= garanzia:
                    copre[i].add(j)

    # Greedy: seleziona la sestina che copre
    # più sestine non ancora coperte
    selezionate_idx = []
    non_coperti     = set(range(n))
    punteggi        = {
        i: len(copre[i] & non_coperti) for i in range(n)
    }

    while non_coperti:
        # Sestina con punteggio massimo
        best = max(range(n),
                   key=lambda i: punteggi.get(i, -1))

        if punteggi.get(best, 0) == 0:
            # Nessuna copre le restanti →
            # aggiungile tutte
            for j in list(non_coperti):
                if j not in selezionate_idx:
                    selezionate_idx.append(j)
            break

        selezionate_idx.append(best)
        newly = copre[best] & non_coperti
        non_coperti -= newly
        non_coperti.discard(best)
        punteggi[best] = -1  # escludi

        # Aggiorna punteggi
        for i in non_coperti:
            if punteggi.get(i, -1) >= 0:
                punteggi[i] = len(copre[i] & non_coperti)

    sistema     = [candidate_list[i] for i in selezionate_idx]
    pool_out    = sorted(set(x for s in sistema for x in s))
    efficienza  = round(
        (1 - len(sistema)/n) * 100, 1
    ) if n > 0 else 0

    return sistema, efficienza, pool_out

# ── Sistema Ridotto manuale (C(N,6)) ─────────────────────
def genera_sistema_ridotto(numeri, garanzia=5):
    """
    Sistema ridotto classico su N numeri inseriti
    manualmente. Trova minimo sottoinsieme di C(N,6)
    tale che ogni combinazione di <garanzia> numeri
    tra gli N sia coperta da almeno una sestina.
    """
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
        nuovi = set(sestina_to_targets[best]) & non_coperti
        non_coperti -= nuovi
        punteggi[best] = -1
        da_aggiornare = set()
        for idx in nuovi:
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Costanti", "🗺️ Mappa 1-90", "📈 Wyckoff",
    "🎯 Candidate", "🔧 Officina", "🔢 Estrazioni",
    "🅱️ Piano B"
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
        sr = df_cost[df_cost['nome']=='spacing_ratio']
        if not sr.empty:
            v = float(sr.iloc[0]['valore_medio'])
            s = float(sr.iloc[0]['std_dev'])

            fig = go.Figure()
            # Traccia fittizia per dare range all'asse x
            fig.add_trace(go.Scatter(
                x=[0.20, 0.70], y=[0.5, 0.5],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
            fig.add_vline(x=0.386, line_dash="dash",
                          line_color="orange", line_width=2,
                          annotation_text="Poisson 0.386",
                          annotation_position="top right",
                          annotation_font_color="orange")
            fig.add_vline(x=0.536, line_dash="dash",
                          line_color="lime", line_width=2,
                          annotation_text="GOE 0.536",
                          annotation_position="top right",
                          annotation_font_color="lime")
            fig.add_vline(x=v, line_color="red", line_width=3,
                          annotation_text=f"Superenalotto {v:.4f}",
                          annotation_position="top left",
                          annotation_font_color="red")
            fig.add_vrect(x0=v-s, x1=v+s,
                          fillcolor="red", opacity=0.15,
                          annotation_text=f"±σ",
                          annotation_position="top left")
            fig.update_layout(
                template="plotly_dark",
                height=220,
                xaxis=dict(
                    range=[0.20, 0.70],
                    title="Spacing Ratio",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)"
                ),
                yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "**Cos'è lo Spacing Ratio?** "
                "Misura come sono distribuiti i gap tra i 6 "
                "numeri di ogni sestina. "
                "**Poisson (0.386)** = numeri estratti "
                "in modo completamente casuale, "
                "gap distribuiti esponenzialmente. "
                "**GOE (0.536)** = sistemi quantistici con "
                "repulsione tra livelli energetici "
                "(governa atomi, neutroni, mercati finanziari). "
                f"**Superenalotto ({v:.4f})** si posiziona "
                "tra i due: né puro caos né struttura forte. "
                "Sigma vicino a zero conferma: "
                "il sistema è essenzialmente random."
            )

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
                    ['numero','freq_assoluta',
                     'freq_relativa','z_score']],
                hide_index=True, use_container_width=True)
        with c2:
            st.write("**Top 10 più ritardatari**")
            st.dataframe(
                df_mappa.nlargest(10, 'ritardo_attuale')[
                    ['numero','ritardo_attuale',
                     'ritardo_medio','ultimo_estratto']],
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
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="Z-score")))
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
            st.info(f"Zona: **{w['zona_tipo']}** | "
                    f"Cicli: {w['cicli_analizzati']}")

            # Vincolo parità se disponibile
            n_p    = w.get('vincolo_n_pari')
            pct_p  = w.get('vincolo_pct_pari')
            logica = w.get('vincolo_logica', '')
            if n_p is not None and str(n_p) != 'nan':
                try:
                    n_p   = int(float(n_p))
                    pct_p = float(pct_p)
                    # Usa logica da DB, fallback semplice
                    if logica and str(logica) != 'nan':
                        descr = f"**{logica}**"
                    else:
                        descr = f"**{n_p}p/{6-n_p}d**"
                    st.success(
                        f"🎲 Vincolo parità: {descr} | "
                        f"Frequenza storica nella fascia: "
                        f"**{pct_p:.1f}%**"
                    )
                except Exception:
                    pass

        st.divider()
        tail    = 500
        df_plot = df_est.tail(tail).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=df_plot['somma'],
            mode='lines', name='Somma',
            line=dict(color='white', width=1)))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_u.tail(tail),
            mode='lines', name='BB Upper(137)',
            line=dict(color='red', dash='dash', width=1)))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_m.tail(tail),
            mode='lines', name='BB Media(137)',
            line=dict(color='yellow', dash='dot', width=1)))
        fig.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=bb_l.tail(tail),
            mode='lines', name='BB Lower(137)',
            line=dict(color='blue', dash='dash', width=1),
            fill='tonexty', fillcolor='rgba(0,0,255,0.05)'))
        if not df_wyk.empty:
            fig.add_hline(y=w['fascia_min'], line_color="gold",
                          line_dash="dash",
                          annotation_text="Target min")
            fig.add_hline(y=w['fascia_max'], line_color="gold",
                          line_dash="dash",
                          annotation_text="Target max")
        fig.update_layout(template="plotly_dark", height=350,
                          margin=dict(l=20,r=20,t=20,b=20),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_plot['data_estrazione'], y=rsi.tail(tail),
            mode='lines', name='RSI(14)',
            line=dict(color='purple', width=1.5)))
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
            fig3.add_vrect(x0=w['fascia_min'],
                           x1=w['fascia_max'],
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
        st.warning("Nessun run disponibile.")
    else:
        run_labels = {
            r: datetime.datetime.fromtimestamp(r)\
               .strftime("%d/%m/%Y %H:%M:%S")
            for r in run_ids
        }
        run_sel = st.selectbox(
            "Seleziona run:", options=run_ids,
            format_func=lambda x: run_labels[x])
        df_freq = carica_candidate_frequenze(run_sel)
        df_cand = carica_candidate(run_sel)

        if df_freq.empty:
            st.warning("Frequenze non disponibili.")
        else:
            # ── Carica vincolo parità dal DB ─────────────────
            df_wyk_t = carica_wyckoff_stato()
            vincolo_n_pari = None
            vincolo_pct    = None
            if not df_wyk_t.empty:
                w_t = df_wyk_t.iloc[0]
                _np = w_t.get('vincolo_n_pari')
                _pp = w_t.get('vincolo_pct_pari')
                if _np is not None and str(_np) != 'nan':
                    try:
                        vincolo_n_pari = int(float(_np))
                        vincolo_pct    = float(_pp)
                    except Exception:
                        pass

            # ── Toggle filtro parità ──────────────────────────
            cols_n = ['n1','n2','n3','n4','n5','n6']
            if vincolo_n_pari is not None:
                usa_parita = st.checkbox(
                    f"🎲 Filtra per vincolo parità: "
                    f"**{vincolo_n_pari}p/{6-vincolo_n_pari}d** "
                    f"({vincolo_pct:.1f}% nella fascia storica)",
                    value=True,
                    key="chk_parita"
                )
                if usa_parita:
                    mask_p = df_cand[cols_n].apply(
                        lambda r: sum(1 for v in r if v % 2 == 0)
                        == vincolo_n_pari, axis=1
                    )
                    df_cand_use = df_cand[mask_p].copy()
                    st.info(
                        f"**{len(df_cand_use):,}** sestine "
                        f"con vincolo parità | "
                        f"**{len(df_cand):,}** totali nel run"
                    )
                else:
                    df_cand_use = df_cand.copy()
                    st.info(f"**{len(df_cand_use):,}** sestine | "
                            f"Run del {run_labels[run_sel]}")
            else:
                df_cand_use = df_cand.copy()
                st.info(f"**{len(df_cand_use):,}** sestine | "
                        f"Run del {run_labels[run_sel]}")

            st.subheader("Frequenza numeri nelle candidate")
            df_fs = df_freq.sort_values('numero')
            col_colors = df_fs['pct'].apply(
                lambda p: '#ff4444' if p >= 15
                else '#ffaa00' if p >= 8 else '#44ff44')
            fig = go.Figure(go.Bar(
                x=df_fs['numero'], y=df_fs['pct'],
                marker_color=col_colors))
            fig.add_hline(y=15, line_dash="dash",
                          line_color="red",
                          annotation_text="ALTA ≥15%")
            fig.add_hline(y=8, line_dash="dash",
                          line_color="orange",
                          annotation_text="MEDIA ≥8%")
            fig.update_layout(template="plotly_dark",
                              height=320,
                              margin=dict(l=20,r=20,t=20,b=20),
                              xaxis_title="Numero",
                              yaxis_title="% presenze",
                              xaxis=dict(dtick=5))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            alta  = df_freq[df_freq['pct'] >= 15]\
                .sort_values('pct', ascending=False)
            media = df_freq[(df_freq['pct'] >= 8) &
                            (df_freq['pct'] < 15)]\
                .sort_values('pct', ascending=False)
            with c1:
                st.write("🔴 **ALTA** (≥15%)")
                for _, r in alta.iterrows():
                    st.write(f"**N.{int(r['numero']):2d}** → "
                             f"{r['pct']:.1f}%")
            with c2:
                st.write("🟡 **MEDIA** (8-15%)")
                for _, r in media.iterrows():
                    st.write(f"**N.{int(r['numero']):2d}** → "
                             f"{r['pct']:.1f}%")
            with c3:
                st.write("🟢 **Pool Wyckoff attivo**")
                if not df_wyk_t.empty:
                    pool_df = carica_pool(
                        int(df_wyk_t.iloc[0]['id']))
                    if not pool_df.empty:
                        nums = sorted(pool_df['numero'].tolist())
                        for i in range(0, len(nums), 5):
                            st.write(" ".join(
                                f"**{n}**" for n in nums[i:i+5]))
                if vincolo_n_pari is not None:
                    st.divider()
                    st.write("🎲 **Vincolo parità**")
                    _log = df_wyk_t.iloc[0].get(
                        'vincolo_logica', '') \
                        if not df_wyk_t.empty else ''
                    if _log and str(_log) != 'nan':
                        st.write(f"**{_log}**")
                    else:
                        st.write(
                            f"**{vincolo_n_pari}p/"
                            f"{6-vincolo_n_pari}d**"
                        )
                    st.write(f"({vincolo_pct:.1f}% nella fascia)")

            st.divider()
            lbl = ("con vincolo parità"
                   if vincolo_n_pari is not None
                   and st.session_state.get("chk_parita", True)
                   else "totali")
            st.subheader(
                f"Prime 50 sestine candidate {lbl} "
                f"({len(df_cand_use):,})"
            )
            if not df_cand_use.empty:
                df_show = df_cand_use.head(50)[cols_n].copy()
                df_show.columns = ['N1','N2','N3','N4','N5','N6']
                df_show['Somma'] = df_show.sum(axis=1)
                df_show['Range'] = df_show['N6'] - df_show['N1']
                df_show.insert(0, '#', range(1, len(df_show)+1))
                st.dataframe(df_show, hide_index=True,
                             use_container_width=True)
                st.download_button(
                    "⬇️ Scarica candidate filtrate (CSV)",
                    df_cand_use[cols_n].to_csv(index=False),
                    f"candidate_parita_{run_sel}.csv",
                    "text/csv",
                    key="dl_cand_parita"
                )

            # ── SEZIONE B: ANALISI PROSSIMITÀ ───────────────
            st.divider()
            st.subheader("🔍 Analisi Prossimità")
            st.caption(
                "Analisi sulle candidate "
                + ("con vincolo parità attivo."
                   if vincolo_n_pari is not None
                   and st.session_state.get("chk_parita", True)
                   else "totali.")
            )

            with st.form("form_prossimita"):
                inp_prox = st.text_input(
                    "I tuoi 6 numeri (es: 7 22 35 48 63 80):",
                    placeholder="7 22 35 48 63 80",
                    key="inp_prox"
                )
                submitted_prox = st.form_submit_button(
                    "🔍 Analizza Prossimità", type="primary"
                )

            if submitted_prox:
                nums_prox = sorted(set(
                    int(n) for n in re.findall(r'\d+', inp_prox)
                    if 1 <= int(n) <= 90
                ))
                if len(nums_prox) != 6:
                    st.error("Inserisci esattamente 6 numeri.")
                elif df_cand_use.empty:
                    st.error("Nessuna candidata caricata.")
                else:
                    set_prox  = set(nums_prox)
                    risultati = []
                    for _, row in df_cand_use.iterrows():
                        s    = tuple(sorted([
                            row['n1'], row['n2'], row['n3'],
                            row['n4'], row['n5'], row['n6']
                        ]))
                        ovlp = len(set_prox & set(s))
                        if ovlp >= 3:
                            risultati.append({
                                'overlap': ovlp,
                                'N1': s[0], 'N2': s[1],
                                'N3': s[2], 'N4': s[3],
                                'N5': s[4], 'N6': s[5],
                                'Somma': sum(s),
                            })

                    n3 = sum(1 for r in risultati if r['overlap']==3)
                    n4 = sum(1 for r in risultati if r['overlap']==4)
                    n5 = sum(1 for r in risultati if r['overlap']==5)
                    n6 = sum(1 for r in risultati if r['overlap']==6)

                    st.info(
                        f"La tua sestina: **{nums_prox}** | "
                        f"Somma: **{sum(nums_prox)}** | "
                        f"Analizzate: **{len(df_cand_use):,}**"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("5 in comune", n5,
                              delta="vicinissime"
                              if n5 > 0 else None)
                    c2.metric("4 in comune", n4)
                    c3.metric("3 in comune", n3)
                    c4.metric("Esatta (6/6)", n6,
                              delta="🎯 presente!"
                              if n6 > 0 else None)

                    if risultati:
                        df_prox = pd.DataFrame(risultati)\
                            .sort_values('overlap', ascending=False)

                        def evidenzia(row):
                            celle = []
                            for col in ['N1','N2','N3',
                                        'N4','N5','N6']:
                                n = row[col]
                                celle.append(
                                    f"**{n}**"
                                    if n in set_prox
                                    else str(n)
                                )
                            return ' - '.join(celle)

                        df_prox['Numeri'] = df_prox.apply(
                            evidenzia, axis=1)
                        df_prox.insert(0, '#',
                                       range(1, len(df_prox)+1))
                        st.subheader(
                            f"{len(df_prox)} candidate "
                            f"con ≥3 in comune"
                        )
                        st.dataframe(
                            df_prox[['#','overlap',
                                     'Numeri','Somma']]\
                                .rename(columns={
                                    'overlap': 'In comune'}),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "Nessuna candidata condivide "
                            "3+ numeri con la tua sestina."
                        )

            # ── SEZIONE C: VERIFICA ESTRAZIONE ──────────────
            st.divider()
            st.subheader("🏆 Verifica Estrazione")
            st.caption(
                "Analisi sulle candidate "
                + ("con vincolo parità attivo."
                   if vincolo_n_pari is not None
                   and st.session_state.get("chk_parita", True)
                   else "totali.")
            )

            with st.form("form_verifica"):
                inp_verif = st.text_input(
                    "Numeri usciti (es: 14 25 38 52 67 81):",
                    placeholder="14 25 38 52 67 81",
                    key="inp_verif"
                )
                submitted_verif = st.form_submit_button(
                    "🏆 Verifica Risultato", type="primary"
                )

            if submitted_verif:
                nums_verif = sorted(set(
                    int(n) for n in re.findall(r'\d+', inp_verif)
                    if 1 <= int(n) <= 90
                ))
                if len(nums_verif) != 6:
                    st.error("Inserisci esattamente 6 numeri.")
                elif df_cand_use.empty:
                    st.error("Nessuna candidata caricata.")
                else:
                    set_verif = set(nums_verif)
                    vincenti  = {3: [], 4: [], 5: [], 6: []}

                    for _, row in df_cand_use.iterrows():
                        s    = tuple(sorted([
                            row['n1'], row['n2'], row['n3'],
                            row['n4'], row['n5'], row['n6']
                        ]))
                        ovlp = len(set_verif & set(s))
                        if ovlp >= 3:
                            vincenti[ovlp].append(s)

                    st.info(
                        f"Estrazione: **{nums_verif}** | "
                        f"Somma: **{sum(nums_verif)}** | "
                        f"Analizzate: **{len(df_cand_use):,}**"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c4.metric("🥇 Punti 6", len(vincenti[6]),
                              delta="JACKPOT! 🎉"
                              if vincenti[6] else None)
                    c3.metric("🥈 Punti 5", len(vincenti[5]))
                    c2.metric("🥉 Punti 4", len(vincenti[4]))
                    c1.metric("✅ Punti 3", len(vincenti[3]))

                    for punti in [6, 5, 4, 3]:
                        if vincenti[punti]:
                            emoji = {6:"🥇",5:"🥈",
                                     4:"🥉",3:"✅"}
                            st.subheader(
                                f"{emoji[punti]} Punti {punti}"
                                f" — {len(vincenti[punti])} "
                                f"schedine"
                            )
                            righe = []
                            for i, s in enumerate(
                                vincenti[punti]
                            ):
                                nums_fmt = [
                                    f"**{n}**"
                                    if n in set_verif
                                    else str(n)
                                    for n in s
                                ]
                                righe.append({
                                    '#':      i+1,
                                    'Numeri': ' - '.join(
                                        nums_fmt),
                                    'Somma':  sum(s),
                                    'Punti':  punti,
                                })
                            st.dataframe(
                                pd.DataFrame(righe),
                                hide_index=True,
                                use_container_width=True
                            )

                    if not any(vincenti.values()):
                        st.warning(
                            "Nessuna candidata ha fatto "
                            "3+ punti con questa estrazione."
                        )

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

    # Mostra totale sestine run attivo
    if run_ids_off:
        run_last = run_ids_off[0]
        df_cand_off_tot = carica_candidate(run_last)
        n_tot_off = len(df_cand_off_tot)
        run_label_last = datetime.datetime.fromtimestamp(
            run_last).strftime("%d/%m/%Y %H:%M")
        st.metric(
            label="📊 Sestine candidate nel run attivo",
            value=f"{n_tot_off:,}",
            delta=f"Run del {run_label_last}",
            delta_color="off"
        )

    # ── SEZIONE A: AUTOMATICA ────────────────────────────
    st.markdown("### 🤖 Sistema Ridotto Automatico")
    st.caption(
        "Prende le candidate già in target Wyckoff. "
        "Trova il minimo sottoinsieme tale che ogni "
        "sestina NON giocata condivida almeno 4 (o 5) "
        "numeri con almeno una sestina giocata."
    )

    if df_wyk_off.empty or not run_ids_off:
        st.warning("Dati non disponibili. Esegui analisi.py.")
    else:
        st.info(
            f"🎯 Target Wyckoff: **{fmin}-{fmax}** | "
            f"Trend: **{w_off['trend'].upper()}** | "
            f"Zona: **{w_off['zona_tipo']}**"
        )

        # Mostra vincolo parità se disponibile
        _np_off  = w_off.get('vincolo_n_pari')
        _pp_off  = w_off.get('vincolo_pct_pari')
        _log_off = w_off.get('vincolo_logica', '')
        if _np_off is not None and str(_np_off) != 'nan':
            try:
                _np_off = int(float(_np_off))
                _pp_off = float(_pp_off)
                if _log_off and str(_log_off) != 'nan':
                    descr_off = f"**{_log_off}**"
                else:
                    descr_off = f"**{_np_off}p/{6-_np_off}d**"
                st.success(
                    f"🎲 Vincolo parità: {descr_off} | "
                    f"Freq. storica: **{_pp_off:.1f}%** | "
                    f"applicato automaticamente alle candidate"
                )
            except Exception:
                pass

        with st.form("form_auto"):
            run_labels_off = {
                r: datetime.datetime.fromtimestamp(r)\
                   .strftime("%d/%m/%Y %H:%M")
                for r in run_ids_off
            }
            run_auto = st.selectbox(
                "Run candidate:",
                options=run_ids_off,
                format_func=lambda x: run_labels_off[x]
            )
            n_cand_auto = st.number_input(
                "Numero candidate da usare (in target):",
                min_value=10,
                max_value=5000,
                value=200,
                step=10,
                help="Quante sestine in target usare. "
                     "Consigliato: 100-500. "
                     "Più alto = ridotto più rappresentativo "
                     "ma calcolo più lento (O(n²))."
            )
            garanzia_auto = st.radio(
                "Garanzia:", options=[4, 5],
                index=1, horizontal=True
            )
            submitted_auto = st.form_submit_button(
                "🎯 Genera Sistema Automatico",
                type="primary"
            )

        if submitted_auto:
            df_cand_auto = carica_candidate(run_auto)

            if df_cand_auto.empty:
                st.error("Nessuna candidata nel run.")
            else:
                cols_n = ['n1','n2','n3','n4','n5','n6']
                df_cand_auto['somma'] = \
                    df_cand_auto[cols_n].sum(axis=1)

                df_in_target = df_cand_auto[
                    (df_cand_auto['somma'] >= fmin) &
                    (df_cand_auto['somma'] <= fmax)
                ]

                # Applica filtro parità se attivo
                _wyk_off2 = carica_wyckoff_stato()
                _vnp = None
                if not _wyk_off2.empty:
                    _np2 = _wyk_off2.iloc[0].get('vincolo_n_pari')
                    if _np2 is not None and str(_np2) != 'nan':
                        try:
                            _vnp = int(float(_np2))
                        except Exception:
                            pass

                if _vnp is not None:
                    mask_off = df_in_target[cols_n].apply(
                        lambda r: sum(
                            1 for v in r if v % 2 == 0
                        ) == _vnp, axis=1
                    )
                    df_in_target_par = df_in_target[mask_off]
                    st.write(
                        f"Candidate nel target {fmin}-{fmax}: "
                        f"**{len(df_in_target):,}** totali | "
                        f"**{len(df_in_target_par):,}** "
                        f"con parità {_vnp}p/{6-_vnp}d"
                    )
                    df_in_target = df_in_target_par
                else:
                    n_in = len(df_in_target)
                    st.write(
                        f"Candidate nel target {fmin}-{fmax}: "
                        f"**{n_in:,}** su "
                        f"**{len(df_cand_auto):,}**"
                    )

                if df_in_target.empty:
                    st.error(
                        "Nessuna candidata nel target. "
                        "Rilancia analisi.py."
                    )
                else:
                    candidate_list = [
                        tuple(sorted([
                            row['n1'], row['n2'], row['n3'],
                            row['n4'], row['n5'], row['n6']
                        ]))
                        for _, row in df_in_target.iterrows()
                    ]
                    n_use = min(int(n_cand_auto),
                                len(candidate_list))
                    candidate_list = candidate_list[:n_use]

                    st.info(
                        f"Usando **{n_use}** candidate in target | "
                        f"Garanzia **{garanzia_auto}** | "
                        f"Ogni sestina non giocata condivide "
                        f"≥{garanzia_auto} numeri con una giocata"
                    )

                    with st.spinner(
                        f"Covering design su {n_use} "
                        f"sestine... (O(n²) = "
                        f"{n_use*n_use:,} confronti)"
                    ):
                        sistema, efficienza, pool_out = \
                            genera_ridotto_da_candidate(
                                candidate_list, garanzia_auto
                            )

                    st.success(
                        f"Sistema ridotto: "
                        f"**{len(sistema)}** sestine | "
                        f"Partenza: **{n_use}** | "
                        f"Riduzione: **{efficienza}%** | "
                        f"Garanzia: **{garanzia_auto}**"
                    )

                    if sistema:
                        st.subheader(
                            f"{len(sistema)} sestine "
                            f"nel target {fmin}-{fmax}"
                        )
                        mostra_sistema(
                            sistema, garanzia_auto, "auto")

                        tutti_a  = [x for s in sistema for x in s]
                        freq_a   = Counter(tutti_a)
                        freq_df_a = pd.DataFrame([
                            {'numero': k, 'presenze': v,
                             'pct': round(v*100/len(sistema), 1)}
                            for k, v in sorted(freq_a.items())
                        ])
                        fig_a = px.bar(
                            freq_df_a, x='numero', y='pct',
                            color='pct',
                            color_continuous_scale='RdYlGn',
                            title="Copertura numeri")
                        fig_a.update_layout(
                            template="plotly_dark", height=250,
                            margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig_a,
                                        use_container_width=True)

    st.divider()

    # ── SEZIONE B: MANUALE ───────────────────────────────
    st.markdown("### ✏️ Selezione Manuale")
    st.caption(
        "Inserisci i tuoi numeri preferiti. "
        "Il sistema filtra le candidate già validate "
        "(Wyckoff + parità) che usano SOLO quei numeri, "
        "poi genera il ridotto su quelle."
    )

    with st.form("form_man"):
        numeri_input = st.text_input(
            "I tuoi numeri (almeno 6, separati da spazio):",
            placeholder="Es: 1 17 31 41 67 70 74 80 83 89"
        )
        c1, c2 = st.columns(2)
        with c1:
            garanzia_man = st.radio(
                "Garanzia:", options=[4, 5],
                index=1, horizontal=True
            )
        with c2:
            run_man = st.selectbox(
                "Run candidate:",
                options=run_ids_off if run_ids_off else [0],
                format_func=lambda x: datetime.datetime
                    .fromtimestamp(x).strftime("%d/%m/%Y %H:%M")
                    if x > 0 else "—"
            )
        submitted_man = st.form_submit_button(
            "🚀 Genera Sistema Manuale",
            type="primary"
        )

    if submitted_man:
        nums_raw = re.findall(r'\d+', numeri_input)
        numeri   = sorted(set(
            int(n) for n in nums_raw if 1 <= int(n) <= 90
        ))

        if len(numeri) < 6:
            st.error("Inserisci almeno 6 numeri (1-90).")
        elif run_man == 0:
            st.error("Nessun run disponibile.")
        else:
            set_numeri = set(numeri)
            df_cand_man = carica_candidate(run_man)

            if df_cand_man.empty:
                st.error("Nessuna candidata nel run.")
            else:
                cols_n = ['n1','n2','n3','n4','n5','n6']

                # Filtra candidate dove TUTTI i 6 numeri
                # sono nel set inserito dall'utente
                mask_man = df_cand_man[cols_n].apply(
                    lambda r: all(v in set_numeri for v in r),
                    axis=1
                )
                df_filtrate = df_cand_man[mask_man]

                st.info(
                    f"I tuoi **{len(numeri)} numeri**: "
                    f"{numeri} | "
                    f"Candidate compatibili: "
                    f"**{len(df_filtrate):,}** su "
                    f"**{len(df_cand_man):,}** totali"
                )

                if df_filtrate.empty:
                    st.warning(
                        "Nessuna candidata usa solo questi numeri. "
                        "Prova ad aggiungerne altri dal pool attivo."
                    )
                else:
                    candidate_list = [
                        tuple(sorted([
                            row['n1'], row['n2'], row['n3'],
                            row['n4'], row['n5'], row['n6']
                        ]))
                        for _, row in df_filtrate.iterrows()
                    ]

                    with st.spinner(
                        f"Covering design su "
                        f"{len(candidate_list)} candidate..."
                    ):
                        sistema_m, efficienza_m, pool_m = \
                            genera_ridotto_da_candidate(
                                candidate_list, garanzia_man
                            )

                    st.success(
                        f"Sistema ridotto: "
                        f"**{len(sistema_m)}** sestine | "
                        f"Partenza: **{len(candidate_list)}** | "
                        f"Riduzione: **{efficienza_m}%** | "
                        f"Garanzia: **{garanzia_man}**"
                    )

                    if sistema_m:
                        st.subheader(
                            f"{len(sistema_m)} sestine "
                            f"nel target {fmin}-{fmax}"
                        )
                        mostra_sistema(
                            sistema_m, garanzia_man, "man")

                        tutti_m   = [x for s in sistema_m
                                     for x in s]
                        freq_m    = Counter(tutti_m)
                        freq_df_m = pd.DataFrame([
                            {'numero': k, 'presenze': v,
                             'pct': round(
                                 v*100/len(sistema_m), 1)}
                            for k, v in sorted(freq_m.items())
                        ])
                        fig_man = px.bar(
                            freq_df_m, x='numero', y='pct',
                            color='pct',
                            color_continuous_scale='RdYlGn',
                            title="Copertura numeri")
                        fig_man.update_layout(
                            template="plotly_dark", height=250,
                            margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(
                            fig_man, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 6 — ULTIME ESTRAZIONI
# ════════════════════════════════════════════════════════
with tab6:
    st.subheader("Ultime Estrazioni")
    n_show = st.slider("Quante estrazioni:", 5, 100, 20)
    res    = supabase.table("estrazioni").select("*")\
        .order("data_estrazione", desc=True)\
        .limit(n_show).execute()
    df_ult = pd.DataFrame(res.data)

    if not df_ult.empty:
        cols_n = ['n1','n2','n3','n4','n5','n6']
        df_ult['somma'] = df_ult[cols_n].sum(axis=1)
        df_ult['range'] = df_ult['n6'] - df_ult['n1']
        st.dataframe(
            df_ult[['data_estrazione'] + cols_n +
                   ['jolly','superstar','somma','range']]\
                .rename(columns={
                    'data_estrazione': 'Data',
                    'n1':'N1','n2':'N2','n3':'N3',
                    'n4':'N4','n5':'N5','n6':'N6',
                    'jolly':'Jolly','superstar':'Superstar'}),
            hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_ult.iloc[::-1],
                         x='data_estrazione', y='somma',
                         title="Somma ultime estrazioni",
                         color='somma',
                         color_continuous_scale='RdYlGn')
            fig.add_hline(y=275.8, line_dash="dash",
                          line_color="white",
                          annotation_text="Media storica")
            fig.update_layout(template="plotly_dark", height=300,
                              margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(df_ult.iloc[::-1],
                          x='data_estrazione', y='range',
                          title="Range ultime estrazioni",
                          color='range',
                          color_continuous_scale='Blues')
            fig2.add_hline(y=65.3, line_dash="dash",
                           line_color="white",
                           annotation_text="Media storica")
            fig2.update_layout(template="plotly_dark", height=300,
                               margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 7 — PIANO B
# ════════════════════════════════════════════════════════
with tab7:
    st.subheader("🅱️ Piano B — Sistema Indipendente da Wyckoff")
    st.caption(
        "Approccio alternativo basato su 3 condizioni statistiche: "
        "**Ritardo collettivo** × **Somma media ultime 10** × "
        "**Parità ultima estrazione**. "
        "Indipendente dal sistema Wyckoff — confronto e verifica."
    )

    df_est_b  = carica_estrazioni()
    df_mappa_b = carica_mappa()

    if df_est_b.empty or df_mappa_b.empty:
        st.warning("Dati non disponibili.")
    else:
        cols_n = ['n1','n2','n3','n4','n5','n6']
        df_est_b['somma'] = df_est_b[cols_n].sum(axis=1)
        df_est_b = df_est_b.sort_values(
            'data_estrazione').reset_index(drop=True)

        # ── CONDIZIONE 1: Top 20 ritardatari da tutti 90 ─────
        st.markdown("### Condizione 1 — Top 20 Ritardatari")
        df_rit_b = df_mappa_b.sort_values(
            'ritardo_attuale', ascending=False).head(20)
        top20_b  = sorted(df_rit_b['numero'].tolist())

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_rit = go.Figure(go.Bar(
                x=df_rit_b['numero'],
                y=df_rit_b['ritardo_attuale'],
                marker_color='#636EFA'
            ))
            fig_rit.update_layout(
                template="plotly_dark", height=250,
                margin=dict(l=20,r=20,t=20,b=20),
                xaxis_title="Numero",
                yaxis_title="Ritardo (estrazioni)"
            )
            st.plotly_chart(fig_rit, use_container_width=True)
        with c2:
            st.write("**Top 20 più attesi (tutti 90):**")
            for i in range(0, len(top20_b), 5):
                st.write(" ".join(
                    f"**{n}**" for n in top20_b[i:i+5]))

        # ── CONDIZIONE 2: Somma media ultime 10 ──────────────
        st.markdown("### Condizione 2 — Somma Media Ultime 10")
        ultime10 = df_est_b.tail(10)
        media_s10 = float(ultime10['somma'].mean())
        margine_b = st.slider(
            "Margine ±", min_value=10, max_value=60,
            value=30, step=5, key="margine_b"
        )
        st.caption(
            f"**Margine ±{margine_b}**: quanto può distaccarsi "
            f"la prossima somma dalla media delle ultime 10 "
            f"({media_s10:.1f}). "
            f"Stretto (±10) = poche candidate concentrate. "
            f"Ampio (±60) = più candidate, meno filtrate."
        )
        fmin_b = int(media_s10 - margine_b)
        fmax_b = int(media_s10 + margine_b)

        st.info(
            f"Media somme ultime 10: **{media_s10:.1f}** | "
            f"Fascia target: **{fmin_b}-{fmax_b}**"
        )

        # ── CONDIZIONE 3: Parità ultima estrazione ───────────
        st.markdown("### Condizione 3 — Parità Ultima Estrazione")
        ultima_b = df_est_b.iloc[-1]
        n_pari_ultima = sum(
            1 for c in cols_n if ultima_b[c] % 2 == 0
        )
        n_disp_ultima = 6 - n_pari_ultima

        st.info(
            f"Ultima estrazione: "
            f"**{int(ultima_b['n1'])},{int(ultima_b['n2'])},"
            f"{int(ultima_b['n3'])},{int(ultima_b['n4'])},"
            f"{int(ultima_b['n5'])},{int(ultima_b['n6'])}** | "
            f"Somma: **{int(ultima_b['somma'])}** | "
            f"Parità: **{n_pari_ultima}p/{n_disp_ultima}d**"
        )

        # Slider per scegliere parità
        parita_b = st.radio(
            "Parità da applicare:",
            options=[
                f"{n_pari_ultima}p/{n_disp_ultima}d (come ultima)",
                f"{n_pari_ultima-1}p/{n_disp_ultima+1}d",
                f"{n_pari_ultima+1}p/{n_disp_ultima-1}d",
            ],
            horizontal=True,
            key="parita_b"
        )
        # Estrai n_pari dalla scelta
        n_pari_b = int(parita_b.split("p/")[0])

        # ── GENERAZIONE CANDIDATE PIANO B ────────────────────
        st.markdown("### Risultato — Candidate Piano B")
        st.caption(
            f"C(20,6) = 38.760 combinazioni | "
            f"Filtro somma [{fmin_b}-{fmax_b}] | "
            f"Filtro parità {n_pari_b}p/{6-n_pari_b}d"
        )

        if st.button("🔄 Calcola Candidate Piano B",
                     type="primary", key="btn_pianob"):

            candidate_b = []
            for sestina in combinations(top20_b, 6):
                somma = sum(sestina)
                if not (fmin_b <= somma <= fmax_b):
                    continue
                n_pari = sum(
                    1 for n in sestina if n % 2 == 0
                )
                if n_pari != n_pari_b:
                    continue
                candidate_b.append(list(sestina))

            # Salva in session_state per il form del ridotto
            st.session_state['candidate_b']  = candidate_b
            st.session_state['fmin_b']       = fmin_b
            st.session_state['fmax_b']       = fmax_b
            st.session_state['n_pari_b']     = n_pari_b

        # Mostra risultati se disponibili in session_state
        if 'candidate_b' in st.session_state:
            candidate_b = st.session_state['candidate_b']
            fmin_b_s    = st.session_state.get('fmin_b', fmin_b)
            fmax_b_s    = st.session_state.get('fmax_b', fmax_b)
            n_pari_b_s  = st.session_state.get('n_pari_b', n_pari_b)

            if not candidate_b:
                st.warning(
                    "Nessuna candidata con questi filtri. "
                    "Prova ad aumentare il margine somma."
                )
            else:
                # Numeri unici nelle candidate
                tutti_b = [n for s in candidate_b for n in s]
                freq_b  = Counter(tutti_b)
                numeri_unici_b = sorted(freq_b.keys())

                st.success(
                    f"**{len(candidate_b):,}** candidate | "
                    f"**{len(numeri_unici_b)}** numeri unici | "
                    f"Somma [{fmin_b}-{fmax_b}] | "
                    f"{n_pari_b}p/{6-n_pari_b}d"
                )

                # Frequenze numeri
                freq_df_b = pd.DataFrame([
                    {'numero': k,
                     'presenze': v,
                     'pct': round(v*100/len(candidate_b), 1)}
                    for k, v in sorted(freq_b.items())
                ])
                fig_b = px.bar(
                    freq_df_b, x='numero', y='pct',
                    color='pct',
                    color_continuous_scale='RdYlGn',
                    title="Frequenza numeri nelle candidate Piano B"
                )
                fig_b.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=20,r=20,t=40,b=20),
                    xaxis=dict(dtick=5)
                )
                st.plotly_chart(fig_b, use_container_width=True)

                # Numeri unici
                st.write("**Numeri unici nelle candidate:**")
                for i in range(0, len(numeri_unici_b), 10):
                    st.write(" ".join(
                        f"**{n}**"
                        for n in numeri_unici_b[i:i+10]
                    ))

                st.divider()

                # Prime 50 candidate
                st.subheader(
                    f"Prime 50 candidate Piano B "
                    f"({len(candidate_b):,} totali)"
                )
                righe_b = []
                for i, s in enumerate(candidate_b[:50]):
                    righe_b.append({
                        '#': i+1,
                        'N1': s[0], 'N2': s[1], 'N3': s[2],
                        'N4': s[3], 'N5': s[4], 'N6': s[5],
                        'Somma': sum(s),
                    })
                st.dataframe(
                    pd.DataFrame(righe_b),
                    hide_index=True,
                    use_container_width=True
                )
                st.download_button(
                    "⬇️ Scarica candidate Piano B (CSV)",
                    pd.DataFrame([
                        {'N1':s[0],'N2':s[1],'N3':s[2],
                         'N4':s[3],'N5':s[4],'N6':s[5],
                         'Somma':sum(s)}
                        for s in candidate_b
                    ]).to_csv(index=False),
                    "candidate_pianob.csv",
                    "text/csv",
                    key="dl_pianob"
                )

                st.divider()

                # ── SISTEMA RIDOTTO PIANO B ───────────────
                st.markdown("### 🔧 Sistema Ridotto Piano B")
                with st.form("form_ridotto_b"):
                    garanzia_b = st.radio(
                        "Garanzia:", options=[4, 5],
                        index=1, horizontal=True,
                        key="gar_b"
                    )
                    n_cand_b = st.number_input(
                        "Candidate da usare per il ridotto:",
                        min_value=10,
                        max_value=len(candidate_b),
                        value=min(300, len(candidate_b)),
                        step=10,
                        key="n_cand_b"
                    )
                    submitted_ridotto_b = st.form_submit_button(
                        "🎯 Genera Sistema Ridotto Piano B",
                        type="primary"
                    )

                if submitted_ridotto_b:
                    cand_list_b = [
                        tuple(s) for s in candidate_b[:int(n_cand_b)]
                    ]
                    with st.spinner(
                        f"Covering design su "
                        f"{len(cand_list_b)} candidate..."
                    ):
                        sistema_b, efficienza_b, pool_b = \
                            genera_ridotto_da_candidate(
                                cand_list_b, garanzia_b
                            )

                    st.success(
                        f"Sistema ridotto: "
                        f"**{len(sistema_b)}** sestine | "
                        f"Partenza: **{len(cand_list_b)}** | "
                        f"Riduzione: **{efficienza_b}%** | "
                        f"Garanzia: **{garanzia_b}**"
                    )

                    if sistema_b:
                        righe_rid_b = []
                        for i, s in enumerate(sistema_b):
                            righe_rid_b.append({
                                '#': i+1,
                                'N1':s[0],'N2':s[1],'N3':s[2],
                                'N4':s[3],'N5':s[4],'N6':s[5],
                                'Somma': sum(s),
                                'Range': s[-1]-s[0],
                            })
                        df_rid_b = pd.DataFrame(righe_rid_b)
                        st.dataframe(
                            df_rid_b,
                            hide_index=True,
                            use_container_width=True
                        )
                        st.download_button(
                            "⬇️ Scarica sistema ridotto (CSV)",
                            df_rid_b.to_csv(index=False),
                            f"ridotto_pianob_g{garanzia_b}.csv",
                            "text/csv",
                            key="dl_rid_b"
                        )

                        # Grafico copertura
                        tutti_rb  = [x for s in sistema_b
                                     for x in s]
                        freq_rb   = Counter(tutti_rb)
                        freq_df_rb = pd.DataFrame([
                            {'numero': k, 'presenze': v,
                             'pct': round(
                                 v*100/len(sistema_b), 1)}
                            for k, v in sorted(freq_rb.items())
                        ])
                        fig_rb = px.bar(
                            freq_df_rb,
                            x='numero', y='pct',
                            color='pct',
                            color_continuous_scale='RdYlGn',
                            title="Copertura numeri nel ridotto"
                        )
                        fig_rb.update_layout(
                            template="plotly_dark", height=250,
                            margin=dict(l=20,r=20,t=40,b=20)
                        )
                        st.plotly_chart(
                            fig_rb, use_container_width=True
                        )
