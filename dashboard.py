"""
SSAS - Dashboard Streamlit
Visualizza analisi strutturale e sestine candidate.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client

# ── Configurazione ────────────────────────────────────────
st.set_page_config(
    page_title="SSAS - Stochastic Structure Analysis",
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

@st.cache_data(ttl=3600)
def carica_ultime_estrazioni(n=20):
    res = supabase.table("estrazioni")\
        .select("*")\
        .order("data_estrazione", desc=True)\
        .limit(n)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=600)
def carica_candidate(run_id=None):
    q = supabase.table("combinazioni_candidate")\
        .select("*")\
        .order("score_armonia", desc=True)
    if run_id:
        q = q.eq("run_id", run_id)
    res = q.limit(10000).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def carica_fingerprint():
    res = supabase.table("fingerprint_estrazioni")\
        .select("*")\
        .order("data_estrazione", desc=False)\
        .limit(10000)\
        .execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def carica_run_ids():
    res = supabase.table("combinazioni_candidate")\
        .select("run_id")\
        .execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return []
    return sorted(df['run_id'].unique().tolist(), reverse=True)

# ── Header ────────────────────────────────────────────────
st.title("🎯 SSAS — Stochastic Structure Analysis System")
st.caption("Analisi strutturale delle estrazioni Superenalotto")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Costanti Sistema",
    "🗺️ Mappa Occupazione",
    "📈 Fingerprint Storico",
    "🎯 Sestine Candidate",
    "🔢 Ultime Estrazioni"
])

# ════════════════════════════════════════════════════════
# TAB 1 — COSTANTI SISTEMA
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Costanti Strutturali del Sistema")
    st.caption("Parametri calcolati su 7304 estrazioni storiche")

    df_cost = carica_costanti()

    if not df_cost.empty:
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)

        spacing = df_cost[df_cost['nome']=='spacing_ratio']
        somma   = df_cost[df_cost['nome']=='somma']
        cv      = df_cost[df_cost['nome']=='cv_gap']
        entropia= df_cost[df_cost['nome']=='entropia_gap']

        if not spacing.empty:
            sr = spacing.iloc[0]
            col1.metric(
                "Spacing Ratio (Wigner-Dyson)",
                f"{sr['valore_medio']:.4f}",
                delta=f"Poisson=0.386 GOE=0.536",
                delta_color="off"
            )
        if not somma.empty:
            s = somma.iloc[0]
            col2.metric(
                "Somma Media",
                f"{s['valore_medio']:.1f}",
                delta=f"±{s['std_dev']:.1f}",
                delta_color="off"
            )
        if not cv.empty:
            c = cv.iloc[0]
            col3.metric(
                "CV Gap (Disordine)",
                f"{c['valore_medio']:.4f}",
                delta=f"±{c['std_dev']:.4f}",
                delta_color="off"
            )
        if not entropia.empty:
            e = entropia.iloc[0]
            col4.metric(
                "Entropia Gap",
                f"{e['valore_medio']:.4f}",
                delta=f"±{e['std_dev']:.4f}",
                delta_color="off"
            )

        st.divider()

        # Tabella costanti con range
        st.subheader("Range Storici (p5 → p95)")
        rows = []
        for _, r in df_cost.iterrows():
            rows.append({
                "Parametro":    r['nome'],
                "Media":        round(r['valore_medio'], 4),
                "Std Dev":      round(r['std_dev'], 4),
                "Min (p5)":     round(r['percentile_5'], 4),
                "Max (p95)":    round(r['percentile_95'], 4),
                "Sigma vs Random": round(r['sigma_da_random'], 3)
                    if r['sigma_da_random'] else "—",
                "Campioni":     r['n_campioni']
            })

        df_display = pd.DataFrame(rows)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )

        # Grafico distribuzione spacing ratio
        st.subheader("Spacing Ratio nel contesto fisico")
        fig = go.Figure()
        fig.add_vline(x=0.386, line_dash="dash",
                      line_color="orange",
                      annotation_text="Poisson 0.386")
        fig.add_vline(x=0.536, line_dash="dash",
                      line_color="green",
                      annotation_text="GOE 0.536")

        if not spacing.empty:
            sr_val = spacing.iloc[0]['valore_medio']
            sr_std = spacing.iloc[0]['std_dev']
            fig.add_vline(x=sr_val, line_color="red",
                          annotation_text=f"Sistema {sr_val:.4f}")
            fig.add_vrect(
                x0=sr_val-sr_std, x1=sr_val+sr_std,
                fillcolor="red", opacity=0.1
            )

        fig.update_layout(
            template="plotly_dark",
            height=200,
            xaxis_title="Spacing Ratio",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — MAPPA OCCUPAZIONE
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Mappa Occupazione 1-90")
    st.caption("Z-score: deviazione dalla frequenza attesa teorica")

    df_mappa = carica_mappa()

    if not df_mappa.empty:
        col1, col2 = st.columns(2)

        # Top 10 più frequenti
        with col1:
            st.write("**Top 10 più frequenti**")
            top10 = df_mappa.nlargest(10, 'freq_assoluta')[
                ['numero','freq_assoluta','freq_relativa','z_score']
            ]
            st.dataframe(top10, hide_index=True,
                        use_container_width=True)

        # Top 10 più ritardatari
        with col2:
            st.write("**Top 10 più ritardatari**")
            rit10 = df_mappa.nlargest(10, 'ritardo_attuale')[
                ['numero','ritardo_attuale','ritardo_medio',
                 'ultimo_estratto']
            ]
            st.dataframe(rit10, hide_index=True,
                        use_container_width=True)

        st.divider()

        # Heatmap Z-score
        st.subheader("Heatmap Z-score (deviazione da atteso)")
        z_vals = df_mappa['z_score'].values
        numeri = df_mappa['numero'].values

        # Griglia 9×10
        grid_z = np.zeros((9, 10))
        grid_n = np.zeros((9, 10), dtype=int)
        for i, (n, z) in enumerate(zip(numeri, z_vals)):
            row = (n-1) // 10
            col = (n-1) % 10
            grid_z[row][col] = z
            grid_n[row][col] = n

        fig = go.Figure(data=go.Heatmap(
            z=grid_z,
            text=grid_n,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Z-score")
        ))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_showticklabels=False,
            yaxis_showticklabels=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rosso = sopra atteso | Blu = sotto atteso")

        # Densità locale
        st.subheader("Densità Locale per numero")
        fig2 = px.bar(
            df_mappa,
            x='numero',
            y='densita_locale',
            color='z_score',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0
        )
        fig2.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — FINGERPRINT STORICO
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Evoluzione Strutturale nel Tempo")

    df_fp = carica_fingerprint()

    if not df_fp.empty:
        df_fp['data_estrazione'] = pd.to_datetime(
            df_fp['data_estrazione'])

        col1, col2 = st.columns(2)

        with col1:
            # Spacing ratio nel tempo
            fig = px.line(
                df_fp.tail(500),
                x='data_estrazione',
                y='spacing_ratio_medio',
                title="Spacing Ratio (ultime 500 estrazioni)"
            )
            fig.add_hline(y=0.386, line_dash="dash",
                         line_color="orange",
                         annotation_text="Poisson")
            fig.add_hline(y=0.536, line_dash="dash",
                         line_color="green",
                         annotation_text="GOE")
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Entropia gap nel tempo
            fig2 = px.line(
                df_fp.tail(500),
                x='data_estrazione',
                y='entropia_gap',
                title="Entropia Gap (ultime 500 estrazioni)"
            )
            fig2.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            # Distribuzione somme
            fig3 = px.histogram(
                df_fp, x='somma',
                nbins=50,
                title="Distribuzione Somme",
                color_discrete_sequence=['#636EFA']
            )
            fig3.add_vline(x=df_fp['somma'].mean(),
                          line_color="red",
                          annotation_text="media")
            fig3.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            # Distribuzione CV gap
            fig4 = px.histogram(
                df_fp, x='cv_gap',
                nbins=50,
                title="Distribuzione CV Gap",
                color_discrete_sequence=['#EF553B']
            )
            fig4.add_vline(x=df_fp['cv_gap'].mean(),
                          line_color="red",
                          annotation_text="media")
            fig4.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Overlap lag1 nel tempo
        st.subheader("Overlap con estrazione precedente")
        fig5 = px.bar(
            df_fp.tail(100),
            x='data_estrazione',
            y='overlap_lag1',
            title="Numeri in comune con estrazione precedente "
                  "(ultime 100)"
        )
        fig5.update_layout(
            template="plotly_dark", height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — SESTINE CANDIDATE
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sestine Candidate")

    run_ids = carica_run_ids()

    if not run_ids:
        st.warning("Nessuna sestina candidata. "
                   "Esegui prima analisi.py.")
    else:
        # Selezione run
        import datetime
        run_labels = {
            r: datetime.datetime.fromtimestamp(r)\
                .strftime("%d/%m/%Y %H:%M:%S")
            for r in run_ids
        }
        run_sel = st.selectbox(
            "Seleziona run:",
            options=run_ids,
            format_func=lambda x: run_labels[x]
        )

        df_cand = carica_candidate(run_sel)

        if not df_cand.empty:
            st.info(f"**{len(df_cand):,}** sestine candidate "
                    f"per questo run.")

            # Calcola fingerprint sulle candidate
            def calc_somma(row):
                return sum([row.n1,row.n2,row.n3,
                           row.n4,row.n5,row.n6])

            cols_n = ['n1','n2','n3','n4','n5','n6']
            df_cand['somma'] = df_cand[cols_n].sum(axis=1)
            df_cand['range_tot'] = (df_cand['n6'] -
                                    df_cand['n1'])
            df_cand['n_pari'] = df_cand[cols_n]\
                .apply(lambda r: sum(x%2==0 for x in r),
                       axis=1)

            col1, col2 = st.columns(2)

            with col1:
                # Distribuzione somme candidate
                fig = px.histogram(
                    df_cand, x='somma',
                    nbins=40,
                    title="Distribuzione Somme Candidate"
                )
                fig.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig,
                               use_container_width=True)

            with col2:
                # Distribuzione range
                fig2 = px.histogram(
                    df_cand, x='range_tot',
                    nbins=40,
                    title="Distribuzione Range Candidate"
                )
                fig2.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2,
                               use_container_width=True)

            st.divider()

            # Frequenza numeri nelle candidate
            st.subheader("Frequenza numeri nelle candidate")
            tutti = []
            for col in cols_n:
                tutti.extend(df_cand[col].tolist())
            freq_series = pd.Series(tutti)\
                .value_counts().sort_index()
            freq_df = pd.DataFrame({
                'numero': freq_series.index,
                'freq':   freq_series.values
            })

            fig3 = px.bar(
                freq_df, x='numero', y='freq',
                title="Numeri più presenti nelle candidate",
                color='freq',
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.divider()

            # Top 20 candidate
            st.subheader("Top 20 Candidate per Score")
            top20 = df_cand.head(20)[
                ['n1','n2','n3','n4','n5','n6',
                 'somma','range_tot','n_pari',
                 'score_armonia']
            ].copy()
            top20.columns = [
                'N1','N2','N3','N4','N5','N6',
                'Somma','Range','Pari','Score'
            ]
            st.dataframe(
                top20,
                hide_index=True,
                use_container_width=True
            )

            # Download
            csv = df_cand[cols_n + ['somma',
                'range_tot','n_pari','score_armonia']]\
                .to_csv(index=False)
            st.download_button(
                "⬇️ Scarica tutte le candidate (CSV)",
                csv,
                f"candidate_{run_sel}.csv",
                "text/csv"
            )

# ════════════════════════════════════════════════════════
# TAB 5 — ULTIME ESTRAZIONI
# ════════════════════════════════════════════════════════
with tab5:
    st.subheader("Ultime Estrazioni")

    n_show = st.slider("Quante estrazioni mostrare:",
                       5, 100, 20)
    df_ult = carica_ultime_estrazioni(n_show)

    if not df_ult.empty:
        # Formatta visualizzazione
        display_cols = ['data_estrazione',
                       'n1','n2','n3','n4','n5','n6',
                       'jolly','superstar']
        df_show = df_ult[display_cols].copy()
        df_show.columns = ['Data','N1','N2','N3',
                          'N4','N5','N6','Jolly',
                          'Superstar']
        st.dataframe(
            df_show,
            hide_index=True,
            use_container_width=True
        )

        # Aggiungi somma e range
        df_ult['somma'] = df_ult[['n1','n2','n3',
                                  'n4','n5','n6']].sum(axis=1)
        df_ult['range'] = df_ult['n6'] - df_ult['n1']

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df_ult,
                x='data_estrazione',
                y='somma',
                title="Somma ultime estrazioni"
            )
            fig.add_hline(y=275.8, line_dash="dash",
                         line_color="red",
                         annotation_text="media storica")
            fig.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(
                df_ult,
                x='data_estrazione',
                y='range',
                title="Range ultime estrazioni"
            )
            fig2.add_hline(y=65.3, line_dash="dash",
                          line_color="red",
                          annotation_text="media storica")
            fig2.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)
