import streamlit as st
from engine.pipeline import pipeline_v2

st.title("SSAS - Core Engine Evoluto")

n = st.slider("Simulazioni", 50000, 300000, 100000)

# placeholder (poi collegheremo a Supabase)
target_h = st.number_input("Target H", value=0.75)
target_delta = st.number_input("Target ΔH", value=0.0)
h_last = st.number_input("H ultimo", value=0.75)

if st.button("Esegui Analisi Evoluta"):
    res = pipeline_v2(n, target_h, target_delta, h_last)

    st.subheader("Stabilità sistema")
    st.metric("Stability Index", f"{res['stability']:.6f}")

    st.subheader("Top numeri")
    top = sorted(res["freq"].items(), key=lambda x: x[1], reverse=True)[:20]
    st.write(top)
