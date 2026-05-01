import streamlit as st
from engine.pipeline import pipeline

st.title("SSAS - Structural Analysis System")

n = st.slider("Numero simulazioni", 10000, 200000, 50000)

if st.button("Esegui Analisi"):
    result = pipeline(n)

    st.subheader("Top numeri")
    top = sorted(result["freq"].items(), key=lambda x: x[1], reverse=True)[:20]
    st.write(top)

    st.subheader("Top coppie")
    top_pairs = sorted(result["pairs"].items(), key=lambda x: x[1], reverse=True)[:20]
    st.write(top_pairs)
