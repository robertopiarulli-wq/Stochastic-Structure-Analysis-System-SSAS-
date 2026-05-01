import streamlit as st
from supabase import create_client

def get_client():
    url = st.secrets["URL_SUPABASE"]
    key = st.secrets["KEY_SUPABASE"]
    return create_client(url, key)

def carica_estrazioni(client):
    res = client.table("estrazioni")\
        .select("id, data_estrazione, n1, n2, n3, n4, n5, n6")\
        .order("data_estrazione", desc=False)\
        .execute()
    return res.data
