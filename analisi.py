# ── Carica estrazioni (tutte, con paginazione) ───────────
print("Caricamento estrazioni...")

tutti_i_dati = []
PAGE = 1000
offset = 0

while True:
    res = supabase.table("estrazioni")\
        .select("id, data_estrazione, n1, n2, n3, n4, n5, n6")\
        .order("data_estrazione", desc=False)\
        .range(offset, offset + PAGE - 1)\
        .execute()
    
    batch = res.data
    if not batch:
        break
    
    tutti_i_dati.extend(batch)
    offset += PAGE
    print(f"  Caricati {len(tutti_i_dati)} finora...")
    
    if len(batch) < PAGE:
        break

df = pd.DataFrame(tutti_i_dati)
df['data_estrazione'] = pd.to_datetime(df['data_estrazione'])
df = df.sort_values('data_estrazione').reset_index(drop=True)
print(f"Caricate {len(df)} estrazioni totali.")
