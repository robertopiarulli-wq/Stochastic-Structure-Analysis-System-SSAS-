from engine.generator import genera_sestina
from engine.metrics import rugosita
from engine.filters import filtro

def run(n=500000, batch=50000, params={}):
    
    risultati = []
    
    for b in range(0, n, batch):
        for _ in range(batch):
            
            s = genera_sestina()
            h = rugosita(s)
            
            ok, score = filtro(
                s,
                h,
                params["target_h"],
                params["delta_target"],
                params["h_prev"],
                params["morsa"]
            )
            
            if ok:
                risultati.append((s, score, h))
                
    risultati.sort(key=lambda x: x[1])
    return risultati
