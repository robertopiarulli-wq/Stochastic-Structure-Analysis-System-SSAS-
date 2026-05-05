"""
SSAS - Modulo LSTM Somme
Predice la prossima somma usando una LSTM
addestrata sulla sequenza storica delle somme.
Complementa Wyckoff con visione dinamica.
"""
import numpy as np
import os

# ── Parametri ────────────────────────────────────────────
SEQ_LEN     = 20    # estrazioni in input
LSTM_UNITS  = 64    # unità per layer
N_LAYERS    = 2     # layer LSTM
DROPOUT     = 0.2
EPOCHS      = 100
BATCH_SIZE  = 32
PATIENCE    = 10    # early stopping
MODEL_PATH  = "moduli/lstm_model.weights.h5"

# ── Preparazione dati ────────────────────────────────────
def prepara_sequenze(somme: list, seq_len: int = SEQ_LEN):
    """
    Crea sequenze (X, y) da lista di somme.
    X[i] = somme[i:i+seq_len]
    y[i] = somme[i+seq_len]
    """
    somme_np = np.array(somme, dtype=np.float32)
    s_min    = somme_np.min()
    s_max    = somme_np.max()

    # Normalizza [0, 1]
    somme_norm = (somme_np - s_min) / (s_max - s_min + 1e-8)

    X, y = [], []
    for i in range(len(somme_norm) - seq_len):
        X.append(somme_norm[i:i+seq_len])
        y.append(somme_norm[i+seq_len])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    return X, y, s_min, s_max


def costruisci_modello(seq_len: int = SEQ_LEN):
    """Costruisce il modello LSTM."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, Input
        )
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError(
            "tensorflow non installato. "
            "Aggiungi 'tensorflow' a requirements.txt"
        )

    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(LSTM_UNITS, return_sequences=(N_LAYERS > 1)),
        Dropout(DROPOUT),
    ])

    for i in range(1, N_LAYERS):
        ret_seq = (i < N_LAYERS - 1)
        model.add(LSTM(LSTM_UNITS, return_sequences=ret_seq))
        model.add(Dropout(DROPOUT))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))   # regressione: somma normalizzata

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model


def addestra_lstm(somme: list, verbose: bool = True):
    """
    Addestra (o riaddestra) il modello LSTM
    sulla sequenza completa delle somme.
    Salva i pesi in MODEL_PATH.
    Restituisce (model, s_min, s_max, history).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError("tensorflow non installato.")

    tf.random.set_seed(42)
    np.random.seed(42)

    X, y, s_min, s_max = prepara_sequenze(somme, SEQ_LEN)

    # Split train/val/test 80/10/10
    n       = len(X)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    X_train, y_train = X[:n_train],        y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], \
                        y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],  y[n_train+n_val:]

    model = costruisci_modello(SEQ_LEN)

    cb = EarlyStopping(
        monitor='val_loss', patience=PATIENCE,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=1 if verbose else 0
    )

    # Valuta su test
    loss_t, mae_t = model.evaluate(X_test, y_test, verbose=0)
    mae_somme = mae_t * (s_max - s_min)  # denormalizzato

    if verbose:
        print(f"  [LSTM] Test MAE: {mae_somme:.1f} somme")
        print(f"  [LSTM] Epoche: {len(history.history['loss'])}")

    # Salva pesi
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_weights(MODEL_PATH)
    if verbose:
        print(f"  [LSTM] Pesi salvati: {MODEL_PATH}")

    return model, s_min, s_max, history


def predici_prossima_somma(somme: list,
                            model=None,
                            s_min: float = None,
                            s_max: float = None,
                            n_simulazioni: int = 100):
    """
    Predice la prossima somma con intervallo di confidenza.
    Usa Monte Carlo Dropout per stimare l'incertezza.

    Restituisce:
      pred_somma:  somma predetta
      conf_low:    limite inferiore 80%
      conf_high:   limite superiore 80%
      std_dev:     deviazione standard (incertezza)
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None

    somme_np   = np.array(somme, dtype=np.float32)
    s_min_loc  = somme_np.min() if s_min is None else s_min
    s_max_loc  = somme_np.max() if s_max is None else s_max
    somme_norm = (somme_np - s_min_loc) / \
                 (s_max_loc - s_min_loc + 1e-8)

    # Ultime SEQ_LEN somme come input
    seq = somme_norm[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)

    # Carica modello se non passato
    if model is None:
        model = costruisci_modello(SEQ_LEN)
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
        else:
            return None

    # Monte Carlo Dropout: n_simulazioni predizioni
    # con dropout attivo → stima incertezza
    @tf.function
    def predict_with_dropout(x):
        return model(x, training=True)

    preds_norm = [
        float(predict_with_dropout(seq)[0, 0])
        for _ in range(n_simulazioni)
    ]

    preds_somme = [
        p * (s_max_loc - s_min_loc) + s_min_loc
        for p in preds_norm
    ]

    pred_arr  = np.array(preds_somme)
    pred_mean = float(pred_arr.mean())
    pred_std  = float(pred_arr.std())
    conf_low  = float(np.percentile(pred_arr, 10))
    conf_high = float(np.percentile(pred_arr, 90))

    return {
        'somma_pred': round(pred_mean, 1),
        'conf_low':   round(conf_low, 1),
        'conf_high':  round(conf_high, 1),
        'std_dev':    round(pred_std, 1),
    }


def merge_fasce(fascia_wyckoff: tuple,
                pred_lstm: dict,
                peso_w: float = 0.6,
                peso_l: float = 0.4):
    """
    Combina la fascia Wyckoff con la previsione LSTM.

    fascia_wyckoff = (fmin, fmax)
    pred_lstm      = dict con somma_pred, conf_low, conf_high

    Restituisce:
      fascia_merged:   (fmin_m, fmax_m)
      concordanza:     'ALTA' / 'MEDIA' / 'BASSA'
      distanza_centri: punti di distanza tra i due centri
    """
    fmin_w, fmax_w  = fascia_wyckoff
    centro_w        = (fmin_w + fmax_w) / 2
    amp_w           = (fmax_w - fmin_w) / 2

    centro_l        = pred_lstm['somma_pred']
    amp_l           = (pred_lstm['conf_high'] -
                       pred_lstm['conf_low']) / 2

    # Centro merged pesato
    centro_m = centro_w * peso_w + centro_l * peso_l

    # Ampiezza: minimo tra le due, leggermente allargata
    amp_m    = min(amp_w, amp_l) * 1.2

    fmin_m = int(round(centro_m - amp_m))
    fmax_m = int(round(centro_m + amp_m))

    # Distanza tra centri
    dist = abs(centro_w - centro_l)

    if dist < 15:
        concordanza = 'ALTA'
    elif dist < 30:
        concordanza = 'MEDIA'
    else:
        concordanza = 'BASSA'

    return {
        'fascia_merged':    (fmin_m, fmax_m),
        'concordanza':      concordanza,
        'distanza_centri':  round(dist, 1),
        'centro_wyckoff':   round(centro_w, 1),
        'centro_lstm':      round(centro_l, 1),
        'amp_wyckoff':      round(amp_w, 1),
        'amp_lstm':         round(amp_l, 1),
    }


def esegui_lstm(df_somme, stato_wyckoff, verbose=True):
    """
    Entry point chiamato da analisi.py.
    
    1. Addestra/aggiorna LSTM sulle somme
    2. Predice prossima somma con incertezza
    3. Merge con fascia Wyckoff
    4. Restituisce stato aggiornato con fascia merged
    """
    somme = df_somme.tolist()

    if verbose:
        print(f"\n  [LSTM] Addestramento su {len(somme)} somme...")

    try:
        model, s_min, s_max, _ = addestra_lstm(
            somme, verbose=verbose
        )

        pred = predici_prossima_somma(
            somme, model, s_min, s_max, n_simulazioni=100
        )

        if pred is None:
            if verbose:
                print("  [LSTM] Predizione fallita, uso solo Wyckoff")
            return stato_wyckoff

        if verbose:
            print(f"  [LSTM] Somma predetta: "
                  f"{pred['somma_pred']:.1f} "
                  f"[{pred['conf_low']:.1f} - "
                  f"{pred['conf_high']:.1f}] "
                  f"(±{pred['std_dev']:.1f})")

        # Merge con Wyckoff
        fascia_w = (stato_wyckoff['fascia_min'],
                    stato_wyckoff['fascia_max'])
        merge    = merge_fasce(fascia_w, pred)

        if verbose:
            print(f"  [LSTM] Concordanza con Wyckoff: "
                  f"{merge['concordanza']} "
                  f"(distanza centri: "
                  f"{merge['distanza_centri']} punti)")
            print(f"  [LSTM] Fascia Wyckoff:  "
                  f"{fascia_w[0]}-{fascia_w[1]}")
            print(f"  [LSTM] Fascia merged:   "
                  f"{merge['fascia_merged'][0]}-"
                  f"{merge['fascia_merged'][1]}")

        # Aggiorna stato con fascia merged
        stato_aggiornato = stato_wyckoff.copy()
        stato_aggiornato['fascia_min']        = \
            merge['fascia_merged'][0]
        stato_aggiornato['fascia_max']        = \
            merge['fascia_merged'][1]
        stato_aggiornato['fascia_lstm_pred']  = \
            pred['somma_pred']
        stato_aggiornato['fascia_lstm_low']   = \
            pred['conf_low']
        stato_aggiornato['fascia_lstm_high']  = \
            pred['conf_high']
        stato_aggiornato['lstm_concordanza']  = \
            merge['concordanza']
        stato_aggiornato['lstm_distanza']     = \
            merge['distanza_centri']

        return stato_aggiornato

    except Exception as e:
        if verbose:
            print(f"  [LSTM] Errore: {e} → uso solo Wyckoff")
        return stato_wyckoff
