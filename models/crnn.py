"""
models/crnn.py — (C) CRNN: CNN + Bidirectional LSTM (small / base / large).

Extracted verbatim from the original main.py build_crnn() with no logic changes.
Architecture:
    3× [Conv2D → BN → Conv2D → MaxPool → Dropout]
    → collapse H
    → BiLSTM stack (variant-dependent)
    → TimeDistributed Dense (CTC softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, TimeDistributed,
)

from config import INPUT_SHAPE, NUM_CLASSES
from models._ctc_utils import ctc_loss_fn


def build_crnn(variant: str = "base") -> tf.keras.Model:
    """
    CRNN with BiLSTM recurrence — three capacity variants.

    Parameters
    ----------
    variant : str
        'small'  — 1× BiLSTM-128
        'base'   — 2× BiLSTM (128 → 64)
        'large'  — 3× BiLSTM (256 → 128 → 64)

    Returns a compiled Keras model.
    """
    if variant not in ("small", "base", "large"):
        raise ValueError(f"variant must be 'small', 'base', or 'large', got '{variant}'")

    inp = Input(shape=INPUT_SHAPE, name="image")
    x = inp

    # CNN backbone — 3 blocks
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.15)(x)

    # Collapse H: (B, H/8, W/8, 128) → (B, W/8, 128)
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="collapse_height"
    )(x)

    # BiLSTM stack
    if variant == "small":
        x = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.25), name="bilstm_1"
        )(x)
    elif variant == "base":
        x = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.25), name="bilstm_1"
        )(x)
        x = Bidirectional(
            LSTM(64,  return_sequences=True, dropout=0.25), name="bilstm_2"
        )(x)
    else:  # large
        x = Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.25), name="bilstm_1"
        )(x)
        x = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.25), name="bilstm_2"
        )(x)
        x = Bidirectional(
            LSTM(64,  return_sequences=True, dropout=0.25), name="bilstm_3"
        )(x)

    # CTC output
    out = TimeDistributed(
        Dense(NUM_CLASSES + 1, activation="softmax"), name="output"
    )(x)

    mdl = models.Model(inp, out, name=f"CRNN_{variant}")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ctc_loss_fn,
    )
    return mdl
