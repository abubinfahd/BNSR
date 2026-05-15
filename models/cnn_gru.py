"""
models/cnn_gru.py — (B) CNN + Bidirectional GRU.

Purpose: controlled LSTM vs GRU comparison under identical CNN backbone.
Architecture:
    3× [Conv2D → BN → Conv2D → MaxPool → Dropout]
    → collapse H
    → BiGRU(128) → BiGRU(64)
    → TimeDistributed Dense (CTC softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Bidirectional, GRU, TimeDistributed,
)

from config import INPUT_SHAPE, NUM_CLASSES
from models._ctc_utils import ctc_loss_fn


def build_cnn_gru() -> tf.keras.Model:
    """
    CNN + Bidirectional GRU — controlled comparison against CRNN (LSTM).

    Returns a compiled Keras model.
    """
    inp = Input(shape=INPUT_SHAPE, name="image")
    x = inp

    # Shared CNN backbone (identical to CRNN)
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.15)(x)

    # Collapse height: (B, H/8, W/8, 128) → (B, W/8, 128)
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="collapse_height"
    )(x)

    # Bidirectional GRU stack (mirrors CRNN-base but with GRU cells)
    x = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.25), name="bigru_1"
    )(x)
    x = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.25), name="bigru_2"
    )(x)

    # CTC output
    out = TimeDistributed(
        Dense(NUM_CLASSES + 1, activation="softmax"), name="output"
    )(x)

    mdl = models.Model(inp, out, name="CNN_GRU")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ctc_loss_fn,
    )
    return mdl
