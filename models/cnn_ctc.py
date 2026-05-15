"""
models/cnn_ctc.py — (A) Weakest baseline: CNN + Dense → CTC, no recurrence.

Purpose: demonstrates the contribution of RNN layers by comparison.
Architecture:
    3× [Conv2D → BN → Conv2D → MaxPool → Dropout]
    → collapse H via GlobalAveragePooling over height axis
    → TimeDistributed Dense (CTC softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    TimeDistributed,
)

from config import INPUT_SHAPE, NUM_CLASSES, CTC_BLANK, CNN_SEQ_LEN
from models._ctc_utils import ctc_loss_fn


def build_cnn_ctc() -> tf.keras.Model:
    """
    Simple CNN → CTC baseline (no recurrent layers).

    Returns a compiled Keras model.
    """
    inp = Input(shape=INPUT_SHAPE, name="image")
    x = inp

    # CNN feature extractor — 3 blocks, same backbone as CRNN
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.15)(x)

    # Collapse height dimension: (B, H/8, W/8, 128) → (B, W/8, 128)
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="collapse_height"
    )(x)

    # Dense projection → intermediate representation
    x = TimeDistributed(Dense(256, activation="relu"), name="dense_proj")(x)
    x = Dropout(0.3)(x)

    # CTC output
    out = TimeDistributed(
        Dense(NUM_CLASSES + 1, activation="softmax"), name="output"
    )(x)

    mdl = models.Model(inp, out, name="CNN_CTC")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ctc_loss_fn,
    )
    return mdl
