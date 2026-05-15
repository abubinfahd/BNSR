"""
models/attention_ocr.py — (D) CNN + BiLSTM + Bahdanau Attention + CTC.

Critical for IJDAR reviewers. Uses attention-weighted context over encoder
states, then a TimeDistributed Dense CTC head — keeping the same loss interface
as all other models (no teacher-forcing required).

Architecture:
    3× [Conv2D → BN → Conv2D → MaxPool → Dropout]   ← shared CNN backbone
    → collapse H
    → BiLSTM encoder  (128 units)
    → Bahdanau self-attention over encoder states
    → Dropout
    → TimeDistributed Dense (CTC softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, TimeDistributed, Attention,
)

from config import INPUT_SHAPE, NUM_CLASSES
from models._ctc_utils import ctc_loss_fn


class BahdanauSelfAttention(tf.keras.layers.Layer):
    """
    Additive (Bahdanau-style) self-attention over a sequence.

    For each time step t, computes a context vector as a weighted sum over
    all encoder hidden states, where the weights are computed using a small
    feed-forward alignment network.

    Input shape  : (B, T, H)
    Output shape : (B, T, H)   — each step attended over all other steps
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_query = Dense(units, use_bias=False)
        self.W_key   = Dense(units, use_bias=False)
        self.V       = Dense(1, use_bias=False)

    def call(self, encoder_output, training=False):
        # encoder_output: (B, T, H)
        query = self.W_query(encoder_output)   # (B, T, units)
        key   = self.W_key(encoder_output)     # (B, T, units)

        # Additive score: (B, T_q, T_k, units)
        query_expanded = tf.expand_dims(query, 2)  # (B, T, 1, units)
        key_expanded   = tf.expand_dims(key,   1)  # (B, 1, T, units)
        score = self.V(tf.nn.tanh(query_expanded + key_expanded))  # (B, T, T, 1)
        score = tf.squeeze(score, axis=-1)     # (B, T, T)

        weights = tf.nn.softmax(score, axis=-1)  # (B, T, T)
        context = tf.matmul(weights, encoder_output)  # (B, T, H)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_attention_ocr() -> tf.keras.Model:
    """
    CNN + BiLSTM + Bahdanau Self-Attention + CTC head.
    Strong IJDAR baseline that satisfies attention-mechanism requirement.

    Returns a compiled Keras model.
    """
    inp = Input(shape=INPUT_SHAPE, name="image")
    x = inp

    # Shared CNN backbone
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.15)(x)

    # Collapse H → (B, W/8, 128)
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="collapse_height"
    )(x)

    # BiLSTM encoder — produces hidden states h_1 … h_T
    enc = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.25), name="bilstm_encoder"
    )(x)  # (B, T, 256)

    # Bahdanau self-attention over encoder states
    attended = BahdanauSelfAttention(units=128, name="bahdanau_attention")(enc)
    # Residual connection: combine attended context with original encoder output
    x = layers.Add(name="residual_add")([enc, attended])
    x = layers.LayerNormalization(name="layer_norm")(x)
    x = Dropout(0.25)(x)

    # Optional second BiLSTM to refine attended features
    x = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.25), name="bilstm_refine"
    )(x)

    # CTC output
    out = TimeDistributed(
        Dense(NUM_CLASSES + 1, activation="softmax"), name="output"
    )(x)

    mdl = models.Model(inp, out, name="Attention_OCR")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ctc_loss_fn,
    )
    return mdl
