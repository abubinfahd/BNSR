"""
models/transformer_ocr.py — (E) Lightweight Transformer OCR + CTC.

Minimal 2-layer Transformer encoder for OCR. Keeps computation feasible on
Kaggle while satisfying the IJDAR requirement for a Transformer baseline.

Architecture:
    Conv2D patch embedding  (stride-based, no separate tokenizer)
    Sinusoidal positional encoding
    2× Transformer encoder block (MultiHeadAttention + FFN)
    CTC head (TimeDistributed Dense softmax)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Dense, Dropout,
    TimeDistributed, LayerNormalization, MultiHeadAttention,
)

from config import INPUT_SHAPE, NUM_CLASSES, IMG_W, IMG_H
from models._ctc_utils import ctc_loss_fn


# ─── Positional Encoding ──────────────────────────────────────────────────────

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Return (1, seq_len, d_model) sinusoidal positional encodings."""
    positions = np.arange(seq_len)[:, np.newaxis]          # (T, 1)
    dims      = np.arange(d_model)[np.newaxis, :]           # (1, d_model)
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)  # (1, T, d_model)


# ─── Transformer Encoder Block ────────────────────────────────────────────────

class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer encoder block:
        MultiHeadAttention → Add & Norm → FFN → Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha  = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn1 = Dense(dff, activation="relu")
        self.ffn2 = Dense(d_model)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Self-attention sub-layer
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Feed-forward sub-layer
        ffn_out = self.ffn2(self.ffn1(x))
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":      self.ffn2.units,
            "num_heads":    self.mha.num_heads,
            "dff":          self.ffn1.units,
            "dropout_rate": self.drop1.rate,
        })
        return config


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_transformer_ocr(
    d_model: int   = 128,
    num_heads: int = 4,
    dff: int       = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    Lightweight Transformer OCR with CTC head.

    Patch embedding:
        Two strided Conv2D layers produce (B, H', W', d_model) feature maps.
        Height is collapsed via mean pooling, yielding (B, T, d_model) tokens.

    Parameters
    ----------
    d_model   : Transformer embedding dimension.
    num_heads : Number of attention heads.
    dff       : Feed-forward inner dimension.
    num_layers: Number of Transformer encoder blocks (default 2).
    dropout   : Dropout rate inside Transformer blocks.

    Returns a compiled Keras model.
    """
    inp = Input(shape=INPUT_SHAPE, name="image")

    # ── Patch embedding (stride-based) ────────────────────────────────────────
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same",
               activation="relu", name="patch_embed_1")(inp)
    x = BatchNormalization()(x)
    x = Conv2D(d_model, (3, 3), strides=(2, 2), padding="same",
               activation="relu", name="patch_embed_2")(x)
    x = BatchNormalization()(x)
    # After 2× stride-2: spatial = (IMG_H/4, IMG_W/4)
    # Further pool height to 1 via Lambda mean
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="collapse_height"
    )(x)   # (B, IMG_W/4, d_model)

    # ── Positional encoding ───────────────────────────────────────────────────
    seq_len = IMG_W // 4
    pos_enc = sinusoidal_positional_encoding(seq_len, d_model)
    x = x + pos_enc   # broadcast over batch

    x = Dropout(dropout)(x)

    # ── Transformer encoder blocks ────────────────────────────────────────────
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout_rate=dropout, name=f"transformer_block_{i}"
        )(x)

    # ── CTC head ──────────────────────────────────────────────────────────────
    out = TimeDistributed(
        Dense(NUM_CLASSES + 1, activation="softmax"), name="output"
    )(x)

    mdl = models.Model(inp, out, name="Transformer_OCR")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ctc_loss_fn,
    )
    return mdl
