"""
models/__init__.py — Registry: map architecture name → build function.
"""

from models.cnn_ctc        import build_cnn_ctc
from models.cnn_gru        import build_cnn_gru
from models.crnn           import build_crnn
from models.attention_ocr  import build_attention_ocr
from models.transformer_ocr import build_transformer_ocr


def build_model(arch: str):
    """
    Factory function — returns a compiled Keras model for the given arch name.

    Supported arch strings
    ----------------------
    cnn_ctc       : (A) CNN → Dense → CTC  (no recurrence)
    cnn_gru       : (B) CNN + Bidirectional GRU
    crnn_small    : (C) CRNN, 1× BiLSTM-128
    crnn_base     : (C) CRNN, 2× BiLSTM
    crnn_large    : (C) CRNN, 3× BiLSTM
    attention     : (D) CNN + BiLSTM + Bahdanau Attention + CTC
    transformer   : (E) Patch Embed + 2-layer Transformer Encoder + CTC
    """
    dispatch = {
        "cnn_ctc":     lambda: build_cnn_ctc(),
        "cnn_gru":     lambda: build_cnn_gru(),
        "crnn_small":  lambda: build_crnn("small"),
        "crnn_base":   lambda: build_crnn("base"),
        "crnn_large":  lambda: build_crnn("large"),
        "attention":   lambda: build_attention_ocr(),
        "transformer": lambda: build_transformer_ocr(),
    }
    if arch not in dispatch:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {list(dispatch.keys())}"
        )
    return dispatch[arch]()


__all__ = ["build_model"]
