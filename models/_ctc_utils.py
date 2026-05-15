"""
models/_ctc_utils.py — Shared CTC loss and greedy decoder used by all models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from config import NUM_CLASSES, CTC_BLANK


def ctc_loss_fn(y_true, y_pred):
    """
    CTC loss compatible with Keras model.compile(loss=...).

    y_true : (B, MAX_LABEL_LENGTH)  int32, padded with -1
    y_pred : (B, T, NUM_CLASSES+1)  float32 softmax probabilities
    """
    batch   = tf.shape(y_pred)[0]
    seq_len = tf.shape(y_pred)[1]
    inp_len = tf.fill([batch], seq_len)
    lbl_len = tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1
    )
    labels  = tf.cast(tf.maximum(y_true, 0), tf.int32)
    # TF ctc_loss expects log-probabilities in time-major format
    logits  = tf.math.log(tf.transpose(y_pred, [1, 0, 2]) + 1e-8)
    loss = tf.nn.ctc_loss(
        labels=labels, logits=logits,
        label_length=lbl_len, logit_length=inp_len,
        logits_time_major=True, blank_index=CTC_BLANK,
    )
    return tf.reduce_mean(loss)


def greedy_decode(preds: np.ndarray) -> np.ndarray:
    """
    CTC greedy decoding.

    Parameters
    ----------
    preds : (B, T, NUM_CLASSES+1)

    Returns
    -------
    decoded : (B, MAX_DECODE_LEN)  int32
    """
    inp_len = np.full(len(preds), preds.shape[1], dtype=np.int32)
    decoded, _ = K.ctc_decode(preds, input_length=inp_len, greedy=True)
    return decoded[0].numpy()


def seqs_to_strs(seqs: np.ndarray, idx_to_char: dict) -> list:
    """Convert decoded integer matrix → list of strings."""
    return [
        "".join(idx_to_char[i] for i in row if 0 <= i < NUM_CLASSES)
        for row in seqs
    ]
