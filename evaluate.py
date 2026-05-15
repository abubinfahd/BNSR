"""
evaluate.py — Evaluation helpers: overall metrics, per-difficulty metrics,
confidence / uncertainty analysis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import jiwer

from config import (
    BATCH_SIZE, NUM_CLASSES, MAX_LABEL_LENGTH,
    idx_to_char, char_to_idx, LEVEL_MAP,
)
from models._ctc_utils import greedy_decode, seqs_to_strs
from data_utils import labels_to_strs


# ─── Overall evaluation ───────────────────────────────────────────────────────

def evaluate_overall(model: tf.keras.Model,
                     X: np.ndarray,
                     Y: np.ndarray,
                     batch: int = BATCH_SIZE) -> dict:
    """
    Compute exact-match accuracy, CER, WER on (X, Y).

    Returns
    -------
    dict with keys: exact, cer, wer, gt (list[str]), pred (list[str])
    """
    preds   = model.predict(X, batch_size=batch, verbose=0)
    decoded = greedy_decode(preds)
    pred_s  = seqs_to_strs(decoded, idx_to_char)
    gt_s    = labels_to_strs(Y)

    exact = sum(g == p for g, p in zip(gt_s, pred_s)) / len(gt_s) * 100
    cer   = jiwer.cer(gt_s, pred_s) * 100
    wer   = jiwer.wer(gt_s, pred_s) * 100

    return dict(exact=exact, cer=cer, wer=wer, gt=gt_s, pred=pred_s)


# ─── Per-difficulty evaluation ────────────────────────────────────────────────

def evaluate_by_difficulty(model: tf.keras.Model,
                            test_df: pd.DataFrame,
                            test_X: np.ndarray,
                            test_Y: np.ndarray,
                            batch: int = BATCH_SIZE) -> pd.DataFrame:
    """
    Evaluate model performance split by difficulty level (easy/medium/hard).

    Parameters
    ----------
    test_df : DataFrame with a 'difficulty' column aligned to test_X / test_Y.

    Returns
    -------
    pd.DataFrame with columns: difficulty, n_samples, exact, cer, wer
    """
    preds   = model.predict(test_X, batch_size=batch, verbose=0)
    decoded = greedy_decode(preds)
    pred_s  = seqs_to_strs(decoded, idx_to_char)
    gt_s    = labels_to_strs(test_Y)

    rows = []
    for level in ["easy", "medium", "hard"]:
        idx = test_df.index[test_df["difficulty"] == level].tolist()
        if len(idx) == 0:
            rows.append(dict(difficulty=level, n_samples=0,
                             exact=float("nan"), cer=float("nan"),
                             wer=float("nan")))
            continue

        # Re-align: test_df may not be 0-indexed after splitting
        rel_idx = [i for i, pos in enumerate(test_df.itertuples())
                   if pos.difficulty == level]

        gt_lvl   = [gt_s[i]   for i in rel_idx]
        pred_lvl = [pred_s[i] for i in rel_idx]

        exact_lvl = sum(g == p for g, p in zip(gt_lvl, pred_lvl)) / len(gt_lvl) * 100
        cer_lvl   = jiwer.cer(gt_lvl, pred_lvl) * 100
        wer_lvl   = jiwer.wer(gt_lvl, pred_lvl) * 100

        rows.append(dict(
            difficulty = level,
            n_samples  = len(gt_lvl),
            exact      = round(exact_lvl, 2),
            cer        = round(cer_lvl,   2),
            wer        = round(wer_lvl,   2),
        ))

    return pd.DataFrame(rows)


# ─── Confidence / uncertainty analysis ───────────────────────────────────────

def confidence_analysis(model: tf.keras.Model,
                         X: np.ndarray,
                         gt_s: list,
                         pred_s: list,
                         batch: int = BATCH_SIZE) -> pd.DataFrame:
    """
    Compute per-sample max-softmax confidence and Shannon entropy.

    Returns
    -------
    pd.DataFrame with columns:
        gt, pred, correct, max_prob_mean, entropy_mean
    One row per sample; max_prob and entropy averaged over time steps.
    """
    preds = model.predict(X, batch_size=batch, verbose=0)  # (B, T, C+1)

    max_probs = preds.max(axis=-1).mean(axis=1)            # (B,)
    eps = 1e-8
    entropy   = -(preds * np.log(preds + eps)).sum(axis=-1).mean(axis=1)  # (B,)

    df = pd.DataFrame({
        "gt":         gt_s,
        "pred":       pred_s,
        "correct":    [g == p for g, p in zip(gt_s, pred_s)],
        "max_prob":   max_probs,
        "entropy":    entropy,
    })
    return df


# ─── Keras callback for per-epoch validation accuracy ────────────────────────

class ValAccCallback(tf.keras.callbacks.Callback):
    """
    Computes exact-match accuracy and CER at each epoch end.
    Stores logs for later plotting.
    """

    def __init__(self, val_X: np.ndarray, val_Y: np.ndarray):
        super().__init__()
        self.val_X = val_X
        self.val_Y = val_Y
        self.history: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        m = evaluate_overall(self.model, self.val_X, self.val_Y)
        logs["val_exact"] = m["exact"]
        logs["val_cer"]   = m["cer"]
        self.history.append(logs.copy())
