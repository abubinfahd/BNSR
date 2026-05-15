"""
train.py — Multi-seed training loop for all architectures × aug degrees.

Outputs
-------
outputs/results/raw_seed_results.csv   — one row per (arch, aug, seed)
outputs/results/multi_seed_summary.csv — mean ± std across seeds per (arch, aug)
outputs/models/<run_name>_seed<seed>.keras
"""

import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    SEEDS, AUG_DEGREES, ARCHITECTURES,
    BATCH_SIZE, EPOCHS, RESULTS_DIR, MODELS_DIR,
)
from data_utils import (
    load_split, normalize, augment_rotation, encode_labels, labels_to_strs,
)
from models import build_model
from evaluate import evaluate_overall, ValAccCallback


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_run(arch: str, n_aug: int, seed: int,
                  train_imgs_raw, train_lbls,
                  val_X, val_Y, test_X, test_Y,
                  smoke_test: bool = False):
    """
    Train a single (arch, aug, seed) configuration.

    Returns
    -------
    dict with keys: arch, n_aug, seed, exact, cer, wer,
                    params, epochs_run, train_time_s, history
    """
    set_seeds(seed)

    # ── Augmentation ──────────────────────────────────────────────────────────
    if n_aug == 0:
        aug_imgs, aug_lbls = train_imgs_raw, train_lbls
    else:
        aug_imgs, aug_lbls = augment_rotation(
            train_imgs_raw, train_lbls, n_aug=n_aug, seed=seed
        )
    train_X = normalize(aug_imgs)
    train_Y, _ = encode_labels(aug_lbls)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(arch)
    n_params = model.count_params()

    run_name = f"{arch}_aug{n_aug}_seed{seed}"
    ckpt_path = os.path.join(MODELS_DIR, f"{run_name}.keras")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    val_cb = ValAccCallback(val_X, val_Y)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
        val_cb,
    ]

    epochs = 2 if smoke_test else EPOCHS

    # ── Training ──────────────────────────────────────────────────────────────
    t0 = time.time()
    hist = model.fit(
        train_X, train_Y,
        validation_data=(val_X, val_Y),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    res = evaluate_overall(model, test_X, test_Y)

    print(
        f"  [{run_name}]  "
        f"Exact={res['exact']:.2f}%  CER={res['cer']:.2f}%  "
        f"WER={res['wer']:.2f}%  t={elapsed:.0f}s"
    )

    return dict(
        arch         = arch,
        n_aug        = n_aug,
        seed         = seed,
        train_size   = len(train_X),
        params       = n_params,
        epochs_run   = len(hist.history["loss"]),
        train_time_s = round(elapsed, 1),
        exact        = round(res["exact"], 4),
        cer          = round(res["cer"],   4),
        wer          = round(res["wer"],   4),
        gt           = res["gt"],
        pred         = res["pred"],
        history      = {
            "loss":      hist.history["loss"],
            "val_loss":  hist.history["val_loss"],
            "val_exact": [h.get("val_exact", float("nan"))
                          for h in val_cb.history],
        },
        model        = model,
    )


def run_multi_seed_training(train_imgs_raw, train_lbls,
                            val_X, val_Y, test_X, test_Y,
                            architectures=None, aug_degrees=None,
                            seeds=None, smoke_test=False):
    """
    Full multi-seed training loop.

    Returns
    -------
    raw_records : list of dicts (one per arch × aug × seed)
    summary_df  : pd.DataFrame with mean ± std columns
    best_run    : dict for the best (arch, aug) by mean exact accuracy
    """
    architectures = architectures or ARCHITECTURES
    aug_degrees   = aug_degrees   or AUG_DEGREES
    seeds         = seeds         or SEEDS

    raw_records = []

    for n_aug in aug_degrees:
        for arch in architectures:
            seed_exact, seed_cer, seed_wer = [], [], []
            seed_runs = []

            for seed in seeds:
                print(f"\n── aug={n_aug}  arch={arch}  seed={seed} ──")
                run = train_one_run(
                    arch, n_aug, seed,
                    train_imgs_raw, train_lbls,
                    val_X, val_Y, test_X, test_Y,
                    smoke_test=smoke_test,
                )
                raw_records.append({
                    k: v for k, v in run.items()
                    if k not in ("gt", "pred", "model", "history")
                })
                seed_exact.append(run["exact"])
                seed_cer.append(run["cer"])
                seed_wer.append(run["wer"])
                seed_runs.append(run)

            # Keep the median-seed run for plots / error analysis
            median_idx = int(np.argsort(seed_exact)[len(seeds) // 2])
            best_seed_run = seed_runs[median_idx]
            best_seed_run["seed_exact_all"] = seed_exact
            best_seed_run["seed_cer_all"]   = seed_cer

    # ── Save raw results ───────────────────────────────────────────────────────
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(
        os.path.join(RESULTS_DIR, "raw_seed_results.csv"), index=False
    )

    # ── Aggregate mean ± std ───────────────────────────────────────────────────
    grp = raw_df.groupby(["arch", "n_aug"])
    summary_rows = []
    for (arch, n_aug), g in grp:
        summary_rows.append(dict(
            arch              = arch,
            n_aug             = n_aug,
            params            = g["params"].iloc[0],
            train_size        = g["train_size"].iloc[0],
            exact_mean        = round(g["exact"].mean(), 4),
            exact_std         = round(g["exact"].std(),  4),
            cer_mean          = round(g["cer"].mean(),   4),
            cer_std           = round(g["cer"].std(),    4),
            wer_mean          = round(g["wer"].mean(),   4),
            wer_std           = round(g["wer"].std(),    4),
            epochs_run_mean   = round(g["epochs_run"].mean(), 1),
            train_time_s_mean = round(g["train_time_s"].mean(), 1),
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df["exact_pm"] = (
        summary_df["exact_mean"].map("{:.2f}".format)
        + " ± "
        + summary_df["exact_std"].map("{:.2f}".format)
    )
    summary_df["cer_pm"] = (
        summary_df["cer_mean"].map("{:.2f}".format)
        + " ± "
        + summary_df["cer_std"].map("{:.2f}".format)
    )
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "multi_seed_summary.csv"), index=False
    )
    print("\n\nMULTI-SEED SUMMARY:")
    print(
        summary_df[["arch", "n_aug", "exact_pm", "cer_pm",
                    "params", "epochs_run_mean"]]
        .to_string(index=False)
    )

    # ── Find best overall (arch, aug) by mean exact ────────────────────────────
    best_row = summary_df.loc[summary_df["exact_mean"].idxmax()]
    print(
        f"\nBest config: arch={best_row['arch']}  "
        f"aug={best_row['n_aug']}  "
        f"exact={best_row['exact_pm']}"
    )

    return raw_records, summary_df, best_seed_run
