"""
train.py — Multi-seed training loop for all architectures × aug degrees.
           Checkpoint-aware: skips already-completed runs on resume.

Outputs
-------
outputs/results/checkpoint.json        — live progress (updated after each run)
outputs/results/raw_seed_results.csv   — one row per (arch, aug, seed)
outputs/results/multi_seed_summary.csv — mean ± std across seeds per (arch, aug)
outputs/models/<arch>_aug<n>_seed<s>.keras
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    SEEDS, AUG_DEGREES, ARCHITECTURES,
    BATCH_SIZE, EPOCHS, RESULTS_DIR, MODELS_DIR, CHECKPOINTS_DIR,
    ensure_dirs,
)
ensure_dirs()  # guarantee all output folders exist

from data_utils import normalize, augment_rotation, encode_labels
from models import build_model
from evaluate import evaluate_overall, ValAccCallback
from checkpoint import (
    load_checkpoint, mark_done, is_done,
    checkpoint_to_records, print_remaining,
)


# ─── Multi-GPU strategy (auto-detected) ───────────────────────────────────────
def _get_strategy():
    """
    Automatically detect and return the best distribution strategy.

    - 2+ GPUs (e.g. Kaggle T4 × 2) → MirroredStrategy  (uses both GPUs)
    - 1 GPU  (e.g. Kaggle P100)     → OneDeviceStrategy  (single GPU)
    - No GPU                         → default CPU strategy
    """
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        print(f"  [GPU] {len(gpus)} GPUs detected → using MirroredStrategy")
        return tf.distribute.MirroredStrategy(), len(gpus)
    elif len(gpus) == 1:
        print(f"  [GPU] 1 GPU detected → single-GPU training")
        return tf.distribute.OneDeviceStrategy('/gpu:0'), 1
    else:
        print("  [GPU] No GPU — using CPU")
        return tf.distribute.get_strategy(), 1


STRATEGY, N_GPUS = _get_strategy()
# Scale batch size linearly with GPU count for efficient multi-GPU utilisation
EFFECTIVE_BATCH  = BATCH_SIZE * N_GPUS
print(f"  [GPU] Effective batch size: {EFFECTIVE_BATCH} "
      f"({BATCH_SIZE} × {N_GPUS} GPUs)")


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_run(arch: str, n_aug: int, seed: int,
                  train_imgs_raw, train_lbls,
                  val_X, val_Y, test_X, test_Y,
                  smoke_test: bool = False) -> dict:
    """
    Train a single (arch, aug, seed) configuration.

    Returns
    -------
    dict with keys: arch, n_aug, seed, exact, cer, wer,
                    params, epochs_run, train_time_s, history, model
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

    # ── Model (built inside strategy scope for multi-GPU support) ───────────────
    with STRATEGY.scope():
        model = build_model(arch)
    n_params = model.count_params()

    run_name   = f"{arch}_aug{n_aug}_seed{seed}"
    # Mid-training best-weight checkpoint (saved by Keras callback each epoch)
    ckpt_path  = os.path.join(CHECKPOINTS_DIR, f"{run_name}.keras")
    # Final saved model location (written explicitly after training is done)
    model_path = os.path.join(MODELS_DIR,      f"{run_name}.keras")

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
        batch_size=EFFECTIVE_BATCH,   # scaled for multi-GPU
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    res = evaluate_overall(model, test_X, test_Y)

    # ── Persist final model to models/ ─────────────────────────────────────
    model.save(model_path)
    print(f"  → Model saved: {model_path}")

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
                            seeds=None, smoke_test=False,
                            resume=True):
    """
    Full multi-seed training loop with checkpoint-based resume support.

    Parameters
    ----------
    resume : bool
        If True (default), load checkpoint.json and skip completed runs.
        Set to False only if you want a clean restart.

    Returns
    -------
    raw_records : list of dicts (one per arch × aug × seed)
    summary_df  : pd.DataFrame with mean ± std columns
    best_run    : dict for the best (arch, aug) by mean exact accuracy
    """
    architectures = architectures or ARCHITECTURES
    aug_degrees   = aug_degrees   or AUG_DEGREES
    seeds         = seeds         or SEEDS

    # ── Load checkpoint ───────────────────────────────────────────────────────
    state = load_checkpoint() if resume else {"completed": []}
    print_remaining(state, architectures, aug_degrees, seeds)

    # Seed-level run storage (for median-run selection)
    arch_aug_runs: dict = {}    # (arch, n_aug) → list of run dicts
    best_run = None

    # ── Restore already-completed records from checkpoint ─────────────────────
    for rec in checkpoint_to_records(state):
        key = (rec["arch"], rec["n_aug"])
        arch_aug_runs.setdefault(key, []).append(rec)

    # ── Main training loop ────────────────────────────────────────────────────
    for n_aug in aug_degrees:
        for arch in architectures:
            key = (arch, n_aug)
            arch_aug_runs.setdefault(key, [])

            for seed in seeds:
                # ── SKIP if already done ──────────────────────────────────────
                if is_done(state, arch, n_aug, seed):
                    print(f"  [SKIP] arch={arch}  aug={n_aug}  seed={seed}  "
                          f"(already in checkpoint)")
                    continue

                print(f"\n── aug={n_aug}  arch={arch}  seed={seed} ──")

                run = train_one_run(
                    arch, n_aug, seed,
                    train_imgs_raw, train_lbls,
                    val_X, val_Y, test_X, test_Y,
                    smoke_test=smoke_test,
                )

                # ── Save to checkpoint IMMEDIATELY ────────────────────────────
                mark_done(state, run)

                arch_aug_runs[key].append(run)

            # ── Pick median-seed run for plots / error analysis ───────────────
            seed_runs   = arch_aug_runs.get(key, [])
            seed_exacts = [r["exact"] for r in seed_runs
                           if isinstance(r, dict) and "exact" in r
                           and "model" in r]   # only runs with model object
            if seed_exacts:
                median_idx   = int(np.argsort(seed_exacts)[len(seed_exacts) // 2])
                candidate    = [r for r in seed_runs if "model" in r][median_idx]
                candidate["seed_exact_all"] = seed_exacts
                if best_run is None or candidate["exact"] > best_run.get("exact", 0):
                    best_run = candidate

    # ── Build full raw_records from checkpoint (includes previous sessions) ───
    all_completed = checkpoint_to_records(state)
    raw_records   = all_completed   # flat list of serialisable dicts

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(
        os.path.join(RESULTS_DIR, "raw_seed_results.csv"), index=False
    )

    # ── Aggregate mean ± std ──────────────────────────────────────────────────
    summary_df = _build_summary(raw_df)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "multi_seed_summary.csv"), index=False
    )

    print("\n\nMULTI-SEED SUMMARY (all completed runs):")
    print(
        summary_df[["arch", "n_aug", "exact_pm", "cer_pm",
                    "params", "epochs_run_mean"]]
        .to_string(index=False)
    )

    # ── Best config ───────────────────────────────────────────────────────────
    best_row = summary_df.loc[summary_df["exact_mean"].idxmax()]
    print(
        f"\nBest config: arch={best_row['arch']}  "
        f"aug={best_row['n_aug']}  "
        f"exact={best_row['exact_pm']}"
    )

    # Fallback if best_run has no model (e.g. fully resumed from checkpoint)
    if best_run is None:
        print(
            "\n  [checkpoint] All runs were loaded from checkpoint. "
            "best_run has no live model — re-training best config for analysis …"
        )
        best_arch = best_row["arch"]
        best_aug  = int(best_row["n_aug"])
        best_seed = seeds[0]
        best_run  = train_one_run(
            best_arch, best_aug, best_seed,
            train_imgs_raw, train_lbls,
            val_X, val_Y, test_X, test_Y,
            smoke_test=smoke_test,
        )

    return raw_records, summary_df, best_run


# ─── Summary helper ───────────────────────────────────────────────────────────

def _build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    grp = raw_df.groupby(["arch", "n_aug"])
    rows = []
    for (arch, n_aug), g in grp:
        rows.append(dict(
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
    df = pd.DataFrame(rows)
    df["exact_pm"] = (df["exact_mean"].map("{:.2f}".format)
                      + " ± " + df["exact_std"].map("{:.2f}".format))
    df["cer_pm"]   = (df["cer_mean"].map("{:.2f}".format)
                      + " ± " + df["cer_std"].map("{:.2f}".format))
    return df
