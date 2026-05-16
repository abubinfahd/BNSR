"""
computational_analysis.py — FLOPs estimate, inference latency (ms/image),
and model size (MB) for all trained architectures.

Outputs
-------
outputs/results/computational_analysis.csv
outputs/figures/fig_compute_tradeoff.pdf
"""

import os
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, FIGURES_DIR, BATCH_SIZE, MODELS_DIR, INPUT_SHAPE,
    ensure_dirs,
)
ensure_dirs()  # guarantee all output folders exist



# ─── Inference latency ───────────────────────────────────────────────────────

def measure_latency(model: tf.keras.Model,
                    X: np.ndarray,
                    n_warmup: int = 5,
                    n_measure: int = 50) -> float:
    """
    Measure mean inference latency in ms/image.

    Uses a compiled tf.function for direct model inference instead of
    model.predict() to avoid the CuDNN LSTM ``max_seq_length <= 0``
    error that occurs when predict() distributes the single-sample
    batch across replica threads and produces a zero-length slice.

    Parameters
    ----------
    n_warmup  : warm-up runs (discarded)
    n_measure : measured runs to average

    Returns
    -------
    float : ms per image
    """
    sample = tf.constant(X[:1], dtype=tf.float32)  # shape (1, H, W, C)

    @tf.function
    def _forward(x):
        return model(x, training=False)

    # Warm-up — also triggers tracing
    for _ in range(n_warmup):
        _forward(sample)

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _forward(sample)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ─── Model size (MB) ─────────────────────────────────────────────────────────

def model_size_mb(model: tf.keras.Model, arch: str, aug: int, seed: int) -> float:
    """
    Return saved .keras file size in MB.
    If the file doesn't exist, estimate from parameter count.
    """
    path = os.path.join(MODELS_DIR, f"{arch}_aug{aug}_seed{seed}.keras")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    # Fallback: float32 params × 4 bytes
    return model.count_params() * 4 / (1024 ** 2)


# ─── FLOPs estimate ──────────────────────────────────────────────────────────

def estimate_flops(model: tf.keras.Model) -> float:
    """
    Approximate multiply-accumulate (MAC) count for one forward pass.

    Uses TensorFlow's profiling API when available, otherwise falls back to
    a parameter-based heuristic (2 × params for dense / 2 × params × kernel
    for conv).

    Returns
    -------
    float : estimated FLOPs (multiply-add counted as 2 ops)
    """
    try:
        # TF2 concrete function approach
        @tf.function
        def forward(x):
            return model(x, training=False)

        concrete = forward.get_concrete_function(
            tf.TensorSpec([1] + list(INPUT_SHAPE), tf.float32)
        )
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            concrete.graph, options=opts
        ).total_float_ops
        return float(flops)
    except Exception:
        # Heuristic fallback
        return float(model.count_params()) * 2.0


# ─── Full computational analysis ─────────────────────────────────────────────

def run_computational_analysis(trained_runs: list,
                                test_X: np.ndarray) -> pd.DataFrame:
    """
    Collect params, FLOPs, latency, size for each trained run.

    Parameters
    ----------
    trained_runs : list of dicts (from train.run_multi_seed_training raw_records
                   + model objects — pass the run dicts that include 'model')
    test_X       : test images numpy array

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    seen = set()

    for run in trained_runs:
        arch = run["arch"]
        if arch in seen:
            continue          # one entry per architecture
        seen.add(arch)

        model = run.get("model")
        if model is None:
            continue

        latency = measure_latency(model, test_X)
        flops   = estimate_flops(model)
        size_mb = model_size_mb(
            model, arch, run.get("n_aug", 0), run.get("seed", 42)
        )

        rows.append(dict(
            arch            = arch,
            params          = model.count_params(),
            params_M        = round(model.count_params() / 1e6, 3),
            flops_G         = round(flops / 1e9, 3),
            latency_ms      = round(latency, 2),
            model_size_mb   = round(size_mb, 2),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(RESULTS_DIR, "computational_analysis.csv"), index=False
    )
    print("\nCOMPUTATIONAL ANALYSIS:")
    print(df.to_string(index=False))

    _plot_compute_tradeoff(df)
    return df


# ─── Figure ──────────────────────────────────────────────────────────────────

def _plot_compute_tradeoff(df: pd.DataFrame):
    """Scatter: latency vs params, bubble = model size."""
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = [
        "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
        "#43A047", "#FB8C00", "#F4511E",
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        color = palette[i % len(palette)]
        ax.scatter(
            row["params_M"], row["latency_ms"],
            s=max(row["model_size_mb"] * 80, 80),
            color=color, alpha=0.8, edgecolors="black", linewidths=0.6,
            label=row["arch"],
        )
        ax.annotate(
            row["arch"],
            (row["params_M"], row["latency_ms"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Inference Latency (ms / image)", fontsize=12)
    ax.set_title(
        "Computational Cost Trade-off\n(bubble size = model size MB)",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, title="Architecture")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig_compute_tradeoff.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()
