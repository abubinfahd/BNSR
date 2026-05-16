"""
plots.py — All publication-quality figures for the IJDAR submission.

Figures generated
-----------------
fig01_dataset_split.pdf
fig02_ablation_heatmap.pdf       — exact-match & CER heatmap (arch × aug)
fig03_aug_effect_line.pdf        — augmentation effect line plot
fig04_params_vs_acc.pdf          — bubble chart (params × accuracy)
fig05_final_metrics.pdf          — best-model bar chart
fig06_overlay_learning_curves.pdf — CRNN-base vs Attention vs CNN-CTC
fig07_difficulty_cer.pdf         — CER per difficulty level per model
fig08_confidence_hist.pdf        — entropy distribution (correct vs incorrect)
fig09_stat_significance.pdf      — p-value heatmap
fig10_latex_table                — generates ablation LaTeX table
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIGURES_DIR, RESULTS_DIR, ARCHITECTURES, ensure_dirs
ensure_dirs()  # guarantee all output folders exist



# ─── Colour palette ───────────────────────────────────────────────────────────
ARCH_COLORS = {
    "cnn_ctc":     "#E53935",
    "cnn_gru":     "#8E24AA",
    "crnn_small":  "#1E88E5",
    "crnn_base":   "#00ACC1",
    "crnn_large":  "#43A047",
    "attention":   "#FB8C00",
    "transformer": "#F4511E",
}
ARCH_MARKERS = {
    "cnn_ctc":     "o",
    "cnn_gru":     "s",
    "crnn_small":  "^",
    "crnn_base":   "D",
    "crnn_large":  "v",
    "attention":   "P",
    "transformer": "*",
}


# ─── fig01: Dataset split pie ─────────────────────────────────────────────────

def plot_dataset_split(train_n: int, val_n: int, test_n: int):
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes  = [train_n, val_n, test_n]
    labels = [
        f"Train\n{train_n:,}\n(70%)",
        f"Val\n{val_n:,}\n(20%)",
        f"Test\n{test_n:,}\n(10%)",
    ]
    colors = ["#2196F3", "#FF9800", "#F44336"]
    wedges, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Dataset Split (before augmentation)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig01_dataset_split.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig02: Ablation heatmap ──────────────────────────────────────────────────

def plot_ablation_heatmap(summary_df: pd.DataFrame):
    """
    Heatmap of mean exact accuracy and mean CER across (arch × aug).
    """
    pivot_exact = summary_df.pivot(
        index="arch", columns="n_aug", values="exact_mean"
    )
    pivot_cer   = summary_df.pivot(
        index="arch", columns="n_aug", values="cer_mean"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, pivot, title, cmap in zip(
        axes,
        [pivot_exact, pivot_cer],
        ["Exact-Match Accuracy (%) ↑", "CER (%) ↓"],
        ["YlGn", "YlOrRd"],
    ):
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"aug={c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Augmentation degree (rotations/image)")
        ax.set_ylabel("Architecture")
        for i, j in itertools.product(
            range(pivot.shape[0]), range(pivot.shape[1])
        ):
            ax.text(
                j, i, f"{pivot.values[i, j]:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="black",
            )
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle(
        "Ablation Study: Architecture × Augmentation (mean over seeds)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig02_ablation_heatmap.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig03: Aug effect line ───────────────────────────────────────────────────

def plot_aug_effect_line(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    for arch in summary_df["arch"].unique():
        sub = summary_df[summary_df["arch"] == arch].sort_values("n_aug")
        ax.errorbar(
            sub["n_aug"], sub["exact_mean"],
            yerr=sub["exact_std"],
            marker=ARCH_MARKERS.get(arch, "o"),
            color=ARCH_COLORS.get(arch, "gray"),
            linewidth=2, capsize=4, label=arch,
        )
    ax.set_xlabel("Augmentation degree (rotations per image)", fontsize=12)
    ax.set_ylabel("Exact-Match Accuracy (%) — mean ± std", fontsize=12)
    ax.set_title("Effect of Rotation Augmentation on Test Accuracy",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig03_aug_effect_line.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig04: Params vs accuracy bubble ─────────────────────────────────────────

def plot_params_vs_acc(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    for arch in summary_df["arch"].unique():
        sub = summary_df[summary_df["arch"] == arch]
        best = sub.loc[sub["exact_mean"].idxmax()]
        ax.scatter(
            best["params"] / 1e6, best["exact_mean"],
            s=best["n_aug"] * 80 + 120,
            color=ARCH_COLORS.get(arch, "gray"),
            alpha=0.8, edgecolors="black", linewidths=0.6,
            label=arch,
        )
    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Best Exact-Match Accuracy (%)", fontsize=12)
    ax.set_title("Model Complexity vs. Accuracy\n(bubble size = best aug degree)",
                 fontweight="bold")
    ax.legend(fontsize=9, title="Architecture"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig04_params_vs_acc.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig05: Final metrics bar ─────────────────────────────────────────────────

def plot_final_metrics(metrics: dict, model_name: str = "Best Model"):
    names  = ["Exact\nAcc (%)", "Precision\n(%)", "Recall\n(%)",
              "F1\n(%)", "CER (%)\n↓", "WER (%)\n↓"]
    vals   = [
        metrics["exact"], metrics["precision"], metrics["recall"],
        metrics["f1"],    metrics["cer"],       metrics["wer"],
    ]
    colors = ["#2196F3", "#4CAF50", "#009688", "#673AB7", "#FF5722", "#F44336"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom",
            fontweight="bold", fontsize=10,
        )
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"{model_name} — Final Evaluation Metrics",
                 fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig05_final_metrics.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig06: Overlay learning curves ──────────────────────────────────────────

def plot_overlay_learning_curves(run_histories: dict):
    """
    Overlay training curves for multiple architectures on the same axes.

    Parameters
    ----------
    run_histories : dict  {arch_label: history_dict}
        history_dict keys: 'loss', 'val_loss', 'val_exact'
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch_label, hist in run_histories.items():
        color  = ARCH_COLORS.get(arch_label, "gray")
        epochs = range(1, len(hist["loss"]) + 1)

        axes[0].plot(epochs, hist["val_loss"], color=color,
                     linewidth=2, label=arch_label)
        axes[1].plot(epochs, hist["val_exact"], color=color,
                     linewidth=2, label=arch_label)

    axes[0].set_title("Validation CTC Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Exact-Match Accuracy (%)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Exact-Match Accuracy (%)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle("Overlay Learning Curves — Architecture Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig06_overlay_learning_curves.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig07: Difficulty CER bar ────────────────────────────────────────────────

def plot_difficulty_cer(difficulty_results: dict):
    """
    Bar chart: CER per difficulty level for each architecture.

    Parameters
    ----------
    difficulty_results : dict  {arch: pd.DataFrame with columns difficulty, cer}
    """
    levels = ["easy", "medium", "hard"]
    archs  = list(difficulty_results.keys())
    x      = np.arange(len(levels))
    width  = 0.8 / max(len(archs), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, arch in enumerate(archs):
        df = difficulty_results[arch]
        cer_vals = [
            df.loc[df["difficulty"] == lvl, "cer"].values[0]
            if lvl in df["difficulty"].values else float("nan")
            for lvl in levels
        ]
        offsets = x + (i - len(archs) / 2 + 0.5) * width
        ax.bar(offsets, cer_vals, width=width,
               color=ARCH_COLORS.get(arch, "gray"),
               alpha=0.85, edgecolor="black", linewidth=0.5,
               label=arch)

    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=12)
    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_ylabel("CER (%) ↓", fontsize=12)
    ax.set_title("CER by Document Difficulty Level — Architecture Comparison",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, title="Architecture"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig07_difficulty_cer.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig08: Confidence / entropy histogram ────────────────────────────────────

def plot_confidence_histogram(conf_df: pd.DataFrame, model_name: str = "Best"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for correct, label, color in [
        (True,  "Correct",   "#43A047"),
        (False, "Incorrect", "#E53935"),
    ]:
        sub = conf_df[conf_df["correct"] == correct]
        axes[0].hist(sub["max_prob"], bins=30, alpha=0.6,
                     color=color, label=label, edgecolor="black", linewidth=0.3)
        axes[1].hist(sub["entropy"],  bins=30, alpha=0.6,
                     color=color, label=label, edgecolor="black", linewidth=0.3)

    axes[0].set_xlabel("Max Softmax Probability (mean over time steps)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence", fontweight="bold")
    axes[0].legend()

    axes[1].set_xlabel("Shannon Entropy (mean over time steps)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Uncertainty (Entropy)", fontweight="bold")
    axes[1].legend()

    fig.suptitle(f"Confidence & Uncertainty Analysis — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig08_confidence_hist.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


# ─── fig09: Statistical significance heatmap ─────────────────────────────────

def plot_stat_significance(stat_df: pd.DataFrame):
    archs = sorted(set(stat_df["model_A"].tolist() + stat_df["model_B"].tolist()))
    n     = len(archs)
    idx   = {a: i for i, a in enumerate(archs)}
    pmat  = np.ones((n, n))

    for _, row in stat_df.iterrows():
        i, j = idx[row["model_A"]], idx[row["model_B"]]
        pmat[i, j] = row["p_value"]
        pmat[j, i] = row["p_value"]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log10(pmat + 1e-10), cmap="RdYlGn_r",
                   vmin=-4, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.04, label="log₁₀(p-value)")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(archs, fontsize=9)
    for i, j in itertools.product(range(n), repeat=2):
        stars = _stars(pmat[i, j])
        ax.text(j, i, stars, ha="center", va="center", fontsize=10,
                color="white" if pmat[i, j] < 0.01 else "black")
    ax.set_title("Paired t-Test p-Values (*** p<0.001, ** p<0.01, * p<0.05)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig09_stat_significance.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─── LaTeX table ─────────────────────────────────────────────────────────────

def save_latex_table(summary_df: pd.DataFrame):
    """
    Save a LaTeX table showing mean ± std for each (arch, aug).
    """
    latex_df = summary_df[[
        "arch", "n_aug", "params", "exact_pm", "cer_pm",
        "epochs_run_mean",
    ]].copy()
    latex_df.columns = [
        "Architecture", "Aug°", "Params",
        "Exact-Acc (mean±std)", "CER (mean±std)", "Epochs",
    ]
    tex = latex_df.to_latex(
        index=False,
        caption=(
            "Ablation study results. "
            "Exact-match accuracy and CER are reported as mean ± std "
            "over \\textit{" + str(3) + "} random seeds."
        ),
        label="tab:ablation",
        escape=False,
    )
    with open(os.path.join(RESULTS_DIR, "ablation_table.tex"), "w",
              encoding="utf-8") as f:
        f.write(tex)
    print("LaTeX table saved → outputs/results/ablation_table.tex")
