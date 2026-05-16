"""
error_analysis.py — Character-level confusion matrix, group-level error
analysis, and visually similar character confusion for the IJDAR pipeline.

Outputs
-------
outputs/results/error_analysis_top30.csv
outputs/results/grouped_confusion.csv
outputs/results/per_sample_errors.csv
outputs/figures/fig_confusion_matrix.pdf
outputs/figures/fig_worst_chars.pdf
outputs/figures/fig_error_dist.pdf
outputs/figures/fig_grouped_errors.pdf
"""

import os
import itertools
import numpy as np
import pandas as pd
import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config import (
    NUM_CLASSES, MAX_LABEL_LENGTH, RESULTS_DIR, FIGURES_DIR,
    char_to_idx, idx_to_char, CHAR_LIST, SIMILAR_GROUPS,
    ENGLISH_DIGITS, BANGLA_DIGITS, BANGLA_LETTERS,
    ensure_dirs,
)
ensure_dirs()  # guarantee all output folders exist



# ─── Helpers ──────────────────────────────────────────────────────────────────

def str_to_padded(s: str, L: int = MAX_LABEL_LENGTH) -> list:
    idxs = [char_to_idx[c] for c in s if c in char_to_idx][:L]
    idxs += [-1] * (L - len(idxs))
    return idxs


def build_char_arrays(gt_s: list, pred_s: list):
    """Build flat character-level true/pred arrays with valid mask."""
    yt = np.array([str_to_padded(s) for s in gt_s])
    yp = np.array([str_to_padded(s) for s in pred_s])
    flat_t = yt.flatten()
    flat_p = yp.flatten()
    mask   = flat_t >= 0
    yt_m   = flat_t[mask]
    yp_m   = np.clip(flat_p[mask], 0, NUM_CLASSES - 1)
    return yt_m, yp_m


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_error_analysis(gt_s: list, pred_s: list, prefix: str = "best"):
    """
    Full error analysis suite.

    Parameters
    ----------
    gt_s   : list of ground-truth strings
    pred_s : list of predicted strings
    prefix : file-name prefix (e.g. 'best', 'attention')
    """
    yt_m, yp_m = build_char_arrays(gt_s, pred_s)
    present     = sorted(set(yt_m.tolist()))
    labels_cm   = [idx_to_char[i] for i in present]

    cm      = confusion_matrix(yt_m, yp_m, labels=present)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum * 100, 0.0)

    # ── (1) Full confusion matrix figure ──────────────────────────────────────
    _plot_confusion_matrix(cm_norm, labels_cm, prefix)

    # ── (2) Per-class accuracy & worst chars ──────────────────────────────────
    per_class_acc = np.where(
        cm.sum(axis=1) > 0,
        cm.diagonal() / cm.sum(axis=1) * 100,
        100.0,
    )
    worst_idx   = np.argsort(per_class_acc)[:10]
    worst_chars = [(labels_cm[i], per_class_acc[i]) for i in worst_idx]
    _plot_worst_chars(worst_chars, prefix)

    # ── (3) Most confused pairs CSV ───────────────────────────────────────────
    confused_pairs = []
    for i in range(len(present)):
        for j in range(len(present)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append(
                    (labels_cm[i], labels_cm[j], cm[i, j], cm_norm[i, j])
                )
    confused_pairs.sort(key=lambda x: -x[2])
    error_df = pd.DataFrame(
        confused_pairs[:30],
        columns=["true", "pred", "count", "pct_of_true"],
    )
    error_df.to_csv(
        os.path.join(RESULTS_DIR, f"error_analysis_top30_{prefix}.csv"),
        index=False,
    )

    # ── (4) Per-sample errors CSV ─────────────────────────────────────────────
    err_rows = [
        {
            "gt": g, "pred": p,
            "cer": jiwer.cer([g], [p]) * 100,
            "len": len(g),
            "correct": g == p,
        }
        for g, p in zip(gt_s, pred_s)
    ]
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(
        os.path.join(RESULTS_DIR, f"per_sample_errors_{prefix}.csv"),
        index=False,
    )
    _plot_error_distributions(err_df, prefix)

    # ── (5) Group-level error analysis ────────────────────────────────────────
    group_df = _group_level_analysis(cm, present, labels_cm, prefix)

    # ── (6) Visually similar character confusion ──────────────────────────────
    _similar_char_analysis(cm, present, labels_cm, prefix)

    print(f"  Error analysis saved  (prefix={prefix})")
    return err_df, group_df


# ─── Group-level analysis ─────────────────────────────────────────────────────

def _group_level_analysis(cm, present, labels_cm, prefix):
    """Compute confusion within and between character groups."""
    groups = {
        "English digits":  set(ENGLISH_DIGITS),
        "Bangla digits":   set(BANGLA_DIGITS),
        "Bangla letters":  set(BANGLA_LETTERS),
    }

    # Map each present char → its group
    char_group = {}
    for char in labels_cm:
        for gname, gset in groups.items():
            if char in gset:
                char_group[char] = gname
                break
        else:
            char_group[char] = "Other"

    group_names = list(groups.keys()) + ["Other"]
    G = len(group_names)
    group_cm = np.zeros((G, G), dtype=int)
    gidx = {g: i for i, g in enumerate(group_names)}

    for i, true_char in enumerate(labels_cm):
        for j, pred_char in enumerate(labels_cm):
            if cm[i, j] == 0:
                continue
            gi = gidx.get(char_group.get(true_char, "Other"), G - 1)
            gj = gidx.get(char_group.get(pred_char, "Other"), G - 1)
            group_cm[gi, gj] += cm[i, j]

    group_df = pd.DataFrame(group_cm, index=group_names, columns=group_names)
    group_df.to_csv(
        os.path.join(RESULTS_DIR, f"grouped_confusion_{prefix}.csv")
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(group_cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_xticks(range(G)); ax.set_yticks(range(G))
    ax.set_xticklabels(group_names, rotation=30, ha="right")
    ax.set_yticklabels(group_names)
    for i, j in itertools.product(range(G), range(G)):
        ax.text(j, i, str(group_cm[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if group_cm[i, j] > group_cm.max() / 2 else "black")
    ax.set_title(f"Group-Level Confusion Matrix ({prefix})", fontweight="bold")
    ax.set_xlabel("Predicted Group"); ax.set_ylabel("True Group")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"fig_grouped_errors_{prefix}.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()
    return group_df


def _similar_char_analysis(cm, present, labels_cm, prefix):
    """Report confusion between visually similar character pairs."""
    rows = []
    char_to_pos = {c: i for i, c in enumerate(labels_cm)}
    for (c1, c2) in SIMILAR_GROUPS:
        if c1 not in char_to_pos or c2 not in char_to_pos:
            continue
        i, j = char_to_pos[c1], char_to_pos[c2]
        rows.append({
            "pair":    f"{c1}↔{c2}",
            "c1_pred_as_c2": int(cm[i, j]),
            "c2_pred_as_c1": int(cm[j, i]),
            "total_confusion": int(cm[i, j] + cm[j, i]),
        })
    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(
        os.path.join(RESULTS_DIR, f"similar_char_confusion_{prefix}.csv"),
        index=False,
    )


# ─── Figures ──────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm_norm, labels_cm, prefix):
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.03, label="% of True Class")
    n = len(labels_cm)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels_cm, rotation=90, fontsize=7)
    ax.set_yticklabels(labels_cm, fontsize=7)
    thresh = cm_norm.max() / 2
    for i, j in itertools.product(range(n), repeat=2):
        v = cm_norm[i, j]
        if v > 1:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if v > thresh else "black")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Character-Level Confusion Matrix — {prefix} (%)",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"fig_confusion_matrix_{prefix}.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


def _plot_worst_chars(worst_chars, prefix):
    fig, ax = plt.subplots(figsize=(9, 5))
    wc_chars = [x[0] for x in worst_chars]
    wc_vals  = [x[1] for x in worst_chars]
    ax.barh(wc_chars, wc_vals, color="#EF5350", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Per-Class Accuracy (%)", fontsize=12)
    ax.set_title(
        f"10 Most Confused Characters — {prefix}",
        fontweight="bold",
    )
    ax.axvline(50, linestyle="--", color="gray", alpha=0.5, label="50% line")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    for i, (v, c) in enumerate(zip(wc_vals, wc_chars)):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"fig_worst_chars_{prefix}.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()


def _plot_error_distributions(err_df, prefix):
    cer_mean = err_df["cer"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(err_df["cer"], bins=30, color="#42A5F5",
                 edgecolor="black", linewidth=0.4)
    axes[0].axvline(cer_mean, color="red", linestyle="--",
                    label=f"Mean CER={cer_mean:.1f}%")
    axes[0].set_xlabel("CER (%)"); axes[0].set_ylabel("Sample Count")
    axes[0].set_title("CER Distribution", fontweight="bold")
    axes[0].legend()

    acc_by_len = err_df.groupby("len")["correct"].mean() * 100
    axes[1].bar(acc_by_len.index, acc_by_len.values,
                color="#66BB6A", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Label Length"); axes[1].set_ylabel("Exact-Match Acc (%)")
    axes[1].set_title("Accuracy by Label Length", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Error Analysis — {prefix}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, f"fig_error_dist_{prefix}.pdf"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()
