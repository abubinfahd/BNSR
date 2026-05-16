"""
run_all.py — Master runner for the IJDAR Bangla Bank OCR pipeline.

Usage
-----
# Full run (all architectures × aug degrees × 3 seeds):
    python run_all.py

# Resume after a Kaggle session timeout (skips completed runs automatically):
    python run_all.py --resume

# Fast smoke-test (2 seeds, 2 epochs, 2 aug levels, 3 architectures):
    python run_all.py --smoke-test

# Start completely fresh (deletes checkpoint):
    python run_all.py --reset

Pipeline steps
--------------
1.  Load data, split 70/20/10, assign difficulty metadata
2.  Multi-seed training loop  (checkpoint-aware, resumes from last run)
3.  Overall evaluation + per-difficulty evaluation
4.  Statistical testing (paired t-tests)
5.  Error analysis (confusion, group errors, similar chars)
6.  Confidence / uncertainty analysis
7.  Computational analysis (FLOPs, latency, MB)
8.  All publication figures
9.  LaTeX table

Novelty framing (IJDAR)
-----------------------
"We present a controlled comparative study of OCR architectures under
structured augmentation and difficulty-aware evaluation for Bangla bank
document recognition."
"""

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="IJDAR Bangla OCR Pipeline")
parser.add_argument(
    "--smoke-test", action="store_true",
    help="Quick validation: 2 seeds, 2 epochs, 2 aug levels, 3 architectures",
)
parser.add_argument(
    "--resume", action="store_true", default=True,
    help="Resume from checkpoint (default: True — always resumes if checkpoint exists)",
)
parser.add_argument(
    "--reset", action="store_true",
    help="Delete checkpoint and start completely fresh",
)
parser.add_argument(
    "--archs", nargs="+", default=None,
    help="Subset of architectures to run (e.g. --archs crnn_base attention)",
)
parser.add_argument(
    "--seeds", nargs="+", type=int, default=None,
    help="Override seeds (e.g. --seeds 42 123)",
)
parser.add_argument(
    "--aug-degrees", nargs="+", type=int, default=None,
    help="Override augmentation degrees (e.g. --aug-degrees 0 3)",
)
args = parser.parse_args()

# ─── Config overrides for smoke-test ─────────────────────────────────────────
from checkpoint import load_checkpoint, print_remaining, reset_checkpoint
from config import (
    SEEDS, ARCHITECTURES, AUG_DEGREES, RESULTS_DIR, FIGURES_DIR,
    INPUT_SHAPE, CHECKPOINTS_DIR, LOGS_DIR, MODELS_DIR,
    ensure_dirs,
)
ensure_dirs()  # create all output folders before anything else


seeds       = args.seeds       or ([42, 123]       if args.smoke_test else SEEDS)
archs       = args.archs       or (["cnn_ctc", "crnn_base", "attention"]
                                   if args.smoke_test else ARCHITECTURES)
aug_degrees = args.aug_degrees or ([0, 3]          if args.smoke_test else AUG_DEGREES)

# ── Handle --reset ────────────────────────────────────────────────────────────
if args.reset:
    reset_checkpoint()

resume = not args.reset   # reset=True → resume=False

print("=" * 65)
print("  IJDAR Bangla OCR — Modular Pipeline")
print(f"  TF {tf.__version__}  |  GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"  Architectures : {archs}")
print(f"  Aug degrees   : {aug_degrees}")
print(f"  Seeds         : {seeds}")
print(f"  Smoke-test    : {args.smoke_test}")
print(f"  Resume mode   : {resume}")
print("=" * 65)

# ── Show checkpoint status before loading data ─────────────────────────────
if resume:
    _ckpt_state = load_checkpoint()
    print_remaining(_ckpt_state, archs, aug_degrees, seeds)
    _done_count = len(_ckpt_state.get("completed", []))
    _total      = len(archs) * len(aug_degrees) * len(seeds)
    if _done_count >= _total:
        print("\n  ✅ All training runs already completed in a previous session!")
        print("  Skipping training — running analysis only …\n")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Data loading & splitting
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading and splitting dataset …")

from data_utils import load_and_split, load_split, normalize, encode_labels

train_df, val_df, test_df = load_and_split(seed=seeds[0])

print("Loading images …")
train_imgs_raw, train_lbls = load_split(train_df, "Train")
val_imgs_raw,   val_lbls   = load_split(val_df,   "Val  ")
test_imgs_raw,  test_lbls  = load_split(test_df,  "Test ")

val_X  = normalize(val_imgs_raw)
test_X = normalize(test_imgs_raw)

val_Y,  _ = encode_labels(val_lbls)
test_Y, _ = encode_labels(test_lbls)

# ── Dataset split pie ─────────────────────────────────────────────────────────
from plots import plot_dataset_split
plot_dataset_split(len(train_df), len(val_df), len(test_df))
print("  fig01_dataset_split.pdf saved")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Multi-seed training
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/9] Multi-seed training …")

from train import run_multi_seed_training

raw_records, summary_df, best_run = run_multi_seed_training(
    train_imgs_raw, train_lbls,
    val_X, val_Y, test_X, test_Y,
    architectures = archs,
    aug_degrees   = aug_degrees,
    seeds         = seeds,
    smoke_test    = args.smoke_test,
    resume        = resume,
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Evaluation  (overall + per-difficulty)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/9] Evaluation (overall + per-difficulty) …")

from evaluate import evaluate_overall, evaluate_by_difficulty, confidence_analysis
from sklearn.metrics import precision_score, recall_score, f1_score
from error_analysis import build_char_arrays
from config import NUM_CLASSES
import numpy as np

best_model = best_run["model"]
best_arch  = best_run["arch"]

res = evaluate_overall(best_model, test_X, test_Y)
gt_s, pred_s = res["gt"], res["pred"]

# Character-level precision / recall / F1
yt_m, yp_m = build_char_arrays(gt_s, pred_s)
kw   = dict(average="macro", zero_division=0)
prec = precision_score(yt_m, yp_m, **kw) * 100
rec  = recall_score   (yt_m, yp_m, **kw) * 100
f1   = f1_score       (yt_m, yp_m, **kw) * 100

best_metrics = dict(
    exact     = round(res["exact"], 2),
    cer       = round(res["cer"],   2),
    wer       = round(res["wer"],   2),
    precision = round(prec, 2),
    recall    = round(rec,  2),
    f1        = round(f1,   2),
)
with open(os.path.join(RESULTS_DIR, "best_model_metrics.json"), "w") as f:
    json.dump(best_metrics, f, indent=2)

print(f"\nBest model ({best_arch}) metrics:")
for k, v in best_metrics.items():
    print(f"  {k:12s}: {v}")

# Per-difficulty evaluation for each architecture
difficulty_results = {}
for run in raw_records:
    arch = run["arch"]
    if arch in difficulty_results:
        continue
    # Re-train is too expensive; use the best-seed model if stored.
    # We use the best_run model for the best arch and skip others for now
    # (full version: store all models in run dicts)

# Full per-difficulty for best model only (others need stored models)
diff_df = evaluate_by_difficulty(best_model, test_df, test_X, test_Y)
diff_df.to_csv(os.path.join(RESULTS_DIR, "difficulty_eval_best.csv"), index=False)
print("\nDifficulty evaluation (best model):")
print(diff_df.to_string(index=False))
difficulty_results[best_arch] = diff_df

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Statistical testing
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/9] Statistical tests (paired t-test) …")

from statistical_tests import run_pairwise_tests, run_augmentation_effect_tests

if len(seeds) >= 2:
    stat_df   = run_pairwise_tests(raw_records)
    aug_test_df = run_augmentation_effect_tests(raw_records)
else:
    print("  Skipping t-tests: need ≥ 2 seeds")
    stat_df = pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Error analysis (best model)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/9] Error analysis …")

from error_analysis import run_error_analysis

err_df, group_df = run_error_analysis(gt_s, pred_s, prefix=best_arch)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Confidence / uncertainty analysis
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6/9] Confidence & uncertainty analysis …")

conf_df = confidence_analysis(best_model, test_X, gt_s, pred_s)
conf_df.to_csv(os.path.join(RESULTS_DIR, "confidence_analysis.csv"), index=False)

print(f"  Correct   — mean max_prob={conf_df[conf_df['correct']]['max_prob'].mean():.3f}  "
      f"entropy={conf_df[conf_df['correct']]['entropy'].mean():.3f}")
print(f"  Incorrect — mean max_prob={conf_df[~conf_df['correct']]['max_prob'].mean():.3f}  "
      f"entropy={conf_df[~conf_df['correct']]['entropy'].mean():.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: Computational analysis
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7/9] Computational analysis (FLOPs, latency, size) …")

from computational_analysis import run_computational_analysis

# Build list of unique-arch run dicts with model objects from best_run
comp_runs = [best_run]   # extend if you store other arch models
comp_df = run_computational_analysis(comp_runs, test_X)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8: All publication figures
# ──────────────────────────────────────────────────────────────────────────────
print("\n[8/9] Generating all figures …")

from plots import (
    plot_ablation_heatmap, plot_aug_effect_line, plot_params_vs_acc,
    plot_final_metrics, plot_overlay_learning_curves,
    plot_difficulty_cer, plot_confidence_histogram, plot_stat_significance,
    save_latex_table,
)

plot_ablation_heatmap(summary_df)
plot_aug_effect_line(summary_df)
plot_params_vs_acc(summary_df)
plot_final_metrics(best_metrics, model_name=best_arch)

# Overlay learning curves — pick representative runs
overlay_archs  = ["cnn_ctc", "crnn_base", "attention"]
overlay_hist   = {}
for rec in raw_records:
    if rec["arch"] in overlay_archs and rec["arch"] not in overlay_hist:
        overlay_hist[rec["arch"]] = rec.get("history",
                                             {"loss": [], "val_loss": [],
                                              "val_exact": []})
if overlay_hist:
    plot_overlay_learning_curves(overlay_hist)

plot_difficulty_cer(difficulty_results)
plot_confidence_histogram(conf_df, model_name=best_arch)

if not stat_df.empty:
    plot_stat_significance(stat_df)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9: LaTeX table
# ──────────────────────────────────────────────────────────────────────────────
print("\n[9/9] Saving LaTeX ablation table …")
save_latex_table(summary_df)

# ── Sample predictions (correct + incorrect) ──────────────────────────────────
correct_idx   = [i for i, (g, p) in enumerate(zip(gt_s, pred_s)) if g == p][:10]
incorrect_idx = [i for i, (g, p) in enumerate(zip(gt_s, pred_s)) if g != p][:10]

import matplotlib.pyplot as plt

def _plot_samples(idxs, title, fname, imgs, gt, pred):
    n = len(idxs)
    if n == 0:
        return
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for k, i in enumerate(idxs):
        axes[k].imshow(imgs[i])
        axes[k].set_title(
            f"GT:  {gt[i]}\nPR: {pred[i]}",
            fontsize=8,
            color="green" if gt[i] == pred[i] else "red",
        )
        axes[k].axis("off")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close()

_plot_samples(correct_idx,   "Correct Predictions",
              os.path.join(FIGURES_DIR, "fig10a_correct_samples.pdf"),
              test_imgs_raw, gt_s, pred_s)
_plot_samples(incorrect_idx, "Incorrect Predictions",
              os.path.join(FIGURES_DIR, "fig10b_incorrect_samples.pdf"),
              test_imgs_raw, gt_s, pred_s)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("  ALL OUTPUTS SAVED TO  ./outputs/")
print("  figures/      — PDF plots ready for IJDAR submission")
print("  models/       — Final .keras models (one per arch×aug×seed)")
print("  checkpoints/  — Keras best-weight checkpoints (mid-training)")
print("  results/      — CSV tables, JSON metrics, LaTeX table")
print("  logs/         — TensorBoard / training logs")
print("═" * 65)
print(f"\n  Best model : {best_arch}")
print(f"  Exact-Match: {best_metrics['exact']:.2f}%")
print(f"  CER        : {best_metrics['cer']:.2f}%")
print(f"  WER        : {best_metrics['wer']:.2f}%")
print("═" * 65)
