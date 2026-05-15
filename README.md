# Bangla Bank Serial-Number OCR — IJDAR Submission Pipeline

> **"A controlled comparative study of OCR architectures under structured augmentation
> and difficulty-aware evaluation for Bangla bank document recognition."**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture Zoo](#architecture-zoo)
4. [Key Contributions](#key-contributions)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Outputs](#outputs)
8. [Configuration](#configuration)
9. [Reproducibility](#reproducibility)
10. [Citation](#citation)

---

## Overview

This repository implements a **fully modular**, IJDAR-ready OCR pipeline for
recognising serial numbers on Bangla bank documents. It goes well beyond a
single-model baseline by providing:

| Feature | Detail |
|---|---|
| **7 architectures** | CNN-CTC, CNN-GRU, CRNN-small/base/large, Attention-OCR, Transformer-OCR |
| **Multi-seed training** | 3 seeds → mean ± std reported |
| **Statistical rigour** | Paired t-tests across model pairs & augmentation conditions |
| **Difficulty-aware eval** | Easy / Medium / Hard split on every metric |
| **Confidence analysis** | Max-softmax probability + Shannon entropy |
| **Group error analysis** | Digit vs digit, Bangla letter groups, visually similar chars |
| **Computational analysis** | FLOPs, latency (ms/image), model size (MB) |
| **Overlay learning curves** | Cross-architecture comparison on same axes |
| **10+ publication figures** | All PDF, 300 DPI, ready for submission |

---

## Project Structure

```
BNSR/
│
├── config.py                   # ★ ALL hyperparameters, paths, seeds, vocab
├── data_utils.py               # Load, split, augment, encode, difficulty assign
│
├── models/
│   ├── __init__.py             # build_model(arch) factory
│   ├── _ctc_utils.py           # Shared CTC loss + greedy decoder
│   ├── cnn_ctc.py              # (A) CNN → Dense → CTC  (no RNN)
│   ├── cnn_gru.py              # (B) CNN + Bidirectional GRU
│   ├── crnn.py                 # (C) CRNN small / base / large
│   ├── attention_ocr.py        # (D) CNN + BiLSTM + Bahdanau Attention + CTC
│   └── transformer_ocr.py      # (E) Patch Embed + 2-layer Transformer + CTC
│
├── train.py                    # Multi-seed training loop
├── evaluate.py                 # Overall, per-difficulty, confidence analysis
├── statistical_tests.py        # Paired t-tests (model pairs + aug effects)
├── error_analysis.py           # Confusion matrix, group errors, similar chars
├── computational_analysis.py   # FLOPs, latency, model size
├── plots.py                    # All 10 publication figures
│
├── run_all.py                  # ★ MASTER RUNNER — replaces main.py
├── main.py                     # Legacy monolithic script (kept for reference)
│
└── outputs/
    ├── figures/                # PDF plots
    ├── models/                 # Keras checkpoints
    └── results/                # CSV, JSON, LaTeX tables
```

---

## Architecture Zoo

### (A) CNN-CTC — Weakest Baseline
```
Input (64×128×3)
  └─ 3× [Conv2D → BN → Conv2D → MaxPool2D → Dropout]
  └─ Collapse H (mean)
  └─ TimeDistributed Dense(256, relu)
  └─ TimeDistributed Dense(C+1, softmax)   ← CTC output
```
**Purpose:** Isolates the contribution of recurrent layers.

---

### (B) CNN-GRU — LSTM vs GRU Comparison
```
Input → CNN backbone (same as above)
  └─ Collapse H
  └─ Bidirectional(GRU(128))
  └─ Bidirectional(GRU(64))
  └─ TimeDistributed Dense(C+1, softmax)
```
**Purpose:** Controlled comparison — only the recurrent cell changes.

---

### (C) CRNN — small / base / large
| Variant | Recurrent Stack | Parameters |
|---------|----------------|------------|
| small   | BiLSTM(128)    | ~1.2M |
| base    | BiLSTM(128) → BiLSTM(64) | ~1.8M |
| large   | BiLSTM(256) → BiLSTM(128) → BiLSTM(64) | ~4.1M |

---

### (D) Attention OCR ⭐ (IJDAR Requirement)
```
Input → CNN backbone
  └─ Collapse H
  └─ Bidirectional(LSTM(128))             ← encoder
  └─ Bahdanau Self-Attention (units=128)  ← attention over encoder states
  └─ Residual Add + LayerNorm
  └─ Bidirectional(LSTM(64))              ← refine
  └─ TimeDistributed Dense(C+1, softmax)
```
Bahdanau additive attention:  
`score(q, k) = V · tanh(W_q · q + W_k · k)`  
`context_t = Σ softmax(score) · encoder_states`

---

### (E) Transformer OCR ⭐ (IJDAR Requirement)
```
Input → Conv2D(stride=2) → Conv2D(stride=2)   ← patch embedding
  └─ Collapse H → (B, W/4, d_model) tokens
  └─ Sinusoidal positional encoding
  └─ 2× Transformer Encoder Block:
       MultiHeadAttention → Add&Norm → FFN → Add&Norm
  └─ TimeDistributed Dense(C+1, softmax)
```
Lightweight (d_model=128, 4 heads, 2 layers) — trainable on Kaggle P100.

---

## Key Contributions

### 1. Multi-Seed Statistical Rigour
```python
SEEDS = [42, 123, 999]
# → outputs/results/multi_seed_summary.csv
# Columns: arch, n_aug, exact_mean, exact_std, cer_mean, cer_std …
# Display: "92.3 ± 0.4"
```

### 2. Paired t-Test Significance
```python
from scipy.stats import ttest_rel
ttest_rel(scores_model_A, scores_model_B)
# → outputs/results/statistical_tests.csv
# Columns: model_A, model_B, t_stat, p_value, significant, stars
```
Stars: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `ns` not significant

### 3. Difficulty-Aware Evaluation
```python
DIFFICULTY_THRESHOLDS = {
    "easy":   (1, 5),    # label length ≤ 5
    "medium": (6, 7),
    "hard":   (8, 9),
}
# → outputs/results/difficulty_eval_best.csv
# Columns: difficulty, n_samples, exact, cer, wer
```
> This becomes a key novelty claim:  
> *"Robustness analysis across document difficulty levels"*

### 4. Confidence & Uncertainty Analysis
```python
max_softmax_prob  → high = confident prediction
shannon_entropy   → high = uncertain prediction
# → outputs/results/confidence_analysis.csv
```

### 5. Group-Level Error Analysis
```
English digits  → English digits  (intra-group confusion)
Bangla digits   → Bangla digits
Bangla digits   ↔ Bangla letters  (inter-group confusion)
```
Visually similar pair analysis: `৫↔৬`, `০↔৮`, `0↔8`, `5↔6`, `1↔7`

---

## Installation

```bash
pip install tensorflow opencv-python jiwer scikit-learn scipy \
            pandas numpy matplotlib tqdm
```

> **Kaggle / Colab**: all packages are pre-installed except `jiwer`.  
> Add `!pip install jiwer` to your first cell.

---

## Usage

### Full pipeline (all 7 archs × 4 aug levels × 3 seeds):
```bash
python run_all.py
```
⚠️ This runs 84 training configurations. On a Kaggle P100 GPU expect ~8–12 hours.

### Quick smoke-test (validates the whole pipeline in ~10 minutes):
```bash
python run_all.py --smoke-test
```
Runs: 3 architectures × 2 aug levels × 2 seeds × 2 epochs.

### ✅ Resume after Kaggle session timeout:
```bash
# Session 1 ends (GPU timeout at 9h) — some runs completed, some not.
# Start a new session and just run the same command again:
python run_all.py
# The pipeline automatically reads checkpoint.json and skips completed runs.
```
Or explicitly:
```bash
python run_all.py --resume
```

### 🔄 Start completely fresh (delete checkpoint):
```bash
python run_all.py --reset
```

### Custom subset:
```bash
# Only attention and transformer, best aug, 2 seeds
python run_all.py \
    --archs attention transformer \
    --aug-degrees 0 5 \
    --seeds 42 123
```

### Legacy single-file run:
```bash
python main.py
```

---

## Outputs

After a full run, the `outputs/` directory contains:

### `outputs/figures/` — Publication-ready PDFs

| File | Description |
|------|-------------|
| `fig01_dataset_split.pdf` | Pie chart of 70/20/10 split |
| `fig02_ablation_heatmap.pdf` | Accuracy & CER heatmap (arch × aug) |
| `fig03_aug_effect_line.pdf` | Augmentation effect per arch (error bars) |
| `fig04_params_vs_acc.pdf` | Bubble chart: complexity vs accuracy |
| `fig05_final_metrics.pdf` | Bar chart: best model final metrics |
| `fig06_overlay_learning_curves.pdf` | Cross-arch training curve overlay |
| `fig07_difficulty_cer.pdf` | CER by Easy / Medium / Hard level |
| `fig08_confidence_hist.pdf` | Entropy distribution: correct vs incorrect |
| `fig09_stat_significance.pdf` | p-value heatmap (paired t-tests) |
| `fig10a/b_*_samples.pdf` | Sample correct / incorrect predictions |
| `fig_confusion_matrix_*.pdf` | Character-level confusion matrix |
| `fig_worst_chars_*.pdf` | 10 hardest characters |
| `fig_grouped_errors_*.pdf` | Group-level confusion |
| `fig_compute_tradeoff.pdf` | Latency vs params bubble chart |

### `outputs/results/` — CSV / JSON / LaTeX

| File | Description |
|------|-------------|
| `raw_seed_results.csv` | One row per (arch, aug, seed) |
| `multi_seed_summary.csv` | Mean ± std over seeds |
| `statistical_tests.csv` | Pairwise t-test results |
| `aug_effect_tests.csv` | Aug-condition t-test results |
| `difficulty_eval_best.csv` | Per-difficulty metrics (best model) |
| `confidence_analysis.csv` | Per-sample max_prob & entropy |
| `computational_analysis.csv` | FLOPs, latency, size |
| `error_analysis_top30_*.csv` | Top 30 confused character pairs |
| `grouped_confusion_*.csv` | Group-level confusion |
| `similar_char_confusion_*.csv` | Visually similar pair errors |
| `per_sample_errors_*.csv` | Per-sample CER |
| `best_model_metrics.json` | Final exact / CER / WER / F1 |
| `ablation_table.tex` | LaTeX table for paper |

---

## Configuration

All settings live in `config.py`. Key parameters:

```python
# Paths
CSV_PATH     = "/kaggle/input/.../labels.csv"
IMAGE_FOLDER = "/kaggle/input/.../cropped_serial_numbers_V3"

# Experiment
SEEDS        = [42, 123, 999]          # multi-seed
AUG_DEGREES  = [0, 1, 3, 5]           # rotations per image
ARCHITECTURES = [ ... ]                # 7 models

# Difficulty proxy (if no CSV column available)
DIFFICULTY_THRESHOLDS = {
    "easy":   (1, 5),
    "medium": (6, 7),
    "hard":   (8, 9),
}
# If your CSV has an explicit difficulty column:
DIFFICULTY_COL = "level"   # set to your column name

# Visually similar character pairs for error analysis
SIMILAR_GROUPS = [("০", "৮"), ("৫", "৬"), ("0", "8"), ...]
```

---

## Reproducibility

| Component | Control |
|-----------|---------|
| Data split | `random_state=seed` in `train_test_split` |
| Augmentation | `np.random.default_rng(seed)` |
| TF training | `tf.random.set_seed(seed)` + `np.random.seed(seed)` |
| Results | `raw_seed_results.csv` stores per-seed scores |
| Models | Checkpointed as `{arch}_aug{n}_seed{s}.keras` |

To reproduce any single run:
```python
from train import train_one_run
result = train_one_run(
    arch="crnn_base", n_aug=3, seed=42,
    train_imgs_raw=..., train_lbls=...,
    val_X=..., val_Y=..., test_X=..., test_Y=...,
)
```

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{YourName2026bangla,
  title   = {A Comparative Study of OCR Architectures for Bangla Bank
             Document Recognition with Difficulty-Aware Evaluation},
  author  = {Your Name},
  journal = {International Journal on Document Analysis and Recognition (IJDAR)},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
