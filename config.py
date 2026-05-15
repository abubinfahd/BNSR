"""
config.py — Central configuration for the IJDAR Bangla OCR pipeline.
All hyperparameters, paths, seeds, and vocabulary live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH     = "/kaggle/input/datasets/abubinfahd/bank-serial-number-images/labels.csv"
IMAGE_FOLDER = "/kaggle/input/datasets/abubinfahd/bank-serial-number-images/cropped_serial_numbers_V3"

OUTPUT_DIR  = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Image / Sequence ─────────────────────────────────────────────────────────
IMG_W, IMG_H     = 128, 64
INPUT_SHAPE      = (IMG_H, IMG_W, 3)
MAX_LABEL_LENGTH = 9
CNN_SEQ_LEN      = IMG_W // 8   # 16 time-steps after 3× MaxPool2D(2,2)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 60

# ─── Multi-seed experiment ────────────────────────────────────────────────────
SEEDS = [42, 123, 999]

# ─── Ablation grid ────────────────────────────────────────────────────────────
AUG_DEGREES   = [0, 1, 3, 5]          # rotations per image (0 = no aug)
ARCHITECTURES = [
    "cnn_ctc",        # (A) Weakest baseline — CNN only
    "cnn_gru",        # (B) CNN + Bidirectional GRU
    "crnn_small",     # (C) CRNN  1× BiLSTM-128
    "crnn_base",      # (C) CRNN  2× BiLSTM
    "crnn_large",     # (C) CRNN  3× BiLSTM
    "attention",      # (D) CNN + BiLSTM + Bahdanau Attention + CTC
    "transformer",    # (E) Patch Embed + 2-layer Transformer + CTC
]

# ─── Difficulty metadata ──────────────────────────────────────────────────────
# Proxy: label length → difficulty bucket
# Override DIFFICULTY_COL if the CSV has an explicit column for difficulty.
DIFFICULTY_COL   = None       # set to column name, e.g. "level", or leave None
DIFFICULTY_THRESHOLDS = {     # label length boundaries (inclusive)
    "easy":   (1, 5),
    "medium": (6, 7),
    "hard":   (8, MAX_LABEL_LENGTH),
}
LEVEL_MAP = {"easy": 0, "medium": 1, "hard": 2}

# ─── Character vocabulary ─────────────────────────────────────────────────────
ENGLISH_DIGITS = "0123456789"
BANGLA_DIGITS  = "০১২৩৪৫৬৭৮৯"
BANGLA_LETTERS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
CHAR_LIST      = ENGLISH_DIGITS + BANGLA_DIGITS + BANGLA_LETTERS
char_to_idx    = {c: i for i, c in enumerate(CHAR_LIST)}
idx_to_char    = {i: c for c, i in char_to_idx.items()}
NUM_CLASSES    = len(CHAR_LIST)
CTC_BLANK      = NUM_CLASSES        # blank index = NUM_CLASSES (last)

# ─── Visually similar Bangla character groups (for error analysis) ────────────
SIMILAR_GROUPS = [
    ("০", "৮"),   # Bangla 0 ↔ 8
    ("৫", "৬"),   # Bangla 5 ↔ 6
    ("৩", "৮"),   # Bangla 3 ↔ 8
    ("0",  "8"),   # English 0 ↔ 8
    ("5",  "6"),   # English 5 ↔ 6
    ("1",  "7"),   # English 1 ↔ 7
]
