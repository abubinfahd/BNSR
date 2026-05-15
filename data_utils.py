"""
data_utils.py — Data loading, splitting, augmentation, encoding, and
difficulty-level assignment for the IJDAR Bangla OCR pipeline.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    CSV_PATH, IMAGE_FOLDER, IMG_W, IMG_H, MAX_LABEL_LENGTH,
    char_to_idx, idx_to_char, NUM_CLASSES,
    DIFFICULTY_COL, DIFFICULTY_THRESHOLDS, LEVEL_MAP,
)


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_img(path: str) -> np.ndarray | None:
    """Load, convert BGR→RGB, and resize a single image."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def load_split(subset_df: pd.DataFrame, desc: str = "Load"):
    """Load all images from a DataFrame split. Returns (imgs, labels)."""
    imgs, labels = [], []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=desc):
        img = load_img(row["image_path"])
        if img is not None:
            imgs.append(img)
            labels.append(row["label"])
    return np.array(imgs, dtype=np.uint8), labels


def normalize(imgs: np.ndarray) -> np.ndarray:
    """Scale uint8 images to [0, 1] float32."""
    return imgs.astype(np.float32) / 255.0


# ─── Data loading & splitting ─────────────────────────────────────────────────

def load_and_split(csv_path: str = CSV_PATH,
                   image_folder: str = IMAGE_FOLDER,
                   seed: int = 42):
    """
    Load the CSV, validate image paths, assign difficulty, then split
    70 / 20 / 10 (train / val / test) — split BEFORE augmentation.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each DataFrame has columns: image_path, label, difficulty
    """
    df = pd.read_csv(csv_path, header=None, names=["image_path", "label"])
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(image_folder, os.path.basename(x))
    )
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Dataset: {len(df):,} valid samples")

    df = assign_difficulty(df)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df,   test_df = train_test_split(temp_df, test_size=1/3, random_state=seed)

    print(
        f"Split  train={len(train_df):,}  "
        f"val={len(val_df):,}  test={len(test_df):,}"
    )
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True)


# ─── Difficulty assignment ────────────────────────────────────────────────────

def assign_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a 'difficulty' column (easy / medium / hard).

    Strategy
    --------
    1. If DIFFICULTY_COL is set and exists in df → use that column directly.
    2. Otherwise → use label length as a proxy (see DIFFICULTY_THRESHOLDS).
    """
    df = df.copy()

    if DIFFICULTY_COL and DIFFICULTY_COL in df.columns:
        df["difficulty"] = df[DIFFICULTY_COL].map(
            lambda v: v.lower().strip() if isinstance(v, str) else "medium"
        )
        # normalise to easy/medium/hard
        mapping = {k: k for k in LEVEL_MAP}
        df["difficulty"] = df["difficulty"].map(mapping).fillna("medium")
    else:
        # proxy: label length
        def _bucket(label: str) -> str:
            n = len(str(label))
            for level, (lo, hi) in DIFFICULTY_THRESHOLDS.items():
                if lo <= n <= hi:
                    return level
            return "hard"
        df["difficulty"] = df["label"].apply(_bucket)

    # Encode as numeric for groupby operations
    df["difficulty_num"] = df["difficulty"].map(LEVEL_MAP)
    return df


# ─── Rotation augmentation ───────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_rotation(imgs: np.ndarray,
                     lbls: list,
                     n_aug: int,
                     seed: int = 42):
    """
    For each image, add n_aug rotated copies sampled from Uniform(-10°, 10°).
    Returns (augmented_imgs, augmented_labels) with originals prepended.
    """
    aug_imgs, aug_lbls = list(imgs), list(lbls)
    rng = np.random.default_rng(seed)
    for img, lbl in zip(imgs, lbls):
        angles = rng.uniform(-10, 10, size=n_aug)
        for a in angles:
            aug_imgs.append(rotate_image(img, a))
            aug_lbls.append(lbl)
    return np.array(aug_imgs, dtype=np.uint8), aug_lbls


# ─── Label encoding ──────────────────────────────────────────────────────────

def encode_labels(label_list: list):
    """
    Encode string labels to padded integer sequences.

    Returns
    -------
    padded : np.ndarray  shape (N, MAX_LABEL_LENGTH)  dtype int32   (-1 = pad)
    lengths : np.ndarray shape (N,)                   dtype int32
    """
    indices = [
        [char_to_idx[c] for c in lbl if c in char_to_idx]
        for lbl in label_list
    ]
    lengths = [len(idx) for idx in indices]
    padded  = pad_sequences(
        indices, maxlen=MAX_LABEL_LENGTH, padding="post", value=-1
    )
    return padded.astype(np.int32), np.array(lengths, dtype=np.int32)


def labels_to_strs(Y: np.ndarray) -> list:
    """Convert padded int matrix back to string labels (skip -1 padding)."""
    return ["".join(idx_to_char[i] for i in row if i >= 0) for row in Y]
