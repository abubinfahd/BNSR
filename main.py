"""
Bangla Bank Serial Number OCR — IJDAR Submission Pipeline
==========================================================
Pipeline:
  1. Load & validate dataset
  2. Train/Val/Test split  (70 / 20 / 10)  — BEFORE augmentation
  3. Rotation-only augmentation on TRAIN split only
  4. CTC-CRNN model (CNN-BiLSTM)
  5. Ablation study  (aug_degree × architecture)
  6. Error analysis  (CER, WER, confusion matrix, worst chars)
  7. All figures / model saved to  ./outputs/
"""

# ─────────────────────────── 0. Imports ──────────────────────────────
import os, warnings, itertools, json, time
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, Dense, Dropout,
    Bidirectional, TimeDistributed, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences

import jiwer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TF {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU'))>0}")

# ─────────────────────────── 1. Config ───────────────────────────────
CSV_PATH     = "/kaggle/input/datasets/abubinfahd/bank-serial-number-images/labels.csv"
IMAGE_FOLDER = "/kaggle/input/datasets/abubinfahd/bank-serial-number-images/cropped_serial_numbers_V3"

IMG_W, IMG_H     = 128, 64
INPUT_SHAPE      = (IMG_H, IMG_W, 3)
MAX_LABEL_LENGTH = 9
BATCH_SIZE       = 32
EPOCHS           = 60

ENGLISH_DIGITS = "0123456789"
BANGLA_DIGITS  = "০১২৩৪৫৬৭৮৯"
BANGLA_LETTERS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
CHAR_LIST      = ENGLISH_DIGITS + BANGLA_DIGITS + BANGLA_LETTERS
char_to_idx    = {c: i for i, c in enumerate(CHAR_LIST)}
idx_to_char    = {i: c for c, i in char_to_idx.items()}
NUM_CLASSES    = len(CHAR_LIST)
CTC_BLANK      = NUM_CLASSES
CNN_SEQ_LEN    = IMG_W // 8   # 16

print(f"Vocab: {NUM_CLASSES} | CTC blank: {CTC_BLANK}")

# ─────────────────────────── 2. Load CSV ─────────────────────────────
df = pd.read_csv(CSV_PATH, header=None, names=["image_path", "label"])
df["image_path"] = df["image_path"].apply(
    lambda x: os.path.join(IMAGE_FOLDER, os.path.basename(x))
)
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Dataset: {len(df):,} samples")

# ─────────────────────────── 3. Split FIRST ──────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=1/3, random_state=SEED)
# → 70 / 20 / 10

print(f"Split  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# ── Save split pie ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
sizes  = [len(train_df), len(val_df), len(test_df)]
labels = [f"Train\n{len(train_df):,}\n(70%)",
          f"Val\n{len(val_df):,}\n(20%)",
          f"Test\n{len(test_df):,}\n(10%)"]
colors = ['#2196F3', '#FF9800', '#F44336']
wedges, _, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
    startangle=90, wedgeprops={'linewidth':1.5,'edgecolor':'white'}
)
for at in autotexts: at.set_fontweight('bold')
ax.set_title('Dataset Split (before augmentation)', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/fig1_dataset_split.pdf", bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────── 4. Load raw images ──────────────────────
def load_img(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

def load_split(subset_df, desc):
    imgs, labels = [], []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=desc):
        img = load_img(row['image_path'])
        if img is not None:
            imgs.append(img)
            labels.append(row['label'])
    return np.array(imgs, dtype=np.uint8), labels

print("Loading images…")
train_imgs_raw, train_lbls = load_split(train_df, "Train")
val_imgs_raw,   val_lbls   = load_split(val_df,   "Val  ")
test_imgs_raw,  test_lbls  = load_split(test_df,  "Test ")

# Normalize (no other preprocessing)
def normalize(imgs): return imgs.astype(np.float32) / 255.0

val_X  = normalize(val_imgs_raw)
test_X = normalize(test_imgs_raw)

# ─────────────────────────── 5. Rotation augmentation ────────────────
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)

def augment_rotation(imgs, lbls, n_aug):
    """n_aug rotations per image from Uniform(-10,10)°"""
    aug_imgs, aug_lbls = list(imgs), list(lbls)
    rng = np.random.default_rng(SEED)
    for img, lbl in zip(imgs, lbls):
        angles = rng.uniform(-10, 10, size=n_aug)
        for a in angles:
            aug_imgs.append(rotate_image(img, a))
            aug_lbls.append(lbl)
    return np.array(aug_imgs, dtype=np.uint8), aug_lbls

# ─────────────────────────── 6. Label encoding ───────────────────────
def encode_labels(label_list):
    indices = [[char_to_idx[c] for c in lbl if c in char_to_idx]
               for lbl in label_list]
    lengths = [len(idx) for idx in indices]
    padded  = pad_sequences(indices, maxlen=MAX_LABEL_LENGTH,
                            padding='post', value=-1)
    return padded.astype(np.int32), np.array(lengths, dtype=np.int32)

val_Y,  val_Ylen  = encode_labels(val_lbls)
test_Y, test_Ylen = encode_labels(test_lbls)

# ─────────────────────────── 7. CTC helpers ──────────────────────────
def ctc_loss_fn(y_true, y_pred):
    batch   = tf.shape(y_pred)[0]
    seqlen  = tf.shape(y_pred)[1]
    inp_len = tf.fill([batch], seqlen)
    lbl_len = tf.reduce_sum(tf.cast(tf.not_equal(y_true,-1), tf.int32), axis=1)
    labels  = tf.cast(tf.maximum(y_true, 0), tf.int32)
    logits  = tf.math.log(tf.transpose(y_pred,[1,0,2]) + 1e-8)
    loss = tf.nn.ctc_loss(labels=labels, logits=logits,
                          label_length=lbl_len, logit_length=inp_len,
                          logits_time_major=True, blank_index=CTC_BLANK)
    return tf.reduce_mean(loss)

def greedy_decode(preds):
    inp_len = np.full(len(preds), preds.shape[1], dtype=np.int32)
    decoded, _ = K.ctc_decode(preds, input_length=inp_len, greedy=True)
    return decoded[0].numpy()

def seqs_to_strs(seqs, ref_labels=None):
    """Convert int matrix → list of strings."""
    return [''.join(idx_to_char[i] for i in row
                    if 0 <= i < NUM_CLASSES) for row in seqs]

def labels_to_strs(Y):
    return [''.join(idx_to_char[i] for i in row if i >= 0) for row in Y]

def evaluate(model, X, Y, batch=BATCH_SIZE):
    preds   = model.predict(X, batch_size=batch, verbose=0)
    decoded = greedy_decode(preds)
    pred_s  = seqs_to_strs(decoded)
    gt_s    = labels_to_strs(Y)
    exact   = sum(g==p for g,p in zip(gt_s,pred_s)) / len(gt_s) * 100
    cer     = jiwer.cer(gt_s, pred_s) * 100
    wer     = jiwer.wer(gt_s, pred_s) * 100
    return dict(exact=exact, cer=cer, wer=wer, gt=gt_s, pred=pred_s)

class ValAccCallback(tf.keras.callbacks.Callback):
    def __init__(self, vX, vY):
        super().__init__()
        self.vX, self.vY = vX, vY
        self.history = []
    def on_epoch_end(self, epoch, logs=None):
        m = evaluate(self.model, self.vX, self.vY)
        logs['val_exact'] = m['exact']
        logs['val_cer']   = m['cer']
        self.history.append(logs.copy())

# ─────────────────────────── 8. Model builder ────────────────────────
def build_crnn(variant='base'):
    """
    variant: 'base'  – 2 BiLSTM layers (128,64)
             'small' – 1 BiLSTM layer  (128)
             'large' – 3 BiLSTM layers (256,128,64)
    """
    inp = Input(shape=INPUT_SHAPE, name='image')
    x = inp
    for filters in [32, 64, 128]:
        x = Conv2D(filters,(3,3),padding='same',activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters,(3,3),padding='same',activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.15)(x)

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)  # collapse H

    if variant == 'small':
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
    elif variant == 'base':
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.25))(x)
    else:  # large
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.25))(x)

    out = TimeDistributed(
        Dense(NUM_CLASSES+1, activation='softmax'), name='output')(x)

    mdl = models.Model(inp, out,
                       name=f'CRNN_{variant}')
    mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=ctc_loss_fn)
    return mdl

# ─────────────────────────── 9. Ablation study ───────────────────────
AUG_DEGREES  = [0, 1, 3, 5]       # n rotations per image (0 = no aug)
ARCHITECTURES = ['small', 'base', 'large']

ablation_results = []   # list of dicts

for n_aug in AUG_DEGREES:
    # Build augmented train set
    if n_aug == 0:
        aug_imgs, aug_lbls = train_imgs_raw, train_lbls
    else:
        print(f"\n── Augmenting: {n_aug}× rotation ──")
        aug_imgs, aug_lbls = augment_rotation(train_imgs_raw, train_lbls, n_aug)

    train_X  = normalize(aug_imgs)
    train_Y, _ = encode_labels(aug_lbls)
    print(f"  Train size after aug: {len(train_X):,}")

    for variant in ARCHITECTURES:
        run_name = f"aug{n_aug}_{variant}"
        print(f"\n{'─'*50}")
        print(f"  RUN: {run_name}")

        model = build_crnn(variant)
        n_params = model.count_params()

        val_cb = ValAccCallback(val_X, val_Y)
        ckpt   = ModelCheckpoint(
            f"outputs/models/{run_name}.keras",
            monitor='val_loss', save_best_only=True, verbose=0
        )
        es  = EarlyStopping(monitor='val_loss', patience=12,
                            restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=1e-6, verbose=0)

        t0 = time.time()
        hist = model.fit(
            train_X, train_Y,
            validation_data=(val_X, val_Y),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es, rlr, val_cb, ckpt],
            verbose=0
        )
        elapsed = time.time() - t0

        # Save training curves per run
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ep = range(1, len(hist.history['loss'])+1)
        axes[0].plot(ep, hist.history['loss'],     label='Train Loss')
        axes[0].plot(ep, hist.history['val_loss'], label='Val Loss')
        axes[0].set_title('CTC Loss'); axes[0].legend()
        axes[0].set_xlabel('Epoch')
        val_exact = [h.get('val_exact', np.nan) for h in val_cb.history]
        axes[1].plot(ep, val_exact, color='green', label='Val Exact-Acc')
        axes[1].set_title('Validation Exact-Match Accuracy')
        axes[1].set_xlabel('Epoch'); axes[1].legend()
        fig.suptitle(f'Training Curves — {run_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"outputs/figures/curves_{run_name}.pdf",
                    bbox_inches='tight', dpi=300)
        plt.close()

        # Test evaluation
        res = evaluate(model, test_X, test_Y)

        row = dict(
            aug_degree = n_aug,
            train_size = len(train_X),
            architecture = variant,
            params     = n_params,
            epochs_run = len(hist.history['loss']),
            train_time_s = round(elapsed, 1),
            test_exact = round(res['exact'], 2),
            test_cer   = round(res['cer'], 2),
            test_wer   = round(res['wer'], 2),
        )
        ablation_results.append(row)
        print(f"  → Exact={res['exact']:.2f}%  CER={res['cer']:.2f}%  WER={res['wer']:.2f}%")

        # Keep predictions for best model (aug5, base)
        if n_aug == max(AUG_DEGREES) and variant == 'base':
            best_gt   = res['gt']
            best_pred = res['pred']
            best_model = model

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv("outputs/results/ablation_table.csv", index=False)
print("\n\nABLATION TABLE:")
print(ablation_df.to_string(index=False))

# ─────────────────────────── 10. Ablation plots ──────────────────────
# Fig A: Exact-match heatmap  (aug_degree × architecture)
pivot_exact = ablation_df.pivot(index='architecture',
                                columns='aug_degree',
                                values='test_exact')
pivot_cer   = ablation_df.pivot(index='architecture',
                                columns='aug_degree',
                                values='test_cer')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, pivot, title, fmt, cmap in zip(
        axes,
        [pivot_exact, pivot_cer],
        ['Exact-Match Accuracy (%)', 'CER (%) ↓'],
        ['.2f', '.2f'],
        ['YlGn', 'YlOrRd']):
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"aug={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Augmentation degree (rotations per image)')
    ax.set_ylabel('Architecture')
    for i, j in itertools.product(range(pivot.shape[0]), range(pivot.shape[1])):
        ax.text(j, i, f"{pivot.values[i,j]:{fmt}}",
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='black')
    plt.colorbar(im, ax=ax, fraction=0.04)
plt.suptitle('Ablation Study: Augmentation Degree × Architecture',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/fig2_ablation_heatmap.pdf",
            bbox_inches='tight', dpi=300)
plt.close()

# Fig B: Line plot — aug_degree effect per architecture
fig, ax = plt.subplots(figsize=(8, 5))
markers = ['o','s','^']
for m, arch in zip(markers, ARCHITECTURES):
    sub = ablation_df[ablation_df.architecture==arch].sort_values('aug_degree')
    ax.plot(sub['aug_degree'], sub['test_exact'],
            marker=m, linewidth=2, label=arch.capitalize())
ax.set_xlabel('Augmentation degree (rotations per image)', fontsize=12)
ax.set_ylabel('Exact-Match Accuracy (%)', fontsize=12)
ax.set_title('Effect of Rotation Augmentation on Test Accuracy', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/fig3_aug_effect_line.pdf",
            bbox_inches='tight', dpi=300)
plt.close()

# Fig C: Parameter count vs accuracy (bubble chart)
fig, ax = plt.subplots(figsize=(8, 5))
colors_map = {'small':'#2196F3','base':'#4CAF50','large':'#FF5722'}
for _, row in ablation_df.iterrows():
    ax.scatter(row['params']/1e6, row['test_exact'],
               s=row['aug_degree']*60+80,
               color=colors_map[row['architecture']], alpha=0.75,
               edgecolors='black', linewidths=0.5)
# legend for arch
for arch, c in colors_map.items():
    ax.scatter([], [], color=c, label=arch.capitalize(), s=100)
ax.legend(title='Architecture')
ax.set_xlabel('Parameters (M)', fontsize=12)
ax.set_ylabel('Test Exact-Match Accuracy (%)', fontsize=12)
ax.set_title('Model Complexity vs. Accuracy\n(bubble size = aug degree)',
             fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_params_vs_acc.pdf",
            bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────── 11. Best model evaluation ───────────────
# Use predictions already computed for best run
gt_s, pred_s = best_gt, best_pred

exact = sum(g==p for g,p in zip(gt_s,pred_s)) / len(gt_s) * 100
cer   = jiwer.cer(gt_s, pred_s) * 100
wer   = jiwer.wer(gt_s, pred_s) * 100

# character-level
def str_to_padded(s, L=MAX_LABEL_LENGTH):
    idxs = [char_to_idx[c] for c in s if c in char_to_idx][:L]
    idxs += [-1]*(L-len(idxs))
    return idxs

yt = np.array([str_to_padded(s) for s in gt_s])
yp = np.array([str_to_padded(s) for s in pred_s])
flat_t = yt.flatten(); flat_p = yp.flatten()
mask   = flat_t >= 0
yt_m   = flat_t[mask]
yp_m   = np.clip(flat_p[mask], 0, NUM_CLASSES-1)

kw = dict(average='macro', zero_division=0)
prec = precision_score(yt_m, yp_m, **kw) * 100
rec  = recall_score   (yt_m, yp_m, **kw) * 100
f1   = f1_score       (yt_m, yp_m, **kw) * 100

# Save summary
summary = dict(exact=round(exact,2), cer=round(cer,2), wer=round(wer,2),
               precision=round(prec,2), recall=round(rec,2), f1=round(f1,2))
with open("outputs/results/best_model_metrics.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\nBEST MODEL METRICS:")
for k,v in summary.items(): print(f"  {k:12s}: {v}")

# ─────────────────────────── 12. Metrics bar chart ───────────────────
fig, ax = plt.subplots(figsize=(9, 5))
names = ['Exact\nAcc (%)', 'Precision\n(%)', 'Recall\n(%)',
         'F1\n(%)', 'CER (%)\n↓', 'WER (%)\n↓']
vals  = [exact, prec, rec, f1, cer, wer]
colors = ['#2196F3','#4CAF50','#009688','#673AB7','#FF5722','#F44336']
bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylim(0, max(vals)*1.15)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Best Model — Final Evaluation Metrics', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/fig5_final_metrics.pdf", bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────── 13. Confusion matrix ────────────────────
present = sorted(set(yt_m.tolist()))
labels_cm = [idx_to_char[i] for i in present]
cm = confusion_matrix(yt_m, yp_m, labels=present)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
plt.colorbar(im, ax=ax, fraction=0.03, label='% of True Class')
ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
ax.set_xticklabels(labels_cm, rotation=90, fontsize=7)
ax.set_yticklabels(labels_cm, fontsize=7)
thresh = cm_norm.max()/2
for i,j in itertools.product(range(len(present)), repeat=2):
    v = cm_norm[i,j]
    if v > 1:
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5,
                color='white' if v > thresh else 'black')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Character-Level Confusion Matrix (% per true class)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/fig6_confusion_matrix.pdf",
            bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────── 14. Error analysis ──────────────────────
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

# Worst 10 characters
worst_idx = np.argsort(per_class_acc)[:10]
worst_chars = [(labels_cm[i], per_class_acc[i]) for i in worst_idx]

# Most confused pairs
confused_pairs = []
for i in range(len(present)):
    for j in range(len(present)):
        if i != j and cm[i,j] > 0:
            confused_pairs.append((labels_cm[i], labels_cm[j], cm[i,j], cm_norm[i,j]))
confused_pairs.sort(key=lambda x: -x[2])

# Save error analysis
error_df = pd.DataFrame(confused_pairs[:30],
                        columns=['true','pred','count','pct_of_true'])
error_df.to_csv("outputs/results/error_analysis_top30.csv", index=False)

# Fig: worst chars bar
fig, ax = plt.subplots(figsize=(9, 5))
wc_chars = [x[0] for x in worst_chars]
wc_vals  = [x[1] for x in worst_chars]
ax.barh(wc_chars, wc_vals, color='#EF5350', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Per-Class Accuracy (%)', fontsize=12)
ax.set_title('10 Most Confused Characters (Lowest Per-Class Accuracy)',
             fontweight='bold')
ax.axvline(50, linestyle='--', color='gray', alpha=0.5, label='50% line')
ax.legend(); ax.grid(axis='x', alpha=0.3)
for i,(v,c) in enumerate(zip(wc_vals, wc_chars)):
    ax.text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/fig7_worst_chars.pdf", bbox_inches='tight', dpi=300)
plt.close()

# Fig: CER distribution by label length
err_rows = []
for g,p in zip(gt_s, pred_s):
    c = jiwer.cer([g],[p])*100
    err_rows.append({'gt':g,'pred':p,'cer':c,'len':len(g),'correct':g==p})
err_df = pd.DataFrame(err_rows)
err_df.to_csv("outputs/results/per_sample_errors.csv", index=False)

fig, axes = plt.subplots(1,2,figsize=(12,5))
# CER distribution
axes[0].hist(err_df['cer'], bins=30, color='#42A5F5', edgecolor='black', linewidth=0.4)
axes[0].set_xlabel('CER (%)', fontsize=12)
axes[0].set_ylabel('Sample Count')
axes[0].set_title('CER Distribution Across Test Samples', fontweight='bold')
axes[0].axvline(cer, color='red', linestyle='--', label=f'Mean CER={cer:.1f}%')
axes[0].legend()
# Accuracy by label length
acc_by_len = err_df.groupby('len')['correct'].mean()*100
axes[1].bar(acc_by_len.index, acc_by_len.values, color='#66BB6A',
            edgecolor='black', linewidth=0.4)
axes[1].set_xlabel('Label Length (characters)', fontsize=12)
axes[1].set_ylabel('Exact-Match Accuracy (%)')
axes[1].set_title('Accuracy by Label Length', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for l,v in acc_by_len.items():
    axes[1].text(l, v+0.5, f'{v:.0f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/fig8_error_analysis.pdf", bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────── 15. Sample predictions ──────────────────
# Show 10 correct + 10 incorrect samples
correct_idx   = [i for i,(g,p) in enumerate(zip(gt_s,pred_s)) if g==p][:10]
incorrect_idx = [i for i,(g,p) in enumerate(zip(gt_s,pred_s)) if g!=p][:10]

def plot_predictions(idxs, title, fname):
    n = len(idxs)
    if n == 0: return
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.2))
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis('off')
    for k, i in enumerate(idxs):
        axes[k].imshow(test_imgs_raw[i])
        axes[k].set_title(f"GT:  {gt_s[i]}\nPR: {pred_s[i]}",
                          fontsize=8, color='green' if gt_s[i]==pred_s[i] else 'red')
        axes[k].axis('off')
    fig.suptitle(title, fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close()

plot_predictions(correct_idx,   "Correct Predictions (Best Model)",
                 "outputs/figures/fig9a_correct_samples.pdf")
plot_predictions(incorrect_idx, "Incorrect Predictions (Best Model)",
                 "outputs/figures/fig9b_incorrect_samples.pdf")

# ─────────────────────────── 16. Final ablation table (LaTeX) ─────────
latex = ablation_df[['architecture','aug_degree','train_size','params',
                      'epochs_run','test_exact','test_cer','test_wer']].copy()
latex.columns = ['Arch','Aug°','Train N','Params','Epochs',
                 'Exact (%)','CER (%)','WER (%)']
with open("outputs/results/ablation_table.tex", 'w') as f:
    f.write(latex.to_latex(index=False, float_format='%.2f',
                           caption="Ablation study results.",
                           label="tab:ablation"))

print("\n" + "═"*60)
print("  ALL OUTPUTS SAVED TO  ./outputs/")
print("  figures/ — PDF plots ready for IJDAR submission")
print("  models/  — Keras model checkpoints")
print("  results/ — CSV tables, JSON metrics, LaTeX table")
print("═"*60)