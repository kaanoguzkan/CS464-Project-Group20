"""
CS464 Group 20 - Wildfire Presence Detection
Random Forest Model: LBP + Color Statistics

Features:
  - Local Binary Patterns (LBP) histogram (uniform, P=24, R=3)
  - Color statistics: mean & std per RGB channel

Usage: Run in Google Colab or locally with Python 3.10+
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file, no blocking)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from itertools import product

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "archive/the_wildfire_dataset_2n_version"
IMG_SIZE = (224, 224)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LBP_RADIUS = 3
LBP_N_POINTS = 24  # 8 * radius
LBP_METHOD = 'uniform'

CLASS_MAP = {'fire': 1, 'nofire': 0}
CLASS_NAMES = ['nofire', 'fire']

OUTPUT_DIR = "rf_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def load_images_from_directory(split_dir):
    """Load and resize images from a split directory."""
    images, labels, corrupted = [], [], 0
    for class_name in ['fire', 'nofire']:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found!")
            continue
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(CLASS_MAP[class_name])
            except Exception as e:
                corrupted += 1
                print(f"  Corrupted: {fpath} ({e})")
    return np.array(images), np.array(labels), corrupted

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

X_train_imgs, y_train, c1 = load_images_from_directory(os.path.join(DATASET_ROOT, "train"))
X_val_imgs, y_val, c2 = load_images_from_directory(os.path.join(DATASET_ROOT, "val"))
X_test_imgs, y_test, c3 = load_images_from_directory(os.path.join(DATASET_ROOT, "test"))

for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    n_fire = np.sum(labels == 1)
    n_no = np.sum(labels == 0)
    print(f"  {name:5s}: {len(labels):4d} imgs | Fire: {n_fire} ({100*n_fire/len(labels):.1f}%) | NoFire: {n_no} ({100*n_no/len(labels):.1f}%)")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_lbp_histogram(image_rgb):
    """Extract normalized LBP histogram from RGB image."""
    gray = rgb2gray(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_color_statistics(image_rgb):
    """Extract mean & std per RGB channel."""
    stats = []
    for ch in range(3):
        channel = image_rgb[:, :, ch].astype(np.float64)
        stats.extend([np.mean(channel), np.std(channel)])
    return np.array(stats)

def extract_all_features(images, desc="Extracting"):
    """Extract LBP + color features for all images."""
    features = []
    total = len(images)
    for i, img in enumerate(images):
        if (i + 1) % 200 == 0 or i == 0 or (i + 1) == total:
            print(f"  {desc}: {i+1}/{total}")
        lbp_hist = extract_lbp_histogram(img)
        color_stats = extract_color_statistics(img)
        features.append(np.concatenate([lbp_hist, color_stats]))
    return np.array(features)

print("\n" + "=" * 60)
print("FEATURE EXTRACTION: LBP + Color Statistics")
print("=" * 60)
print(f"  LBP: radius={LBP_RADIUS}, n_points={LBP_N_POINTS}, method='{LBP_METHOD}'")
print(f"  Color: mean & std per RGB channel (6 features)")

X_train = extract_all_features(X_train_imgs, "Train")
X_val = extract_all_features(X_val_imgs, "Val")
X_test = extract_all_features(X_test_imgs, "Test")

n_lbp = LBP_N_POINTS + 2
feature_names = [f"LBP_bin_{i}" for i in range(n_lbp)] + \
                ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std"]
print(f"\n  Feature vector size: {X_train.shape[1]} ({n_lbp} LBP + 6 color)")

# Free memory
del X_train_imgs, X_val_imgs, X_test_imgs

# ============================================================
# BASELINE RANDOM FOREST
# ============================================================
print("\n" + "=" * 60)
print("BASELINE RANDOM FOREST (n_estimators=100)")
print("=" * 60)

rf_baseline = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)
t0 = time.time()
rf_baseline.fit(X_train, y_train)
print(f"  Training time: {time.time()-t0:.2f}s")

y_val_pred = rf_baseline.predict(X_val)
y_val_proba = rf_baseline.predict_proba(X_val)[:, 1]
print(f"  Val Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"  Val Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"  Val Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"  Val F1-Score:  {f1_score(y_val, y_val_pred):.4f}")
print(f"  Val ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")

# ============================================================
# HYPERPARAMETER TUNING (Validation Set)
# ============================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (using validation set)")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

keys = list(param_grid.keys())
vals = list(param_grid.values())
combos = list(product(*vals))
print(f"  Total combinations: {len(combos)}")

best_f1 = 0
best_params = {}
best_model = None
tuning_log = []

t0 = time.time()
for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    rf = RandomForestClassifier(
        **params, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    val_f1 = f1_score(y_val, y_pred)
    val_acc = accuracy_score(y_val, y_pred)
    val_rec = recall_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_proba)

    tuning_log.append({**params, 'val_f1': val_f1, 'val_acc': val_acc,
                        'val_recall': val_rec, 'val_auc': val_auc})

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_params = params
        best_model = rf

    if (i + 1) % 50 == 0 or (i + 1) == len(combos):
        print(f"    [{i+1}/{len(combos)}] Best Val F1: {best_f1:.4f}")

print(f"\n  Tuning time: {time.time()-t0:.1f}s")
print(f"  Best hyperparameters:")
for k, v in best_params.items():
    print(f"    {k}: {v}")
print(f"  Best validation F1: {best_f1:.4f}")

# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n  Test Accuracy:  {test_acc:.4f}")
print(f"  Test Precision: {test_prec:.4f}")
print(f"  Test Recall:    {test_rec:.4f}")
print(f"  Test F1-Score:  {test_f1:.4f}")
print(f"  Test ROC-AUC:   {test_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

# ============================================================
# THRESHOLD TUNING (Optimize F1 / Recall)
# ============================================================
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

thresholds = np.arange(0.1, 0.91, 0.05)
threshold_results = []
for t in thresholds:
    y_t = (y_test_proba >= t).astype(int)
    if len(np.unique(y_t)) < 2:
        continue
    threshold_results.append({
        'threshold': t,
        'f1': f1_score(y_test, y_t),
        'recall': recall_score(y_test, y_t),
        'precision': precision_score(y_test, y_t),
        'accuracy': accuracy_score(y_test, y_t),
    })

best_thresh = max(threshold_results, key=lambda x: x['f1'])
print(f"  Best threshold for F1: {best_thresh['threshold']:.2f} "
      f"(F1={best_thresh['f1']:.4f}, Recall={best_thresh['recall']:.4f})")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Random Forest – Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rf_confusion_matrix.png")

# --- ROC Curve ---
fig, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, label=f'RF (AUC = {test_auc:.4f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Random Forest – ROC Curve (Test Set)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rf_roc_curve.png")

# --- Feature Importance ---
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_k = len(feature_names)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(top_k), importances[sorted_idx[:top_k]][::-1], color='steelblue')
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_k]][::-1])
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Random Forest – Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rf_feature_importance.png")

# --- Threshold vs Metrics ---
fig, ax = plt.subplots(figsize=(8, 5))
ts = [r['threshold'] for r in threshold_results]
ax.plot(ts, [r['f1'] for r in threshold_results], 'o-', label='F1')
ax.plot(ts, [r['recall'] for r in threshold_results], 's-', label='Recall')
ax.plot(ts, [r['precision'] for r in threshold_results], '^-', label='Precision')
ax.axvline(x=best_thresh['threshold'], color='red', linestyle='--', alpha=0.7, label=f"Best F1 @ {best_thresh['threshold']:.2f}")
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_title('Random Forest – Threshold Tuning')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_threshold_tuning.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rf_threshold_tuning.png")

# --- Hyperparameter Sensitivity: n_estimators ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for param_name, ax in zip(['n_estimators', 'max_depth'], axes):
    unique_vals = sorted(set(r[param_name] for r in tuning_log if r[param_name] is not None))
    mean_f1s = []
    for v in unique_vals:
        f1s = [r['val_f1'] for r in tuning_log if r[param_name] == v]
        mean_f1s.append(np.mean(f1s))
    ax.plot(unique_vals, mean_f1s, 'o-', linewidth=2)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Mean Val F1')
    ax.set_title(f'Sensitivity: {param_name}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_hyperparam_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: rf_hyperparam_sensitivity.png")

# ============================================================
# ABLATION STUDY: LBP-only vs Color-only vs LBP+Color
# ============================================================
print("\n" + "=" * 60)
print("ABLATION STUDY: Feature Groups")
print("=" * 60)

ablation_configs = {
    'LBP only':   (slice(0, n_lbp), slice(0, n_lbp), slice(0, n_lbp)),
    'Color only': (slice(n_lbp, None), slice(n_lbp, None), slice(n_lbp, None)),
    'LBP+Color':  (slice(None), slice(None), slice(None)),
}

ablation_results = {}
for name, (tr_s, va_s, te_s) in ablation_configs.items():
    rf_abl = RandomForestClassifier(
        **best_params, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_abl.fit(X_train[:, tr_s], y_train)
    y_p = rf_abl.predict(X_test[:, te_s])
    y_pr = rf_abl.predict_proba(X_test[:, te_s])[:, 1]
    res = {
        'accuracy': accuracy_score(y_test, y_p),
        'precision': precision_score(y_test, y_p),
        'recall': recall_score(y_test, y_p),
        'f1': f1_score(y_test, y_p),
        'auc': roc_auc_score(y_test, y_pr),
    }
    ablation_results[name] = res
    print(f"  {name:12s} -> F1: {res['f1']:.4f} | Acc: {res['accuracy']:.4f} | AUC: {res['auc']:.4f}")

# ============================================================
# SAVE MODEL & RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL & RESULTS")
print("=" * 60)

joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'rf_best_model.joblib'))
joblib.dump({'feature_names': feature_names, 'best_params': best_params,
             'test_metrics': {'accuracy': test_acc, 'precision': test_prec,
                              'recall': test_rec, 'f1': test_f1, 'auc': test_auc}},
            os.path.join(OUTPUT_DIR, 'rf_metadata.joblib'))

print(f"  Model saved to {OUTPUT_DIR}/rf_best_model.joblib")
print(f"  Metadata saved to {OUTPUT_DIR}/rf_metadata.joblib")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Best Hyperparameters: {best_params}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Test Precision: {test_prec:.4f}")
print(f"  Test Recall:    {test_rec:.4f}")
print(f"  Test F1-Score:  {test_f1:.4f}")
print(f"  Test ROC-AUC:   {test_auc:.4f}")
print(f"  Best Threshold: {best_thresh['threshold']:.2f} (F1={best_thresh['f1']:.4f})")
print("=" * 60)
print("DONE")
