"""
CS464 Group 20 - Wildfire Presence Detection
XGBoost Model: HOG + LBP + Color Histogram + Color Statistics

This is the first local baseline script for XGBoost.
It follows the same train/val/test directory convention as the Random Forest script:
  DATASET_ROOT/
    train/{fire,nofire}
    val/{fire,nofire}
    test/{fire,nofire}
"""

import os
import time
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "archive/the_wildfire_dataset_2n_version"
IMG_SIZE = (224, 224)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CLASS_MAP = {"fire": 1, "nofire": 0}
CLASS_NAMES = ["nofire", "fire"]

OUTPUT_DIR = "xgboost_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP
LBP_RADIUS = 3
LBP_N_POINTS = 24
LBP_METHOD = "uniform"

# Color
HIST_BINS = 32


# ============================================================
# DATA LOADING
# ============================================================
def load_images_from_directory(split_dir):
    """Load and resize images from a split directory."""
    images, labels, corrupted = [], [], 0
    for class_name in ["fire", "nofire"]:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found")
            continue
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(CLASS_MAP[class_name])
            except Exception as err:
                corrupted += 1
                print(f"  Corrupted: {fpath} ({err})")
    return np.array(images), np.array(labels), corrupted


def print_split_stats(name, labels):
    n_fire = np.sum(labels == 1)
    n_no = np.sum(labels == 0)
    total = len(labels)
    print(
        f"  {name:5s}: {total:4d} imgs | "
        f"Fire: {n_fire} ({100 * n_fire / total:.1f}%) | "
        f"NoFire: {n_no} ({100 * n_no / total:.1f}%)"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_hog_features(image_rgb):
    """Extract HOG vector from grayscale image."""
    gray = rgb2gray(image_rgb)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_lbp_histogram(image_rgb):
    """Extract normalized LBP histogram."""
    gray = rgb2gray(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_color_histogram(image_rgb):
    """Extract RGB histogram (density-normalized per channel)."""
    arr = image_rgb.astype(np.float32) / 255.0
    channel_hists = [
        np.histogram(arr[:, :, c], bins=HIST_BINS, range=(0, 1), density=True)[0]
        for c in range(3)
    ]
    return np.concatenate(channel_hists)


def extract_color_statistics(image_rgb):
    """Extract mean and std per RGB channel."""
    stats = []
    for c in range(3):
        ch = image_rgb[:, :, c].astype(np.float64)
        stats.extend([np.mean(ch), np.std(ch)])
    return np.array(stats)


def extract_all_features(images, desc="Extracting"):
    """Extract concatenated feature vectors for all images."""
    vectors = []
    total = len(images)
    for i, img in enumerate(images):
        if (i + 1) % 200 == 0 or i == 0 or (i + 1) == total:
            print(f"  {desc}: {i + 1}/{total}")

        f_hog = extract_hog_features(img)
        f_lbp = extract_lbp_histogram(img)
        f_hist = extract_color_histogram(img)
        f_stats = extract_color_statistics(img)
        vectors.append(np.concatenate([f_hog, f_lbp, f_hist, f_stats]))
    return np.array(vectors)


# ============================================================
# TRAIN + EVALUATE
# ============================================================
def evaluate_and_print(y_true, y_pred, y_proba, prefix=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"{prefix}Accuracy:  {acc:.4f}")
    print(f"{prefix}Precision: {prec:.4f}")
    print(f"{prefix}Recall:    {rec:.4f}")
    print(f"{prefix}F1-Score:  {f1:.4f}")
    print(f"{prefix}ROC-AUC:   {auc:.4f}")
    return acc, prec, rec, f1, auc


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost - Confusion Matrix (Test Set)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "xgb_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    X_train_imgs, y_train, c1 = load_images_from_directory(os.path.join(DATASET_ROOT, "train"))
    X_val_imgs, y_val, c2 = load_images_from_directory(os.path.join(DATASET_ROOT, "val"))
    X_test_imgs, y_test, c3 = load_images_from_directory(os.path.join(DATASET_ROOT, "test"))
    print(f"  Corrupted images skipped: {c1 + c2 + c3}")

    print_split_stats("Train", y_train)
    print_split_stats("Val", y_val)
    print_split_stats("Test", y_test)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION: HOG + LBP + COLOR")
    print("=" * 60)
    t0 = time.time()
    X_train = extract_all_features(X_train_imgs, "Train")
    X_val = extract_all_features(X_val_imgs, "Val")
    X_test = extract_all_features(X_test_imgs, "Test")
    print(f"  Feature extraction time: {time.time() - t0:.1f}s")
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Standardize for stabler optimization.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    del X_train_imgs, X_val_imgs, X_test_imgs

    print("\n" + "=" * 60)
    print("TRAINING BASELINE XGBOOST")
    print("=" * 60)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    t1 = time.time()
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )
    print(f"  Training time: {time.time() - t1:.2f}s")

    print("\nValidation metrics:")
    y_val_pred = model.predict(X_val_s)
    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    evaluate_and_print(y_val, y_val_pred, y_val_proba, prefix="  ")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    y_test_pred = model.predict(X_test_s)
    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    acc, prec, rec, f1, auc = evaluate_and_print(y_test, y_test_pred, y_test_proba, prefix="  ")

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

    save_confusion_matrix(y_test, y_test_pred)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "xgb_metrics.npz"),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
    )
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'xgb_metrics.npz')}")

    print("\nDONE")


if __name__ == "__main__":
    main()
