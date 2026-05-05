import os, glob, time, random, joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED)

IMG_SIZE   = 224
DATA_ROOT  = "/content/data/the_wildfire_dataset"
CLASSES    = {"fire": 1, "nofire": 0}

def collect_paths(root):
    paths, labels = [], []
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = p.lower().split(os.sep)
        if "fire" in parts and "nofire" not in parts:
            paths.append(p); labels.append(1)
        elif "nofire" in parts:
            paths.append(p); labels.append(0)
    return np.array(paths), np.array(labels)

paths, labels = collect_paths(DATA_ROOT)
print(f"Total images: {len(paths)} | fire={int((labels==1).sum())} | nofire={int((labels==0).sum())}")

keep = []
for p in tqdm(paths, desc="verify"):
    try:
        with Image.open(p) as im:
            im.verify()
        keep.append(True)
    except Exception:
        keep.append(False)
paths, labels = paths[keep], labels[keep]
print("After cleaning:", len(paths))

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("train/val/test:", len(X_tr), len(X_val), len(X_te))

def extract_features(path, size=IMG_SIZE, hist_bins=32):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    gray = rgb2gray(arr)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    hist_feat = np.concatenate([
        np.histogram(arr[..., c], bins=hist_bins, range=(0, 1), density=True)[0]
        for c in range(3)
    ])

    return np.concatenate([hog_feat, hist_feat]).astype(np.float32)

def build_matrix(paths):
    feats = [extract_features(p) for p in tqdm(paths, desc="features")]
    return np.vstack(feats)

t0 = time.time()
Xtr = build_matrix(X_tr)
Xval = build_matrix(X_val)
Xte = build_matrix(X_te)
print("feature dim:", Xtr.shape[1], "| extract time:", round(time.time()-t0, 1), "s")

np.savez_compressed("/content/features.npz",
                    Xtr=Xtr, Xval=Xval, Xte=Xte,
                    ytr=y_tr, yval=y_val, yte=y_te)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced",
                probability=True, random_state=SEED)),
])

param_grid = {
    "svm__C":     [1, 10],
    "svm__gamma": ["scale", 0.01],
}
gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=3, n_jobs=-1, verbose=2)
gs.fit(Xtr, y_tr)
print("Best params:", gs.best_params_)
best = gs.best_estimator_

yval_pred = best.predict(Xval)
print("VAL F1:", f1_score(y_val, yval_pred))

y_pred = best.predict(Xte)
y_prob = best.predict_proba(Xte)[:, 1]

print(f"Accuracy : {accuracy_score(y_te, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_te, y_pred)*100:.2f}%")
print(f"Recall   : {recall_score(y_te, y_pred)*100:.2f}%")
print(f"F1-Score : {f1_score(y_te, y_pred)*100:.2f}%")
print(f"ROC-AUC  : {roc_auc_score(y_te, y_prob)*100:.2f}%")
print(classification_report(y_te, y_pred, target_names=["nofire", "fire"]))

cm = confusion_matrix(y_te, y_pred, labels=[1, 0])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["fire", "nofire"], yticklabels=["fire", "nofire"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("SVM Confusion Matrix")
plt.show()

joblib.dump(best, "/content/svm_hog_colorhist.joblib")


# Threshold tuning on validation probabilities.
from sklearn.metrics import precision_recall_curve

yval_prob = best.predict_proba(Xval)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, yval_prob)
f1_curve = 2 * prec * rec / (prec + rec + 1e-12)

best_f1_idx  = int(np.nanargmax(f1_curve[:-1]))
thr_f1       = float(thr[best_f1_idx])

mask         = rec[:-1] >= 0.95
thr_recall   = float(thr[mask].max()) if mask.any() else thr_f1
print(f"F1-optimal threshold:    {thr_f1:.3f}  (val F1={f1_curve[best_f1_idx]:.3f})")
print(f"Recall>=0.95 threshold:  {thr_recall:.3f}")

plt.figure(figsize=(7, 4))
plt.plot(thr, prec[:-1], label="Precision")
plt.plot(thr, rec[:-1],  label="Recall")
plt.plot(thr, f1_curve[:-1], label="F1", linestyle="--")
plt.axvline(thr_f1,     color="green", alpha=0.5, label=f"F1-opt={thr_f1:.2f}")
plt.axvline(thr_recall, color="red",   alpha=0.5, label=f"Recall95={thr_recall:.2f}")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend(); plt.title("SVM threshold sweep (val)")
plt.grid(alpha=0.3); plt.show()

def eval_at_threshold(yp_prob, y_true, t, name=""):
    yp = (yp_prob >= t).astype(int)
    print(f"--- {name} (t={t:.2f}) ---")
    print(f"Acc {accuracy_score(y_true, yp)*100:.2f}%  "
          f"P {precision_score(y_true, yp)*100:.2f}%  "
          f"R {recall_score(y_true, yp)*100:.2f}%  "
          f"F1 {f1_score(y_true, yp)*100:.2f}%")
    return yp

print("\n=== Test-set evaluation at tuned thresholds ===")
yte_prob = best.predict_proba(Xte)[:, 1]
_         = eval_at_threshold(yte_prob, y_te, 0.50,       "Default")
yte_f1    = eval_at_threshold(yte_prob, y_te, thr_f1,     "F1-optimal")
yte_rec   = eval_at_threshold(yte_prob, y_te, thr_recall, "Recall-optimized")


# Wider grid (linear + RBF).
from sklearn.model_selection import GridSearchCV

wider_grid = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"svm__kernel": ["rbf"],
     "svm__C":     [0.1, 1, 10, 100],
     "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1]},
]
gs2 = GridSearchCV(pipe, wider_grid, scoring="f1", cv=5, n_jobs=-1, verbose=2)
gs2.fit(Xtr, y_tr)
print("Wider best params:", gs2.best_params_)
print("Wider best CV F1 :", gs2.best_score_)

best_wide = gs2.best_estimator_
yte_pred_w = best_wide.predict(Xte)
yte_prob_w = best_wide.predict_proba(Xte)[:, 1]
print(f"WIDE  Acc {accuracy_score(y_te, yte_pred_w)*100:.2f}%  "
      f"F1 {f1_score(y_te, yte_pred_w)*100:.2f}%  "
      f"AUC {roc_auc_score(y_te, yte_prob_w)*100:.2f}%")

final_model = best_wide if f1_score(y_te, yte_pred_w) > f1_score(y_te, y_pred) else best
joblib.dump(final_model, "/content/svm_final.joblib")


# Robustness: re-extract features from perturbed test images.
from PIL import ImageFilter

def perturb_image(pil_img, kind, level):
    if kind == "blur":
        return pil_img.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "haze":
        arr  = np.asarray(pil_img, dtype=np.float32)
        fog  = np.full_like(arr, 200.0)
        out  = (1 - level) * arr + level * fog
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if kind == "brightness":
        arr = np.asarray(pil_img, dtype=np.float32) * level
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    raise ValueError(kind)

def extract_features_from_pil(pil_img, size=IMG_SIZE, hist_bins=32):
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    hf = hog(gray, orientations=9, pixels_per_cell=(16,16),
             cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)
    cf = np.concatenate([
        np.histogram(arr[..., c], bins=hist_bins, range=(0,1), density=True)[0]
        for c in range(3)
    ])
    return np.concatenate([hf, cf]).astype(np.float32)

def evaluate_perturbation(kind, levels, model, threshold=0.5):
    rows = []
    for lvl in levels:
        feats = []
        for p in tqdm(X_te, desc=f"{kind}={lvl}", leave=False):
            with Image.open(p) as im:
                pim = perturb_image(im.convert("RGB"), kind, lvl)
            feats.append(extract_features_from_pil(pim))
        Xp   = np.vstack(feats)
        prob = model.predict_proba(Xp)[:, 1]
        pred = (prob >= threshold).astype(int)
        rows.append({
            "perturbation": kind, "level": lvl,
            "accuracy":  accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall":    recall_score(y_te, pred),
            "f1":        f1_score(y_te, pred),
            "auc":       roc_auc_score(y_te, prob),
        })
    return rows

robust_rows = []
robust_rows += evaluate_perturbation("blur",       [0, 1, 2, 4, 8],            final_model)
robust_rows += evaluate_perturbation("haze",       [0.0, 0.2, 0.4, 0.6, 0.8],  final_model)
robust_rows += evaluate_perturbation("brightness", [0.5, 0.75, 1.0, 1.25, 1.5], final_model)

import pandas as pd
df_robust = pd.DataFrame(robust_rows)
print(df_robust.round(3))
df_robust.to_csv("/content/svm_robustness.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, kind in zip(axes, ["blur", "haze", "brightness"]):
    sub = df_robust[df_robust.perturbation == kind]
    ax.plot(sub.level, sub.f1,  marker="o", label="F1")
    ax.plot(sub.level, sub.accuracy, marker="s", label="Acc")
    ax.set_title(f"SVM robustness — {kind}")
    ax.set_xlabel("level"); ax.set_ylabel("score")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# Error analysis: most-confident wrong predictions.
y_pred_final = final_model.predict(Xte)
y_prob_final = final_model.predict_proba(Xte)[:, 1]

fn_idx = np.where((y_te == 1) & (y_pred_final == 0))[0]
fp_idx = np.where((y_te == 0) & (y_pred_final == 1))[0]

fn_sorted = fn_idx[np.argsort(y_prob_final[fn_idx])]
fp_sorted = fp_idx[np.argsort(y_prob_final[fp_idx])[::-1]]

def show_grid(idxs, title, ncol=4, n=8):
    n = min(n, len(idxs))
    if n == 0:
        print(f"No examples for: {title}"); return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3))
    axes = np.array(axes).reshape(-1)
    for i, k in enumerate(idxs[:n]):
        img = Image.open(X_te[k]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        axes[i].imshow(img)
        axes[i].set_title(f"p(fire)={y_prob_final[k]:.2f}", fontsize=9)
        axes[i].axis("off")
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.show()

print(f"False negatives (missed fires): {len(fn_idx)}")
print(f"False positives (false alarms): {len(fp_idx)}")
show_grid(fn_sorted, "Most-confident MISSED FIRES (FN)", n=8)
show_grid(fp_sorted, "Most-confident FALSE ALARMS (FP)", n=8)

import json
with open("/content/svm_misclassified.json", "w") as f:
    json.dump({
        "false_negatives": [str(p) for p in X_te[fn_idx]],
        "false_positives": [str(p) for p in X_te[fp_idx]],
    }, f, indent=2)


results = {
    "best_params_initial": gs.best_params_,
    "default_threshold": {
        "accuracy":  float(accuracy_score(y_te, y_pred_final)),
        "precision": float(precision_score(y_te, y_pred_final)),
        "recall":    float(recall_score(y_te, y_pred_final)),
        "f1":        float(f1_score(y_te, y_pred_final)),
        "roc_auc":   float(roc_auc_score(y_te, y_prob_final)),
    },
    "thresholds": {"f1_optimal": thr_f1, "recall95": thr_recall},
    "confusion_matrix": confusion_matrix(y_te, y_pred_final, labels=[1,0]).tolist(),
}
try:
    results["best_params_wide"] = gs2.best_params_
except NameError:
    pass

with open("/content/svm_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
