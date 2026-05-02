# CLAUDE.md — Wildfire Detection Progress Report Implementation Plan (Google Colab)

## Project Context
CS464 Machine Learning course, Group 20. Wildfire presence detection from RGB images. Progress report due **April 5, 5 PM**. We need to train **at least one model** and present initial evaluation results.

## Dataset
- **Source:** The Wildfire Dataset by elmadafri on Kaggle
- **URL:** https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset
- **Task:** Binary classification — wildfire vs. non-wildfire

## Goal
Produce a **single Colab notebook** (`wildfire_progress.ipynb`) that runs top-to-bottom with no manual intervention (aside from initial Kaggle download). Each section should be a clearly labeled cell group.

## Colab Setup

### Cell 1: Environment & GPU Check
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```
- Runtime → Change runtime type → **T4 GPU**
- All pre-installed on Colab: torch, torchvision, sklearn, matplotlib, seaborn, PIL
- Only install if needed: `!pip install -q scikit-image`

### Cell 2: Dataset Download via Kaggle API
```python
# Option A: Kaggle API
!pip install -q kaggle
# User uploads kaggle.json manually or mounts Drive
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d elmadafri/the-wildfire-dataset -p /content/data --unzip

# Option B: If dataset is already on Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# !unzip /content/drive/MyDrive/wildfire-dataset.zip -d /content/data
```
- Provide **both options** in the notebook, let user uncomment whichever applies
- After download, print the folder structure and total file count

### Cell 3: Mount Google Drive (for saving outputs)
```python
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = '/content/drive/MyDrive/CS464_Wildfire/outputs'
CHECKPOINT_DIR = '/content/drive/MyDrive/CS464_Wildfire/checkpoints'
!mkdir -p {OUTPUT_DIR} {CHECKPOINT_DIR}
```
- Save all outputs and checkpoints to Drive so they persist after runtime disconnects

## Implementation Steps

### Cell Group 4: Dataset Exploration
- Count total images per class (wildfire vs. non-wildfire)
- Report image resolution statistics (min, max, mean) by sampling ~500 images
- Report class balance ratio
- Print a summary table
- Display a class distribution bar chart inline
- Save chart to `{OUTPUT_DIR}/class_distribution.png`

### Cell Group 5: Preprocessing & DataLoaders
- Remove any corrupted/unreadable images (try-except on PIL open)
- Stratified 70/15/15 train/val/test split using `sklearn.model_selection.train_test_split`
- Resize all images to 224×224
- Normalize with ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Training augmentations: RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- Validation/Test: Resize(256) + CenterCrop(224) + Normalize
- Handle class imbalance via `WeightedRandomSampler`
- Return PyTorch DataLoaders with `batch_size=32`, `num_workers=2`, `pin_memory=True`
- Display a grid of 8 sample augmented training images inline for sanity check

### Cell Group 6: Model Definition — EfficientNet-B0
- Load `torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')`
- Replace classifier head: `nn.Linear(1280, 2)`
- Move model to GPU
- Print model summary (total params, trainable params)

### Cell Group 7: Training Loop
- **Phase 1 — Frozen backbone** (5 epochs):
  - Freeze all layers except classifier head
  - Optimizer: Adam, LR=1e-3
  - Loss: CrossEntropyLoss with class weights
- **Phase 2 — Full fine-tune** (10 epochs):
  - Unfreeze all layers
  - Optimizer: Adam, LR=1e-4
  - Same loss
- Track per-epoch: train loss, train acc, val loss, val acc, val F1
- Save best model checkpoint (by val F1) to `{CHECKPOINT_DIR}/best_efficientnet.pth`
- Print a training progress table after each epoch
- Plot training/validation loss and accuracy curves inline at the end
- Save plots to `{OUTPUT_DIR}/training_curves.png`
- Use `random_state=42` / `torch.manual_seed(42)` everywhere

### Cell Group 8: Evaluation
- Load best checkpoint
- Evaluate on the held-out test set
- Compute and print: **Accuracy, Precision, Recall, F1-score, ROC-AUC**
- Generate and display confusion matrix inline (use seaborn heatmap)
- Save confusion matrix to `{OUTPUT_DIR}/confusion_matrix.png`
- Save metrics to `{OUTPUT_DIR}/metrics.json`:
```json
{
  "model": "EfficientNet-B0 (fine-tuned)",
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "roc_auc": 0.0
}
```

### Cell Group 9: Summary
- Print a clean final summary:
```
============ RESULTS ============
Model: EfficientNet-B0 (fine-tuned)
Accuracy:  X.XX%
Precision: X.XX%
Recall:    X.XX%
F1-score:  X.XX%
ROC-AUC:   X.XX
=================================
Outputs saved to Google Drive:
  - {OUTPUT_DIR}/class_distribution.png
  - {OUTPUT_DIR}/training_curves.png
  - {OUTPUT_DIR}/confusion_matrix.png
  - {OUTPUT_DIR}/metrics.json
```

## Colab-Specific Notes
- **All plots must display inline** (`plt.show()`) AND save to Drive
- **All checkpoints and outputs go to Google Drive** — Colab local storage is wiped on disconnect
- Use `tqdm` for progress bars (pre-installed on Colab)
- Use `torch.cuda.amp` (mixed precision) to speed up training and reduce VRAM usage on T4
- Keep `num_workers=2` — Colab crashes with higher values
- Add `torch.cuda.empty_cache()` between Phase 1 and Phase 2 training
- If runtime disconnects mid-training, the checkpoint on Drive lets us resume

## Important Notes
- **Do NOT do hyperparameter tuning across models** — not required for progress report
- **Do NOT compare models** — just train EfficientNet-B0 and report results
- **Do NOT implement robustness testing yet** — that's future work
- **Single notebook, runs top-to-bottom** — no separate .py files
- Use `random_state=42` everywhere for reproducibility
