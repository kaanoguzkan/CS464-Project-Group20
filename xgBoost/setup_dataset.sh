#!/usr/bin/env bash
set -euo pipefail

# CS464 Group 20 - Wildfire dataset setup (local)
# This script downloads and unzips:
#   elmadafri/the-wildfire-dataset
# into:
#   archive/the_wildfire_dataset_2n_version
#
# Requirements:
#   1) kaggle CLI installed
#   2) ~/.kaggle/kaggle.json exists (chmod 600)

DATA_DIR="archive"
ZIP_PATH="${DATA_DIR}/the-wildfire-dataset.zip"
TARGET_DIR="${DATA_DIR}/the_wildfire_dataset_2n_version"

echo "== Wildfire dataset setup =="
echo "Target directory: ${TARGET_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found."
  echo "Install with one of:"
  echo "  pip install kaggle"
  echo "or"
  echo "  pip3 install kaggle"
  exit 1
fi

if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
  echo "ERROR: ${HOME}/.kaggle/kaggle.json not found."
  echo "Steps:"
  echo "  1) Kaggle -> Account -> Create New API Token"
  echo "  2) mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
  echo "  3) chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

mkdir -p "${DATA_DIR}"

if [ -d "${TARGET_DIR}/train" ] && [ -d "${TARGET_DIR}/val" ] && [ -d "${TARGET_DIR}/test" ]; then
  echo "Dataset already appears prepared at ${TARGET_DIR}"
  echo "Skipping download."
  exit 0
fi

echo "Downloading dataset from Kaggle..."
kaggle datasets download -d elmadafri/the-wildfire-dataset -p "${DATA_DIR}"

if [ ! -f "${ZIP_PATH}" ]; then
  echo "ERROR: download completed but zip not found at ${ZIP_PATH}"
  exit 1
fi

echo "Unzipping..."
unzip -o "${ZIP_PATH}" -d "${DATA_DIR}" >/dev/null

echo "Checking expected split folders..."
if [ -d "${TARGET_DIR}/train/fire" ] && [ -d "${TARGET_DIR}/train/nofire" ] \
  && [ -d "${TARGET_DIR}/val/fire" ] && [ -d "${TARGET_DIR}/val/nofire" ] \
  && [ -d "${TARGET_DIR}/test/fire" ] && [ -d "${TARGET_DIR}/test/nofire" ]; then
  echo "Dataset is ready."
else
  echo "WARNING: expected split structure not fully found."
  echo "Please inspect: ${TARGET_DIR}"
fi

echo
echo "Next step:"
echo "  python \"xgboost/xgboost_model.py\""
