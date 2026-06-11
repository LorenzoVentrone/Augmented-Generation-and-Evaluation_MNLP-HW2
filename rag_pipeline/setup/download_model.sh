#!/usr/bin/env bash
# Download a model into the scratch models directory, either from the Hugging Face
# Hub (model id) or from a shared Google Drive folder (used for the HW1 fine-tuned
# retriever, which lives on Drive rather than on the Hub).
#
# Run this on a LOGIN node: compute nodes have no network access, so every job
# downstream loads these local copies with TRANSFORMERS_OFFLINE=1.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <huggingface-model-id|google-drive-folder-url> [target-name] [models-dir]" >&2
    echo "Example: $0 sentence-transformers/all-MiniLM-L6-v2 all-MiniLM-L6-v2" >&2
    echo "Example: $0 'https://drive.google.com/drive/folders/...' bge-m3-mnrl-hardneg" >&2
    exit 1
fi

SOURCE="$1"
TARGET_NAME="${2:-}"
MODELS_DIR="${3:-/leonardo/home/userexternal/ltam0000/my_scratch/MNLP_FUNTORI_HWII/scratch/models}"

# Derive a target folder name when not given explicitly: last path segment of the
# HF id, or a placeholder for Drive folders (whose URL carries no usable name).
if [[ -z "$TARGET_NAME" ]]; then
    if [[ "$SOURCE" == *"drive.google.com/drive/folders/"* ]]; then
        TARGET_NAME="gdrive_model"
    else
        TARGET_NAME="${SOURCE##*/}"
    fi
fi

TARGET_DIR="$MODELS_DIR/$TARGET_NAME"

mkdir -p "$TARGET_DIR"

SOURCE="$SOURCE" TARGET_DIR="$TARGET_DIR" ~/.venv/bin/python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

source = os.environ["SOURCE"]
target_dir = os.environ["TARGET_DIR"]
target_path = Path(target_dir)

target_path.mkdir(parents=True, exist_ok=True)

is_drive_folder = "drive.google.com/drive/folders/" in source or source.startswith("https://drive.google.com/")

if is_drive_folder:
    from gdown import download_folder

    download_folder(url=source, output=target_dir, quiet=False, use_cookies=False)
    print(f"downloaded google-drive folder -> {target_dir}")
else:
    # HF_TOKEN is only needed for gated models; public ones download anonymously
    token = os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id=source,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
    )

    print(f"downloaded {source} -> {target_dir}")
PY
