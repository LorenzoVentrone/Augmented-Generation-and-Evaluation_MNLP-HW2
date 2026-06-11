#!/usr/bin/env bash
# Download the official homework dataset from the Hugging Face Hub and dump each
# split as JSONL under <dest>/<split>/<split>.jsonl.
#
# Run this on a LOGIN node: compute nodes have no network access, so every job
# downstream reads these local copies with TRANSFORMERS_OFFLINE=1.
#
# Usage: ./download_dataset.sh [dest-dir]
set -euo pipefail

DEST_DIR="${1:-/leonardo/home/userexternal/ltam0000/my_scratch/MNLP_FUNTORI_HWII/scratch/dataset}"

mkdir -p "$DEST_DIR"

DEST_DIR="$DEST_DIR" ~/.venv/bin/python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import os

dest_dir = Path(os.environ["DEST_DIR"])
dataset = load_dataset("sapienzanlp-course-materials/hw-mnlp-2026")

# One subfolder per split, one JSONL per split (the layout every other script expects)
for split_name in ("train", "test", "blind"):
    split_dataset = dataset[split_name]
    split_dir = dest_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{split_name}.jsonl"
    split_dataset.to_json(str(split_path), orient="records", lines=True)
    print(f"{split_name}: {len(split_dataset)} examples -> {split_path}")
PY
