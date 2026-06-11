"""Shared dataset-loading helper for the RAG/corrective scripts.

Handles every way a split can be referenced:
  - a save_to_disk DatasetDict directory (has dataset_dict.json) -> index by split
  - a save_to_disk single-Dataset directory (dataset_info.json/state.json) -> as-is
  - a directory of raw data files (e.g. .../train/train.jsonl) -> load_dataset(json|parquet)
  - a single .jsonl/.json/.parquet file
  - a HuggingFace hub id -> load_dataset(..., split=split)
"""

import glob
import os

from datasets import DatasetDict, load_dataset, load_from_disk

# extension glob -> datasets builder name
_FILE_FORMATS = (("*.jsonl", "json"), ("*.json", "json"), ("*.parquet", "parquet"))


def _looks_like_save_to_disk(path):
    return any(
        os.path.exists(os.path.join(path, m))
        for m in ("dataset_dict.json", "dataset_info.json", "state.json")
    )


def _load_data_files(path):
    """Load a directory of data files or a single data file into one Dataset."""
    if os.path.isdir(path):
        for pattern, fmt in _FILE_FORMATS:
            files = sorted(glob.glob(os.path.join(path, pattern)))
            if files:
                return load_dataset(fmt, data_files=files, split="train")
        raise FileNotFoundError(
            f"No .jsonl/.json/.parquet files found in directory: {path}"
        )
    fmt = "parquet" if path.lower().endswith(".parquet") else "json"
    return load_dataset(fmt, data_files=path, split="train")


def load_split_any(dataset_name, split):
    if os.path.isdir(dataset_name):
        if _looks_like_save_to_disk(dataset_name):
            obj = load_from_disk(dataset_name)
            return obj[split] if isinstance(obj, DatasetDict) else obj
        return _load_data_files(dataset_name)
    if os.path.isfile(dataset_name):
        return _load_data_files(dataset_name)
    return load_dataset(dataset_name, split=split)
