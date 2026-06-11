#!/usr/bin/env python3
"""Two-stage retrieval over the homework dataset (HW1 architecture).

For every query, rank its `candidate_chunks` with:
  1. a bi-encoder (our HW1 bge-m3 fine-tuned with MNRL + hard negatives):
     cosine similarity between query and chunks, keep the top-k pool;
  2. a cross-encoder reranker (bge-reranker-v2-m3): rescore the surviving
     (query, chunk) pairs and sort them.

Each split is then re-written as JSONL with an extra `retrieved_index` field
(the ranked candidate indices, best first). Downstream consumers:
  - the generation step builds RAG prompts from the first 3 indices
    (top_k = 3 is mandated by the homework specs);
  - the SFT dataset builder uses the ranking to decide whether the gold chunk
    was actually retrieved (corrective-RAG training).
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm


DEFAULT_DATASET_DIR = Path("/leonardo/home/userexternal/ltam0000/my_scratch/MNLP_FUNTORI_HWII/scratch/dataset")
DEFAULT_OUTPUT_DIR = Path("/leonardo/home/userexternal/ltam0000/my_scratch/MNLP_FUNTORI_HWII/scratch/dataset_retrieved")
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich dataset splits with a reranked retrieved_index field and save them as JSONL."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing <split>/<split>.jsonl files (see setup/download_dataset.sh).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the enriched JSONL files will be written.",
    )
    parser.add_argument(
        "--retriever-model",
        required=True,
        type=Path,
        help="Local path to the bi-encoder retriever (the HW1 fine-tuned bge-m3).",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder used for the second-stage rescoring (local path or HF id).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of ranked indices to keep in retrieved_index.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=32,
        help="Batch size for CrossEncoder.predict.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional limit for quick smoke tests; use -1 to process the full split.",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to enrich, e.g. 'blind' or 'train,test,blind'.",
    )
    return parser.parse_args()


def resolve_split_path(dataset_dir: Path, split_name: str) -> Path:
    """Accept both the canonical layout (<split>/<split>.jsonl) and a flat one."""
    preferred = dataset_dir / split_name / f"{split_name}.jsonl"
    if preferred.exists():
        return preferred

    fallback = dataset_dir / f"{split_name}.jsonl"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Missing JSONL for split '{split_name}' under {dataset_dir}")


def load_jsonl_split(dataset_dir: Path, split_name: str):
    split_path = resolve_split_path(dataset_dir, split_name)
    print(f"Loading {split_name} split from {split_path}")
    return load_dataset("json", data_files=str(split_path), split="train")


def build_models(retriever_model_path: Path, reranker_model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = SentenceTransformer(str(retriever_model_path), device=device)
    reranker = CrossEncoder(reranker_model, device=device)
    print(f"retriever loaded from: {retriever_model_path}")
    print(f"reranker loaded from: {reranker_model}")
    print(f"device: {device}")
    return retriever, reranker, device


@torch.no_grad()
def rank_candidates(query, candidates, retriever, reranker, device, top_k=10, encode_batch_size=64, reranker_batch_size=32):
    """Return the indices of `candidates` sorted by relevance (best first).

    Stage 1 (recall): bi-encoder cosine similarity selects a pool of `top_k`
    candidates out of the full list.
    Stage 2 (precision): the cross-encoder rescores only the pooled
    (query, chunk) pairs, which is far cheaper than scoring every candidate.
    """
    if not candidates:
        return []
    q = retriever.encode(query, convert_to_tensor=True, device=device, batch_size=encode_batch_size)
    c = retriever.encode(candidates, convert_to_tensor=True, device=device, batch_size=encode_batch_size)
    pool_size = min(top_k, len(candidates))
    if pool_size == 0:
        return []

    cos = F.cosine_similarity(q.unsqueeze(0), c, dim=1)
    pool_idx = torch.topk(cos, pool_size).indices.cpu().tolist()
    scores = reranker.predict(
        [[query, candidates[i]] for i in pool_idx],
        batch_size=reranker_batch_size,
        show_progress_bar=False,
    )
    order = sorted(range(pool_size), key=lambda i: scores[i], reverse=True)
    return [pool_idx[i] for i in order][:top_k]


def enrich_split(dataset_split, retriever, reranker, device, top_k=10, max_samples=-1, encode_batch_size=64, reranker_batch_size=32):
    """Run retrieval on every record of a split and append `retrieved_index`."""
    samples = dataset_split
    if max_samples != -1:
        samples = dataset_split.select(range(min(max_samples, len(dataset_split))))
        print(f"Smoke-test mode: processing {len(samples)} samples")

    enriched_records = []
    for sample in tqdm(samples, desc="Retrieval + reranking"):
        record = dict(sample)
        record["retrieved_index"] = rank_candidates(
            query=sample["query"],
            candidates=sample["candidate_chunks"],
            retriever=retriever,
            reranker=reranker,
            device=device,
            top_k=top_k,
            encode_batch_size=encode_batch_size,
            reranker_batch_size=reranker_batch_size,
        )
        enriched_records.append(record)

    return enriched_records


def write_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    retriever, reranker, device = build_models(args.retriever_model, args.reranker_model)

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"splits to process: {split_names}")

    for split_name in split_names:
        split = load_jsonl_split(args.dataset_dir, split_name)
        print(f"{split_name} split: {len(split)} rows")

        records = enrich_split(
            split,
            retriever=retriever,
            reranker=reranker,
            device=device,
            top_k=args.top_k,
            max_samples=args.max_samples,
            encode_batch_size=args.encode_batch_size,
            reranker_batch_size=args.reranker_batch_size,
        )

        # Mirror the input layout: <output-dir>/<split>/<split>.jsonl
        out_path = args.output_dir / split_name / f"{split_name}.jsonl"
        write_jsonl(records, out_path)
        print(f"saved {split_name}: {out_path}")


if __name__ == "__main__":
    main()
