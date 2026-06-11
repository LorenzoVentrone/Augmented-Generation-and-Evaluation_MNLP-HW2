"""
Build the corrective-RAG SFT dataset from the real retriever output.

We precompute the context from OUR semantic
retriever's actual top-k, and add explicit REFUSAL targets when the correct chunk was
not retrieved.

INPUT: the base dataset, augmented by an extra column holding the
retriever's top-10 candidate_chunks indices (most relevant first), e.g.
    example["retrieved_ids"] == [11, 3, 8, 0, 17, 2, 9, 1, 5, 14]

Strict answerable criterion (the dataset has exactly one correct chunk; the answer
lives only there):
    answerable ->  answer_pos ∈ retrieved_ids[:top_k]
    refusal    ->  otherwise   → target = a random phrase from refusal_templates.py

Output: a HuggingFace DatasetDict saved to disk with precomputed fields, already split
into train/eval so training is deterministic and reproducible across experiments:
    {query_id, query, context_chunks: list[str], target: str, is_refusal: int}

train.py consumes this with `--data_mode precomputed`.

Usage:
    python build_sft_dataset.py \
        --dataset_name assets/dataset/hw-mnlp-2026-retrieved \
        --split train \
        --retrieved_field retrieved_ids \
        --top_k 3 \
        --eval_split_ratio 0.02 \
        --seed 42 \
        --out_dir assets/dataset/hw-mnlp-2026-corrective \
        --max_refusal_ratio 0.0
"""

import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict

from data_utils import load_split_any
from refusal_templates import pick_refusal


def parse_args():
    p = argparse.ArgumentParser(description="Build corrective-RAG SFT dataset")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="train",
                   help="Source split to build training data from (usually 'train')")
    p.add_argument("--retrieved_field", type=str, default="retrieved_index",
                   help="Dataset column with the retriever's top-k candidate_chunks "
                        "INDICES (most relevant first).")
    p.add_argument("--top_k", type=int, default=3,
                   help="How many of the retrieved indices to put in the context "
                        "(<= len of the field, which holds top-10).")
    p.add_argument("--eval_split_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_refusal_ratio", type=float, default=0.0,
                   help="If >0 and refusal share exceeds it, downsample ONLY refusal "
                        "examples to this ratio (answerable examples are never dropped). "
                        "0 = keep everything, just warn.")
    return p.parse_args()


def build_records(ds, field, top_k, rng):
    """Produce the precomputed records + stats. One record per source example."""
    records = []
    n_refusal = 0
    n_answerable = 0
    n_no_answer_pos = 0

    for ex in ds:
        candidates = list(ex["candidate_chunks"])
        retrieved = list(ex[field])
        topk_idx = retrieved[:top_k]
        # real retriever order — no reordering: we want the true positional distribution
        context_chunks = [candidates[i] for i in topk_idx]

        pos = ex.get("answer_pos", None)
        if pos is None or pos < 0:
            n_no_answer_pos += 1
            answerable = False  # no known correct chunk → safest as unanswerable
        else:
            answerable = pos in topk_idx

        short_answers = ex.get("short_answer", []) or []
        if answerable:
            target = short_answers[0] if short_answers else candidates[pos][:100]
            is_refusal = 0
            n_answerable += 1
        else:
            target = pick_refusal(rng)
            is_refusal = 1
            n_refusal += 1

        records.append({
            "query_id": ex["query_id"],
            "query": ex["query"],
            "context_chunks": context_chunks,
            "target": target,
            "is_refusal": is_refusal,
        })

    stats = {
        "n_total": len(records),
        "n_answerable": n_answerable,
        "n_refusal": n_refusal,
        "n_no_answer_pos": n_no_answer_pos,
    }
    return records, stats


def maybe_downsample_refusal(records, max_ratio, rng):
    """Drop ONLY refusal records to bring their share down to max_ratio. Never drops
    answerable records."""
    if max_ratio <= 0 or max_ratio >= 1.0:
        return records
    answerable = [r for r in records if r["is_refusal"] == 0]
    refusal = [r for r in records if r["is_refusal"] == 1]
    n_ans = len(answerable)
    # solve r / (n_ans + r) <= max_ratio  ->  r <= max_ratio*n_ans/(1-max_ratio)
    allowed = int(max_ratio * n_ans / (1 - max_ratio))
    if len(refusal) <= allowed:
        return records
    rng.shuffle(refusal)
    merged = answerable + refusal[:allowed]
    rng.shuffle(merged)
    print(f"  downsampled refusal {len(refusal)} -> {allowed} (cap ratio {max_ratio:.2f})")
    return merged


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    ds = load_split_any(args.dataset_name, args.split)
    if args.retrieved_field not in ds.column_names:
        raise SystemExit(
            f"Column '{args.retrieved_field}' not in dataset. Available columns: "
            f"{ds.column_names}."
        )

    records, stats = build_records(ds, args.retrieved_field, args.top_k, rng)

    print(f"Built {stats['n_total']} records "
          f"(answerable={stats['n_answerable']}, refusal={stats['n_refusal']})")
    if stats["n_total"]:
        ratio = stats["n_refusal"] / stats["n_total"]
        print(f"  refusal ratio: {100*ratio:.1f}%")
        if ratio > 0.35:
            print("  refusal ratio > 35%")
    if stats["n_no_answer_pos"]:
        print(f"  note: {stats['n_no_answer_pos']} examples had no answer_pos "
              f"(treated as refusal).")

    records = maybe_downsample_refusal(records, args.max_refusal_ratio, rng)

    full = Dataset.from_list(records)
    if args.eval_split_ratio > 0:
        split = full.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
        dd = DatasetDict({"train": split["train"], "eval": split["test"]})
    else:
        dd = DatasetDict({"train": full})

    os.makedirs(args.out_dir, exist_ok=True)
    dd.save_to_disk(args.out_dir)
    print(f"Saved DatasetDict to {args.out_dir}: "
          + ", ".join(f"{k}={len(v)}" for k, v in dd.items()))

    meta = {**stats, "top_k": args.top_k, "seed": args.seed,
            "retrieved_field": args.retrieved_field,
            "eval_split_ratio": args.eval_split_ratio,
            "max_refusal_ratio": args.max_refusal_ratio,
            "final_sizes": {k: len(v) for k, v in dd.items()}}
    with open(os.path.join(args.out_dir, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
