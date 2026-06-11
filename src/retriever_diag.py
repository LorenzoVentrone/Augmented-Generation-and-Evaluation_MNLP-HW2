"""
Diagnose the semantic retriever BEFORE building the corrective-RAG SFT dataset.

Reads the retriever's top-k indices from a dataset column. Computes:
  - recall@k for several k  (= fraction of queries whose correct chunk is in top-k)
  - => refusal-rate @k       (= 1 - recall@k, the fraction that becomes refusal)
  - position distribution of the correct chunk within the retrieved list

This decides the top_k to use (§1 of CORRECTIVE_RAG_PLAN.md). Run on the login node.

Usage:
    python retriever_diag.py \
        --dataset_name <dataset_dir> \
        --split train \
        --retrieved_field retrieved_ids \
        --ks 1 3 5 10 \
        --out <out_dir>
"""

import argparse
import os
from collections import Counter

from data_utils import load_split_any


def parse_args():
    p = argparse.ArgumentParser(description="Retriever recall@k / refusal-rate diagnostic")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--retrieved_field", type=str, default="retrieved_index",
                   help="Dataset column with the retriever's top-k candidate_chunks indices")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--out", type=str, default=None, help="Optional markdown report path")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_split_any(args.dataset_name, args.split)
    if args.retrieved_field not in ds.column_names:
        raise SystemExit(
            f"Column '{args.retrieved_field}' not in dataset. Available columns: "
            f"{ds.column_names}. Set --retrieved_field to the colleague's column name."
        )

    ks = sorted(set(args.ks))
    field_len = len(ds[0][args.retrieved_field]) 
    ks = [k for k in ks if k <= field_len]
    max_k = max(ks) if ks else field_len

    n = 0
    no_answer_pos = 0
    hits_at = {k: 0 for k in ks}
    rank_positions = Counter()  
    found_anywhere = 0

    for ex in ds:
        n += 1
        pos = ex.get("answer_pos", None)
        if pos is None or pos < 0:
            no_answer_pos += 1
            continue
        retrieved = list(ex[args.retrieved_field])
        if pos in retrieved:
            r = retrieved.index(pos)
            rank_positions[r] += 1
            found_anywhere += 1
            for k in ks:
                if r < k:
                    hits_at[k] += 1

    usable = n - no_answer_pos
    denom = usable if usable > 0 else 1

    lines = []
    lines.append(f"# Retriever diagnostic — split={args.split}, field={args.retrieved_field}")
    lines.append("")
    lines.append(f"- examples: {n}")
    lines.append(f"- usable (answer_pos present): {usable}")
    if no_answer_pos:
        lines.append(f"- skipped (no answer_pos): {no_answer_pos}")
    lines.append(f"- retrieved list length (top-N): {field_len}")
    lines.append(f"- correct chunk present in top-{field_len}: "
                 f"{found_anywhere}/{usable} ({100*found_anywhere/denom:.1f}%)")
    lines.append("")
    lines.append("| k | recall@k | refusal-rate@k |")
    lines.append("|---|---|---|")
    for k in ks:
        rec = hits_at[k] / denom
        lines.append(f"| {k} | {100*rec:.1f}% | {100*(1-rec):.1f}% |")
    lines.append("")
    lines.append("## Position of the correct chunk within the retrieved list")
    lines.append("")
    lines.append("| rank (0-based) | count | cum. recall |")
    lines.append("|---|---|---|")
    cum = 0
    for r in range(field_len):
        c = rank_positions.get(r, 0)
        cum += c
        lines.append(f"| {r} | {c} | {100*cum/denom:.1f}% |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Pick k so refusal-rate@k is moderate (target ≤ ~35%). If refusal-rate@3 "
                 "is high, prefer k=5. Then set `--top_k` in build_sft_dataset.py / training.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
