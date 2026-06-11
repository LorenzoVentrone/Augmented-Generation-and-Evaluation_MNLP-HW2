"""
Recompute EM / sub-EM / METEOR from existing prediction JSONL files — no model,
no GPU, no regeneration. Joins predictions to gold answers by query_id, so you can
re-score (e.g. after tweaking normalization in rag_metrics.py) or compare several
runs (r16 vs r32, rag vs oracle) in seconds.

The prediction files are the ones written by evaluate.py:
    {"query_id": "...", "retrieved_chunks": [...], "augmented_prompt": "...",
     "generated_answer": "..."}
Gold answers come from the dataset split (matched by query_id), since the output
JSONL doesn't store them.

Usage:
    python score_jsonl.py \
        --dataset_name .../assets/dataset/hw-mnlp-2026 --split test \
        --inputs results/minerva-ft-r16-test-rag.jsonl \
                 results/minerva-ft-r32-test-rag.jsonl \
                 results/minerva-ft-r32-test-oracle.jsonl
"""

import argparse
import json
import os

import rag_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Re-score prediction JSONL files")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Dataset (HF id or save_to_disk dir) to pull gold short_answer from "
                        "(not needed with --gold_field)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--inputs", type=str, nargs="+", required=True,
                   help="One or more prediction JSONL files to score")
    p.add_argument("--gold_field", type=str, default=None,
                   help="Read gold from this field IN the JSONL (e.g. ground_truth) instead "
                        "of joining to a dataset by query_id. Self-contained scoring.")
    return p.parse_args()


def _golds_list(val):
    if isinstance(val, list):
        return [str(g) for g in val if str(g).strip()] or [""]
    return [str(val)] if str(val).strip() else [""]


def load_gold_map(dataset_name, split):
    from datasets import load_dataset, load_from_disk  # lazy: only when joining to a dataset
    if os.path.isdir(dataset_name) and os.path.exists(
        os.path.join(dataset_name, "dataset_dict.json")
    ):
        ds = load_from_disk(dataset_name)[split]
    else:
        ds = load_dataset(dataset_name, split=split)
    if "short_answer" not in ds.column_names:
        raise SystemExit(
            f"Split '{split}' has no 'short_answer' field (blind split?) — "
            "cannot score. Use a split with gold answers."
        )
    return {ex["query_id"]: ex["short_answer"] for ex in ds}


def score_file(path, gold, gold_field=None):
    em = sub = 0
    meteor_sum = 0.0
    n = missing = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            qid = rec.get("query_id")
            gen = rec.get("generated_answer", "")
            if gold_field:
                golds = _golds_list(rec.get(gold_field, []))
            else:
                if qid not in gold:
                    missing += 1
                    continue
                golds = gold[qid]
            s = rag_metrics.score_all(gen, golds)
            em += s["em"]
            sub += s["sub_em"]
            meteor_sum += s["meteor"]
            n += 1
    return {"n": n, "em": em, "sub": sub, "meteor": meteor_sum, "missing": missing}


def main():
    args = parse_args()

    if not rag_metrics.meteor_available():
        print("WARNING: nltk METEOR unavailable — METEOR column will be 0.0. "
              "Install nltk + wordnet/omw-1.4 for real scores.\n")

    gold = None
    if args.gold_field:
        print(f"Reading gold from JSONL field '{args.gold_field}' (self-contained).\n")
    else:
        if not args.dataset_name:
            raise SystemExit("Provide --dataset_name, or use --gold_field to score from the JSONL.")
        print(f"Loading gold answers: {args.dataset_name} [{args.split}]")
        gold = load_gold_map(args.dataset_name, args.split)
        print(f"Gold answers for {len(gold)} query_ids.\n")

    name_w = max(len(os.path.basename(p)) for p in args.inputs)
    header = f"{'file':<{name_w}}  {'n':>5}  {'EM':>8}  {'sub-EM':>8}  {'METEOR':>7}"
    print(header)
    print("-" * len(header))

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"{os.path.basename(path):<{name_w}}  (file not found)")
            continue
        r = score_file(path, gold, gold_field=args.gold_field)
        n = r["n"] or 1  # avoid div-by-zero on empty files
        em_pct = 100 * r["em"] / n
        sub_pct = 100 * r["sub"] / n
        met = r["meteor"] / n
        print(f"{os.path.basename(path):<{name_w}}  {r['n']:>5}  "
              f"{em_pct:>7.2f}%  {sub_pct:>7.2f}%  {met:>7.4f}")
        if r["missing"]:
            print(f"{'':<{name_w}}  ({r['missing']} predictions had a query_id "
                  f"not in the {args.split} split — skipped)")


if __name__ == "__main__":
    main()
