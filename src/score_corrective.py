"""
Score a Corrective-RAG prediction JSONL, splitting the test set into ANSWERABLE vs
UNANSWERABLE using the retriever ranking, because plain EM is meaningless once the
model abstains.

Split rule (strict, same as build_sft_dataset.py):
    answerable    ⇔  answer_pos ∈ retrieved_ids[:top_k]   (correct chunk was retrieved)
    unanswerable  ⇔  otherwise

Metrics:
  - ANSWERABLE: EM / sub-EM / METEOR (a refusal here is a failure → counts as wrong),
    plus over-refusal rate (model refused although the answer was in context = lost EM).
  - UNANSWERABLE: abstention recall (% correctly refused) and hallucination rate
    (% that produced a non-refusal answer = faithfulness failure).
  - OVERALL: answer-or-abstain accuracy = (correct answers on answerable
    + correct abstentions on unanswerable) / total.

Abstention is detected with refusal_templates.is_refusal_text (template + key-phrase
match, robust to paraphrase).

Usage:
    python score_corrective.py \
        --dataset_name assets/dataset/hw-mnlp-2026-retrieved --split test \
        --retrieved_field retrieved_ids --top_k 3 \
        --inputs results/qwen25-3b-v3-test-rag.jsonl
"""

import argparse
import json
import os

import rag_metrics
from refusal_templates import is_refusal_text


def parse_args():
    p = argparse.ArgumentParser(description="Score corrective-RAG predictions (answerable/unanswerable)")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Dataset with retrieved_index + answer_pos + short_answer "
                        "(not needed with --from_jsonl)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--retrieved_field", type=str, default="retrieved_index",
                   help="Dataset column with the retriever's top-k candidate_chunks indices")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--inputs", type=str, nargs="+", required=True)
    # self-contained mode: read everything from the prediction JSONL itself
    p.add_argument("--from_jsonl", action="store_true",
                   help="Read answerable + gold from each prediction record instead of an "
                        "external dataset: answerable = ground_truth_index ∈ retrieved_chunks, "
                        "gold = ground_truth. No dataset / query_id join needed.")
    p.add_argument("--answer_pos_field", type=str, default="ground_truth_index")
    p.add_argument("--retrieved_in_rec", type=str, default="retrieved_chunks")
    p.add_argument("--gold_field", type=str, default="ground_truth")
    return p.parse_args()


def _golds_list(val):
    if isinstance(val, list):
        return [str(g) for g in val if str(g).strip()] or [""]
    return [str(val)] if str(val).strip() else [""]


def build_gold(ds, field, top_k):
    """query_id -> {gold: [...], answerable: bool}."""
    info = {}
    for ex in ds:
        qid = ex["query_id"]
        pos = ex.get("answer_pos", None)
        retrieved = list(ex[field]) if field in ex else []
        if pos is None or pos < 0:
            answerable = False
        else:
            answerable = pos in retrieved[:top_k]
        info[qid] = {
            "gold": ex.get("short_answer", []),
            "answerable": answerable,
        }
    return info


def score_file(path, info, from_jsonl=False, top_k=3,
               answer_pos_field="ground_truth_index",
               retrieved_in_rec="retrieved_chunks", gold_field="ground_truth"):
    # answerable bucket
    a_n = a_em = a_sub = 0
    a_meteor = 0.0
    a_over_refusal = 0          # refused although answerable
    # unanswerable bucket
    u_n = u_abstain = 0         # correctly refused
    # bookkeeping
    missing = 0

    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            qid = rec.get("query_id")
            gen = rec.get("generated_answer", "")
            if from_jsonl:
                pos = rec.get(answer_pos_field, None)
                retrieved = rec.get(retrieved_in_rec, []) or []
                answerable = pos is not None and pos in list(retrieved)[:top_k]
                meta = {"answerable": answerable, "gold": _golds_list(rec.get(gold_field, []))}
            else:
                meta = info.get(qid)
                if meta is None:
                    missing += 1
                    continue
            refused = is_refusal_text(gen)
            if meta["answerable"]:
                a_n += 1
                if refused:
                    a_over_refusal += 1
                    # a refusal scores 0 on all answer metrics → don't add
                    continue
                s = rag_metrics.score_all(gen, meta["gold"])
                a_em += s["em"]
                a_sub += s["sub_em"]
                a_meteor += s["meteor"]
            else:
                u_n += 1
                if refused:
                    u_abstain += 1

    return {
        "a_n": a_n, "a_em": a_em, "a_sub": a_sub, "a_meteor": a_meteor,
        "a_over_refusal": a_over_refusal,
        "u_n": u_n, "u_abstain": u_abstain,
        "missing": missing,
    }


def main():
    args = parse_args()

    if not rag_metrics.meteor_available():
        print("WARNING: nltk METEOR unavailable — METEOR will be 0.0.\n")

    info = None
    if args.from_jsonl:
        print(f"Self-contained mode (answerable/gold read from each JSONL) @ top_k={args.top_k}\n")
    else:
        if not args.dataset_name:
            raise SystemExit("Provide --dataset_name, or use --from_jsonl to score from the JSONL.")
        from data_utils import load_split_any  # lazy: only when joining to a dataset
        ds = load_split_any(args.dataset_name, args.split)
        if args.retrieved_field not in ds.column_names:
            raise SystemExit(
                f"Column '{args.retrieved_field}' not in dataset. Available columns: "
                f"{ds.column_names}. Set --retrieved_field to the colleague's column name."
            )
        info = build_gold(ds, args.retrieved_field, args.top_k)
        n_ans = sum(1 for v in info.values() if v["answerable"])
        n_unans = len(info) - n_ans
        print(f"Split '{args.split}' @ top_k={args.top_k}: "
              f"{n_ans} answerable / {n_unans} unanswerable "
              f"(refusal target rate {100*n_unans/max(len(info),1):.1f}%)\n")

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: (file not found)")
            continue
        r = score_file(path, info, from_jsonl=args.from_jsonl, top_k=args.top_k,
                       answer_pos_field=args.answer_pos_field,
                       retrieved_in_rec=args.retrieved_in_rec, gold_field=args.gold_field)
        an = r["a_n"] or 1
        un = r["u_n"] or 1
        em = 100 * r["a_em"] / an
        sub = 100 * r["a_sub"] / an
        met = r["a_meteor"] / an
        over = 100 * r["a_over_refusal"] / an
        abst = 100 * r["u_abstain"] / un
        hall = 100 - abst
        total = r["a_n"] + r["u_n"]
        correct = r["a_em"] + r["u_abstain"]   # exact answers + correct abstentions
        aoa = 100 * correct / max(total, 1)

        print(f"=== {os.path.basename(path)} ===")
        print(f"  ANSWERABLE   (n={r['a_n']}): "
              f"EM {em:.2f}%  sub-EM {sub:.2f}%  METEOR {met:.4f}  "
              f"| over-refusal {over:.2f}%")
        print(f"  UNANSWERABLE (n={r['u_n']}): "
              f"abstention recall {abst:.2f}%  | hallucination {hall:.2f}%")
        print(f"  OVERALL: answer-or-abstain accuracy {aoa:.2f}%  (n={total})")
        if r["missing"]:
            print(f"  ({r['missing']} predictions had a query_id not in the split — skipped)")
        print()


if __name__ == "__main__":
    main()
