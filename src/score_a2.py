"""
A2 additional evaluation metrics over an inference JSONL — no re-inference.

Reads the teammate's per-example records:
  {"query_id", "retrieved_chunks", "augmented_prompt", "generated_answer",
   "ground_truth" (short_answer, str or list), "ground_truth_index",
   "retrieved_chunks_text" (list of the top-k chunk texts)}

Computes two complementary alternative metrics (A2):

  1. BERTScore (P/R/F1)  — semantic similarity of `generated_answer` vs `ground_truth`,
     catching paraphrases/synonyms that Exact Match misses. Reference-based.

  2. NLI faithfulness    — does the retrieved context ENTAIL the generated answer?
     For each answer we run an NLI model over (premise = each retrieved chunk,
     hypothesis = generated_answer) and take the MAX entailment probability across
     chunks (an answer is faithful if supported by at least one retrieved passage).
     This is the groundedness / anti-hallucination signal of A2. Refusal answers are
     skipped (an abstention is not a factual claim to verify); they stay covered by the
     corrective abstention metrics.

Everything is post-hoc: it runs scoring models over the saved answers, it does NOT
re-run the RAG generation. Uses GPU if available, else CPU.

Usage:
  python score_a2.py --inputs results/foo.jsonl results/bar.jsonl
  # optional: --output_dir results/a2  (writes per-example scores)
  #           --nli_model <hf id or local dir>  --bertscore_model <hf id or local dir>
  #           --no_bertscore / --no_nli  to run only one
"""

import argparse
import json
import os

from refusal_templates import is_refusal_text


def parse_args():
    p = argparse.ArgumentParser(description="A2 metrics (BERTScore + NLI faithfulness) over inference JSONL")
    p.add_argument("--inputs", type=str, nargs="+", required=True,
                   help="One or more inference JSONL files (teammate's format)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="If set, write per-example scores alongside the inputs")
    p.add_argument("--nli_model", type=str,
                   default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                   help="NLI model (entailment classifier). Pre-download on the login node.")
    p.add_argument("--bertscore_model", type=str, default=None,
                   help="Override the BERTScore backbone (default: lang='en' → roberta-large)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hypothesis", type=str, default="answer", choices=["answer", "qa"],
                   help="NLI hypothesis: 'answer' = bare generated answer (conservative on "
                        "short spans); 'qa' = query + answer (a fuller proposition → "
                        "sharper, more discriminative entailment). 'qa' extracts the query "
                        "from augmented_prompt.")
    p.add_argument("--no_rescale", action="store_true",
                   help="Disable BERTScore baseline rescaling. Rescaling spreads scores "
                        "to ~[0,1] vs a random baseline but compresses very short answers "
                        "near 0; --no_rescale gives the raw F1 (~0.85-0.92), comparable "
                        "to the literature and easier to read on short spans.")
    p.add_argument("--no_bertscore", action="store_true")
    p.add_argument("--no_nli", action="store_true")
    return p.parse_args()


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def golds_of(rec):
    """ground_truth may be a str or a list[str] — always return a non-empty list."""
    gt = rec.get("ground_truth", "")
    if isinstance(gt, list):
        golds = [str(g) for g in gt if str(g).strip()]
    else:
        golds = [str(gt)] if str(gt).strip() else []
    return golds or [""]


def chunks_of(rec):
    """The retrieved chunk texts that were actually in the prompt."""
    ch = rec.get("retrieved_chunks_text") or []
    return [c for c in ch if isinstance(c, str) and c.strip()]


# common chat-template / format tokens that may trail the query in augmented_prompt
_QUERY_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "[/INST]", "</s>", "<|eot_id|>")


def extract_query(augmented_prompt):
    """Pull the question out of the prompt, which ends with
    'Reply to this question: <query>'. Returns None if not found."""
    if not augmented_prompt:
        return None
    marker = "Reply to this question:"
    idx = augmented_prompt.rfind(marker)
    if idx == -1:
        return None
    q = augmented_prompt[idx + len(marker):].strip()
    q = q.split("\n", 1)[0]                     # query is a single line
    for tok in _QUERY_STOP_TOKENS:              # drop trailing template tokens
        q = q.split(tok, 1)[0]
    return q.strip() or None


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(records, model_type, batch_size, rescale=True):
    """Return list of F1 (and P/R) per record, generated_answer vs ground_truth."""
    from bert_score import score as bertscore_score

    cands = [(r.get("generated_answer") or " ").strip() or " " for r in records]
    refs = [golds_of(r) for r in records]  # list of lists (multi-ref → max match)

    if model_type:
        # explicit model: baseline rescaling not available for arbitrary backbones
        kwargs = dict(model_type=model_type, batch_size=batch_size, verbose=False)
    else:
        kwargs = dict(lang="en", rescale_with_baseline=rescale, batch_size=batch_size,
                      verbose=False)
    P, R, F1 = bertscore_score(cands, refs, **kwargs)
    return P.tolist(), R.tolist(), F1.tolist()


# ---------------------------------------------------------------------------
# NLI faithfulness
# ---------------------------------------------------------------------------

def _entailment_index(model):
    """Find the logit index for the 'entailment' class from the model config."""
    id2label = getattr(model.config, "id2label", {}) or {}
    for idx, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(idx)
    # MNLI default order is [contradiction, neutral, entailment]
    return 2


def compute_nli_faithfulness(records, nli_model, batch_size, hypothesis_mode="answer"):
    """For each NON-refusal record, max entailment prob of (chunk -> hypothesis) across
    the retrieved chunks. The hypothesis is the bare answer ('answer' mode) or the
    query+answer ('qa' mode). Returns dict idx -> faithfulness (None if refusal/no ctx)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(device).eval()
    ent_idx = _entailment_index(model)

    # build (premise, hypothesis) pairs, remembering which record each belongs to
    pairs, owner = [], []
    scores = {}  # idx in records -> faithfulness or None
    n_qa_fallback = 0
    for i, r in enumerate(records):
        ans = (r.get("generated_answer") or "").strip()
        if not ans or is_refusal_text(ans):
            scores[i] = None
            continue
        chunks = chunks_of(r)
        if not chunks:
            scores[i] = None
            continue
        hyp = ans
        if hypothesis_mode == "qa":
            q = extract_query(r.get("augmented_prompt"))
            if q:
                hyp = f"{q} {ans}"
            else:
                n_qa_fallback += 1  # query not found → fall back to bare answer
        for c in chunks:
            pairs.append((c, hyp))
            owner.append(i)
    if hypothesis_mode == "qa" and n_qa_fallback:
        print(f"  [qa] query not extractable for {n_qa_fallback} records "
              f"→ used bare answer for those")

    # batched NLI forward passes; accumulate max entailment per record
    per_record_max = {}
    with torch.no_grad():
        for s in range(0, len(pairs), batch_size):
            batch = pairs[s:s + batch_size]
            prem = [p for p, _ in batch]
            hyp = [h for _, h in batch]
            enc = tok(prem, hyp, return_tensors="pt", truncation=True,
                      max_length=512, padding=True).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, ent_idx].tolist()
            for j, prob in enumerate(probs):
                i = owner[s + j]
                per_record_max[i] = max(per_record_max.get(i, 0.0), prob)

    for i in per_record_max:
        scores[i] = per_record_max[i]
    return scores


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: (file not found)\n")
            continue
        records = load_records(path)
        n = len(records)
        n_refusal = sum(1 for r in records
                        if is_refusal_text((r.get("generated_answer") or "").strip()))

        print(f"=== {os.path.basename(path)}  (n={n}, refusals={n_refusal}) ===")

        bs_f1 = bs_p = bs_r = None
        if not args.no_bertscore:
            P, R, F1 = compute_bertscore(records, args.bertscore_model, args.batch_size,
                                         rescale=not args.no_rescale)
            bs_p = sum(P) / len(P)
            bs_r = sum(R) / len(R)
            bs_f1 = sum(F1) / len(F1)
            tag = "raw" if args.no_rescale else "rescaled"
            print(f"  BERTScore [{tag}]  P={bs_p:.4f}  R={bs_r:.4f}  F1={bs_f1:.4f}  "
                  f"(vs ground_truth, all {n} examples)")

        faith = None
        if not args.no_nli:
            faith = compute_nli_faithfulness(records, args.nli_model, args.batch_size,
                                             hypothesis_mode=args.hypothesis)
            vals = [v for v in faith.values() if v is not None]
            if vals:
                avg = sum(vals) / len(vals)
                grounded = sum(1 for v in vals if v >= 0.5)
                print(f"  NLI faithfulness [{args.hypothesis}]  avg={avg:.4f}  "
                      f"grounded(≥0.5)={grounded}/{len(vals)} ({100*grounded/len(vals):.1f}%)  "
                      f"[over {len(vals)} non-refusal answers]")
            else:
                print("  NLI faithfulness: no scorable (non-refusal, with-context) answers")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out = os.path.join(args.output_dir,
                               os.path.basename(path).replace(".jsonl", "-a2.jsonl"))
            with open(out, "w", encoding="utf-8") as f:
                for i, r in enumerate(records):
                    rec = {"query_id": r.get("query_id")}
                    if not args.no_bertscore:
                        rec["bertscore_f1"] = F1[i]
                    if not args.no_nli:
                        rec["nli_faithfulness"] = faith.get(i)
                    f.write(json.dumps(rec) + "\n")
            print(f"  per-example scores -> {out}")
        print()


if __name__ == "__main__":
    main()
