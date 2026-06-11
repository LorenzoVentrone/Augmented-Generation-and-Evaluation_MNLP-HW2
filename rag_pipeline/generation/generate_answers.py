#!/usr/bin/env python3
"""Answer generation for HW2 from precomputed retrieval JSONL files.

Reads a dataset split enriched with `retrieved_index` (see
retrieval/build_retrieval_index.py), builds the strategy-specific augmented
prompt for each query, runs greedy generation with a small LM and writes one
JSONL record per query.

Supported prompting strategies:
  - baseline: question only (parametric knowledge of the model);
  - rag:      top-3 retrieved chunks + question (the realistic scenario);
  - oracle:   like rag, but the gold chunk is forced in first position
              (retrieval upper bound, homework rule);
  - rag_ft:   exact replica of the prompt used to fine-tune the LoRA models
              (see src/train.py), with "1." chunk numbering. Use it only with
              the fine-tuned models, which were trained on this format.

Fine-tuned models are loaded as LoRA adapters over their base model
(--is-lora); --base-model-path overrides the base path stored in the adapter
config, which points to a filesystem location that may not be reachable.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATES = {
    "baseline": (
        "Answer the following question briefly and concisely.\n\n"
        "Question: {query}"
    ),
    "rag": (
        "Use the following context to answer the question briefly and concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}"
    ),
    # rag_ft: exact replica of the fine-tuning prompt (src/train.py). The context
    # must be numbered "1.", "2." (see build_context_from_indices numbering="dot").
    "rag_ft": (
        "Given the following information:\n{context}\n\n"
        "Reply to this question: {query}"
    ),
    "oracle": (
        "Use the following context to answer the question briefly and concisely. "
        "The correct answer is definitely contained in the text below:\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers for HW2 from precomputed retrieval JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input split JSONL (test/blind) with candidate_chunks.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Where to write generated results JSONL.")
    parser.add_argument("--model-path", required=True, help="Model path (local dir) or HF id.")
    parser.add_argument("--family", choices=["chat", "alpaca"], default="chat", help="Prompt family (chat template vs Alpaca format).")
    parser.add_argument("--strategy", choices=["baseline", "rag", "oracle", "rag_ft"], default="rag")
    parser.add_argument("--top-k-retrieved", type=int, default=5, help="How many retrieved indices to emit in output.")
    parser.add_argument("--top-k-context", type=int, default=3, help="How many chunks to include in the prompt context (specs mandate 3).")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=-1, help="Use -1 for full split.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--is-lora", action="store_true", help="Load model as LoRA adapter over its base model.")
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Override the base model path for a LoRA adapter (else read from adapter_config.json). "
        "Use this when the base path stored in the adapter is not accessible/offline.",
    )
    parser.add_argument("--retrieval-json", type=Path, default=None, help="Optional JSON/JSONL with query_id -> retrieved_index.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_retrieval_map(path: Optional[Path]) -> Dict[str, List[int]]:
    """Load an external query_id -> retrieved_index map.

    Only needed when the input split does not already carry `retrieved_index`.
    Accepts two formats: a single JSON object keyed by query_id (whose values
    are either retrieval records or plain index lists), or a JSONL file with
    one record per line.
    """
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        # Sniff the format from the first character: '{' means one JSON object
        first = f.read(1)
        f.seek(0)
        if first == "{":
            obj = json.load(f)
            out = {}
            for qid, payload in obj.items():
                if isinstance(payload, dict):
                    out[qid] = payload.get("retrieved_index", [])
                elif isinstance(payload, list):
                    out[qid] = payload
            return out

        out = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            if qid is not None:
                out[qid] = row.get("retrieved_index", [])
        return out


def load_model_and_tokenizer(model_path: str, trust_remote_code: bool, is_lora: bool, base_model_path: Optional[str] = None):
    """Load either a plain HF checkpoint or a LoRA adapter merged over its base."""
    if not is_lora:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            device_map="auto",
        ).eval()
        return model, tokenizer

    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError("peft is required when using --is-lora. Install it in the venv first.") from exc

    # Resolve the base model: prefer the explicit override (local, offline-friendly),
    # otherwise fall back to whatever was stored in the adapter config.
    if base_model_path:
        base = base_model_path
    else:
        peft_config = PeftConfig.from_pretrained(model_path)
        base = peft_config.base_model_name_or_path
    print(f"LoRA adapter: {model_path}")
    print(f"base model  : {base}")

    # The adapter dir is self-contained (ships its own tokenizer), so load the tokenizer
    # from there; fall back to the base model if the adapter lacks tokenizer files.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code)

    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    # Merge the adapter weights into the base for faster inference
    model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload().eval()
    return model, tokenizer


def build_context_from_indices(candidate_chunks: List[str], indices: List[int], top_k_context: int, numbering: str = "bracket") -> str:
    """Join the first `top_k_context` chunks referenced by `indices` into one string.

    numbering="bracket" produces "[1] ..." (rag/oracle prompts);
    numbering="dot" produces "1. ..." — the exact format seen during fine-tuning.
    Out-of-range indices are skipped defensively.
    """
    selected = []
    for idx in indices[:top_k_context]:
        if 0 <= idx < len(candidate_chunks):
            selected.append(candidate_chunks[idx])
    if numbering == "dot":
        return "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(selected))
    return "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(selected))


def build_task_prompt(strategy: str, query: str, candidate_chunks: List[str], retrieved_index: List[int], answer_pos: Optional[int], top_k_context: int) -> str:
    """Build the strategy-specific task prompt for one query."""
    template = PROMPT_TEMPLATES[strategy]

    if strategy == "baseline":
        return template.format(query=query)

    if strategy == "rag":
        context = build_context_from_indices(candidate_chunks, retrieved_index, top_k_context)
        return template.format(query=query, context=context)

    if strategy == "rag_ft":
        # Same retrieved context as rag (retriever order) but with the "1." numbering
        # used during fine-tuning
        context = build_context_from_indices(candidate_chunks, retrieved_index, top_k_context, numbering="dot")
        return template.format(query=query, context=context)

    # oracle: the gold chunk must ALWAYS be in the prompt, in FIRST position
    # (homework rule). The remaining slots keep the retriever order, gold
    # duplicates removed. On the blind split answer_pos is absent or out of
    # range, so the context degrades gracefully to plain RAG.
    oracle_indices = list(retrieved_index)
    if answer_pos is not None and 0 <= answer_pos < len(candidate_chunks):
        oracle_indices = [answer_pos] + [i for i in oracle_indices if i != answer_pos]
    context = build_context_from_indices(candidate_chunks, oracle_indices, top_k_context)
    return template.format(query=query, context=context)


def format_prompt_for_model(task_prompt: str, family: str, tokenizer) -> str:
    """Wrap the task prompt in the input format expected by the model family."""
    if family == "chat":
        msgs = [{"role": "user", "content": task_prompt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Italian Alpaca instruction format (Minerva-style models)
    return (
        "Di seguito e riportata un'istruzione che descrive un'attivita. "
        "Scrivi una risposta che soddisfi adeguatamente la richiesta.\n\n"
        f"### Istruzione:\n{task_prompt}\n\n### Risposta:\n"
    )


@torch.no_grad()
def generate_answer(model, tokenizer, formatted_prompt: str, max_new_tokens: int) -> str:
    """Greedy generation plus cleanup of the raw model output."""
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding, fully reproducible
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Cut everything after typical continuation markers: small models tend to keep
    # going, inventing new questions or section headers
    for stop in ["###", "\n\n", "Risposta:", "Question:", "Answer:"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()
    return answer


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_samples != -1:
        rows = rows[: args.max_samples]

    retrieval_map = load_retrieval_map(args.retrieval_json)

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        is_lora=args.is_lora,
        base_model_path=args.base_model_path,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="Generation"):
            query_id = row["query_id"]
            query = row["query"]
            candidate_chunks = row.get("candidate_chunks", [])
            answer_pos = row.get("answer_pos")

            # Prefer the ranking embedded in the input split; fall back to the
            # external map passed via --retrieval-json
            retrieved_index = row.get("retrieved_index")
            if retrieved_index is None:
                retrieved_index = retrieval_map.get(query_id)

            if retrieved_index is None:
                raise ValueError(
                    "Missing retrieved_index. Provide it in input JSONL or pass --retrieval-json."
                )

            task_prompt = build_task_prompt(
                strategy=args.strategy,
                query=query,
                candidate_chunks=candidate_chunks,
                retrieved_index=retrieved_index,
                answer_pos=answer_pos,
                top_k_context=args.top_k_context,
            )
            augmented_prompt = format_prompt_for_model(task_prompt, args.family, tokenizer)
            generated_answer = generate_answer(model, tokenizer, augmented_prompt, args.max_new_tokens)

            # Superset of the official submission schema: the first four fields are
            # the mandated ones; the rest are internal extras used for evaluation
            # and debugging, stripped away before submission.
            result = {
                "query_id": query_id,
                "retrieved_chunks": retrieved_index[: min(3, len(retrieved_index))],
                "augmented_prompt": augmented_prompt,
                "generated_answer": generated_answer,
                "query": query,
                "retrieved_index": retrieved_index,
                "retrieved_top_k_index": retrieved_index[: args.top_k_retrieved],
                "ground_truth": row.get("short_answer", ""),
                "ground_truth_index": answer_pos,
                "retrieved_chunks_text": [
                    candidate_chunks[i]
                    for i in retrieved_index[: min(3, len(retrieved_index))]
                    if 0 <= i < len(candidate_chunks)
                ],
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
