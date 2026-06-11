import argparse
import math
import os
import random

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for RAG")

    # Model & data
    parser.add_argument("--model_name", type=str,
                        default="/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/assets/model/minerva-3b-instruct-v1.0",
                        help="HF model ID or local model directory")
    parser.add_argument("--dataset_name", type=str,
                        default="/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/assets/dataset/hw-mnlp-2026",
                        help="HF dataset ID or local dataset directory")
    parser.add_argument("--output_dir", type=str, required=True)

    # RAG format
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of chunks in the augmented prompt (oracle mode "
                             "only; in precomputed mode the context is already built)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_mode", type=str, default="oracle",
                        choices=["oracle", "precomputed"],
                        help="oracle: build context on the fly (correct chunk first + "
                             "random distractors) from the raw dataset. precomputed: "
                             "read context_chunks + target already built by "
                             "build_sft_dataset.py (real retriever top-k + refusal "
                             "targets, the Corrective-RAG paradigm). See "
                             "CORRECTIVE_RAG_PLAN.md.")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=1536,
                        help="Ceiling on tokenized sequence length. measure_lengths.py "
                             "reports the real max RAG prompt at 1456 tokens, so 1536 "
                             "covers EVERY example with zero truncation and zero "
                             "filtering. It is only a safety cap: with dynamic padding "
                             "and batch=1 it costs memory only on the longest example, "
                             "not on every step. Any instance still exceeding it is "
                             "dropped (not truncated) in prepare_dataset, so the model "
                             "never trains on corrupted/cut context.")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--eval_split_ratio", type=float, default=0.02,
                        help="Fraction of the train split to reserve for validation")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop training after this many consecutive evals with no "
                             "eval_loss improvement (patience absorbs noisy single-step "
                             "bumps). 0 disables early stopping. The best checkpoint is "
                             "kept regardless via load_best_model_at_end.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="Minimum eval_loss decrease to count as an improvement; "
                             "ignores fluctuations smaller than this.")

    # Quantization
    parser.add_argument("--no_qlora", action="store_true",
                        help="Disable 4-bit quantization (use full LoRA instead)")

    # HPC / misc
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Override HF_HOME/cache dir (useful on scratch storage)")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preparation


# Fallback chat template for models that ship without one (e.g. Minerva base
# checkpoints saved without tokenizer_config.json chat_template field).
# Uses the Mistral [INST] format which matches Minerva's training convention.
# Includes {{ bos_token }} explicitly so we can tokenize with
# add_special_tokens=False (avoiding a doubled BOS) while still getting a BOS.

_FALLBACK_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "[INST] {{ message['content'] }}\n[/INST]"
    "{% elif message['role'] == 'assistant' %}"
    " {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def build_example(example: dict, tokenizer, top_k: int, rng: random.Random,
                  max_seq_length: int) -> dict:
    """
    Pre-tokenize one example into input_ids + labels with the prompt region masked.

    Oracle setup: correct chunk is always first, then top_k-1 randomly sampled
    chunks from the remaining candidates.

    Masking is done by PREFIX LENGTH, not by searching for a response-template
    token subsequence. We render the prompt alone (up to the generation point)
    and the full prompt+answer, tokenize both, and set labels[:len(prompt_ids)]
    to -100. This is robust to SentencePiece's context-dependent tokenization:
    unlike DataCollatorForCompletionOnlyLM (which silently drops an instance when
    the marker ids don't match in-context), a boundary off-by-one here only
    mislabels at most one token instead of discarding the whole example. The EOS
    token falls in the unmasked region, so the model also learns where to stop.

    Returns {"input_ids", "labels", "attention_mask"}; the lists are empty if the
    example is longer than max_seq_length (dropped, never truncated — filtered out
    in prepare_dataset). Empty (not absent) keeps a consistent map() schema.
    """
    correct_chunk: str = example["answer"]
    candidates: list[str] = list(example["candidate_chunks"])

    other_chunks = [c for c in candidates if c != correct_chunk]
    rng.shuffle(other_chunks)
    context_chunks = [correct_chunk] + other_chunks[: top_k - 1]

    context = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(context_chunks))

    short_answers: list[str] = example["short_answer"]
    target = short_answers[0] if short_answers else correct_chunk[:100]

    user_content = (
        f"Given the following information:\n{context}\n\n"
        f"Reply to this question: {example['query']}"
    )

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    # Template already emits BOS/EOS textually, add_special_tokens=False so we don't double them.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Drop, never truncate: a longer sequence would force cutting context or the answer.
    if len(full_ids) > max_seq_length:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(n_prompt):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def build_example_precomputed(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    The context chunks are the real retriever top-k (in retrieved order, NOT reordered to put the
    correct chunk first), and the target may be a refusal phrase when the retriever
    missed the correct chunk.

    Same robust prefix-length masking and same drop-don't-truncate policy as
    build_example.
    """
    context_chunks: list[str] = list(example["context_chunks"])
    context = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(context_chunks))
    target: str = example["target"]

    user_content = (
        f"Given the following information:\n{context}\n\n"
        f"Reply to this question: {example['query']}"
    )

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_length:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(n_prompt):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def prepare_dataset(tokenizer, dataset_name: str, top_k: int, seed: int, eval_split_ratio: float,
                    max_seq_length: int, data_mode: str = "oracle"):
    rng = random.Random(seed)

    if os.path.isdir(dataset_name) and os.path.exists(os.path.join(dataset_name, "dataset_dict.json")):
        raw = load_from_disk(dataset_name)
    else:
        raw = load_dataset(dataset_name)

    # precomputed mode: build_sft_dataset.py already produced train/eval splits with context_chunks + target.
    if data_mode == "precomputed":
        train_data = raw["train"]
        eval_data = raw["eval"] if "eval" in raw else None

        def _format_pre(example):
            return build_example_precomputed(example, tokenizer, max_seq_length)

        def _encode_pre(ds, name):
            n_before = len(ds)
            ds = ds.map(_format_pre, remove_columns=ds.column_names, load_from_cache_file=False)
            ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0, load_from_cache_file=False)
            n_kept = len(ds)
            print(f"{name}: kept {n_kept}/{n_before} instances "
                  f"(dropped {n_before - n_kept} over max_seq_length={max_seq_length}; "
                  f"no truncation) [precomputed/corrective mode]")
            return ds

        train_data = _encode_pre(train_data, "Train")
        if eval_data is not None:
            eval_data = _encode_pre(eval_data, "Eval")
        return train_data, eval_data

    train_data = raw["train"]
    eval_data = None
    if eval_split_ratio > 0:
        split = train_data.train_test_split(test_size=eval_split_ratio, seed=seed)
        train_data = split["train"]
        eval_data = split["test"]

    def _format(example):
        return build_example(example, tokenizer, top_k, rng, max_seq_length)

    def _encode_and_filter(ds, name):
        n_before = len(ds)
        ds = ds.map(_format, remove_columns=ds.column_names, load_from_cache_file=False)
        # build_example returns {} (no input_ids) for over-length examples → drop.
        ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0, load_from_cache_file=False)
        n_kept = len(ds)
        dropped = n_before - n_kept
        print(f"{name}: kept {n_kept}/{n_before} instances "
              f"(dropped {dropped} over max_seq_length={max_seq_length}; "
              f"no truncation, no chunk corruption)")
        return ds

    train_data = _encode_and_filter(train_data, "Train")
    if eval_data is not None:
        eval_data = _encode_and_filter(eval_data, "Eval")

    return train_data, eval_data


# ---------------------------------------------------------------------------
# Model & tokenizer

def load_model_and_tokenizer(args):
    cache_dir = args.hf_cache_dir  # None → default HF cache

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # required for SFT causal loss masking

    
    native_template = getattr(tokenizer, "chat_template", None)
    if not native_template:
        print("No native chat_template found on the tokenizer — using fallback "
              "[INST]...[/INST] template.")
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE
    else:
        print("Using tokenizer's native chat_template:")
    print(tokenizer.chat_template)

    bnb_config = None
    if not args.no_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.config.use_cache = False 

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_lora_config(args) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)

    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        tokenizer,
        args.dataset_name,
        args.top_k,
        args.seed,
        args.eval_split_ratio,
        args.max_seq_length,
        data_mode=args.data_mode,
    )
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Validation examples: {len(eval_dataset)}")

    # Sanity-print one pre-tokenized example
    sample = train_dataset[0]
    answer_ids = [t for t in sample["labels"] if t != -100]
    print("Sample full text:\n", tokenizer.decode(sample["input_ids"])[:600])
    print("Sample answer region (unmasked labels):\n", tokenizer.decode(answer_ids))

    # Apply LoRA. 
    lora_config = build_lora_config(args)
    if not args.no_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Pads input_ids with pad_token_id and labels with -100, builds attention masks.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding=True,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_train_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = max(1, int(total_train_steps * args.warmup_ratio))

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_total_limit=5,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Early stopping
    callbacks = []
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
        print(f"Early stopping enabled: patience={args.early_stopping_patience}, "
              f"threshold={args.early_stopping_threshold} (metric: eval_loss)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}/final_adapter")
    trainer.model.save_pretrained(f"{args.output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/final_adapter")

    print("Done.")


if __name__ == "__main__":
    main()
