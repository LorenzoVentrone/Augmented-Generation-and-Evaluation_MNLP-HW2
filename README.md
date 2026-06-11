# Augmented Generation and Evaluation — MNLP 2026 (HW2)

Retrieval-Augmented Generation (RAG) pipeline with **QLoRA fine-tuning** of small,
open-weight language models, built for Homework 2 of the *Multilingual Natural
Language Processing* course (Sapienza University of Rome, Prof. Roberto Navigli).

Given a query and the chunks retrieved for it, a small LM must generate a short,
factually correct answer. We fine-tune the generators with a **corrective** objective
that trains on the retriever's *real* top-k and teaches the model **when to abstain**
if the answer was not retrieved.

> **Where things ran.** Fine-tuning and full-scale (2,000-query) inference were run on
> the CINECA Leonardo HPC cluster; this repository holds those scripts. The submitted
> Colab notebook runs the same inference pipeline end-to-end on a small sample (~10
> queries) and contains no fine-tuning code. Generated outputs (JSONL) are shared via
> Google Drive.

---

## Models

| Role | Model | Variants |
|---|---|---|
| Generator | `Qwen2.5-0.5B-Instruct` | off-the-shelf + corrective fine-tuned |
| Generator | `SmolLM2-360M-Instruct` | off-the-shelf + corrective fine-tuned |
| LLM judge | `Qwen2.5-1.5B-Instruct` | evaluation only (never generates answers) |

**Retriever (from HW1):** a fine-tuned `bge-m3` bi-encoder followed by a
`bge-reranker-v2-m3` cross-encoder produce the top-k `retrieved_index` used here.

---

## Highlights

- **Three inference settings:** `baseline` (query only), `rag` (top-k retrieved
  chunks), `oracle` (gold chunk placed first, the generation ceiling).
- **Corrective fine-tuning:** training on the retriever's actual top-k, with an explicit
  **refusal** target whenever the gold chunk was not retrieved, so the model abstains
  instead of hallucinating (additional requirement **A4**; the abstention behaviour is
  evaluated as a faithfulness signal).
- **Robust completion-only masking:** the prompt is masked by prefix length, so no
  training instance is ever silently dropped, with **no truncation** of context.
- **Evaluation:** EM, sub-EM, METEOR, plus **A2** semantic metrics (BERTScore, NLI
  faithfulness) and a corrective split into *answerable* / *unanswerable*
  (abstention recall and hallucination rate), with an LLM judge and Cohen's kappa.

---

## Repository structure

```
src/
├── train.py               # QLoRA fine-tuning (--data_mode oracle | precomputed)
├── build_sft_dataset.py   # build the corrective SFT dataset from the retriever top-k
├── retriever_diag.py      # retriever recall@k diagnostic (to choose top_k)
├── data_utils.py          # split loader (save_to_disk dir / jsonl-parquet / HF id)
├── refusal_templates.py   # ~50 abstention phrases + a refusal detector for scoring
├── rag_metrics.py         # EM / sub-EM / METEOR with SQuAD-style normalization
├── evaluate.py            # baseline/RAG/oracle inference + metrics
├── score_jsonl.py         # re-score EM/sub-EM/METEOR from a predictions JSONL
├── score_corrective.py    # answerable/unanswerable split: abstention + hallucination
├── score_a2.py            # A2: BERTScore + NLI faithfulness
└── scripts/
    ├── slurm_job_small.sh # single-GPU fine-tuning: <TAG> <MODEL_DIR> [DATA_MODE]
    └── run_all_eval.sh    # full evaluation (baseline/RAG/oracle) for one model
```

---

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

## Dataset

Loaded from the Hugging Face Hub:

```python
from datasets import load_dataset
ds = load_dataset("sapienzanlp-course-materials/hw-mnlp-2026")
```

| field | meaning |
|---|---|
| `query` / `query_id` | the question and its id |
| `candidate_chunks` | the pool of candidate passages |
| `answer` / `answer_pos` | the gold passage and its index in `candidate_chunks` |
| `short_answer` | list of acceptable short answers (the generation target) |
| `retrieved_index` | the HW1 retriever's top-k `candidate_chunks` indices |

Splits: **train/dev** (8,000), **test** (2,000), **blind** (1,322, no answers).

---

## Corrective dataset rule (strict)

The dataset has exactly one correct chunk, so the answerable criterion is exact:

```
answerable    if   answer_pos in retrieved_index[:top_k]   ->  target = short_answer
unanswerable  otherwise                                    ->  target = a refusal phrase
```

With `top_k = 3` the retriever's recall@3 is ~88%, giving ~12% refusal examples.

---

## Usage

**1. (optional) Diagnose the retriever to choose `top_k`:**
```bash
python src/retriever_diag.py --dataset_name <dataset_with_retrieved_index> \
    --split train --retrieved_field retrieved_index --ks 1 3 5 10
```

**2. Build the corrective SFT dataset:**
```bash
python src/build_sft_dataset.py --dataset_name <dataset_with_retrieved_index> \
    --split train --retrieved_field retrieved_index --top_k 3 \
    --out_dir assets/dataset/hw-mnlp-2026-corrective
```

**3. Fine-tune (single GPU, parametric SLURM job):**
```bash
# corrective fine-tuning on the dataset from step 2
sbatch src/scripts/slurm_job_small.sh qwen-0.5b /path/to/Qwen2.5-0.5B-Instruct precomputed
# (the default DATA_MODE is `oracle`, which builds an oracle context on the fly)
```
`train.py` is model-agnostic: it uses the model's native chat template when available
and falls back to an `[INST]` template otherwise.

**4. Inference + evaluation.** Run all metrics for one model over the three settings:
```bash
sbatch src/scripts/run_all_eval.sh <MODEL_TAG> <DIR_WITH_JSONL>
```
or score individual prediction files:
```bash
python src/score_jsonl.py     --gold_field ground_truth --inputs preds.jsonl   # EM/sub-EM/METEOR
python src/score_corrective.py --from_jsonl --top_k 3   --inputs preds_rag.jsonl # abstention
python src/score_a2.py        --inputs preds.jsonl                              # BERTScore + NLI
```

---

## Results (test split, 2,000 queries)

Corrective fine-tuning lifts EM by more than 20x while improving every other metric.

| Model | Setting | EM | sub-EM | METEOR | BERTScore | Faith. | Abst. |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | RAG | 1.35 | 44.90 | 34.76 | 0.847 | 0.676 | 0.00 |
| **Qwen2.5-0.5B-ft** | RAG | **40.25** | 51.75 | 49.02 | 0.915 | 0.850 | 20.73 |
| SmolLM2-360M | RAG | 1.90 | 41.80 | 32.90 | 0.845 | 0.638 | 0.41 |
| **SmolLM2-360M-ft** | RAG | **41.85** | 47.10 | 45.96 | 0.919 | 0.849 | 18.70 |

EM/sub-EM/METEOR are percentages; BERTScore is raw F1; Faithfulness is mean NLI
entailment over non-refusal answers; Abst. is abstention recall on the unanswerable
subset. The full table (all settings, baseline/oracle included) is in the report.

LLM judge (Qwen2.5-1.5B) on the best system marks 71.5% of answers correct; the judge
agrees with the human consensus at Cohen's kappa = 0.86, and the two annotators agree
at kappa = 0.93.

---

## License

Released under the [MIT License](LICENSE) © 2026 Lorenzo Ventrone.
