# RAG pipeline (CINECA Leonardo)

Retrieval and answer-generation code used to produce the HW2 outputs on the CINECA
Leonardo cluster (SLURM). The fine-tuning and evaluation code lives in [`src/`](../src);
the Colab notebook consumes the JSONL files produced here.

## Layout

```
rag_pipeline/
├── setup/
│   ├── download_dataset.sh        # HF dataset -> scratch (one JSONL per split)
│   └── download_model.sh          # HF id or Google Drive folder -> scratch/models
├── retrieval/
│   ├── build_retrieval_index.py   # bi-encoder + reranker, adds retrieved_index
│   └── run_retrieval.sh           # SLURM launcher
└── generation/
    ├── generate_answers.py        # prompt building + greedy generation
    └── run_generation_array.sh    # SLURM array launcher (models x splits x strategies)
```

## Environment

- Deployed on Leonardo at `$BASE/src/rag_pipeline` with
  `BASE=/leonardo/home/userexternal/ltam0000` (the `#SBATCH` log paths assume this
  location; adjust them if you deploy elsewhere).
- Python venv at `$BASE/.venv`, built on a **login node** (`pip install -r requirements.txt`).
- Compute nodes have **no network access**: datasets and models must be downloaded
  beforehand with the `setup/` scripts, and every job runs with `TRANSFORMERS_OFFLINE=1`
  and the HF caches pointed at scratch.
- SLURM account `IscrC_MNLP26`, partition `boost_usr_prod`, 1 GPU per job.

## 1. Setup (login node)

```bash
./setup/download_dataset.sh
./setup/download_model.sh 'https://drive.google.com/drive/folders/<HW1-retriever>' bge-m3-mnrl-hardneg
./setup/download_model.sh BAAI/bge-reranker-v2-m3 bge-reranker-v2-m3
./setup/download_model.sh HuggingFaceTB/SmolLM2-360M-Instruct smollm2-360m-instruct
./setup/download_model.sh Qwen/Qwen2.5-0.5B-Instruct qwen2.5-0.5b-instruct
```

## 2. Retrieval

Enriches each split with `retrieved_index`: the candidate chunks ranked by the HW1
bi-encoder (`bge-m3-mnrl-hardneg`, cosine, top-10 pool) and rescored by the
`bge-reranker-v2-m3` cross-encoder. Downstream, RAG prompts use the first 3 indices
(`top_k = 3` is mandated by the homework specs).

```bash
sbatch retrieval/run_retrieval.sh                 # default: train,test
SPLITS=blind sbatch retrieval/run_retrieval.sh    # blind split
```

## 3. Generation

One SLURM array task per (model, split, strategy) combination. Print the task map
first, then submit the matrix:

```bash
DRY_RUN=1 ./generation/run_generation_array.sh    # prints "[i] model=... split=... strategy=..."
sbatch --array=0-9 generation/run_generation_array.sh
```

Useful overrides (environment variables):

```bash
SPLITS_CSV=test,blind sbatch --array=0-... generation/run_generation_array.sh
MAX_SAMPLES=50 sbatch --array=0-0 generation/run_generation_array.sh   # smoke test
QWEN_LORA= SMOL_LORA= sbatch --array=0-5 generation/run_generation_array.sh  # base models only
```

### Prompting strategies

| Strategy   | Context                          | Used with                         |
|------------|----------------------------------|-----------------------------------|
| `baseline` | none (question only)             | base models (parametric knowledge)|
| `rag`      | top-3 retrieved chunks, `[1]`    | base models (realistic scenario)  |
| `oracle`   | gold chunk forced first, `[1]`   | base models (upper bound)         |
| `rag_ft`   | top-3 retrieved chunks, `1.`     | fine-tuned models ONLY            |

`rag_ft` replicates the exact fine-tuning prompt of [`src/train.py`](../src/train.py)
("Given the following information... Reply to this question:"), so the fine-tuned
models see at inference time the same format they were trained on. Running it on
base models, or other strategies on fine-tuned models, is not meaningful.

### Fine-tuned models (A4)

The QLoRA adapters live under `/leonardo_work/IscrC_MNLP26/Funtori/models/*/final_adapter`.
Their `adapter_config.json` stores a base-model path inside another user's home,
unreadable from this account: the launcher overrides it with the local base copy via
`--base-model-path` (variables `SMOL_LORA_BASE` / `QWEN_LORA_BASE`).

## Output

One JSONL per (model, split, strategy) under
`$OUTPUT_ROOT/<model>/<split>_<strategy>.jsonl`. Each record is a superset of the
official submission schema:

```json
{
  "query_id": "...",
  "retrieved_chunks": [0, 23, 10],
  "augmented_prompt": "...",
  "generated_answer": "...",
  "query": "...",
  "retrieved_index": [...],
  "retrieved_top_k_index": [...],
  "ground_truth": ["..."],
  "ground_truth_index": 0,
  "retrieved_chunks_text": ["..."]
}
```

The first four fields are the ones required by the specs; the rest are internal
extras (evaluation, debugging) and are stripped away before submission.
