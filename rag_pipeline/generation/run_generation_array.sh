#!/bin/bash
#SBATCH --job-name=mnlp_generation
#SBATCH --account=IscrC_MNLP26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/ltam0000/src/rag_pipeline/generation/logs/generation_%A_%a.out
#SBATCH --error=/leonardo/home/userexternal/ltam0000/src/rag_pipeline/generation/logs/generation_%A_%a.err

# SLURM array launcher for generate_answers.py.
# The job matrix is MODELS x SPLITS x STRATEGIES: one array task per combination.
# Run without sbatch (or with DRY_RUN=1) to print the index -> task map first:
#   DRY_RUN=1 ./run_generation_array.sh
# then submit the whole matrix in one shot:
#   sbatch --array=0-<N-1> run_generation_array.sh

set -euo pipefail

BASE=/leonardo/home/userexternal/ltam0000
SCRATCH=$BASE/my_scratch/MNLP_FUNTORI_HWII/scratch
PY_SCRIPT=$BASE/src/rag_pipeline/generation/generate_answers.py

INPUT_ROOT=${INPUT_ROOT:-$BASE/my_scratch/MNLP_FUNTORI_HWII/output}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BASE/my_scratch/MNLP_FUNTORI_HWII/output/generated_answers}
TOP_K_RETRIEVED=${TOP_K_RETRIEVED:-5}
TOP_K_CONTEXT=${TOP_K_CONTEXT:-3}     # the specs mandate 3 chunks in the prompt
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-80}
MAX_SAMPLES=${MAX_SAMPLES:--1}        # -1 = full split; small values for smoke tests

# Splits and strategies to run for the BASE models (comma-separated).
SPLITS_CSV=${SPLITS_CSV:-test}
# SPLITS_CSV=${SPLITS_CSV:-test,blind}
STRATEGIES_CSV=${STRATEGIES_CSV:-baseline,rag,oracle}

# Fine-tuned models only run the prompt format they were trained on: the SFT
# (src/train.py) uses the "Given the following information... Reply to this
# question:" prompt with "1." numbering, replicated by the rag_ft strategy.
# Running baseline/oracle on them would be meaningless.
FT_STRATEGIES=${FT_STRATEGIES:-rag_ft}

# --- Base (instruct) models, local copies under scratch ---
SMOL_MODEL=${SMOL_MODEL:-$SCRATCH/models/smollm2-360m-instruct}
QWEN_MODEL=${QWEN_MODEL:-$SCRATCH/models/qwen2.5-0.5b-instruct}
MINERVA_MODEL=${MINERVA_MODEL:-}

# --- Fine-tuned models: LoRA adapters [A4] ---
# The "precomputed" adapters live in the final_adapter/ subfolder. Their
# adapter_config.json stores a base path inside another user's home, which is
# not readable from this account: the local base below overrides it.
# To exclude a model, pass its variable empty, e.g.:  QWEN_LORA= sbatch ...
FT_DIR=${FT_DIR:-/leonardo_work/IscrC_MNLP26/Funtori/models}
SMOL_LORA=${SMOL_LORA:-$FT_DIR/smollm2-360m_rag_ft_precomputed/final_adapter}
QWEN_LORA=${QWEN_LORA:-$FT_DIR/qwen-0.5b_rag_ft_precomputed/final_adapter}
# Minerva is currently excluded. To re-enable it:
#   MINERVA_LORA=$FT_DIR/minerva-1b_rag_ft_precomputed/final_adapter MINERVA_MODEL=<base> sbatch ...
MINERVA_LORA=${MINERVA_LORA:-}

# Local base model for each LoRA adapter (overrides base_model_name_or_path)
SMOL_LORA_BASE=${SMOL_LORA_BASE:-$SMOL_MODEL}
QWEN_LORA_BASE=${QWEN_LORA_BASE:-$QWEN_MODEL}
MINERVA_LORA_BASE=${MINERVA_LORA_BASE:-$MINERVA_MODEL}

mkdir -p "$OUTPUT_ROOT" "$BASE/src/rag_pipeline/generation/logs"

module purge
module load profile/deeplrn python/3.11.7 cuda/12.1
source "$BASE/.venv/bin/activate"

# Compute nodes have no network access: everything must already be in the local
# HF cache / scratch (see setup/), hence TRANSFORMERS_OFFLINE=1.
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$BASE/my_scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=$BASE/my_scratch/hf_cache
export HF_DATASETS_CACHE=$BASE/my_scratch/hf_cache
export TRANSFORMERS_OFFLINE=1

# Model registry. Entry format: "name|path|family|flags|base|strats"
#   flags  = --trust-remote-code and/or --is-lora (space-separated)
#   base   = LOCAL base model path for LoRA adapters (empty for full models)
#   strats = per-model strategy CSV (empty = use the global STRATEGIES_CSV).
#            Fine-tuned entries set it to FT_STRATEGIES; base models leave it empty.
MODELS=(
  "smollm2|$SMOL_MODEL|chat|--trust-remote-code||"
  "qwen05b|$QWEN_MODEL|chat|||"
)
if [[ -n "$MINERVA_MODEL" ]]; then
  MODELS+=("minerva|$MINERVA_MODEL|alpaca|--trust-remote-code||")
fi
# LoRA fine-tuned models (added only when their adapter path is set)
if [[ -n "$SMOL_LORA" ]]; then
  MODELS+=("smollm2-ft|$SMOL_LORA|chat|--trust-remote-code --is-lora|$SMOL_LORA_BASE|$FT_STRATEGIES")
fi
if [[ -n "$QWEN_LORA" ]]; then
  MODELS+=("qwen05b-ft|$QWEN_LORA|chat|--is-lora|$QWEN_LORA_BASE|$FT_STRATEGIES")
fi
if [[ -n "$MINERVA_LORA" ]]; then
  MODELS+=("minerva-ft|$MINERVA_LORA|alpaca|--trust-remote-code --is-lora|$MINERVA_LORA_BASE|$FT_STRATEGIES")
fi

IFS=',' read -r -a SPLITS <<< "$SPLITS_CSV"
IFS=',' read -r -a STRATEGIES <<< "$STRATEGIES_CSV"

# Expand the MODELS x SPLITS x STRATEGIES matrix into a flat task list; each
# array task picks its entry by SLURM_ARRAY_TASK_ID.
TASKS=()
for m in "${MODELS[@]}"; do
  IFS='|' read -r name path family flags base strats <<< "$m"
  # Effective strategies for this model: per-model list when set, global otherwise
  if [[ -n "$strats" ]]; then
    IFS=',' read -r -a MODEL_STRATS <<< "$strats"
  else
    MODEL_STRATS=("${STRATEGIES[@]}")
  fi
  for split in "${SPLITS[@]}"; do
    for strategy in "${MODEL_STRATS[@]}"; do
      in_file="$INPUT_ROOT/$split/$split.jsonl"
      out_file="$OUTPUT_ROOT/$name/${split}_${strategy}.jsonl"
      TASKS+=("$name|$path|$family|$flags|$strategy|$split|$in_file|$out_file|$base")
    done
  done
done

# Outside an array job, print the index -> task map (useful to pick --array).
# DRY_RUN=1 prints the list and exits without running anything.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Total tasks: ${#TASKS[@]}  (submit with --array=0-$(( ${#TASKS[@]} - 1 )))"
  for i in "${!TASKS[@]}"; do
    IFS='|' read -r n p f fl s sp _ _ b <<< "${TASKS[$i]}"
    echo "  [$i] model=$n split=$sp strategy=$s flags='$fl' base='$b'"
  done
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
  fi
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#TASKS[@]} )); then
  echo "TASK_ID out of range: $TASK_ID (valid: 0..$(( ${#TASKS[@]} - 1 )))"
  exit 1
fi

IFS='|' read -r MODEL_NAME MODEL_PATH FAMILY FLAGS STRATEGY SPLIT INPUT_FILE OUTPUT_FILE BASE_MODEL <<< "${TASKS[$TASK_ID]}"

# For LoRA models, pass the local base as an override (when set)
BASE_FLAG=()
if [[ -n "$BASE_MODEL" ]]; then
  BASE_FLAG=(--base-model-path "$BASE_MODEL")
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: missing input file $INPUT_FILE"
  echo "Run the retrieval step first so each record has retrieved_index."
  exit 1
fi

echo "Job ID          : ${SLURM_JOB_ID:-N/A}"
echo "Task ID         : $TASK_ID"
echo "Model           : $MODEL_NAME"
echo "Model path      : $MODEL_PATH"
echo "Family          : $FAMILY"
echo "Flags           : $FLAGS"
echo "Base override   : ${BASE_MODEL:-<from adapter config>}"
echo "Strategy        : $STRATEGY"
echo "Split           : $SPLIT"
echo "Input           : $INPUT_FILE"
echo "Output          : $OUTPUT_FILE"
echo "Top-k retrieved : $TOP_K_RETRIEVED"
echo "Top-k context   : $TOP_K_CONTEXT"
echo "Max new tokens  : $MAX_NEW_TOKENS"
echo "Max samples     : $MAX_SAMPLES"

python3 "$PY_SCRIPT" \
  --input-jsonl "$INPUT_FILE" \
  --output-jsonl "$OUTPUT_FILE" \
  --model-path "$MODEL_PATH" \
  --family "$FAMILY" \
  --strategy "$STRATEGY" \
  --top-k-retrieved "$TOP_K_RETRIEVED" \
  --top-k-context "$TOP_K_CONTEXT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-samples "$MAX_SAMPLES" \
  $FLAGS \
  ${BASE_FLAG[@]+"${BASE_FLAG[@]}"}

echo "Finished at: $(date)"
