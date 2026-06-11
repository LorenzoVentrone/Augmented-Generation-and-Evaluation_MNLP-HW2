#!/bin/bash
# Parametric SLURM job to QLoRA-fine-tune ANY small LM (<=1B) on a SINGLE GPU.
#
# Pass the model as POSITIONAL ARGUMENTS:
#   sbatch slurm_job_small.sh <MODEL_TAG> <MODEL_DIR> [DATA_MODE]
#
# e.g.:
#   sbatch slurm_job_small.sh smollm2-360m /leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/assets/model/smollm2-360m-instruct
#   sbatch slurm_job_small.sh qwen-0.5b    /leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/assets/model/qwen2.5-0.5b
#   sbatch slurm_job_small.sh minerva-1b   /leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/assets/model/minerva-1b-base-v1.0
#
# DATA_MODE (3rd arg, optional): oracle (default) | precomputed (corrective dataset).
# LORA_R / LORA_ALPHA (default 16 / 32).

#SBATCH --job-name=small_rag_ft
#SBATCH --account=IscrC_MNLP26
#SBATCH --output=/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/finetune/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/finetune/logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load profile/deeplrn
module load cuda/12.2

WORKDIR=/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning
cd "$WORKDIR/finetune"
source "$WORKDIR/../.venv/bin/activate" 2>/dev/null || source /leonardo/home/userexternal/lventron/src/.venv/bin/activate

export HF_HOME=/leonardo_scratch/large/userexternal/lventron/.cache/huggingface
export TRANSFORMERS_CACHE=/leonardo_scratch/large/userexternal/lventron/.cache/transformers
export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/lventron/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORKDIR/finetune/logs"

# positional args (preferred), falling back to env vars
MODEL_TAG="${1:-${MODEL_TAG:-}}"
MODEL_DIR="${2:-${MODEL_DIR:-}}"
DATA_MODE="${3:-${DATA_MODE:-oracle}}"
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

if [ -z "$MODEL_TAG" ] || [ -z "$MODEL_DIR" ]; then
    echo "ERROR: usage: sbatch slurm_job_small.sh <MODEL_TAG> <MODEL_DIR> [DATA_MODE]" >&2
    exit 1
fi
# fail fast if MODEL_DIR is wrong
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR does not exist: '$MODEL_DIR'" >&2
    echo "       Pass the FULL absolute path as the 2nd argument (no shell variable)." >&2
    exit 1
fi

if [ "$DATA_MODE" = "precomputed" ]; then
    DATASET_DIR=${DATASET_DIR:-$WORKDIR/finetune/assets/dataset/hw-mnlp-2026-corrective}
else
    DATASET_DIR=${DATASET_DIR:-$WORKDIR/assets/dataset/hw-mnlp-2026}
fi

OUTPUT_DIR=/leonardo_scratch/large/userexternal/lventron/${MODEL_TAG}_rag_ft_${DATA_MODE}_$

echo "=== ${MODEL_TAG} | mode=${DATA_MODE} | r=${LORA_R}/${LORA_ALPHA} ==="
echo "model:   $MODEL_DIR"
echo "dataset: $DATASET_DIR"
echo "output:  $OUTPUT_DIR"
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Single GPU.
python -u train.py \
    --model_name "$MODEL_DIR" \
    --dataset_name "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --data_mode "$DATA_MODE" \
    --top_k 3 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --max_seq_length 1536 \
    --logging_steps 25 \
    --eval_steps 50 \
    --save_steps 50 \
    --eval_split_ratio 0.02

echo "Training complete. Output at: $OUTPUT_DIR"
