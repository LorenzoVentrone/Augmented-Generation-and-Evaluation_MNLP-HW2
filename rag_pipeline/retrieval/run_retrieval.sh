#!/bin/bash
#SBATCH --job-name=mnlp_retrieval
#SBATCH --account=IscrC_MNLP26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/ltam0000/src/rag_pipeline/retrieval/logs/retrieval_%j.out
#SBATCH --error=/leonardo/home/userexternal/ltam0000/src/rag_pipeline/retrieval/logs/retrieval_%j.err

# SLURM launcher for build_retrieval_index.py: enriches the dataset splits with the
# reranked `retrieved_index` field. Submit from a login node:
#   sbatch run_retrieval.sh
# Every knob below can be overridden via environment variables, e.g.:
#   SPLITS=blind sbatch run_retrieval.sh

set -euo pipefail

BASE=/leonardo/home/userexternal/ltam0000
SCRATCH=$BASE/my_scratch/MNLP_FUNTORI_HWII/scratch
PYTHON_SCRIPT=$BASE/src/rag_pipeline/retrieval/build_retrieval_index.py

mkdir -p "$BASE/src/rag_pipeline/retrieval/logs"

# Tunable knobs (override via environment variables)
DATASET_DIR=${DATASET_DIR:-$SCRATCH/dataset}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/my_scratch/MNLP_FUNTORI_HWII/output}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-$SCRATCH/models/bge-m3-mnrl-hardneg}   # HW1 fine-tuned bge-m3
RERANKER_MODEL=${RERANKER_MODEL:-$SCRATCH/models/bge-reranker-v2-m3}      # off-the-shelf cross-encoder
TOP_K=${TOP_K:-10}               # length of the ranked index list kept per query
MAX_SAMPLES=${MAX_SAMPLES:--1}   # -1 = full split; small values for smoke tests
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-256}
RERANKER_BATCH_SIZE=${RERANKER_BATCH_SIZE:-64}
SPLITS=${SPLITS:-train,test}     # comma-separated; use SPLITS=blind for the blind split

mkdir -p "$OUTPUT_DIR"

echo "Job ID          : ${SLURM_JOB_ID:-N/A}"
echo "Running on host : $(hostname)"
echo "Starting time   : $(date)"
echo "Submit dir      : ${SLURM_SUBMIT_DIR:-$PWD}"
echo ""
echo "Dataset dir     : $DATASET_DIR"
echo "Output dir      : $OUTPUT_DIR"
echo "Retriever model : $RETRIEVER_MODEL"
echo "Reranker model  : $RERANKER_MODEL"
echo "Top-k           : $TOP_K"
echo "Max samples     : $MAX_SAMPLES"
echo ""

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

python3 "$PYTHON_SCRIPT" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --retriever-model "$RETRIEVER_MODEL" \
  --reranker-model "$RERANKER_MODEL" \
  --top-k "$TOP_K" \
  --max-samples "$MAX_SAMPLES" \
  --encode-batch-size "$ENCODE_BATCH_SIZE" \
  --reranker-batch-size "$RERANKER_BATCH_SIZE" \
  --splits "$SPLITS"

echo ""
echo "Finished at: $(date)"
