#!/bin/bash
# Run the evaluation for one model over the chosen settings, self-contained (reads
# gold/answerable directly from the teammate's JSONL — no external dataset). 1 GPU.
#
#   sbatch run_all_eval.sh <MODEL_TAG> <INPUT_DIR> [SETTINGS] [SUFFIX]
#
# SETTINGS (3rd arg, optional): comma-separated subset of baseline,rag,oracle.
#   default = baseline,rag,oracle  (base models with all 3 files)
#   e.g.    = rag                  (models that only have a rag file)
#
# SUFFIX (4th arg, optional): inserted before .jsonl in the filenames. Default "".
#   base models  -> files: test_baseline.jsonl  test_rag.jsonl  test_oracle.jsonl
#   FT models    -> SUFFIX=_ft, SETTINGS=rag  ->  test_rag_ft.jsonl
#
# A setting requested but whose file is missing is skipped with a warning.
#
# Metric routing (see EVALUATION_METRICS.md):
#   EM/sub-EM/METEOR (score_jsonl)        -> every chosen+present setting (B2.1)
#   BERTScore (score_a2)                  -> every chosen+present setting
#   NLI faithfulness (score_a2)           -> rag + oracle only (baseline has no context)
#   abstention/hallucination (corrective) -> rag only

#SBATCH --job-name=eval_all
#SBATCH --account=IscrC_MNLP26
#SBATCH --output=/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/finetune/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/finetune/logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

module purge
module load profile/deeplrn
module load cuda/12.2

cd /leonardo/home/userexternal/lventron/src/Minerva3B_RAG_Finetuning/finetune
source /leonardo/home/userexternal/lventron/src/.venv/bin/activate

# Only HF_HOME (warm-up cached the A2 models here). NO TRANSFORMERS_CACHE.
export HF_HOME=/leonardo_scratch/large/userexternal/lventron/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA=$HOME/nltk_data            # for METEOR (wordnet/omw)

mkdir -p logs results/a2

MODEL_TAG="${1:-}"
INPUT_DIR="${2:-}"
SETTINGS="${3:-baseline,rag,oracle}"
SUFFIX="${4:-}"
if [ -z "$MODEL_TAG" ] || [ -z "$INPUT_DIR" ]; then
    echo "ERROR: usage: sbatch run_all_eval.sh <MODEL_TAG> <INPUT_DIR> [SETTINGS] [SUFFIX]" >&2
    exit 1
fi

want() { case ",$SETTINGS," in *",$1,"*) return 0;; *) return 1;; esac; }

BASE="$INPUT_DIR/test_baseline${SUFFIX}.jsonl"
RAG="$INPUT_DIR/test_rag${SUFFIX}.jsonl"
ORACLE="$INPUT_DIR/test_oracle${SUFFIX}.jsonl"

# Resolve which settings are both requested AND present.
DO_BASE=0; DO_RAG=0; DO_ORACLE=0
JSONL_ALL=()      # all chosen+present files -> EM/sub-EM/METEOR + BERTScore
A2_NLI=()         # rag/oracle chosen+present -> BERTScore + NLI
for s in baseline rag oracle; do
    if want "$s"; then
        f="$INPUT_DIR/test_${s}${SUFFIX}.jsonl"
        if [ -f "$f" ]; then
            JSONL_ALL+=("$f")
            case "$s" in
                baseline) DO_BASE=1 ;;
                rag)      DO_RAG=1;    A2_NLI+=("$f") ;;
                oracle)   DO_ORACLE=1; A2_NLI+=("$f") ;;
            esac
        else
            echo "WARN: setting '$s' requested but $f not found — skipping"
        fi
    fi
done

if [ ${#JSONL_ALL[@]} -eq 0 ]; then
    echo "ERROR: none of the requested settings ($SETTINGS) have files in $INPUT_DIR" >&2
    exit 1
fi

A2OUT="results/a2/${MODEL_TAG}_2"
mkdir -p "$A2OUT"

echo "############################################################"
echo "# EVAL  model=$MODEL_TAG   settings=$SETTINGS"
echo "#       files: ${JSONL_ALL[*]}"
echo "############################################################"
nvidia-smi || true
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo
echo "============================================================"
echo "[1] EM / sub-EM / METEOR  (B2.1)"
echo "============================================================"
python -u score_jsonl.py --gold_field ground_truth --inputs "${JSONL_ALL[@]}"

echo
echo "============================================================"
echo "[2] A2 — BERTScore (raw + rescaled)  &  NLI faithfulness"
echo "============================================================"
if [ "$DO_BASE" -eq 1 ]; then
    echo "--- baseline (BERTScore only, no context for NLI) ---"
    python -u score_a2.py --inputs "$BASE" --no_nli --no_rescale --output_dir "$A2OUT"   # RAW (saved per-example)
    python -u score_a2.py --inputs "$BASE" --no_nli                                        # rescaled (aggregate, ref)
fi
if [ ${#A2_NLI[@]} -gt 0 ]; then
    echo "--- rag/oracle (BERTScore + NLI faithfulness) ---"
    python -u score_a2.py --inputs "${A2_NLI[@]}" --no_rescale --output_dir "$A2OUT"   # RAW BERTScore + NLI (saved)
    python -u score_a2.py --inputs "${A2_NLI[@]}" --no_nli                              # rescaled BERTScore (aggregate, ref)
fi

if [ "$DO_RAG" -eq 1 ]; then
    echo
    echo "============================================================"
    echo "[3] Corrective abstention / hallucination  (RAG only)"
    echo "============================================================"
    python -u score_corrective.py --from_jsonl --top_k 3 --inputs "$RAG"
fi

echo
echo "DONE. Per-example A2 scores in $A2OUT/"
