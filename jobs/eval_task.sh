#!/bin/bash
#SBATCH --job-name=steerling-eval
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Single-task evaluation job (1 GPU).
# Called by launch_all_evals.sh — not meant to be submitted directly.
# REPO_ROOT is passed as an env var by the launcher.

set -e

TASK="$1"
MODEL="${2:-asalam91/steerling-test}"
REPO_ROOT="${REPO_ROOT:?REPO_ROOT must be set}"

if [ -z "$TASK" ]; then
    echo "Usage: sbatch jobs/eval_task.sh <task> [model]"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/eval_results"

echo "Task:   $TASK"
echo "Model:  $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Time:   $(date)"

source "$REPO_ROOT/.venv/bin/activate"

export TORCH_COMPILE_DISABLE=1

python scripts/evaluate.py \
    --model "$MODEL" \
    --tasks "$TASK" \
    --results-dir "$REPO_ROOT/eval_results" \
    --device cuda

echo "Finished $TASK at $(date)"
