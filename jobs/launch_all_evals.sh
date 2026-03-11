#!/bin/bash
# Launch all evaluation tasks in parallel (1 GPU each).
#
# Usage:
#   bash jobs/launch_all_evals.sh [model]
#
# Example:
#   bash jobs/launch_all_evals.sh asalam91/steerling-test
#   bash jobs/launch_all_evals.sh guidelabs/steerling-8b

set -e

MODEL="${1:-asalam91/steerling-test}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/eval_results"

TASKS=(
    hellaswag
    arc_challenge
    winogrande
    piqa
    mmlu
    gsm8k
)

echo "Launching ${#TASKS[@]} eval jobs for model: $MODEL"
echo "Partition: research | 1 GPU per task"
echo "Logs:      $REPO_ROOT/logs/"
echo "Results:   $REPO_ROOT/eval_results/"
echo ""

for task in "${TASKS[@]}"; do
    job_id=$(sbatch \
        --job-name="eval-${task}" \
        --output="$REPO_ROOT/logs/eval_${task}_%j.out" \
        --error="$REPO_ROOT/logs/eval_${task}_%j.err" \
        --export="ALL,REPO_ROOT=$REPO_ROOT" \
        --parsable \
        "$REPO_ROOT/jobs/eval_task.sh" "$task" "$MODEL")
    echo "  $task -> job $job_id"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
