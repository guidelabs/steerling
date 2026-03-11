#!/bin/bash
# Steerling Evaluation Script
#
# Runs lm-eval-harness benchmarks on a Steerling model.
# No SLURM — runs locally on a single GPU.
#
# Usage:
#   bash scripts/eval_steerling_lm_eval.sh
#   MODEL_PATH=/path/to/local bash scripts/eval_steerling_lm_eval.sh
#   TASKS="hellaswag mmlu" bash scripts/eval_steerling_lm_eval.sh
#   DEVICE=cuda:1 bash scripts/eval_steerling_lm_eval.sh
#
# Prerequisites:
#   pip install -e .
#   pip install lm-eval

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-guidelabs/steerling-8b}"
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"

echo "============================================"
echo "Steerling Evaluation"
echo "============================================"
echo "Model:   ${MODEL_PATH}"
echo "Device:  ${DEVICE}"
echo "Results: ${RESULTS_DIR}"
echo "============================================"

# Disable torch.compile to avoid Triton kernel issues
export TORCH_COMPILE_DISABLE=1

TASKS="${TASKS:-hellaswag arc_challenge winogrande piqa openbookqa mmlu gsm8k}"

python scripts/evaluate.py \
    --model "${MODEL_PATH}" \
    --tasks ${TASKS} \
    --results-dir "${RESULTS_DIR}" \
    --device "${DEVICE}"
