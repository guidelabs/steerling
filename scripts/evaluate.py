#!/usr/bin/env python3
"""
Steerling evaluation CLI.

Reads task settings from steerling.configs.evaluation and runs
lm-eval-harness benchmarks.

Usage:
    python scripts/evaluate.py --model guidelabs/steerling-8b --tasks hellaswag arc_challenge
    python scripts/evaluate.py --model /path/to/local --tasks mmlu --device cuda:0
    python scripts/evaluate.py --model guidelabs/steerling-8b  # runs all default tasks
"""

from __future__ import annotations

import argparse
import logging

from steerling.configs.evaluation import ALL_TASKS
from steerling.evaluation.lm_harness_wrapper import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Steerling model with lm-eval-harness")
    parser.add_argument(
        "--model",
        type=str,
        default="guidelabs/steerling-8b",
        help="HuggingFace repo ID or local path to model",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Tasks to evaluate. Default: all ({', '.join(ALL_TASKS)})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()
    tasks = args.tasks or ALL_TASKS

    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Results: {args.results_dir}")

    results = run_evaluation(
        model_path=args.model,
        tasks=tasks,
        results_dir=args.results_dir,
        device=args.device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for task_name, task_results in results.items():
        task_metrics = task_results.get("results", {}).get(task_name, {})
        if task_metrics:
            print(f"\n{task_name}:")
            for metric, value in task_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
