"""
Evaluation configuration for Steerling models.

Tasks and settings are derived strictly from eval_steerling_lm_eval.sh.
Only includes what is explicitly defined in that script.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskSettings(BaseModel):
    """Settings for a single lm-eval-harness task."""

    num_fewshot: int = Field(default=0, ge=0)
    batch_size: int = Field(default=16, gt=0)
    mc_num: int = Field(default=128, gt=0, description="Monte Carlo samples for likelihood estimation")
    mc_batch_size: int = Field(default=32, gt=0, description="Batch size per MC forward pass")
    cfg: float = Field(default=0.0, ge=0.0, description="Classifier-free guidance scale")
    gen_length: int | None = Field(default=None, gt=0, description="Max generation tokens (generation tasks only)")
    steps: int | None = Field(default=None, gt=0, description="Diffusion steps (generation tasks only)")

    model_config = {"extra": "forbid"}


# fmt: off
TASK_DEFAULTS: dict[str, TaskSettings] = {
    "hellaswag":     TaskSettings(num_fewshot=0, batch_size=32, mc_num=128, mc_batch_size=32, cfg=1.0),
    "arc_challenge": TaskSettings(num_fewshot=0, batch_size=64, mc_num=128, mc_batch_size=32, cfg=1.0),
    "winogrande":    TaskSettings(num_fewshot=5, batch_size=64, mc_num=128, mc_batch_size=32, cfg=1.5),
    "piqa":          TaskSettings(num_fewshot=0, batch_size=64, mc_num=128, mc_batch_size=32, cfg=0.5),
    "mmlu":          TaskSettings(num_fewshot=5, batch_size=8,  mc_num=1,   mc_batch_size=1,  cfg=0.0),
    "gsm8k":         TaskSettings(num_fewshot=4, batch_size=8,  cfg=0.0, gen_length=256, steps=256),
}
# fmt: on

ALL_TASKS = list(TASK_DEFAULTS.keys())


def get_task_settings(task_name: str, overrides: dict | None = None) -> TaskSettings:
    """
    Get settings for a task. Falls back to defaults if task not found.

    Args:
        task_name: lm-eval task name
        overrides: Optional dict to override defaults

    Returns:
        TaskSettings for the task
    """
    base = TASK_DEFAULTS.get(task_name, TaskSettings())
    if overrides:
        return base.model_copy(update=overrides)
    return base
