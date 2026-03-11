"""
Checkpoint utilities for loading Steerling models.

Supports:
- Local directory with safetensors
- HuggingFace Hub download
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

# HuggingFace Hub
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Keys injected by HuggingFace that are not part of the model config
HF_KEYS = {"model_type", "transformers_version", "auto_map", "architectures"}


def load_config(model_name_or_path: str) -> dict:
    """
    Load config.json from a local directory or HuggingFace Hub.

    Strips HuggingFace-specific keys before returning.

    Returns:
        Parsed config dictionary
    """

    path = Path(model_name_or_path)

    if path.is_dir():
        config_file = path / "config.json"
    else:
        config_file = Path(hf_hub_download(model_name_or_path, "config.json"))

    with open(config_file) as f:
        config = json.load(f)

    for key in HF_KEYS:
        config.pop(key, None)

    return config


def load_state_dict(model_name_or_path: str) -> dict[str, torch.Tensor]:
    """
    Load model weights from safetensors (local or HuggingFace Hub).

    Handles both single-file and sharded safetensors.

    Returns:
        Complete state dict
    """

    path = Path(model_name_or_path)

    if path.is_dir():
        # Local directory
        safetensor_files = sorted(path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")

        state_dict: dict[str, torch.Tensor] = {}
        for sf in safetensor_files:
            state_dict.update(load_file(str(sf), device="cpu"))

        logger.info(f"Loaded {len(state_dict)} tensors from {len(safetensor_files)} file(s)")
        return state_dict
    else:
        try:
            # Try single file first
            sf_path = hf_hub_download(model_name_or_path, "model.safetensors")
            state_dict = load_file(sf_path, device="cpu")
            logger.info(f"Loaded {len(state_dict)} tensors from single file")
            return state_dict
        except Exception:
            pass

        # Try sharded files
        try:
            index_path = hf_hub_download(model_name_or_path, "model.safetensors.index.json")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find model weights at '{model_name_or_path}'. "
                "Expected model.safetensors or model.safetensors.index.json"
            ) from e

        with open(index_path) as f:
            index = json.load(f)

        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_name in shard_files:
            shard_path = hf_hub_download(model_name_or_path, shard_name)
            state_dict.update(load_file(shard_path, device="cpu"))

        logger.info(f"Loaded {len(state_dict)} tensors from {len(shard_files)} shard(s)")
        return state_dict
