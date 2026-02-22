"""
Convert ScaleX checkpoints to Steerling safetensors format.

Usage (run from your ScaleX environment):
    python scripts/convert_weights.py \
        --checkpoint /fss/shared/models/bl_iguide_midtraining/models-67528/checkpoints/last.ckpt \
        --output ./steerling-8b-hf/ \
        --config /fss/shared/models/bl_iguide_midtraining/models-67528/config.json

This script:
1. Loads a ScaleX DCP or portable checkpoint
2. Extracts model weights (strips optimizer state)
3. Converts to bfloat16
4. Saves as safetensors (sharded if > 5GB)
5. Writes config.json in Steerling format
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_scalex_checkpoint(checkpoint_path: Path, config_path: Path | None = None) -> dict[str, torch.Tensor]:
    """Load a ScaleX checkpoint and return a clean state_dict."""
    if checkpoint_path.is_dir():
        # DCP format — requires scalex installed
        try:
            from scalex.inference.checkpoint_utils import (
                build_model_and_tokenizer,
                load_dcp_weights,
                load_hparams,
                materialize_model,
            )
        except ImportError as e:
            raise ImportError(
                "ScaleX package is required to convert DCP checkpoints. "
                "Run this script from your ScaleX environment."
            ) from e

        hparams = load_hparams(checkpoint_path, config_path)
        model, tokenizer, model_config, _, _, is_interpretable = build_model_and_tokenizer(hparams)
        materialize_model(model, device="cpu")
        layout = load_dcp_weights(model, checkpoint_path)
        logger.info(f"Loaded DCP checkpoint (layout: {layout})")
        return model.state_dict()
    else:
        # Portable .pt file
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", {}))
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        return state_dict


def build_steerling_config(scalex_config_path: Path) -> dict:
    """Convert ScaleX config.json to Steerling config format."""
    with open(scalex_config_path) as f:
        sx = json.load(f)

    model = sx["model"]
    tokenizer = sx["tokenizer"]
    concept = sx.get("concept")
    steering = sx.get("steering")

    config: dict = {
        "model_type": "causal_diffusion",
        "interpretable": model.get("interpretable", False),
        "n_layers": model["n_layers"],
        "n_head": model["n_head"],
        "n_embd": model["n_embd"],
        "block_size": model["block_size"],
        "n_kv_heads": model.get("n_kv_heads"),
        "diff_block_size": model["diff_block_size"],
        "use_rms_norm": model["use_rms_norm"],
        "norm_eps": model["norm_eps"],
        "norm_order": model["norm_order"],
        "use_qk_norm": model["use_qk_norm"],
        "use_rope": model["use_rope"],
        "rope_base": model["rope_base"],
        "rope_full_precision": model["rope_full_precision"],
        "mlp_type": model["mlp_type"],
        "activation": model["activation"],
        "mlp_ratio": model["mlp_ratio"],
        "intermediate_size": model.get("intermediate_size"),
        "use_bias": model["use_bias"],
        "clip_qkv": model.get("clip_qkv"),
        "weight_sharing": model["weight_sharing"],
        "pad_token_id": tokenizer["pad_token_id"],
        "bos_token_id": tokenizer["bos_token_id"],
        "eos_token_id": tokenizer["eos_token_id"],
        "endofchunk_token_id": tokenizer["endofchunk_token_id"],
        "mask_token_id": tokenizer["mask_token_id"],
        "vocab_size": tokenizer["vocab_size"],
    }

    if concept is not None and model.get("interpretable", False):
        config["concept"] = {
            "n_concepts": concept["n_concepts"],
            "n_unknown_concepts": concept["n_unknown_concepts"],
            "max_concepts": concept.get("max_concepts", 16),
            "concept_dim": concept["concept_dim"],
            "use_attention_known": concept["use_attention_known"],
            "use_attention_unknown": concept["use_attention_unknown"],
            "topk_known": concept["topk_known"],
            "topk_known_features": concept.get("topk_known_features"),
            "unknown_topk": concept.get("unknown_topk"),
            "use_unknown": concept["use_unknown"],
            "apply_topk_to_unknown": concept.get("apply_topk_to_unknown", False),
            "topk_on_logits": concept.get("topk_on_logits", False),
            "factorize_unknown": concept.get("factorize_unknown", False),
            "factorize_rank": concept.get("factorize_rank", 256),
            "use_epsilon_correction": concept.get("use_epsilon_correction", True),
            "block_size": concept.get("block_size", 4096),
            "pad_multiple": concept.get("pad_multiple", 16),
            "store_unknown_weights": concept.get("store_unknown_weights", False),
            "inject_layer": steering["inject_layer"] if steering else 16,
            "inject_alpha": steering["inject_alpha"] if steering else 1.0,
        }

    return config


def save_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: Path,
    shard_size_bytes: int = 5 * 1024 * 1024 * 1024,
):
    """Save state dict as safetensors, sharding if necessary."""
    from safetensors.torch import save_file

    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    logger.info(f"Total model size: {total_size / 1e9:.2f} GB ({len(state_dict)} tensors)")

    if total_size <= shard_size_bytes:
        save_file(state_dict, output_dir / "model.safetensors")
        logger.info("Saved single file: model.safetensors")
        return

    # Sharded
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    weight_map: dict[str, str] = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    total_shards = len(shards)
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, output_dir / shard_name)
        for key in shard:
            weight_map[key] = shard_name
        logger.info(f"Saved shard: {shard_name} ({len(shard)} tensors)")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Convert ScaleX checkpoint to Steerling safetensors")
    parser.add_argument("--checkpoint", required=True, help="Path to ScaleX checkpoint (DCP dir or .pt file)")
    parser.add_argument("--output", required=True, help="Output directory for safetensors + config")
    parser.add_argument("--config", default=None, help="Path to ScaleX config.json")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find config
    config_file = Path(args.config) if args.config else None
    if config_file is None:
        for parent in [checkpoint_path.parent, checkpoint_path.parent.parent]:
            candidate = parent / "config.json"
            if candidate.exists():
                config_file = candidate
                break

    if config_file is None:
        raise FileNotFoundError("Could not find config.json. Pass --config explicitly.")

    logger.info(f"Using config: {config_file}")

    # Load
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_scalex_checkpoint(checkpoint_path, config_file)

    # Convert dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    target_dtype = dtype_map[args.dtype]
    logger.info(f"Converting to {target_dtype}...")
    state_dict = {k: v.to(target_dtype) if v.is_floating_point() else v for k, v in state_dict.items()}

    # Handle weight tying
    tok_key = "transformer.tok_emb.weight"
    lm_key = "transformer.lm_head.weight"
    if (
        tok_key in state_dict
        and lm_key in state_dict
        and torch.equal(state_dict[tok_key], state_dict[lm_key])
    ):
        del state_dict[lm_key]
        logger.info("Removed duplicate lm_head.weight (tied to tok_emb.weight)")

    # Save
    save_safetensors(state_dict, output_dir)

    # Save config
    steerling_config = build_steerling_config(config_file)
    with open(output_dir / "config.json", "w") as f:
        json.dump(steerling_config, f, indent=2)

    # Summary
    n_params = sum(v.numel() for v in state_dict.values())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"  Output:     {output_dir}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Size:       {total_bytes / 1e9:.2f} GB")
    print(f"  Dtype:      {target_dtype}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
