#!/usr/bin/env python3
"""Upload Steerling model to HuggingFace Hub.

Usage:
    python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id org/model-name
    python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id org/model-name --skip-weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HF_DIR = Path(__file__).resolve().parent.parent.parent / "hf"

CODE_FILES = [
    "config.json",
    "configuration_steerling.py",
    "modeling_steerling.py",
    "tokenization_steerling.py",
    "tokenizer_config.json",
]

WEIGHT_PATTERNS = [
    "model-*.safetensors",
    "model.safetensors.index.json",
    "model.safetensors",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Steerling to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to safetensor weights directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID (e.g. org/model-name)")
    parser.add_argument("--skip-weights", action="store_true", help="Skip uploading weight files")
    args = parser.parse_args()

    weights_dir = Path(args.model_path)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    api.create_repo(repo_id=args.repo_id, private=True, exist_ok=True)
    print(f"Repo: https://huggingface.co/{args.repo_id}")

    # Upload code files
    print("\n--- Uploading code files ---")
    for filename in CODE_FILES:
        filepath = HF_DIR / filename
        if not filepath.exists():
            print(f"  SKIP (not found): {filepath}")
            continue
        print(f"  {filename}")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=args.repo_id,
            commit_message=f"Update {filename}",
        )

    if args.skip_weights:
        print("\n--- Skipping weights (--skip-weights flag set) ---")
        print(f"\nDone! https://huggingface.co/{args.repo_id}")
        return

    # Upload weights
    print("\n--- Uploading weights ---")
    weight_files: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        weight_files.extend(weights_dir.glob(pattern))
    weight_files = sorted(set(weight_files))

    if not weight_files:
        print(f"  WARNING: no safetensor files found in {weights_dir}")
    else:
        for wf in weight_files:
            size_gb = wf.stat().st_size / 1e9
            print(f"  {wf.name} ({size_gb:.1f} GB)")
            api.upload_file(
                path_or_fileobj=str(wf),
                path_in_repo=wf.name,
                repo_id=args.repo_id,
                commit_message=f"Upload {wf.name}",
            )

    print(f"\nDone! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()