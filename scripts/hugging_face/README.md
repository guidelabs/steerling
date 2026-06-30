# Uploading Steerling to HuggingFace

## Steps

1. **Build HF-compatible files** from a training config:

   ```bash
   # Base model
   python scripts/hugging_face/build_hf_files.py --config /path/to/training_config.py

   # Instruct model
   python scripts/hugging_face/build_hf_files.py --config /path/to/training_config.py --instruct

   # Custom output directory (default: hf/)
   python scripts/hugging_face/build_hf_files.py --config /path/to/training_config.py --output-dir custom/hf/path
   ```

   This generates the following files in `hf/`:
   - `config.json` — model configuration
   - `configuration_steerling.py` — HF config class
   - `modeling_steerling.py` — HF model class (inlined from steerling source)
   - `tokenization_steerling.py` — HF tokenizer class
   - `tokenizer_config.json` — tokenizer configuration

2. **Upload to HuggingFace Hub**:

   ```bash
   # Upload code files + weights
   python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id guidelabs/steerling-8b

   # Use a different HF files directory (e.g. for instruct)
   python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id guidelabs/steerling-8b-instruct --hf-dir hf-instruct

   # Upload code files only (skip weights)
   python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id guidelabs/steerling-8b --skip-weights
   ```

   This uploads the generated HF files and safetensor weights to the specified repo.

## Notes

- You must be logged in to HuggingFace (`huggingface-cli login`) before uploading.
- The `hf/` output directory is gitignored — generated files are not committed.
- Weight files should be in safetensors format (`model.safetensors` or sharded `model-*.safetensors`).
