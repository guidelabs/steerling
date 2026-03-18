# Uploading Steerling to HuggingFace

## Steps

1. **Build HF-compatible files** from a training config:

   ```bash
   python scripts/hugging_face/build_hf_files.py --config /path/to/training_config.py
   ```

   This generates the following files in `hf/`:
   - `config.json` — model configuration
   - `configuration_steerling.py` — HF config class
   - `modeling_steerling.py` — HF model class (inlined from steerling source)
   - `tokenization_steerling.py` — HF tokenizer class
   - `tokenizer_config.json` — tokenizer configuration

2. **Upload to HuggingFace Hub**:

   ```bash
   python scripts/hugging_face/upload_to_hf.py --model-path /path/to/weights --repo-id guidelabs/steerling-8b
   ```

   This uploads the generated HF files and safetensor weights to the specified repo.

## Notes

- You must be logged in to HuggingFace (`huggingface-cli login`) before uploading.
- The `hf/` output directory is gitignored — generated files are not committed.
- Weight files should be in safetensors format (`model.safetensors` or sharded `model-*.safetensors`).
