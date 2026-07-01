# Steerling Notebooks

Interactive walkthroughs for generation, interpretability, and concept steering with [Steerling-8B](https://huggingface.co/guidelabs/steerling-8b).

## Requirements

- Python >= 3.13
- GPU with >= 18 GB VRAM (H100, A100, A6000, RTX 4090)
- CUDA 12.8

Install the package and notebook dependencies:

```bash
uv sync --extra dev
source .venv/bin/activate
jupyter notebook notebooks/
```

The `dev` extra includes `steerling[notebook]` (Jupyter and ipykernel).

## Directory layout

Notebooks are grouped by model checkpoint:

| Directory | Model | HuggingFace ID |
|-----------|-------|----------------|
| [`base_model/`](base_model/) | Base | [`guidelabs/steerling-8b`](https://huggingface.co/guidelabs/steerling-8b) |
| [`instruct_model/`](instruct_model/) | Instruct | [`guidelabs/steerling-8b-instruct`](https://huggingface.co/guidelabs/steerling-8b-instruct) |

Each directory contains the same set of workflows adapted for that checkpoint. Use the **base** notebooks for plain text completion and the **instruct** notebooks for chat-style generation with the Llama 3-style template.

[`search_concepts.ipynb`](search_concepts.ipynb) lives at the top level and applies to both checkpoints.

## Recommended order

1. **[Finding Concepts to Steer](search_concepts.ipynb)** — search the concept catalog and confirm a concept ID against the loaded weights before steering.
2. **[Text Generation](base_model/generation.ipynb)** or **[Text Generation (Instruct)](instruct_model/generation_instruct.ipynb)** — block-by-block diffusion generation.
3. **[Logit Contribution Analysis](base_model/logit_contribution.ipynb)** — per-token decomposition into known, discovered, and residual components.
4. **[Chunk-to-Concept Attribution](base_model/chunk_level_concept_attribution.ipynb)** — attribute generated text to known concepts.
5. **[Concept Amplify](base_model/amplify_concept.ipynb)** or **[Concept Suppression](base_model/suppress_concept.ipynb)** — steer generation toward or away from a target concept.

For instruct workflows, use the corresponding notebooks under [`instruct_model/`](instruct_model/).

## Notebooks

### Shared

| Notebook | Description |
|----------|-------------|
| [search_concepts.ipynb](search_concepts.ipynb) | Search the concept catalog by meaning and verify a concept ID with `concept_top_tokens` |

### Base model (`guidelabs/steerling-8b`)

| Notebook | Description |
|----------|-------------|
| [generation.ipynb](base_model/generation.ipynb) | Text generation via confidence-based block unmasking |
| [logit_contribution.ipynb](base_model/logit_contribution.ipynb) | Decompose each predicted token's logit into known, discovered, and residual contributions |
| [chunk_level_concept_attribution.ipynb](base_model/chunk_level_concept_attribution.ipynb) | Attribute generated text chunks to known concepts |
| [amplify_concept.ipynb](base_model/amplify_concept.ipynb) | Amplify a target concept during generation via residual-stream injection |
| [suppress_concept.ipynb](base_model/suppress_concept.ipynb) | Suppress a target concept via negative injection and ReLU logit masking |

### Instruct model (`guidelabs/steerling-8b-instruct`)

| Notebook | Description |
|----------|-------------|
| [generation_instruct.ipynb](instruct_model/generation_instruct.ipynb) | Chat-style generation with system, user, and assistant roles |
| [logit_contribution_instruct.ipynb](instruct_model/logit_contribution_instruct.ipynb) | Per-token logit decomposition during instruct generation |
| [chunk_level_concept_attribution_instruct.ipynb](instruct_model/chunk_level_concept_attribution_instruct.ipynb) | Chunk-level concept attribution for instruct outputs |
| [amplify_concept_instruct.ipynb](instruct_model/amplify_concept_instruct.ipynb) | Amplify a concept during instruct generation |
| [suppress_concept_instruct.ipynb](instruct_model/suppress_concept_instruct.ipynb) | Suppress a concept during instruct generation |

## Notes

- First run downloads ~17 GB of model weights from HuggingFace Hub; later runs load from cache.
- Concept IDs are checkpoint-specific. Always confirm a concept with `search_concepts.ipynb` before using it in amplify or suppress notebooks.
- Steering notebooks require an **interpretable** Steerling checkpoint (both public models qualify).

For installation, architecture, and evaluation details, see the [project README](../README.md).
