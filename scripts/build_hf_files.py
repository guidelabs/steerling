#!/usr/bin/env python3
"""Generate HF-compatible files for Steerling.

Reads model and concept configs from a scalex training config file
and inlines source files — no hardcoded values.

Usage:
    python scripts/build_hf_files_v3.py --config /path/to/training_config.py
    python scripts/build_hf_files_v3.py --config /path/to/training_config.py --output-dir custom/hf/path
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
# Utilities
# ===========================================================================

def _read_source(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text()


def _load_training_config(config_path: Path) -> tuple[dict, dict]:
    source = config_path.read_text()

    variables: dict[str, object] = {}
    for m in re.finditer(r"^(\w+)\s*=\s*(.+?)$", source, re.MULTILINE):
        name, expr = m.group(1), m.group(2).strip()
        try:
            variables[name] = ast.literal_eval(expr)
        except (ValueError, SyntaxError):
            pass

    def _resolve(val_str: str) -> object:
        val_str = val_str.strip().rstrip(",")
        try:
            return ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            if val_str in variables:
                return variables[val_str]
            return None

    def _extract_constructor(var_name: str) -> dict:
        pattern = rf"{var_name}\s*=\s*\w+\((.*?)\)"
        m = re.search(pattern, source, re.DOTALL)
        if m is None:
            raise ValueError(f"Could not find '{var_name} = ...(...)' in {config_path}")
        body = m.group(1)
        result = {}
        for kwm in re.finditer(r"(\w+)\s*=\s*([^,\n]+)", body):
            key = kwm.group(1)
            val = _resolve(kwm.group(2))
            if val is not None:
                result[key] = val
        return result

    arch = _extract_constructor("model_config")
    concept = _extract_constructor("concept_config")
    return arch, concept


def _strip_docstrings(source: str) -> str:
    tree = ast.parse(source)

    class Stripper(ast.NodeTransformer):
        def _strip(self, node):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
            return node
        visit_Module = _strip
        visit_ClassDef = _strip
        visit_FunctionDef = _strip
        visit_AsyncFunctionDef = _strip

    ast.fix_missing_locations(Stripper().visit(tree))
    return ast.unparse(tree)


def compact_json(obj: dict) -> str:
    raw = json.dumps(obj, indent=2)
    def _collapse(m):
        elements = [line.strip().rstrip(",") for line in m.group(1).strip().splitlines()]
        return "[" + ", ".join(elements) + "]"
    return re.sub(r'\[\s*\n((?:\s+(?:"[^"]*"|null),?\s*\n)+)\s*\]', _collapse, raw) + "\n"


# ===========================================================================
# JSON configs
# ===========================================================================

def build_config_json(arch: dict, concept: dict) -> dict:
    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.configs.concept import ConceptConfig
    from steerling.data.tokenizer import SteerlingTokenizer

    tok = SteerlingTokenizer()
    arch_defaults = {name: f.default for name, f in CausalDiffusionConfig.model_fields.items() if not f.is_required()}
    concept_defaults = {name: f.default for name, f in ConceptConfig.model_fields.items() if not f.is_required()}
    a = {**arch_defaults, **arch}
    c = {**concept_defaults, **concept}

    return {
        "model_type": "steerling",
        "auto_map": {
            "AutoConfig": "configuration_steerling.SteerlingConfig",
            "AutoModel": "modeling_steerling.SteerlingForCausalLM",
            "AutoModelForCausalLM": "modeling_steerling.SteerlingForCausalLM",
            "AutoTokenizer": ["tokenization_steerling.SteerlingTokenizer", None],
        },
        "architectures": ["SteerlingForCausalLM"],
        "interpretable": a.get("interpretable", False),
        "vocab_size": tok.vocab_size,
        "n_layers": a["n_layers"],
        "n_head": a["n_head"],
        "n_embd": a["n_embd"],
        "n_kv_heads": a["n_kv_heads"],
        "block_size": a["block_size"],
        "diff_block_size": a["diff_block_size"],
        "use_rms_norm": a["use_rms_norm"],
        "norm_eps": a["norm_eps"],
        "norm_order": a["norm_order"],
        "use_qk_norm": a["use_qk_norm"],
        "use_rope": a["use_rope"],
        "rope_base": a["rope_base"],
        "rope_full_precision": a["rope_full_precision"],
        "clip_qkv": a["clip_qkv"],
        "mlp_type": a["mlp_type"],
        "activation": a["activation"],
        "mlp_ratio": a["mlp_ratio"],
        "intermediate_size": a["intermediate_size"],
        "use_bias": a["use_bias"],
        "weight_sharing": a["weight_sharing"],
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "mask_token_id": tok.mask_token_id,
        "endofchunk_token_id": tok.endofchunk_token_id,
        "n_concepts": c["n_concepts"],
        "n_unknown_concepts": c["n_unknown_concepts"],
        "concept_dim": c["concept_dim"],
        "use_attention_known": c["use_attention_known"],
        "use_attention_unknown": c["use_attention_unknown"],
        "topk_known": c["topk_known"],
        "topk_known_features": c["topk_known_features"],
        "unknown_topk": c["unknown_topk"],
        "use_unknown": c["use_unknown"],
        "apply_topk_to_unknown": c["apply_topk_to_unknown"],
        "topk_on_logits": c["topk_on_logits"],
        "factorize_unknown": c["factorize_unknown"],
        "factorize_rank": c["factorize_rank"],
        "use_epsilon_correction": c["use_epsilon_correction"],
        "concept_block_size": c["block_size"],
        "pad_multiple": c["pad_multiple"],
        "store_unknown_weights": c["store_unknown_weights"],
        "inject_layer": c["inject_layer"],
        "inject_alpha": c["inject_alpha"],
        "torch_dtype": "bfloat16",
        "transformers_version": "4.48.0",
    }


def build_tokenizer_config_json() -> dict:
    from steerling.data.tokenizer import SteerlingTokenizer
    tok = SteerlingTokenizer()
    return {
        "tokenizer_class": "SteerlingTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_steerling.SteerlingTokenizer", None]},
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|endoftext|>",
        "additional_special_tokens": ["<|endofchunk|>", "<|mask|>"],
        "encoding_name": SteerlingTokenizer.ENCODING_NAME,
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "endofchunk_token_id": tok.endofchunk_token_id,
        "mask_token_id": tok.mask_token_id,
    }


# ===========================================================================
# Python files
# ===========================================================================

def build_configuration_py(arch: dict, concept: dict) -> str:
    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.configs.concept import ConceptConfig
    from steerling.data.tokenizer import SteerlingTokenizer

    tok = SteerlingTokenizer()
    arch_defaults = {name: f.default for name, f in CausalDiffusionConfig.model_fields.items() if not f.is_required()}
    concept_defaults = {name: f.default for name, f in ConceptConfig.model_fields.items() if not f.is_required()}
    arch = {**arch_defaults, **arch}
    concept = {**concept_defaults, **concept}

    assigns = [
        "interpretable",
        "n_layers", "n_head", "n_embd", "n_kv_heads", "block_size", "diff_block_size",
        "use_rms_norm", "norm_eps", "norm_order", "use_qk_norm",
        "use_rope", "rope_base", "rope_full_precision", "clip_qkv",
        "mlp_type", "activation", "mlp_ratio", "intermediate_size", "use_bias", "weight_sharing",
        "mask_token_id", "endofchunk_token_id",
        "n_concepts", "n_unknown_concepts", "concept_dim",
        "use_attention_known", "use_attention_unknown",
        "topk_known", "topk_known_features", "unknown_topk",
        "use_unknown", "apply_topk_to_unknown", "topk_on_logits",
        "factorize_unknown", "factorize_rank", "use_epsilon_correction",
        "concept_block_size", "pad_multiple", "store_unknown_weights",
        "inject_layer", "inject_alpha",
    ]

    params = {**arch, **concept,
        "vocab_size": tok.vocab_size,
        "mask_token_id": tok.mask_token_id,
        "endofchunk_token_id": tok.endofchunk_token_id,
        "concept_block_size": concept["block_size"],
    }

    def fmt(v):
        if isinstance(v, str):
            return f'"{v}"'
        return repr(v)

    param_lines = "\n".join(
        f"        {k}={fmt(params[k])}," for k in assigns if k in params
    )
    assign_lines = "\n".join(f"        self.{k} = {k}" for k in assigns)

    return f'''\
from transformers import PretrainedConfig


class SteerlingConfig(PretrainedConfig):
    model_type = "steerling"

    def __init__(
        self,
        vocab_size={tok.vocab_size},
{param_lines}
        **kwargs,
    ):
{assign_lines}
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=kwargs.pop("pad_token_id", {tok.pad_token_id}),
            bos_token_id=kwargs.pop("bos_token_id", {tok.bos_token_id}),
            eos_token_id=kwargs.pop("eos_token_id", {tok.eos_token_id}),
            **kwargs,
        )
'''


def build_tokenization_py() -> str:
    from steerling.data.tokenizer import SteerlingTokenizer
    tok = SteerlingTokenizer()
    enc = SteerlingTokenizer.ENCODING_NAME

    core = _read_source("steerling/data/tokenizer.py")
    core = _strip_docstrings(core)
    core = re.sub(r"^import numpy.*\n", "", core, flags=re.MULTILINE)
    core = re.sub(r"^import torch.*\n", "", core, flags=re.MULTILINE)
    core = re.sub(r"^from torch.*\n", "", core, flags=re.MULTILINE)
    core = re.sub(r"^from __future__.*\n", "", core, flags=re.MULTILINE)
    core = core.replace("list[int] | np.ndarray | torch.Tensor", "list[int]")
    core = re.sub(r"\s*if isinstance\(tokens, torch\.Tensor\):.*?\.tolist\(\)\n", "\n", core, flags=re.DOTALL)
    core = re.sub(r"\s*if isinstance\(tokens, np\.ndarray\):.*?\.tolist\(\)\n", "\n", core, flags=re.DOTALL)
    core = core.replace("class SteerlingTokenizer:", "class _SteerlingTokenizer:")

    return f'''\
from __future__ import annotations
from typing import Any
import tiktoken
from transformers import PreTrainedTokenizer

{core}

class SteerlingTokenizer(PreTrainedTokenizer):
    vocab_files_names: dict[str, str] = {{}}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, encoding_name="{enc}", pad_token_id={tok.pad_token_id},
                 bos_token_id={tok.bos_token_id}, eos_token_id={tok.eos_token_id},
                 endofchunk_token_id={tok.endofchunk_token_id}, mask_token_id={tok.mask_token_id}, **kwargs):
        self._core = _SteerlingTokenizer()
        self._endofchunk_token_id = endofchunk_token_id
        self._mask_token_id = mask_token_id
        for k in ("pad_token", "bos_token", "eos_token", "additional_special_tokens"):
            kwargs.pop(k, None)
        super().__init__(pad_token="<|pad|>", bos_token="<|bos|>", eos_token="<|endoftext|>",
                         additional_special_tokens=["<|endofchunk|>", "<|mask|>"], **kwargs)

    @property
    def vocab_size(self): return self._core.vocab_size
    @property
    def endofchunk_token_id(self): return self._core.endofchunk_token_id
    @property
    def mask_token_id(self): return self._core.mask_token_id

    def get_vocab(self): return dict(self._core._tokenizer._special_tokens)

    def _tokenize(self, text, **kwargs):
        return [str(i) for i in self._core._tokenizer.encode(text, disallowed_special=())]

    def _convert_token_to_id(self, token):
        special = self._core._tokenizer._special_tokens
        if token in special: return special[token]
        try: return int(token)
        except ValueError:
            ids = self._core._tokenizer.encode(token, disallowed_special=())
            return ids[0] if ids else self._core.pad_token_id

    def _convert_id_to_token(self, index):
        for name, idx in self._core._tokenizer._special_tokens.items():
            if idx == index: return name
        try: return self._core._tokenizer.decode([index])
        except Exception: return f"<|token_{{index}}|>"

    def convert_tokens_to_string(self, tokens):
        ids, special = [], self._core._tokenizer._special_tokens
        for t in tokens:
            if t in special: continue
            try:
                tid = int(t)
                if tid not in self._core._special_token_ids: ids.append(tid)
            except ValueError:
                ids.extend(self._core._tokenizer.encode(t, disallowed_special=()))
        return self._core._tokenizer.decode(ids)

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return self._core.decode(list(token_ids) if not isinstance(token_ids, list) else token_ids,
                                 skip_special_tokens=skip_special_tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return token_ids_0

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()
'''


def build_modeling_py() -> str:
    sources = [
        "steerling/models/layers/primitives.py",
        "steerling/models/layers/causal_diffusion_layers.py",
        "steerling/models/causal_diffusion.py",
        "steerling/models/interpretable/outputs.py",
        "steerling/models/interpretable/concept_head.py",
        "steerling/models/interpretable/interpretable_causal_diffusion.py",
    ]

    _drop = re.compile(r"^\s*(from \.|from steerling|import steerling)")
    _keep = re.compile(r"from \.configuration_steerling import")

    seen: set[str] = set()
    all_imports: list[str] = []
    all_body: list[str] = []

    for path in sources:
        src = _strip_docstrings(_read_source(path))
        lines = src.splitlines()

        imports, body = [], []
        past = False
        skip_type_checking = False

        for line in lines:
            if past:
                body.append(line)
                continue
            stripped = line.strip()

            if stripped == "if TYPE_CHECKING:":
                skip_type_checking = True
                continue
            if skip_type_checking:
                if stripped and not line[0].isspace():
                    skip_type_checking = False
                else:
                    continue

            if re.match(r"^(import |from )", stripped):
                if _drop.match(stripped) and not _keep.match(stripped):
                    continue
                if line not in seen:
                    seen.add(line)
                    imports.append(line)
            elif stripped == "" or stripped.startswith("#"):
                pass
            else:
                past = True
                body.append(line)

        all_imports.extend(imports)
        all_body.append(f"\n# {'=' * 70}")
        all_body.append(f"# {path}")
        all_body.append(f"# {'=' * 70}\n")
        all_body.extend(body)

    future = [line for line in all_imports if "from __future__" in line]
    rest   = [line for line in all_imports if "from __future__" not in line]

    header = "# Auto-generated by scripts/build_hf_files_v3.py — do not edit manually.\n\n"
    body_str = "\n".join(all_body)

    hf_wrapper = (
        "\n"
        "from transformers import PreTrainedModel\n"
        "from .configuration_steerling import SteerlingConfig\n"
        "\n"
        "\n"
        "# CausalDiffusionLM is the backbone — alias to HF-friendly name\n"
        "SteerlingBackbone = CausalDiffusionLM\n"
        "\n"
        "\n"
        "class SteerlingForCausalLM(PreTrainedModel):\n"
        "    config_class = SteerlingConfig\n"
        "    supports_gradient_checkpointing = False\n"
        '    _tied_weights_keys = ["transformer.lm_head.weight"]\n'
        "\n"
        "    def __init__(self, config: SteerlingConfig):\n"
        "        super().__init__(config)\n"
        "        # SteerlingConfig has all fields from both arch and concept configs\n"
        "        self.concept_config = config\n"
        "        self.transformer = SteerlingBackbone(config, config.vocab_size)\n"
        "        self.known_head = ConceptHead(\n"
        "            n_concepts=config.n_concepts,\n"
        "            concept_dim=config.concept_dim,\n"
        "            n_embd=config.n_embd,\n"
        "            is_unknown=False,\n"
        "            use_attention=config.use_attention_known,\n"
        "            topk=config.topk_known,\n"
        "            topk_features=config.topk_known_features,\n"
        "            block_size=config.concept_block_size,\n"
        "            pad_multiple=config.pad_multiple,\n"
        "            store_unknown_weights=False,\n"
        "            apply_topk_to_unknown=False,\n"
        "            topk_on_logits=config.topk_on_logits,\n"
        "            factorize=False,\n"
        "        )\n"
        "        if config.use_unknown:\n"
        "            self.unknown_head = ConceptHead(\n"
        "                n_concepts=config.n_unknown_concepts,\n"
        "                concept_dim=config.concept_dim,\n"
        "                n_embd=config.n_embd,\n"
        "                is_unknown=True,\n"
        "                use_attention=config.use_attention_unknown,\n"
        "                topk=config.unknown_topk,\n"
        "                block_size=config.concept_block_size,\n"
        "                pad_multiple=config.pad_multiple,\n"
        "                store_unknown_weights=config.store_unknown_weights,\n"
        "                apply_topk_to_unknown=config.apply_topk_to_unknown,\n"
        "                topk_on_logits=config.topk_on_logits,\n"
        "                factorize=config.factorize_unknown,\n"
        "                factorize_rank=config.factorize_rank,\n"
        "            )\n"
        "        else:\n"
        "            self.unknown_head = None\n"
        "        self.post_init()\n"
        "\n"
        "    def _init_weights(self, module):\n"
        "        pass\n"
        "\n"
        "    def _tie_weights(self):\n"
        "        if self.config.weight_sharing:\n"
        "            self.transformer.lm_head.weight = self.transformer.tok_emb.weight\n"
        "\n"
        "    def forward(self, input_ids=None, **kwargs):\n"
        "        if self.config.interpretable:\n"
        "            return InterpretableCausalDiffusionLM.forward(self, input_ids, **kwargs)\n"
        "        else:\n"
        "            kwargs.pop('minimal_output', None)\n"
        "            return CausalDiffusionLM.forward(self, input_ids, **kwargs)\n"
    )

    return (
        "\n".join(future) + "\n"
        + header
        + "\n".join(rest) + "\n\n"
        + body_str
        + hf_wrapper
    )


# ===========================================================================
# File registry & entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to scalex training config file (defines model_config and concept_config)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "hf")
    parser.add_argument("--files", nargs="+", default=None)
    args = parser.parse_args()

    arch, concept = _load_training_config(args.config)

    FILES = {
        "config.json":                lambda: compact_json(build_config_json(arch, concept)),
        "tokenizer_config.json":      lambda: compact_json(build_tokenizer_config_json()),
        "configuration_steerling.py": lambda: build_configuration_py(arch, concept),
        "tokenization_steerling.py":  build_tokenization_py,
        "modeling_steerling.py":      build_modeling_py,
    }

    if args.files:
        unknown = set(args.files) - set(FILES)
        if unknown:
            parser.error(f"Unknown files: {unknown}. Valid: {list(FILES)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    to_gen = args.files or list(FILES)

    print(f"Generating HF files in {args.output_dir}/")
    for fname in to_gen:
        try:
            content = FILES[fname]()
            if fname.endswith(".py"):
                ast.parse(content)
            (args.output_dir / fname).write_text(content)
            print(f"  ✓ {fname} ({len(content.splitlines())} lines)")
        except SyntaxError as e:
            print(f"  ✗ {fname} — SYNTAX ERROR line {e.lineno}: {e.msg}")
            print(f"    {e.text}")
            raise
        except Exception as e:
            print(f"  ✗ {fname} — ERROR: {e}")
            raise
    print("Done.")


if __name__ == "__main__":
    main()