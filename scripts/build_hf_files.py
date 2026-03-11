#!/usr/bin/env python3
"""Generate HF-compatible files for Steerling.

Dynamically reads model/tokenizer/concept configs from the steerling package
and inlines source files — no hardcoded values.

Usage:
    python scripts/build_hf_files.py
    python scripts/build_hf_files.py --output-dir custom/hf/path
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


def _pydantic_defaults(model_cls) -> dict:
    return {
        name: field.default
        for name, field in model_cls.model_fields.items()
        if not field.is_required()
    }


def _strip_docstrings(source: str) -> str:
    """Remove all docstrings from Python source via AST round-trip."""
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
        elements = [l.strip().rstrip(",") for l in m.group(1).strip().splitlines()]
        return "[" + ", ".join(elements) + "]"
    return re.sub(r'\[\s*\n((?:\s+(?:"[^"]*"|null),?\s*\n)+)\s*\]', _collapse, raw) + "\n"


# ===========================================================================
# JSON configs
# ===========================================================================

def build_config_json() -> dict:
    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.configs.concept import ConceptConfig
    from steerling.data.tokenizer import SteerlingTokenizer

    arch = _pydantic_defaults(CausalDiffusionConfig)
    concept = _pydantic_defaults(ConceptConfig)
    tok = SteerlingTokenizer()

    return {
        "model_type": "steerling",
        "auto_map": {
            "AutoConfig": "configuration_steerling.SteerlingConfig",
            "AutoModel": "modeling_steerling.SteerlingForCausalLM",
            "AutoModelForCausalLM": "modeling_steerling.SteerlingForCausalLM",
            "AutoTokenizer": ["tokenization_steerling.SteerlingTokenizer", None],
        },
        "architectures": ["SteerlingForCausalLM"],
        "vocab_size": tok.vocab_size,
        "n_layers": arch["n_layers"],
        "n_head": arch["n_head"],
        "n_embd": arch["n_embd"],
        "n_kv_heads": arch["n_kv_heads"],
        "block_size": arch["block_size"],
        "diff_block_size": arch["diff_block_size"],
        "use_rms_norm": arch["use_rms_norm"],
        "norm_eps": arch["norm_eps"],
        "norm_order": arch["norm_order"],
        "use_qk_norm": arch["use_qk_norm"],
        "use_rope": arch["use_rope"],
        "rope_base": arch["rope_base"],
        "rope_full_precision": arch["rope_full_precision"],
        "clip_qkv": arch["clip_qkv"],
        "mlp_type": arch["mlp_type"],
        "activation": arch["activation"],
        "mlp_ratio": arch["mlp_ratio"],
        "intermediate_size": arch["intermediate_size"],
        "use_bias": arch["use_bias"],
        "weight_sharing": arch["weight_sharing"],
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "mask_token_id": tok.mask_token_id,
        "endofchunk_token_id": tok.endofchunk_token_id,
        "n_concepts": concept["n_concepts"],
        "n_unknown_concepts": concept["n_unknown_concepts"],
        "concept_dim": concept["concept_dim"],
        "use_attention_known": concept["use_attention_known"],
        "use_attention_unknown": concept["use_attention_unknown"],
        "topk_known": concept["topk_known"],
        "topk_known_features": concept["topk_known_features"],
        "unknown_topk": concept["unknown_topk"],
        "use_unknown": concept["use_unknown"],
        "apply_topk_to_unknown": concept["apply_topk_to_unknown"],
        "topk_on_logits": concept["topk_on_logits"],
        "factorize_unknown": concept["factorize_unknown"],
        "factorize_rank": concept["factorize_rank"],
        "use_epsilon_correction": concept["use_epsilon_correction"],
        "concept_block_size": concept["block_size"],
        "pad_multiple": concept["pad_multiple"],
        "store_unknown_weights": concept["store_unknown_weights"],
        "inject_layer": concept["inject_layer"],
        "inject_alpha": concept["inject_alpha"],
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

def build_configuration_py() -> str:
    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.configs.concept import ConceptConfig
    from steerling.data.tokenizer import SteerlingTokenizer

    arch = _pydantic_defaults(CausalDiffusionConfig)
    concept = _pydantic_defaults(ConceptConfig)
    tok = SteerlingTokenizer()

    assigns = [
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
        if isinstance(v, str): return f'"{v}"'
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

    # Read core tokenizer, strip docstrings and steerling-internal imports
    core = _read_source("steerling/data/tokenizer.py")
    core = _strip_docstrings(core)
    # Remove numpy/torch deps not needed in HF wrapper
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
        # Strip docstrings FIRST before any other processing
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

            # Skip TYPE_CHECKING blocks with internal imports
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
                pass  # skip blanks/comments between imports
            else:
                past = True
                body.append(line)

        all_imports.extend(imports)
        all_body.append(f"\n# {'=' * 70}")
        all_body.append(f"# {path}")
        all_body.append(f"# {'=' * 70}\n")
        all_body.extend(body)

    future = [l for l in all_imports if "from __future__" in l]
    rest   = [l for l in all_imports if "from __future__" not in l]

    header = '''\
# Auto-generated by scripts/build_hf_files.py — do not edit manually.
# Shims for config type hints
class _Cfg:
    def __getattr__(self, n): return None
CausalDiffusionConfig = _Cfg
ConceptConfig = _Cfg

'''

    body = "\n".join(all_body)

    # PreTrainedModel wrapper so HF's from_pretrained / register_for_auto_class work
    hf_wrapper = '''
from transformers import PreTrainedModel
from .configuration_steerling import SteerlingConfig


class SteerlingForCausalLM(PreTrainedModel):
    config_class = SteerlingConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: SteerlingConfig):
        super().__init__(config)
        from steerling.configs.causal_diffusion import CausalDiffusionConfig as _ArchCfg
        from steerling.configs.concept import ConceptConfig as _ConceptCfg

        arch = _ArchCfg(
            n_layers=config.n_layers, n_head=config.n_head, n_embd=config.n_embd,
            n_kv_heads=config.n_kv_heads, block_size=config.block_size,
            diff_block_size=config.diff_block_size, use_rms_norm=config.use_rms_norm,
            norm_eps=config.norm_eps, norm_order=config.norm_order,
            use_qk_norm=config.use_qk_norm, use_rope=config.use_rope,
            rope_base=config.rope_base, rope_full_precision=config.rope_full_precision,
            clip_qkv=config.clip_qkv, mlp_type=config.mlp_type,
            activation=config.activation, mlp_ratio=config.mlp_ratio,
            intermediate_size=config.intermediate_size, use_bias=config.use_bias,
            weight_sharing=config.weight_sharing,
        )
        concept = _ConceptCfg(
            n_concepts=config.n_concepts, n_unknown_concepts=config.n_unknown_concepts,
            concept_dim=config.concept_dim, use_attention_known=config.use_attention_known,
            use_attention_unknown=config.use_attention_unknown, topk_known=config.topk_known,
            topk_known_features=config.topk_known_features, unknown_topk=config.unknown_topk,
            use_unknown=config.use_unknown, apply_topk_to_unknown=config.apply_topk_to_unknown,
            topk_on_logits=config.topk_on_logits, factorize_unknown=config.factorize_unknown,
            factorize_rank=config.factorize_rank, use_epsilon_correction=config.use_epsilon_correction,
            block_size=config.concept_block_size, pad_multiple=config.pad_multiple,
            store_unknown_weights=config.store_unknown_weights,
            inject_layer=config.inject_layer, inject_alpha=config.inject_alpha,
        )
        self.model = InterpretableCausalDiffusionLM(arch, concept, config.vocab_size)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return self.model(input_ids, **kwargs)

    def _init_weights(self, module):
        pass  # weights loaded from checkpoint
'''

    return (
        "\n".join(future) + "\n"
        + header
        + "\n".join(rest) + "\n\n"
        + body
        + hf_wrapper
    )


# ===========================================================================
# File registry & entry point
# ===========================================================================

FILES = {
    "config.json":                lambda: compact_json(build_config_json()),
    "tokenizer_config.json":      lambda: compact_json(build_tokenizer_config_json()),
    "configuration_steerling.py": build_configuration_py,
    "tokenization_steerling.py":  build_tokenization_py,
    "modeling_steerling.py":      build_modeling_py,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "hf")
    parser.add_argument("--files", nargs="+", choices=list(FILES), default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    to_gen = args.files or list(FILES)

    print(f"Generating HF files in {args.output_dir}/")
    for fname in to_gen:
        try:
            content = FILES[fname]()
            if fname.endswith(".py"):
                ast.parse(content)  # validate before writing
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