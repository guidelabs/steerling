"""Tests for configuration classes."""

import tempfile

import pytest
from pydantic import ValidationError

from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.configs.generation import GenerationConfig


class TestCausalDiffusionConfig:
    def test_defaults(self):
        config = CausalDiffusionConfig()
        assert config.n_layers == 32
        assert config.n_head == 32
        assert config.n_embd == 4096
        assert config.weight_sharing is True

    def test_validation_embd_head(self):
        with pytest.raises(ValueError, match="divisible"):
            CausalDiffusionConfig(n_embd=100, n_head=3)

    def test_validation_kv_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            CausalDiffusionConfig(n_head=32, n_kv_heads=5)

    def test_json_roundtrip(self, tiny_config):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            tiny_config.to_json(f.name)
            loaded = CausalDiffusionConfig.from_json(f.name)
        assert loaded.n_layers == tiny_config.n_layers
        assert loaded.n_embd == tiny_config.n_embd

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CausalDiffusionConfig(n_layers=32, bogus_field=True)


class TestConceptConfig:
    def test_defaults(self):
        config = ConceptConfig()
        assert config.n_concepts == 33732
        assert config.factorize_unknown is True
        assert config.inject_layer == 16

    def test_tiny(self, tiny_concept_config):
        assert tiny_concept_config.n_concepts == 32
        assert tiny_concept_config.factorize_unknown is False


class TestGenerationConfig:
    def test_defaults(self):
        config = GenerationConfig()
        assert config.max_new_tokens == 1024
        assert config.steps is None
        assert config.temperature == 1.2
        assert config.cfg_scale == 0.0
        assert config.seed is None
        assert config.stop_tokens is None

    def test_stop_tokens(self):
        config = GenerationConfig(stop_tokens=[1, 2, 3])
        assert config.stop_tokens == [1, 2, 3]

    def test_validation(self):
        with pytest.raises(ValidationError):
            GenerationConfig(max_new_tokens=-1)
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            GenerationConfig(bogus_field=True)
