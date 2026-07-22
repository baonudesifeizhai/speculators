"""Unit tests for speculators.models.utils config-resolution helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig

from speculators.models.utils import (
    resolve_draft_intermediate_size,
    resolve_verifier_text_config,
    resolve_verifier_weight_prefix,
)


def _fake_verifier(**fields) -> PretrainedConfig:
    """Lightweight stand-in verifier config (the resolver only reads attributes)."""
    return cast("PretrainedConfig", SimpleNamespace(**fields))


@pytest.mark.smoke
def test_resolve_verifier_text_config_prefers_qwen3_omni_thinker():
    thinker_text = _fake_verifier(hidden_size=2048, num_hidden_layers=48)
    talker_text = _fake_verifier(hidden_size=1024, num_hidden_layers=20)
    root = _fake_verifier(
        thinker_config=_fake_verifier(text_config=thinker_text),
        talker_config=_fake_verifier(text_config=talker_text),
    )

    assert resolve_verifier_text_config(root) is thinker_text


@pytest.mark.smoke
def test_resolve_verifier_text_config_keeps_common_multimodal_fallback():
    text_config = _fake_verifier(hidden_size=4096)
    root = _fake_verifier(text_config=text_config)

    assert resolve_verifier_text_config(root) is text_config


@pytest.mark.smoke
def test_resolve_verifier_weight_prefix_targets_qwen3_omni_thinker():
    root = _fake_verifier(thinker_config=_fake_verifier())

    assert resolve_verifier_weight_prefix(root) == "thinker."
    assert resolve_verifier_weight_prefix(_fake_verifier()) is None


# ---------------------------------------------------------------------------
# resolve_draft_intermediate_size
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_resolve_uses_dense_intermediate_size_directly():
    # A dense verifier's intermediate_size is mirrored verbatim, even when a
    # hidden_size is also present (dense takes precedence over the 3x fallback).
    verifier = _fake_verifier(intermediate_size=11008, hidden_size=4096)

    assert resolve_draft_intermediate_size(verifier) == 11008


@pytest.mark.smoke
def test_resolve_moe_falls_back_to_3x_hidden_size():
    # MoE verifier: no dense intermediate_size -> draft uses 3 * hidden_size.
    verifier = _fake_verifier(hidden_size=2048)

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 6144


@pytest.mark.smoke
def test_resolve_moe_does_not_reuse_per_expert_intermediate_size():
    verifier = _fake_verifier(
        hidden_size=2048,
        intermediate_size=768,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
    )

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 6144


@pytest.mark.smoke
def test_resolve_ignores_moe_expert_fields():
    # Expert fields are irrelevant now: with no dense intermediate_size the draft
    # width is purely 3 * hidden_size regardless of the MoE routing config.
    verifier = _fake_verifier(
        hidden_size=1024,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        shared_expert_intermediate_size=2048,
    )

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 3072


@pytest.mark.smoke
def test_resolve_requires_intermediate_or_hidden_size():
    # Degenerate config with neither field -> explicit error pointing at --draft-config.
    verifier = _fake_verifier()

    with pytest.raises(ValueError, match="--draft-config"):
        resolve_draft_intermediate_size(verifier)
