import warnings
from functools import partial

import torch
from transformers import AutoConfig, PretrainedConfig


def resolve_verifier_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the decoder config used by a verifier for text generation.

    Most multimodal verifiers expose ``text_config`` at the root. Qwen3-Omni
    has one additional model-stage level instead::

        Qwen3OmniMoeConfig.thinker_config.text_config

    Speculator training targets the Thinker decoder, not the Talker codec
    decoder, so resolve the Thinker explicitly before applying the common
    ``text_config`` fallback.
    """
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is not None:
        config = thinker_config

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        config = text_config

    return config


def resolve_verifier_weight_prefix(config: PretrainedConfig) -> str | None:
    """Return the checkpoint prefix for the verifier's generation stage."""
    if getattr(config, "thinker_config", None) is not None:
        return "thinker."
    return None


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(verifier_name_or_path: str) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
    return resolve_verifier_text_config(verifier_config)


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
) -> list[int]:
    if target_layer_ids is not None:
        return target_layer_ids

    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    target_layer_ids = [2, num_layers // 2, num_layers - 3]
    warnings.warn(
        DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
        stacklevel=3,
    )
    return target_layer_ids


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers either omit ``intermediate_size`` or use it for one routed
      expert. That width must not be reused for the independent dense draft, which
      falls back to the widely used ``3 * hidden_size`` gated-MLP ratio. Pass
      ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    expert_count_fields = (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
    )
    is_moe = any(
        (count := getattr(verifier_config, field, None)) is not None and int(count) > 0
        for field in expert_count_fields
    )

    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None and not is_moe:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier is MoE or has no dense intermediate_size; using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size
