from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import (
    MODALITY_NAMES,
    ConfidenceHead,
    MarkovHead,
    ModalityLogitHead,
    ModalityRouter,
    infer_anchor_modalities,
)
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

        self.modality_heads = nn.ModuleDict()
        self.modality_router: ModalityRouter | None = None
        if config.modality_head_rank > 0:
            missing = set(MODALITY_NAMES[1:]) - set(config.modality_token_ids)
            if missing:
                raise ValueError(
                    "modality_token_ids is missing routes for: "
                    + ", ".join(sorted(missing))
                )
            self.modality_heads = nn.ModuleDict(
                {
                    name: ModalityLogitHead(
                        hidden_size,
                        self.draft_vocab_size,
                        config.modality_head_rank,
                    )
                    for name in MODALITY_NAMES
                }
            )
            self.modality_router = ModalityRouter(hidden_size, len(MODALITY_NAMES))

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        modality_head_rank = kwargs.get("modality_head_rank", 0)
        modality_token_ids = kwargs.get("modality_token_ids") or {}
        if modality_head_rank > 0 and not modality_token_ids:
            modality_token_ids = cls._resolve_modality_token_ids(
                kwargs["verifier_name_or_path"]
            )

        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            modality_head_rank=modality_head_rank,
            modality_token_ids=modality_token_ids,
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _resolve_modality_token_ids(verifier_name_or_path: str) -> dict[str, list[int]]:
        """Resolve Qwen-style modality pad IDs from the verifier tokenizer."""
        from speculators.data_generation.preprocessing import (  # noqa: PLC0415
            get_tokenizer,
            load_processor,
        )

        processor = load_processor(verifier_name_or_path)
        tokenizer = get_tokenizer(processor)
        token_names = {
            "image": "<|image_pad|>",
            "audio": "<|audio_pad|>",
            "video": "<|video_pad|>",
        }
        vocab = tokenizer.get_vocab()
        missing = [token for token in token_names.values() if token not in vocab]
        if missing:
            raise ValueError(
                "Could not enable modality heads because the verifier tokenizer "
                f"does not define: {', '.join(missing)}"
            )
        return {name: [int(vocab[token])] for name, token in token_names.items()}

    @torch.compiler.disable
    def _apply_modality_heads(
        self,
        hidden_blocks: torch.Tensor,
        logits_blocks: torch.Tensor,
        block_modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exactly one low-rank residual head to each anchor block."""
        residual = torch.zeros_like(logits_blocks)
        for modality_id, name in enumerate(MODALITY_NAMES):
            selected = block_modality_ids == modality_id
            if torch.any(selected):
                residual[selected] = self.modality_heads[name](
                    hidden_blocks[selected]
                ).to(residual.dtype)
        return logits_blocks + residual

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        modality_router_alpha = kwargs.get("modality_router_alpha", 0.1)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "modality_router_alpha": modality_router_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        modality_router_alpha: float = 0.1,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        aligned_modality_ids = None
        modality_router_logits = None
        if self.modality_heads:
            anchor_positions = anchored_block_indices.view(num_blocks, block)[:, 0]
            block_modality_ids = infer_anchor_modalities(
                input_ids,
                document_ids,
                anchor_positions,
                self.config.modality_token_ids,
            )
            if self.modality_router is None:
                raise RuntimeError("modality heads require a modality router")
            modality_router_logits = self.modality_router(hidden_blocks)
            logits_blocks = logits.view(num_blocks, block, -1)
            logits = self._apply_modality_heads(
                hidden_blocks,
                logits_blocks,
                block_modality_ids,
            ).view(1, mask_tokens_size, -1)
            aligned_modality_ids = block_modality_ids.repeat_interleave(block).view(
                1, mask_tokens_size
            )

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            modality_ids=aligned_modality_ids,
            modality_names=MODALITY_NAMES if aligned_modality_ids is not None else None,
            modality_router_logits=modality_router_logits,
            modality_router_alpha=modality_router_alpha,
        )
        return None, loss, metrics
