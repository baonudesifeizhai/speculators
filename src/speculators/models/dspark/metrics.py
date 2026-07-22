"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    softmax,
)

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    dpace_loss_decay,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(  # noqa: C901
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
    modality_ids: torch.Tensor | None = None,  # [1, T] or None
    modality_names: tuple[str, ...] | None = None,
    modality_router_logits: torch.Tensor | None = None,  # [num_blocks, num_modalities]
    modality_router_alpha: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
    )

    # Analytical per-position acceptance rate = distributional overlap.
    with torch.no_grad():
        draft_p = softmax(logits.float(), dim=-1)
        target_p = softmax(targets.float(), dim=-1)
        accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)  # [1, T]
        # Per-block cumulative acceptance product over the actual draft slots,
        # shared by the accept-length and calibration metrics. DSpark predicts
        # from slot 0; DFlash-style training reserves slot 0 for the anchor.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        mask_blocks = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)
        draft_start = 0 if sample_from_anchor else 1
        draft_mask = mask_blocks[:, draft_start:]
        draft_accept = accept_blocks[:, draft_start:]
        accept_prefix = (draft_accept * draft_mask).cumprod(dim=-1)
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, draft_start:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    if modality_router_logits is not None:
        if modality_ids is None or modality_names is None:
            raise ValueError(
                "modality IDs and names are required with modality router logits"
            )
        if modality_ids.shape != accept_rate.shape:
            raise ValueError(
                "modality_ids must match acceptance-rate shape, got "
                f"{modality_ids.shape} and {accept_rate.shape}"
            )
        modality_ids = modality_ids.to(device=device, dtype=torch.long)
        if modality_router_logits.shape != (num_blocks, len(modality_names)):
            raise ValueError(
                "modality_router_logits must have shape "
                f"({num_blocks}, {len(modality_names)}), got "
                f"{tuple(modality_router_logits.shape)}"
            )

        router_targets = modality_ids.view(num_blocks, block_size)[:, 0].long()
        # Equalize the total contribution of every modality present in this
        # packed batch so minority audio/video routes cannot be ignored.
        valid_targets = router_targets * block_valid.long()
        counts = torch.bincount(
            valid_targets,
            weights=block_valid,
            minlength=len(modality_names),
        ).to(modality_router_logits.dtype)
        present = (counts > 0).to(counts.dtype)
        class_weights = torch.where(
            counts > 0,
            block_valid.sum() / (present.sum().clamp_min(1.0) * counts.clamp_min(1.0)),
            torch.zeros_like(counts),
        )
        router_ce = cross_entropy(
            modality_router_logits.float(),
            router_targets,
            weight=class_weights.float(),
            reduction="none",
        )
        router_loss = (router_ce * block_valid).sum() / block_valid.sum().clamp_min(1.0)
        loss = loss + modality_router_alpha * router_loss

        with torch.no_grad():
            router_pred = modality_router_logits.argmax(dim=-1)
            router_correct = (router_pred == router_targets).to(block_valid.dtype)
            metrics["modality_router_loss_sum"] = router_loss.detach().clone()
            metrics["modality_router_loss_total"] = torch.ones((), device=device)
            metrics["modality_router_acc_sum"] = (router_correct * block_valid).sum()
            metrics["modality_router_acc_total"] = block_valid.sum()
            for modality_id, name in enumerate(modality_names):
                route_mask = block_valid * (router_targets == modality_id).to(
                    block_valid.dtype
                )
                metrics[f"modality_router_acc_{name}_sum"] = (
                    router_correct * route_mask
                ).sum()
                metrics[f"modality_router_acc_{name}_total"] = route_mask.sum()

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)
        # Unlike the position-decayed TV training loss, this is the unweighted
        # per-token TV distance. It therefore obeys raw_tv = 1 - accept_rate.
        metrics["raw_tv_sum"] = ((1.0 - accept_rate) * mask_f).sum()
        metrics["raw_tv_total"] = mask_f.sum().clamp_min(1.0)

        for pos in range(draft_start, block_size):
            pos_mask = mask_blocks[:, pos]
            metrics[f"accept_rate_position_{pos}_sum"] = (
                accept_blocks[:, pos] * pos_mask
            ).sum()
            metrics[f"accept_rate_position_{pos}_total"] = pos_mask.sum()

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Split the same diagnostics by routed input modality. Do not clamp subgroup
    # totals: after DDP reduction, an absent modality must contribute zero rather
    # than a fake denominator of one.
    if modality_ids is not None:
        if modality_names is None:
            raise ValueError("modality_names is required when modality_ids is set")
        if modality_ids.shape != accept_rate.shape:
            raise ValueError(
                "modality_ids must match acceptance-rate shape, got "
                f"{modality_ids.shape} and {accept_rate.shape}"
            )

        with torch.no_grad():
            modality_ids = modality_ids.to(device=device, dtype=torch.long)
            modality_blocks = modality_ids.view(num_blocks, block_size)
            block_modality_ids = modality_blocks[:, 0]
            for modality_id, name in enumerate(modality_names):
                modality_mask = (modality_ids == modality_id).to(mask_f.dtype)
                token_mask = mask_f * modality_mask
                metrics[f"accept_rate_{name}_sum"] = (accept_rate * token_mask).sum()
                metrics[f"accept_rate_{name}_total"] = token_mask.sum()
                metrics[f"raw_tv_{name}_sum"] = ((1.0 - accept_rate) * token_mask).sum()
                metrics[f"raw_tv_{name}_total"] = token_mask.sum()

                for pos in range(draft_start, block_size):
                    pos_mask = mask_blocks[:, pos] * (
                        modality_blocks[:, pos] == modality_id
                    ).to(mask_blocks.dtype)
                    metrics[f"accept_rate_{name}_position_{pos}_sum"] = (
                        accept_blocks[:, pos] * pos_mask
                    ).sum()
                    metrics[f"accept_rate_{name}_position_{pos}_total"] = pos_mask.sum()

                modality_block_mask = block_valid * (
                    block_modality_ids == modality_id
                ).to(block_valid.dtype)
                metrics[f"accept_len_{name}_sum"] = (
                    per_block_len * modality_block_mask
                ).sum()
                metrics[f"accept_len_{name}_total"] = modality_block_mask.sum()

    # Per-position greedy accuracy
    # Start position: 0 if sample_from_anchor else 1 (skip anchor)
    start_pos = 0 if sample_from_anchor else 1
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics
