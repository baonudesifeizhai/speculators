"""Unit tests for the DSpark loss and metrics."""

import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; position 0 is the anchor (masked), position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the anchor token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_sample_from_anchor_counts_position_zero(self):
        # DSpark's slot 0 is a real speculative prediction. With two perfect
        # draft slots, the expected emitted length is 2 accepted + 1 verifier
        # fallback/bonus token.
        ids = torch.tensor([[1, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.ones((1, 2), dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=True,
        )
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 3.0) < 1e-2

    def test_raw_tv_is_exact_acceptance_complement(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        targets = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
        loss_mask = torch.ones((1, 2), dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        raw_tv = metrics["raw_tv_sum"] / metrics["raw_tv_total"]
        assert torch.allclose(accept + raw_tv, torch.tensor(1.0), atol=1e-6)

    def test_modality_metrics_are_separate(self):
        # First block (text) is perfect; second block (image) is fully wrong.
        draft_ids = torch.tensor([[0, 1, 2, 3]])
        target_ids = torch.tensor([[0, 1, 4, 5]])
        logits = _ids_to_logits(draft_ids, 8)
        targets = _ids_to_logits(target_ids, 8)
        loss_mask = torch.ones((1, 4), dtype=torch.float32)
        modality_ids = torch.tensor([[0, 0, 1, 1]])
        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            modality_ids=modality_ids,
            modality_names=("text", "image", "audio", "video"),
        )
        text_accept = (
            metrics["accept_rate_text_sum"] / metrics["accept_rate_text_total"]
        )
        image_accept = (
            metrics["accept_rate_image_sum"] / metrics["accept_rate_image_total"]
        )
        assert float(text_accept) > 0.99
        assert float(image_accept) < 0.01
        assert float(metrics["accept_rate_audio_total"]) == 0.0
        assert "accept_rate_video_position_0_sum" in metrics

    def test_modality_router_loss_and_per_modality_accuracy(self):
        ids = torch.tensor([[0, 1, 2, 3]])
        logits = _ids_to_logits(ids, 8)
        loss_mask = torch.ones((1, 4), dtype=torch.float32)
        modality_ids = torch.tensor([[0, 0, 1, 1]])
        router_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        base_loss, _ = compute_metrics(
            logits,
            logits,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            modality_ids=modality_ids,
            modality_names=("text", "image"),
        )
        routed_loss, metrics = compute_metrics(
            logits,
            logits,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            modality_ids=modality_ids,
            modality_names=("text", "image"),
            modality_router_logits=router_logits,
            modality_router_alpha=0.1,
        )
        assert float(routed_loss - base_loss) < 1e-3
        acc = metrics["modality_router_acc_sum"] / metrics["modality_router_acc_total"]
        assert float(acc) == 1.0
        assert float(metrics["modality_router_acc_text_total"]) == 1.0

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}'),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}'),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "raw_tv_sum",
            "accept_rate_position_0_sum",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())
