"""Unit tests for DSpark's Markov, confidence, and modality heads."""

import pytest
import torch

from speculators.models.dspark.model_definitions import (
    ConfidenceHead,
    MarkovHead,
    ModalityLogitHead,
    ModalityRouter,
    infer_anchor_modalities,
)


class TestMarkovHead:
    def _head(self, head_type="vanilla", r=8, vv=50, dv=20, h=16):
        torch.manual_seed(0)
        return MarkovHead(
            verifier_vocab_size=vv,
            draft_vocab_size=dv,
            markov_rank=r,
            hidden_size=h,
            head_type=head_type,
        )

    @pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
    def test_block_bias_shape(self, head_type):
        head = self._head(head_type)
        n, b, h = 3, 4, 16
        prev = torch.randint(0, 50, (n, b))
        hidden = torch.randn(n, b, h)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        assert bias.shape == (n, b, 20)
        assert torch.isfinite(bias).all()

    def test_vanilla_is_low_rank_factorization(self):
        head = self._head("vanilla")
        prev = torch.randint(0, 50, (2, 4))
        hidden = torch.zeros(2, 4, 16)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        expected = head.markov_w2(head.markov_w1(prev))
        assert torch.allclose(bias, expected, atol=1e-5)

    def test_bias_depends_on_prev_token(self):
        head = self._head("vanilla")
        hidden = torch.zeros(1, 1, 16)
        bias_a = head.block_bias(
            prev_token_ids=torch.tensor([[1]]), hidden_states=hidden
        )
        bias_b = head.block_bias(
            prev_token_ids=torch.tensor([[2]]), hidden_states=hidden
        )
        assert not torch.allclose(bias_a, bias_b)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=0,
                hidden_size=16,
            )

    def test_invalid_head_type_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=8,
                hidden_size=16,
                head_type="bogus",
            )


class TestConfidenceHead:
    def test_output_shape(self):
        head = ConfidenceHead(input_dim=24)
        features = torch.randn(3, 4, 24)
        out = head(features)
        assert out.shape == (3, 4)
        assert torch.isfinite(out).all()


class TestModalityLogitHead:
    def test_zero_initialized_residual_and_trainable_output(self):
        head = ModalityLogitHead(hidden_size=16, draft_vocab_size=20, rank=4)
        hidden = torch.randn(3, 2, 16)
        out = head(hidden)
        assert out.shape == (3, 2, 20)
        assert torch.count_nonzero(out) == 0
        out.sum().backward()
        assert head.up_proj.weight.grad is not None
        assert torch.count_nonzero(head.up_proj.weight.grad) > 0

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            ModalityLogitHead(hidden_size=16, draft_vocab_size=20, rank=0)


class TestModalityRouter:
    def test_routes_one_label_per_block(self):
        router = ModalityRouter(hidden_size=16, num_modalities=4)
        hidden = torch.randn(3, 7, 16)
        assert router(hidden).shape == (3, 4)


class TestInferAnchorModalities:
    def test_routes_packed_documents_from_special_tokens(self):
        # Four packed documents: text, image, audio, video. The multimodal pad
        # tokens can occur anywhere in the prompt portion of each document.
        input_ids = torch.tensor(
            [[10, 11, 12, 101, 13, 14, 102, 15, 16, 103, 17, 18, 0, 0]]
        )
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, -1, -1]])
        anchor_positions = torch.tensor([1, 4, 7, 10])
        modalities = infer_anchor_modalities(
            input_ids,
            document_ids,
            anchor_positions,
            {"image": [101], "audio": [102], "video": [103]},
        )
        assert modalities.tolist() == [0, 1, 2, 3]

    def test_mixed_document_uses_video_audio_image_priority(self):
        input_ids = torch.tensor([[101, 102, 103, 10]])
        document_ids = torch.zeros_like(input_ids)
        modalities = infer_anchor_modalities(
            input_ids,
            document_ids,
            torch.tensor([3]),
            {"image": [101], "audio": [102], "video": [103]},
        )
        assert modalities.item() == 3
