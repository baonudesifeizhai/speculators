"""Sequential, confidence, and modality heads for the DSpark draft model."""

import torch
from torch import nn

__all__ = [
    "MODALITY_NAMES",
    "ConfidenceHead",
    "MarkovHead",
    "ModalityLogitHead",
    "ModalityRouter",
    "infer_anchor_modalities",
]


MODALITY_NAMES = ("text", "image", "audio", "video")


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias ``B = W1 @ W2``.

    ``W1`` indexes the verifier vocabulary (the previous token id); ``W2`` projects
    to the draft vocabulary so the bias adds onto the DFlash logits.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] projection over [state; prev_emb; hidden].
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up W1 embeddings for the given previous-token ids."""
        return self.markov_w1(token_ids.long())

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [N, block_size, r]
    ) -> torch.Tensor:
        """Return the per-position logit bias, shape [N, block_size, draft_vocab]."""
        if prev_emb is None:
            prev_emb = self.prev_embeddings(prev_token_ids)
        prev_emb = prev_emb.to(self.markov_w2.weight.dtype)

        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            return self.markov_w2(gate * prev_emb)

        # rnn: maintain a recurrent state across block positions.
        hidden_states = hidden_states.to(prev_emb.dtype)
        num_blocks, block_size, _ = prev_emb.shape
        state = prev_emb.new_zeros(num_blocks, self.markov_rank)
        outputs = []
        for k in range(block_size):
            z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
            gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class ModalityLogitHead(nn.Module):
    """Low-rank, modality-specific residual over the shared draft logits.

    DSpark's main LM head is copied from the verifier and deliberately frozen.
    Cloning that full vocabulary projection once per modality would therefore be
    both wasteful and ineffective.  This head instead learns a compact residual
    from the shared draft hidden state into the draft vocabulary.

    The output projection starts at zero, so enabling modality heads preserves
    the vanilla DSpark logits at initialization.
    """

    def __init__(self, hidden_size: int, draft_vocab_size: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"modality head rank must be > 0, got {rank}")
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(rank, draft_vocab_size, bias=False)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.act(self.down_proj(hidden_states)))


class ModalityRouter(nn.Module):
    """Predict the request modality from the first draft slot of each block.

    Training uses labels recovered from cached Qwen multimodal special tokens.
    Serving no longer has those prompt tokens at every decode step, so this
    lightweight router lets the checkpoint select the same head from its fused
    Thinker-conditioned hidden state.
    """

    def __init__(self, hidden_size: int, num_modalities: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_modalities, bias=True)

    def forward(self, block_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(block_hidden_states[:, 0])


@torch.compiler.disable
def infer_anchor_modalities(
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    modality_token_ids: dict[str, list[int]],
) -> torch.Tensor:
    """Infer each anchor's input modality from special tokens in its document.

    The cached Qwen3-Omni token IDs retain ``image_pad``, ``audio_pad``, and
    ``video_pad`` tokens even though those tokens are excluded from the draft
    vocabulary and loss mask.  Routing from the cached IDs avoids rewriting the
    existing multi-terabyte hidden-state cache with explicit modality labels.

    Documents containing more than one modality use the highest-priority route
    ``video > audio > image > text``.  The current Thinker dataset has one primary
    modality per row, but the deterministic precedence keeps mixed requests safe.
    """
    input_ids = input_ids.reshape(-1).long()
    document_ids = document_ids.reshape(-1).long()
    anchor_positions = anchor_positions.reshape(-1).long()
    if input_ids.shape != document_ids.shape:
        raise ValueError(
            "input_ids and document_ids must have the same flattened shape"
        )

    token_modalities = torch.zeros_like(input_ids)
    for modality_id, name in enumerate(MODALITY_NAMES[1:], start=1):
        matches = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in modality_token_ids.get(name, []):
            matches |= input_ids == int(token_id)
        token_modalities = torch.where(
            matches,
            torch.full_like(token_modalities, modality_id),
            token_modalities,
        )

    # document_ids are dense non-negative indices produced by the collator.
    # Allocate by sequence length so this remains static-shaped under packing.
    document_modalities = torch.zeros(
        input_ids.numel(), dtype=torch.long, device=input_ids.device
    )
    valid = document_ids >= 0
    if torch.any(valid):
        document_modalities.scatter_reduce_(
            0,
            document_ids[valid],
            token_modalities[valid],
            reduce="amax",
            include_self=True,
        )

    anchor_documents = document_ids[anchor_positions]
    safe_documents = anchor_documents.clamp_min(0)
    modalities = document_modalities[safe_documents]
    return torch.where(anchor_documents >= 0, modalities, torch.zeros_like(modalities))
