"""Condition router module for mapping text embeddings to expert mixture weights.

Maps pooled CLIP text embeddings to a set of K routing weights using a
configurable MLP. The output weights are normalized via softmax so they
sum to 1 across experts.

This module is designed as a standalone analysis component and is not yet
connected to the UNet or training pipeline.

Example usage::

    router = ConditionRouter(input_dim=768, num_experts=4)
    pooled_emb = torch.randn(2, 768)  # (B, D)
    weights = router(pooled_emb)       # (B, K) with softmax normalization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class ConditionRouterConfig:
    """Configuration for the ConditionRouter MLP.

    Parameters
    ----------
    input_dim : int
        Dimension of the input pooled text embedding (e.g., 768 for CLIP ViT-L).
    num_experts : int
        Number of experts (K) to route to.
    hidden_dims : List[int]
        Hidden layer dimensions for the MLP. Empty list means a single
        linear layer from input_dim to num_experts.
    dropout : float
        Dropout probability applied after each hidden layer.
    temperature : float
        Temperature for the softmax. Lower values produce sharper routing.
    """

    input_dim: int = 768
    num_experts: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    dropout: float = 0.0
    temperature: float = 1.0


class ConditionRouter(nn.Module):
    """Maps pooled text embeddings to mixture weights over K experts.

    A small configurable MLP followed by softmax normalization.

    Parameters
    ----------
    input_dim : int
        Dimension of input pooled text embedding.
    num_experts : int
        Number of experts (K) to produce weights for.
    hidden_dims : Sequence[int]
        Hidden layer dimensions. Default is a single hidden layer of 256.
    dropout : float
        Dropout probability after hidden layers.
    temperature : float
        Softmax temperature. Lower = sharper routing distributions.

    Example
    -------
    >>> router = ConditionRouter(input_dim=768, num_experts=4)
    >>> pooled = torch.randn(2, 768)
    >>> weights = router(pooled)
    >>> weights.shape
    torch.Size([2, 4])
    >>> weights.sum(dim=-1)  # Should be all 1s
    tensor([1., 1.])
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 4,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256]

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dims = list(hidden_dims)
        self.dropout_prob = dropout
        self.temperature = temperature

        # Build MLP layers
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection to num_experts (no activation - softmax applied in forward)
        layers.append(nn.Linear(prev_dim, num_experts))

        self.mlp = nn.Sequential(*layers)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        """Compute routing weights from pooled text embeddings.

        Parameters
        ----------
        pooled_embedding : torch.Tensor
            Pooled text embedding of shape ``(B, D)`` where D = input_dim.

        Returns
        -------
        torch.Tensor
            Routing weights of shape ``(B, K)`` that sum to 1 along dim=-1.
        """
        logits = self.mlp(pooled_embedding)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights

    def get_logits(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        """Return raw logits before softmax (useful for analysis).

        Parameters
        ----------
        pooled_embedding : torch.Tensor
            Pooled text embedding of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, K)``.
        """
        return self.mlp(pooled_embedding)

    @classmethod
    def from_config(cls, config: ConditionRouterConfig) -> "ConditionRouter":
        """Construct a ConditionRouter from a config dataclass."""
        return cls(
            input_dim=config.input_dim,
            num_experts=config.num_experts,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            temperature=config.temperature,
        )

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"num_experts={self.num_experts}, "
            f"hidden_dims={self.hidden_dims}, "
            f"temperature={self.temperature}"
        )


def pool_sequence_embeddings(
    sequence_embeddings: torch.Tensor,
    method: str = "first",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool sequence embeddings to a single vector per sample.

    Utility function to convert CLIP sequence outputs ``(B, seq_len, D)``
    to pooled vectors ``(B, D)``.

    Parameters
    ----------
    sequence_embeddings : torch.Tensor
        Sequence embeddings of shape ``(B, seq_len, D)``.
    method : str
        Pooling method:
        - ``"first"``: Take the first token (NOTE: for CLIP this is [SOT],
          which doesn't vary with content - prefer "eot" or "mean").
        - ``"mean"``: Mean over all tokens.
        - ``"last"``: Take the last token in the sequence.
        - ``"eot"``: Take the last non-padded token (EOT position).
          Requires ``attention_mask``.
    attention_mask : torch.Tensor, optional
        Attention mask of shape ``(B, seq_len)`` with 1s for real tokens.
        Required for "eot" method.

    Returns
    -------
    torch.Tensor
        Pooled embeddings of shape ``(B, D)``.

    Note
    ----
    For CLIP text embeddings, the proper pooled representation comes from
    the ``pooler_output`` of the text encoder (which extracts from EOT
    position with a learned projection). If you have access to the encoder
    outputs directly, prefer using ``pooler_output`` over this function.
    """
    if method == "first":
        return sequence_embeddings[:, 0, :]
    elif method == "mean":
        return sequence_embeddings.mean(dim=1)
    elif method == "last":
        return sequence_embeddings[:, -1, :]
    elif method == "eot":
        if attention_mask is None:
            raise ValueError("attention_mask required for 'eot' pooling method")
        # Find the last 1 in each attention mask row (EOT position)
        # attention_mask shape: (B, seq_len)
        # Sum gives count of non-padded tokens, minus 1 gives last position
        seq_lengths = attention_mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(sequence_embeddings.size(0), device=sequence_embeddings.device)
        return sequence_embeddings[batch_indices, seq_lengths, :]
    else:
        raise ValueError(f"Unknown pooling method: {method!r}. Use 'first', 'mean', 'last', or 'eot'.")
