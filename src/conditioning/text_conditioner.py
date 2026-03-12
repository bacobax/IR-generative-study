"""CLIP-based text conditioner with classifier-free guidance support.

Tokenises text prompts, encodes them with a frozen CLIP text encoder,
and supports conditioning dropout for CFG training.  During inference,
:meth:`prepare_cfg_pair` returns both conditional and null embeddings
for the CFG velocity combination.

The null embedding is the encoding of the empty string ``""``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import CLIPTokenizer, CLIPTextModel

from src.conditioning.base_conditioner import BaseConditioner
from src.core.registry import REGISTRIES


class TextConditioner(BaseConditioner):
    """CLIP text conditioner for flow-matching models with CFG.

    Parameters
    ----------
    encoder_name : str
        HuggingFace model id for the CLIP text encoder.
    max_length : int
        Maximum token sequence length (CLIP default is 77).
    cond_drop_prob : float
        Probability of replacing text embeddings with null embeddings
        during training (enables classifier-free guidance).
    device : str or torch.device
        Device for the frozen text encoder.
    """

    def __init__(
        self,
        encoder_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        cond_drop_prob: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.cond_drop_prob = cond_drop_prob
        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(encoder_name)
        self.text_encoder.to(device)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self._null_embedding: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        """Hidden size of the CLIP text encoder (cross_attention_dim)."""
        return self.text_encoder.config.hidden_size

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "TextConditioner":
        self.device = device
        self.text_encoder.to(device)
        self._null_embedding = None
        return self

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Tokenise and encode a list of prompts.

        Returns
        -------
        torch.Tensor of shape ``(B, seq_len, hidden_dim)``.
        """
        dev = device or self.device
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(dev)
        attention_mask = tokens["attention_mask"].to(dev)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def null_embedding(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Cached null (empty-string) embedding expanded to *batch_size*."""
        dev = device or self.device
        if (
            self._null_embedding is None
            or str(self._null_embedding.device) != str(dev)
        ):
            self._null_embedding = self.encode_text([""], dev)
        return self._null_embedding.expand(batch_size, -1, -1)

    # ------------------------------------------------------------------
    # BaseConditioner interface
    # ------------------------------------------------------------------
    def prepare_for_training(
        self,
        batch: Any,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Encode text from *batch* with conditioning dropout.

        Expects ``batch["text"]`` to be a list of strings.
        With probability :attr:`cond_drop_prob`, individual samples are
        replaced with the null embedding so the model learns both
        conditional and unconditional behaviour.
        """
        texts: List[str] = batch["text"]
        embeddings = self.encode_text(texts, device)

        if self.cond_drop_prob > 0:
            drop_mask = torch.rand(len(texts)) < self.cond_drop_prob
            if drop_mask.any():
                null_emb = self.null_embedding(1, device)
                embeddings = embeddings.clone()
                embeddings[drop_mask] = null_emb[0]

        return {"encoder_hidden_states": embeddings}

    def prepare_for_sampling(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Return null embeddings (unconditional generation)."""
        return {"encoder_hidden_states": self.null_embedding(batch_size, device)}

    # ------------------------------------------------------------------
    # CFG-specific helpers
    # ------------------------------------------------------------------
    def prepare_conditional(
        self,
        prompts: List[str],
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Encode prompts for conditional-only sampling (no CFG)."""
        return {"encoder_hidden_states": self.encode_text(prompts, device)}

    def prepare_cfg_pair(
        self,
        prompts: List[str],
        device: torch.device | str = "cpu",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(cond_kwargs, uncond_kwargs)`` for CFG sampling.

        Both dicts contain ``"encoder_hidden_states"``; the second uses
        the null embedding.
        """
        batch_size = len(prompts)
        cond = {"encoder_hidden_states": self.encode_text(prompts, device)}
        uncond = {"encoder_hidden_states": self.null_embedding(batch_size, device)}
        return cond, uncond


# ── registry ──────────────────────────────────────────────────────────────
REGISTRIES.conditioning.register("text_cfg")(TextConditioner)
