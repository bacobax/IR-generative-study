"""Abstract base class for inference-time guidance modules.

Any guidance object that can be plugged into ``FlowMatchingSampler``
must implement this interface.  The three methods mirror the existing
duck-typed protocol used by ``ScorePredictorGuidance`` in the legacy
``fm_src/guidance`` package.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch


class BaseGuidance(abc.ABC):
    """Minimal interface for inference-time guidance.

    Subclasses must implement :meth:`energy`, :meth:`guidance_grad`, and
    :meth:`log_scores`.  The ``FlowMatchingSampler`` calls these methods
    during the Euler integration loop.
    """

    @abc.abstractmethod
    def energy(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
    ) -> torch.Tensor:
        """Compute per-sample scalar energy E(z).

        Parameters
        ----------
        z : (B, C, H, W)
            Current latent samples.
        t : float, optional
            Flow time in [0, 1).
        pipeline : object, optional
            Reference to the sampler (for VAE access, etc.).

        Returns
        -------
        Tensor of shape (B,) — one energy value per sample.
        """

    @abc.abstractmethod
    def guidance_grad(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the guidance gradient to add to the velocity field.

        Parameters
        ----------
        z : (B, C, H, W)
            Current latent, may have ``requires_grad=True``.
        t : float, optional
            Flow time.
        pipeline : object, optional
            Reference to the sampler.
        velocity : (B, C, H, W), optional
            Velocity predicted by the FM model.

        Returns
        -------
        Tensor of same shape as *z* — the gradient signal.
        """

    @abc.abstractmethod
    def log_scores(self, z: torch.Tensor) -> Dict[str, float]:
        """Return human-readable diagnostic scores for *z* (no grad).

        The returned dict is used for TensorBoard logging and comparison
        between guided vs. unguided samples.

        Returns
        -------
        Dict mapping string keys to float scores.
        """
