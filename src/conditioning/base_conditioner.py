"""Abstract base class for conditioning modules.

A conditioner prepares extra keyword arguments that are passed to the
UNet alongside the noisy input and timestep.  This covers both training
(where a data batch may contain labels, text, etc.) and inference (where
conditioning may be sampled or fixed).

Concrete implementations include:

* ``NoConditioner``  — pass-through, reproducing the current unconditional flow.
* (future) ``ClassConditioner``, ``TextConditioner``, ``DINOConditioner``, …
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch


class BaseConditioner(abc.ABC):
    """Minimal interface for conditioning modules.

    Subclasses must implement :meth:`prepare_for_training` and
    :meth:`prepare_for_sampling`.  Both return a ``Dict[str, Any]``
    whose contents are unpacked as extra keyword arguments into the
    UNet forward call::

        unet_out = unet(zt, t, **conditioner.prepare_for_training(batch))
    """

    @abc.abstractmethod
    def prepare_for_training(
        self,
        batch: Any,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Derive conditioning kwargs from a training mini-batch.

        Parameters
        ----------
        batch : Any
            Whatever the DataLoader yields.  Implementations decide what
            to extract (e.g. class labels, text tokens).
        device : str or torch.device
            Target device for any tensors produced.

        Returns
        -------
        Dict that will be unpacked into the UNet forward call.
        An empty dict means unconditional.
        """

    @abc.abstractmethod
    def prepare_for_sampling(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Produce conditioning kwargs for inference / sampling.

        Parameters
        ----------
        batch_size : int
            Number of samples being generated.
        device : str or torch.device
            Target device.

        Returns
        -------
        Dict that will be unpacked into the UNet forward call.
        """
