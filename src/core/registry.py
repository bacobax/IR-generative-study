"""Lightweight decorator-based component registry for the FM system.

Provides five named registries so that model builders, trainers, samplers,
guidance modules, and conditioning modules can be looked up by string name.

Usage::

    from src.core.registry import REGISTRIES

    # Register a new component
    @REGISTRIES.model_builder.register("my_unet")
    def build_my_unet(config, *, device="cpu"):
        ...

    # Look up and call
    builder = REGISTRIES.model_builder["my_unet"]
    unet = builder(config, device=device)

Each registry keeps an internal dict and raises a clear ``KeyError`` if a
name is not found, listing all available keys.

No dynamic import magic — components must be imported (and therefore
registered) before they can be resolved.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class Registry:
    """A single named registry mapping string keys to callables."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Decorator for registration
    # ------------------------------------------------------------------
    def register(self, name: str, *, default: bool = False):
        """Decorator that registers *fn* under *name*.

        Parameters
        ----------
        name : str
            Lookup key.
        default : bool
            If ``True`` this entry is also stored as ``"__default__"``.
        """

        def decorator(fn):
            if name in self._entries:
                raise ValueError(
                    f"[{self.name}] '{name}' is already registered. "
                    f"Registered keys: {self.list()}"
                )
            self._entries[name] = fn
            if default:
                self._entries["__default__"] = fn
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> Any:
        if name not in self._entries:
            raise KeyError(
                f"[{self.name}] '{name}' not found. "
                f"Available: {self.list()}"
            )
        return self._entries[name]

    def get(self, name: Optional[str] = None) -> Any:
        """Return *name* if given, otherwise the ``__default__`` entry."""
        if name is not None:
            return self[name]
        if "__default__" in self._entries:
            return self._entries["__default__"]
        raise KeyError(
            f"[{self.name}] No name given and no default registered. "
            f"Available: {self.list()}"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def list(self) -> List[str]:
        """Return all registered keys (excluding ``__default__``)."""
        return [k for k in sorted(self._entries) if k != "__default__"]

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={self.list()})"


# ═══════════════════════════════════════════════════════════════════════════
# Global registry container
# ═══════════════════════════════════════════════════════════════════════════

class _Registries:
    """Container holding all five FM-system registries."""

    def __init__(self) -> None:
        self.model_builder = Registry("model_builder")
        self.trainer = Registry("trainer")
        self.sampler = Registry("sampler")
        self.guidance = Registry("guidance")
        self.conditioning = Registry("conditioning")

    def summary(self) -> str:
        lines = []
        for attr in ("model_builder", "trainer", "sampler", "guidance", "conditioning"):
            reg: Registry = getattr(self, attr)
            keys = reg.list()
            default = "__default__" in reg._entries
            lines.append(f"  {attr}: {keys}  (default={'yes' if default else 'no'})")
        return "FM Component Registries:\n" + "\n".join(lines)


REGISTRIES = _Registries()
