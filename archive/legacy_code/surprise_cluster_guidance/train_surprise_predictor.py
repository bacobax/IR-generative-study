#!/usr/bin/env python3
"""Compatibility wrapper for scripts.standalone.train_surprise_predictor."""

from importlib import import_module as _import_module

_impl = _import_module("scripts.standalone.train_surprise_predictor")
globals().update({
    k: v for k, v in vars(_impl).items()
    if k not in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"}
})

if __name__ == "__main__":
    _impl.main()
