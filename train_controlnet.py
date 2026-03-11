"""Compatibility entrypoint for ControlNet flow-matching training (stage 2).

The real implementation lives in ``src.cli.train_controlnet``.  This file is
kept so that ``python train_controlnet.py <flags>`` still works exactly as
before.
"""

from src.cli.train_controlnet import main

if __name__ == "__main__":
    main()

