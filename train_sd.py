"""Compatibility entrypoint for Stable Diffusion LoRA training.

The real implementation lives in ``src.cli.train_sd``.  This file is kept
so that ``python train_sd.py <flags>`` still works exactly as before.
"""

from src.cli.train_sd import main

if __name__ == "__main__":
    main()

