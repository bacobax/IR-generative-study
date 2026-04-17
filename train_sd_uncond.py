"""Compatibility entrypoint for unconditional latent Stable Diffusion training.

The real implementation lives in ``src.cli.train_sd_uncond``. This file is kept
so that ``python train_sd_uncond.py <flags>`` works like the other thin wrappers.
"""

from src.cli.train_sd_uncond import main

if __name__ == "__main__":
    main()
