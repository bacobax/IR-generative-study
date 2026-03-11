"""Compatibility entrypoint for VAE training.

The real implementation lives in ``src.cli.train_vae``.  This file is kept
so that ``python train_vae.py <flags>`` still works exactly as before.
"""

from src.cli.train_vae import main

if __name__ == "__main__":
    main()

