"""Convenience wrapper for flow-matching training.

The real implementation lives in ``src.cli.train``.  This file is kept
for backward-compatibility so that ``python train_sfm.py <flags>`` still
works exactly as before.
"""

from src.cli.train import main

if __name__ == "__main__":
    main()