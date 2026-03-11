#!/usr/bin/env python3
"""Compatibility entrypoint for synthetic dataset generation.

The real implementation lives in ``src.cli.generate``.  This file is kept
so that ``python generate_datasets.py <flags>`` still works exactly as
before.
"""

from src.cli.generate import main

if __name__ == "__main__":
    main()
