"""Tests for mini-batch optimal transport assignment helpers."""

from __future__ import annotations

import builtins
import itertools

import torch

from src.core.ot import solve_assignment


def _brute_force_assignment(cost_matrix: torch.Tensor) -> torch.Tensor:
    n = int(cost_matrix.shape[0])
    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        total = sum(float(cost_matrix[row, col]) for row, col in enumerate(perm))
        if total < best_cost:
            best_cost = total
            best_perm = perm
    return torch.tensor(best_perm, dtype=torch.long)


def test_solve_assignment_uses_exact_fallback_when_scipy_import_fails(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.optimize":
            raise ImportError("blocked scipy.optimize")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    cost_matrix = torch.tensor(
        [
            [4.0, 1.0, 3.0, 9.0],
            [2.0, 0.0, 5.0, 8.0],
            [3.0, 2.0, 2.0, 6.0],
            [7.0, 5.0, 6.0, 1.0],
        ]
    )

    assert torch.equal(solve_assignment(cost_matrix), _brute_force_assignment(cost_matrix))
