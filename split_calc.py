#!/usr/bin/env python3
"""
Compute phase-B / phase-C epoch suggestions for split configurations
under a total-class-exposure balance constraint.

Definitions
-----------
Let:
- base classes = B
- incremental classes = I
- test classes are ignored here
- n_k = number of samples for class k
- A = phase_a epochs
- B_ep = phase_b epochs
- C_ep = phase_c epochs
- replay_every = r
- rho ~= 1 / r

Approximate total class exposure of split S = (B, I):
    H(S) ~= A * sum_{k in B} n_k
            + B_ep * sum_{i in I} n_i
            + (1 + rho) * C_ep * sum_{i in I} n_i

Within-split diagnostic:
    R(S) = (sum_{i in I} T_i) / (sum_{k in B} T_k)

With the same approximation:
    R(S) ~= ((B_ep + C_ep) * sum_{i in I} n_i) /
            (A * sum_{k in B} n_k + rho * C_ep * sum_{i in I} n_i)

This script:
1. stores the class-count distribution,
2. asks the user for the reference split B base classes and phase-A epochs,
3. computes the target H(S_ref),
4. lets the user enter candidate base/incremental configurations,
5. suggests phase-B / phase-C epochs that match H(S_ref),
6. computes the corresponding R metric.

By default, the script preserves a chosen phase-B : phase-C ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


CLASS_COUNTS: Dict[int, int] = {
    0: 6,
    1: 2710,
    2: 1517,
    3: 1142,
    4: 625,
    5: 327,
    6: 122,
    7: 81,
    8: 36,
    9: 30,
    10: 24,
    11: 20,
    12: 19,
    13: 29,
    14: 13,
    15: 3,
    16: 8,
    17: 2,
    18: 3,
    19: 8,
}


@dataclass
class Split:
    base: List[int]
    incremental: List[int]

    def validate(self, counts: Dict[int, int]) -> None:
        base_set = set(self.base)
        inc_set = set(self.incremental)

        unknown = sorted((base_set | inc_set) - set(counts))
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}")

        overlap = sorted(base_set & inc_set)
        if overlap:
            raise ValueError(f"Classes cannot be both base and incremental: {overlap}")

        if not self.base:
            raise ValueError("Base classes cannot be empty.")


def sum_counts(classes: List[int], counts: Dict[int, int]) -> int:
    return sum(counts[c] for c in classes)


def rho_from_replay_every(replay_every: int) -> float:
    if replay_every <= 0:
        raise ValueError("replay_every must be >= 1")
    return 1.0 / replay_every


def total_exposure(
    split: Split,
    phase_a_epochs: float,
    phase_b_epochs: float,
    phase_c_epochs: float,
    counts: Dict[int, int],
    replay_every: int,
) -> float:
    split.validate(counts)
    rho = rho_from_replay_every(replay_every)

    base_sum = sum_counts(split.base, counts)
    inc_sum = sum_counts(split.incremental, counts)

    return (
        phase_a_epochs * base_sum
        + phase_b_epochs * inc_sum
        + (1.0 + rho) * phase_c_epochs * inc_sum
    )


def exposure_ratio_R(
    split: Split,
    phase_a_epochs: float,
    phase_b_epochs: float,
    phase_c_epochs: float,
    counts: Dict[int, int],
    replay_every: int,
) -> float:
    split.validate(counts)
    rho = rho_from_replay_every(replay_every)

    base_sum = sum_counts(split.base, counts)
    inc_sum = sum_counts(split.incremental, counts)

    if inc_sum == 0:
        return 0.0

    numerator = (phase_b_epochs + phase_c_epochs) * inc_sum
    denominator = phase_a_epochs * base_sum + rho * phase_c_epochs * inc_sum

    if denominator == 0:
        return math.inf
    return numerator / denominator


def suggest_phase_b_c(
    target_H: float,
    split: Split,
    phase_a_epochs: float,
    counts: Dict[int, int],
    replay_every: int,
    b_to_c_ratio: float,
) -> Tuple[float, float]:
    """
    Solve for phase-B and phase-C epochs so that H(split) ~= target_H,
    preserving phase_b / phase_c = b_to_c_ratio.

    Let B_ep = alpha * C_ep.
    Then:
        target_H = A * sum_base + (alpha + 1 + rho) * C_ep * sum_inc
    """
    split.validate(counts)
    rho = rho_from_replay_every(replay_every)

    base_sum = sum_counts(split.base, counts)
    inc_sum = sum_counts(split.incremental, counts)

    current_base_budget = phase_a_epochs * base_sum
    residual = target_H - current_base_budget

    if inc_sum == 0:
        # No incremental classes => no B/C needed.
        return 0.0, 0.0

    denom = (b_to_c_ratio + 1.0 + rho) * inc_sum
    if denom <= 0:
        raise ValueError("Invalid denominator while solving for phase B/C epochs.")

    phase_c = residual / denom
    phase_b = b_to_c_ratio * phase_c

    return phase_b, phase_c


def fmt_list(xs: List[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def parse_int_list(prompt: str) -> List[int]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def print_distribution(counts: Dict[int, int]) -> None:
    print("\nClass count distribution:")
    for k in sorted(counts):
        print(f"  class {k:>2}: {counts[k]}")


def main() -> None:
    print("=== Exposure-balanced phase-B / phase-C suggester ===")
    print_distribution(CLASS_COUNTS)

    print("\nReference split S_ref (this is your config B baseline).")
    ref_base = parse_int_list("Enter reference base classes, comma-separated: ")
    ref_split = Split(base=ref_base, incremental=[])

    phase_a = float(input("Enter fixed phase A epochs: ").strip())
    replay_every = int(input("Enter replay_every (usually 1): ").strip())
    b_to_c_ratio = float(
        input("Enter desired phase_B / phase_C ratio (e.g. 0.5 for 30:60): ").strip()
    )

    target_H = total_exposure(
        split=ref_split,
        phase_a_epochs=phase_a,
        phase_b_epochs=0.0,
        phase_c_epochs=0.0,
        counts=CLASS_COUNTS,
        replay_every=replay_every,
    )

    print("\nReference configuration:")
    print(f"  base        = {fmt_list(ref_split.base)}")
    print(f"  incremental = []")
    print(f"  phase_a     = {phase_a}")
    print(f"  target H    = {target_H:.4f}")

    while True:
        print("\n--- Candidate split ---")
        cand_base = parse_int_list("Enter candidate base classes, comma-separated: ")
        cand_inc = parse_int_list("Enter candidate incremental classes, comma-separated: ")

        candidate = Split(base=cand_base, incremental=cand_inc)
        candidate.validate(CLASS_COUNTS)

        phase_b, phase_c = suggest_phase_b_c(
            target_H=target_H,
            split=candidate,
            phase_a_epochs=phase_a,
            counts=CLASS_COUNTS,
            replay_every=replay_every,
            b_to_c_ratio=b_to_c_ratio,
        )

        H_candidate = total_exposure(
            split=candidate,
            phase_a_epochs=phase_a,
            phase_b_epochs=phase_b,
            phase_c_epochs=phase_c,
            counts=CLASS_COUNTS,
            replay_every=replay_every,
        )

        R_candidate = exposure_ratio_R(
            split=candidate,
            phase_a_epochs=phase_a,
            phase_b_epochs=phase_b,
            phase_c_epochs=phase_c,
            counts=CLASS_COUNTS,
            replay_every=replay_every,
        )

        base_sum = sum_counts(candidate.base, CLASS_COUNTS)
        inc_sum = sum_counts(candidate.incremental, CLASS_COUNTS)

        print("\nSuggested configuration:")
        print(f"  base classes        = {fmt_list(candidate.base)}")
        print(f"  incremental classes = {fmt_list(candidate.incremental)}")
        print(f"  sum base counts     = {base_sum}")
        print(f"  sum incr counts     = {inc_sum}")
        print(f"  fixed phase_a       = {phase_a:.4f}")
        print(f"  suggested phase_b   = {phase_b:.4f}")
        print(f"  suggested phase_c   = {phase_c:.4f}")
        print(f"  matched H(S)        = {H_candidate:.4f}")
        print(f"  target H(S_ref)     = {target_H:.4f}")
        print(f"  abs diff            = {abs(H_candidate - target_H):.8f}")
        print(f"  R(S)                = {R_candidate:.6f}")

        if candidate.incremental:
            print("\nRounded integer suggestions:")
            print(f"  phase_b ≈ {round(phase_b)}")
            print(f"  phase_c ≈ {round(phase_c)}")
        else:
            print("\nNo incremental classes: phase_b = phase_c = 0")

        again = input("\nEvaluate another split? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()