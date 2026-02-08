"""Reduced controllability test via output-only derivative-lifted Gramians.

This script demonstrates the *reduced* data-driven Hautus test
(Theorem thm:ct-dd-hautus-reduced), which certifies controllability
using only the output trajectory y — no input lifting required.

Steps:
    1. Generate an LTI system (A, B, C, D)
    2. Simulate via SDE:  dx = (Ax + Bu)dt + β dW,  y = Cx + Du + δv
    3. Compute the finite candidate set σ(K^y_{L,K})
       (Theorem thm:ct-dd-hautus-reduced-finite-lambda)
    4. For each candidate λ, compute H_{L,K}(λ) = Γ_K(y_λ) and check rank
    5. Compare with the full behavioral test G_{L,K}(λ)
    6. Visualize results

Usage:
    python examples/controllability_reduced.py
    python examples/controllability_reduced.py --system coupled_spring
    python examples/controllability_reduced.py --system two_spring
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LinearSDE,
    simulate,
    compute_H_LK,
    compute_K_LK_reduced,
    compute_G_LK,
    compute_K_LK,
    check_controllability,
    check_persistent_excitation,
    check_controllability_reduced,
    compute_derivative_lift,
    plot_trajectories,
    plot_eigenvalues,
    plot_gramian_eigenvalues,
    plot_controllability_margin,
    compute_observability_index,
    smooth_signal,
)
from src.systems import get_system, print_system_info, SYSTEMS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reduced controllability test via output-only Gramians"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="two_spring",
        choices=list(SYSTEMS.keys()),
        help=f"System to analyze. Available: {list(SYSTEMS.keys())}",
    )
    return parser.parse_args()


def main(system_name: str):
    # =========================================================================
    # Configuration
    # =========================================================================
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Using device: {device}")
    print(f"Selected system: {system_name}")

    # Derivative lift parameters
    # For the reduced test (Thm thm:ct-dd-hautus-reduced):
    #   K ≥ ℓ(B),  L ≥ K+1.
    # For the finite candidate set (Thm thm:ct-dd-hautus-reduced-finite-lambda),
    # rank(S_y) = Kp is required for equivalence over all λ.
    # We compute the tightest feasible (L, K) after loading the system.

    # Simulation parameters
    T = 100.0   # Final time
    dt = 0.01  # Time step

    # Noise intensities
    beta_scale = 0.1   # Process noise
    delta_scale = 0.1  # Measurement noise

    # Smoothing
    smooth_y = False
    smoothing_window = 11
    smoothing_sigma = 10.0
    smoothing_mode = "gaussian"
    numerical_threshold = 1e-8

    # =========================================================================
    # Generate system
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"LOADING SYSTEM: {system_name.upper()}")
    print("=" * 60)

    if system_name == "random":
        A, B, C, D = get_system(system_name, device=device, dtype=dtype, seed=42)
    else:
        A, B, C, D = get_system(system_name, device=device, dtype=dtype)

    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Input dimension
    p = C.shape[0]  # Output dimension
    q = min(n, 1)
    r = min(p, 1)

    # Noise matrices
    Beta = beta_scale * torch.randn(n, q, device=device, dtype=dtype)
    Delta = delta_scale * torch.randn(p, r, device=device, dtype=dtype)

    # System info
    print(f"\nSystem dimensions: n={n}, m={m}, p={p}")

    # Classical controllability / observability checks
    C_mat = torch.cat(
        [torch.linalg.matrix_power(A, k) @ B for k in range(n)], dim=1
    )
    rank_C = torch.linalg.matrix_rank(C_mat).item()
    is_controllable_AB = rank_C == n
    print(f"\nControllability matrix rank: {rank_C}/{n} → controllable: {is_controllable_AB}")

    O_mat = torch.cat(
        [C @ torch.linalg.matrix_power(A, k) for k in range(n)], dim=0
    )
    rank_O = torch.linalg.matrix_rank(O_mat).item()
    is_observable_AC = rank_O == n
    print(f"Observability matrix rank:  {rank_O}/{n} → observable:    {is_observable_AC}")

    ell = compute_observability_index(C, A)
    print(f"Observability index ell(B): {ell}")

    # Tight reduced-test choice: K = ell, L = K+1.
    K = ell
    L = K + 1
    print(f"Using derivative lifts: L={L}, K={K}")

    # =========================================================================
    # Simulate SDE
    # =========================================================================
    print("\n" + "=" * 60)
    print("SIMULATING STOCHASTIC SYSTEM")
    print("=" * 60)

    sde = LinearSDE(A, B, C, D, Beta, Delta)
    ts, x, u, y = simulate(sde, T, dt)

    y_raw = y
    if smooth_y:
        y = smooth_signal(
            y,
            window_size=smoothing_window,
            sigma=smoothing_sigma,
            mode=smoothing_mode,
        )
        print(
            f"Applied {smoothing_mode} smoothing to y "
            f"(window={smoothing_window}, sigma={smoothing_sigma})."
        )

    print(f"\nSimulation complete:")
    print(f"  T={T}, dt={dt}, N={len(ts)}")
    print(f"  u: {u.shape},  y: {y.shape}")

    # =========================================================================
    # Check persistent excitation of u at order L+n+1 (full-test assumption)
    # =========================================================================
    print("\n" + "=" * 60)
    print("INPUT PERSISTENT EXCITATION CHECK")
    print("=" * 60)

    pe_order = L + n + 1
    pe_result = check_persistent_excitation(
        u,
        order=pe_order,
        dt=dt,
        threshold=numerical_threshold,
    )
    print(f"\nPE order tested: {pe_order}")
    print(
        f"Γ_{pe_order}(u) rank: {pe_result['rank']}/{pe_result['full_dimension']} "
        f"(threshold={numerical_threshold:.1e})"
    )
    print(f"min eigenvalue(Γ_{pe_order}(u)): {pe_result['min_eigenvalue']:.6e}")
    print(f"u is persistently exciting of order {pe_order}: {pe_result['is_persistently_exciting']}")

    # =========================================================================
    # Check lifted state-input Gramian Γ_{L,1}(u,x) invertibility (assumption)
    # =========================================================================
    Lambda_L_u = compute_derivative_lift(u, L, dt)  # (N', Lm)
    N_common = min(Lambda_L_u.shape[0], x.shape[0])
    Lambda_L_u = Lambda_L_u[:N_common]
    x_trunc = x[:N_common]
    Z = torch.cat([Lambda_L_u, x_trunc], dim=-1)  # (N', Lm+n)
    Gamma_L1 = Z.T @ Z * dt
    eigvals_G = torch.linalg.eigvalsh(Gamma_L1).real
    min_eig_G = eigvals_G.min().item()
    rank_G = (eigvals_G > numerical_threshold).sum().item()
    invertible_G = rank_G == Gamma_L1.shape[0]
    is_pd_G = min_eig_G > numerical_threshold
    print(
        f"\nΓ_{{L,1}}(u,x) rank: {rank_G}/{Gamma_L1.shape[0]} "
        f"(threshold={numerical_threshold:.1e})"
    )
    print(f"min eigenvalue(Γ_{{L,1}}(u,x)): {min_eig_G:.6e}")
    print(f"Γ_{{L,1}}(u,x) invertible: {invertible_G}")
    print(f"Γ_{{L,1}}(u,x) positive definite: {is_pd_G}")
    if not invertible_G:
        print(
            "  WARNING: Γ_{L,1}(u,x) is not invertible; "
            "Theorem thm:ct-dd-hautus-reduced assumptions may fail."
        )

    # =========================================================================
    # Reduced test: finite candidate set  σ(K^y_{L,K})
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"REDUCED TEST — K^y_{{L,K}} with L={L}, K={K}")
    print("=" * 60)

    K_matrix_red, cand_red = compute_K_LK_reduced(y, L, K, dt)
    rank_Sy = len(cand_red)

    print(f"\nK^y_{{L,K}} shape: {K_matrix_red.shape}")
    print(
        f"Candidate eigenvalues (|σ| ≤ r = rank(S_y) = {rank_Sy} ≤ Kp = {K*p}):"
    )
    for i, lam in enumerate(cand_red[:10]):
        print(f"  μ_{i+1} = {lam.real:.4f} + {lam.imag:.4f}i")
    if len(cand_red) > 10:
        print(f"  ... ({len(cand_red) - 10} more)")
    if rank_Sy < K * p:
        print(
            "  WARNING: rank(S_y) < Kp, so the finite-candidate reduction "
            "is not conclusive for full-rank H_{L,K}(λ)."
        )

    # =========================================================================
    # Compute H_{L,K}(λ) at candidate λ and a few extra values
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"COMPUTING H_{{L,K}}(λ) GRAMIANS")
    print("=" * 60)

    expected_rank = K * p
    print(f"\nExpected rank: Kp = {expected_rank}")


    # =========================================================================
    # Controllability check — reduced test
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONTROLLABILITY CHECK — REDUCED (output-only)")
    print("=" * 60)

    result_red = check_controllability_reduced(
        y,
        L,
        K,
        T,
        dt,
        candidate_lambdas=cand_red,
        threshold=1e-6,
    )

    print(f"\nExpected rank: {result_red['expected_rank']}")
    print(f"rank(S_y): {result_red['rank_Sy']} (full: {result_red['full_rank_Sy']})")
    print(f"Ranks at candidate λ: {result_red['ranks'].cpu().numpy()}")
    print(f"Min margin: {result_red['min_eigenvalues'].min():.6e}")
    print(f"\nIS CONTROLLABLE (reduced): {result_red['is_controllable']}")

    # =========================================================================
    # Controllability check — full behavioral test (for comparison)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONTROLLABILITY CHECK — FULL BEHAVIORAL (for comparison)")
    print("=" * 60)

    K = ell
    L = K

    _, cand_full = compute_K_LK(u, y, L, K, n, 0.0, dt)

    result_full = check_controllability(
        u,
        y,
        L,
        K,
        n,
        m,
        T,
        dt,
        candidate_lambdas=cand_full,
        threshold=1e-6,
    )

    print(f"\nExpected rank: {result_full['expected_rank']}")
    print(f"Ranks at candidate λ: {result_full['ranks'].cpu().numpy()}")
    print(f"Min margin: {result_full['min_eigenvalues'].min():.6e}")
    print(f"\nIS CONTROLLABLE (full):    {result_full['is_controllable']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Classical (A,B) controllable:      {is_controllable_AB}")
    print(
        f"  Reduced  test (output-only):       {result_red['is_controllable']} "
        f"(rank(S_y)=Kp: {result_red['full_rank_Sy']})"
    )
    print(f"  Full behavioral test (u + y):      {result_full['is_controllable']}")

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. System trajectories
    fig1 = plot_trajectories(ts, x, u, y)
    fig1.savefig(os.path.join(output_dir, "reduced_trajectories.pdf"), bbox_inches="tight")
    print("  Saved: reduced_trajectories.pdf")
    plt.show()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return {
        "A": A, "B": B, "C": C, "D": D,
        "ts": ts, "x": x, "u": u, "y": y, "y_raw": y_raw,
        "K_matrix_reduced": K_matrix_red,
        "candidate_lambdas_reduced": cand_red,
        "candidate_lambdas_full": cand_full,
        "result_reduced": result_red,
        "result_full": result_full,
    }


if __name__ == "__main__":
    args = parse_args()
    results = main(system_name=args.system)
