"""Experiment: Behavioral (input-output) controllability test.

Validates the behavioral Gramian G_{L,K}(λ) and reduced H_{L,K}(λ) 
using only input-output data (u, y), without access to the state x.
Compares full and reduced tests on controllable vs uncontrollable systems.

This directly validates Theorems 2.7 and 2.14 from the paper.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LinearSDE,
    simulate,
    compute_G_LK,
    compute_K_LK,
    check_controllability,
    compute_observability_index,
    smooth_signal,
)
from src.gramians import (
    compute_K_LK_reduced,
    check_controllability_reduced,
    check_persistent_excitation,
)
from src.systems import get_system, print_system_info


def run_behavioral_test(system_name, T=50.0, dt=0.01, beta_scale=0.0, delta_scale=0.0, seed=42):
    """Run full and reduced behavioral tests on a given system.
    
    Returns dict with test results.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Load system
    if system_name == "random":
        A, B, C, D = get_system(system_name, device=device, dtype=dtype, seed=seed)
    else:
        A, B, C, D = get_system(system_name, device=device, dtype=dtype)
    
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    q = min(n, 2)
    r = min(p, 2)
    
    # Noise matrices
    torch.manual_seed(seed)
    Beta = beta_scale * torch.randn(n, q, device=device, dtype=dtype)
    Delta = delta_scale * torch.randn(p, r, device=device, dtype=dtype)
    
    # True controllability
    C_mat = torch.cat([torch.linalg.matrix_power(A, k) @ B for k in range(n)], dim=1)
    true_rank = torch.linalg.matrix_rank(C_mat).item()
    is_controllable = (true_rank == n)
    
    # Compute ell(B)
    ell = compute_observability_index(C, A)
    K = ell
    L_full = K
    L_red = K + 1  # reduced test needs L >= K + 1
    
    # Simulate
    sde = LinearSDE(A, B, C, D, Beta, Delta)
    ts, x, u, y = simulate(sde, T, dt, seed=seed)
    
    # For noisy data, smooth y for derivative stability
    if beta_scale > 0 or delta_scale > 0:
        y_smooth = smooth_signal(y, window_size=11, sigma=2.0, mode="gaussian")
    else:
        y_smooth = y
    
    # Full behavioral test: G_{L,K}(λ) using (u, y) only
    K_matrix_full, cand_full = compute_K_LK(u, y_smooth, L_full, K, n, lam=0.0, dt=dt)
    result_full = check_controllability(
        u, y_smooth, L_full, K, n, m, T, dt,
        candidate_lambdas=cand_full.numpy(),
        threshold=1e-6,
    )
    
    # Reduced test: H_{L,K}(λ) using y only
    K_matrix_red, cand_red = compute_K_LK_reduced(y_smooth, L_red, K, dt)
    result_red = check_controllability_reduced(
        y_smooth, L_red, K, T, dt,
        candidate_lambdas=cand_red.numpy(),
    )
    
    return {
        "system": system_name,
        "n": n, "m": m, "p": p,
        "is_controllable_true": is_controllable,
        "full_test_result": result_full.get("is_controllable", None),
        "reduced_test_result": result_red.get("is_controllable", None),
        "full_min_eigenvalues": result_full.get("min_eigenvalues", []),
        "reduced_min_eigenvalues": result_red.get("min_eigenvalues", []),
        "candidate_lambdas_full": cand_full,
        "candidate_lambdas_red": cand_red,
        "eigvals_A": torch.linalg.eigvals(A),
    }


def main():
    """Run behavioral tests on multiple systems and generate comparison figure."""
    
    systems = ["coupled_spring", "two_spring", "random", "double_integrator"]
    results = []
    
    print("=" * 70)
    print("BEHAVIORAL (INPUT-OUTPUT) CONTROLLABILITY TEST")
    print("=" * 70)
    
    for sys_name in systems:
        print(f"\n--- Testing system: {sys_name} ---")
        try:
            res = run_behavioral_test(sys_name, T=50.0, dt=0.01, beta_scale=0.0, delta_scale=0.0)
            results.append(res)
            print(f"  True controllable: {res['is_controllable_true']}")
            print(f"  Full test result:  {res['full_test_result']}")
            print(f"  Reduced test result: {res['reduced_test_result']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Generate figure: bar chart of minimum eigenvalues at candidates
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)
    
    for idx, res in enumerate(results):
        ax = axes[0, idx]
        
        full_eigs = res["full_min_eigenvalues"]
        red_eigs = res["reduced_min_eigenvalues"]
        
        if len(full_eigs) > 0:
            ax.semilogy(range(len(full_eigs)), np.abs(full_eigs) + 1e-16, 
                       'o-', label="Full $G_{L,K}$", markersize=4)
        if len(red_eigs) > 0:
            ax.semilogy(range(len(red_eigs)), np.abs(red_eigs) + 1e-16,
                       's--', label="Reduced $H_{L,K}$", markersize=4)
        
        ax.axhline(y=1e-6, color='r', linestyle=':', alpha=0.7, label="Threshold")
        ax.set_title(f"{res['system']}\n(ctrl={res['is_controllable_true']})",
                    fontsize=10)
        ax.set_xlabel("Candidate index")
        if idx == 0:
            ax.set_ylabel("Min eigenvalue")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Behavioral Controllability Test: Full vs Reduced", fontsize=12, y=1.02)
    plt.tight_layout()
    
    os.makedirs("examples/figures", exist_ok=True)
    fig.savefig("examples/figures/behavioral_test_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("examples/figures/behavioral_test_comparison.png", bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to examples/figures/behavioral_test_comparison.pdf")
    plt.close(fig)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'System':<20} {'True':>8} {'Full':>8} {'Reduced':>8} {'Match':>8}")
    print("-" * 52)
    for res in results:
        match = (res["full_test_result"] == res["is_controllable_true"])
        print(f"{res['system']:<20} {str(res['is_controllable_true']):>8} "
              f"{str(res['full_test_result']):>8} {str(res['reduced_test_result']):>8} "
              f"{'✓' if match else '✗':>8}")


if __name__ == "__main__":
    main()
