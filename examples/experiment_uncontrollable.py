"""Experiment: Uncontrollable system detection (negative test).

Tests that the data-driven Hautus test correctly identifies systems that
are NOT controllable. Uses the two_spring system (decoupled springs where
the second mass is unreachable) as the primary example.

Also compares against a controllable variant (coupled_spring) to show
that the test can both accept and reject appropriately.
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
)
from src.systems import get_system, print_system_info


def run_test(system_name, T=50.0, dt=0.01, beta_scale=0.0, delta_scale=0.0, seed=42):
    """Run controllability test on a system and return detailed results."""
    device = torch.device("cpu")
    dtype = torch.float64
    
    A, B, C, D = get_system(system_name, device=device, dtype=dtype)
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    q = min(n, 2)
    r = min(p, 2)
    
    torch.manual_seed(seed)
    Beta = beta_scale * torch.randn(n, q, device=device, dtype=dtype)
    Delta = delta_scale * torch.randn(p, r, device=device, dtype=dtype)
    
    # True controllability
    C_mat = torch.cat([torch.linalg.matrix_power(A, k) @ B for k in range(n)], dim=1)
    true_rank = torch.linalg.matrix_rank(C_mat).item()
    is_controllable_true = (true_rank == n)
    
    # Eigenvalues of A
    eigvals_A = torch.linalg.eigvals(A)
    
    # Compute ell(B)
    ell = compute_observability_index(C, A)
    K = ell
    L = K
    
    # Simulate
    sde = LinearSDE(A, B, C, D, Beta, Delta)
    ts, x, u, y = simulate(sde, T, dt, seed=seed)
    
    # Full test
    K_matrix, cand = compute_K_LK(u, y, L, K, n, lam=0.0, dt=dt)
    result = check_controllability(u, y, L, K, n, m, T, dt,
                                    candidate_lambdas=cand.numpy(), threshold=1e-6)
    
    # Also compute G_{L,K}(λ) at the true eigenvalues of A  
    test_lambdas = eigvals_A.numpy()
    margins_at_eigvals = []
    for lam_val in test_lambdas:
        G = compute_G_LK(u, y, L, K, lam_val, T, dt)
        eigs_G = torch.linalg.eigvalsh(G.real if G.is_complex() else G)
        margins_at_eigvals.append(eigs_G.numpy())
    
    return {
        "system": system_name,
        "n": n, "m": m, "p": p, "ell": ell, "L": L, "K": K,
        "is_controllable_true": is_controllable_true,
        "true_controllability_rank": true_rank,
        "test_result": result.get("is_controllable", None),
        "min_eigenvalues": result.get("min_eigenvalues", []),
        "candidate_lambdas": cand.numpy(),
        "eigvals_A": eigvals_A.numpy(),
        "margins_at_eigvals": margins_at_eigvals,
    }


def main():
    """Compare controllable vs uncontrollable systems."""
    
    print("=" * 70)
    print("UNCONTROLLABLE SYSTEM DETECTION (NEGATIVE TEST)")
    print("=" * 70)
    
    systems = [
        ("two_spring", "Decoupled springs (NOT controllable)"),
        ("coupled_spring", "Coupled springs (controllable)"),
    ]
    
    results = []
    for sys_name, description in systems:
        print(f"\n{'='*60}")
        print(f"Testing: {sys_name} — {description}")
        print(f"{'='*60}")
        
        res = run_test(sys_name, T=50.0, dt=0.01, beta_scale=0.0)
        results.append(res)
        
        print(f"  Dimensions: n={res['n']}, m={res['m']}, p={res['p']}")
        print(f"  Observability index: ell={res['ell']}")
        print(f"  True controllability rank: {res['true_controllability_rank']}/{res['n']}")
        print(f"  True controllable: {res['is_controllable_true']}")
        print(f"  Test result: {res['test_result']}")
        
        print(f"\n  Eigenvalues of A:")
        for i, ev in enumerate(res['eigvals_A']):
            print(f"    λ_{i+1} = {ev.real:.4f} + {ev.imag:.4f}j")
        
        print(f"\n  Min eigenvalues of G_{{L,K}}(λ) at candidates:")
        for i, (lam, eig) in enumerate(zip(res['candidate_lambdas'][:10], res['min_eigenvalues'][:10])):
            marker = "✓" if eig > 1e-6 else "✗ (RANK DROP)"
            print(f"    λ={lam:.4f}: σ_min = {eig:.6e}  {marker}")
    
    # Generate figure
    os.makedirs("examples/figures", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    for row, res in enumerate(results):
        # Left: candidate eigenvalues in complex plane with margin color
        ax = axes[row, 0]
        cands = res['candidate_lambdas']
        min_eigs = np.array(res['min_eigenvalues'])
        
        # Plot candidates colored by margin
        if len(cands) > 0 and len(min_eigs) > 0:
            n_plot = min(len(cands), len(min_eigs))
            scatter = ax.scatter(
                cands[:n_plot].real, cands[:n_plot].imag,
                c=np.log10(np.abs(min_eigs[:n_plot]) + 1e-16),
                cmap='RdYlGn', s=80, edgecolors='black', linewidths=0.5,
                vmin=-12, vmax=0,
            )
            plt.colorbar(scatter, ax=ax, label="$\\log_{10}\\sigma_{\\min}$")
        
        # Plot true eigenvalues of A
        ax.scatter(res['eigvals_A'].real, res['eigvals_A'].imag,
                  marker='x', color='blue', s=100, linewidths=2, label='$\\sigma(A)$', zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Re($\\lambda$)")
        ax.set_ylabel("Im($\\lambda$)")
        ax.set_title(f"{res['system']}: candidates in $\\mathbb{{C}}$\n"
                     f"(ctrl={res['is_controllable_true']}, test={res['test_result']})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right: eigenvalue spectrum of G_{L,K}(λ) at eigenvalues of A
        ax = axes[row, 1]
        for i, (ev, eigs_G) in enumerate(zip(res['eigvals_A'], res['margins_at_eigvals'])):
            label = f"$\\lambda = {ev.real:.2f}{'+' if ev.imag >= 0 else ''}{ev.imag:.2f}j$"
            ax.semilogy(range(1, len(eigs_G) + 1), np.sort(eigs_G)[::-1] + 1e-16,
                       'o-', markersize=3, label=label)
        
        expected_rank = res['L'] * res['m'] + res['n']
        ax.axvline(x=expected_rank, color='red', linestyle='--', alpha=0.7,
                   label=f"Expected rank $Lm+n={expected_rank}$")
        ax.axhline(y=1e-6, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue magnitude")
        ax.set_title(f"{res['system']}: eigenvalue spectrum of $G_{{L,K}}(\\lambda)$")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig("examples/figures/uncontrollable_detection.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("examples/figures/uncontrollable_detection.png", bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to examples/figures/uncontrollable_detection.pdf")
    plt.close(fig)
    
    # Noisy version
    print("\n" + "=" * 70)
    print("NOISY CASE (β=0.1, δ=0.05)")
    print("=" * 70)
    
    results_noisy = []
    for sys_name, description in systems:
        print(f"\n--- {sys_name}: {description} ---")
        res = run_test(sys_name, T=100.0, dt=0.01, beta_scale=0.1, delta_scale=0.05)
        results_noisy.append(res)
        print(f"  True controllable: {res['is_controllable_true']}")
        print(f"  Test result: {res['test_result']}")
        print(f"  Min margins: {[f'{v:.2e}' for v in res['min_eigenvalues'][:5]]}")


if __name__ == "__main__":
    main()
