"""Main example: Controllability analysis via derivative-lifted Gramians.

This script demonstrates:
1. Generate an LTI system (A, B, C, D) from various example systems
2. Simulate the stochastic system using torchsde:
   dx = (Ax + Bu)dt + β dW(t)    (process noise)
   y  = Cx + Du + δv(t)           (measurement noise)
3. Compute G_{L,K}(λ) and K_{L,K}(λ) matrices
4. Check controllability via rank conditions
5. Visualize results

Usage:
    python examples/main.py                    # Use default random system
    python examples/main.py --system random
    python examples/main.py --system two_spring
    python examples/main.py --system coupled_spring
    python examples/main.py --system double_integrator
    python examples/main.py --system inverted_pendulum
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
    compute_G_LK,
    compute_K_LK,
    check_controllability,
    plot_trajectories,
    plot_eigenvalues,
    plot_gramian_eigenvalues,
    plot_controllability_margin,
)
from systems import get_system, print_system_info, SYSTEMS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Controllability analysis via derivative-lifted Gramians"
    )
    parser.add_argument(
        "--system", 
        type=str, 
        default="two_spring",
        choices=list(SYSTEMS.keys()),
        help=f"System to analyze. Available: {list(SYSTEMS.keys())}"
    )
    return parser.parse_args()


def main(system_name: str = "random"):
    # =========================================================================
    # Configuration
    # =========================================================================
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Using device: {device}")
    print(f"Selected system: {system_name}")
    
    # Derivative lift parameters
    L = 3   # Input derivative levels
    K = 3   # Output derivative levels
    
    # Simulation parameters
    T = 10.0       # Final time
    dt = 0.01      # Time step
    
    # Noise intensities
    beta_scale = 0.1   # Process noise intensity
    delta_scale = 0.05 # Measurement noise intensity
    
    # =========================================================================
    # Generate system from examples
    # =========================================================================
    print("\n" + "="*60)
    print(f"LOADING SYSTEM: {system_name.upper()}")
    print("="*60)
    
    # Get system matrices
    if system_name == "random":
        A, B, C, D = get_system(system_name, device=device, dtype=dtype, seed=42)
    else:
        A, B, C, D = get_system(system_name, device=device, dtype=dtype)
    
    # Extract dimensions from the system
    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Input dimension
    p = C.shape[0]  # Output dimension
    q = min(n, 2)   # Process noise dimension
    r = min(p, 2)   # Measurement noise dimension
    
    # Noise matrices
    Beta = beta_scale * torch.randn(n, q, device=device, dtype=dtype)
    Delta = delta_scale * torch.randn(p, r, device=device, dtype=dtype)
    
    # Check eigenvalues of A
    eigvals_A = torch.linalg.eigvals(A)
    print(f"\nSystem dimensions: n={n}, m={m}, p={p}")
    print(f"Eigenvalues of A:")
    for i, ev in enumerate(eigvals_A):
        print(f"  λ_{i+1} = {ev.real:.4f} + {ev.imag:.4f}i")
    print(f"Max Re(λ) = {eigvals_A.real.max():.4f} (should be < 0 for stability)")
    
    # Check controllability of (A, B)
    C_mat = torch.cat([torch.linalg.matrix_power(A, k) @ B for k in range(n)], dim=1)
    rank_C = torch.linalg.matrix_rank(C_mat).item()
    print(f"\nControllability matrix rank: {rank_C} (should be {n} for controllable)")
    is_controllable_AB = (rank_C == n)
    print(f"(A, B) is controllable: {is_controllable_AB}")
    
    # Check observability of (A, C)
    O_mat = torch.cat([C @ torch.linalg.matrix_power(A, k) for k in range(n)], dim=0)
    rank_O = torch.linalg.matrix_rank(O_mat).item()
    print(f"Observability matrix rank: {rank_O} (should be {n} for observable)")
    is_observable_AC = (rank_O == n)
    print(f"(A, C) is observable: {is_observable_AC}")
    
    # =========================================================================
    # Simulate SDE
    # =========================================================================
    print("\n" + "="*60)
    print("SIMULATING STOCHASTIC SYSTEM")
    print("="*60)
    
    sde = LinearSDE(A, B, C, D, Beta, Delta)
    ts, x, u, y = simulate(sde, T, dt)
    
    print(f"\nSimulation complete:")
    print(f"  Time horizon: T = {T}")
    print(f"  Time step: dt = {dt}")
    print(f"  Number of samples: N = {len(ts)}")
    print(f"  State trajectory shape: {x.shape}")
    print(f"  Input trajectory shape: {u.shape}")
    print(f"  Output trajectory shape: {y.shape}")
    
    # =========================================================================
    # Compute K_{L,K}(λ) and candidate eigenvalues
    # =========================================================================
    print("\n" + "="*60)
    print(f"COMPUTING K_{{L,K}}(λ) with L={L}, K={K}")
    print("="*60)
    
    K_matrix, candidate_lambdas = compute_K_LK(u, y, L, K, lam=0.0, dt=dt)
    
    print(f"\nK_{{L,K}} matrix shape: {K_matrix.shape}")
    print(f"Number of candidate eigenvalues: {len(candidate_lambdas)}")
    print(f"\nCandidate eigenvalues (from K_{{L,K}}):")
    for i, lam in enumerate(candidate_lambdas[:10]):
        print(f"  μ_{i+1} = {lam.real:.4f} + {lam.imag:.4f}i")
    if len(candidate_lambdas) > 10:
        print(f"  ... ({len(candidate_lambdas) - 10} more)")
    
    # =========================================================================
    # Compute G_{L,K}(λ) at multiple λ values
    # =========================================================================
    print("\n" + "="*60)
    print(f"COMPUTING G_{{L,K}}(λ) GRAMIANS")
    print("="*60)
    
    expected_rank = L * m + n
    print(f"\nExpected rank for controllability: Lm + n = {L}×{m} + {n} = {expected_rank}")
    
    # Test at a few candidate lambdas and some fixed values
    test_lambdas = [0.0, 0.5j, -0.5, -0.5 + 0.5j]
    if len(candidate_lambdas) > 0:
        test_lambdas.extend([candidate_lambdas[0].item(), candidate_lambdas[-1].item()])
    
    eigenvalues_list = []
    lambda_labels = []
    margins = []
    
    for lam in test_lambdas:
        G = compute_G_LK(u, y, L, K, lam, dt)
        
        # Eigenvalues
        eigvals = torch.linalg.eigvalsh(G).real
        eigvals_sorted = torch.sort(eigvals, descending=True).values
        eigenvalues_list.append(eigvals_sorted)
        
        # Label
        if isinstance(lam, complex):
            label = f"λ = {lam.real:.2f}{lam.imag:+.2f}i"
        else:
            label = f"λ = {lam:.2f}"
        lambda_labels.append(label)
        
        # Controllability margin
        if len(eigvals_sorted) >= expected_rank:
            margin = eigvals_sorted[expected_rank - 1].item()
        else:
            margin = 0.0
        margins.append(margin)
        
        print(f"\n{label}:")
        print(f"  G_{{L,K}} shape: {G.shape}")
        print(f"  Top 5 eigenvalues: {eigvals_sorted[:5].cpu().numpy()}")
        print(f"  λ_{{Lm+n}} = {margin:.6f}")
        print(f"  Thresholded rank (τ=1e-6): {(eigvals > 1e-6).sum().item()}")
    
    # =========================================================================
    # Check controllability
    # =========================================================================
    print("\n" + "="*60)
    print("CONTROLLABILITY CHECK")
    print("="*60)
    
    result = check_controllability(
        u, y, L, K, n, m, dt,
        candidate_lambdas=candidate_lambdas[:20] if len(candidate_lambdas) > 20 else candidate_lambdas,
        threshold=1e-6
    )
    
    print(f"\nExpected rank: {result['expected_rank']}")
    print(f"Ranks at candidate λ: {result['ranks'].cpu().numpy()}")
    print(f"Min margin: {result['min_eigenvalues'].min():.6e}")
    print(f"\nIS CONTROLLABLE: {result['is_controllable']}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. System trajectories
    fig1 = plot_trajectories(ts, x, u, y)
    fig1.savefig(os.path.join(output_dir, "trajectories.pdf"), bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, "trajectories.png"), dpi=150, bbox_inches='tight')
    print("  Saved: trajectories.pdf/png")
    
    # 2. Eigenvalues of A
    fig2 = plot_eigenvalues(eigvals_A, title="Eigenvalues of System Matrix $A$")
    fig2.savefig(os.path.join(output_dir, "eigenvalues_A.pdf"), bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "eigenvalues_A.png"), dpi=150, bbox_inches='tight')
    print("  Saved: eigenvalues_A.pdf/png")
    
    # 3. Eigenvalue decay of G_{L,K}(λ)
    fig3 = plot_gramian_eigenvalues(eigenvalues_list, lambda_labels, expected_rank)
    fig3.savefig(os.path.join(output_dir, "gramian_eigenvalues.pdf"), bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, "gramian_eigenvalues.png"), dpi=150, bbox_inches='tight')
    print("  Saved: gramian_eigenvalues.pdf/png")
    
    # 4. Controllability margins
    test_lambdas_complex = torch.tensor([complex(l) for l in test_lambdas])
    margins_tensor = torch.tensor(margins)
    fig4 = plot_controllability_margin(test_lambdas_complex, margins_tensor)
    fig4.savefig(os.path.join(output_dir, "controllability_margin.pdf"), bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, "controllability_margin.png"), dpi=150, bbox_inches='tight')
    print("  Saved: controllability_margin.pdf/png")
    
    # 5. Candidate eigenvalues from K_{L,K}
    fig5 = plot_eigenvalues(candidate_lambdas, title=r"Candidate Eigenvalues from $K_{L,K}$")
    fig5.savefig(os.path.join(output_dir, "candidate_eigenvalues.pdf"), bbox_inches='tight')
    fig5.savefig(os.path.join(output_dir, "candidate_eigenvalues.png"), dpi=150, bbox_inches='tight')
    print("  Saved: candidate_eigenvalues.pdf/png")
    
    print(f"\nAll figures saved to: {output_dir}")
    
    # Show plots
    plt.show()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    
    return {
        "A": A, "B": B, "C": C, "D": D,
        "ts": ts, "x": x, "u": u, "y": y,
        "K_matrix": K_matrix,
        "candidate_lambdas": candidate_lambdas,
        "controllability_result": result,
    }


if __name__ == "__main__":
    args = parse_args()
    results = main(system_name=args.system)
