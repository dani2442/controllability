"""Experiment: Input conditioning comparison under Sobolev budgets.

Compares different input designs on conditioning of the Gramian Γ_L(u):
  1. Random input
  2. L^2-optimal (cosine basis, Corollary 4.2)
  3. H^1-optimal (Sobolev-weighted cosine basis, Proposition 4.1)
  4. Sinusoidal PE (multisine, Proposition 4.4)

Validates Propositions 4.1, 4.3, and 4.4 from the paper.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gramians import (
    compute_persistent_excitation_gramian,
    check_persistent_excitation,
    compute_G_LK,
    compute_K_LK,
    check_controllability,
)
from src.utils import compute_observability_index
from src import LinearSDE, simulate
from src.systems import get_system


def l2_optimal_input(T, dt, m, Q=None):
    """L^2-optimal input: first m cosine basis functions (Corollary 4.2).
    
    Satisfies Γ_1(u) = (1/m) I_m.
    """
    N = int(T / dt) + 1
    ts = torch.linspace(0, T, N, dtype=torch.float64)
    
    u = torch.zeros(N, m, dtype=torch.float64)
    for k in range(m):
        if k == 0:
            u[:, k] = 1.0 / np.sqrt(T)
        else:
            u[:, k] = np.sqrt(2.0 / T) * torch.cos(k * np.pi * ts / T)
    
    # Normalize to unit H^0 (L^2) norm
    norm_sq = torch.sum(u**2) * dt
    u = u / torch.sqrt(norm_sq)
    
    if Q is not None:
        u = u @ Q.T
    
    return ts, u


def h1_optimal_input(T, dt, m, Q=None):
    """H^1-optimal input: Sobolev-weighted cosine basis (Proposition 4.1 with p=1).
    
    Uses the first m Neumann eigenfunctions with H^1 normalization.
    """
    N = int(T / dt) + 1
    ts = torch.linspace(0, T, N, dtype=torch.float64)
    
    # Compute Sobolev weights s_k = 1 + (k*pi/T)^2
    psi = torch.zeros(N, m, dtype=torch.float64)
    s = torch.zeros(m, dtype=torch.float64)
    
    for k in range(m):
        omega_k = k * np.pi / T
        s[k] = 1.0 + omega_k**2  # H^1 weight
        if k == 0:
            psi[:, k] = 1.0 / np.sqrt(T)
        else:
            psi[:, k] = np.sqrt(2.0 / T) * torch.cos(k * np.pi * ts / T)
    
    alpha = 1.0 / torch.sum(s).item()
    u = np.sqrt(alpha) * psi
    
    if Q is not None:
        u = u @ Q.T
    
    return ts, u


def sinusoidal_pe_input(T, dt, m, L, p=1, Q=None):
    """Sinusoidal PE input with explicit conditioning (Proposition 4.4).
    
    Uses mL distinct frequencies to guarantee Γ_L(u) > 0.
    """
    N = int(T / dt) + 1
    ts = torch.linspace(0, T, N, dtype=torch.float64)
    
    u = torch.zeros(N, m, dtype=torch.float64)
    
    for i in range(m):
        for j in range(L):
            idx = i * L + j + 1  # 1-based index
            omega = 2 * np.pi * idx / T
            S_p = sum(omega**(2*k) for k in range(p + 1))
            a_sq = 2.0 / (T * m * L * S_p)
            u[:, i] += np.sqrt(a_sq) * torch.cos(omega * ts)
    
    if Q is not None:
        u = u @ Q.T
    
    return ts, u


def random_input(T, dt, m, seed=42):
    """Random smooth input (band-limited noise)."""
    torch.manual_seed(seed)
    N = int(T / dt) + 1
    ts = torch.linspace(0, T, N, dtype=torch.float64)
    
    # Generate random input as sum of random sinusoids
    n_freqs = 10
    u = torch.zeros(N, m, dtype=torch.float64)
    for i in range(m):
        for k in range(1, n_freqs + 1):
            amp = torch.randn(1, dtype=torch.float64).item() / n_freqs
            freq = k * 2 * np.pi / T
            phase = torch.rand(1, dtype=torch.float64).item() * 2 * np.pi
            u[:, i] += amp * torch.cos(freq * ts + phase)
    
    # Normalize to unit L^2 norm
    norm_sq = torch.sum(u**2) * dt
    u = u / torch.sqrt(norm_sq)
    
    return ts, u


def compute_conditioning(u, L, dt):
    """Compute conditioning metrics for Γ_L(u)."""
    gramian = compute_persistent_excitation_gramian(u, order=L, dt=dt)
    eigenvalues = torch.linalg.eigvalsh(gramian)
    
    lmin = eigenvalues[0].item()
    lmax = eigenvalues[-1].item()
    cond = lmax / max(lmin, 1e-16)
    trace = eigenvalues.sum().item()
    
    return {
        "lambda_min": lmin,
        "lambda_max": lmax,
        "condition_number": cond,
        "trace": trace,
        "eigenvalues": eigenvalues.numpy(),
    }


def main():
    """Compare input designs on conditioning of Γ_L(u)."""
    
    print("=" * 70)
    print("INPUT CONDITIONING COMPARISON UNDER SOBOLEV BUDGETS")
    print("=" * 70)
    
    # Parameters
    T = 20.0
    dt = 0.01
    m = 3
    
    L_values = [1, 2, 3, 4]
    
    # Input designs
    designs = {
        "Random": lambda T, dt, m, L: random_input(T, dt, m),
        "$L^2$-optimal": lambda T, dt, m, L: l2_optimal_input(T, dt, m),
        "$H^1$-optimal": lambda T, dt, m, L: h1_optimal_input(T, dt, m),
        "Sinusoidal PE": lambda T, dt, m, L: sinusoidal_pe_input(T, dt, m, L, p=1),
    }
    
    # Collect results
    results = {name: {"lambda_min": [], "cond": []} for name in designs}
    
    for L in L_values:
        print(f"\n--- L = {L} ---")
        for name, gen_fn in designs.items():
            ts, u = gen_fn(T, dt, m, L)
            metrics = compute_conditioning(u, L, dt)
            results[name]["lambda_min"].append(metrics["lambda_min"])
            results[name]["cond"].append(metrics["condition_number"])
            print(f"  {name:20s}: λ_min = {metrics['lambda_min']:.6f}, "
                  f"cond = {metrics['condition_number']:.2f}")
    
    # Also test downstream effect on controllability test accuracy
    print("\n" + "=" * 70)
    print("DOWNSTREAM EFFECT ON CONTROLLABILITY CERTIFICATION")
    print("=" * 70)
    
    device = torch.device("cpu")
    dtype = torch.float64
    A, B, C, D = get_system("coupled_spring", device=device, dtype=dtype)
    n = A.shape[0]
    ell = compute_observability_index(C, A)
    K = ell
    L_test = K
    
    print(f"System: coupled_spring (n={n}, m={B.shape[1]}, p={C.shape[0]})")
    print(f"Using L={L_test}, K={K}")
    
    beta_scale = 0.1
    torch.manual_seed(42)
    Beta = beta_scale * torch.randn(n, 2, device=device, dtype=dtype)
    Delta = 0.05 * torch.randn(C.shape[0], 1, device=device, dtype=dtype)
    
    downstream_results = {}
    
    for name, gen_fn in designs.items():
        ts_input, u_designed = gen_fn(T, dt, B.shape[1], L_test)
        
        # Create SDE with designed input
        control_fn = lambda t, x, u_data=u_designed, dt_val=dt: u_data[min(int(t.item() / dt_val), len(u_data)-1)].unsqueeze(0).expand(x.shape[0], -1)
        
        sde = LinearSDE(A, B, C, D, Beta, Delta, control_fn=control_fn)
        ts, x, u, y = simulate(sde, T, dt, seed=42)
        
        # Run controllability test
        K_matrix, cand = compute_K_LK(u, y, L_test, K, n, lam=0.0, dt=dt)
        result = check_controllability(u, y, L_test, K, n, B.shape[1], T, dt,
                                       candidate_lambdas=cand.numpy(), threshold=1e-6)
        
        min_margin = min(result.get("min_eigenvalues", [0]))
        downstream_results[name] = {
            "is_controllable": result.get("is_controllable", False),
            "min_margin": min_margin,
        }
        print(f"  {name:20s}: controllable = {result.get('is_controllable', '?')}, "
              f"min margin = {min_margin:.6f}")
    
    # Generate figures
    os.makedirs("examples/figures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Panel 1: λ_min(Γ_L) vs L
    ax = axes[0]
    markers = ['o-', 's--', 'D-.', '^:']
    for (name, data), marker in zip(results.items(), markers):
        ax.semilogy(L_values, [max(v, 1e-16) for v in data["lambda_min"]], 
                    marker, label=name, linewidth=1.5, markersize=6)
    ax.set_xlabel("Lift order $L$")
    ax.set_ylabel("$\\lambda_{\\min}(\\Gamma_L(u))$")
    ax.set_title("Min eigenvalue vs lift order")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    
    # Panel 2: Condition number vs L
    ax = axes[1]
    for (name, data), marker in zip(results.items(), markers):
        ax.semilogy(L_values, data["cond"], marker, label=name, linewidth=1.5, markersize=6)
    ax.set_xlabel("Lift order $L$")
    ax.set_ylabel("Condition number")  
    ax.set_title("Conditioning vs lift order")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    
    # Panel 3: Downstream controllability margins
    ax = axes[2]
    names = list(downstream_results.keys())
    margins = [downstream_results[n]["min_margin"] for n in names]
    colors = ['#2196F3' if downstream_results[n]["is_controllable"] else '#F44336' for n in names]
    bars = ax.bar(range(len(names)), margins, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel("Min controllability margin")
    ax.set_title("Downstream certification margin")
    ax.axhline(y=1e-6, color='r', linestyle=':', alpha=0.7, label="Threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig("examples/figures/input_conditioning_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("examples/figures/input_conditioning_comparison.png", bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to examples/figures/input_conditioning_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
