"""Compare full vs reduced controllability margins with/without noise.

This example:
1. Simulates two trajectories up to T=1000s: noiseless and noisy.
2. Runs both finite-candidate tests:
   - full: rank test on G_{L,K}(lambda)
   - reduced: rank test on H_{L,K}(lambda)
3. For each horizon, estimates the minimum eigenvalue margin:
   - min eigenvalue of G (full test)
   - min eigenvalue of H (reduced test)
4. Plots these estimated minimum eigenvalues vs horizon.

Usage:
    python examples/controllability_noise_convergence.py
    python examples/controllability_noise_convergence.py --system coupled_spring
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    LinearSDE,
    check_controllability,
    check_controllability_reduced,
    compute_K_LK,
    compute_K_LK_reduced,
    compute_observability_index,
    simulate,
)
from src.systems import SYSTEMS, get_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convergence of estimated minimum eigenvalues of G and H (noise vs no-noise)."
    )
    parser.add_argument(
        "--system",
        type=str,
        default="coupled_spring",
        choices=list(SYSTEMS.keys()),
    )
    parser.add_argument("--T", type=float, default=100.0, help="Final horizon (seconds).")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--num-horizons",
        type=int,
        default=10,
        help="Number of smaller horizons from min-T up to T.",
    )
    parser.add_argument("--min-T", type=float, default=10.0, help="Smallest horizon.")
    parser.add_argument("--beta-scale", type=float, default=0.1, help="Process noise scale.")
    parser.add_argument(
        "--delta-scale", type=float, default=0.1, help="Measurement noise scale."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def simulate_case(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    T: float,
    dt: float,
    beta_scale: float,
    delta_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.shape[0]
    p = C.shape[0]
    q = min(n, 1)
    r = min(p, 1)
    Beta = beta_scale * torch.randn(n, q, device=A.device, dtype=A.dtype)
    Delta = delta_scale * torch.randn(p, r, device=A.device, dtype=A.dtype)
    sde = LinearSDE(A, B, C, D, Beta, Delta)
    return simulate(sde, T, dt, seed=seed)


def estimate_min_eig_paths(
    u: torch.Tensor,
    y: torch.Tensor,
    horizons: np.ndarray,
    n: int,
    m: int,
    L_full: int,
    K_full: int,
    L_red: int,
    K_red: int,
    dt: float,
    threshold: float = 1e-6,
) -> Dict[str, np.ndarray]:
    out = {
        "G": np.full(len(horizons), np.nan, dtype=np.float64),
        "H": np.full(len(horizons), np.nan, dtype=np.float64),
    }
    for i, Ti in enumerate(horizons):
        n_i = int(round(Ti / dt)) + 1
        u_i = u[:n_i]
        y_i = y[:n_i]
        T_i = (n_i - 1) * dt

        try:
            _, cand_full = compute_K_LK(u_i, y_i, L_full, K_full, n, 0.0, dt)
            res_full = check_controllability(
                u_i,
                y_i,
                L_full,
                K_full,
                n,
                m,
                T_i,
                dt,
                candidate_lambdas=cand_full,
                threshold=threshold,
            )
            out["G"][i] = float(res_full["min_eigenvalues"].min().item())
        except Exception:
            pass

        try:
            _, cand_red = compute_K_LK_reduced(y_i, L_red, K_red, dt)
            res_red = check_controllability_reduced(
                y_i,
                L_red,
                K_red,
                T_i,
                dt,
                candidate_lambdas=cand_red,
                threshold=threshold,
            )
            out["H"][i] = float(res_red["min_eigenvalues"].min().item())
        except Exception:
            pass

    return out


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    if args.system == "random":
        A, B, C, D = get_system(args.system, device=device, dtype=dtype, seed=args.seed)
    else:
        A, B, C, D = get_system(args.system, device=device, dtype=dtype)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    ell = compute_observability_index(C, A)
    # Tight choices under corrected bounds.
    K_full = ell
    L_full = K_full
    K_red = ell
    L_red = K_red + 1

    horizons = np.unique(np.round(np.geomspace(args.min_T, args.T, args.num_horizons)).astype(int))
    horizons = horizons[horizons >= int(np.ceil(args.min_T))]
    if len(horizons) == 0 or horizons[-1] != int(round(args.T)):
        horizons = np.append(horizons, int(round(args.T)))
    horizons = horizons.astype(float)

    print(f"System: {args.system}, n={n}, m={m}, p={p}")
    print(
        f"L_full={L_full}, K_full={K_full}, "
        f"L_red={L_red}, K_red={K_red}, ell={ell}, T={args.T}, dt={args.dt}"
    )
    print("Simulating trajectories (no-noise and noisy) ...")

    _, _, u_clean, y_clean = simulate_case(
        A,
        B,
        C,
        D,
        args.T,
        args.dt,
        beta_scale=0.0,
        delta_scale=0.0,
        seed=args.seed,
    )
    _, _, u_noisy, y_noisy = simulate_case(
        A,
        B,
        C,
        D,
        args.T,
        args.dt,
        beta_scale=args.beta_scale,
        delta_scale=args.delta_scale,
        seed=args.seed + 1,
    )

    est_clean = estimate_min_eig_paths(
        u_clean, y_clean, horizons, n, m, L_full, K_full, L_red, K_red, args.dt
    )
    est_noisy = estimate_min_eig_paths(
        u_noisy, y_noisy, horizons, n, m, L_full, K_full, L_red, K_red, args.dt
    )

    print("\nFinal-horizon controllability checks:")
    _, cand_full_ref = compute_K_LK(u_clean, y_clean, L_full, K_full, n, 0.0, args.dt)
    _, cand_red_ref = compute_K_LK_reduced(y_clean, L_red, K_red, args.dt)
    full_res_clean = check_controllability(
        u_clean, y_clean, L_full, K_full, n, m, args.T, args.dt, candidate_lambdas=cand_full_ref
    )
    red_res_clean = check_controllability_reduced(
        y_clean, L_red, K_red, args.T, args.dt, candidate_lambdas=cand_red_ref
    )
    _, cand_full_noisy = compute_K_LK(u_noisy, y_noisy, L_full, K_full, n, 0.0, args.dt)
    _, cand_red_noisy = compute_K_LK_reduced(y_noisy, L_red, K_red, args.dt)
    full_res_noisy = check_controllability(
        u_noisy, y_noisy, L_full, K_full, n, m, args.T, args.dt, candidate_lambdas=cand_full_noisy
    )
    red_res_noisy = check_controllability_reduced(
        y_noisy, L_red, K_red, args.T, args.dt, candidate_lambdas=cand_red_noisy
    )
    print(
        f"  no-noise: full={full_res_clean['is_controllable']}, "
        f"reduced={red_res_clean['is_controllable']}"
    )
    print(
        f"  noisy:    full={full_res_noisy['is_controllable']}, "
        f"reduced={red_res_noisy['is_controllable']}"
    )
    print(
        f"  est. min eig G (clean/noisy): "
        f"{full_res_clean['min_eigenvalues'].min().item():.6e} / "
        f"{full_res_noisy['min_eigenvalues'].min().item():.6e}"
    )
    print(
        f"  est. min eig H (clean/noisy): "
        f"{red_res_clean['min_eigenvalues'].min().item():.6e} / "
        f"{red_res_noisy['min_eigenvalues'].min().item():.6e}"
    )

    print("\nConvergence table (estimated minimum eigenvalues):")
    print("  T      min_eig_G_clean  min_eig_G_noisy  min_eig_H_clean   min_eig_H_noisy")
    for i, Ti in enumerate(horizons):
        print(
            f"  {Ti:6.1f}  {est_clean['G'][i]:15.6e}  {est_noisy['G'][i]:15.6e}  "
            f"{est_clean['H'][i]:15.6e}  {est_noisy['H'][i]:15.6e}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    curves = [
        ("min eig G, no-noise", est_clean["G"], "o-"),
        ("min eig G, noisy", est_noisy["G"], "o--"),
        ("min eig H, no-noise", est_clean["H"], "s-"),
        ("min eig H, noisy", est_noisy["H"], "s--"),
    ]
    for label, vals, style in curves:
        valid = np.isfinite(vals) & np.isfinite(horizons) & (horizons > 0)
        if valid.any():
            ax.plot(horizons[valid], vals[valid], style, linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Estimated minimum eigenvalue")
    ax.set_title("Estimated minimum eigenvalues of G and H")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    out_pdf = os.path.join(output_dir, f"full_vs_reduced_noise_min_eigs_{args.system}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved: {out_pdf}")


if __name__ == "__main__":
    main()
