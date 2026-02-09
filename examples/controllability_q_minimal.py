"""Minimal basis-free Pi_{L,K}(lambda) diagnostics.

This example:
1. Simulates one trajectory (u, y).
2. Builds observable quotient coordinates from Xi = Lambda_{L,K}(u, y).
3. Computes data-driven Pi_hat_{L,K}(lambda) = Q_hat Q_hat^*.
4. Computes model reference Pi_mod_{L,K}(lambda) = N (M^* M)^{-1} N^*.
5. Reports per-lambda rank/eigenvalue/model-mismatch diagnostics and saves one figure.

Usage:
    python examples/controllability_q_minimal.py
    python examples/controllability_q_minimal.py --system coupled_spring
"""

import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    LinearSDE,
    compute_K_LK_reduced,
    compute_N_LK_lambda,
    compute_observable_quotient_coordinates,
    compute_Q_LK_from_coordinates,
    compute_observability_index,
    compute_lift_matrix,
    simulate,
    smooth_signal,
    plot_trajectories
)
from src.systems import SYSTEMS, get_system  # noqa: E402


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def compute_model_pi_lambda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
) -> torch.Tensor:
    """Compute Pi_mod(lambda) = N (M^* M)^{-1} N^*."""
    N_lam = compute_N_LK_lambda(A=A, B=B, C=C, D=D, L=L, K=K, lam=lam)
    work_dtype = N_lam.dtype if torch.is_complex(N_lam) else _complex_dtype_from_real(N_lam.dtype)

    M = compute_lift_matrix(C=C, A=A, B=B, D=D, L=L, K=K).to(dtype=work_dtype)
    G_M = M.conj().T @ M
    G_M = 0.5 * (G_M + G_M.conj().T)

    rhs = N_lam.conj().T
    try:
        solved = torch.linalg.solve(G_M, rhs)
    except RuntimeError:
        solved = torch.linalg.lstsq(G_M, rhs).solution

    pi_mod = N_lam @ solved
    return 0.5 * (pi_mod + pi_mod.conj().T)


def compute_data_pi_lambda(
    y: torch.Tensor,
    K: int,
    lam: complex,
    dt: float,
    xi: torch.Tensor,
    gamma_xi: torch.Tensor,
) -> torch.Tensor:
    """Compute Pi_hat(lambda) = Q_hat(lambda) Q_hat(lambda)^*."""
    Q_hat = compute_Q_LK_from_coordinates(
        y=y,
        K=K,
        lam=lam,
        dt=dt,
        xi=xi,
        Gamma_xi=gamma_xi,
    )
    pi_hat = Q_hat @ Q_hat.conj().T
    return 0.5 * (pi_hat + pi_hat.conj().T)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Pi_{L,K}(lambda) basis-free diagnostics")
    parser.add_argument("--system", type=str, default="two_spring", choices=list(SYSTEMS.keys()))
    parser.add_argument("--T", type=float, default=100.0, help="Final horizon")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-scale", type=float, default=0.1, help="Process noise scale")
    parser.add_argument("--delta-scale", type=float, default=0.1, help="Measurement noise scale")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--rank-threshold", type=float, default=1e-8)
    parser.add_argument("--smooth-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--smoothing-sigma", type=float, default=5.0)
    parser.add_argument(
        "--smoothing-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "moving_average"],
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


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
    p = C.shape[0]
    ell = compute_observability_index(C, A)
    K = ell
    L = K + 1

    print(f"System={args.system}, n={n}, m={B.shape[1]}, p={p}, ell={ell}, L={L}, K={K}")

    q = min(n, 1)
    r = min(p, 1)
    if args.beta_scale == 0.0:
        Beta = torch.zeros(n, q, device=device, dtype=dtype)
    else:
        Beta = args.beta_scale * torch.randn(n, q, device=device, dtype=dtype)
    if args.delta_scale == 0.0:
        Delta = torch.zeros(p, r, device=device, dtype=dtype)
    else:
        Delta = args.delta_scale * torch.randn(p, r, device=device, dtype=dtype)

    sde = LinearSDE(A, B, C, D, Beta, Delta)
    ts, x, u, y = simulate(sde, args.T, args.dt, seed=args.seed)

    y_old = y.clone()
    if args.smooth_y:
        y = smooth_signal(
            y,
            window_size=args.smoothing_window,
            sigma=args.smoothing_sigma,
            mode=args.smoothing_mode,
        )
        print(
            f"Smoothed y with {args.smoothing_mode} "
            f"(window={args.smoothing_window}, sigma={args.smoothing_sigma})."
        )

    _, candidate_lambdas = compute_K_LK_reduced(y, L, K, args.dt)
    if args.max_candidates > 0:
        candidate_lambdas = candidate_lambdas[: args.max_candidates]
    if len(candidate_lambdas) == 0:
        raise RuntimeError("No candidate lambdas produced.")

    # candidate_lambdas = [0.5 + 1.0j, 0.5 - 1.0j][: args.max_candidates]

    coords = compute_observable_quotient_coordinates(
        u=u,
        y=y,
        L=L,
        K=K,
        n=n,
        dt=args.dt,
    )

    print("\nCoordinate diagnostics:")
    print(f"  Xi shape: {tuple(coords['Xi'].shape)}")
    print(f"  xi shape: {tuple(coords['xi'].shape)}")
    print(f"  target rank r = Lm+n: {coords['target_rank']}")
    print(f"  numerical rank(Xi): {coords['numerical_rank']}")
    print(f"  cond(Gamma_xi): {coords['condition_number']:.3e}")

    xi = coords["xi"]
    gamma = coords["Gamma_xi"]

    pi_min_eigs: List[float] = []
    pi_ranks: List[int] = []
    pi_abs_errors: List[float] = []
    pi_rel_errors: List[float] = []

    print("\nPer-lambda diagnostics:")
    for i, lam in enumerate(candidate_lambdas):
        lam_val = lam.item() if torch.is_tensor(lam) else lam
        pi_data = compute_data_pi_lambda(
            y=y,
            K=K,
            lam=lam_val,
            dt=args.dt,
            xi=xi,
            gamma_xi=gamma,
        )
        pi_mod = compute_model_pi_lambda(
            A=A,
            B=B,
            C=C,
            D=D,
            L=L,
            K=K,
            lam=lam_val,
        )

        eigs_pi = torch.linalg.eigvalsh(pi_data).real
        eigs_mod = torch.linalg.eigvalsh(pi_mod).real

        fig, axes = plt.subplots(2, 2, figsize=(10, 4))
        axes[0, 0].imshow(pi_data.real.cpu().numpy(), aspect="auto")
        axes[0, 1].imshow(pi_mod.real.cpu().numpy(), aspect="auto")
        axes[1, 0].plot(eigs_pi.cpu().numpy(), marker="o")
        axes[1, 1].plot(eigs_mod.cpu().numpy(), marker="o")
        #set colorbar for the top row
        for ax in axes[0, :]:
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        axes[0, 0].set_title(r"$\widehat{\Pi}_{L,K}(\lambda)$")
        axes[0, 1].set_title(r"$\Pi^{\mathrm{mod}}_{L,K}(\lambda)$")
        axes[1, 0].set_title(r"Eigenvalues of $\widehat{\Pi}_{L,K}(\lambda)$")
        axes[1, 1].set_title(r"Eigenvalues of $\Pi^{\mathrm{mod}}_{L,K}(\lambda)$")
        fig.tight_layout()
        # fig.savefig(os.path.join("figures", f"pi_lambda_{i:02d}.pdf"), bbox_inches="tight")
        
        rank_pi = int((eigs_pi > args.rank_threshold).sum().item())
        min_eig_pi = float(eigs_pi.min().item())
        abs_err = float(torch.linalg.matrix_norm(pi_data - pi_mod, ord=2).item())
        mod_norm = float(torch.linalg.matrix_norm(pi_mod, ord=2).item())
        rel_err = abs_err / max(mod_norm, 1e-16)

        pi_min_eigs.append(min_eig_pi)
        pi_ranks.append(rank_pi)
        pi_abs_errors.append(abs_err)
        pi_rel_errors.append(rel_err)

        print(
            f"  [{i:02d}] lambda={lam_val.real:+.4f}{lam_val.imag:+.4f}i, "
            f"rank(Pi)={rank_pi}/{K*p}, lambda_min(Pi)={min_eig_pi:.3e}, "
            f"||Pi_data-Pi_mod||_2={abs_err:.3e}, rel={rel_err:.3e}"
        )

    
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    idx = np.arange(len(candidate_lambdas))

    fig1 = plot_trajectories(ts, x, u, y_old)
    out_pdf = os.path.join(out_dir, f"trajectories_{args.system}.pdf")
    fig1.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved figure: {out_pdf}")
    print

    if args.show:
        plt.show()
    plt.close(fig1)


if __name__ == "__main__":
    main()
