"""Minimal noisy/clean/real-Pi convergence of Pi controllability diagnostics.

This example mirrors the quotient/Pi notation used in
`examples/controllability_q_minimal.py` and keeps only two final plots:

    min_{lambda in Lambda_cand} lambda_min(Pi_hat_{L,K}(lambda; T))
    max_{lambda in Lambda_cand} ||Pi_hat_{L,K}(lambda; T)||_2

evaluated across horizons for clean and noisy trajectories, and compared
against the model/real Pi reference computed from (A, B, C, D).

Usage:
    python examples/controllability_noise_convergence.py
    python examples/controllability_noise_convergence.py --system coupled_spring
"""

import argparse
import os
import sys
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    add_friction_cli_args,
    build_sde,
    compute_K_LK_reduced,
    compute_N_LK_lambda,
    compute_Q_LK_from_coordinates,
    compute_observable_quotient_coordinates,
    compute_observability_index,
    compute_lift_matrix,
    friction_params_from_namespace,
    load_system_with_friction,
    simulate,
    smooth_signal,
)
from src.systems import SYSTEMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal clean/noisy convergence with real-Pi references."
    )
    parser.add_argument("--system", type=str, default="coupled_spring", choices=list(SYSTEMS.keys()))
    parser.add_argument(
        "--friction-model",
        type=str,
        default="none",
        choices=["none", "coulomb", "stribeck"],
        help="Friction model for spring systems",
    )
    add_friction_cli_args(parser)
    parser.add_argument("--T", type=float, default=100.0, help="Final horizon")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--min-T", type=float, default=10.0, help="Minimum horizon")
    parser.add_argument("--num-horizons", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-scale", type=float, default=0.1, help="Process noise scale")
    parser.add_argument("--delta-scale", type=float, default=0.1, help="Measurement noise scale")
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--smooth-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument(
        "--smoothing-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "moving_average"],
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _as_complex_lambda(lam: complex) -> complex:
    return complex(lam)


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def simulate_case(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dynamics_fn,
    T: float,
    dt: float,
    beta_scale: float,
    delta_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n = A.shape[0]
    p = C.shape[0]
    q = min(n, 1)
    r = min(p, 1)

    if beta_scale == 0.0:
        Beta = torch.zeros(n, q, device=A.device, dtype=A.dtype)
    else:
        Beta = beta_scale * torch.randn(n, q, device=A.device, dtype=A.dtype)

    if delta_scale == 0.0:
        Delta = torch.zeros(p, r, device=A.device, dtype=A.dtype)
    else:
        Delta = delta_scale * torch.randn(p, r, device=A.device, dtype=A.dtype)

    sde = build_sde(A, B, C, D, Beta, Delta, dynamics_fn=dynamics_fn)
    _, _, u, y = simulate(sde, T, dt, seed=seed)
    return u, y


def maybe_smooth(y: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.smooth_y:
        return y
    return smooth_signal(
        y,
        window_size=args.smoothing_window,
        sigma=args.smoothing_sigma,
        mode=args.smoothing_mode,
    )


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


def compute_model_pi_lambda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
) -> torch.Tensor:
    """Compute model/real Pi(lambda) = N (M^* M)^{-1} N^*."""
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


def build_horizons(min_T: float, T: float, num_horizons: int) -> np.ndarray:
    horizons = np.unique(np.round(np.geomspace(min_T, T, num_horizons)).astype(int))
    horizons = horizons[horizons >= int(np.ceil(min_T))]
    if len(horizons) == 0 or horizons[-1] != int(round(T)):
        horizons = np.append(horizons, int(round(T)))
    return horizons.astype(float)


def pi_margin_and_spectral_norm_at_horizon(
    u: torch.Tensor,
    y: torch.Tensor,
    Ti: float,
    dt: float,
    L: int,
    K: int,
    n: int,
    candidate_lambdas: Sequence[complex],
    rank_tol: float,
) -> Tuple[float, float]:
    n_i = int(round(Ti / dt)) + 1
    u_i = u[:n_i]
    y_i = y[:n_i]

    coords = compute_observable_quotient_coordinates(
        u=u_i,
        y=y_i,
        L=L,
        K=K,
        n=n,
        dt=dt,
        rank_tol=rank_tol,
    )

    per_lambda_mins: List[float] = []
    per_lambda_spec_norms: List[float] = []
    for lam in candidate_lambdas:
        pi_hat = compute_data_pi_lambda(
            y=y_i,
            K=K,
            lam=lam,
            dt=dt,
            xi=coords["xi"],
            gamma_xi=coords["Gamma_xi"],
        )
        min_eig = float(torch.linalg.eigvalsh(pi_hat).real.min().item())
        spec_norm = float(torch.linalg.matrix_norm(pi_hat, ord=2).item())
        per_lambda_mins.append(min_eig)
        per_lambda_spec_norms.append(spec_norm)

    margin = float(np.min(np.array(per_lambda_mins, dtype=np.float64)))
    spectral = float(np.max(np.array(per_lambda_spec_norms, dtype=np.float64)))
    return margin, spectral


def pi_margin_and_spectral_norm_from_map(
    pi_map: dict,
    candidate_lambdas: Sequence[complex],
) -> Tuple[float, float]:
    per_lambda_mins: List[float] = []
    per_lambda_spec_norms: List[float] = []
    for lam in candidate_lambdas:
        pi_lam = pi_map[lam]
        per_lambda_mins.append(float(torch.linalg.eigvalsh(pi_lam).real.min().item()))
        per_lambda_spec_norms.append(float(torch.linalg.matrix_norm(pi_lam, ord=2).item()))
    margin = float(np.min(np.array(per_lambda_mins, dtype=np.float64)))
    spectral = float(np.max(np.array(per_lambda_spec_norms, dtype=np.float64)))
    return margin, spectral


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    friction_params = friction_params_from_namespace(args)
    A, B, C, D, dynamics_fn = load_system_with_friction(
        system_name=args.system,
        device=device,
        dtype=dtype,
        seed=args.seed,
        friction_model=args.friction_model,
        friction_params=friction_params,
    )

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    ell = compute_observability_index(C, A)
    K = ell
    L = K + 1

    print(
        f"System={args.system}, friction={args.friction_model}, "
        f"n={n}, m={m}, p={p}, ell={ell}, L={L}, K={K}"
    )

    u_clean, y_clean = simulate_case(
        A,
        B,
        C,
        D,
        dynamics_fn,
        args.T,
        args.dt,
        beta_scale=0.0,
        delta_scale=0.0,
        seed=args.seed,
    )
    u_noisy, y_noisy = simulate_case(
        A,
        B,
        C,
        D,
        dynamics_fn,
        args.T,
        args.dt,
        beta_scale=args.beta_scale,
        delta_scale=args.delta_scale,
        seed=args.seed,
    )

    y_clean = maybe_smooth(y_clean, args)
    y_noisy = maybe_smooth(y_noisy, args)

    try:
        _, candidate_lambdas_raw = compute_K_LK_reduced(
            y_clean,
            L,
            K,
            args.dt,
            rank_tol=args.rank_tol,
        )
        candidate_lambdas = [
            _as_complex_lambda(l.item() if torch.is_tensor(l) else l)
            for l in candidate_lambdas_raw
        ]
    except ValueError:
        candidate_lambdas = [0.5 + 1.0j, 0.5 - 1.0j]

    if args.max_candidates > 0:
        candidate_lambdas = candidate_lambdas[: args.max_candidates]
    if len(candidate_lambdas) == 0:
        raise RuntimeError("No candidate lambdas available.")

    print(f"Using {len(candidate_lambdas)} candidate lambdas.")

    horizons = build_horizons(args.min_T, args.T, args.num_horizons)

    margins_clean = np.full(len(horizons), np.nan, dtype=np.float64)
    margins_noisy = np.full(len(horizons), np.nan, dtype=np.float64)
    specs_clean = np.full(len(horizons), np.nan, dtype=np.float64)
    specs_noisy = np.full(len(horizons), np.nan, dtype=np.float64)
    margins_real = np.full(len(horizons), np.nan, dtype=np.float64)
    specs_real = np.full(len(horizons), np.nan, dtype=np.float64)

    pi_real_map = {
        lam: compute_model_pi_lambda(
            A=A,
            B=B,
            C=C,
            D=D,
            L=L,
            K=K,
            lam=lam,
        )
        for lam in candidate_lambdas
    }
    margin_real, spec_real = pi_margin_and_spectral_norm_from_map(pi_real_map, candidate_lambdas)
    margins_real[:] = margin_real
    specs_real[:] = spec_real

    print("\nEstimating Pi-hat margin and spectral-norm by horizon...")
    for i, Ti in enumerate(horizons):
        try:
            margins_clean[i], specs_clean[i] = pi_margin_and_spectral_norm_at_horizon(
                u=u_clean,
                y=y_clean,
                Ti=Ti,
                dt=args.dt,
                L=L,
                K=K,
                n=n,
                candidate_lambdas=candidate_lambdas,
                rank_tol=args.rank_tol,
            )
        except ValueError:
            pass

        try:
            margins_noisy[i], specs_noisy[i] = pi_margin_and_spectral_norm_at_horizon(
                u=u_noisy,
                y=y_noisy,
                Ti=Ti,
                dt=args.dt,
                L=L,
                K=K,
                n=n,
                candidate_lambdas=candidate_lambdas,
                rank_tol=args.rank_tol,
            )
        except ValueError:
            pass

        clean_str = f"{margins_clean[i]:.3e}" if np.isfinite(margins_clean[i]) else "nan"
        noisy_str = f"{margins_noisy[i]:.3e}" if np.isfinite(margins_noisy[i]) else "nan"
        real_str = f"{margins_real[i]:.3e}" if np.isfinite(margins_real[i]) else "nan"
        clean_spec_str = f"{specs_clean[i]:.3e}" if np.isfinite(specs_clean[i]) else "nan"
        noisy_spec_str = f"{specs_noisy[i]:.3e}" if np.isfinite(specs_noisy[i]) else "nan"
        real_spec_str = f"{specs_real[i]:.3e}" if np.isfinite(specs_real[i]) else "nan"
        print(
            f"  T={Ti:7.1f}: margin(clean={clean_str}, noisy={noisy_str}, real={real_str}), "
            f"spec(clean={clean_spec_str}, noisy={noisy_spec_str}, real={real_spec_str})"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    valid_clean = np.isfinite(margins_clean)
    valid_noisy = np.isfinite(margins_noisy)
    valid_real = np.isfinite(margins_real)
    if valid_clean.any():
        ax1.plot(
            horizons[valid_clean],
            margins_clean[valid_clean],
            "o-",
            linewidth=2,
            markersize=5,
            label="clean",
        )
    if valid_noisy.any():
        ax1.plot(
            horizons[valid_noisy],
            margins_noisy[valid_noisy],
            "s--",
            linewidth=2,
            markersize=5,
            label="noisy",
        )
    if valid_real.any():
        ax1.plot(
            horizons[valid_real],
            margins_real[valid_real],
            "^-.",
            linewidth=2,
            markersize=5,
            label=r"$\Pi$ real (model)",
        )

    ax1.set_xlabel("Horizon T")
    ax1.set_ylabel(
        r"$\min_{\lambda\in\Lambda_{\mathrm{cand}}}\lambda_{\min}(\widehat{\Pi}_{L,K}(\lambda;T))$"
    )
    ax1.set_title(r"Convergence of $\widehat{\Pi}_{L,K}$ margin")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    valid_spec_clean = np.isfinite(specs_clean)
    valid_spec_noisy = np.isfinite(specs_noisy)
    valid_spec_real = np.isfinite(specs_real)
    if valid_spec_clean.any():
        ax2.plot(
            horizons[valid_spec_clean],
            specs_clean[valid_spec_clean],
            "o-",
            linewidth=2,
            markersize=5,
            label="clean",
        )
    if valid_spec_noisy.any():
        ax2.plot(
            horizons[valid_spec_noisy],
            specs_noisy[valid_spec_noisy],
            "s--",
            linewidth=2,
            markersize=5,
            label="noisy",
        )
    if valid_spec_real.any():
        ax2.plot(
            horizons[valid_spec_real],
            specs_real[valid_spec_real],
            "^-.",
            linewidth=2,
            markersize=5,
            label=r"$\Pi$ real (model)",
        )

    ax2.set_xlabel("Horizon T")
    ax2.set_ylabel(
        r"$\max_{\lambda\in\Lambda_{\mathrm{cand}}}\|\widehat{\Pi}_{L,K}(\lambda;T)\|_2$"
    )
    ax2.set_title(r"Spectral norm of $\widehat{\Pi}_{L,K}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    fig.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    out_pdf = os.path.join(
        output_dir,
        f"q_noise_convergence_minimal_{args.system}_{args.friction_model}.pdf",
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved figure: {out_pdf}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
