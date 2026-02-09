"""Empirical sweep to visualize the asymptotic Pi-bias scaling in Δt.

This script is a companion to the discussion in
`paper/sections/hautus_general_alternative.tex` (EIV bias under stochastic noise).

It compares:
    Pi_hat_{L,K}(λ;T) = Q_hat Q_hat^*
to
    Pi_mod_{L,K}(λ)   = N(M^* M)^{-1} N^*

across a list of sampling steps `dt`, holding noise intensity fixed.

Typical usage (measurement-noise dominated):
    python examples/pi_bias_scaling_sweep.py --noise measurement --delta-scale 0.05 --beta-scale 0.0

Process-noise dominated:
    python examples/pi_bias_scaling_sweep.py --noise process --beta-scale 0.05 --delta-scale 0.0
"""

import argparse
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    LinearSDE,
    compute_N_LK_lambda,
    compute_Q_LK_from_coordinates,
    compute_lift_matrix,
    compute_observable_quotient_coordinates,
    compute_observability_index,
    simulate,
    smooth_signal,
)
from src.systems import SYSTEMS, get_system  # noqa: E402


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def parse_dts(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    out = [float(p) for p in parts]
    if any(dt <= 0.0 for dt in out):
        raise argparse.ArgumentTypeError("All dt must be positive.")
    return out


def parse_lambdas(value: str) -> List[complex]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    out: List[complex] = []
    for p in parts:
        out.append(complex(p))
    return out


def compute_pi_mod(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    lam: complex,
) -> torch.Tensor:
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


def compute_pi_hat(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    dt: float,
    lam: complex,
    rank_tol: float,
) -> torch.Tensor:
    coords = compute_observable_quotient_coordinates(
        u=u,
        y=y,
        L=L,
        K=K,
        n=n,
        dt=dt,
        rank_tol=rank_tol,
    )
    Q_hat = compute_Q_LK_from_coordinates(
        y=y,
        K=K,
        lam=lam,
        dt=dt,
        xi=coords["xi"],
        Gamma_xi=coords["Gamma_xi"],
    )
    pi_hat = Q_hat @ Q_hat.conj().T
    return 0.5 * (pi_hat + pi_hat.conj().T)


def rate_fit_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    valid = np.isfinite(x) & (x > 0.0) & np.isfinite(y) & (y > 0.0)
    xv = x[valid]
    yv = y[valid]
    if len(yv) < 2:
        return float("nan"), float("nan"), int(len(yv))
    slope, intercept = np.polyfit(np.log(xv), np.log(yv), deg=1)
    return float(slope), float(intercept), int(len(yv))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical dt-sweep of Pi_hat vs Pi_mod mismatch.")
    parser.add_argument("--system", type=str, default="coupled_spring", choices=list(SYSTEMS.keys()))
    parser.add_argument("--T", type=float, default=400.0, help="Horizon (larger helps reveal plateau).")
    parser.add_argument(
        "--dts",
        type=parse_dts,
        default=parse_dts("0.2,0.1,0.05,0.025"),
        help="Comma-separated dt list, e.g. '0.2,0.1,0.05'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-scale", type=float, default=0.05)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--noise",
        type=str,
        default="both",
        choices=["measurement", "process", "both"],
        help="Which noise source(s) to include.",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=parse_lambdas("0.5+1j,0.5-1j"),
        help="Comma-separated complex lambdas, e.g. '0.5+1j,0.5-1j'.",
    )
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--smooth-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument(
        "--smoothing-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "moving_average"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    print(f"Noise mode={args.noise}, beta_scale={args.beta_scale}, delta_scale={args.delta_scale}")
    print(f"Lambdas={args.lambdas}")

    dt_vals: List[float] = []
    err_vals: List[float] = []

    for dt in args.dts:
        torch.manual_seed(args.seed)

        q = min(n, 1)
        r = min(p, 1)
        if args.noise in ("process", "both") and args.beta_scale != 0.0:
            Beta = args.beta_scale * torch.randn(n, q, device=device, dtype=dtype)
        else:
            Beta = torch.zeros(n, q, device=device, dtype=dtype)

        if args.noise in ("measurement", "both") and args.delta_scale != 0.0:
            Delta = args.delta_scale * torch.randn(p, r, device=device, dtype=dtype)
        else:
            Delta = torch.zeros(p, r, device=device, dtype=dtype)

        sde = LinearSDE(A, B, C, D, Beta, Delta)
        _, _, u, y = simulate(sde, args.T, dt, seed=args.seed)

        if args.smooth_y:
            y = smooth_signal(
                y,
                window_size=args.smoothing_window,
                sigma=args.smoothing_sigma,
                mode=args.smoothing_mode,
            )

        max_err = 0.0
        for lam in args.lambdas:
            pi_hat = compute_pi_hat(u=u, y=y, L=L, K=K, n=n, dt=dt, lam=lam, rank_tol=args.rank_tol)
            pi_mod = compute_pi_mod(A=A, B=B, C=C, D=D, L=L, K=K, lam=lam)
            err = float(torch.linalg.matrix_norm(pi_hat - pi_mod, ord=2).item())
            max_err = max(max_err, err)

        dt_vals.append(float(dt))
        err_vals.append(float(max_err))
        print(f"dt={dt:>8.5f} | max_lambda ||Pi_hat - Pi_mod||_2 = {max_err:.3e}")

    slope, intercept, n_used = rate_fit_loglog(np.array(dt_vals), np.array(err_vals))
    if np.isfinite(slope):
        print(f"\nLog-log fit using {n_used} points: log(err) ≈ {slope:+.3f} log(dt) + {intercept:+.3f}")
        print(f"Implied power law: err ≈ const * dt^{slope:+.3f}")


if __name__ == "__main__":
    main()

