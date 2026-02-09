r"""Convergence of basis-free minimum-eigenvalue Pi-errors under stochastic noise.

This example estimates the rate

    |\lambda_min(Pi_hat_{L,K}(lambda; T)) - \lambda_min(Pi_mod_{L,K}(lambda))|
    = O_P(T^{-1/2})

on a fixed finite candidate set Lambda_cand.

For each horizon T_i and noisy run j, define

    e_j(T_i) = max_{lambda in Lambda_cand}
               |\lambda_min(Pi_hat^{(j)}_{L,K}(lambda; T_i))
                - \lambda_min(Pi_mod_{L,K}(lambda))|.

The script reports aggregated statistics over Monte Carlo runs and checks:
1) log-log slope of mean error vs T is close to -1/2,
2) log-log slope of sqrt(T) * mean error is close to 0.
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    LinearSDE,
    compute_K_LK_reduced,
    compute_N_LK_lambda,
    compute_Q_LK_from_coordinates,
    compute_observable_quotient_coordinates,
    compute_observability_index,
    compute_lift_matrix,
    simulate,
    smooth_signal,
)
from src.systems import SYSTEMS, get_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convergence of minimum-eigenvalue Pi_hat-to-Pi_mod errors under stochastic noise."
    )
    parser.add_argument("--system", type=str, default="random", choices=list(SYSTEMS.keys()))
    parser.add_argument("--T", type=float, default=200.0, help="Final horizon")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--min-T", type=float, default=1.0, help="Minimum horizon")
    parser.add_argument("--num-horizons", type=int, default=8)
    parser.add_argument("--mc-runs", type=int, default=2, help="Monte Carlo noisy realizations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-scale", type=float, default=0.01, help="Process noise scale")
    parser.add_argument("--delta-scale", type=float, default=0.01, help="Measurement noise scale")
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--smooth-y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--smoothing-sigma", type=float, default=10.0)
    parser.add_argument(
        "--smoothing-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "moving_average"],
    )
    parser.add_argument("--logy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--assert-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert T^{-1/2} rate diagnostics from fitted slopes.",
    )
    parser.add_argument(
        "--rate-slope-target",
        type=float,
        default=-0.5,
        help="Target log-log slope for mean error vs T.",
    )
    parser.add_argument(
        "--rate-slope-tol",
        type=float,
        default=0.20,
        help="Tolerance around rate-slope-target.",
    )
    parser.add_argument(
        "--scaled-slope-abs-max",
        type=float,
        default=0.20,
        help="Max allowed absolute slope for sqrt(T)*mean_error vs T.",
    )
    return parser.parse_args()


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def _as_complex_lambda(lam: complex) -> complex:
    return complex(lam)


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

    sde = LinearSDE(A, B, C, D, Beta, Delta)
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


def compute_model_pi_map(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    L: int,
    K: int,
    candidate_lambdas: Sequence[complex],
) -> Dict[complex, torch.Tensor]:
    """Compute Pi_mod(lambda)=N(M^*M)^{-1}N^* for each lambda."""
    work_dtype = A.dtype if torch.is_complex(A) else _complex_dtype_from_real(A.dtype)
    M = compute_lift_matrix(C=C, A=A, B=B, D=D, L=L, K=K).to(dtype=work_dtype)
    G_M = M.conj().T @ M
    G_M = 0.5 * (G_M + G_M.conj().T)

    pi_mod_map: Dict[complex, torch.Tensor] = {}
    for lam in candidate_lambdas:
        N_lam = compute_N_LK_lambda(A=A, B=B, C=C, D=D, L=L, K=K, lam=lam)
        rhs = N_lam.conj().T
        try:
            solved = torch.linalg.solve(G_M, rhs)
        except RuntimeError:
            solved = torch.linalg.lstsq(G_M, rhs).solution
        pi_mod = N_lam @ solved
        pi_mod_map[lam] = 0.5 * (pi_mod + pi_mod.conj().T)
    return pi_mod_map


def compute_pi_bundle_for_horizon(
    u: torch.Tensor,
    y: torch.Tensor,
    L: int,
    K: int,
    n: int,
    dt: float,
    T: float,
    candidate_lambdas: Sequence[complex],
    rank_tol: float,
) -> Dict[str, object]:
    coords = compute_observable_quotient_coordinates(
        u=u,
        y=y,
        L=L,
        K=K,
        n=n,
        dt=dt,
        rank_tol=rank_tol,
    )
    pi_hat_map: Dict[complex, torch.Tensor] = {}
    for lam in candidate_lambdas:
        Q_hat = compute_Q_LK_from_coordinates(
            y=y,
            K=K,
            lam=lam,
            dt=dt,
            xi=coords["xi"],
            Gamma_xi=coords["Gamma_xi"],
        )
        pi_hat = Q_hat @ Q_hat.conj().T
        pi_hat_map[lam] = 0.5 * (pi_hat + pi_hat.conj().T)

    return {
        "Pi_map": pi_hat_map,
        "target_rank": coords["target_rank"],
        "numerical_rank": coords["numerical_rank"],
    }


def _min_eigval(mat: torch.Tensor) -> float:
    return float(torch.linalg.eigvalsh(mat).real.min().item())


def worst_case_min_eig_error(
    pi_hat_map: Dict[complex, torch.Tensor],
    pi_mod_map: Dict[complex, torch.Tensor],
    candidate_lambdas: Sequence[complex],
) -> float:
    errs = [
        abs(_min_eigval(pi_hat_map[lam]) - _min_eigval(pi_mod_map[lam]))
        for lam in candidate_lambdas
    ]
    return float(np.max(np.array(errs, dtype=np.float64)))


def worst_case_spectral_matrix_error(
    pi_hat_map: Dict[complex, torch.Tensor],
    pi_mod_map: Dict[complex, torch.Tensor],
    candidate_lambdas: Sequence[complex],
) -> float:
    errs = [
        float(torch.linalg.matrix_norm(pi_hat_map[lam] - pi_mod_map[lam], ord=2).item())
        for lam in candidate_lambdas
    ]
    return float(np.max(np.array(errs, dtype=np.float64)))


def rate_fit_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    valid = np.isfinite(x) & (x > 0.0) & np.isfinite(y) & (y > 0.0)
    xv = x[valid]
    yv = y[valid]
    if len(yv) < 2:
        return float("nan"), float("nan"), int(len(yv))
    slope, intercept = np.polyfit(np.log(xv), np.log(yv), deg=1)
    return float(slope), float(intercept), int(len(yv))


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
    K = ell
    L = K + 1

    print(
        f"System={args.system}, n={n}, m={m}, p={p}, ell={ell}, "
        f"L={L}, K={K}, T={args.T}, dt={args.dt}, mc_runs={args.mc_runs}"
    )

    u_clean, y_clean = simulate_case(
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
    y_clean = maybe_smooth(y_clean, args)

    try:
        _, candidate_lambdas_raw = compute_K_LK_reduced(
            y_clean,
            L,
            K,
            args.dt,
            rank_tol=args.rank_tol,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Failed to build candidate lambda set from clean final-horizon data. "
            "Try increasing T, reducing K, or loosening rank tolerance."
        ) from exc

    candidate_lambdas = [
        _as_complex_lambda(l.item() if torch.is_tensor(l) else l) for l in candidate_lambdas_raw
    ]
    if args.max_candidates > 0:
        candidate_lambdas = candidate_lambdas[: args.max_candidates]
    if len(candidate_lambdas) == 0:
        raise RuntimeError("No candidate lambdas produced from clean final-horizon data.")
    print(f"Using {len(candidate_lambdas)} fixed candidate lambdas from clean final horizon.")

    pi_mod_map = compute_model_pi_map(
        A=A,
        B=B,
        C=C,
        D=D,
        L=L,
        K=K,
        candidate_lambdas=candidate_lambdas,
    )

    noisy_runs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for j in range(args.mc_runs):
        u_j, y_j = simulate_case(
            A,
            B,
            C,
            D,
            args.T,
            args.dt,
            beta_scale=args.beta_scale,
            delta_scale=args.delta_scale,
            seed=args.seed + 1000 + j,
        )
        y_j = maybe_smooth(y_j, args)
        noisy_runs.append((u_j, y_j))

    horizons = np.unique(np.round(np.geomspace(args.min_T, args.T, args.num_horizons)).astype(int))
    horizons = horizons[horizons >= int(np.ceil(args.min_T))]
    if len(horizons) == 0 or horizons[-1] != int(round(args.T)):
        horizons = np.append(horizons, int(round(args.T)))
    horizons = horizons.astype(float)

    clean_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    mean_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    std_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    median_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    q90_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    q95_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    scaled_mean_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    clean_spec_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    mean_spec_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    std_spec_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    q95_spec_errors = np.full(len(horizons), np.nan, dtype=np.float64)
    valid_counts = np.zeros(len(horizons), dtype=np.int64)

    print("\nEvaluating worst-case minimum-eigenvalue errors per horizon...")
    for i, Ti in enumerate(horizons):
        n_i = int(round(Ti / args.dt)) + 1
        u_clean_i = u_clean[:n_i]
        y_clean_i = y_clean[:n_i]

        try:
            clean_bundle = compute_pi_bundle_for_horizon(
                u=u_clean_i,
                y=y_clean_i,
                L=L,
                K=K,
                n=n,
                dt=args.dt,
                T=args.T,
                candidate_lambdas=candidate_lambdas,
                rank_tol=args.rank_tol,
            )
        except ValueError:
            print(f"  T={Ti:7.1f}: clean rank condition failed -> skipping")
            continue

        clean_errors[i] = worst_case_min_eig_error(
            pi_hat_map=clean_bundle["Pi_map"],
            pi_mod_map=pi_mod_map,
            candidate_lambdas=candidate_lambdas,
        )
        clean_spec_errors[i] = worst_case_spectral_matrix_error(
            pi_hat_map=clean_bundle["Pi_map"],
            pi_mod_map=pi_mod_map,
            candidate_lambdas=candidate_lambdas,
        )

        run_errors: List[float] = []
        run_spec_errors: List[float] = []
        for (u_noisy_full, y_noisy_full) in noisy_runs:
            u_noisy_i = u_noisy_full[:n_i]
            y_noisy_i = y_noisy_full[:n_i]
            try:
                noisy_bundle = compute_pi_bundle_for_horizon(
                    u=u_noisy_i,
                    y=y_noisy_i,
                    L=L,
                    K=K,
                    n=n,
                    dt=args.dt,
                    T=args.T,
                    candidate_lambdas=candidate_lambdas,
                    rank_tol=args.rank_tol,
                )
            except ValueError:
                continue

            run_errors.append(
                worst_case_min_eig_error(
                    pi_hat_map=noisy_bundle["Pi_map"],
                    pi_mod_map=pi_mod_map,
                    candidate_lambdas=candidate_lambdas,
                )
            )
            run_spec_errors.append(
                worst_case_spectral_matrix_error(
                    pi_hat_map=noisy_bundle["Pi_map"],
                    pi_mod_map=pi_mod_map,
                    candidate_lambdas=candidate_lambdas,
                )
            )

        if run_errors:
            run_np = np.array(run_errors, dtype=np.float64)
            run_spec_np = np.array(run_spec_errors, dtype=np.float64)
            mean_errors[i] = float(run_np.mean())
            std_errors[i] = float(run_np.std(ddof=0))
            median_errors[i] = float(np.median(run_np))
            q90_errors[i] = float(np.quantile(run_np, 0.90))
            q95_errors[i] = float(np.quantile(run_np, 0.95))
            scaled_mean_errors[i] = float(np.sqrt(Ti) * mean_errors[i])
            mean_spec_errors[i] = float(run_spec_np.mean())
            std_spec_errors[i] = float(run_spec_np.std(ddof=0))
            q95_spec_errors[i] = float(np.quantile(run_spec_np, 0.95))
            valid_counts[i] = int(len(run_errors))

            print(
                f"  T={Ti:7.1f}: clean={clean_errors[i]:.3e}, "
                f"mean={mean_errors[i]:.3e}, std={std_errors[i]:.3e}, "
                f"q95={q95_errors[i]:.3e}, sqrt(T)*mean={scaled_mean_errors[i]:.3e}, "
                f"spec_mean={mean_spec_errors[i]:.3e}, "
                f"valid_runs={valid_counts[i]:d}/{args.mc_runs}"
            )
        else:
            print(f"  T={Ti:7.1f}: no valid noisy runs")

    print("\nSummary table (worst-case min-eig gap over fixed candidate lambdas):")
    print("  Horizon         mean_e          std_e           q95_e      sqrt(T)*mean_e    ValidRuns")
    for Ti, em, es, eq95, esm, vc in zip(
        horizons,
        mean_errors,
        std_errors,
        q95_errors,
        scaled_mean_errors,
        valid_counts,
    ):
        em_s = f"{em:.3e}" if np.isfinite(em) else "nan"
        es_s = f"{es:.3e}" if np.isfinite(es) else "nan"
        eq95_s = f"{eq95:.3e}" if np.isfinite(eq95) else "nan"
        esm_s = f"{esm:.3e}" if np.isfinite(esm) else "nan"
        print(f"  {Ti:7.1f}   {em_s:>12}   {es_s:>12}   {eq95_s:>12}   {esm_s:>14}   {vc:3d}/{args.mc_runs}")

    slope_err, intercept_err, n_err = rate_fit_loglog(horizons, mean_errors)
    slope_scaled, intercept_scaled, n_scaled = rate_fit_loglog(horizons, scaled_mean_errors)

    slope_err_ok = np.isfinite(slope_err) and abs(slope_err - args.rate_slope_target) <= args.rate_slope_tol
    slope_scaled_ok = np.isfinite(slope_scaled) and abs(slope_scaled) <= args.scaled_slope_abs_max
    rate_ok = slope_err_ok and slope_scaled_ok

    print("\nRate diagnostics:")
    print(f"  fit_points_mean={n_err}, slope_mean={slope_err:.6f}, intercept_mean={intercept_err:.6f}")
    print(f"  fit_points_scaled={n_scaled}, slope_scaled={slope_scaled:.6f}, intercept_scaled={intercept_scaled:.6f}")
    print(
        f"  criteria: |slope_mean-({args.rate_slope_target:.3f})| <= {args.rate_slope_tol:.3f} "
        f"and |slope_scaled| <= {args.scaled_slope_abs_max:.3f}"
    )
    print(f"  rate_pass={rate_ok}")

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q_noise_convergence.pdf")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.2))

    valid_clean = np.isfinite(clean_errors)
    if valid_clean.any():
        ax1.plot(horizons[valid_clean], clean_errors[valid_clean], "--", marker="^", label="clean-vs-model")

    valid_mean = np.isfinite(mean_errors)
    if valid_mean.any():
        x = horizons[valid_mean]
        y = mean_errors[valid_mean]
        s = std_errors[valid_mean]
        ax1.plot(x, y, marker="o", label=r"$\mathbb{E}[e(T)]$")
        ax1.fill_between(x, np.maximum(y - s, 1e-16), y + s, alpha=0.2)

    valid_q95 = np.isfinite(q95_errors)
    if valid_q95.any():
        ax1.plot(horizons[valid_q95], q95_errors[valid_q95], marker="s", linestyle="-.", label=r"$q_{0.95}(e(T))$")

    if valid_mean.any():
        idx0 = np.where(valid_mean)[0][0]
        c_ref = mean_errors[idx0] * np.sqrt(horizons[idx0])
        ref_curve = c_ref / np.sqrt(horizons[valid_mean])
        ax1.plot(horizons[valid_mean], ref_curve, ":", color="k", label=r"$c\,T^{-1/2}$")

    if args.logy:
        ax1.set_yscale("log")

    ax1.set_xlabel("Horizon T")
    ax1.set_ylabel(
        r"$\max_{\lambda}|\lambda_{\min}(\widehat{\Pi}(\lambda;T))-\lambda_{\min}(\Pi^{\mathrm{mod}}(\lambda))|$"
    )
    ax1.set_title(r"$\lambda_{\min}$-error decay")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    valid_scaled = np.isfinite(scaled_mean_errors)
    if valid_scaled.any():
        x = horizons[valid_scaled]
        y = scaled_mean_errors[valid_scaled]
        ax2.plot(x, y, marker="d", label=r"$\sqrt{T}\,\mathbb{E}[e(T)]$")
        ax2.axhline(y=float(y[0]), linestyle=":", color="k", label="first-horizon level")

    ax2.set_xlabel("Horizon T")
    ax2.set_ylabel(r"$\sqrt{T}\,\mathbb{E}[e(T)]$")
    ax2.set_title("Scaled error stability")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    if np.isfinite(slope_err) and np.isfinite(slope_scaled):
        ax2.text(
            0.02,
            0.03,
            f"slope_mean={slope_err:.3f}\nslope_scaled={slope_scaled:.3f}",
            transform=ax2.transAxes,
            fontsize=10,
        )

    valid_clean_spec = np.isfinite(clean_spec_errors)
    if valid_clean_spec.any():
        ax3.plot(
            horizons[valid_clean_spec],
            clean_spec_errors[valid_clean_spec],
            "--",
            marker="^",
            label="clean-vs-model",
        )

    valid_mean_spec = np.isfinite(mean_spec_errors)
    if valid_mean_spec.any():
        x = horizons[valid_mean_spec]
        y = mean_spec_errors[valid_mean_spec]
        s = std_spec_errors[valid_mean_spec]
        ax3.plot(x, y, marker="o", label=r"$\mathbb{E}[\max_{\lambda}\|\widehat{\Pi}-\Pi^{\mathrm{mod}}\|_2]$")
        ax3.fill_between(x, np.maximum(y - s, 1e-16), y + s, alpha=0.2)

    valid_q95_spec = np.isfinite(q95_spec_errors)
    if valid_q95_spec.any():
        ax3.plot(
            horizons[valid_q95_spec],
            q95_spec_errors[valid_q95_spec],
            marker="s",
            linestyle="-.",
            label=r"$q_{0.95}(\max_{\lambda}\|\widehat{\Pi}-\Pi^{\mathrm{mod}}\|_2)$",
        )

    if args.logy:
        ax3.set_yscale("log")

    ax3.set_xlabel("Horizon T")
    ax3.set_ylabel(r"$\max_{\lambda}\|\widehat{\Pi}_{L,K}(\lambda;T)-\Pi^{\mathrm{mod}}_{L,K}(\lambda)\|_2$")
    ax3.set_title("Spectral-norm matrix error")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)

    if args.assert_rate and not rate_ok:
        raise SystemExit(
            "Rate assertion failed: "
            f"slope_mean={slope_err:.6f}, slope_scaled={slope_scaled:.6f}."
        )


if __name__ == "__main__":
    main()
