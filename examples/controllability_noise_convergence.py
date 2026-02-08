"""Compare full vs reduced controllability margins with/without noise.

This example:
1. Simulates two trajectories up to T=1000s: noiseless and noisy.
2. Runs both finite-candidate tests:
   - full: rank test on G_{L,K}(lambda)
   - reduced: rank test on H_{L,K}(lambda)
3. Uses the no-noise final-horizon test result as a reference "true" margin.
4. For each horizon, estimates the minimum test margin using the same
   reference candidate sets, with and without noise.
5. Plots estimated margins vs horizon and compares them to the true reference.

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
    smooth_signal,
)
from src.systems import SYSTEMS, get_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convergence of estimated minimum eigenvalues of G and H (noise vs no-noise)."
    )
    parser.add_argument(
        "--system",
        type=str,
        default="random",
        choices=list(SYSTEMS.keys()),
    )
    parser.add_argument("--T", type=float, default=1000.0, help="Final horizon (seconds).")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step.")
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
    parser.add_argument(
        "--smooth-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply smoothing to y before derivative-based checks (default: enabled).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Odd smoothing window size.",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=10.0,
        help="Gaussian sigma when smoothing mode is gaussian.",
    )
    parser.add_argument(
        "--smoothing-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "moving_average"],
        help="Smoothing kernel type.",
    )
    parser.add_argument(
        "--numerical-threshold",
        type=float,
        default=1e-8,
        help="Numerical rank/eigenvalue threshold for controllability checks.",
    )
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
    # Keep all randomness in this case tied to `seed` so runs are reproducible
    # and comparable across examples.
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
    threshold: float = 1e-8,
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
        # Use the same seed path as `controllability_reduced.py` so
        # final-horizon noisy margins are directly comparable.
        seed=args.seed,
    )

    if args.smooth_y:
        y_clean = smooth_signal(
            y_clean,
            window_size=args.smoothing_window,
            sigma=args.smoothing_sigma,
            mode=args.smoothing_mode,
        )
        y_noisy = smooth_signal(
            y_noisy,
            window_size=args.smoothing_window,
            sigma=args.smoothing_sigma,
            mode=args.smoothing_mode,
        )
        print(
            "Applied "
            f"{args.smoothing_mode} smoothing to y "
            f"(window={args.smoothing_window}, sigma={args.smoothing_sigma})."
        )

    # Reference candidate sets and "true" margins:
    # use the no-noise, final-horizon test result as baseline.
    full_res_ref = check_controllability(
        u_clean,
        y_clean,
        L_full,
        K_full,
        n,
        m,
        args.T,
        args.dt,
        threshold=args.numerical_threshold,
    )
    red_res_ref = check_controllability_reduced(
        y_clean,
        L_red,
        K_red,
        args.T,
        args.dt,
        threshold=args.numerical_threshold,
    )
    true_margin_G = float(full_res_ref["min_eigenvalues"].min().item())
    true_margin_H = float(red_res_ref["min_eigenvalues"].min().item())
    full_res_noisy_ref = check_controllability(
        u_noisy,
        y_noisy,
        L_full,
        K_full,
        n,
        m,
        args.T,
        args.dt,
        threshold=args.numerical_threshold,
    )
    noisy_margin_G = float(full_res_noisy_ref["min_eigenvalues"].min().item())
    red_res_noisy_ref = check_controllability_reduced(
        y_noisy,
        L_red,
        K_red,
        args.T,
        args.dt,
        threshold=args.numerical_threshold,
    )
    noisy_margin_H = float(red_res_noisy_ref["min_eigenvalues"].min().item())
    print(
        f"Reference margins at T={args.T:.1f}: "
        f"G_clean={true_margin_G:.6e}, G_noisy={noisy_margin_G:.6e}, "
        f"H_clean={true_margin_H:.6e}, H_noisy={noisy_margin_H:.6e}"
    )

    est_clean = estimate_min_eig_paths(
        u_clean,
        y_clean,
        horizons,
        n,
        m,
        L_full,
        K_full,
        L_red,
        K_red,
        args.dt,
        threshold=args.numerical_threshold,
    )
    est_noisy = estimate_min_eig_paths(
        u_noisy,
        y_noisy,
        horizons,
        n,
        m,
        L_full,
        K_full,
        L_red,
        K_red,
        args.dt,
        threshold=args.numerical_threshold,
    )

    print("\nConvergence table (estimated minimum margins vs clean reference):")
    print(
        "  T      G_clean         G_noisy         |G_clean-clean_ref|  "
        "|G_noisy-clean_ref|  H_clean         H_noisy         "
        "|H_clean-clean_ref|  |H_noisy-clean_ref|"
    )
    for i, Ti in enumerate(horizons):
        g_clean = est_clean["G"][i]
        g_noisy = est_noisy["G"][i]
        h_clean = est_clean["H"][i]
        h_noisy = est_noisy["H"][i]
        print(
            f"  {Ti:6.1f}  {g_clean:13.6e}  {g_noisy:13.6e}  "
            f"{abs(g_clean - true_margin_G):13.6e}  {abs(g_noisy - true_margin_G):13.6e}  "
            f"{h_clean:13.6e}  {h_noisy:13.6e}  "
            f"{abs(h_clean - true_margin_H):13.6e}  {abs(h_noisy - true_margin_H):13.6e}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    plot_cfg = [
        (
            axes[0],
            "Full test margin (G)",
            est_clean["G"],
            est_noisy["G"],
            true_margin_G,
        ),
        (
            axes[1],
            "Reduced test margin (H)",
            est_clean["H"],
            est_noisy["H"],
            true_margin_H,
        ),
    ]
    for ax, title, vals_clean, vals_noisy, true_val in plot_cfg:
        valid_clean = np.isfinite(vals_clean) & np.isfinite(horizons) & (horizons > 0)
        valid_noisy = np.isfinite(vals_noisy) & np.isfinite(horizons) & (horizons > 0)
        if valid_clean.any():
            ax.loglog(
                horizons[valid_clean],
                vals_clean[valid_clean],
                "o-",
                linewidth=2,
                markersize=5,
                label="no-noise",
            )
            ax.loglog(
                horizons[valid_noisy],
                vals_noisy[valid_noisy],
                "s--",
                linewidth=2,
                markersize=5,
                label="noisy",
            )
        ax.axhline(
            true_val,
            color="k",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label="clean reference",
        )
        ax.set_xlabel("Horizon T")
        ax.set_ylabel("Minimum margin")
        ax.set_title(title)
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
