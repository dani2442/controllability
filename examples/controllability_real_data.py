"""Controllability analysis from real experimental data.

This script demonstrates the data-driven Hautus controllability test
applied to *real* input-output measurements — no known system matrices
(A, B, C, D) required.

Dataset: BAB multisine_01 (05_multisine_01.mat) from
  https://github.com/helonayala/sysid

Pipeline:
    1. Load and preprocess real (u, y) data via the Dataset class.
    2. Convert numpy signals to torch tensors.
    3. Sweep over assumed state dimensions n ∈ {2, 3, 4, 5}:
       a. Compute candidate eigenvalue set via K_{L,K}(u, y).
       b. Full behavioral test: check_controllability (uses u and y).
       c. Reduced (output-only) test: check_controllability_reduced (uses y only).
    4. Print summary table and save eigenvalue-decay plots.

Usage:
    python examples/controllability_real_data.py
    python examples/controllability_real_data.py --dataset multisine_05
    python examples/controllability_real_data.py --n 3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    compute_G_LK,
    compute_K_LK,
    compute_K_LK_reduced,
    check_controllability,
    check_controllability_reduced,
    check_persistent_excitation,
    smooth_signal,
    plot_gramian_eigenvalues,
)
from src.datasets import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data-driven controllability analysis on real data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multisine_05",
        help="BAB experiment key or alias (default: multisine_05 = 05_multisine_01.mat)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Assumed state dimension(s) to sweep (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--resample-factor",
        type=int,
        default=50,
        help="Downsample factor during preprocessing (default: 50)",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Gaussian smoothing to y before analysis",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Smoothing window size (default: 11)",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (default: 2.0)",
    )
    parser.add_argument(
        "--detrend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mean from u and y (default: True)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Eigenvalue threshold for rank computation (default: 1e-6)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Max candidate eigenvalues to test (default: 20)",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show matplotlib plots interactively",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def numpy_to_torch(
    arr: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a 1-D numpy array to a (N, 1) torch tensor (SISO convention)."""
    arr = np.asarray(arr, dtype=np.float64).flatten()
    return torch.from_numpy(arr).unsqueeze(-1).to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Device: {device}")

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"LOADING DATASET: {args.dataset}")
    print("=" * 60)

    ds = Dataset.from_bab_experiment(
        args.dataset,
        preprocess=True,
        resample_factor=args.resample_factor,
    )
    print(f"  {ds}")
    print(f"  Samples after preprocessing: {len(ds)}")
    print(f"  Sampling rate: {ds.sampling_rate:.2f} Hz")

    # Optional detrending
    if args.detrend:
        ds = ds.preprocess(detrend=True)
        print("  Applied detrending (zero-mean u, y)")

    # ── 2. Convert to torch ──────────────────────────────────────────────────
    u = numpy_to_torch(ds.u, device, dtype)  # (N, 1)
    y = numpy_to_torch(ds.y, device, dtype)  # (N, 1)

    # Apply smoothing
    if args.smooth:
        y = smooth_signal(
            y,
            window_size=args.smoothing_window,
            sigma=args.smoothing_sigma,
            mode="gaussian",
        )
        print(
            f"  Applied Gaussian smoothing to y "
            f"(window={args.smoothing_window}, sigma={args.smoothing_sigma})"
        )

    dt = float(np.median(np.diff(ds.t)))
    T = float(ds.t[-1] - ds.t[0])
    N = u.shape[0]
    m = 1  # SISO input dimension
    p = 1  # SISO output dimension

    print(f"\n  Signal shapes: u={tuple(u.shape)}, y={tuple(y.shape)}")
    print(f"  dt = {dt:.6f} s,  T = {T:.2f} s,  N = {N}")

    # ── 3. Quick data summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  u: mean={ds.u.mean():.4f}, std={ds.u.std():.4f}, "
          f"min={ds.u.min():.4f}, max={ds.u.max():.4f}")
    print(f"  y: mean={ds.y.mean():.4f}, std={ds.y.std():.4f}, "
          f"min={ds.y.min():.4f}, max={ds.y.max():.4f}")

    # ── 4. Sweep over assumed state dimensions ───────────────────────────────
    state_dims = args.n
    summary_rows = []

    for n in state_dims:
        print("\n" + "=" * 60)
        print(f"ASSUMED STATE DIMENSION n = {n}")
        print("=" * 60)

        # Heuristic: set L = K = n (minimal admissible choice when ell ≈ n)
        L = n
        K = n

        print(f"  Derivative lift orders: L={L}, K={K}")
        print(f"  Full-test expected rank: Lm + n = {L * m + n}")
        print(f"  Reduced-test expected rank: Kp = {K * p}")

        # ── 4a. Persistent excitation check ──────────────────────────────────
        pe_order = L + n + 1
        pe_result = check_persistent_excitation(
            u, order=pe_order, dt=dt, threshold=args.threshold,
        )
        print(f"\n  PE check (order {pe_order}):")
        print(f"    rank(Γ) = {pe_result['rank']}/{pe_result['full_dimension']}")
        print(f"    min eigenvalue = {pe_result['min_eigenvalue']:.6e}")
        print(f"    PE satisfied: {pe_result['is_persistently_exciting']}")

        # ── 4b. Full behavioral test (u + y) ─────────────────────────────────
        print(f"\n  FULL BEHAVIORAL TEST (u + y)")

        _, cand_full = compute_K_LK(u, y, L, K, n, lam=0.0, dt=dt)
        n_cand = min(len(cand_full), args.max_candidates)
        cand_full = cand_full[:n_cand]
        print(f"    Candidate eigenvalues: {len(cand_full)}")

        for i, lam in enumerate(cand_full[:5]):
            print(f"      μ_{i+1} = {lam.real:.4f} + {lam.imag:.4f}i")
        if len(cand_full) > 5:
            print(f"      ... ({len(cand_full) - 5} more)")

        result_full = check_controllability(
            u, y, L, K, n, m, T, dt,
            candidate_lambdas=cand_full,
            threshold=args.threshold,
        )
        print(f"    Expected rank: {result_full['expected_rank']}")
        print(f"    Ranks: {result_full['ranks'].cpu().numpy()}")
        print(f"    Min margin: {result_full['min_eigenvalues'].min():.6e}")
        print(f"    IS CONTROLLABLE (full): {result_full['is_controllable']}")

        # ── 4c. Reduced test (y only) ────────────────────────────────────────
        print(f"\n  REDUCED TEST (output-only)")

        L_red = K + 1  # Reduced test requires L >= K + 1
        _, cand_red = compute_K_LK_reduced(y, L_red, K, dt)
        n_cand_red = min(len(cand_red), args.max_candidates)
        cand_red = cand_red[:n_cand_red]
        print(f"    Candidate eigenvalues: {len(cand_red)}")

        result_red = check_controllability_reduced(
            y, L_red, K, T, dt,
            candidate_lambdas=cand_red,
            threshold=args.threshold,
        )
        print(f"    Expected rank: {result_red['expected_rank']}")
        print(f"    rank(S_y) = {result_red['rank_Sy']} "
              f"(full rank: {result_red['full_rank_Sy']})")
        print(f"    Ranks: {result_red['ranks'].cpu().numpy()}")
        print(f"    Min margin: {result_red['min_eigenvalues'].min():.6e}")
        print(f"    IS CONTROLLABLE (reduced): {result_red['is_controllable']}")

        # ── 4d. Eigenvalue decay at a few candidate λ ────────────────────────
        test_lambdas = [0.0]
        if len(cand_full) > 0:
            test_lambdas.append(cand_full[0].item())
        if len(cand_full) > 1:
            test_lambdas.append(cand_full[-1].item())

        eigenvalue_lists = []
        lambda_labels = []
        for lam in test_lambdas:
            G = compute_G_LK(u, y, L, K, lam, T, dt)
            eigvals = torch.linalg.eigvalsh(G).real
            eigvals_sorted = torch.sort(eigvals, descending=True).values
            eigenvalue_lists.append(eigvals_sorted)
            if isinstance(lam, complex):
                label = f"λ = {lam.real:.2f}{lam.imag:+.2f}i"
            else:
                label = f"λ = {lam:.2f}"
            lambda_labels.append(label)

        # Save eigenvalue decay plot
        output_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(output_dir, exist_ok=True)

        expected_rank_full = L * m + n
        fig = plot_gramian_eigenvalues(
            eigenvalue_lists, lambda_labels, expected_rank_full
        )
        fig.suptitle(
            f"Real data ({args.dataset}) — G_{{L,K}}(λ) eigenvalue decay, n={n}",
            fontsize=12,
        )
        fig.savefig(
            os.path.join(output_dir, f"real_data_eigdecay_n{n}.pdf"),
            bbox_inches="tight",
        )
        print(f"\n  Saved: figures/real_data_eigdecay_n{n}.pdf")
        plt.close(fig)

        # Collect summary
        summary_rows.append({
            "n": n,
            "L": L,
            "K": K,
            "PE": pe_result["is_persistently_exciting"],
            "full_controllable": result_full["is_controllable"],
            "full_min_margin": result_full["min_eigenvalues"].min().item(),
            "red_controllable": result_red["is_controllable"],
            "red_full_rank_Sy": result_red["full_rank_Sy"],
            "red_min_margin": result_red["min_eigenvalues"].min().item(),
        })

    # ── 5. Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = (
        f"{'n':>3} {'L':>3} {'K':>3} │ {'PE':>5} │ "
        f"{'Full':>5} {'margin':>12} │ "
        f"{'Red.':>5} {'S_y full':>8} {'margin':>12}"
    )
    print(header)
    print("─" * len(header))
    for row in summary_rows:
        print(
            f"{row['n']:>3} {row['L']:>3} {row['K']:>3} │ "
            f"{str(row['PE']):>5} │ "
            f"{str(row['full_controllable']):>5} {row['full_min_margin']:>12.4e} │ "
            f"{str(row['red_controllable']):>5} {str(row['red_full_rank_Sy']):>8} "
            f"{row['red_min_margin']:>12.4e}"
        )

    # ── 6. Input / output plot ────────────────────────────────────────────────
    fig_data, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t_np = ds.t

    axes[0].plot(t_np, ds.u, linewidth=0.6)
    axes[0].set_ylabel("u (input)")
    axes[0].set_title(f"Real data: {args.dataset}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_np, ds.y, linewidth=0.6)
    axes[1].set_ylabel("y (output)")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig_data.tight_layout()
    fig_data.savefig(
        os.path.join(output_dir, "real_data_signals.pdf"),
        bbox_inches="tight",
    )
    print(f"\n  Saved: figures/real_data_signals.pdf")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
