"""Demo: exactly-unbiased Pi estimator under i.i.d. Gaussian samples.

This script validates the split-sample/Wishart-correction estimator implemented in:
    - src.gramians.compute_unbiased_Q_wishart
    - src.gramians.compute_unbiased_Pi_wishart

Model:
    xi_i ~ CN(0, Sigma_xi)
    eta_i = Q xi_i + e_i,   E[e_i | xi_i] = 0

Then Pi = Q Q* and the estimator Pi_hat_unb is unbiased (Hermitian, not PSD).

Usage:
    python examples/pi_unbiased_iid_demo.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compute_unbiased_Pi_wishart  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unbiased Pi demo (i.i.d. Gaussian samples).")
    p.add_argument("--N", type=int, default=2000, help="Number of i.i.d. samples per trial.")
    p.add_argument("--mc", type=int, default=200, help="Monte Carlo trials.")
    p.add_argument("--r", type=int, default=6, help="Regressor dimension.")
    p.add_argument("--d", type=int, default=4, help="Response dimension.")
    p.add_argument("--noise-scale", type=float, default=0.5, help="Additive eta noise scale.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sample_proper_complex_gaussian(N: int, cov: torch.Tensor) -> torch.Tensor:
    """Samples CN(0, cov) as rows (N, r)."""
    r = cov.shape[0]
    L = torch.linalg.cholesky(cov)
    real_dtype = torch.float64 if cov.dtype in (torch.complex128, torch.float64) else torch.float32
    a = torch.randn(N, r, device=cov.device, dtype=real_dtype)
    b = torch.randn(N, r, device=cov.device, dtype=real_dtype)
    z = (a + 1j * b).to(dtype=cov.dtype) / np.sqrt(2.0)
    # Row-stacked convention: if xi_i = L z_i (column), then xi_i^T = z_i^T L^T.
    return z @ L.T


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.complex128

    r = args.r
    d = args.d

    # Random well-conditioned covariance.
    G = torch.randn(r, r, device=device, dtype=torch.float64)
    Sigma_xi = (G.T @ G) + 0.5 * torch.eye(r, device=device, dtype=torch.float64)
    Sigma_xi = Sigma_xi.to(dtype)

    # Random Q and Pi.
    Q = (torch.randn(d, r, device=device, dtype=torch.float64) + 1j * torch.randn(d, r, device=device, dtype=torch.float64)) / np.sqrt(2.0)
    Q = Q.to(dtype)
    Pi_true = Q @ Q.conj().T
    Pi_true = 0.5 * (Pi_true + Pi_true.conj().T)

    errs: list[float] = []
    min_eigs: list[float] = []
    for t in range(args.mc):
        xi = sample_proper_complex_gaussian(args.N, Sigma_xi)
        a = torch.randn(args.N, d, device=device, dtype=torch.float64)
        b = torch.randn(args.N, d, device=device, dtype=torch.float64)
        e = args.noise_scale * (a + 1j * b).to(dtype=dtype) / np.sqrt(2.0)
        eta = xi @ Q.T + e  # rows: eta_i^T = xi_i^T Q^T + e_i^T

        Pi_hat = compute_unbiased_Pi_wishart(xi=xi, eta=eta, assume="complex", split=0.25)
        err = float(torch.linalg.matrix_norm(Pi_hat - Pi_true, ord=2).real.item())
        errs.append(err)
        min_eigs.append(float(torch.linalg.eigvalsh(Pi_hat).real.min().item()))

    mean_err = float(np.mean(np.array(errs)))
    std_err = float(np.std(np.array(errs)))
    mean_min_eig = float(np.mean(np.array(min_eigs)))
    true_min_eig = float(torch.linalg.eigvalsh(Pi_true).real.min().item())

    print(f"N={args.N}, mc={args.mc}, r={r}, d={d}")
    print(f"True  lambda_min(Pi) = {true_min_eig:.6g}")
    print(f"Mean  lambda_min(Pi_hat_unb) = {mean_min_eig:.6g}  (not unbiased)")
    print(f"Mean ||Pi_hat_unb - Pi||_2 = {mean_err:.3e} ± {std_err:.3e}")


if __name__ == "__main__":
    main()
