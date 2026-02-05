import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gramians import compute_K_LK


class TestComputeKLK(unittest.TestCase):
    def test_expected_rank_shapes(self):
        torch.manual_seed(0)
        dtype = torch.float64

        N = 120
        m, p = 2, 1
        L, K = 2, 2
        dt = 0.1

        u = torch.randn(N, m, dtype=dtype)
        y = torch.randn(N, p, dtype=dtype)

        d = L * m + K * p
        r = d - 1

        K_matrix, eigs = compute_K_LK(u, y, L, K, lam=0.0, dt=dt, expected_rank=r)
        self.assertEqual(tuple(K_matrix.shape), (r, r))
        self.assertEqual(tuple(eigs.shape), (r,))

    def test_eigenvalues_invariant_full_rank_basis(self):
        torch.manual_seed(1)
        dtype = torch.float64

        N = 200
        m, p = 2, 2
        L, K = 2, 2
        dt = 0.05

        u = torch.randn(N, m, dtype=dtype)
        y = torch.randn(N, p, dtype=dtype)

        d = L * m + K * p

        _, eig_svd = compute_K_LK(
            u, y, L, K, lam=0.0, dt=dt, expected_rank=d, basis_method="svd"
        )
        _, eig_qr = compute_K_LK(
            u, y, L, K, lam=0.0, dt=dt, expected_rank=d, basis_method="qr"
        )

        # Eigenvalues should match as sets (ordering can vary for conjugate pairs).
        eig_svd_np = eig_svd.detach().cpu().numpy().tolist()
        eig_qr_np = eig_qr.detach().cpu().numpy().tolist()

        remaining = list(eig_qr_np)
        max_abs_diff = 0.0
        for z in eig_svd_np:
            j = min(range(len(remaining)), key=lambda k: abs(remaining[k] - z))
            max_abs_diff = max(max_abs_diff, abs(remaining[j] - z))
            remaining.pop(j)

        self.assertLessEqual(max_abs_diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
