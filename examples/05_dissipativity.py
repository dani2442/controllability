"""Example 5: data-driven dissipativity verification (Section 8).

Exact case: Theorem 8.1. Noisy case: Theorem 8.3.

We verify two supply rates on the damped oscillator:
    - L^2 gain <= gamma  (S = diag(gamma^2 I_m, -I_p))
    - passivity         (S = [[0, I/2]; [I/2, 0]])
"""

import _common  # noqa: F401
import numpy as np
import torch

from src import (
    FourierSineBasis, LinearSDE, ball_Pi, compute_io_lifts,
    damped_oscillator, dissipativity_exact, dissipativity_noisy,
    make_multisine_control, simulate,
)


def _l2_gain_supply(gamma: float, m: int, p: int) -> torch.Tensor:
    S = torch.zeros(m + p, m + p, dtype=torch.float64)
    S[:m, :m] = gamma * gamma * torch.eye(m)
    S[m:, m:] = -torch.eye(p)
    return S


def _passivity_supply(m: int, p: int) -> torch.Tensor:
    if m != p:
        raise ValueError("Passivity requires m == p.")
    S = torch.zeros(m + p, m + p, dtype=torch.float64)
    S[:m, m:] = 0.5 * torch.eye(m)
    S[m:, :m] = 0.5 * torch.eye(m)
    return S


def main():
    torch.manual_seed(0)
    A, B, C, D = damped_oscillator(omega=1.5, zeta=0.1)
    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    T = 20.0
    dt = 0.005
    ctrl = make_multisine_control(m, n_harmonics=12, T=T, seed=1)
    G = 0.0 * torch.eye(n)
    sde = LinearSDE(A, B, C, D, G=G, control_fn=ctrl)
    traj = simulate(sde, T=T, dt=dt, seed=2)

    basis = FourierSineBasis(N=80, T=T)
    lifts = compute_io_lifts(
        basis=basis, ts=traj.ts, u=traj.u, y=traj.y, x=traj.x,
        n=n, L=1, K=1,
    )
    y_lift_0 = lifts.Y[:p]  # <y, Phi>  (first block of Y_K)

    # --- L^2 gain tests ---
    for gamma in (1.0, 2.5, 10.0):
        S = _l2_gain_supply(gamma, m, p)
        try:
            res = dissipativity_exact(lifts, y_lift_0, S)
            tag = "OK" if res.success else "infeasible"
            print(f"[exact] L^2 gain <= {gamma:>4}: {tag}")
        except Exception as e:
            print(f"[exact] L^2 gain <= {gamma}: error ({e})")

    # --- Passivity ---
    S_pass = _passivity_supply(min(m, p), min(m, p))
    if m == p:
        try:
            res = dissipativity_exact(lifts, y_lift_0, S_pass)
            tag = "OK" if res.success else "infeasible"
            print(f"[exact] passivity: {tag}")
        except Exception as e:
            print(f"[exact] passivity: error ({e})")
    else:
        print(f"[exact] passivity skipped (m={m} != p={p})")

    # --- Noisy case: Theorem 8.3 ---
    print()
    N_basis = len(basis)
    Pi = ball_Pi(q=n + p, N=N_basis, noise_bound=0.2)
    for gamma in (2.5, 10.0):
        S = _l2_gain_supply(gamma, m, p)
        try:
            res = dissipativity_noisy(lifts=lifts, y_lift_0=y_lift_0, S=S, Pi=Pi)
            tag = "OK" if res.success else "infeasible"
            print(f"[noisy] L^2 gain <= {gamma:>4}: {tag}")
        except Exception as e:
            print(f"[noisy] L^2 gain <= {gamma}: error ({e})")


if __name__ == "__main__":
    main()
