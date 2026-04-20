"""Example 3: data-driven state-feedback stabilization.

Exact case: Theorem 6.1 / Remark 6.2.
Noisy case: Theorem 6.3 (quadratic stabilization via the induced QMI).

We use the coupled-spring system with optional additive process noise.
"""

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import (
    FourierSineBasis, LinearSDE, ball_Pi, compute_state_lifts,
    coupled_spring, make_multisine_control, simulate, stabilize_exact,
    stabilize_noisy,
)


def _close_loop(A, B, K, x0, T, dt):
    """Small forward-Euler simulation of dx = (A + BK) x, x(0)=x0."""
    Acl = A + B @ K
    N = int(round(T / dt)) + 1
    ts = torch.linspace(0.0, T, N, dtype=A.dtype)
    xs = torch.zeros(N, A.shape[0], dtype=A.dtype)
    xs[0] = x0
    for k in range(N - 1):
        xs[k + 1] = xs[k] + dt * xs[k] @ Acl.T
    return ts, xs


def main():
    torch.manual_seed(0)
    A, B, C, D = coupled_spring()
    n, m = A.shape[0], B.shape[1]

    T = 15.0
    dt = 0.01
    ctrl = make_multisine_control(m, n_harmonics=10, T=T, seed=1)
    G = 0.02 * torch.eye(n)  # small process noise
    sde = LinearSDE(A, B, C, D, G=G, control_fn=ctrl)
    traj = simulate(sde, T=T, dt=dt, seed=2)

    basis = FourierSineBasis(N=60, T=T)
    lifts = compute_state_lifts(basis=basis, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)

    # Exact LMI (Remark 6.2)
    stab = stabilize_exact(lifts)
    print(f"[exact] success={stab.success}")
    Acl = A + B @ stab.K
    eigs = np.linalg.eigvals(Acl.numpy())
    print(f"[exact] closed-loop eigvals: {eigs.round(3)}")
    print(f"[exact] max Re(eig): {eigs.real.max():.4f}")

    # Noisy LMI (Theorem 6.3) with a spherical noise bound
    N_basis = len(basis)
    Pi = ball_Pi(q=n, N=N_basis, noise_bound=0.5)
    stab_n = stabilize_noisy(lifts=lifts, Pi=Pi)
    print(f"\n[noisy] success={stab_n.success}")
    Acl_n = A + B @ stab_n.K
    eigs_n = np.linalg.eigvals(Acl_n.numpy())
    print(f"[noisy] closed-loop eigvals: {eigs_n.round(3)}")
    print(f"[noisy] max Re(eig): {eigs_n.real.max():.4f}")

    # Visualize the closed-loop response under both gains.
    x0 = torch.tensor([1.0, 0.0, -0.5, 0.0], dtype=torch.float64)
    ts_e, xs_e = _close_loop(A, B, stab.K, x0, T=8.0, dt=0.01)
    ts_n, xs_n = _close_loop(A, B, stab_n.K, x0, T=8.0, dt=0.01)

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for i in range(n):
        ax[0].plot(ts_e, xs_e[:, i], label=f"$x_{i+1}$")
        ax[1].plot(ts_n, xs_n[:, i], label=f"$x_{i+1}$")
    ax[0].set_title("Exact LMI (Remark 6.2)")
    ax[1].set_title("Noisy LMI (Theorem 6.3)")
    for a in ax:
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
        a.legend(ncol=2, fontsize=8)
    ax[0].set_ylabel("state")
    fig.suptitle("Closed-loop response under data-driven state feedback")
    fig.tight_layout()
    path = f"{_common.FIG_DIR}/03_stabilization.png"
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
