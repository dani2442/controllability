"""Example 4: data-driven continuous-time LQR (Theorem 7.1).

Given state data informative for identification, we solve the CARE on the
identified plant. We compare the data-driven gain to the model-based gain
(same CARE solved with the true (A, B)) and plot the closed-loop response.
"""

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from src import (
    FourierSineBasis, LinearSDE, compute_state_lifts,
    coupled_spring, make_multisine_control, simulate, solve_lqr_exact,
)


def _close_loop(A, B, K, x0, T, dt):
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

    T = 20.0
    dt = 0.005
    ctrl = make_multisine_control(m, n_harmonics=12, T=T, seed=1)
    G = 0.0 * torch.eye(n)
    sde = LinearSDE(A, B, C, D, G=G, control_fn=ctrl)
    traj = simulate(sde, T=T, dt=dt, seed=2)

    basis = FourierSineBasis(N=80, T=T)
    lifts = compute_state_lifts(basis=basis, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)

    Q = torch.eye(n)
    R = 0.1 * torch.eye(m)

    res = solve_lqr_exact(lifts, Q=Q, R=R)
    print(f"informative for LQR: {res.informative}")
    print(f"||A_hat - A|| = {(res.A0 - A).norm().item():.3e}")
    print(f"||B_hat - B|| = {(res.B0 - B).norm().item():.3e}")

    # Compare to model-based oracle gain.
    P_true = solve_continuous_are(A.numpy(), B.numpy(), Q.numpy(), R.numpy())
    K_true = -np.linalg.solve(R.numpy(), B.numpy().T @ P_true)
    print(f"||K_data - K_true||_F = {np.linalg.norm(res.K.numpy() - K_true):.3e}")

    Acl_data = A + B @ res.K
    Acl_true = A + B @ torch.tensor(K_true, dtype=A.dtype)
    print(f"max Re(eig) data-driven gain = {np.linalg.eigvals(Acl_data.numpy()).real.max():.4f}")
    print(f"max Re(eig) model-based gain = {np.linalg.eigvals(Acl_true.numpy()).real.max():.4f}")

    x0 = torch.tensor([1.0, 0.0, -0.5, 0.0], dtype=torch.float64)
    ts_d, xs_d = _close_loop(A, B, res.K, x0, T=10.0, dt=0.01)
    ts_m, xs_m = _close_loop(A, B, torch.tensor(K_true, dtype=A.dtype), x0, T=10.0, dt=0.01)

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for i in range(n):
        ax[0].plot(ts_d, xs_d[:, i], label=f"$x_{i+1}$")
        ax[1].plot(ts_m, xs_m[:, i], label=f"$x_{i+1}$")
    ax[0].set_title("Data-driven LQR (Theorem 7.1)")
    ax[1].set_title("Model-based LQR (oracle)")
    for a in ax:
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
        a.legend(ncol=2, fontsize=8)
    ax[0].set_ylabel("state")
    fig.suptitle("LQR closed-loop: data-driven vs. oracle gain")
    fig.tight_layout()
    path = f"{_common.FIG_DIR}/04_lqr.png"
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
