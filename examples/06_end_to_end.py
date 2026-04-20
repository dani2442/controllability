"""Example 6: end-to-end pipeline on a lightly-damped coupled spring.

From a single noisy trajectory, we run the full informativity pipeline:
    1. Persistent excitation (Definition 2.3).
    2. Data-driven Hautus test (Theorems 5.1 / 5.3).
    3. Identification (Theorem 4.1).
    4. Noisy stabilization (Theorem 6.3).
    5. LQR on the identified plant (Theorem 7.1).
    6. Closed-loop L^2-gain verification via Theorem 8.1.
"""

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import (
    FourierSineBasis, LinearSDE, ball_Pi, check_persistent_excitation,
    compute_io_lifts, compute_state_lifts, coupled_spring,
    dissipativity_exact, identify_state_data, io_hautus_test,
    make_multisine_control, simulate, solve_lqr_exact, stabilize_noisy,
    state_hautus_test,
)


def _l2_gain_supply(gamma, m, p):
    S = torch.zeros(m + p, m + p, dtype=torch.float64)
    S[:m, :m] = gamma * gamma * torch.eye(m)
    S[m:, m:] = -torch.eye(p)
    return S


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
    A, B, C, D = coupled_spring(b1=0.05, b2=0.05)  # lightly damped
    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    T = 20.0
    dt = 0.005
    ctrl = make_multisine_control(m, n_harmonics=15, T=T, seed=1)
    G = 0.02 * torch.eye(n)  # additive process noise
    sde = LinearSDE(A, B, C, D, G=G, control_fn=ctrl)
    traj = simulate(sde, T=T, dt=dt, seed=2)

    # --- 1. PE of the control signal ---
    basis = FourierSineBasis(N=80, T=T)
    pe = check_persistent_excitation(basis, traj.ts, traj.u, L=n + 1)
    print(f"[1] PE order {pe.L}: rank {pe.rank}/{pe.expected_rank}, "
          f"min_sv={pe.min_singular_value:.2e}, is_pe={pe.is_pe}")

    # --- 2. Hautus tests ---
    lifts = compute_state_lifts(basis=basis, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)
    hs = state_hautus_test(basis=basis, ts=traj.ts, x=traj.x, lifts=lifts)
    print(f"[2a] state Hautus: controllable={hs.is_controllable}, "
          f"stabilizable={hs.is_stabilizable}")

    lifts_io = compute_io_lifts(
        basis=basis, ts=traj.ts, u=traj.u, y=traj.y, x=traj.x,
        n=n, L=2, K=2,
    )
    hi = io_hautus_test(
        basis=basis, ts=traj.ts, u=traj.u, y=traj.y,
        L=2, K=2, n=n, m=m, lifts=lifts_io,
    )
    print(f"[2b] I/O   Hautus: controllable={hi.is_controllable}, "
          f"stabilizable={hi.is_stabilizable}")

    # --- 3. Identification ---
    ident = identify_state_data(lifts)
    print(f"[3] ||A_hat - A|| = {(ident.A - A).norm().item():.3e}, "
          f"||B_hat - B|| = {(ident.B - B).norm().item():.3e}")

    # --- 4. Noisy stabilization ---
    Pi = ball_Pi(q=n, N=len(basis), noise_bound=0.5)
    stab = stabilize_noisy(lifts=lifts, Pi=Pi)
    Acl = A + B @ stab.K
    eigs = np.linalg.eigvals(Acl.numpy())
    print(f"[4] noisy stabilization success={stab.success}, "
          f"max Re(eig)={eigs.real.max():.4f}")

    # --- 5. LQR on the identified plant ---
    Q = torch.eye(n)
    R = 0.1 * torch.eye(m)
    lqr = solve_lqr_exact(lifts, Q=Q, R=R)
    Acl_lqr = A + B @ lqr.K
    eigs_lqr = np.linalg.eigvals(Acl_lqr.numpy())
    print(f"[5] LQR informative={lqr.informative}, "
          f"max Re(eig)={eigs_lqr.real.max():.4f}")

    # --- 6. L^2-gain test via Theorem 8.1 (uses L=K=1 I/O lifts) ---
    lifts_io_1 = compute_io_lifts(
        basis=basis, ts=traj.ts, u=traj.u, y=traj.y, x=traj.x,
        n=n, L=1, K=1,
    )
    y_lift_0 = lifts_io_1.Y[:p]
    for gamma in (1.0, 10.0, 50.0):
        S = _l2_gain_supply(gamma, m, p)
        res = dissipativity_exact(lifts_io_1, y_lift_0, S)
        tag = "certified" if res.success else "infeasible"
        print(f"[6] L^2-gain <= {gamma:>5}: {tag}")

    # --- Closed-loop plot ---
    x0 = torch.tensor([1.0, 0.0, -0.5, 0.0], dtype=torch.float64)
    ts_s, xs_s = _close_loop(A, B, stab.K, x0, T=12.0, dt=0.005)
    ts_l, xs_l = _close_loop(A, B, lqr.K, x0, T=12.0, dt=0.005)

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for i in range(n):
        ax[0].plot(ts_s, xs_s[:, i], label=f"$x_{i+1}$")
        ax[1].plot(ts_l, xs_l[:, i], label=f"$x_{i+1}$")
    ax[0].set_title("Noisy stabilization (Thm 6.3)")
    ax[1].set_title("Data-driven LQR (Thm 7.1)")
    for a in ax:
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
        a.legend(ncol=2, fontsize=8)
    ax[0].set_ylabel("state")
    fig.suptitle("End-to-end: stabilization and LQR from one noisy trajectory")
    fig.tight_layout()
    path = f"{_common.FIG_DIR}/06_end_to_end.png"
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
