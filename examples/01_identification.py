"""Example 1: recover (A, B) from noisy state data.

Implements Theorem 4.1 of paper_informativity/sections/identification.tex:
    im Z_1^{x,u} = R^{n+m}  =>  Sigma_{D_x} = {(A*, B*)},
    [A* B*] = <Dx, Phi> R_x.

We generate a trajectory of a stable random LTI system with additive
Brownian process noise, form the weak lifts against a Fourier-sine
basis, and recover (A, B) as the singular value N -> infty.
"""

import _common  # noqa: F401
import numpy as np
import torch
import matplotlib.pyplot as plt

from src import (
    FourierSineBasis, LinearSDE, check_identifiability, check_persistent_excitation,
    compute_state_lifts, identify_state_data, make_multisine_control, random_stable,
    simulate,
)


def main():
    torch.manual_seed(0)
    A, B, C, D = random_stable(n=5, m=2, p=3, seed=7, stability_margin=0.5)
    n, m = A.shape[0], B.shape[1]

    T = 20.0
    dt = 0.005
    ctrl = make_multisine_control(m, n_harmonics=10, T=T, seed=1)
    G = 0.0 * torch.eye(n)  # zero-noise: torchsde still runs (see sde.py docstring)
    sde = LinearSDE(A, B, C, D, G=G, control_fn=ctrl)
    traj = simulate(sde, T=T, dt=dt, seed=2)

    # PE test
    basis = FourierSineBasis(N=80, T=T)
    pe = check_persistent_excitation(basis, traj.ts, traj.u, L=n + 1)
    print(f"PE of order {pe.L}: rank={pe.rank}/{pe.expected_rank}, "
          f"min_sv={pe.min_singular_value:.3e}, is_pe={pe.is_pe}")

    # Identification
    lifts = compute_state_lifts(basis=basis, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)
    surj, rank_Z, _ = check_identifiability(lifts)
    print(f"im Z surjective? {surj} (rank {rank_Z} / {lifts.n + lifts.m})")
    ident = identify_state_data(lifts)
    print(f"||A_hat - A|| = {(ident.A - A).norm().item():.3e}")
    print(f"||B_hat - B|| = {(ident.B - B).norm().item():.3e}")

    # Plot identification error vs. basis size
    Ns = [10, 20, 40, 80, 160]
    errs_A, errs_B = [], []
    for N in Ns:
        b = FourierSineBasis(N=N, T=T)
        li = compute_state_lifts(basis=b, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)
        r = identify_state_data(li)
        errs_A.append((r.A - A).norm().item())
        errs_B.append((r.B - B).norm().item())

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(Ns, errs_A, "o-", label=r"$\|A-\hat A\|_F$")
    ax.loglog(Ns, errs_B, "s-", label=r"$\|B-\hat B\|_F$")
    ax.set_xlabel("basis size $N$")
    ax.set_ylabel("identification error")
    ax.set_title("Identification error vs. test-function basis size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = f"{_common.FIG_DIR}/01_identification.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
