"""Example 2: data-driven Hautus tests for controllability/stabilizability.

Implements Theorem 5.1 (state) and 5.3 (I/O) of
paper_informativity/sections/controllability.tex. We generate data from
the controllable coupled-spring system and from an uncontrollable
two-spring system, and verify the tests distinguish them.
"""

import _common  # noqa: F401
import numpy as np
import torch

from src import (
    FourierSineBasis, LinearSDE, compute_state_lifts, compute_io_lifts,
    coupled_spring, io_hautus_test, make_multisine_control, simulate,
    state_hautus_test,
)


def _simulate_trajectory(A, B, C, D, T, dt, *, seed, n_harmonics=10):
    n = A.shape[0]
    m = B.shape[1]
    ctrl = make_multisine_control(m, n_harmonics=n_harmonics, T=T, seed=seed)
    sde = LinearSDE(A, B, C, D, control_fn=ctrl)
    return simulate(sde, T=T, dt=dt, seed=seed)


def _uncontrollable_two_spring():
    """Two masses with the same spring constant but no coupling: block-diagonal."""
    n = 4; m = 1; p = 2
    A = torch.tensor(
        [[0, 1, 0, 0],
         [-1, -0.1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, -1, -0.1]], dtype=torch.float64,
    )
    B = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float64)
    C = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float64)
    D = torch.zeros(p, m, dtype=torch.float64)
    return A, B, C, D


def _grid_lambdas():
    """Fallback grid when Lemma 5.2's finite candidate set is empty."""
    reals = np.linspace(-2.0, 2.0, 11)
    imags = np.linspace(-2.0, 2.0, 11)
    grid = [complex(r, i) for r in reals for i in imags]
    return torch.tensor(grid, dtype=torch.complex128)


def _run(label, A, B, C, D, T=15.0, dt=0.01):
    print(f"\n=== {label} ===")
    traj = _simulate_trajectory(A, B, C, D, T, dt, seed=3)
    basis = FourierSineBasis(N=60, T=T)
    grid = _grid_lambdas()

    # state-based test
    lifts = compute_state_lifts(basis=basis, ts=traj.ts, u=traj.u, x=traj.x, L=1, K=1)
    hs = state_hautus_test(
        basis=basis, ts=traj.ts, x=traj.x, lifts=lifts, grid_lambdas=grid,
    )
    print(f"[state] checked {hs.candidate_lambdas.numel()} candidate(s)")
    print(f"[state] controllable={hs.is_controllable}, stabilizable={hs.is_stabilizable}")

    # I/O-based test
    n = A.shape[0]
    lifts_io = compute_io_lifts(basis=basis, ts=traj.ts, u=traj.u, y=traj.y, n=n, L=2, K=2)
    hi = io_hautus_test(
        basis=basis, ts=traj.ts, u=traj.u, y=traj.y,
        L=2, K=2, n=n, m=B.shape[1], lifts=lifts_io, grid_lambdas=grid,
    )
    print(f"[io]    checked {hi.candidate_lambdas.numel()} candidate(s)")
    print(f"[io]    controllable={hi.is_controllable}, stabilizable={hi.is_stabilizable}")


def main():
    _run("Coupled spring (controllable)", *coupled_spring())
    _run("Two independent springs (NOT controllable)", *_uncontrollable_two_spring())


if __name__ == "__main__":
    main()
