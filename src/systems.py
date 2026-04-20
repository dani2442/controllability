"""A small library of linear time-invariant test systems.

Each factory returns ``(A, B, C, D)`` as torch tensors. Used by the examples
and the notebooks.
"""

from __future__ import annotations

from typing import Tuple

import torch


System = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def double_integrator(*, device: torch.device = torch.device("cpu"),
                      dtype: torch.dtype = torch.float64) -> System:
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device, dtype=dtype)
    B = torch.tensor([[0.0], [1.0]], device=device, dtype=dtype)
    C = torch.eye(2, device=device, dtype=dtype)
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    return A, B, C, D


def damped_oscillator(*, omega: float = 1.5, zeta: float = 0.1,
                      device: torch.device = torch.device("cpu"),
                      dtype: torch.dtype = torch.float64) -> System:
    A = torch.tensor(
        [[0.0, 1.0], [-omega * omega, -2.0 * zeta * omega]],
        device=device, dtype=dtype,
    )
    B = torch.tensor([[0.0], [1.0]], device=device, dtype=dtype)
    C = torch.eye(2, device=device, dtype=dtype)
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    return A, B, C, D


def coupled_spring(
    *,
    m1: float = 1.0, m2: float = 1.0,
    k1: float = 1.0, k2: float = 1.0,
    b1: float = 0.1, b2: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> System:
    """Controllable 4-state coupled spring-damper with scalar input."""
    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(k1 + k2) / m1, -b1 / m1, k2 / m1, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [k2 / m2, 0.0, -k2 / m2, -b2 / m2],
        ],
        device=device, dtype=dtype,
    )
    B = torch.tensor([[0.0], [1.0 / m1], [0.0], [0.0]], device=device, dtype=dtype)
    C = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        device=device, dtype=dtype,
    )
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    return A, B, C, D


def random_stable(
    n: int = 4, m: int = 2, p: int = 3,
    *,
    stability_margin: float = 0.2,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> System:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    A = torch.randn(n, n, generator=gen, device=device, dtype=dtype)
    eigvals = torch.linalg.eigvals(A)
    shift = eigvals.real.max().item() + stability_margin
    A = A - shift * torch.eye(n, device=device, dtype=dtype)
    B = torch.randn(n, m, generator=gen, device=device, dtype=dtype)
    C = torch.randn(p, n, generator=gen, device=device, dtype=dtype)
    D = 0.1 * torch.randn(p, m, generator=gen, device=device, dtype=dtype)
    return A, B, C, D


def unstable_inverted_pendulum(*, device: torch.device = torch.device("cpu"),
                               dtype: torch.dtype = torch.float64) -> System:
    """Inverted pendulum on a cart, linearized around the upright equilibrium.

    Unstable, controllable, observable with two measured outputs.
    """
    M, m, l, b, g = 1.0, 0.1, 1.0, 0.1, 9.81
    denom = M + m
    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -b / denom, m * g / denom, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -b / (l * denom), denom * g / (l * denom), 0.0],
        ],
        device=device, dtype=dtype,
    )
    B = torch.tensor(
        [[0.0], [1.0 / denom], [0.0], [1.0 / (l * denom)]],
        device=device, dtype=dtype,
    )
    C = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        device=device, dtype=dtype,
    )
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    return A, B, C, D


__all__ = [
    "System",
    "double_integrator",
    "damped_oscillator",
    "coupled_spring",
    "random_stable",
    "unstable_inverted_pendulum",
]
