"""Continuous-time simulator for the paper's data model.

State-space model (matching eq. 8.1 of
`paper_informativity/sections/dissipativity.tex`):

    dx = (A x + B u) dt + G dW_t,        W Brownian d_w-dim
    y  = C x + D u + H xi(t),            xi discrete-time white noise

The simulator supports the noiseless case (G = 0 and/or H = 0) and uses
`torchsde.sdeint` in every case; when the diffusion is zero, the SDE
integrator reduces to a standard ODE and produces a deterministic
trajectory up to the integrator's tolerance. This is what the paper
means by "when the noise vanishes, Pi collapses to the exact-data model".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
import torchsde


ControlFn = Callable[[torch.Tensor], torch.Tensor]


def _as_tensor(
    M: torch.Tensor | None,
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    fill: float = 0.0,
) -> torch.Tensor:
    if M is None:
        return torch.full(shape, fill, device=device, dtype=dtype)
    return M.to(device=device, dtype=dtype)


class LinearSDE(torchsde.SDEIto):
    """Constant-coefficient linear SDE with additive process noise.

    Args:
        A, B, C, D: state-space matrices.
        G:  (n, d_w) process-noise coupling (defaults to zeros -> deterministic).
        H:  (p, d_m) measurement-noise coupling (defaults to zeros).
        control_fn: callable t -> u(t) with signature
                    ``u(t: scalar Tensor) -> (m,) Tensor``.
    """

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        *,
        G: Optional[torch.Tensor] = None,
        H: Optional[torch.Tensor] = None,
        control_fn: Optional[ControlFn] = None,
    ):
        super().__init__(noise_type="additive")
        device, dtype = A.device, A.dtype
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]

        # Process noise: default to zero -> use a 1-dim dummy Brownian that is
        # multiplied by a zero column, so torchsde still has something to sample.
        if G is None:
            self.d_w = 1
            self.G = torch.zeros(self.n, 1, device=device, dtype=dtype)
            self.process_noise_is_zero = True
        else:
            self.d_w = G.shape[1]
            self.G = _as_tensor(G, (self.n, self.d_w), device=device, dtype=dtype)
            self.process_noise_is_zero = bool(torch.allclose(self.G, torch.zeros_like(self.G)))

        if H is None:
            self.d_m = 1
            self.H = torch.zeros(self.p, 1, device=device, dtype=dtype)
            self.measurement_noise_is_zero = True
        else:
            self.d_m = H.shape[1]
            self.H = _as_tensor(H, (self.p, self.d_m), device=device, dtype=dtype)
            self.measurement_noise_is_zero = bool(torch.allclose(self.H, torch.zeros_like(self.H)))

        self._control_fn = control_fn

    # ---- control ----
    def u(self, t: torch.Tensor) -> torch.Tensor:
        if self._control_fn is None:
            return torch.zeros(self.m, device=self.A.device, dtype=self.A.dtype)
        return self._control_fn(t)

    # ---- SDEIto interface ----
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift: (A x + B u) per batch row."""
        ut = self.u(t).to(x.dtype)
        if ut.ndim == 1:
            ut = ut.unsqueeze(0).expand(x.shape[0], -1)
        return x @ self.A.T + ut @ self.B.T

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion G (constant, additive). Returns (batch, n, d_w)."""
        batch = x.shape[0]
        return self.G.unsqueeze(0).expand(batch, -1, -1).to(x.dtype)

    # ---- output ----
    def observe(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        add_noise: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        y = x @ self.C.T + u @ self.D.T
        if add_noise and not self.measurement_noise_is_zero:
            shape = (*y.shape[:-1], self.d_m)
            if rng is None:
                xi = torch.randn(shape, device=y.device, dtype=y.dtype)
            else:
                xi = torch.randn(shape, device=y.device, dtype=y.dtype, generator=rng)
            y = y + xi @ self.H.T
        return y


@dataclass
class TrajectoryData:
    """Output of :func:`simulate`."""

    ts: torch.Tensor
    x: torch.Tensor
    u: torch.Tensor
    y: torch.Tensor
    dt: float = field(init=False)

    def __post_init__(self):
        self.dt = float(self.ts[1].item() - self.ts[0].item())


def simulate(
    sde: LinearSDE,
    T: float,
    dt: float,
    *,
    x0: Optional[torch.Tensor] = None,
    seed: int = 0,
    method: str = "srk",
    add_measurement_noise: bool = True,
) -> TrajectoryData:
    """Simulate the SDE on [0, T] with sampling step `dt`.

    Uses `torchsde.sdeint`; when G has zero columns this reproduces a
    classical ODE solution.
    """
    device, dtype = sde.A.device, sde.A.dtype
    N = int(round(T / dt)) + 1
    ts = torch.linspace(0.0, T, N, device=device, dtype=dtype)

    if x0 is None:
        x0 = torch.zeros(1, sde.n, device=device, dtype=dtype)
    elif x0.ndim == 1:
        x0 = x0.unsqueeze(0).to(device=device, dtype=dtype)

    bm = torchsde.BrownianInterval(
        t0=float(ts[0].item()),
        t1=float(ts[-1].item()),
        size=(1, sde.d_w),
        dtype=dtype,
        device=device,
        entropy=seed,
        dt=dt,
        levy_area_approximation="space-time",
    )

    xs = torchsde.sdeint(sde, x0, ts, method=method, dt=dt, bm=bm)  # (N, 1, n)
    xs = xs.squeeze(1)  # (N, n)

    us = torch.stack([sde.u(ts[i]) for i in range(N)], dim=0).to(dtype)  # (N, m)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 1)
    ys = sde.observe(xs, us, add_noise=add_measurement_noise, rng=rng)

    return TrajectoryData(ts=ts, x=xs, u=us, y=ys)


def make_multisine_control(
    m: int,
    *,
    n_harmonics: int = 6,
    frequencies: Optional[torch.Tensor] = None,
    amplitudes: Optional[torch.Tensor] = None,
    phases: Optional[torch.Tensor] = None,
    T: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    seed: int = 7,
) -> ControlFn:
    """Build a deterministic multisine u(t) with `n_harmonics` per channel.

    u_i(t) = sum_k a_{ik} sin(2 pi f_{ik} t + phi_{ik}).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if frequencies is None:
        base = torch.arange(1, n_harmonics + 1, device=device, dtype=dtype) / max(T, 1e-12)
        frequencies = base.unsqueeze(0).expand(m, -1).clone()
        frequencies = frequencies + 0.05 * torch.arange(m, device=device, dtype=dtype).unsqueeze(1)
    if amplitudes is None:
        amplitudes = 1.0 + 0.5 * torch.rand(m, n_harmonics, generator=gen, device=device, dtype=dtype)
    if phases is None:
        phases = 2.0 * torch.pi * torch.rand(m, n_harmonics, generator=gen, device=device, dtype=dtype)

    freqs = frequencies.to(device=device, dtype=dtype)
    amps = amplitudes.to(device=device, dtype=dtype)
    phis = phases.to(device=device, dtype=dtype)

    def u_of_t(t: torch.Tensor) -> torch.Tensor:
        t_val = t if t.ndim == 0 else t[0]
        arg = 2.0 * torch.pi * freqs * t_val + phis  # (m, n_h)
        return (amps * torch.sin(arg)).sum(dim=-1)

    return u_of_t


__all__ = [
    "LinearSDE",
    "TrajectoryData",
    "simulate",
    "make_multisine_control",
]
