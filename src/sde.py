"""SDE simulation for controlled linear systems with input-output.

System model:
    dx(t) = (Ax + Bu)dt + β dW(t)    (process noise)
    y(t)  = Cx + Du + δ v(t)          (measurement noise)
"""

import argparse
import torch
import torchsde
import numpy as np
from typing import Callable, Optional, Tuple

DynamicsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LinearSDE(torchsde.SDEIto):
    """Linear SDE with additive process noise and output equation.
    
    State equation:  dx(t) = (Ax + Bu)dt + β dW(t)
    Output equation: y(t)  = Cx + Du + δ v(t)
    
    Attributes:
        A: System matrix (n, n)
        B: Input matrix (n, m)
        C: Output matrix (p, n)
        D: Feedthrough matrix (p, m)
        Beta: Process noise intensity (n, q)
        Delta: Measurement noise intensity (p, r)
    """
    
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        Beta: torch.Tensor,
        Delta: Optional[torch.Tensor] = None,
        control_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize the linear SDE.
        
        Args:
            A: System matrix (n, n)
            B: Input matrix (n, m)
            C: Output matrix (p, n)
            D: Feedthrough matrix (p, m)
            Beta: Process noise intensity (n, q)
            Delta: Measurement noise intensity (p, r), defaults to zeros
            control_fn: Optional control function u(t, x) -> (batch, m)
        """
        super().__init__(noise_type="additive")
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Beta = Beta
        
        # Dimensions
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]
        self.q = Beta.shape[1]
        
        # Measurement noise
        if Delta is None:
            self.Delta = torch.zeros(self.p, 1, device=A.device, dtype=A.dtype)
            self.r = 1
        else:
            self.Delta = Delta
            self.r = Delta.shape[1]
        
        self._control_fn = control_fn

    def u(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute control input at time t.
        
        Default: sinusoidal inputs with different frequencies.
        """
        if self._control_fn is not None:
            return self._control_fn(t, x)
        
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Sinusoidal inputs with varying frequencies
        freqs = torch.linspace(1.0, 3.0, self.m, device=device, dtype=dtype)
        t_val = t.item() if t.ndim == 0 else t[0].item()
        #u_val = torch.sin((np.sin(0.5*t_val) + np.cos(0.5*t_val)) * freqs)
        u_val = torch.sin(.1039*(0.122*np.cos(0.0523*t_val)+np.sin(t_val + 1.303857103*np.sin(0.1233939*t_val) + 0.5*t_val) + np.sin(0.5*t_val)) * freqs)
        return u_val.unsqueeze(0).expand(batch_size, -1)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift: f(t, x) = Ax + Bu."""
        ut = self.u(t, x).to(x.dtype)
        return x @ self.A.T + ut @ self.B.T

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion: g(t, x) = β (additive)."""
        batch_size = x.shape[0]
        return self.Beta.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)

    def output(self, x: torch.Tensor, u: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Compute output y = Cx + Du + δv.
        
        Args:
            x: State (batch, n) or (N, batch, n)
            u: Input (batch, m) or (N, batch, m)
            add_noise: Whether to add measurement noise
            
        Returns:
            y: Output (batch, p) or (N, batch, p)
        """
        y = x @ self.C.T + u @ self.D.T
        
        if add_noise:
            noise = torch.randn(*y.shape[:-1], self.r, device=y.device, dtype=y.dtype)
            y = y + noise @ self.Delta.T
        
        return y


class NonlinearSDE(LinearSDE):
    """SDE wrapper with state-dependent drift f(x, u)."""

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        Beta: torch.Tensor,
        Delta: Optional[torch.Tensor],
        dynamics_fn: DynamicsFn,
        control_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(A, B, C, D, Beta, Delta, control_fn=control_fn)
        self._dynamics_fn = dynamics_fn

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ut = self.u(t, x).to(x.dtype)
        return self._dynamics_fn(x, ut)


def build_sde(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    Beta: torch.Tensor,
    Delta: Optional[torch.Tensor] = None,
    dynamics_fn: Optional[DynamicsFn] = None,
    control_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> LinearSDE:
    """Construct linear or nonlinear SDE depending on `dynamics_fn`."""
    if dynamics_fn is None:
        return LinearSDE(A, B, C, D, Beta, Delta, control_fn=control_fn)
    return NonlinearSDE(A, B, C, D, Beta, Delta, dynamics_fn, control_fn=control_fn)


def make_friction_params(
    *,
    fc1: float = 0.0,
    fc2: float = 0.0,
    fs1: float = 0.0,
    fs2: float = 0.0,
    vs1: float = 1.0,
    vs2: float = 1.0,
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    b1: float = 0.0,
    b2: float = 0.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Build friction parameter dict for systems with Coulomb/Stribeck friction."""
    return {
        "Fc1": fc1,
        "Fc2": fc2,
        "Fs1": fs1 if fs1 > 0.0 else fc1,
        "Fs2": fs2 if fs2 > 0.0 else fc2,
        "vs1": vs1,
        "vs2": vs2,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "B1": b1,
        "B2": b2,
        "eps": eps,
    }


def add_friction_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach shared friction arguments to an argparse parser."""
    parser.add_argument("--fc1", type=float, default=0.05)
    parser.add_argument("--fc2", type=float, default=0.05)
    parser.add_argument("--fs1", type=float, default=0.05)
    parser.add_argument("--fs2", type=float, default=0.05)
    parser.add_argument("--vs1", type=float, default=1.0)
    parser.add_argument("--vs2", type=float, default=1.0)
    parser.add_argument("--alpha1", type=float, default=1.0)
    parser.add_argument("--alpha2", type=float, default=1.0)
    parser.add_argument("--bf1", type=float, default=0.0, help="Stribeck viscous term B1")
    parser.add_argument("--bf2", type=float, default=0.0, help="Stribeck viscous term B2")
    parser.add_argument("--friction-eps", type=float, default=1e-6)
    return parser


def friction_params_from_namespace(args: argparse.Namespace) -> dict[str, float]:
    """Extract friction parameter dict from parsed CLI namespace."""
    return make_friction_params(
        fc1=float(args.fc1),
        fc2=float(args.fc2),
        fs1=float(args.fs1),
        fs2=float(args.fs2),
        vs1=float(args.vs1),
        vs2=float(args.vs2),
        alpha1=float(args.alpha1),
        alpha2=float(args.alpha2),
        b1=float(args.bf1),
        b2=float(args.bf2),
        eps=float(args.friction_eps),
    )


def load_system_with_friction(
    *,
    system_name: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    friction_model: str = "none",
    friction_params: Optional[dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[DynamicsFn]]:
    """Load a system and optional nonlinear friction drift from src.systems."""
    from src.systems import get_system

    if friction_model == "none":
        kwargs = {"device": device, "dtype": dtype}
        if system_name == "random" and seed is not None:
            kwargs["seed"] = seed
        A, B, C, D = get_system(system_name, **kwargs)
        return A, B, C, D, None

    if system_name not in {"two_spring", "coupled_spring"}:
        raise ValueError("Nonlinear friction is supported only for two_spring and coupled_spring.")

    A, B, C, D, dynamics_fn = get_system(
        system_name,
        device=device,
        dtype=dtype,
        friction_model=friction_model,
        friction_params=friction_params,
        return_dynamics=True,
    )
    return A, B, C, D, dynamics_fn


def simulate(
    sde: LinearSDE,
    T: float,
    dt: float,
    x0: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the SDE and compute outputs.
    
    Args:
        sde: The LinearSDE object
        T: Final time
        dt: Time step
        x0: Initial condition (n,), defaults to zeros
        seed: Random seed
        
    Returns:
        ts: Time grid (N,)
        x: State trajectory (N, n)
        u: Input trajectory (N, m)
        y: Output trajectory (N, p)
    """
    torch.manual_seed(seed)
    
    device = sde.A.device
    dtype = sde.A.dtype
    
    # Time grid
    N = int(T / dt) + 1
    ts = torch.linspace(0.0, T, N, device=device, dtype=dtype)
    
    # Initial condition
    if x0 is None:
        x0 = torch.zeros(1, sde.n, device=device, dtype=dtype)
    elif x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    
    # Brownian motion
    bm = torchsde.BrownianInterval(
        t0=float(ts[0].item()),
        t1=float(ts[-1].item()),
        size=(1, sde.q),
        dtype=dtype,
        device=device,
        entropy=seed,
        dt=dt,
        levy_area_approximation="space-time",
    )
    
    # Simulate
    x_traj = torchsde.sdeint(sde, x0, ts, method="srk", dt=dt, bm=bm)  # (N, 1, n)
    x_traj = x_traj.squeeze(1)  # (N, n)
    
    # Compute inputs at each time step
    u_traj = torch.stack([sde.u(ts[i], x_traj[i:i+1]).squeeze(0) for i in range(N)])  # (N, m)
    
    # Compute outputs
    y_traj = sde.output(x_traj, u_traj, add_noise=True)  # (N, p)
    
    return ts, x_traj, u_traj, y_traj
