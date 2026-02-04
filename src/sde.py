"""SDE simulation for controlled linear systems with input-output.

System model:
    dx(t) = (Ax + Bu)dt + β dW(t)    (process noise)
    y(t)  = Cx + Du + δ v(t)          (measurement noise)
"""

import torch
import torchsde
from typing import Callable, Optional, Tuple


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
        super().__init__(noise_type="general")
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
        u_val = torch.sin(t_val * freqs)
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
    bm = torchsde.BrownianTree(
        t0=ts[0],
        t1=ts[-1],
        w0=torch.zeros(1, sde.q, device=device, dtype=dtype),
        entropy=seed,
    )
    
    # Simulate
    x_traj = torchsde.sdeint(sde, x0, ts, method="euler", dt=dt, bm=bm)  # (N, 1, n)
    x_traj = x_traj.squeeze(1)  # (N, n)
    
    # Compute inputs at each time step
    u_traj = torch.stack([sde.u(ts[i], x_traj[i:i+1]).squeeze(0) for i in range(N)])  # (N, m)
    
    # Compute outputs
    y_traj = sde.output(x_traj, u_traj, add_noise=True)  # (N, p)
    
    return ts, x_traj, u_traj, y_traj
