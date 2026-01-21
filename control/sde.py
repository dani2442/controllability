"""SDE simulation for controlled linear systems.

This module provides classes for simulating stochastic differential equations
of the form:
    dx(t) = (A x(t) + B u(t)) dt + β dW(t)
"""

import torch
import torchsde
from typing import Callable, Optional


class ControlledLinearSDE(torchsde.SDEIto):
    """Controlled linear SDE with additive noise.
    
    dx(t) = (A x(t) + B u(t)) dt + β dW(t)
    
    Attributes:
        A: System matrix of shape (n, n)
        B: Input matrix of shape (n, m)
        Beta: Noise matrix of shape (n, q)
        control_fn: Optional custom control function u(t, x)
    """
    
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        Beta: torch.Tensor,
        control_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize the controlled linear SDE.
        
        Args:
            A: System matrix (n, n)
            B: Input matrix (n, m)
            Beta: Noise matrix (n, q)
            control_fn: Optional control function u(t, x) -> (batch, m)
                       If None, uses sinusoidal inputs
        """
        super().__init__(noise_type="general")
        self.A = A
        self.B = B
        self.Beta = Beta
        self._control_fn = control_fn
        
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.q = Beta.shape[1]

    def u(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute control input at time t.
        
        Default implementation uses sinusoidal inputs with different frequencies.
        
        Args:
            t: Time (scalar or tensor)
            x: State of shape (batch, n)
            
        Returns:
            Control input of shape (batch, m)
        """
        if self._control_fn is not None:
            return self._control_fn(t, x)
        
        batch_size = x.shape[0]
        m = self.m
        device = x.device
        dtype = x.dtype
        
        w = torch.linspace(1, 5, m, device=device, dtype=dtype)
        
        if t.ndim == 0:
            # Scalar time -> (batch, m)
            u = torch.sin(t * w)
            return u.unsqueeze(0).expand(batch_size, -1)
        else:
            # Time vector (N,) -> (N, batch, m)
            u = torch.sin(t[:, None] * w[None, :])
            return u.unsqueeze(1).expand(-1, batch_size, -1)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift coefficient: f(t, x) = A x + B u(t, x).
        
        Args:
            t: Time
            x: State of shape (batch, n)
            
        Returns:
            Drift of shape (batch, n)
        """
        ut = self.u(t, x).to(x)
        Ax = x @ self.A.T
        But = ut @ self.B.T
        return Ax + But

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient: g(t, x) = β (additive noise).
        
        Args:
            t: Time
            x: State of shape (batch, n)
            
        Returns:
            Diffusion matrix of shape (batch, n, q)
        """
        batch_size = x.shape[0]
        return self.Beta.unsqueeze(0).expand(batch_size, -1, -1).to(x)


def simulate_sde(
    sde: ControlledLinearSDE,
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float,
    method: str = "euler",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate the SDE and compute control inputs.
    
    Args:
        sde: The SDE object
        x0: Initial condition of shape (batch, n)
        ts: Time grid of shape (N,)
        dt: Minimum time step
        method: Integration method ("euler", "milstein", etc.)
        
    Returns:
        x: State trajectory of shape (N, batch, n)
        u: Input trajectory of shape (N, batch, m)
    """
    # Use BrownianTree which handles long time horizons efficiently
    bm = torchsde.BrownianTree(
        t0=ts[0],
        t1=ts[-1],
        w0=torch.zeros(x0.shape[0], sde.q, device=x0.device, dtype=x0.dtype),
        entropy=0,  # Fixed seed for reproducibility within simulation
    )
    
    # Simulate state trajectory
    x = torchsde.sdeint(sde, x0, ts, method=method, dt=dt, bm=bm)
    
    # Compute inputs at each time step
    batch_size = x0.shape[0]
    u = sde.u(ts, x[:, 0, :])  # Get u for first batch element to get shape
    if u.ndim == 2:
        # Need to expand for batch
        u = u.unsqueeze(1).expand(-1, batch_size, -1)
    
    return x, u


def sdeint_safe(
    sde: ControlledLinearSDE,
    x0: torch.Tensor,
    ts: torch.Tensor,
    dt: float,
    method: str = "euler",
) -> torch.Tensor:
    """Safe wrapper around torchsde.sdeint that handles long time horizons.
    
    This function creates a proper BrownianTree to avoid recursion issues
    with long time simulations.
    
    Args:
        sde: The SDE object
        x0: Initial condition of shape (batch, n)
        ts: Time grid of shape (N,)
        dt: Time step
        method: Integration method
        
    Returns:
        x: State trajectory of shape (N, batch, n)
    """
    # Use BrownianTree which handles long time horizons efficiently
    bm = torchsde.BrownianTree(
        t0=ts[0],
        t1=ts[-1],
        w0=torch.zeros(x0.shape[0], sde.q, device=x0.device, dtype=x0.dtype),
    )
    return torchsde.sdeint(sde, x0, ts, method=method, dt=dt, bm=bm)


def create_time_grid(T: float, dt: float, device: torch.device) -> torch.Tensor:
    """Create a uniform time grid.
    
    Args:
        T: Final time
        dt: Time step
        device: Target device
        
    Returns:
        Time grid of shape (N,)
    """
    return torch.linspace(0.0, T, int(T / dt) + 1, device=device)
