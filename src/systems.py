"""Example dynamical systems for controllability analysis.

This module provides functions that generate (A, B, C, D) state-space representations
for various LTI systems used in controllability analysis examples.

Each function returns:
    A: System matrix (n, n)
    B: Input matrix (n, m)
    C: Output matrix (p, n)
    D: Feedthrough matrix (p, m)

System dynamics:
    dx/dt = Ax + Bu
    y = Cx + Du
"""

import torch
from typing import Tuple, Optional


def random_stable_system(
    n: int = 4,
    m: int = 2,
    p: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    stability_margin: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random stable LTI system.
    
    Creates random matrices such that all eigenvalues of A have negative real parts.
    This is the default example used in main.py.
    
    Args:
        n: State dimension (default: 4)
        m: Input dimension (default: 2)
        p: Output dimension (default: 3)
        device: Target device (default: CPU)
        dtype: Data type (default: float64)
        stability_margin: Minimum distance of eigenvalues from imaginary axis
        seed: Random seed for reproducibility
        
    Returns:
        A: System matrix (n, n) with all eigenvalues having Re(λ) < -stability_margin
        B: Input matrix (n, m)
        C: Output matrix (p, n)
        D: Feedthrough matrix (p, m)
    """
    from src import generate_stable_system
    
    if seed is not None:
        torch.manual_seed(seed)
    
    return generate_stable_system(n, m, p, device=device, dtype=dtype, stability_margin=stability_margin)


def two_spring_system(
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 1.0,
    k2: float = 1.0,
    b1: float = 0.1,
    b2: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two independent spring-mass-damper systems.
    
    Physical setup:
        - Two independent mass-spring-damper oscillators
        - Only mass 1 is actuated (force applied to m1)
        - Both positions are measured
    
    State vector: x = [x1, v1, x2, v2]^T
        x1, x2: positions of masses 1 and 2
        v1, v2: velocities of masses 1 and 2
    
    System matrices:
        A = [[0,    1,    0,    0   ],
             [-k1/m1, -b1/m1, 0,    0   ],
             [0,    0,    0,    1   ],
             [0,    0,    -k2/m2, -b2/m2]]
             
        B = [[0], [1/m1], [0], [0]]
        
    Note: This system is NOT fully controllable since mass 2 is decoupled.
    
    Args:
        m1, m2: Masses (default: 1.0)
        k1, k2: Spring constants (default: 1.0)
        b1, b2: Damping coefficients (default: 0.1)
        device: Target device
        dtype: Data type
        
    Returns:
        A, B, C, D: State-space matrices
    """
    
    # State dimension n=4, input dimension m=1, output dimension p=2
    A = torch.tensor([
        [0.0,       1.0,        0.0,       0.0],
        [-k1/m1,   -b1/m1,      0.0,       0.0],
        [0.0,       0.0,        0.0,       1.0],
        [0.0,       0.0,       -k2/m2,    -b2/m2],
    ], device=device, dtype=dtype)
    
    B = torch.tensor([
        [0.0],
        [1.0/m1],
        [0.0],
        [0.0],
    ], device=device, dtype=dtype)
    
    # Output: both positions
    C = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # y1 = x1
        [0.0, 0.0, 1.0, 0.0],  # y2 = x2
    ], device=device, dtype=dtype)
    
    # No direct feedthrough
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    
    return A, B, C, D


def coupled_spring_system(
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 1.0,
    k2: float = 1.0,
    b1: float = 0.1,
    b2: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coupled (concatenated) spring-mass-damper system.
    
    Physical setup:
        Wall --[k1, b1]-- m1 --[k2]-- m2
        
        - Mass 1 connected to wall via spring k1 and damper b1
        - Mass 2 connected to mass 1 via spring k2
        - Mass 2 has damper b2 (to ground or inherent)
        - Only mass 1 is actuated
        - Both positions are measured
    
    State vector: x = [x1, v1, x2, v2]^T
        x1, x2: positions of masses 1 and 2
        v1, v2: velocities of masses 1 and 2
    
    System matrices:
        A = [[0,           1,       0,        0      ],
             [-(k1+k2)/m1, -b1/m1,  k2/m1,    0      ],
             [0,           0,       0,        1      ],
             [k2/m2,       0,       -k2/m2,   -b2/m2 ]]
             
        B = [[0], [1/m1], [0], [0]]
        
    Note: This system IS fully controllable since the coupling spring k2
    allows force to propagate from m1 to m2.
    
    Args:
        m1, m2: Masses (default: 1.0)
        k1: Spring constant connecting m1 to wall (default: 1.0)
        k2: Spring constant connecting m1 to m2 (default: 1.0)
        b1: Damping coefficient for m1 (default: 0.1)
        b2: Damping coefficient for m2 (default: 0.1)
        device: Target device
        dtype: Data type
        
    Returns:
        A, B, C, D: State-space matrices
    """
    
    # State dimension n=4, input dimension m=1, output dimension p=2
    A = torch.tensor([
        [0.0,             1.0,       0.0,       0.0],
        [-(k1+k2)/m1,    -b1/m1,     k2/m1,     0.0],
        [0.0,             0.0,       0.0,       1.0],
        [k2/m2,           0.0,      -k2/m2,    -b2/m2],
    ], device=device, dtype=dtype)
    
    B = torch.tensor([
        [0.0],
        [1.0/m1],
        [0.0],
        [0.0],
    ], device=device, dtype=dtype)
    
    # Output: both positions
    C = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # y1 = x1
        [0.0, 0.0, 1.0, 0.0],  # y2 = x2
    ], device=device, dtype=dtype)
    
    # No direct feedthrough
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    
    return A, B, C, D


def double_integrator(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple double integrator system (point mass with force input).
    
    Physical setup:
        d²x/dt² = u  (Newton's second law with m=1)
    
    State vector: x = [position, velocity]^T
    
    System matrices:
        A = [[0, 1], [0, 0]]
        B = [[0], [1]]
        C = [[1, 0]]  (measure position)
        D = [[0]]
        
    This is a minimal controllable and observable system.
    
    Args:
        device: Target device
        dtype: Data type
        
    Returns:
        A, B, C, D: State-space matrices
    """
    if device is None:
        device = torch.device("cpu")
    
    A = torch.tensor([
        [0.0, 1.0],
        [0.0, 0.0],
    ], device=device, dtype=dtype)
    
    B = torch.tensor([
        [0.0],
        [1.0],
    ], device=device, dtype=dtype)
    
    C = torch.tensor([
        [1.0, 0.0],
    ], device=device, dtype=dtype)
    
    D = torch.zeros(1, 1, device=device, dtype=dtype)
    
    return A, B, C, D


def inverted_pendulum(
    M: float = 1.0,
    m: float = 0.1,
    l: float = 1.0,
    b: float = 0.1,
    g: float = 9.81,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linearized inverted pendulum on a cart.
    
    Physical setup:
        - Cart of mass M on a track
        - Pendulum of mass m and length l attached to cart
        - Damping coefficient b for cart
        - Linearized around upright equilibrium (θ = 0)
    
    State vector: x = [cart_position, cart_velocity, angle, angular_velocity]^T
    
    Args:
        M: Cart mass (default: 1.0)
        m: Pendulum mass (default: 0.1)
        l: Pendulum length (default: 1.0)
        b: Cart damping coefficient (default: 0.1)
        g: Gravitational acceleration (default: 9.81)
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        
    Returns:
        A, B, C, D: State-space matrices (linearized around θ=0)
    """
    
    # Denominator terms
    denom = M + m
    
    A = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -b/denom, m*g/denom, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -b/(l*denom), (M+m)*g/(l*denom), 0.0],
    ], device=device, dtype=dtype)
    
    B = torch.tensor([
        [0.0],
        [1.0/denom],
        [0.0],
        [1.0/(l*denom)],
    ], device=device, dtype=dtype)
    
    # Output: cart position and pendulum angle
    C = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # cart position
        [0.0, 0.0, 1.0, 0.0],  # pendulum angle
    ], device=device, dtype=dtype)
    
    D = torch.zeros(2, 1, device=device, dtype=dtype)
    
    return A, B, C, D


# Dictionary of all available systems for easy access
SYSTEMS = {
    "random": random_stable_system,
    "two_spring": two_spring_system,
    "coupled_spring": coupled_spring_system,
    "double_integrator": double_integrator,
    "inverted_pendulum": inverted_pendulum,
}


def get_system(name: str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get a system by name.
    
    Args:
        name: System name (one of: random, two_spring, coupled_spring, 
              double_integrator, inverted_pendulum)
        **kwargs: Additional arguments passed to the system function
        
    Returns:
        A, B, C, D: State-space matrices
        
    Example:
        >>> A, B, C, D = get_system("coupled_spring", m1=2.0, k1=0.5)
    """
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system: {name}. Available: {list(SYSTEMS.keys())}")
    return SYSTEMS[name](**kwargs)


def print_system_info(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, name: str = "System"):
    """Print information about a system including controllability and observability.
    
    Args:
        A, B, C, D: State-space matrices
        name: System name for display
    """
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Dimensions: n={n} (states), m={m} (inputs), p={p} (outputs)")
    
    # Eigenvalues
    eigvals = torch.linalg.eigvals(A)
    print(f"\nEigenvalues of A:")
    for i, ev in enumerate(eigvals):
        stability = "stable" if ev.real < 0 else ("marginal" if ev.real == 0 else "UNSTABLE")
        print(f"  λ_{i+1} = {ev.real:.4f} + {ev.imag:.4f}i  ({stability})")
    
    # Controllability
    C_mat = torch.cat([torch.linalg.matrix_power(A, k) @ B for k in range(n)], dim=1)
    rank_C = torch.linalg.matrix_rank(C_mat).item()
    print(f"\nControllability matrix rank: {rank_C}/{n}")
    print(f"Controllable: {rank_C == n}")
    
    # Observability
    O_mat = torch.cat([C @ torch.linalg.matrix_power(A, k) for k in range(n)], dim=0)
    rank_O = torch.linalg.matrix_rank(O_mat).item()
    print(f"Observability matrix rank: {rank_O}/{n}")
    print(f"Observable: {rank_O == n}")


if __name__ == "__main__":
    # Demo: print info for all systems
    print("Available example systems for controllability analysis")
    print("="*60)
    
    for name, func in SYSTEMS.items():
        if name == "random":
            A, B, C, D = func(seed=42)
        else:
            A, B, C, D = func()
        print_system_info(A, B, C, D, name=name.replace("_", " ").title())
