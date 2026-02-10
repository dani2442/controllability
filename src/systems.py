"""Example dynamical systems for controllability analysis.

This module mainly provides linear state-space systems (A, B, C, D).
For frictional spring systems, it can also return nonlinear dynamics callables.
"""

import torch
from typing import Callable, Optional, Tuple

SpringDynamics = Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]
LinearSystem = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
NonlinearSystem = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SpringDynamics]


def _get_param(params: dict[str, float], *keys: str, default: float) -> float:
    """Read a scalar parameter from multiple key aliases."""
    for key in keys:
        if key in params:
            return float(params[key])
    return float(default)


def _smooth_abs(v: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(v * v + eps * eps)


def _smooth_sign(v: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.tanh(v / eps)


def _coulomb_force(v: torch.Tensor, fc: float, eps: float) -> torch.Tensor:
    return fc * _smooth_sign(v, eps)


def _stribeck_force(
    v: torch.Tensor,
    fc: float,
    fs: float,
    vs: float,
    alpha: float,
    viscous: float,
    eps: float,
) -> torch.Tensor:
    if vs <= 0.0:
        raise ValueError("Stribeck parameter 'vs' must be > 0.")
    if alpha <= 0.0:
        raise ValueError("Stribeck parameter 'alpha' must be > 0.")
    abs_v = _smooth_abs(v, eps)
    g = fc + (fs - fc) * torch.exp(-torch.pow(abs_v / vs, alpha))
    return g * _smooth_sign(v, eps) + viscous * v


def _normalize_state(x: torch.Tensor, state_dim: int) -> tuple[torch.Tensor, bool]:
    if x.ndim == 1:
        if x.shape[0] != state_dim:
            raise ValueError(f"Expected state size {state_dim}, got {x.shape[0]}.")
        return x.unsqueeze(0), True
    if x.ndim < 2 or x.shape[-1] != state_dim:
        raise ValueError(f"Expected state tensor with trailing dimension {state_dim}.")
    return x, False


def _normalize_input(
    u: torch.Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(u):
        u_tensor = torch.tensor(u, device=device, dtype=dtype)
    else:
        u_tensor = u.to(device=device, dtype=dtype)

    if u_tensor.ndim == 0:
        return u_tensor.expand(batch_size)
    if u_tensor.ndim == 1:
        if u_tensor.shape[0] == 1:
            return u_tensor.expand(batch_size)
        if u_tensor.shape[0] == batch_size:
            return u_tensor
    if u_tensor.ndim == 2 and u_tensor.shape == (batch_size, 1):
        return u_tensor.squeeze(-1)
    raise ValueError("Input u must be scalar, (batch,), or (batch, 1).")


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
    friction_model: str = "none",
    friction_params: Optional[dict[str, float]] = None,
    return_dynamics: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> LinearSystem | NonlinearSystem:
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
    
    Friction:
        - `friction_model="none"`: linear spring-damper model (default)
        - `friction_model="coulomb"`: F_fi(v_i) = F_c,i * sgn(v_i)
        - `friction_model="stribeck"`:
          F_fi(v_i) = (F_c,i + (F_s,i-F_c,i)exp(-( |v_i|/v_s,i )^alpha_i))sgn(v_i) + B_i v_i
        - set `return_dynamics=True` to get the nonlinear drift callable.

    Args:
        m1, m2: Masses (default: 1.0)
        k1, k2: Spring constants (default: 1.0)
        b1, b2: Damping coefficients (default: 0.1)
        friction_model: One of {"none", "coulomb", "stribeck"}
        friction_params: Model parameters. Supported aliases:
            coulomb: Fc1/Fc2 (or fc1/fc2), eps
            stribeck: Fc1/Fc2, Fs1/Fs2, vs1/vs2, alpha1/alpha2, B1/B2, eps
        return_dynamics: Return nonlinear drift f(x, u) in addition to (A, B, C, D)
        device: Target device
        dtype: Data type
        
    Returns:
        If return_dynamics is False:
            A, B, C, D
        If return_dynamics is True:
            A, B, C, D, dynamics
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
    
    model = friction_model.lower()
    if model not in {"none", "coulomb", "stribeck"}:
        raise ValueError("friction_model must be one of {'none', 'coulomb', 'stribeck'}.")
    if model == "none" and not return_dynamics:
        return A, B, C, D

    params = {} if friction_params is None else dict(friction_params)
    eps = _get_param(params, "eps", "epsilon", default=1e-6)
    if eps <= 0.0:
        raise ValueError("Friction smoothing parameter 'eps' must be > 0.")

    fc1 = _get_param(params, "Fc1", "fc1", default=0.5)
    fc2 = _get_param(params, "Fc2", "fc2", default=0.5)
    fs1 = _get_param(params, "Fs1", "fs1", default=fc1)
    fs2 = _get_param(params, "Fs2", "fs2", default=fc2)
    vs1 = _get_param(params, "vs1", "v_s1", default=1.0)
    vs2 = _get_param(params, "vs2", "v_s2", default=1.0)
    alpha1 = _get_param(params, "alpha1", default=1.0)
    alpha2 = _get_param(params, "alpha2", default=1.0)
    bv1 = _get_param(params, "B1", "b_f1", default=0.0)
    bv2 = _get_param(params, "B2", "b_f2", default=0.0)

    if model in {"coulomb", "stribeck"}:
        if fc1 < 0.0 or fc2 < 0.0:
            raise ValueError("Coulomb friction levels Fc1/Fc2 must be >= 0.")
    if model == "stribeck":
        if fs1 < 0.0 or fs2 < 0.0:
            raise ValueError("Static friction levels Fs1/Fs2 must be >= 0.")

    def friction_forces(v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if model == "none":
            zeros = torch.zeros_like(v1)
            return zeros, torch.zeros_like(v2)
        if model == "coulomb":
            return _coulomb_force(v1, fc1, eps), _coulomb_force(v2, fc2, eps)
        return (
            _stribeck_force(v1, fc1, fs1, vs1, alpha1, bv1, eps),
            _stribeck_force(v2, fc2, fs2, vs2, alpha2, bv2, eps),
        )

    def dynamics(x: torch.Tensor, u: torch.Tensor | float) -> torch.Tensor:
        x_batch, squeeze = _normalize_state(x, state_dim=4)
        u_batch = _normalize_input(u, x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype)

        x1 = x_batch[:, 0]
        v1 = x_batch[:, 1]
        x2 = x_batch[:, 2]
        v2 = x_batch[:, 3]

        ff1, ff2 = friction_forces(v1, v2)
        dx1 = v1
        dv1 = (-k1 * x1 - b1 * v1 - ff1 + u_batch) / m1
        dx2 = v2
        dv2 = (-k2 * x2 - b2 * v2 - ff2) / m2
        xdot = torch.stack([dx1, dv1, dx2, dv2], dim=-1)
        return xdot[0] if squeeze else xdot

    if model != "none" and not return_dynamics:
        raise ValueError(
            "Coulomb/Stribeck friction is nonlinear. Set return_dynamics=True to receive f(x, u)."
        )

    if return_dynamics:
        return A, B, C, D, dynamics
    return A, B, C, D


def coupled_spring_system(
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 1.0,
    k2: float = 1.0,
    b1: float = 0.1,
    b2: float = 0.1,
    friction_model: str = "none",
    friction_params: Optional[dict[str, float]] = None,
    return_dynamics: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> LinearSystem | NonlinearSystem:
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
    
    Friction:
        - `friction_model="none"`: linear spring-damper model (default)
        - `friction_model="coulomb"`: F_fi(v_i) = F_c,i * sgn(v_i)
        - `friction_model="stribeck"`:
          F_fi(v_i) = (F_c,i + (F_s,i-F_c,i)exp(-( |v_i|/v_s,i )^alpha_i))sgn(v_i) + B_i v_i
        - set `return_dynamics=True` to get the nonlinear drift callable.

    Args:
        m1, m2: Masses (default: 1.0)
        k1: Spring constant connecting m1 to wall (default: 1.0)
        k2: Spring constant connecting m1 to m2 (default: 1.0)
        b1: Damping coefficient for m1 (default: 0.1)
        b2: Damping coefficient for m2 (default: 0.1)
        friction_model: One of {"none", "coulomb", "stribeck"}
        friction_params: Model parameters. Supported aliases:
            coulomb: Fc1/Fc2 (or fc1/fc2), eps
            stribeck: Fc1/Fc2, Fs1/Fs2, vs1/vs2, alpha1/alpha2, B1/B2, eps
        return_dynamics: Return nonlinear drift f(x, u) in addition to (A, B, C, D)
        device: Target device
        dtype: Data type
        
    Returns:
        If return_dynamics is False:
            A, B, C, D
        If return_dynamics is True:
            A, B, C, D, dynamics
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
    
    model = friction_model.lower()
    if model not in {"none", "coulomb", "stribeck"}:
        raise ValueError("friction_model must be one of {'none', 'coulomb', 'stribeck'}.")
    if model == "none" and not return_dynamics:
        return A, B, C, D

    params = {} if friction_params is None else dict(friction_params)
    eps = _get_param(params, "eps", "epsilon", default=1e-6)
    if eps <= 0.0:
        raise ValueError("Friction smoothing parameter 'eps' must be > 0.")

    fc1 = _get_param(params, "Fc1", "fc1", default=0.0)
    fc2 = _get_param(params, "Fc2", "fc2", default=0.0)
    fs1 = _get_param(params, "Fs1", "fs1", default=fc1)
    fs2 = _get_param(params, "Fs2", "fs2", default=fc2)
    vs1 = _get_param(params, "vs1", "v_s1", default=1.0)
    vs2 = _get_param(params, "vs2", "v_s2", default=1.0)
    alpha1 = _get_param(params, "alpha1", default=1.0)
    alpha2 = _get_param(params, "alpha2", default=1.0)
    bv1 = _get_param(params, "B1", "b_f1", default=0.0)
    bv2 = _get_param(params, "B2", "b_f2", default=0.0)

    if model in {"coulomb", "stribeck"}:
        if fc1 < 0.0 or fc2 < 0.0:
            raise ValueError("Coulomb friction levels Fc1/Fc2 must be >= 0.")
    if model == "stribeck":
        if fs1 < 0.0 or fs2 < 0.0:
            raise ValueError("Static friction levels Fs1/Fs2 must be >= 0.")

    def friction_forces(v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if model == "none":
            zeros = torch.zeros_like(v1)
            return zeros, torch.zeros_like(v2)
        if model == "coulomb":
            return _coulomb_force(v1, fc1, eps), _coulomb_force(v2, fc2, eps)
        return (
            _stribeck_force(v1, fc1, fs1, vs1, alpha1, bv1, eps),
            _stribeck_force(v2, fc2, fs2, vs2, alpha2, bv2, eps),
        )

    def dynamics(x: torch.Tensor, u: torch.Tensor | float) -> torch.Tensor:
        x_batch, squeeze = _normalize_state(x, state_dim=4)
        u_batch = _normalize_input(u, x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype)

        x1 = x_batch[:, 0]
        v1 = x_batch[:, 1]
        x2 = x_batch[:, 2]
        v2 = x_batch[:, 3]

        ff1, ff2 = friction_forces(v1, v2)
        dx1 = v1
        dv1 = (-k1 * x1 - k2 * (x1 - x2) - b1 * v1 - ff1 + u_batch) / m1
        dx2 = v2
        dv2 = (-k2 * (x2 - x1) - b2 * v2 - ff2) / m2
        xdot = torch.stack([dx1, dv1, dx2, dv2], dim=-1)
        return xdot[0] if squeeze else xdot

    if model != "none" and not return_dynamics:
        raise ValueError(
            "Coulomb/Stribeck friction is nonlinear. Set return_dynamics=True to receive f(x, u)."
        )

    if return_dynamics:
        return A, B, C, D, dynamics
    return A, B, C, D


def spring_system_dahl_lugre(
    configuration: str = "coupled",
    friction_model: str = "dahl",
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 1.0,
    k2: float = 1.0,
    b1: float = 0.1,
    b2: float = 0.1,
    friction_params: Optional[dict[str, float]] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> NonlinearSystem:
    """Augmented spring system with internal friction states (Dahl or LuGre).

    State ordering: x = [x1, v1, x2, v2, z1, z2]^T.
    The returned drift callable implements the exact nonlinear model.

    Args:
        configuration: "two" (independent springs) or "coupled" (coupling spring k2)
        friction_model: "dahl" or "lugre"
        friction_params:
            dahl keys: sigma1, sigma2, Fc1, Fc2, eps
            lugre keys: Fc1, Fc2, Fs1, Fs2, vs1, vs2, alpha1, alpha2,
                        sigma0_1, sigma0_2, sigma1_1, sigma1_2, sigma2_1, sigma2_2,
                        g_min, eps

    Returns:
        A, B, C, D, dynamics
    """
    config = configuration.lower()
    if config not in {"two", "coupled"}:
        raise ValueError("configuration must be one of {'two', 'coupled'}.")

    model = friction_model.lower()
    if model not in {"dahl", "lugre"}:
        raise ValueError("friction_model must be one of {'dahl', 'lugre'}.")

    params = {} if friction_params is None else dict(friction_params)
    eps = _get_param(params, "eps", "epsilon", default=1e-6)
    if eps <= 0.0:
        raise ValueError("Friction smoothing parameter 'eps' must be > 0.")

    # Base linear part (without nonlinear internal-state attenuation terms).
    A = torch.zeros(6, 6, device=device, dtype=dtype)
    A[0, 1] = 1.0
    A[2, 3] = 1.0
    A[4, 1] = 1.0
    A[5, 3] = 1.0

    if config == "two":
        A[1, 0] = -k1 / m1
        A[1, 1] = -b1 / m1
        A[3, 2] = -k2 / m2
        A[3, 3] = -b2 / m2
    else:
        A[1, 0] = -(k1 + k2) / m1
        A[1, 1] = -b1 / m1
        A[1, 2] = k2 / m1
        A[3, 0] = k2 / m2
        A[3, 2] = -k2 / m2
        A[3, 3] = -b2 / m2

    if model == "dahl":
        sigma_d1 = _get_param(params, "sigma1", default=1.0)
        sigma_d2 = _get_param(params, "sigma2", default=1.0)
        fc_d1 = _get_param(params, "Fc1", "fc1", default=1.0)
        fc_d2 = _get_param(params, "Fc2", "fc2", default=1.0)
        if fc_d1 <= 0.0 or fc_d2 <= 0.0:
            raise ValueError("Dahl parameters Fc1/Fc2 must be > 0.")
        A[1, 4] = -sigma_d1 / m1
        A[3, 5] = -sigma_d2 / m2
    else:
        fc_l1 = _get_param(params, "Fc1", "fc1", default=1.0)
        fc_l2 = _get_param(params, "Fc2", "fc2", default=1.0)
        fs_l1 = _get_param(params, "Fs1", "fs1", default=fc_l1)
        fs_l2 = _get_param(params, "Fs2", "fs2", default=fc_l2)
        vs_l1 = _get_param(params, "vs1", "v_s1", default=1.0)
        vs_l2 = _get_param(params, "vs2", "v_s2", default=1.0)
        alpha_l1 = _get_param(params, "alpha1", default=1.0)
        alpha_l2 = _get_param(params, "alpha2", default=1.0)
        sigma0_l1 = _get_param(params, "sigma0_1", default=1.0)
        sigma0_l2 = _get_param(params, "sigma0_2", default=1.0)
        sigma1_l1 = _get_param(params, "sigma1_1", default=0.0)
        sigma1_l2 = _get_param(params, "sigma1_2", default=0.0)
        sigma2_l1 = _get_param(params, "sigma2_1", default=0.0)
        sigma2_l2 = _get_param(params, "sigma2_2", default=0.0)
        g_min = _get_param(params, "g_min", default=1e-6)
        if g_min <= 0.0:
            raise ValueError("LuGre guard 'g_min' must be > 0.")
        if vs_l1 <= 0.0 or vs_l2 <= 0.0:
            raise ValueError("LuGre parameters vs1/vs2 must be > 0.")
        if alpha_l1 <= 0.0 or alpha_l2 <= 0.0:
            raise ValueError("LuGre parameters alpha1/alpha2 must be > 0.")

        A[1, 1] -= (sigma1_l1 + sigma2_l1) / m1
        A[1, 4] = -sigma0_l1 / m1
        A[3, 3] -= (sigma1_l2 + sigma2_l2) / m2
        A[3, 5] = -sigma0_l2 / m2

    B = torch.tensor([[0.0], [1.0 / m1], [0.0], [0.0], [0.0], [0.0]], device=device, dtype=dtype)
    C = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y1 = x1
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # y2 = x2
        ],
        device=device,
        dtype=dtype,
    )
    D = torch.zeros(2, 1, device=device, dtype=dtype)

    def dynamics(x: torch.Tensor, u: torch.Tensor | float) -> torch.Tensor:
        x_batch, squeeze = _normalize_state(x, state_dim=6)
        u_batch = _normalize_input(u, x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype)

        x1 = x_batch[:, 0]
        v1 = x_batch[:, 1]
        x2 = x_batch[:, 2]
        v2 = x_batch[:, 3]
        z1 = x_batch[:, 4]
        z2 = x_batch[:, 5]

        if model == "dahl":
            abs_v1 = _smooth_abs(v1, eps)
            abs_v2 = _smooth_abs(v2, eps)
            dz1 = v1 - (abs_v1 / fc_d1) * z1
            dz2 = v2 - (abs_v2 / fc_d2) * z2
            ff1 = sigma_d1 * z1
            ff2 = sigma_d2 * z2
        else:
            abs_v1 = _smooth_abs(v1, eps)
            abs_v2 = _smooth_abs(v2, eps)
            g1 = fc_l1 + (fs_l1 - fc_l1) * torch.exp(-torch.pow(abs_v1 / vs_l1, alpha_l1))
            g2 = fc_l2 + (fs_l2 - fc_l2) * torch.exp(-torch.pow(abs_v2 / vs_l2, alpha_l2))
            g1 = torch.clamp(g1, min=g_min)
            g2 = torch.clamp(g2, min=g_min)

            dz1 = v1 - (abs_v1 / g1) * z1
            dz2 = v2 - (abs_v2 / g2) * z2
            ff1 = sigma0_l1 * z1 + sigma1_l1 * dz1 + sigma2_l1 * v1
            ff2 = sigma0_l2 * z2 + sigma1_l2 * dz2 + sigma2_l2 * v2

        dx1 = v1
        dx2 = v2
        if config == "two":
            dv1 = (-k1 * x1 - b1 * v1 - ff1 + u_batch) / m1
            dv2 = (-k2 * x2 - b2 * v2 - ff2) / m2
        else:
            dv1 = (-k1 * x1 - k2 * (x1 - x2) - b1 * v1 - ff1 + u_batch) / m1
            dv2 = (-k2 * (x2 - x1) - b2 * v2 - ff2) / m2

        xdot = torch.stack([dx1, dv1, dx2, dv2, dz1, dz2], dim=-1)
        return xdot[0] if squeeze else xdot

    return A, B, C, D, dynamics


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
