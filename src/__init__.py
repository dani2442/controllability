"""Data-driven controllability analysis for continuous-time systems.

This package implements the data-driven Hautus tests for LTI systems
with input-output measurements:

    dx = (Ax + Bu)dt + β dW(t)    (state equation with process noise)
    y  = Cx + Du + δv(t)           (output equation with measurement noise)

Main components:
    - sde: SDE simulation with torchsde
    - gramians: G_{L,K}(λ) and K_{L,K}(λ) computation
    - utils: System generation and matrix utilities
    - visualization: Plotting functions
"""

from .utils import (
    generate_stable_system,
    compute_lift_matrix,
    compute_observability_matrix,
    compute_observability_index,
    compute_toeplitz_matrix,
    smooth_signal,
)

from .sde import (
    LinearSDE,
    simulate,
)

from .gramians import (
    compute_G_LK,
    compute_H_LK,
    compute_K_LK,
    compute_K_LK_reduced,
    compute_model_basis_R,
    compute_Sk_lambda,
    compute_N_LK_lambda,
    compute_Q_LK_model,
    compute_basis_alignment,
    aligned_q_error,
    compute_observable_quotient_coordinates,
    compute_Q_LK_from_coordinates,
    compute_Q_LK,
    compute_persistent_excitation_gramian,
    compute_filtered_signal,
    compute_derivative_lift,
    compute_filtered_derivative_lift,
    check_persistent_excitation,
    check_controllability,
    check_controllability_reduced,
    check_controllability_observable_quotient,
)

from .visualization import (
    plot_trajectories,
    plot_eigenvalues,
    plot_gramian_eigenvalues,
    plot_controllability_margin,
)

__all__ = [
    # Utils
    "generate_stable_system",
    "compute_lift_matrix",
    "compute_observability_matrix",
    "compute_observability_index",
    "compute_toeplitz_matrix",
    "smooth_signal",
    # SDE
    "LinearSDE",
    "simulate",
    # Gramians
    "compute_G_LK",
    "compute_H_LK",
    "compute_K_LK",
    "compute_K_LK_reduced",
    "compute_model_basis_R",
    "compute_Sk_lambda",
    "compute_N_LK_lambda",
    "compute_Q_LK_model",
    "compute_basis_alignment",
    "aligned_q_error",
    "compute_observable_quotient_coordinates",
    "compute_Q_LK_from_coordinates",
    "compute_Q_LK",
    "compute_persistent_excitation_gramian",
    "compute_filtered_signal",
    "compute_derivative_lift",
    "compute_filtered_derivative_lift",
    "check_persistent_excitation",
    "check_controllability",
    "check_controllability_reduced",
    "check_controllability_observable_quotient",
    # Visualization
    "plot_trajectories",
    "plot_eigenvalues",
    "plot_gramian_eigenvalues",
    "plot_controllability_margin",
]
