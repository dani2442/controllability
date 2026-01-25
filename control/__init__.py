"""Control package for data-driven controllability analysis.

This package implements data-driven Hautus tests for continuous-time systems,
including both time-domain and frequency-domain formulations.

Main modules:
    - sde: SDE simulation for controlled linear systems
    - gramians: Gramian computation functions
    - hautus: Hautus test computations
    - visualization: Plotting utilities
    - utils: Utility functions
"""

from .utils import (
    complex_dtype_from_real,
    to_complex,
    stack_z,
    ensure_dt_vector,
    make_stable_A,
)

from .gramians import (
    gramian_Sz_time,
    integral_xxH_time,
    integral_xdot_xH_time,
    compute_candidate_eigenvalues,
    gramian_Sx_from_fft,
)

from .hautus import (
    cross_moment_H_time,
    cross_moment_H_fft,
    cross_moment_H_laplace,
    gramian_G_from_H,
    estimate_Hautus_matrix,
    true_Hautus_matrix,
    hautus_test,
    check_controllability,
    compare_with_true,
)

from .sde import (
    ControlledLinearSDE,
    simulate_sde,
    sdeint_safe,
    create_time_grid,
)

from .visualization import (
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series,
    plot_error_vs_T,
    plot_matrix_comparison,
    plot_singular_values,
    plot_error_bound_comparison,
    plot_controllability_margin,
)

__all__ = [
    # Utils
    "complex_dtype_from_real",
    "to_complex",
    "stack_z",
    "ensure_dt_vector",
    "make_stable_A",
    # Gramians
    "gramian_Sz_time",
    "integral_xxH_time",
    "integral_xdot_xH_time",
    "compute_candidate_eigenvalues",
    "gramian_Sx_from_fft",
    # Hautus
    "cross_moment_H_time",
    "cross_moment_H_fft",
    "gramian_G_from_H",
    "estimate_Hautus_matrix",
    "true_Hautus_matrix",
    "hautus_test",
    "check_controllability",
    "compare_with_true",
    # SDE
    "ControlledLinearSDE",
    "simulate_sde",
    "sdeint_safe",
    "create_time_grid",
    # Visualization
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_time_series",
    "plot_error_vs_T",
    "plot_matrix_comparison",
    "plot_singular_values",
    "plot_error_bound_comparison",
    "plot_controllability_margin",
]
