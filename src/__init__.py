"""Continuous-time informativity framework.

Implements the main results of `paper_informativity/main.tex`:
weak-derivative lifts, identification, data-driven Hautus tests,
state-feedback stabilization (exact & noisy), LQR, and dissipativity
(exact & noisy). See the per-module docstrings for which theorem is
being implemented.
"""

from .bases import (
    BumpBasis,
    FourierSineBasis,
    HatBasis,
    Quadrature,
    TestFunctionBasis,
    pair,
)
from .controllability import (
    HautusResult,
    finite_candidates_io,
    finite_candidates_state,
    io_hautus_test,
    state_hautus_test,
)
from .dissipativity import (
    DissipativityResult,
    dissipativity_exact,
    dissipativity_noisy,
)
from .identification import (
    IdentificationResult,
    check_identifiability,
    identify_state_data,
    identify_state_output_data,
)
from .lifts import (
    LiftData,
    compute_filtered_io_lift,
    compute_filtered_state_lift,
    compute_io_lifts,
    compute_state_lifts,
)
from .lqr import (
    LQRResult,
    is_detectable,
    is_stabilizable,
    solve_lqr_exact,
)
from .persistent_excitation import (
    PersistentExcitationResult,
    check_persistent_excitation,
)
from .qmi import (
    InducedQMI,
    ball_Pi,
    build_induced,
    induced_qmi_matrix,
    is_in_Pi_qN,
    make_exact_Pi,
    schur_complement_22,
)
from .sde import (
    LinearSDE,
    TrajectoryData,
    make_multisine_control,
    simulate,
)
from .stabilization import (
    StabilizationResult,
    build_Nx_phi,
    stabilize_exact,
    stabilize_noisy,
)
from .systems import (
    coupled_spring,
    damped_oscillator,
    double_integrator,
    random_stable,
    unstable_inverted_pendulum,
)

__all__ = [
    # Bases
    "TestFunctionBasis", "FourierSineBasis", "BumpBasis", "HatBasis",
    "Quadrature", "pair",
    # Systems
    "double_integrator", "damped_oscillator", "coupled_spring",
    "random_stable", "unstable_inverted_pendulum",
    # SDE
    "LinearSDE", "TrajectoryData", "simulate", "make_multisine_control",
    # Lifts
    "LiftData", "compute_state_lifts", "compute_io_lifts",
    "compute_filtered_state_lift", "compute_filtered_io_lift",
    # QMI
    "InducedQMI", "schur_complement_22", "is_in_Pi_qN", "induced_qmi_matrix",
    "make_exact_Pi", "ball_Pi", "build_induced",
    # PE
    "PersistentExcitationResult", "check_persistent_excitation",
    # Identification
    "IdentificationResult", "check_identifiability", "identify_state_data",
    "identify_state_output_data",
    # Controllability
    "HautusResult", "finite_candidates_state", "finite_candidates_io",
    "state_hautus_test", "io_hautus_test",
    # Stabilization
    "StabilizationResult", "stabilize_exact", "stabilize_noisy",
    "build_Nx_phi",
    # LQR
    "LQRResult", "solve_lqr_exact", "is_stabilizable", "is_detectable",
    # Dissipativity
    "DissipativityResult", "dissipativity_exact", "dissipativity_noisy",
]
