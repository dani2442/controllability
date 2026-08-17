"""The input--output windowed lemma and the regulator built on it."""

import numpy as np
import pytest

from ddinf.heat import heat_system
from ddinf.lqr_io import behaviour_basis, io_shift_library, solve_io_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian, trajectory_cost
from ddinf.moments import trapezoid_weights
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system

PAST, HORIZON, DT, LENGTH, DWELL = .25, .5, .01, 6.0, 4


def _heat_case():
    """A real example small enough to run inside the test suite."""
    sys = heat_system("neumann", n_elems=4, nu=1.0)
    xi = sys.meta["mesh"].nodes[sys.meta["free"]]
    weights = LqrWeights.make(sys, terminal=0.0, control=.05)
    t = uniform_grid(HORIZON, DT)

    conditioning = simulate(sys, Prbs(DWELL * DT, seed=11, horizon=PAST + DT),
                            uniform_grid(PAST, DT), 1.0 + .2 * np.cos(np.pi * xi),
                            theta=.5)
    x0 = conditioning.x[:, -1]
    record = simulate(sys, Prbs(DWELL * DT, seed=3, horizon=LENGTH + DT),
                      uniform_grid(LENGTH, DT), np.zeros(sys.n), theta=.5)
    return sys, weights, t, conditioning, x0, record


def _library(record, size):
    return io_shift_library(record, past=PAST, horizon=HORIZON, n_windows=size)


def _behaviour_dimension(sys):
    """``m n_w + n``: the dimension of the sampled behaviour on the window."""
    return sys.m * (int(round((PAST + HORIZON) / DT)) + 1) + sys.n


def _flat(library, weights):
    """The library as rows of an ``L^2(-T_ini, T)`` Gram factor."""
    root = np.sqrt(weights)
    return np.concatenate([(library.u * root).reshape(library.n_windows, -1),
                           (library.y * root).reshape(library.n_windows, -1)],
                          axis=1)


def _span_residual(sys, record, size):
    """Relative distance of an independent trajectory to the library span."""
    basis = behaviour_basis(_library(record, size))
    rng = np.random.default_rng(5)
    tt = basis.windows.t
    u = np.stack([np.sin(2.3 * tt) + .4 * rng.standard_normal(tt.size)
                  for _ in range(sys.m)])
    fresh = simulate(
        sys,
        lambda s: np.stack([np.interp(np.asarray(s), tt, ui) for ui in u]),
        tt, rng.standard_normal(sys.n), theta=.5,
    )

    w = trapezoid_weights(tt)
    rows = _flat(basis.windows, w)  # orthonormal in the window metric
    target = np.concatenate([(fresh.u * np.sqrt(w)).ravel(),
                             (fresh.y * np.sqrt(w)).ravel()])
    residual = np.linalg.norm(rows.T @ (rows @ target) - target)
    return residual / np.linalg.norm(target), basis


def test_shifted_windows_span_the_sampled_behaviour():
    """The discrete form of ``eq:windowed-io-fundamental-lemma``.

    Enough shifts of one record span every input--output window the system can
    produce, so an independent trajectory -- a fresh initial state and a fresh
    input -- lies in their span to solver precision.  This is the property the
    regulator rests on, and it is not implied by the nominal library size.
    """
    sys = wave_system("dirichlet", n_elems=3, speed=.5)
    record = simulate(sys, Prbs(DWELL * DT, seed=3, horizon=LENGTH + DT),
                      uniform_grid(LENGTH, DT), np.zeros(sys.n), theta=.5)
    residual, basis = _span_residual(sys, record, 2 * _behaviour_dimension(sys))
    assert basis.rank == _behaviour_dimension(sys)
    assert residual < 1e-9


def test_a_smoothing_semigroup_loses_a_direction_whatever_the_library_size():
    """The closure form is not a formality: the heat record never spans.

    The fastest mode of the semi-discrete heat operator has decayed below
    rounding by the end of every window, so one direction of the behaviour is
    missing from the span at any number of shifts -- and stays missing when the
    shifts are doubled, which a rank shortfall from too few windows would not.
    """
    sys, _, _, _, _, record = _heat_case()
    dimension = _behaviour_dimension(sys)
    residual, basis = _span_residual(sys, record, 2 * dimension)
    doubled, doubled_basis = _span_residual(sys, record, 4 * dimension)

    assert basis.rank == dimension - 1 == doubled_basis.rank
    assert 1e-9 < residual < 1e-2
    assert abs(doubled - residual) < .5 * residual


def test_behaviour_basis_is_orthonormal_and_bounded_by_the_behaviour_dimension():
    sys, _, _, _, _, record = _heat_case()
    library = _library(record, 400)
    basis = behaviour_basis(library)

    rows = _flat(basis.windows, trapezoid_weights(basis.windows.t))
    assert np.max(abs(rows @ rows.T - np.eye(basis.rank))) < 1e-8
    assert basis.rank <= library.behaviour_dimension(sys.n)
    assert basis.spectrum[basis.rank - 1] > basis.rank_tol * basis.spectrum[0]


def test_io_lqr_reproduces_the_riccati_optimum_and_is_a_true_trajectory():
    """The regulator sees no state, only the past and the windows.

    Its accuracy is limited by the sampled junction between the conditioning
    window and the control horizon, hence the looser tolerance than the graph
    regulator; but the cost it predicts must be the cost the plant really pays,
    which is what makes the reconstruction a trajectory rather than a fit.
    """
    sys, weights, t, conditioning, x0, record = _heat_case()
    library = _library(record, 2 * _behaviour_dimension(sys))
    solution = solve_io_lqr(behaviour_basis(library), weights,
                            conditioning.u, conditioning.y, rho=1e-8)

    optimum = riccati_hamiltonian(sys, weights, t).optimal_cost(x0)
    assert abs(solution.cost - optimum) / optimum < 2e-2

    replay = simulate(sys, solution.input_callable(), t, x0, theta=.5)
    paid = trajectory_cost(replay, sys, weights)
    assert abs(paid - solution.cost) / optimum < 1e-3
    assert solution.past_defect < 1e-5


def test_reconstructed_input_carries_no_sample_nyquist_ripple():
    """Guards the control-term quadrature of ``ddinf.moments.trapezoid_weights``.

    Weighting the control term with Simpson's rule instead makes the discrete
    optimiser split a given effective input between neighbouring samples in the
    ratio 4:2, which shows up here as an odd-even component of the input.
    """
    sys, weights, t, conditioning, x0, record = _heat_case()
    library = _library(record, 2 * _behaviour_dimension(sys))
    solution = solve_io_lqr(behaviour_basis(library), weights,
                            conditioning.u, conditioning.y, rho=1e-8)

    def odd_even(u):
        return np.linalg.norm(u - np.convolve(u, [.25, .5, .25], "same")) \
            / np.linalg.norm(u)

    optimal = riccati_hamiltonian(sys, weights, t).closed_loop(x0)
    assert odd_even(solution.u[0]) < 2 * odd_even(optimal.u[0]) + 1e-3


def test_a_terminal_state_weight_is_refused():
    """``<x(T), G x(T)>`` is not a functional of the measured input and output."""
    sys, _, _, conditioning, _, record = _heat_case()
    library = _library(record, 200)
    weights = LqrWeights.make(sys, terminal=1.0, control=.05)
    with pytest.raises(ValueError, match="terminal"):
        solve_io_lqr(behaviour_basis(library), weights,
                     conditioning.u, conditioning.y)
