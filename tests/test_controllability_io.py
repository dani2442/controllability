"""The windowed input--output controllability test of ``thm:io-window-controllability``.

The reference is always the model-based Fattorini--Hautus condition, which none
of the tested routines ever sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddinf.controllability_io import (io_shift_windows, io_window_controllability,
                                      _ExponentialFit)
from ddinf.signals import Prbs
from ddinf.systems import LinearSystem
from ddinf.timestepping import simulate, uniform_grid

DT = .02
WINDOW = 2.0
SHIFTS = 40.0


def _prbs(length: float, amplitude: float = 3.0, seed: int = 3):
    signal = Prbs(2 * DT, seed=seed, horizon=length + DT)
    return lambda t: amplitude * signal(t)


def _modal(lams: np.ndarray, gains: np.ndarray, observed: np.ndarray
           ) -> LinearSystem:
    """Diagonal realisation; ``gains[k] == 0`` makes mode ``k`` unreachable."""
    n = lams.size
    return LinearSystem(name="modal", A=np.diag(lams), B=gains.reshape(n, 1),
                        C=observed.reshape(1, n), MX=np.eye(n), MW=np.eye(n))


def _run(sys: LinearSystem, x0: np.ndarray, *, shifts: float = SHIFTS, **kwargs):
    length = shifts + WINDOW
    record = simulate(sys, _prbs(length), uniform_grid(length, DT), x0, theta=.5)
    windows = io_shift_windows(record, horizon=WINDOW, spread=shifts)
    return io_window_controllability(windows, n_states=sys.n, **kwargs)


# ------------------------------------------------------------------ windows


def test_windows_reproduce_the_l2_pairing():
    """``theta' w(s)`` is the quadrature of ``int v u + int g y`` on the window."""
    sys = _modal(np.array([-.3, -1.1]), np.array([1., 1.]), np.array([1., .7]))
    length = 6.0
    record = simulate(sys, _prbs(length), uniform_grid(length, DT),
                      np.array([1., .5]), theta=.5)
    windows = io_shift_windows(record, horizon=2.0, spread=2.0)

    rng = np.random.default_rng(0)
    theta = rng.normal(size=windows.data.shape[0])
    v, g = windows.kernels(theta)
    values = windows.evaluate(theta)

    shift = 37  # an arbitrary shift, in samples
    piece = slice(shift, shift + windows.n_w)
    expected = np.trapezoid(v[0] * record.u[0, piece], windows.t) \
        + np.trapezoid(g[0] * record.y[0, piece], windows.t)
    assert values[shift] == pytest.approx(expected, rel=1e-12)
    # The weighting is chosen so that the coefficient norm is the L^2 norm of
    # the kernel pair; everything downstream measures functionals with it.
    kernel_norm = np.sqrt(np.trapezoid(v[0] ** 2 + g[0] ** 2, windows.t))
    assert np.linalg.norm(theta) == pytest.approx(kernel_norm, rel=1e-12)


def test_shift_count_is_capped_by_the_record():
    sys = _modal(np.array([-.5]), np.array([1.]), np.array([1.]))
    record = simulate(sys, _prbs(4.0), uniform_grid(4.0, DT), np.array([1.]),
                      theta=.5)
    assert io_shift_windows(record, horizon=1.0).n_s == record.n_samples - 50
    with pytest.raises(ValueError, match="at most"):
        io_shift_windows(record, horizon=1.0, spread=9.0)


def test_resolved_rank_stops_at_the_behaviour_dimension():
    """The shifted windows span ``m n_w + n`` directions and no more."""
    sys = _modal(np.array([-.4, -1.3, -2.2]), np.ones(3), np.array([1., .8, .6]))
    report = _run(sys, np.array([1., .5, .3]))
    assert report.numerical_rank <= report.dimension
    assert report.numerical_rank >= report.dimension - 5
    # Below the cliff the spectrum is round-off, not signal.
    assert report.singular_values[report.dimension] < 1e-25 * report.singular_values[0]


# ------------------------------------------------------------------ the test


def test_reachable_modes_are_not_obstructions():
    sys = _modal(np.array([-.35, -1.2, -2.4]), np.ones(3), np.array([1., .8, .6]))
    report = _run(sys, np.array([1., .5, .3]))
    assert report.approximately_controllable
    assert report.obstructions == []


def test_unreachable_mode_is_recovered():
    """``gains[1] = 0`` leaves ``lambda = -1.2`` uncontrollable, and visible in ``y``."""
    lams = np.array([-.35, -1.2, -2.4])
    sys = _modal(lams, np.array([1., 0., 1.]), np.array([1., .8, .6]))
    report = _run(sys, np.array([1., .5, .3]))

    assert not report.approximately_controllable
    found = [c.lam for c in report.obstructions]
    assert len(found) == 1
    assert found[0].real == pytest.approx(-1.2, abs=1e-3)
    assert found[0].imag == pytest.approx(0.0, abs=1e-6)


def test_the_recovered_kernels_satisfy_the_predicate():
    """``theta`` comes back in window coordinates and reproduces ``kappa e^{lambda s}``.

    The whole point of the test is that the certificate is a functional of the
    measured window, so it has to be usable as one: the returned coefficients
    must reshape into a kernel pair on ``[0, T]`` and pairing them against the
    record must give the exponential the fit claimed.
    """
    lams = np.array([-.35, -1.2, -2.4])
    sys = _modal(lams, np.array([1., 0., 1.]), np.array([1., .8, .6]))
    length = SHIFTS + WINDOW
    record = simulate(sys, _prbs(length), uniform_grid(length, DT),
                      np.array([1., .5, .3]), theta=.5)
    windows = io_shift_windows(record, horizon=WINDOW, spread=SHIFTS)
    report = io_window_controllability(windows, n_states=sys.n)

    candidate = report.obstructions[0]
    assert candidate.theta.shape == (windows.data.shape[0],)
    v, g = candidate.kernels(windows)
    assert v.shape == (windows.m, windows.n_w)
    assert g.shape == (windows.p, windows.n_w)

    values = windows.evaluate(candidate.theta)
    reference = np.exp(candidate.lam * windows.s)
    scale = np.vdot(reference, values) / np.vdot(reference, reference)
    weights = np.sqrt(np.abs(np.gradient(windows.s)))
    relative = (np.linalg.norm((values - scale * reference) * weights)
                / np.linalg.norm(values * weights))
    assert relative < 1e-2
    assert abs(scale) > 0  # the kappa != 0 clause, built into the normalisation
    assert candidate.functional_norm == pytest.approx(
        np.linalg.norm(candidate.theta), rel=1e-12)


def test_unreachable_mode_absent_from_the_output_is_invisible():
    """``eta`` outside the range of ``O_T^*`` cannot be reached by any window.

    The theorem assumes exact observability for exactly this reason: a mode the
    output does not see is not an obstruction the input--output test can
    produce, even though the state test would return it.
    """
    lams = np.array([-.35, -1.2, -2.4])
    sys = _modal(lams, np.array([1., 0., 1.]), np.array([1., 0., .6]))
    report = _run(sys, np.array([1., .5, .3]))
    assert report.approximately_controllable


def test_conjugate_pair_is_returned_as_a_pair():
    """A real system with a complex unreachable mode yields both of them."""
    A = np.array([[-.5, 2.0, 0.], [-2.0, -.5, 0.], [0., 0., -1.5]])
    sys = LinearSystem(name="rotating", A=A, B=np.array([[0.], [0.], [1.]]),
                       C=np.array([[1., .3, .7]]), MX=np.eye(3), MW=np.eye(3))
    report = _run(sys, np.array([1., .4, .6]))

    found = sorted((c.lam for c in report.obstructions), key=lambda z: z.imag)
    assert len(found) == 2
    assert found[0] == pytest.approx(np.conj(found[1]), abs=1e-6)
    assert found[1].real == pytest.approx(-.5, abs=1e-3)
    assert abs(found[1].imag) == pytest.approx(2.0, abs=1e-3)


def test_a_multisine_record_fabricates_obstructions_at_its_own_lines():
    """Output-window informativity is a hypothesis, not a formality.

    A length-``T`` kernel can null every line of a narrow multisine but one, so
    the window pairing is itself a pure exponential and the test reports the
    line frequency as an obstruction of a controllable system.  With so few
    lines the individual modal transients separate as well, and those come back
    as obstructions too.
    """
    from ddinf.signals import multisine

    lams = np.array([-.35, -1.2, -2.4])
    sys = _modal(lams, np.ones(3), np.array([1., .8, .6]))
    length = SHIFTS + WINDOW
    lines = np.array([1.3, 2.9, 5.1])
    record = simulate(sys, multisine(lines, seed=1), uniform_grid(length, DT),
                      np.array([1., .5, .3]), theta=.5)
    windows = io_shift_windows(record, horizon=WINDOW, spread=SHIFTS)
    report = io_window_controllability(windows, n_states=sys.n)

    # Every mode of this system is reachable, so the truth is "no obstruction".
    assert not report.approximately_controllable
    assert report.numerical_rank < report.dimension // 4
    found = [c.lam for c in report.obstructions]
    for line in lines:
        assert any(abs(abs(lam.imag) - line) < 1e-2 and abs(lam.real) < 1e-2
                   for lam in found), f"line {line} not reported"


# ------------------------------------------------------- the exponential fit


def test_residual_is_scale_free_and_finite_far_from_the_data():
    sys = _modal(np.array([-.4, -1.3]), np.ones(2), np.array([1., .7]))
    length = SHIFTS + WINDOW
    record = simulate(sys, _prbs(length), uniform_grid(length, DT),
                      np.array([1., .5]), theta=.5)
    windows = io_shift_windows(record, horizon=WINDOW, spread=SHIFTS)
    fit = _ExponentialFit(windows, windows.data)

    # A growing exponential must not overflow into a nan and poison the
    # ordering of the candidates.
    for lam in (5.0, 50.0, 500.0, -500.0):
        assert 0.0 <= fit.residual(lam) <= 1.0
    # Past the Nyquist rate of the shift grid the exponential aliases, and the
    # fit refuses to report a distance at all.
    assert fit.residual(1j * (windows.nyquist + 1.0)) == 1.0
