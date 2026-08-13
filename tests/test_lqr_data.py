import numpy as np

from ddinf.heat import heat_system
from ddinf.lqr_data import estimate_graph, solve_graph_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian
from ddinf.lqr_window import kernel_library, shift_library, solve_window_lqr
from ddinf.moments import hat_tests
from ddinf.signals import Prbs
from ddinf.systems import LinearSystem
from ddinf.timestepping import Record, simulate, uniform_grid


def test_data_lqr_selects_minimum_cost_trajectory_with_exact_initial_state():
    one = np.eye(1)
    sys = LinearSystem("static", np.zeros((1, 1)), one, one, one, one)
    weights = LqrWeights.make(sys, terminal=0.0, control=1.0)
    t = np.linspace(0, 1, 101)
    z1 = Record(t, np.ones((1, len(t))), np.ones((1, len(t))), np.ones((1, len(t))))
    z2 = Record(t, np.zeros((1, len(t))), 2 * np.ones((1, len(t))),
                2 * np.ones((1, len(t))))
    sol = solve_window_lqr([z1, z2], sys, weights, np.array([1.0]), rho=1e-10)
    assert sol.initial_defect < 1e-8
    assert np.isfinite(sol.cost)


def _heat_case(dt=.01, horizon=.5, length=4.0, dwell=4):
    """A real example small enough to run inside the test suite."""
    sys = heat_system("neumann", n_elems=4, nu=1.0)
    xi = sys.meta["mesh"].nodes[sys.meta["free"]]
    x0 = 1.0 + .2 * np.cos(np.pi * xi)
    t = uniform_grid(horizon, dt)
    weights = LqrWeights.make(sys, terminal=1.0, control=.5)
    ref = riccati_hamiltonian(sys, weights, t)
    record = simulate(sys, Prbs(dwell * dt, seed=3, horizon=length + dt),
                      uniform_grid(length, dt), np.zeros(sys.n), theta=.5)
    return sys, x0, weights, t, ref, record


def test_window_lqr_reproduces_the_riccati_optimum_on_the_heat_equation():
    """A sufficiently rich window library recovers the model-based optimum."""
    sys, x0, weights, t, ref, record = _heat_case()
    sol = solve_window_lqr(shift_library(record, t[-1], 2 * t.size), sys, weights,
                           x0, rho=1e-8)
    optimum = ref.optimal_cost(x0)
    assert abs(sol.cost - optimum) / abs(optimum) < 1e-4
    assert sol.initial_defect < 1e-6
    # the reconstructed signals, not merely the cost, match the optimum
    opt = ref.closed_loop(x0)
    assert (np.linalg.norm(sol.record.u[0] - opt.u[0])
            / np.linalg.norm(opt.u[0])) < 5e-2


def test_reconstructed_input_carries_no_sample_nyquist_ripple():
    """Guards the control-term quadrature of ``ddinf.moments.trapezoid_weights``.

    Weighting the control term with Simpson's rule instead makes the discrete
    optimiser split a given effective input between neighbouring samples in the
    ratio 4:2, which shows up here as an odd-even component of about 0.3.
    """
    sys, x0, weights, t, ref, record = _heat_case()
    sol = solve_window_lqr(shift_library(record, t[-1], 2 * t.size), sys, weights,
                           x0, rho=1e-8)

    def odd_even(u):
        return np.linalg.norm(u - np.convolve(u, [.25, .5, .25], "same")) \
            / np.linalg.norm(u)

    assert odd_even(sol.record.u[0]) < 2 * odd_even(ref.closed_loop(x0).u[0]) + 1e-3


def test_kernel_library_windows_are_trajectories_of_the_same_system():
    """Smoothed convolutional library from the window-informativity note.

    A convolution of trajectories is a trajectory, and the theta scheme is
    linear, so each window must be *exactly* the record the same scheme produces
    from that window's own initial state and input.
    """
    sys, x0, weights, t, ref, record = _heat_case()
    n_s = 9
    kernels = np.exp(-((np.linspace(0, 1, n_s)[None, :]
                        - np.linspace(.2, .8, 6)[:, None]) ** 2) / .02)
    lib = kernel_library(record, t[-1], kernels, theta_len=n_s * record.dt)
    assert len(lib) == 6
    for z in lib:
        again = simulate(sys, lambda tt, z=z: np.interp(tt, z.t, z.u[0])[None, :],
                         z.t, z.x[:, 0], theta=.5)
        assert np.max(abs(again.x - z.x)) < 1e-10 * max(np.max(abs(z.x)), 1.0)
        assert np.max(abs(again.y - z.y)) < 1e-10 * max(np.max(abs(z.y)), 1.0)


def test_moment_graph_accepts_an_independent_heat_trajectory():
    """The learned range, rather than a shifted library, encodes consistency."""
    sys, x0, weights, t, ref, record = _heat_case()
    graph = estimate_graph(
        record.window(0.0, t[-1]),
        hat_tests(t, 4 * (sys.m + sys.n)),
        sys.MW,
        derivative_metric=sys.MX,
        rank_tol=1e-10,
    )
    assert graph.is_full

    other = simulate(
        sys,
        lambda tt: np.sin(3.0 * np.asarray(tt))[None, :],
        t,
        x0,
        theta=.5,
    )
    u = .5 * (other.u[:, :-1] + other.u[:, 1:])
    x = .5 * (other.x[:, :-1] + other.x[:, 1:])
    dx = np.diff(other.x, axis=1) / other.dt
    y = .5 * (other.y[:, :-1] + other.y[:, 1:])
    assert graph.residual(np.vstack([u, x, dx, y])) < 1e-10


def test_graph_lqr_reproduces_the_riccati_optimum_with_exact_initial_state():
    """The synthesis-range QP is the main-paper numerical regulator."""
    sys, x0, weights, t, ref, record = _heat_case()
    graph = estimate_graph(
        record.window(0.0, t[-1]),
        hat_tests(t, 4 * (sys.m + sys.n)),
        sys.MW,
        derivative_metric=sys.MX,
        rank_tol=1e-10,
    )
    sol = solve_graph_lqr(graph, t, weights, x0)
    optimum = ref.optimal_cost(x0)
    assert abs(sol.cost - optimum) / abs(optimum) < 1e-4
    assert sol.initial_defect < 1e-10
    assert sol.graph_residual < 1e-10
    opt = ref.closed_loop(x0)
    assert (np.linalg.norm(sol.record.u[0] - opt.u[0])
            / np.linalg.norm(opt.u[0])) < 5e-2
