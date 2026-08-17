import numpy as np

from ddinf.heat import heat_system
from ddinf.lqr_data import estimate_graph, solve_graph_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian
from ddinf.moments import hat_tests
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid


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
