import numpy as np

from ddinf.lqr.riccati import LqrWeights, riccati_hamiltonian, riccati_ivp, trajectory_cost
from ddinf.systems import LinearSystem


def scalar_system():
    one = np.eye(1)
    return LinearSystem("scalar", np.array([[-1.0]]), one, one, one, one)


def test_hamiltonian_and_ivp_riccati_agree():
    sys = scalar_system()
    weights = LqrWeights.make(sys, terminal=.3, control=.7)
    t = np.linspace(0, .5, 101)
    a = riccati_hamiltonian(sys, weights, t)
    b = riccati_ivp(sys, weights, t)
    assert np.max(abs(a.P - b.P)) < 1e-9


def test_riccati_survives_a_stiff_long_horizon():
    """The Riccati transform must be restarted each step, not accumulated.

    Propagating ``exp(-H s)`` over the whole grid overflows once
    ``s |lambda_max(A)|`` is large and ``Phi11 + Phi12 G`` turns singular; the
    semi-discrete heat operator reaches that regime on any useful horizon.
    """
    from ddinf.systems.heat import heat_system

    sys = heat_system("neumann", n_elems=8, nu=1.0)
    weights = LqrWeights.make(sys, terminal=1.0, control=.5)
    t = np.linspace(0, 4.0, 401)
    a = riccati_hamiltonian(sys, weights, t)
    assert np.all(np.isfinite(a.P))
    b = riccati_ivp(sys, weights, t)
    assert np.max(abs(a.P - b.P)) < 1e-8 * max(np.max(abs(b.P)), 1.0)


def test_optimal_trajectory_has_riccati_cost():
    sys = scalar_system()
    weights = LqrWeights.make(sys, terminal=.3, control=.7)
    t = np.linspace(0, .5, 1001)
    sol = riccati_hamiltonian(sys, weights, t)
    rec = sol.closed_loop(np.array([1.0]))
    assert abs(trajectory_cost(rec, sys, weights) - sol.optimal_cost(np.array([1.0]))) < 1e-7
