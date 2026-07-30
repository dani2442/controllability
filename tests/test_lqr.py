import numpy as np

from ddinf.lqr_model import LqrWeights, riccati_hamiltonian, riccati_ivp, trajectory_cost
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


def test_optimal_trajectory_has_riccati_cost():
    sys = scalar_system()
    weights = LqrWeights.make(sys, terminal=.3, control=.7)
    t = np.linspace(0, .5, 1001)
    sol = riccati_hamiltonian(sys, weights, t)
    rec = sol.closed_loop(np.array([1.0]))
    assert abs(trajectory_cost(rec, sys, weights) - sol.optimal_cost(np.array([1.0]))) < 1e-7
