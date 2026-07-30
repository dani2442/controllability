import numpy as np

from ddinf.moments import moments, sine_tests
from ddinf.systems import LinearSystem
from ddinf.timestepping import Record


def test_weak_derivative_moment_for_exact_exponential():
    t = np.linspace(0, 1, 2001)
    lam = -2.0
    x = np.exp(lam * t)[None, :]
    rec = Record(t, np.zeros((1, t.size)), x, x)
    mom = moments(rec, sine_tests(t, 5))
    assert np.linalg.norm(mom.X1 - lam * mom.X0) / np.linalg.norm(mom.X1) < 1e-8


def test_dynamics_identity_for_exact_record():
    t = np.linspace(0, 1, 2001)
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    u = np.ones((1, t.size))
    x = (1 - np.exp(-t))[None, :]
    sys = LinearSystem("scalar", A, B, np.eye(1), np.eye(1), np.eye(1))
    rec = Record(t, u, x, x)
    mom = moments(rec, sine_tests(t, 5))
    assert mom.dynamics_residual(sys.A, sys.B) < 1e-8
