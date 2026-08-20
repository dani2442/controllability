import numpy as np

from ddinf.data.moments import moments, sine_tests, theta_moments
from ddinf.systems import LinearSystem
from ddinf.data.records import Record


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


def test_theta_moments_match_the_discrete_dynamics_to_roundoff():
    """The graph estimator uses the weak form consistent with time stepping."""
    t = np.linspace(0, 1, 101)
    A = np.array([[-2.0]])
    B = np.array([[1.0]])
    sys = LinearSystem("scalar", A, B, np.eye(1), np.eye(1), np.eye(1))
    from ddinf.data.records import simulate

    rec = simulate(sys, lambda tt: np.cos(np.asarray(tt))[None, :], t,
                   np.array([.3]), theta=.5)
    mom = theta_moments(rec, sine_tests(t, 12), theta=.5)
    assert mom.dynamics_residual(A, B) < 1e-12
