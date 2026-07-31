import numpy as np

from ddinf.heat import fem_eigenvalues, heat_system
from ddinf.spectral import heat_modal


def test_heat_modal_coefficients():
    modal = heat_modal("dirichlet", n_modes=4)
    n = np.arange(1, 5)
    assert np.allclose(modal.lam, -(n * np.pi) ** 2)
    assert np.allclose(modal.b[:, 0], np.sqrt(2) * n * np.pi)


def test_neumann_modal_matches_pritchard_salamon_example_4_6():
    """Neumann control at 0, Dirichlet at 1: lambda_n = -(n-1/2)^2 pi^2."""
    modal = heat_modal("neumann", n_modes=5, nu=1.0)
    a = (np.arange(1, 6) - 0.5) * np.pi
    assert np.allclose(modal.lam, -a**2)
    # b_n = -nu phi_n(0) = -sqrt2 for every mode: no unreachable mode.
    assert np.allclose(modal.b[:, 0], -np.sqrt(2.0))
    # The lift D(xi) = xi - 1 realises the boundary condition it is meant to.
    assert np.isclose(modal.lift(np.array([1.0]))[0], 0.0)
    assert np.allclose(modal.d, -np.sqrt(2.0) / a**2)


def test_fem_eigenvalue_converges_quadratically():
    errors = []
    for ne in (16, 32):
        sys = heat_system("dirichlet", n_elems=ne)
        errors.append(abs(fem_eigenvalues(sys, 1)[0].real + np.pi**2))
    assert errors[1] < .27 * errors[0]


def test_neumann_fem_agrees_with_its_modal_reference():
    for ne in (32, 64):
        sys = heat_system("neumann", n_elems=ne)
        assert abs(fem_eigenvalues(sys, 1)[0].real + 0.25 * np.pi**2) < 6.0 / ne**2


def test_symmetric_control_misses_even_modes():
    modal = heat_modal("dirichlet_sym", n_modes=8)
    assert np.array_equal(modal.uncontrollable_modes(), np.array([1, 3, 5, 7]))
