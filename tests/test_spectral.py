import numpy as np

from ddinf.heat import fem_eigenvalues, heat_system
from ddinf.spectral import heat_modal


def test_heat_modal_coefficients():
    modal = heat_modal("dirichlet", n_modes=4)
    n = np.arange(1, 5)
    assert np.allclose(modal.lam, -(n * np.pi) ** 2)
    assert np.allclose(modal.b[:, 0], np.sqrt(2) * n * np.pi)


def test_fem_eigenvalue_converges_quadratically():
    errors = []
    for ne in (16, 32):
        sys = heat_system("dirichlet", n_elems=ne)
        errors.append(abs(fem_eigenvalues(sys, 1)[0].real + np.pi**2))
    assert errors[1] < .27 * errors[0]


def test_symmetric_control_misses_even_modes():
    modal = heat_modal("dirichlet_sym", n_modes=8)
    assert np.array_equal(modal.uncontrollable_modes(), np.array([1, 3, 5, 7]))
