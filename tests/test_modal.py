import numpy as np

from ddinf.systems.heat import fem_eigenvalues, heat_system
from ddinf.systems.modal import heat_modal


def test_heat_modal_coefficients():
    modal = heat_modal("dirichlet", n_modes=4)
    n = np.arange(1, 5)
    assert np.allclose(modal.lam, -(n * np.pi) ** 2)
    assert np.allclose(modal.b[:, 0], np.sqrt(2) * n * np.pi)


def test_neumann_modal_matches_pritchard_salamon_example_4_6():
    """Neumann data at both ends: lambda_0=0 and lambda_n=-n^2*pi^2."""
    modal = heat_modal("neumann", n_modes=5, nu=1.0)
    n = np.arange(5)
    assert np.allclose(modal.lam, -(n * np.pi) ** 2)
    assert np.isclose(modal.b[0, 0], -1.0)
    assert np.allclose(modal.b[1:, 0], -np.sqrt(2.0))
    assert modal.lift is None


def test_fem_eigenvalue_converges_quadratically():
    errors = []
    for ne in (16, 32):
        sys = heat_system("dirichlet", n_elems=ne)
        errors.append(abs(fem_eigenvalues(sys, 1)[0].real + np.pi**2))
    assert errors[1] < .27 * errors[0]


def test_neumann_fem_agrees_with_its_modal_reference():
    for ne in (32, 64):
        sys = heat_system("neumann", n_elems=ne)
        eig = fem_eigenvalues(sys, 2)
        assert abs(eig[0].real) < 1e-10
        assert abs(eig[1].real + np.pi**2) < 9.0 / ne**2


def test_neumann_fem_has_the_exact_controlled_constant_mode():
    """Example 4.6 has phi_0=1, lambda_0=0, and b_0=-nu."""
    nu = 0.3
    sys = heat_system("neumann", n_elems=24, nu=nu)
    constant = np.ones(sys.n)
    assert np.linalg.norm(sys.A @ constant) < 1e-11
    assert np.isclose(constant @ sys.MX @ sys.B[:, 0], -nu)


def test_heat_default_output_is_point_evaluation():
    xi0 = 0.6
    sys = heat_system("neumann", n_elems=24, obs_center=xi0)
    nodal_values = 1.0 + 2.0 * sys.meta["mesh"].nodes
    assert sys.meta["observation"] == "point"
    assert np.isclose((sys.C @ nodal_values)[0], 1.0 + 2.0 * xi0)


def test_symmetric_control_misses_even_modes():
    modal = heat_modal("dirichlet_sym", n_modes=8)
    assert np.array_equal(modal.uncontrollable_modes(), np.array([1, 3, 5, 7]))
