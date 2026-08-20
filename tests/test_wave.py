import numpy as np

from ddinf.controllability.state import hautus_uncontrollable
from ddinf.systems.wave import exact_frequencies, fem_frequencies, wave_system


def test_frequencies_converge_quadratically():
    errors = []
    for ne in (16, 32):
        sys = wave_system("dirichlet", n_elems=ne)
        errors.append(abs(fem_frequencies(sys, 1)[0] - exact_frequencies(1)[0]))
    assert errors[1] < .3 * errors[0]


def test_generator_spectrum_is_purely_imaginary():
    sys = wave_system("dirichlet", n_elems=12)
    assert np.max(np.abs(np.linalg.eigvals(sys.A).real)) < 1e-9


def test_symmetric_control_misses_even_modes():
    """z(t,0) = z(t,1) = u(t) cannot reach modes antisymmetric about xi = 1/2."""
    sys = wave_system("dirichlet_sym", n_elems=12, speed=1.0)
    bad = np.sort(np.abs(np.array([lam for lam, _ in hautus_uncontrollable(sys)]).imag))
    freqs = fem_frequencies(sys)
    assert np.allclose(bad[::2], freqs[1::2])  # exactly the even modes
    assert not hautus_uncontrollable(wave_system("dirichlet", n_elems=12))


def test_energy_and_state_inner_products_are_positive_definite():
    sys = wave_system("dirichlet", n_elems=10)
    for G in (sys.MX, sys.MW):
        assert np.min(np.linalg.eigvalsh(0.5 * (G + G.T))) > 0
