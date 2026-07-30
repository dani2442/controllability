import numpy as np

from ddinf.fem import Mesh1D, mass_matrix, stiffness_matrix


def test_p1_matrices_have_exact_integrals():
    mesh = Mesh1D(4)
    M = mass_matrix(mesh)
    K = stiffness_matrix(mesh)
    ones = np.ones(mesh.n_nodes)
    assert np.isclose(ones @ M @ ones, 1.0)
    assert np.isclose(ones @ K @ ones, 0.0)
    assert np.allclose(M, M.T)
    assert np.allclose(K, K.T)


def test_lumped_mass_preserves_total_mass():
    mesh = Mesh1D(9)
    assert np.isclose(np.trace(mass_matrix(mesh, lumped=True)), 1.0)
