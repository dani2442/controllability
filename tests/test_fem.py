import numpy as np

from ddinf.systems.fem import Mesh1D, mass_matrix, point_evaluation, stiffness_matrix


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


def test_point_evaluation_is_exact_for_a_p1_function():
    mesh = Mesh1D(8)
    nodal_values = 2.0 - 3.0 * mesh.nodes
    for xi0 in (0.0, 0.17, 0.5, 0.93, 1.0):
        assert np.isclose(point_evaluation(mesh, xi0) @ nodal_values,
                          2.0 - 3.0 * xi0)
