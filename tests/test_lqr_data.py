import numpy as np

from ddinf.lqr_data import solve_data_lqr
from ddinf.lqr_model import LqrWeights
from ddinf.systems import LinearSystem
from ddinf.timestepping import Record


def test_data_lqr_selects_minimum_cost_trajectory_with_exact_initial_state():
    one = np.eye(1)
    sys = LinearSystem("static", np.zeros((1, 1)), one, one, one, one)
    weights = LqrWeights.make(sys, terminal=0.0, control=1.0)
    t = np.linspace(0, 1, 101)
    z1 = Record(t, np.ones((1, len(t))), np.ones((1, len(t))), np.ones((1, len(t))))
    z2 = Record(t, np.zeros((1, len(t))), 2 * np.ones((1, len(t))),
                2 * np.ones((1, len(t))))
    sol = solve_data_lqr([z1, z2], sys, weights, np.array([1.0]), rho=1e-10)
    assert sol.initial_defect < 1e-8
    assert np.isfinite(sol.cost)
