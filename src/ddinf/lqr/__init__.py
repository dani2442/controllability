"""The finite-horizon regulator: one reference and two data-driven realizations.

:mod:`ddinf.lqr.riccati` is the model-based optimum every data-driven result is
scored against.  :mod:`ddinf.lqr.graph` learns the system graph from weak
moments of an input--state--output record; :mod:`ddinf.lqr.window` spans the
finite-horizon behavior with shifted input--output windows and never sees a
state.
"""

from .graph import GraphBasis, estimate_graph, solve_graph_lqr
from .riccati import LqrWeights, riccati_hamiltonian, trajectory_cost
from .window import (BehaviorBasis, behavior_basis, io_shift_library,
                     solve_io_lqr)

__all__ = ["BehaviorBasis", "GraphBasis", "LqrWeights", "behavior_basis",
           "estimate_graph", "io_shift_library", "riccati_hamiltonian",
           "solve_graph_lqr", "solve_io_lqr", "trajectory_cost"]
