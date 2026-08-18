"""Graph discretisations of the data-driven finite-horizon LQR.

The graph can be learned either from pointwise theta-method stages or from weak
moments.  In the latter, derivative-free construction the data satisfy

    D = [U0; X0; X1; Y0] = Gamma_h [U0; X0] = Gamma_h Z.

If ``Z`` has full row rank, the columns of ``D`` span the finite-dimensional
system graph ``im Gamma_h``.  A metric-weighted SVD provides a stable basis of
that graph without identifying ``A``, ``B`` or ``C``.  Candidate trajectories
are then constrained, interval by interval, to this learned graph and the LQR
cost is minimized by a sparse equality-constrained quadratic solve.

The input--output counterpart, which reads no state at all, is
``ddinf.lqr_io``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import bmat, block_diag as sparse_block_diag, lil_matrix
from scipy.sparse.linalg import spsolve

from .lqr_model import LqrWeights
from .moments import Moments, TestFunctions, theta_moments
from .timestepping import Record


def _metric_root(metric: np.ndarray) -> np.ndarray:
    """Return ``R`` such that ``||R z||_2^2 = z.T @ metric @ z``."""
    metric = np.asarray(metric, dtype=float)
    return np.linalg.cholesky(0.5 * (metric + metric.T)).T


def _positive_metric(metric: np.ndarray | None, dimension: int) -> np.ndarray:
    if metric is None:
        return np.eye(dimension)
    metric = np.asarray(metric, dtype=float)
    if metric.shape != (dimension, dimension):
        raise ValueError(f"metric must have shape {(dimension, dimension)}")
    return metric


def _resolved_graph(
    Z: np.ndarray,
    D: np.ndarray,
    *,
    input_metric: np.ndarray,
    state_metric: np.ndarray,
    derivative_metric: np.ndarray,
    output_metric: np.ndarray,
    rank_tol: float,
    moments: Moments | None,
) -> "GraphBasis":
    """Build a metric-orthogonal graph basis from ``D = Gamma_h Z``."""
    domain_root = block_diag(
        _metric_root(input_metric), _metric_root(state_metric)
    )
    ambient_root = block_diag(
        _metric_root(input_metric),
        _metric_root(state_metric),
        _metric_root(derivative_metric),
        _metric_root(output_metric),
    )

    _, singular_values, Vt = np.linalg.svd(domain_root @ Z, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > rank_tol * singular_values[0]))
    if rank == 0:
        raise ValueError("the record has zero resolved input-state rank")

    # Dividing by Sigma makes the first two blocks a metric-orthonormal basis
    # of the resolved input-state directions before they are lifted by Gamma.
    trial = D @ (Vt[:rank].T / singular_values[:rank])
    scaled_basis, _ = np.linalg.qr(ambient_root @ trial, mode="reduced")
    # Complete QR supplies an orthonormal basis of the ambient complement.
    complete, _ = np.linalg.qr(scaled_basis, mode="complete")
    complement = complete[:, rank:]
    constraint = complement.T @ ambient_root

    return GraphBasis(
        basis=trial,
        constraint=constraint,
        singular_values=singular_values,
        rank=rank,
        rank_tol=rank_tol,
        moments=moments,
        input_metric=input_metric,
        state_metric=state_metric,
        derivative_metric=derivative_metric,
        output_metric=output_metric,
    )


@dataclass
class GraphBasis:
    """Resolved finite-dimensional approximation of ``im Gamma``.

    ``constraint`` is a row matrix ``K`` satisfying ``K d = 0`` precisely for
    vectors in the resolved graph.  Its columns are ordered as
    ``d = (u, x, xdot, y)``.
    """

    basis: np.ndarray
    constraint: np.ndarray
    singular_values: np.ndarray
    rank: int
    rank_tol: float
    moments: Moments | None
    input_metric: np.ndarray
    state_metric: np.ndarray
    derivative_metric: np.ndarray
    output_metric: np.ndarray

    @property
    def m(self) -> int:
        return self.input_metric.shape[0]

    @property
    def n(self) -> int:
        return self.state_metric.shape[0]

    @property
    def p(self) -> int:
        return self.output_metric.shape[0]

    @property
    def domain_dimension(self) -> int:
        return self.m + self.n

    @property
    def ambient_dimension(self) -> int:
        return self.m + 2 * self.n + self.p

    @property
    def is_full(self) -> bool:
        return self.rank == self.domain_dimension

    @property
    def smallest_resolved(self) -> float:
        return float(self.singular_values[self.rank - 1]) if self.rank else 0.0

    def residual(self, data: np.ndarray) -> float:
        """Relative distance of one or more ``(u,x,xdot,y)`` columns to the graph."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        root = block_diag(
            _metric_root(self.input_metric),
            _metric_root(self.state_metric),
            _metric_root(self.derivative_metric),
            _metric_root(self.output_metric),
        )
        return float(np.linalg.norm(self.constraint @ data)
                     / max(np.linalg.norm(root @ data), 1e-300))


def estimate_graph(record: Record, tests: TestFunctions, state_metric: np.ndarray, *,
                   derivative_metric: np.ndarray | None = None,
                   input_metric: np.ndarray | None = None,
                   output_metric: np.ndarray | None = None,
                   rank_tol: float = 1e-10, theta: float = 0.5) -> GraphBasis:
    """Estimate the resolved system graph from one record's weak moments.

    ``rank_tol`` is relative to the largest singular value of the moment map
    ``Z = [U0; X0]``.  The derivative metric defaults to the state metric; this
    changes conditioning but not the exact finite-dimensional graph.
    """
    mom = theta_moments(record, tests, theta=theta)
    m, n, p = mom.U0.shape[0], mom.X0.shape[0], mom.Y0.shape[0]
    input_metric = _positive_metric(input_metric, m)
    state_metric = _positive_metric(state_metric, n)
    derivative_metric = _positive_metric(
        state_metric if derivative_metric is None else derivative_metric, n
    )
    output_metric = _positive_metric(output_metric, p)

    return _resolved_graph(
        np.vstack([mom.U0, mom.X0]),
        np.vstack([mom.U0, mom.X0, mom.X1, mom.Y0]),
        input_metric=input_metric,
        state_metric=state_metric,
        derivative_metric=derivative_metric,
        output_metric=output_metric,
        moments=mom,
        rank_tol=rank_tol,
    )


def estimate_sampled_graph(
    record: Record,
    state_metric: np.ndarray,
    *,
    derivative_metric: np.ndarray | None = None,
    input_metric: np.ndarray | None = None,
    output_metric: np.ndarray | None = None,
    rank_tol: float = 1e-10,
    theta: float = 0.5,
) -> GraphBasis:
    """Estimate the graph directly from sampled theta-method stage data.

    On interval ``k`` the estimator uses the column

    ``(u_theta, x_theta, (x[k+1]-x[k])/dt, y_theta)``.

    This is the direct finite-dimensional realization of the pointwise graph
    theorem.  Unlike :func:`estimate_graph`, it forms a difference quotient of
    the measured state, so the weak-moment estimator remains preferable for
    noisy data.  A uniform grid is required to match :func:`solve_graph_lqr`.
    """
    steps = np.diff(record.t)
    if steps.size == 0 or not np.allclose(steps, steps[0]):
        raise ValueError(
            "the sampled graph currently requires a uniform time grid"
        )

    m, n, p = record.u.shape[0], record.x.shape[0], record.y.shape[0]
    input_metric = _positive_metric(input_metric, m)
    state_metric = _positive_metric(state_metric, n)
    derivative_metric = _positive_metric(
        state_metric if derivative_metric is None else derivative_metric, n
    )
    output_metric = _positive_metric(output_metric, p)

    def stage(values: np.ndarray) -> np.ndarray:
        return (1.0 - theta) * values[:, :-1] + theta * values[:, 1:]

    U = stage(record.u)
    X = stage(record.x)
    Xdot = np.diff(record.x, axis=1) / float(steps[0])
    Y = stage(record.y)
    return _resolved_graph(
        np.vstack([U, X]),
        np.vstack([U, X, Xdot, Y]),
        input_metric=input_metric,
        state_metric=state_metric,
        derivative_metric=derivative_metric,
        output_metric=output_metric,
        moments=None,
        rank_tol=rank_tol,
    )


@dataclass
class GraphLqrSolution:
    """Solution of the graph-constrained discrete regulator."""

    record: Record
    stage_t: np.ndarray
    stage_u: np.ndarray
    stage_y: np.ndarray
    cost: float
    initial_defect: float
    graph_residual: float
    graph: GraphBasis


def _stage_to_nodes(stage: np.ndarray) -> np.ndarray:
    """Return a plotting representation of interval-stage values on the nodes."""
    nodes = np.empty((stage.shape[0], stage.shape[1] + 1))
    nodes[:, 0] = stage[:, 0]
    nodes[:, -1] = stage[:, -1]
    if stage.shape[1] > 1:
        nodes[:, 1:-1] = 0.5 * (stage[:, :-1] + stage[:, 1:])
    return nodes


def solve_graph_lqr(graph: GraphBasis, t: np.ndarray, weights: LqrWeights,
                    x0: np.ndarray, *, theta: float = 0.5) -> GraphLqrSolution:
    """Solve the LQR using only the graph learned from measured data.

    On interval ``k`` the tuple

    ``(u_k, (1-theta)x_k + theta*x_{k+1}, (x_{k+1}-x_k)/dt, y_k)``

    is constrained to the learned graph.  The initial condition is imposed
    exactly.  Controls and outputs are interval-stage variables, avoiding the
    checkerboard ambiguity of nodal controls under Crank--Nicolson.
    """
    if not graph.is_full:
        raise ValueError(
            f"resolved moment rank {graph.rank}/{graph.domain_dimension}; "
            "a full graph is required for LQR"
        )
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a one-dimensional grid with at least two nodes")
    steps = np.diff(t)
    if not np.allclose(steps, steps[0]):
        raise ValueError("the graph LQR currently requires a uniform time grid")
    dt = float(steps[0])
    x0 = np.asarray(x0, dtype=float)
    m, n, p = graph.m, graph.n, graph.p
    if x0.shape != (n,):
        raise ValueError(f"x0 must have shape {(n,)}")
    if weights.R.shape != (m, m) or weights.G.shape != (n, n):
        raise ValueError("LQR weights are incompatible with the learned graph")

    n_steps = t.size - 1
    nx = n * (n_steps + 1)
    nu = m * n_steps
    ny = p * n_steps
    nvar = nx + nu + ny

    # Objective z.T H z.  Zero state blocks are harmless because the graph
    # constraints and coercive control cost determine a unique trajectory.
    blocks = [np.zeros((n, n)) for _ in range(n_steps)] + [weights.G]
    blocks += [dt * weights.R for _ in range(n_steps)]
    blocks += [dt * graph.output_metric for _ in range(n_steps)]
    H = sparse_block_diag(blocks, format="csc")

    K = graph.constraint
    c = K.shape[0]
    Ku = K[:, :m]
    Kx = K[:, m:m + n]
    Kd = K[:, m + n:m + 2 * n]
    Ky = K[:, m + 2 * n:]
    Aeq = lil_matrix((n + c * n_steps, nvar))
    beq = np.zeros(n + c * n_steps)
    Aeq[:n, :n] = np.eye(n)
    beq[:n] = x0

    u_offset = nx
    y_offset = nx + nu
    left = (1.0 - theta) * Kx - Kd / dt
    right = theta * Kx + Kd / dt
    for k in range(n_steps):
        rows = slice(n + k * c, n + (k + 1) * c)
        Aeq[rows, k * n:(k + 1) * n] = left
        Aeq[rows, (k + 1) * n:(k + 2) * n] = right
        Aeq[rows, u_offset + k * m:u_offset + (k + 1) * m] = Ku
        Aeq[rows, y_offset + k * p:y_offset + (k + 1) * p] = Ky
    Aeq = Aeq.tocsc()

    kkt = bmat([[H, Aeq.T], [Aeq, None]], format="csc")
    rhs = np.concatenate([np.zeros(nvar), beq])
    solution = spsolve(kkt, rhs)
    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError("graph-constrained KKT solve failed")
    primal = solution[:nvar]

    X = primal[:nx].reshape(n_steps + 1, n).T
    U = primal[u_offset:y_offset].reshape(n_steps, m).T
    Y = primal[y_offset:].reshape(n_steps, p).T
    x_theta = (1.0 - theta) * X[:, :-1] + theta * X[:, 1:]
    xdot = np.diff(X, axis=1) / dt
    stage_data = np.vstack([U, x_theta, xdot, Y])
    record = Record(t, _stage_to_nodes(U), X, _stage_to_nodes(Y))
    defect = X[:, 0] - x0
    cost = (float(X[:, -1] @ weights.G @ X[:, -1])
            + dt * float(np.einsum("ik,ij,jk->", U, weights.R, U))
            + dt * float(np.einsum("ik,ij,jk->", Y, graph.output_metric, Y)))
    return GraphLqrSolution(
        record=record,
        stage_t=(t[:-1] + t[1:]) / 2.0,
        stage_u=U,
        stage_y=Y,
        cost=cost,
        initial_defect=float(np.sqrt(max(defect @ graph.state_metric @ defect, 0.0))),
        graph_residual=graph.residual(stage_data),
        graph=graph,
    )
