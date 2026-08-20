"""Input--output windows: the discretization of ``thm:windowed-io-fundamental-lemma``.

The lemmas of ``sec:fundamental-lemma`` read the state.  When only the input
and the output are measured, ``subsec:window-informativity`` lifts whole
windows instead: with ``f_s = f(s + .)|_{(0,T)}``, a ``(T, Theta)``-state-window
informative record satisfies

    closure span{ (ubar_s, ybar_s) : 0 < s < Theta } = closure B_T^{u,y},
    B_T^{u,y} = { (u, F_T u + O_T x0) } = im M_T,

so the shifted windows of one record span the finite-horizon input--output
behavior.  Sampled, this is a library ``(u_i, y_i)`` of measured windows, and a
trial signal is a combination ``z_g = sum_i g_i (u_i, y_i)``.  Neither a state
sample, nor a state derivative, nor ``O_T`` enters the construction.

Cost functionals of a window
----------------------------
The window carries no state, so the two state-dependent items of
``eq:lqr-cost`` have to be supplied by the input--output data itself:

* **Initial condition.**  Each window is extended backwards by ``T_ini``, and
  the past part of the combination is matched to the measured past
  ``(u_ini, y_ini)`` of the plant, up to and including the junction sample at
  which the regulator takes over.  By the shift identity of
  ``lem:finite-horizon-output`` this pins
  the state at the junction: it is unique as soon as ``(A, C)`` is
  approximately observable in time ``T_ini`` (``def:observability``), and the
  inversion is stable under exact observability.  All three examples are only
  approximately observable, with a compact ``O_T`` whose singular values decay
  (``sec:numerics``), so the match is imposed as a least-squares fit with
  weight ``1/rho`` rather than as an equality.  This is the closure form of the
  theorem, and ``rho`` is the one parameter the state methods do not need.

  Including the junction sample is what makes the reconstruction a genuine
  trajectory: under the theta scheme the state at the junction is driven by
  the two-point average that straddles it, so leaving that sample to the
  regulator would let it move the initial state by ``theta dt B du`` and buy a
  cost below the true optimum.  The price is that the regulator inherits one
  sample of the conditioning input, which costs ``O(dt)`` in the achieved cost
  -- the accuracy floor of this method, against ``O(dt^2)`` for the
  interval-stage graph regulator of :mod:`ddinf.lqr.graph`.
* **Terminal weight.**  ``<x(T), G_T x(T)>`` is not a functional of ``(u, y)``
  and cannot be represented here at all; :func:`solve_io_lqr` therefore refuses
  a nonzero ``G``.  The comparison in ``experiments/lqr.py`` runs both
  regulators with ``G_T = 0`` so that they minimize the same functional.

The resulting problem is the small dense system

    (rho H + E) g = e,

with ``H`` the cost Gram matrix of the future halves of the library and
``(E, e)`` the normal equations of the past match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .riccati import LqrWeights
from ..data.moments import quadrature_weights, trapezoid_weights
from ..data.records import Record


def _metric_root(metric: np.ndarray) -> np.ndarray:
    """``R`` with ``R' R = metric``, so that ``||R z||_2^2 = <z, metric z>``."""
    return np.linalg.cholesky(0.5 * (metric + metric.T)).T


def _flatten(signal: np.ndarray, root: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(N, d, T)`` signals as ``(N, d*T)`` rows of a Euclidean Gram factor.

    The row of window ``i`` is ``sqrt(w_t) * root @ z_i(t)``, so that a product
    of two rows is the quadrature of ``<z_i(t), metric z_j(t)>``.
    """
    scaled = np.einsum("ab,nbt->nat", root, signal) * np.sqrt(weights)
    return scaled.reshape(signal.shape[0], -1)


@dataclass
class WindowLibrary:
    """Shifted input--output windows of one record, past part included.

    The window clock runs over ``[-T_ini, T]``; sample ``n_past`` is the
    junction ``t = 0`` at which the regulator starts.  Only ``u`` and ``y`` are
    stored: a library is exactly what an input--output record supplies.
    """

    t: np.ndarray  # (n_w,) window clock, t[n_past] = 0
    u: np.ndarray  # (N, m, n_w)
    y: np.ndarray  # (N, p, n_w)
    n_past: int
    starts: np.ndarray  # sample offset of each window in the record

    @property
    def n_windows(self) -> int:
        return self.u.shape[0]

    @property
    def m(self) -> int:
        return self.u.shape[1]

    @property
    def p(self) -> int:
        return self.y.shape[1]

    @property
    def n_samples(self) -> int:
        return self.t.size

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def past(self) -> slice:
        return slice(0, self.n_past + 1)

    @property
    def future(self) -> slice:
        return slice(self.n_past, self.n_samples)

    @property
    def t_future(self) -> np.ndarray:
        return self.t[self.future] - self.t[self.n_past]

    def behavior_dimension(self, n_states: int) -> int:
        """``m n_w + n``: the dimension of the sampled behavior on the window.

        The reference size a library is measured against.  It uses the state
        dimension of the model and is a scoring quantity only, never an input
        of :func:`solve_io_lqr`.
        """
        return self.m * self.n_samples + n_states


def io_shift_library(record: Record, *, past: float, horizon: float,
                     n_windows: int, spread: float | None = None
                     ) -> WindowLibrary:
    """``n_windows`` shifted ``[-T_ini, T]`` windows of the measured ``(u, y)``.

    By time invariance each window is again an input--output trajectory of the
    same system, which is what makes the library a subspace of the behavior;
    the theorem says the span fills it out as the shifts fill ``(0, Theta)``.
    """
    n_past = int(round(past / record.dt))
    n_w = n_past + int(round(horizon / record.dt)) + 1
    if n_w > record.n_samples:
        raise ValueError("record is shorter than the past plus the horizon")
    last = record.n_samples - n_w
    if spread is not None:
        last = min(last, int(round(spread / record.dt)))
    if last < n_windows - 1:
        raise ValueError(f"record too short for {n_windows} distinct shifts")
    starts = np.unique(np.round(np.linspace(0, last, n_windows)).astype(int))
    u = np.stack([record.u[:, k:k + n_w] for k in starts])
    y = np.stack([record.y[:, k:k + n_w] for k in starts])
    return WindowLibrary(t=record.t[:n_w] - record.t[n_past], u=u, y=y,
                         n_past=n_past, starts=starts)


@dataclass
class BehaviorBasis:
    """The resolved part of the behavior spanned by a window library.

    ``windows`` are combinations of the measured windows that are orthonormal
    in the ``L^2`` metric of the window, ``spectrum`` holds the ``mu_j`` of
    ``eq:num-resolved`` -- squared singular values of the library in that
    metric -- and ``rank`` is ``r_eps``.  Truncating at ``rank_tol`` is what
    makes the closure form of ``thm:windowed-io-fundamental-lemma`` usable in
    finite precision: the discarded directions are combinations of windows that
    nearly cancel, and keeping them only amplifies rounding error.
    """

    windows: WindowLibrary
    coefficients: np.ndarray  # (N, r), columns are the combinations kept
    spectrum: np.ndarray
    rank: int
    rank_tol: float

    @property
    def n_windows(self) -> int:
        return self.coefficients.shape[0]

    @property
    def smallest_resolved(self) -> float:
        return float(self.spectrum[self.rank - 1]) if self.rank else 0.0


def behavior_basis(library: WindowLibrary, *, rank_tol: float = 1e-10
                   ) -> BehaviorBasis:
    """Metric-weighted SVD of the window library, truncated at ``rank_tol``.

    The counterpart, for input--output data, of the weighted SVD of ``Z_q`` in
    :func:`ddinf.lqr.graph.estimate_graph`: both replace a raw data matrix by a
    well-scaled basis of the subspace it resolves, at the same relative
    threshold on the squared singular values.
    """
    w = trapezoid_weights(library.t)
    eye_u, eye_y = np.eye(library.m), np.eye(library.p)
    L = np.concatenate([_flatten(library.u, eye_u, w),
                        _flatten(library.y, eye_y, w)], axis=1)
    left, singular, _ = np.linalg.svd(L, full_matrices=False)
    spectrum = singular ** 2
    if spectrum.size == 0 or spectrum[0] <= 0:
        raise ValueError("the window library is identically zero")
    rank = int(np.sum(spectrum > rank_tol * spectrum[0]))
    coefficients = left[:, :rank] / singular[:rank]

    def combine(signal: np.ndarray) -> np.ndarray:
        flat = coefficients.T @ signal.reshape(signal.shape[0], -1)
        return flat.reshape(rank, *signal.shape[1:])

    windows = WindowLibrary(
        t=library.t,
        u=combine(library.u),
        y=combine(library.y),
        n_past=library.n_past,
        starts=library.starts,
    )
    return BehaviorBasis(windows=windows, coefficients=coefficients,
                          spectrum=spectrum, rank=rank, rank_tol=rank_tol)


@dataclass
class IoLqrSolution:
    """Regulator reconstructed inside the measured input--output behavior."""

    g: np.ndarray
    t: np.ndarray  # control clock 0..T
    u: np.ndarray  # (m, n_future)
    y: np.ndarray  # (p, n_future)
    past_t: np.ndarray  # conditioning clock -T_ini..0, junction sample included
    past_u: np.ndarray
    past_y: np.ndarray
    cost: float
    past_defect: float
    rho: float
    behavior: BehaviorBasis

    @property
    def n_windows(self) -> int:
        return self.behavior.n_windows

    def input_callable(self):
        """The recovered input as a function of time, for a scoring simulation."""
        t, u = self.t, self.u
        return lambda s: np.stack([np.interp(np.asarray(s, dtype=float), t, ui)
                                   for ui in u])


def assemble_io_lqr(behavior: BehaviorBasis, weights: LqrWeights,
                    u_past: np.ndarray, y_past: np.ndarray, *,
                    output_metric: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cost Gram ``H`` and past-match normal equations ``(E, e)``.

    ``H`` uses the quadrature split of :mod:`ddinf.data.moments`: Simpson for the
    output, trapezoid for the control, because the optimizer chooses the input
    sample by sample and the alternating Simpson weights would price the
    sample-Nyquist direction unevenly.
    """
    library = behavior.windows
    if np.any(weights.G):
        raise ValueError(
            "the terminal state cost is not a functional of (u, y); "
            "run the input-output regulator with LqrWeights terminal=0"
        )
    output_metric = np.eye(library.p) if output_metric is None else output_metric
    n_f = library.t_future
    root_r, root_y = _metric_root(weights.R), _metric_root(output_metric)

    control = _flatten(library.u[:, :, library.future], root_r,
                       trapezoid_weights(n_f))
    output = _flatten(library.y[:, :, library.future], root_y,
                      quadrature_weights(n_f))
    H = control @ control.T + output @ output.T
    H = 0.5 * (H + H.T)

    # The past match weighs input and output alike, in the L^2 metric of the
    # conditioning window: both are measured signals of the same record.
    w_p = trapezoid_weights(library.t[library.past])
    eye_u, eye_y = np.eye(library.m), np.eye(library.p)
    P = np.concatenate([_flatten(library.u[:, :, library.past], eye_u, w_p),
                        _flatten(library.y[:, :, library.past], eye_y, w_p)],
                       axis=1)
    target = np.concatenate([
        (u_past * np.sqrt(w_p)).ravel(), (y_past * np.sqrt(w_p)).ravel(),
    ])
    if target.size != P.shape[1]:
        raise ValueError("the past segment and the library windows disagree in shape")
    return H, P @ P.T, P @ target


def solve_io_lqr(behavior: BehaviorBasis, weights: LqrWeights,
                 u_past: np.ndarray, y_past: np.ndarray, *, rho: float = 1e-8,
                 output_metric: np.ndarray | None = None) -> IoLqrSolution:
    """Minimize the measured cost over the behavior spanned by the windows.

    Solves ``(rho H + E) g = e``, the stationarity condition of

        J(z_g) + rho^{-1} || (u_g, y_g)|_past - (u_ini, y_ini) ||^2_{L^2},

    and returns the combination ``z_g`` on the control horizon.  ``rho`` is a
    regularisation of the closure form: the past match is a least-squares
    problem whenever ``O_{T_ini}`` fails to be bounded below, which is the
    generic situation in infinite dimensions.
    """
    library = behavior.windows
    H, E, e = assemble_io_lqr(behavior, weights, u_past, y_past,
                              output_metric=output_metric)
    # rho H + E is symmetric positive semidefinite and can be singular in the
    # directions the past window does not see; a symmetric eigendecomposition
    # inverts it on its resolved range, where a dense least-squares driver on
    # the same matrix is both slower and prone to convergence failures.
    values, vectors = np.linalg.eigh(rho * H + E)
    keep = values > np.finfo(float).eps * max(values[-1], 0.0) * values.size
    g = vectors[:, keep] @ ((vectors[:, keep].T @ e) / values[keep])

    u = np.einsum("n,nat->at", g, library.u)
    y = np.einsum("n,nat->at", g, library.y)
    defect = np.concatenate([u[:, library.past] - u_past,
                             y[:, library.past] - y_past])
    w_p = trapezoid_weights(library.t[library.past])
    return IoLqrSolution(
        g=g,
        t=library.t_future,
        u=u[:, library.future],
        y=y[:, library.future],
        past_t=library.t[library.past],
        past_u=u[:, library.past],
        past_y=y[:, library.past],
        cost=float(g @ H @ g),
        past_defect=float(np.sqrt(max(np.sum(w_p * defect ** 2), 0.0))),
        rho=rho,
        behavior=behavior,
    )
