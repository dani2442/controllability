"""The dynamic synthesis operator of Appendix ``app:willems-gramian``, discretised.

The dynamic synthesis operator of a record is

    W_(u,x,y) phi = ( int phi u,  int phi x,  -int phi' x,  int phi y ),
    phi in H_0^1(I),

and the whole point of it is that it never differentiates the measured state:
the third component is the derivative moment, obtained by moving the derivative
onto the test function.  Evaluating it on a finite family ``phi_1..phi_q``
gives the four moment matrices below, from which every data-driven quantity in
this project is assembled.

Because the record satisfies ``x' = A x + B u`` exactly, the moment matrices
satisfy the *exact* linear identity

    X1 = A X0 + B U0,      Y0 = C X0,

which is the finite-dimensional shadow of the factorisation
``W = Gamma o Z`` in the proof of ``thm:willems-gramian``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timestepping import Record


@dataclass
class TestFunctions:
    """A family ``phi_i`` in ``H_0^1(I)`` sampled on the record's time grid."""

    phi: np.ndarray  # (q, T)
    dphi: np.ndarray  # (q, T)
    weights: np.ndarray  # (T,) composite trapezoid weights
    kind: str

    @property
    def q(self) -> int:
        return self.phi.shape[0]

    def integrate(self, z: np.ndarray) -> np.ndarray:
        """``int phi_i z dt`` for a signal ``z`` of shape ``(d, T)`` -> ``(d, q)``."""
        return z @ (self.weights[:, None] * self.phi.T)

    def integrate_d(self, z: np.ndarray) -> np.ndarray:
        """``-int phi_i' z dt`` -> ``(d, q)``."""
        return -(z @ (self.weights[:, None] * self.dphi.T))


def quadrature_weights(t: np.ndarray) -> np.ndarray:
    """Composite Simpson weights on a uniform grid (trapezoid if the count is even).

    The moments are the only place where the sampled record is integrated, so
    the order of this rule caps the accuracy of everything downstream: with the
    trapezoid rule the identity ``X1 = A X0 + B U0`` holds only to ``O(dt^2)``,
    with Simpson to ``O(dt^4)``, which puts it below the error of the time
    stepping itself.

    Do *not* use this rule to weight a term that the optimiser controls
    pointwise in the input -- see :func:`trapezoid_weights`.
    """
    dt = t[1] - t[0]
    if t.size % 2 == 1:
        w = np.ones(t.size)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * dt / 3.0
    return trapezoid_weights(t)


def trapezoid_weights(t: np.ndarray) -> np.ndarray:
    """Composite trapezoid weights -- the rule the time stepping itself uses.

    Required wherever the quadrature weights multiply a quantity the optimiser
    is free to choose sample by sample, i.e. the control term of the LQR cost.
    The theta scheme forces the input through ``theta u^{k+1} + (1-theta) u^k``,
    which for Crank--Nicolson is the two-point average ``(u_k + u_{k+1})/2``: the
    odd--even component of a sampled input is invisible to the state.  Charging
    it with the *alternating* Simpson weights ``4 dt/3, 2 dt/3`` therefore makes
    a genuinely free direction cheap on the even samples and expensive on the
    odd ones, and the discrete minimiser splits a given effective input between
    neighbours in inverse proportion to the weights -- a spurious 2:1
    sample-Nyquist ripple in an input the continuous problem wants smooth.  A
    uniform weight prices every sample alike and the ripple disappears.

    This is the lumped form of the exact ``int |u|^2`` of the piecewise-linear
    interpolant that the theta scheme implicitly integrates, so it is the
    consistent choice rather than merely the safe one.
    """
    dt = t[1] - t[0]
    w = np.full(t.size, float(dt))
    w[0] = w[-1] = dt / 2.0
    return w


def hat_tests(t: np.ndarray, q: int) -> TestFunctions:
    """``q`` P1 hat functions on a partition of ``I``, vanishing at the ends.

    Local supports keep the moments well scaled and make each column of the
    moment matrices a local time-average of the record.  The knots are snapped
    to *even* sample indices so that no Simpson panel straddles the kink of a
    hat or the jump of its derivative -- otherwise the quadrature drops to
    first order and swamps the moment identity.
    """
    n_t = t.size
    idx = np.unique(2 * np.round(np.linspace(0, n_t - 1, q + 2) / 2).astype(int))
    if idx.size < q + 2:
        raise ValueError(f"time grid too coarse for q={q} hats ({n_t} samples)")
    phi = np.zeros((q, n_t))
    dphi = np.zeros((q, n_t))
    for i in range(q):
        a, c, b = idx[i], idx[i + 1], idx[i + 2]
        up = slice(a, c + 1)
        dn = slice(c, b + 1)
        rise, fall = 1.0 / (t[c] - t[a]), -1.0 / (t[b] - t[c])
        phi[i, up] = (t[up] - t[a]) / (t[c] - t[a])
        phi[i, dn] = (t[b] - t[dn]) / (t[b] - t[c])
        dphi[i, up] = rise
        dphi[i, dn] = fall
        # phi' is a step function whose jumps sit exactly on quadrature nodes.
        # A composite rule splits the weight of an *interior* node evenly
        # between the panels on either side, so such a node must carry the mean
        # of the two one-sided limits; using either limit alone costs an order
        # of accuracy.  The two ends of the grid carry a half weight already,
        # so there the one-sided limit is the correct value.
        dphi[i, a] = rise if a == 0 else 0.5 * rise
        dphi[i, c] = 0.5 * (rise + fall)
        dphi[i, b] = fall if b == n_t - 1 else 0.5 * fall
    return TestFunctions(phi, dphi, quadrature_weights(t), "hat")


def sine_tests(t: np.ndarray, q: int) -> TestFunctions:
    """``phi_k(t) = sin(k pi (t-t0)/L)``, a smooth orthogonal basis of ``H_0^1(I)``."""
    t0, t1 = float(t[0]), float(t[-1])
    L = t1 - t0
    k = np.arange(1, q + 1)[:, None]
    s = (t[None, :] - t0)
    phi = np.sin(k * np.pi * s / L)
    dphi = (k * np.pi / L) * np.cos(k * np.pi * s / L)
    return TestFunctions(phi, dphi, quadrature_weights(t), "sine")


@dataclass
class Moments:
    """The image of the dynamic synthesis operator on a finite test family."""

    X0: np.ndarray  # (n, q)   int phi x
    X1: np.ndarray  # (n, q)  -int phi' x
    U0: np.ndarray  # (m, q)   int phi u
    Y0: np.ndarray  # (p, q)   int phi y
    tests: TestFunctions

    @property
    def q(self) -> int:
        return self.X0.shape[1]

    def stacked(self) -> np.ndarray:
        """``[X0; U0]`` -- the moment matrix whose row rank is informativity."""
        return np.vstack([self.X0, self.U0])

    def dynamics_residual(self, A: np.ndarray, B: np.ndarray) -> float:
        """Relative violation of ``X1 = A X0 + B U0``; a consistency check only.

        Uses the model, so it is never part of a data-driven computation -- it
        verifies that the discretisation really produces trajectories of the
        semi-discrete system.
        """
        r = self.X1 - A @ self.X0 - B @ self.U0
        return float(np.linalg.norm(r) / max(np.linalg.norm(self.X1), 1e-300))


def moments(record: Record, tests: TestFunctions) -> Moments:
    """Evaluate the dynamic synthesis operator on ``tests``."""
    return Moments(
        X0=tests.integrate(record.x),
        X1=tests.integrate_d(record.x),
        U0=tests.integrate(record.u),
        Y0=tests.integrate(record.y),
        tests=tests,
    )


def theta_moments(record: Record, tests: TestFunctions, *, theta: float = 0.5
                  ) -> Moments:
    """Weak moments consistent with a theta-method sampled trajectory.

    On each interval the theta scheme satisfies

    ``x[k+1]-x[k] = dt * (A x_theta[k] + B u_theta[k])``.

    Testing this identity with the stage value of ``phi`` gives
    ``X1 = A X0 + B U0`` to linear-solver precision.  The derivative moment is
    still weak: ``sum phi_theta * (x[k+1]-x[k])`` is the discrete
    summation-by-parts counterpart of ``-int phi' x`` and never forms a
    pointwise state derivative.
    """
    if tests.phi.shape[1] != record.n_samples:
        raise ValueError("test functions and record must use the same time grid")
    dt = np.diff(record.t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("theta moments currently require a uniform time grid")
    phi_stage = ((1.0 - theta) * tests.phi[:, :-1]
                 + theta * tests.phi[:, 1:])
    u_stage = ((1.0 - theta) * record.u[:, :-1]
               + theta * record.u[:, 1:])
    x_stage = ((1.0 - theta) * record.x[:, :-1]
               + theta * record.x[:, 1:])
    y_stage = ((1.0 - theta) * record.y[:, :-1]
               + theta * record.y[:, 1:])
    weighted_tests = float(dt[0]) * phi_stage.T
    return Moments(
        X0=x_stage @ weighted_tests,
        X1=np.diff(record.x, axis=1) @ phi_stage.T,
        U0=u_stage @ weighted_tests,
        Y0=y_stage @ weighted_tests,
        tests=tests,
    )
