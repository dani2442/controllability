"""The input--output controllability test of ``thm:io-window-controllability``.

Proposition ``prop:data-fattorini-hautus`` reads the state: it looks for a
functional ``eta`` of ``X`` whose pairing with the record is a pure exponential.
When only ``(u, y)`` is measured the state is latent, and the theorem replaces
``eta`` by a functional of a whole length-``T`` input--output *window*,

    l(u, y) = int_0^T <v(t), u(t)>_U dt + int_0^T <g(t), y(t)>_Y dt,

asking instead that the shifted record satisfy

    l(ubar_s, ybar_s) = kappa e^{lambda s},   s in (0, 2T + Theta).   (eq:io-window-fh)

Why a window functional is the right substitute
-----------------------------------------------
By ``lem:finite-horizon-output`` the window at shift ``s`` is
``(ubar_s, F_T ubar_s + O_T xbar(s))``, so the functional pulls back to

    l(ubar_s, ybar_s) = <alpha, ubar_s> + <eta, xbar(s)>,
    alpha := v + F_T^* g,      eta := O_T^* g.                       (eq:io-window-pullback)

The input kernel ``v`` is free, so ``alpha`` can always be annulled, and what is
left is exactly the state predicate of ``prop:data-fattorini-hautus`` restricted
to ``eta`` in the range of ``O_T^*``.  Exact observability makes that range all
of ``X``, which is why the theorem assumes it: the window functional then
reaches every state functional the state test could have used.

What the record has to supply is stronger than before.  The step from
"``l`` is exponential on the record" to "``eta`` is a Fattorini--Hautus
obstruction" needs ``alpha = 0``, and the proof gets it from output-window
informativity at horizon ``3T``: the measured windows must span the behaviour,
or else an ``alpha != 0`` supported on the unexcited directions survives.  This
is not a technicality.  On a multisine record the shifted windows span only a
few dozen behaviour directions, a length-``T`` FIR kernel can null every line
but one, and ``<alpha, ubar_s>`` is then *itself* a pure exponential
``e^{i omega_k s}``: the test returns one spurious obstruction per line.  A
broadband record has far more lines than the window has taps and no such kernel
exists.  ``experiments/exp02_controllability.py`` shows both.

Discretisation
--------------
The window is sampled on the record's own grid, and the pairing is the
quadrature of the ``L^2`` product.  Folding ``sqrt(w_j)`` into both sides,

    w(s) := ( sqrt(w_j) ubar(s + t_j), sqrt(w_j) ybar(s + t_j) )_j,
    theta := ( sqrt(w_j) v(t_j), sqrt(w_j) g(t_j) )_j,

gives ``theta' w(s) = l(ubar_s, ybar_s)`` and ``||theta||_2 = ||(v,g)||_{L^2}``,
so the Euclidean geometry of the coefficient vector is the ``L^2`` geometry of
the kernels.  The record is generated on this grid, so the sampled window
relation ``y = F u + O x`` holds exactly and the span of ``{w(s)}`` has the
behaviour dimension ``m n_w + n`` it should.

Finding the obstructions is then a two-stage problem, and both stages are
needed.  Testing (eq:io-window-fh) against ``phi_i in H_0^1`` and integrating by
parts turns it into the pencil ``theta' W1 = lambda theta' W0`` -- the same
weak form as the state test, now on the window moments.  The pencil supplies
the candidate ``lambda``; its *eigenvectors* are useless here, because ``theta``
ranges over a few hundred directions whose singular values span five orders of
magnitude and the true obstruction sits among the weak ones.  Each candidate is
therefore re-decided by the literal predicate: (eq:io-window-fh) says that
``e^{lambda s}`` lies in the span of ``{s -> theta' w(s)}``, so

    rho(lambda) := dist_{L^2(0,2T+Theta)}( e^{lambda s}, span ) / ||e^{lambda s}||

is a well-posed least-squares quantity, and ``lambda`` is polished by minimising
it.  Normalising the target to unit amplitude builds the theorem's ``kappa != 0``
clause into the formulation; what the fit reports instead is ``||theta||``, the
size of the kernel pair that a unit exponential costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize

from .controllability import ControllabilityReport
from .moments import hat_tests, trapezoid_weights
from .timestepping import Record


@dataclass
class WindowRecord:
    """Sampled length-``T`` windows of an input--output record, shifted over ``s``.

    ``data`` holds one column per shift, the column being the pair
    ``(ubar_s, ybar_s)`` stacked and weighted by ``sqrt`` of the window
    quadrature, so that a Euclidean product with a coefficient vector is the
    ``L^2`` pairing of ``eq:io-window-fh``.  Only ``u`` and ``y`` are stored:
    a window record is exactly what input--output data supplies.
    """

    s: np.ndarray  # (n_s,) shift clock, the variable of eq:io-window-fh
    t: np.ndarray  # (n_w,) window clock, [0, T]
    data: np.ndarray  # ((m+p) n_w, n_s)
    m: int
    p: int

    @property
    def n_w(self) -> int:
        return self.t.size

    @property
    def n_s(self) -> int:
        return self.s.size

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def horizon(self) -> float:
        """``T``, the window length."""
        return float(self.t[-1])

    @property
    def spread(self) -> float:
        """``2T + Theta``, the range of shifts the predicate is tested on."""
        return float(self.s[-1])

    @property
    def nyquist(self) -> float:
        """``pi/dt``: beyond it, ``e^{lambda s}`` aliases on the shift grid."""
        return math.pi / self.dt

    def behaviour_dimension(self, n_states: int) -> int:
        """``m n_w + n``, the dimension of the sampled behaviour on the window.

        The reference size the resolved rank is reported against.  It uses the
        state dimension of the model and is a scoring quantity only, never an
        input of :func:`io_window_controllability`.
        """
        return self.m * self.n_w + n_states

    def kernels(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undo the quadrature weighting: ``theta`` as the kernel pair ``(v, g)``."""
        root = np.sqrt(trapezoid_weights(self.t))
        blocks = np.asarray(theta).reshape(self.m + self.p, self.n_w) / root
        return blocks[: self.m], blocks[self.m :]

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """``s -> l(ubar_s, ybar_s)`` for the functional with coefficients ``theta``."""
        return np.asarray(theta) @ self.data


def io_shift_windows(record: Record, *, horizon: float,
                     spread: float | None = None) -> WindowRecord:
    """Length-``horizon`` windows of ``(u, y)`` at every shift in ``(0, spread)``.

    By time invariance each window is again an input--output trajectory of the
    same system, so all of them lie in ``B_T^{u,y}``; the theorem asks that they
    fill it out.  ``spread`` defaults to everything the record allows, which for
    a record of length ``3T + Theta`` is the ``2T + Theta`` of the theorem.
    """
    dt = record.dt
    n_w = int(round(horizon / dt)) + 1
    if n_w > record.n_samples:
        raise ValueError("the record is shorter than the window")
    n_s = record.n_samples - n_w + 1 if spread is None else int(round(spread / dt)) + 1
    if n_s < 1 or n_s + n_w - 1 > record.n_samples:
        raise ValueError(
            f"a record of {record.n_samples} samples carries at most "
            f"{record.n_samples - n_w + 1} shifts of a {n_w}-sample window"
        )
    root = np.sqrt(trapezoid_weights(record.t[:n_w]))[:, None]
    index = np.arange(n_w)[:, None] + np.arange(n_s)[None, :]
    blocks = [signal[a][index] * root
              for signal in (record.u, record.y) for a in range(signal.shape[0])]
    return WindowRecord(s=record.t[:n_s], t=record.t[:n_w] - record.t[0],
                        data=np.vstack(blocks), m=record.u.shape[0],
                        p=record.y.shape[0])


@dataclass
class WindowCandidate:
    """A candidate ``(lambda, v, g)`` of ``eq:io-window-fh`` read off the windows."""

    lam: complex
    theta: np.ndarray = field(repr=False)  # window functional, weighted coordinates
    exp_residual: float  # rho(lambda): relative L^2 distance to the window span
    functional_norm: float  # ||(v, g)||_{L^2}, the cost of a unit exponential
    accepted: bool = False

    def kernels(self, windows: WindowRecord) -> tuple[np.ndarray, np.ndarray]:
        """The kernels ``(v, g)`` of ``eq:io-window-fh`` on ``[0, T]``."""
        return windows.kernels(self.theta)


class _ExponentialFit:
    """``rho(lambda)`` and its minimiser over the span of the shifted windows.

    The span is represented by an orthonormal basis of the *row* space of the
    window matrix in the ``L^2(0, 2T+Theta)`` metric of the shift variable, so
    the distance is one projection and the minimisation over ``lambda`` is the
    variable-projection form of the joint problem in ``(lambda, v, g)``.
    """

    def __init__(self, windows: WindowRecord, resolved: np.ndarray,
                 lift: np.ndarray | None = None) -> None:
        self.windows = windows
        self.resolved = resolved  # (r, n_s), the rows the record actually resolves
        # Coefficients are found against ``resolved``; ``lift`` carries them back
        # to the ((m+p) n_w) coordinates in which they are the kernel pair.
        self.lift = np.eye(resolved.shape[0]) if lift is None else lift
        self.root = np.sqrt(trapezoid_weights(windows.s))
        left, singular, right = np.linalg.svd(resolved * self.root,
                                              full_matrices=False)
        keep = singular > singular[0] * 1e-14
        self.basis = right[keep]  # orthonormal rows spanning the fitted subspace
        self.left = left[:, keep]
        self.singular = singular[keep]

    def target(self, lam: complex) -> np.ndarray | None:
        """``e^{lambda s}`` in the weighted coordinates, or ``None`` if unusable.

        The exponential is normalised by its own peak before anything else is
        done with it.  ``rho`` is a relative residual, so the scale is free, and
        without it a candidate with a large positive real part overflows and
        poisons the ordering of the candidates with a ``nan``.
        """
        lam = complex(lam)
        # Beyond the Nyquist rate of the shift grid, e^{lambda s} is
        # indistinguishable from its aliases and the fit is meaningless.
        if not np.isfinite(lam) or abs(lam.imag) > self.windows.nyquist:
            return None
        exponent = lam * self.windows.s
        b = np.exp(exponent - np.max(exponent.real)) * self.root
        return b if np.all(np.isfinite(b)) and np.any(b) else None

    def residual(self, lam: complex) -> float:
        """Relative ``L^2`` distance from ``e^{lambda s}`` to the resolved span."""
        b = self.target(lam)
        if b is None:
            return 1.0
        norm = float(np.linalg.norm(b))
        coef = self.basis.conj() @ b
        gap = norm**2 - float(np.vdot(coef, coef).real)
        residual = math.sqrt(max(gap, 0.0)) / norm
        return float(residual) if math.isfinite(residual) else 1.0

    def refine(self, lam: complex, *, maxiter: int = 250,
               slack: float = 1e-3) -> complex:
        """Polish ``lambda`` by minimising ``log10 rho``, a smooth 2-parameter fit.

        The record is real, so ``rho(conj lambda) = rho(lambda)`` and the real
        axis is a critical line of the objective; a real mode is therefore
        returned a hair off it, at an imaginary part far below what the shift
        range could resolve.  The candidate is snapped back whenever dropping
        the imaginary part costs nothing -- for a genuinely complex mode it
        costs everything, so the test is self-certifying.
        """
        result = minimize(
            lambda p: math.log10(max(self.residual(complex(p[0], p[1])), 1e-16)),
            x0=[complex(lam).real, complex(lam).imag], method="Nelder-Mead",
            options=dict(xatol=1e-9, fatol=1e-9, maxiter=maxiter),
        )
        polished = complex(result.x[0], result.x[1])
        real = complex(result.x[0], 0.0)
        if self.residual(real) <= (1.0 + slack) * max(self.residual(polished), 1e-16):
            return real
        return polished

    def functional(self, lam: complex) -> tuple[np.ndarray, float]:
        """The minimising ``theta`` in the window coordinates, and its ``L^2`` norm.

        Solves ``min_theta ||theta' G - e^{lambda s}||`` for ``G`` the weighted
        resolved windows, which is ``theta = U Sigma^{-1} V' b`` on the pseudo-
        inverse.  The exponential is normalised to unit peak, so the norm that
        comes back is the size of the kernel pair a unit exponential costs.
        """
        b = self.target(lam)
        if b is None:
            return np.zeros(self.lift.shape[0]), np.inf
        coef = self.lift @ ((self.left / self.singular) @ (self.basis.conj() @ b))
        return coef, float(np.linalg.norm(coef))


def _pencil_candidates(fit_rows: np.ndarray, s: np.ndarray, n_tests: int,
                       rank_tol: float) -> np.ndarray:
    """Candidate ``lambda`` from the weak form of ``eq:io-window-fh``.

    Testing the identity against ``phi_i in H_0^1(0, 2T+Theta)`` and moving the
    derivative onto the test function gives ``theta' W1 = lambda theta' W0``,
    the window counterpart of the moment pencil of the state test.  Only the
    eigenvalues are used; the eigenvectors are re-derived by the least-squares
    fit, which is what makes the two-stage form necessary.
    """
    tests = hat_tests(s, n_tests)
    W0 = tests.integrate(fit_rows)
    W1 = tests.integrate_d(fit_rows)
    left, singular, right = np.linalg.svd(W0, full_matrices=False)
    rank = int(np.sum(singular**2 > rank_tol * singular[0] ** 2))
    P, V = left[:, :rank], right[:rank].T
    lams = eig(a=(P.T @ W1 @ V).T, b=(P.T @ W0 @ V).T, left=False, right=False)
    return lams[np.isfinite(lams)]


def io_window_controllability(
    windows: WindowRecord, *, n_states: int, rank_tol: float = 1e-10,
    residual_tol: float = 3e-3, n_tests: int | None = None,
    prefilter: float = 0.5, min_refined: int = 8,
) -> ControllabilityReport:
    """Run the test of ``thm:io-window-controllability`` on measured windows.

    ``n_states`` only sizes the ``m n_w + n`` the resolved rank is reported
    against; no model quantity enters the decision, which sees the shifted
    windows and nothing else.
    """
    root = np.sqrt(trapezoid_weights(windows.s))
    left, singular, _ = np.linalg.svd(windows.data * root, full_matrices=False)
    spectrum = singular**2
    if spectrum.size == 0 or spectrum[0] <= 0:
        raise ValueError("the window record is identically zero")
    rank = int(np.sum(spectrum > rank_tol * spectrum[0]))
    # Restricting theta to the resolved left singular directions is the closure
    # form of the theorem in finite precision: a theta outside them pairs with
    # every measured window to within round-off, so it certifies nothing.
    resolved = left[:, :rank].T @ windows.data

    fit = _ExponentialFit(windows, resolved, lift=left[:, :rank])
    if n_tests is None:
        n_tests = min(int(math.ceil(1.5 * rank)), (windows.n_s - 1) // 2 - 2)
    lams = _pencil_candidates(resolved, windows.s, n_tests, rank_tol)

    # The pencil returns one candidate per resolved direction and polishing them
    # all is wasteful: the refinement is local, so a candidate whose raw
    # residual is already this far from every exponential in the span cannot
    # descend onto an obstruction that a closer candidate has not found too.
    # The best few are always polished anyway, so that a record with no
    # obstruction still reports how close its nearest miss came.
    ordered = sorted(lams, key=fit.residual)
    candidates: list[WindowCandidate] = []
    for rank_of_candidate, lam in enumerate(ordered):
        if rank_of_candidate >= min_refined and fit.residual(lam) > prefilter:
            break
        polished = fit.refine(lam)
        if any(abs(polished - c.lam) < 1e-6 * max(1.0, abs(polished))
               for c in candidates):
            continue
        residual = fit.residual(polished)
        theta, norm = fit.functional(polished)
        candidates.append(WindowCandidate(
            lam=polished, theta=theta, exp_residual=residual,
            functional_norm=norm, accepted=residual < residual_tol,
        ))

    candidates.sort(key=lambda c: (not c.accepted, c.exp_residual))
    return ControllabilityReport(
        approximately_controllable=not any(c.accepted for c in candidates),
        candidates=candidates,
        numerical_rank=rank,
        dimension=windows.behaviour_dimension(n_states),
        rank_tol=rank_tol,
        residual_tol=residual_tol,
        singular_values=spectrum,
    )
