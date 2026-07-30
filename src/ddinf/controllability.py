"""The data-driven controllability test of Proposition ``prop:data-fattorini-hautus``.

The test says: given an approximately informative record, the system fails to
be approximately controllable exactly when the record carries a pure
exponential, i.e. when there are ``lambda``, ``eta != 0`` and ``kappa`` with

    <eta, x_bar(t)> = kappa e^{lambda t}   on I.                       (eq:data-fh)

In moment form this is an ordinary matrix problem.  Testing the identity
against ``phi_i in H_0^1(I)`` and integrating by parts, ``<eta, x_bar> = kappa
e^{lambda t}`` says precisely that ``eta' X1 = lambda eta' X0``, so the
candidates are the **left eigenvectors of the pencil ``(X1, X0)``** built from
the measured moments -- no model anywhere.

Why this is the right object: the record satisfies ``X1 = A X0 + B U0``
exactly, so

    eta'X1 - lambda eta'X0 = eta'(A - lambda I) X0 + eta'B U0,

and when ``[X0; U0]`` has full row rank -- i.e. when the record is informative
-- the left-hand side vanishes if and only if ``A'eta = lambda eta`` *and*
``B'eta = 0``, which is the Fattorini--Hautus obstruction.  Controllable modes
leave the residual ``eta'B U0 != 0`` and are rejected.

Two independent certificates are computed for every candidate: the pencil
backward error, and the fit residual of the literal predicate (eq:data-fh) on
the sampled record.  A candidate is accepted only if both are small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eig

from .informativity import moment_spectrum
from .moments import Moments
from .systems import LinearSystem
from .timestepping import Record


@dataclass
class ModeCandidate:
    """A candidate obstruction ``(lambda, eta)`` extracted from the record."""

    lam: complex
    eta: np.ndarray  # functional: pairs with the state as eta @ x
    pencil_residual: float
    exp_residual: float
    kappa: complex
    accepted: bool = False

    def eta_as_function(self, sys: LinearSystem) -> np.ndarray:
        """``eta`` seen as an element of ``X``: ``psi`` with ``<psi, x>_X = eta'x``."""
        psi = np.linalg.solve(sys.MX, np.real(self.eta))
        return psi / np.linalg.norm(psi)


@dataclass
class ControllabilityReport:
    approximately_controllable: bool
    candidates: list[ModeCandidate]
    numerical_rank: int
    dimension: int
    rank_tol: float
    residual_tol: float
    singular_values: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def obstructions(self) -> list[ModeCandidate]:
        return [c for c in self.candidates if c.accepted]

    def __str__(self) -> str:
        head = ("approximately controllable"
                if self.approximately_controllable
                else f"NOT approximately controllable ({len(self.obstructions)} obstructions)")
        return (f"{head}; record informative to numerical rank "
                f"{self.numerical_rank}/{self.dimension} at tol {self.rank_tol:.1e}")


def _fit_exponential(t: np.ndarray, y: np.ndarray, lam: complex) -> tuple[complex, float]:
    """Least-squares ``kappa`` in ``y(t) ~ kappa e^{lambda t}`` and its relative residual."""
    basis = np.exp(lam * t)
    denom = float(np.real(np.vdot(basis, basis)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, np.inf
    kappa = np.vdot(basis, y) / denom
    resid = np.linalg.norm(y - kappa * basis)
    scale = np.linalg.norm(y)
    return kappa, float(resid / scale) if scale > 0 else np.inf


def data_driven_controllability(
    mom: Moments, record: Record, sys: LinearSystem, *,
    rank_tol: float = 1e-10, residual_tol: float = 1e-6,
    kappa_tol: float = 1e-8, space: str = "X",
) -> ControllabilityReport:
    """Run the test of ``prop:data-fattorini-hautus`` on measured moments.

    ``sys`` is used only for the inner product that reports informativity and
    for reshaping ``eta``; neither ``A``, ``B`` nor ``C`` enters the decision.
    """
    X0, X1 = mom.X0, mom.X1
    n, q = X0.shape
    if q < n:
        raise ValueError(f"need at least n={n} test functions, got q={q}")

    spec = moment_spectrum(mom.stacked(), sys, space=space, tol=rank_tol)

    # Reduce the (n x q) pencil to a square one by projecting the test-function
    # index onto the n leading right singular directions of X0; every candidate
    # is afterwards re-checked against all q moments, so the projection cannot
    # manufacture an obstruction.
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    V = Vt[:n].T  # (q, n)
    lams, etas = eig(a=(X1 @ V).T, b=(X0 @ V).T, left=False, right=True)

    scale_1 = np.linalg.norm(X1)
    scale_0 = np.linalg.norm(X0)
    candidates: list[ModeCandidate] = []
    for lam, eta in zip(lams, etas.T):
        if not np.isfinite(lam):
            continue
        eta = eta / np.linalg.norm(eta)
        # The pairing is the complex-*bilinear* extension of the real inner
        # product, as the paper prescribes -- no conjugation.  Using the
        # sesquilinear one instead silently misses every complex mode.
        resid = np.linalg.norm(eta @ X1 - lam * (eta @ X0))
        pencil_residual = float(resid / (scale_1 + abs(lam) * scale_0))

        y_eta = eta @ record.x  # <eta, x_bar(t)>
        kappa, exp_residual = _fit_exponential(record.t, y_eta, lam)

        accepted = (pencil_residual < residual_tol
                    and exp_residual < residual_tol
                    and abs(kappa) > kappa_tol * max(np.linalg.norm(y_eta), 1e-300))
        candidates.append(ModeCandidate(lam, eta, pencil_residual, exp_residual,
                                        kappa, accepted))

    candidates.sort(key=lambda c: (not c.accepted, -c.lam.real))
    return ControllabilityReport(
        approximately_controllable=not any(c.accepted for c in candidates),
        candidates=candidates,
        numerical_rank=spec.numerical_rank,
        dimension=spec.dimension,
        rank_tol=rank_tol,
        residual_tol=residual_tol,
        singular_values=spec.eigenvalues,
    )


def hautus_modes(sys: LinearSystem, tol: float = 1e-8) -> list[tuple[complex, np.ndarray, float]]:
    """Model-based baseline: ``(lambda, eta, |eta'B|)`` over the left eigenvectors of ``A``.

    The finite-dimensional Hautus test -- the reference the data-driven result
    is checked against, never used inside it.
    """
    lams, etas = np.linalg.eig(sys.A.T)
    out = []
    for lam, eta in zip(lams, etas.T):
        eta = eta / np.linalg.norm(eta)
        out.append((lam, eta, float(np.linalg.norm(eta @ sys.B))))
    out.sort(key=lambda r: -r[0].real)
    return out


def hautus_uncontrollable(sys: LinearSystem, tol: float = 1e-8
                          ) -> list[tuple[complex, np.ndarray]]:
    """The modes the model says are unreachable: ``A'eta = lambda eta``, ``B'eta = 0``."""
    return [(lam, eta) for lam, eta, gain in hautus_modes(sys) if gain < tol]
