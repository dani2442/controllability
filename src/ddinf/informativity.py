"""Informativity diagnostics (Definitions ``def:hilbert-informative`` / ``-W``).

The Gramian of a signal ``z in L^2(I;H)`` is ``G(z)zeta = int <z(t),zeta>_H z(t) dt``,
and the record is *approximately informative* when ``G`` is injective.  In
coordinates, with the inner product of ``H`` represented by a Gram matrix ``S``
(``<a,b>_H = a' S b``), one has

    <G(z) zeta, zeta>_H = zeta' S ( int z z' dt ) S zeta,

so ``G`` is represented by ``Z S`` with ``Z = int z z' dt``, and its spectrum is
that of the symmetric positive semidefinite matrix ``L' Z L`` where ``S = L L'``.
That spectrum is what these functions return.

The caveat the paper states (Remark following ``prop:data-fattorini-hautus``) is
visible here and is not a numerical artefact: on an infinite-dimensional space
``G`` is compact, so its eigenvalues accumulate at zero and its range is dense
but not closed.  Under discretisation the eigenvalues decay geometrically, and
"informative" can only ever mean "informative down to a threshold".  Every
routine here therefore reports a *numerical rank* together with the threshold
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moments import trapezoid_weights
from .systems import LinearSystem
from .timestepping import Record


@dataclass
class GramianSpectrum:
    """Eigenvalues of the Gramian operator, largest first."""

    eigenvalues: np.ndarray
    space: str
    tol: float

    @property
    def numerical_rank(self) -> int:
        return int(np.sum(self.eigenvalues > self.tol * self.eigenvalues[0]))

    @property
    def dimension(self) -> int:
        return self.eigenvalues.size

    def __str__(self) -> str:
        return (f"Gramian on U x {self.space}: dim {self.dimension}, "
                f"numerical rank {self.numerical_rank} at tol {self.tol:.1e}, "
                f"spread {self.eigenvalues[0]:.3e} .. {self.eigenvalues[-1]:.3e}")


def gram_matrix(sys: LinearSystem, space: str = "X") -> np.ndarray:
    """Block Gram matrix ``S`` of ``U x X`` or ``U x W`` (input space Euclidean)."""
    inner = {"X": sys.MX, "W": sys.MW}[space]
    S = np.zeros((sys.m + sys.n, sys.m + sys.n))
    S[: sys.m, : sys.m] = np.eye(sys.m)
    S[sys.m :, sys.m :] = inner
    return S


def gramian_spectrum(record: Record, sys: LinearSystem, *, space: str = "X",
                     tol: float = 1e-12) -> GramianSpectrum:
    """Spectrum of ``G(u_bar, x_bar)`` on ``U x X`` (or ``U x W``)."""
    z = np.vstack([record.u, record.x])  # (m+n, T)
    Z = (z * trapezoid_weights(record.t)) @ z.T  # int z z' dt

    S = gram_matrix(sys, space)
    L = np.linalg.cholesky(S)
    sym = L.T @ Z @ L
    ev = np.linalg.eigvalsh(0.5 * (sym + sym.T))[::-1]
    return GramianSpectrum(np.maximum(ev, 0.0), space, tol)


def moment_spectrum(stacked: np.ndarray, sys: LinearSystem, *, space: str = "X",
                    tol: float = 1e-12) -> GramianSpectrum:
    """Singular values of the moment matrix ``[X0; U0]``, in the ``U x H`` metric.

    By characterization ``item:wfl-synthesis`` of ``thm:willems-gramian`` the
    record is informative exactly when
    the range of the synthesis operator is dense, i.e. when this matrix has full
    row rank.
    Reported on the same footing as :func:`gramian_spectrum` so the two views
    can be compared.
    """
    S = gram_matrix(sys, space)
    L = np.linalg.cholesky(S)
    reordered = np.vstack([stacked[sys.n :], stacked[: sys.n]])  # [U0; X0] -> U x H order
    sv = np.linalg.svd(L.T @ reordered, compute_uv=False)
    return GramianSpectrum(sv**2, space, tol)
