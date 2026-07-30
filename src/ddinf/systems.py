"""The plug-in interface between the discretised examples and the algorithms.

Every example (heat equation, retarded equation, ...) is reduced to a
:class:`LinearSystem`: a semi-discrete realisation ``x' = A x + B u``,
``y = C x`` together with the two Gram matrices that represent the Hilbert
structures the theory uses,

* ``MX`` -- the inner product of the state space ``X`` (the FEM mass matrix),
* ``MW`` -- the inner product of the finer space ``W`` in which informativity is
  required when ``C`` is unbounded (Definition ``def:hilbert-informative-W``).

Nothing downstream of this module knows which PDE it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LinearSystem:
    """Semi-discrete realisation plus the Hilbert structures of ``X`` and ``W``."""

    name: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    MX: np.ndarray
    MW: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B must be (n, m)")
        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError("C must be (p, n)")
        for name, G in (("MX", self.MX), ("MW", self.MW)):
            if G.shape != (n, n):
                raise ValueError(f"{name} must be (n, n)")

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def p(self) -> int:
        return self.C.shape[0]

    # -- Hilbert structure -------------------------------------------------

    def inner_X(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``<a, b>_X``; broadcasts over trailing columns."""
        return np.einsum("i...,ij,j...->...", a, self.MX, b)

    def inner_W(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``<a, b>_W``."""
        return np.einsum("i...,ij,j...->...", a, self.MW, b)

    def norm_X(self, a: np.ndarray) -> float:
        return float(np.sqrt(max(self.inner_X(a, a), 0.0)))

    def norm_W(self, a: np.ndarray) -> float:
        return float(np.sqrt(max(self.inner_W(a, a), 0.0)))

    def control_operator_norm(self) -> float:
        """``||B||_{L(U, X)}`` -- the discrete shadow of the unboundedness of ``B``."""
        # largest singular value of B measured in the X norm on the range
        LX = np.linalg.cholesky(self.MX)
        return float(np.linalg.norm(LX.T @ self.B, 2))
