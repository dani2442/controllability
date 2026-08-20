"""Closed-form modal references for the heat equation examples.

These are the ground truth the finite-element discretisation of
:mod:`ddinf.heat` is measured against.  Each configuration is diagonal in its
own eigenbasis, ``A phi_n = lambda_n phi_n``, and the modal control
coefficients ``b_n`` decide the Fattorini--Hautus obstructions in closed form:
``b_n = 0`` is exactly the ``n``-th mode being unreachable.  That is what makes
``dirichlet_sym`` usable as the uncontrollable comparison of ``exp02`` -- its
obstruction is known before any data is generated.

For boundary control the modal coefficients come from a lift.  With ``D`` a
fixed function carrying the inhomogeneous boundary condition,
``d_n = <D, phi_n>`` and integration by parts gives ``b_n = -lambda_n d_n``.

Modal data (see Pritchard--Salamon Ex. 4.6 and ``sections/lqr.tex``); the last
row is the verbatim example of the cited paper and includes the constant mode:

===========================  =======================  ==================================  =========================
example                      ``lambda_n``             ``phi_n``                           ``b_n``
===========================  =======================  ==================================  =========================
Dirichlet control at 0       ``-nu n^2 pi^2``         ``sqrt2 sin(n pi xi)``              ``sqrt2 nu n pi``
Dirichlet control both ends  ``-nu n^2 pi^2``         ``sqrt2 sin(n pi xi)``              ``sqrt2 nu n pi (1-(-1)^n)``
Neumann at both endpoints    ``-nu n^2 pi^2``, n>=0  ``1`` (n=0), ``sqrt2 cos(n pi xi)`` ``-nu phi_n(0)``
===========================  =======================  ==================================  =========================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ModalSystem:
    """Diagonal realisation of an example in its own eigenbasis."""

    name: str
    lam: np.ndarray  # (N,)
    b: np.ndarray  # (N, m) modal control coefficients
    phi: Callable[[np.ndarray], np.ndarray]  # xi -> (N, len(xi))
    lift: Callable[[np.ndarray], np.ndarray] | None = None  # D(xi)

    def uncontrollable_modes(self, tol: float = 1e-12) -> np.ndarray:
        """Indices ``n`` with ``b_n = 0`` -- the Fattorini--Hautus obstructions."""
        return np.where(np.linalg.norm(self.b, axis=1) <= tol)[0]


def heat_modal(kind: str = "dirichlet", n_modes: int = 400, nu: float = 1.0) -> ModalSystem:
    """Closed-form modal data for the three heat-equation configurations.

    ``nu`` is the diffusivity of ``x_t = nu x_xx``.  For Dirichlet control it
    scales ``b_n = -lambda_n <D, phi_n>``; for Neumann control the weak
    boundary term gives ``b_n = -nu phi_n(0)`` directly.
    """
    if kind in ("dirichlet", "dirichlet_sym"):
        n = np.arange(1, n_modes + 1)
        lam = -nu * ((n * np.pi) ** 2)

        def phi(xi: np.ndarray, n=n) -> np.ndarray:
            return np.sqrt(2.0) * np.sin(np.outer(n * np.pi, np.asarray(xi, dtype=float)))

        if kind == "dirichlet":
            # D u = (1-xi) u,  <D, phi_n> = sqrt(2)/(n pi)
            lift = lambda xi: 1.0 - np.asarray(xi, dtype=float)  # noqa: E731
            d = np.sqrt(2.0) / (n * np.pi)
        else:
            # D u = u (both ends),  <1, phi_n> = sqrt(2)(1-(-1)^n)/(n pi), so the
            # even modes -- those antisymmetric about xi = 1/2 -- are unreachable
            lift = lambda xi: np.ones_like(np.asarray(xi, dtype=float))  # noqa: E731
            d = np.sqrt(2.0) * (1.0 - (-1.0) ** n) / (n * np.pi)
        return ModalSystem(f"heat-{kind}", lam, (-lam * d)[:, None], phi, lift=lift)

    if kind == "neumann":
        # Neumann control at xi = 0, homogeneous Neumann data at xi = 1
        # (Pritchard--Salamon Ex. 4.6).  Indexing starts at n=0 because the
        # homogeneous generator has the normalized constant eigenfunction.
        n = np.arange(n_modes)
        lam = -nu * (n * np.pi) ** 2

        def phi(xi: np.ndarray, n=n) -> np.ndarray:
            xi = np.asarray(xi, dtype=float)
            values = np.cos(np.outer(n * np.pi, xi))
            if n.size > 1:
                values[1:] *= np.sqrt(2.0)
            return values

        b = -nu * phi(np.array([0.0]))[:, 0][:, None]  # b_n = -nu phi_n(0)
        return ModalSystem("heat-neumann", lam, b, phi)

    raise ValueError(f"unknown heat configuration {kind!r}")
