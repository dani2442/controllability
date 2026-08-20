"""Probing inputs for the record.

The default is the paper's own harmonically persistently exciting signal
(``lem:harmonic-pe``) multiplied by ``e^{sigma t}``, which
Theorem ``thm:analytic-universal-sufficiency`` shows to be sufficient for
informativity of *every* approximately controllable analytic system.  A plain
well-separated multisine is provided for comparison: the frequency clustering
``omega_k -> nu_j`` that makes the harmonic signal theoretically sufficient is
also what makes it numerically stiff, and ``exp04`` quantifies the difference.

Signals with an exponential-sum representation carry their ``(coefficient,
exponent)`` pairs, so that a frequency of the probing input can be compared
directly with a recovered obstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpSum:
    """``u(t) = Re( sum_k coeffs[k] exp(exponents[k] t) )`` with values in ``R^m``."""

    coeffs: np.ndarray  # (K, m) complex
    exponents: np.ndarray  # (K,) complex
    label: str = "exp-sum"

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 2:
            raise ValueError("coeffs must be (K, m)")
        if self.coeffs.shape[0] != self.exponents.shape[0]:
            raise ValueError("coeffs and exponents disagree in length")

    @property
    def m(self) -> int:
        return self.coeffs.shape[1]

    def __call__(self, t: np.ndarray | float) -> np.ndarray:
        """Values at ``t``; shape ``(m,)`` for scalar ``t``, else ``(m, len(t))``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        phases = np.exp(np.outer(self.exponents, t_arr))  # (K, T)
        vals = np.real(self.coeffs.T @ phases)  # (m, T)
        return vals[:, 0] if np.isscalar(t) or np.ndim(t) == 0 else vals


def harmonic_pe(n_terms: int = 8, sigma: float = 0.0, m: int = 1,
                directions: np.ndarray | None = None) -> ExpSum:
    """``e^{sigma t} h(t)`` with ``h`` the harmonic PE signal of ``eq:harmonic-pe-example``.

    ``h(t) = sum_{j,k} 2^{-j-k} cos(omega_{j,k} t) phi_j`` with
    ``omega_{j,k} = 2 - 2^{-j} - 2^{-j-k}``.  For each direction ``phi_j`` the
    frequencies increase to ``nu_j = 2 - 2^{-j}``; it is this accumulation that
    drives the identity-theorem argument of the paper.

    ``n_terms`` truncates the inner sum over ``k``; ``m`` is the input
    dimension, and ``directions`` (``(m, m)``, columns ``phi_j``) defaults to
    the standard basis.
    """
    if directions is None:
        directions = np.eye(m)
    coeffs: list[np.ndarray] = []
    exponents: list[complex] = []
    for j in range(1, m + 1):
        phi_j = directions[:, j - 1]
        for k in range(1, n_terms + 1):
            omega = 2.0 - 2.0 ** (-j) - 2.0 ** (-j - k)
            amp = 2.0 ** (-j - k)
            # cos(omega t) = (e^{i omega t} + e^{-i omega t}) / 2
            coeffs.append(0.5 * amp * phi_j)
            exponents.append(sigma + 1j * omega)
            coeffs.append(0.5 * amp * phi_j)
            exponents.append(sigma - 1j * omega)
    return ExpSum(np.array(coeffs, dtype=complex), np.array(exponents, dtype=complex),
                  label=f"harmonic PE (K={n_terms}, sigma={sigma:g})")


def multisine(freqs: np.ndarray, *, amps: np.ndarray | None = None,
              phases: np.ndarray | None = None, sigma: float = 0.0,
              m: int = 1, seed: int = 0) -> ExpSum:
    """Well-separated multisine ``e^{sigma t} sum_k a_k cos(omega_k t + varphi_k)``."""
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    rng = np.random.default_rng(seed)
    if amps is None:
        amps = np.ones_like(freqs)
    if phases is None:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(freqs), m))
    coeffs: list[np.ndarray] = []
    exponents: list[complex] = []
    for k, omega in enumerate(freqs):
        for i in range(m):
            e_i = np.zeros(m)
            e_i[i] = 1.0
            coeffs.append(0.5 * amps[k] * np.exp(1j * phases[k, i]) * e_i)
            exponents.append(sigma + 1j * omega)
            coeffs.append(0.5 * amps[k] * np.exp(-1j * phases[k, i]) * e_i)
            exponents.append(sigma - 1j * omega)
    return ExpSum(np.array(coeffs, dtype=complex), np.array(exponents, dtype=complex),
                  label=f"multisine ({len(freqs)} lines, sigma={sigma:g})")


@dataclass(frozen=True)
class Prbs:
    """Pseudo-random binary sequence held over intervals of length ``dwell``.

    No exponential-sum representation, hence no closed-form mild solution; used
    only as a practical baseline in the conditioning study.
    """

    dwell: float
    m: int = 1
    seed: int = 0
    horizon: float = 100.0
    label: str = "PRBS"

    def _levels(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = int(np.ceil(self.horizon / self.dwell)) + 2
        return rng.choice([-1.0, 1.0], size=(n, self.m))

    def __call__(self, t: np.ndarray | float) -> np.ndarray:
        levels = self._levels()
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.clip((t_arr / self.dwell).astype(int), 0, levels.shape[0] - 1)
        vals = levels[idx].T  # (m, T)
        return vals[:, 0] if np.ndim(t) == 0 else vals
