"""CLI and output utilities shared by all experiments."""

from __future__ import annotations

import argparse

import numpy as np


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quality", choices=("quick", "paper"), default="quick",
                   help="quick is intended for iteration; paper uses finer grids")
    return p


def tex_num(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        return f"\\num{{{value:.{digits}e}}}"
    return f"\\num{{{value:.{digits}g}}}"


def tex_complex(value: complex, digits: int = 4) -> str:
    """``a`` or ``a \\pm b i`` -- obstructions of real systems come in pairs."""
    z = complex(value)
    if not np.isfinite(z):
        return "---"
    scale = max(abs(z), 1.0)
    re = 0.0 if abs(z.real) < 1e-8 * scale else z.real
    if abs(z.imag) < 1e-8 * scale:
        return f"${re:.{digits}g}$"
    return f"${re:.{digits}g}\\pm{abs(z.imag):.{digits}g}\\mathrm{{i}}$"


def nearest(values, target: complex) -> complex:
    """The entry of ``values`` closest to ``target`` (``nan`` if there is none).

    Obstructions must be matched to their reference by proximity, never by
    ordering: the semi-discrete generators here carry high-frequency modes whose
    position in any ordering is an artefact of the mesh.
    """
    arr = np.asarray(list(values), dtype=complex)
    if arr.size == 0:
        return complex("nan")
    return complex(arr[np.argmin(np.abs(arr - target))])
