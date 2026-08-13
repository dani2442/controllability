"""Shared plotting and LaTeX-output helpers for the paper experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper_wfl2" / "figures"
TABLES = ROOT / "paper_wfl2" / "tables"

# One graded colormap per example family, so that a colour identifies the
# system across every figure of the paper and the shades within it identify the
# configuration, the probing input or the method.
FAMILY_COLORMAP = {"heat": "Blues", "wave": "Oranges", "delay": "Greens"}


def family_colors(family: str, n: int = 1, *, dark: float = .85,
                  light: float = .45) -> list:
    """``n`` shades of the family colormap, darkest first.

    The range stops short of both ends of the colormap: the palest shades are
    invisible on paper and the darkest are indistinguishable from black.
    """
    cmap = plt.get_cmap(FAMILY_COLORMAP[family])
    levels = [dark] if n == 1 else np.linspace(dark, light, n)
    return [cmap(float(level)) for level in levels]


def configure() -> None:
    """Apply a compact style that embeds fonts and remains legible in two columns."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> Path:
    configure()
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


def write_table(name: str, body: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path
