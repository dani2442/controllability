"""Shared plotting and LaTeX-output helpers for the paper experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper_wfl2" / "figures"
TABLES = ROOT / "paper_wfl2" / "tables"


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
