"""Control Controllability Examples

This directory contains examples demonstrating the data-driven Hautus tests
for continuous-time linear systems.

Examples:
---------

1. basic_hautus_test.py
   Basic demonstration of the Hautus test comparing estimated vs. true matrices.

2. error_convergence.py
   Validates the O(T^{-1/2}) error rate from Proposition 1.

3. finite_candidate_check.py
   Demonstrates finite candidate λ checking from Corollary 2.

4. method_comparison.py
   Compares time-domain and FFT-based methods.

5. trajectory_visualization.py
   Various trajectory visualizations (2D, 3D, time series).

Usage:
------
Run any example from the examples directory:

    cd examples
    python basic_hautus_test.py

Or from the project root:

    python examples/basic_hautus_test.py

Requirements:
-------------
- torch
- torchsde
- numpy
- matplotlib
"""

from __future__ import annotations

from pathlib import Path
import typing as _t

# Directory where example scripts should save figures (project-root/images)
_IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(
    filename: str,
    fig=None,
    dpi: int = 150,
    filetype: _t.Optional[str] = None,
    **kwargs,
) -> Path:
    """Save a matplotlib figure into the repository `images/` folder.

    Args:
        filename: Basename of the image file, e.g. 'plot' or 'plot.png'.
        fig: Optional matplotlib Figure object. If None, `plt.savefig` is used.
        dpi: Dots-per-inch for the saved figure.
        filetype: Optional file type/extension to enforce, e.g. '.pdf' or 'png'.
        **kwargs: Passed to `savefig`.

    Behavior:
        - If `filetype` is provided, the file will be saved with that extension
          (leading dot optional). If `filename` already has an extension, it
          will be replaced by `filetype`.
        - If `filetype` is None, `filename` is used verbatim.

    Returns:
        Path to the saved image file.
    """
    # Normalize filetype
    ft = None
    if filetype:
        ft = filetype.lstrip('.')

    target = _IMAGES_DIR / filename
    if ft is not None:
        # Ensure extension matches requested filetype
        target = target.with_suffix('.' + ft)

    if fig is None:
        # Defer import so this module has no hard plt dependency at import-time
        import matplotlib.pyplot as plt

        save_kwargs = dict(dpi=dpi, **kwargs)
        if ft is not None:
            save_kwargs.setdefault('format', ft)

        plt.savefig(str(target), **save_kwargs)
    else:
        save_kwargs = dict(dpi=dpi, **kwargs)
        if ft is not None:
            save_kwargs.setdefault('format', ft)

        fig.savefig(str(target), **save_kwargs)

    return target
