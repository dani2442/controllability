"""Numerics for *Data-Driven Control in Infinite-Dimensional Spaces* (``paper/``).

The library is grouped by what a module reads rather than by which example it
serves:

``ddinf.systems``
    the semi-discrete plants -- heat, wave, retarded equation -- behind the
    common :class:`~ddinf.systems.base.LinearSystem` interface, plus the P1
    finite elements and closed-form modal references they are built and scored
    against.
``ddinf.data``
    the measured record and everything computed from it alone: probing
    signals, the sampled record, weak moments, Gramian spectra.
``ddinf.controllability``
    the two data-driven Fattorini--Hautus tests, one per data class
    (``state`` reads ``(u, x)``, ``window`` reads ``(u, y)``).
``ddinf.lqr``
    the finite-horizon regulator: the Riccati reference and the two
    data-driven realizations (``graph`` reads ``(u, x, y)``, ``window`` reads
    ``(u, y)``).
``ddinf.paper``
    figure and table output into the paper source tree.
"""

from .data.records import Record, simulate, uniform_grid
from .systems import LinearSystem

__all__ = ["LinearSystem", "Record", "simulate", "uniform_grid"]
