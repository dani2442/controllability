"""The record and the quantities read off it.

Nothing here knows a model: :mod:`ddinf.data.signals` designs the probing
input, :mod:`ddinf.data.records` samples the trajectory it generates,
:mod:`ddinf.data.moments` evaluates the dynamic synthesis operator on it, and
:mod:`ddinf.data.informativity` reports how many directions the result
resolves.
"""

from .records import Record, simulate, uniform_grid

__all__ = ["Record", "simulate", "uniform_grid"]
