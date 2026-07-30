"""Numerics for *Data-Driven Control in Infinite-Dimensional Spaces* (paper_wfl2)."""

from .systems import LinearSystem
from .timestepping import Record, simulate, uniform_grid

__all__ = ["LinearSystem", "Record", "simulate", "uniform_grid"]
