"""The semi-discrete examples, behind one interface.

:class:`~ddinf.systems.base.LinearSystem` is the plug-in point: a realisation
``x' = A x + B u``, ``y = C x`` plus the Gram matrices of the two Hilbert
structures the theory uses.  :mod:`ddinf.systems.fem` supplies the P1
ingredients, :mod:`ddinf.systems.heat`, :mod:`ddinf.systems.wave` and
:mod:`ddinf.systems.delay` the three examples of the paper, and
:mod:`ddinf.systems.modal` the closed-form spectra they are validated against.
"""

from .base import LinearSystem

__all__ = ["LinearSystem"]
