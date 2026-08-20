"""Data-driven approximate controllability, one module per data class.

:mod:`ddinf.controllability.state` implements the input--state test of
``prop:data-fattorini-hautus``; :mod:`ddinf.controllability.window` implements
the input--output window test of ``thm:io-window-controllability``, which reads
no state at all.
"""

from .state import (ControllabilityReport, ModeCandidate,
                    data_driven_controllability, hautus_uncontrollable)
from .window import (WindowCandidate, WindowRecord, io_shift_windows,
                     io_window_controllability)

__all__ = ["ControllabilityReport", "ModeCandidate", "WindowCandidate",
           "WindowRecord", "data_driven_controllability",
           "hautus_uncontrollable", "io_shift_windows",
           "io_window_controllability"]
