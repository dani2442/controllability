"""The two data-driven controllability tests of the paper, on all three examples.

Each family is run twice: once in a configuration that is approximately
controllable, where both tests must reject every candidate, and once in a
configuration whose obstruction is known in closed form, where both must
recover it.  The two tests differ in what they are allowed to read.

``i-s``
    Proposition ``prop:data-fattorini-hautus``: the record is ``(ubar, xbar)``
    and the obstruction is a state functional ``eta`` with
    ``<eta, xbar(t)> = kappa e^{lambda t}``.  Implemented in
    :mod:`ddinf.controllability`.

``i-o``
    Theorem ``thm:io-window-controllability``: the record is ``(ubar, ybar)``
    and the obstruction is a functional of a length-``T`` input--output window
    whose value along the shifted record is a pure exponential.  Implemented in
    :mod:`ddinf.controllability_io`.

The two are run on different records, and that is the point rather than an
accident.  The state test needs the record to be informative on ``U x X``
(``lem:informative-gramian``), which the smooth multisine supplies.  The window test
needs the *shifted windows* to span the input--output behaviour, a strictly
stronger demand: a length-``T`` FIR kernel can null every line of a multisine
but one, so on such a record the window pairing ``<alpha, ubar_s>`` is itself a
pure exponential and the test returns one spurious obstruction per line.
:func:`multisine_window_failure` measures exactly that, and it is why the
window records use a broadband PRBS instead.

Nothing but the record enters either decision; the closed-form values and the
model-based Hautus modes are computed only to score them.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ddinf.controllability import data_driven_controllability, hautus_uncontrollable
from ddinf.controllability_io import io_shift_windows, io_window_controllability
from ddinf.delay import controllable_pair, delay_system, uncontrollable_pair
from ddinf.heat import heat_system
from ddinf.moments import moments, sine_tests
from ddinf.plotting import configure, family_colors, savefig, write_table
from ddinf.signals import Prbs, multisine
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import nearest, parser, tex_complex

NU = .02  # diffusivity; see the module docstring of ddinf.heat
SPEED = .5  # wave speed, chosen so the excited band covers the leading modes

RANK_TOL = 1e-10
RESIDUAL_TOL = 3e-3

# --- the window records -------------------------------------------------
# The window is sampled on the record's own grid, so the record is generated at
# the rate the window pairing is evaluated at and the sampled window relation
# ``y = F u + O x`` holds exactly.  WINDOW is the ``T`` of the theorem; SHIFTS
# is its ``2T + Theta``, and the record therefore runs to ``3T + Theta``.  A
# long shift range is what separates the two verdicts: a mode that the input
# reaches only weakly -- the constant Neumann mode is driven through ``nu``
# alone -- imitates an exponential over a short range and stops doing so over a
# long one.
WINDOW = 4.0
SHIFTS = 80.0
WINDOW_DT = .02
PRBS_DWELL = 2  # samples held per level
PRBS_AMPLITUDE = 3.0


class _Scaled:
    """A probing signal at a chosen amplitude; ``Prbs`` itself is always unit."""

    def __init__(self, signal, amplitude: float) -> None:
        self.signal, self.amplitude = signal, amplitude

    def __call__(self, t):
        return self.amplitude * self.signal(t)


def _prbs(length: float) -> _Scaled:
    return _Scaled(Prbs(PRBS_DWELL * WINDOW_DT, seed=3, horizon=length + WINDOW_DT),
                   PRBS_AMPLITUDE)


def _rng_state(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n) / np.sqrt(n)


def _cases(quality: str) -> dict:
    """``label -> (system, initial state, horizon, reference obstruction, space)``."""
    ne = 16 if quality == "quick" else 20
    # The window test is the binding constraint on the delay mesh: its
    # obstruction is reached through O_T^*, whose inversion is the unstable one
    # of sec:numerics, and on six delay elements the exponential fit
    # stalls just above the acceptance threshold.
    n_tau = 8 if quality == "quick" else 12

    heat_c = heat_system("neumann", n_elems=ne, nu=NU)
    xi_c = heat_c.meta["mesh"].nodes[heat_c.meta["free"]]
    x0_heat_c = 1.0 + np.cos(np.pi * xi_c) + .2 * np.cos(2 * np.pi * xi_c)
    heat_u = heat_system("dirichlet_sym", n_elems=ne, nu=NU)
    xi_u = heat_u.meta["mesh"].nodes[heat_u.meta["free"]]
    x0_heat_u = np.sin(np.pi * xi_u) + .2 * np.sin(2 * np.pi * xi_u)

    delay_u, delay_c = uncontrollable_pair(), controllable_pair()
    lambert = delay_u["obstruction"]
    delay_ref = complex(lambert[np.argmax(lambert.real)])

    def delay(data: dict):
        return delay_system(data["A0"], data["A1"], data["B0"], h=data["h"], n_tau=n_tau)

    wave_c = wave_system("dirichlet", n_elems=ne, speed=SPEED)
    wave_u = wave_system("dirichlet_sym", n_elems=ne, speed=SPEED)
    xi = wave_c.meta["mesh"].nodes[wave_c.meta["free"]]
    # The predicate of prop:data-fattorini-hautus needs kappa != 0, so the
    # initial state has to put energy into the mode that the input cannot
    # reach; a record in which the unreachable mode is simply absent carries no
    # evidence either way.
    x0_wave = np.concatenate([np.sin(np.pi * xi) + .2 * np.sin(2 * np.pi * xi),
                              np.zeros(xi.size)])

    return {
        "heat, one-sided": (heat_c, x0_heat_c, 8.0, None, "X"),
        "heat, symmetric": (heat_u, x0_heat_u, 8.0,
                            -NU * (2 * np.pi) ** 2, "X"),
        "wave, one-sided": (wave_c, x0_wave, 8.0, None, "W"),
        "wave, symmetric": (wave_u, x0_wave, 8.0, 2j * np.pi * SPEED, "W"),
        "delay, coupled": (delay(delay_c), _rng_state(delay(delay_c).n, 1), 10.0,
                           None, "W"),
        "delay, decoupled": (delay(delay_u), _rng_state(delay(delay_u).n, 1), 10.0,
                             delay_ref, "W"),
    }


def _state_test(sys, x0, horizon, space, signal, dt):
    """Proposition ``prop:data-fattorini-hautus`` on a state record."""
    rec = simulate(sys, signal, uniform_grid(horizon, dt), x0, theta=.5)
    mom = moments(rec, sine_tests(rec.t, max(30, sys.n + 8)))
    report = data_driven_controllability(
        mom, rec, sys, rank_tol=RANK_TOL, residual_tol=RESIDUAL_TOL,
        kappa_tol=1e-10, space=space)
    return report, mom.dynamics_residual(sys.A, sys.B)


def _window_test(sys, x0, *, shifts: float = SHIFTS):
    """Theorem ``thm:io-window-controllability`` on an input--output record."""
    length = shifts + WINDOW
    rec = simulate(sys, _prbs(length), uniform_grid(length, WINDOW_DT), x0, theta=.5)
    windows = io_shift_windows(rec, horizon=WINDOW, spread=shifts)
    return io_window_controllability(windows, n_states=sys.n, rank_tol=RANK_TOL,
                                     residual_tol=RESIDUAL_TOL)


def multisine_window_failure(quality: str = "quick") -> dict:
    """What the window test does on a record the theorem's hypothesis excludes.

    Run on the *controllable* one-sided heat equation with the multisine of the
    state test.  The shifted windows then span a few dozen behaviour directions
    instead of ``m n_w + n``, a length-``T`` kernel can isolate a single line,
    and the test reports its frequency as an obstruction.  Reported in the text,
    not in the table: it scores the record, not the system.
    """
    sys, x0, _, _, _ = _cases(quality)["heat, one-sided"]
    length = SHIFTS + WINDOW
    rec = simulate(sys, _multisine(), uniform_grid(length, WINDOW_DT), x0, theta=.5)
    windows = io_shift_windows(rec, horizon=WINDOW, spread=SHIFTS)
    report = io_window_controllability(windows, n_states=sys.n, rank_tol=RANK_TOL,
                                       residual_tol=RESIDUAL_TOL)
    return {"report": report, "n_spurious": len(report.obstructions),
            "frequencies": sorted(abs(c.lam.imag) for c in report.obstructions),
            "resolved": report.numerical_rank, "dimension": report.dimension}


def _multisine():
    return multisine(np.geomspace(.2, 20.0, 20), amps=np.geomspace(1.0, .2, 20),
                     seed=2)


def _row(label: str, truth: str, method: str, report, reference) -> str:
    """One table line; the configuration and its verdict are printed once."""
    found = [c.lam for c in report.obstructions]
    head = f"{label} & {truth}" if method == "i-s" else " & "
    cells = f"{report.numerical_rank}/{report.dimension} & {len(found)}"
    if reference is None:
        return f"{head} & {method} & {cells} & --- & --- \\\\"
    return (f"{head} & {method} & {cells} & {tex_complex(nearest(found, reference))}"
            f" & {tex_complex(reference)} \\\\")


def run(quality: str = "quick") -> dict:
    configure()
    dt = .002 if quality == "quick" else .001
    signal = _multisine()

    state_reports, window_reports, rows, diagnostics = {}, {}, [], {}
    for i, (label, (sys, x0, horizon, reference, space)) in enumerate(
            _cases(quality).items()):
        if i:  # blank line between configurations
            rows.append(r"\addlinespace")

        state, residual = _state_test(sys, x0, horizon, space, signal, dt)
        window = _window_test(sys, x0)
        state_reports[label], window_reports[label] = state, window

        model = [lam for lam, _ in hautus_uncontrollable(sys)]
        # The tabulated mark is the *true* property of the configuration, known
        # in closed form and never read from the data: a reference obstruction
        # was supplied exactly when the system fails to be approximately
        # controllable.  What each test produces is the count of surviving
        # candidates; the experiment succeeds when both agree with it.
        truth = r"\cmark" if reference is None else r"\xmark"
        for method, report in (("i-s", state), ("i-o", window)):
            verdict = r"\xmark" if report.obstructions else r"\cmark"
            if verdict != truth:
                raise AssertionError(
                    f"{label} [{method}]: test says {verdict}, truth is {truth}")
            rows.append(_row(label, truth, method, report, reference))

        entry = {
            "residual": residual,
            "state_found": len(state.obstructions),
            "window_found": len(window.obstructions),
            "state_resolved": (state.numerical_rank, state.dimension),
            "window_resolved": (window.numerical_rank, window.dimension),
            # The residual of the best *rejected* candidate: the margin by
            # which the window verdict cleared the acceptance threshold.
            "window_margin": min((c.exp_residual for c in window.candidates
                                  if not c.accepted), default=float("nan")),
            "window_residuals": [c.exp_residual for c in window.obstructions],
        }
        if reference is not None:
            lam_state = nearest([c.lam for c in state.obstructions], reference)
            lam_window = nearest([c.lam for c in window.obstructions], reference)
            lam_model = nearest(model, reference)
            # What each data-driven test controls is the distance to the
            # obstruction of the *discretised* system; the distance from that to
            # the closed form is the discretisation error of the mesh.
            entry |= {"lam_state": lam_state, "lam_window": lam_window,
                      "lam_model": lam_model, "reference": reference,
                      "state_error": abs(lam_state - lam_model),
                      "window_error": abs(lam_window - lam_model),
                      "discretisation_error": abs(lam_model - reference)}
        diagnostics[label] = entry

    # One colormap per family; within a family the darker shade is the
    # approximately controllable configuration, drawn solid, and the lighter
    # one its uncontrollable comparison, drawn dashed.
    shades = {family: iter(family_colors(family, 2))
              for family in ("heat", "wave", "delay")}
    colors = {label: next(shades[label.split(",")[0]]) for label in state_reports}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # The moment spectra have a few dozen points each and are read point by
    # point; the window spectra have four hundred and are read as a curve, so
    # markers there would merge into a band and hide the cliff.
    panels = ((axes[0], state_reports, "input--state moments", 2.5),
              (axes[1], window_reports, "input--output windows", 0.0))
    for ax, reports, title, marker_size in panels:
        for i, (label, report) in enumerate(reports.items()):
            s = report.singular_values
            ax.semilogy(np.arange(1, len(s) + 1), s / s[0],
                        "o-" if i % 2 == 0 else "o--", ms=marker_size, lw=1.1,
                        color=colors[label], label=label)
        ax.axhline(RANK_TOL, color="k", ls=":", lw=1)
        ax.set_xlabel("direction index $j$")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("normalized squared singular value")
    # Below the line and at the left edge: the only corner no spectrum reaches.
    axes[0].annotate("threshold $10^{-10}$", (1, RANK_TOL),
                     textcoords="offset points", xytext=(2, -4), va="top",
                     fontsize=6.5)
    handles = [Line2D([0], [0], color=colors[label], marker="o", ms=2.5,
                      ls="-" if i % 2 == 0 else "--", label=label)
               for i, label in enumerate(state_reports)]
    fig.tight_layout()
    # Anchored below the axes rather than inside them: the window spectra fill
    # their panel edge to edge and leave no free band for a legend.
    fig.legend(handles=handles, fontsize=6.2, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, .02), columnspacing=1.0, handlelength=1.6)
    fig_path = savefig(fig, "controllability.pdf")

    table = write_table("controllability.tex", r"""\begin{tabular}{l@{\hspace{4pt}}clrrll}
\toprule
\multicolumn{2}{l}{Configuration} & data & resolved & found & recovered $\lambda$ &
exact $\lambda$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "state_reports": state_reports,
            "window_reports": window_reports, "diagnostics": diagnostics}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, d in result["diagnostics"].items():
        print(label, d)
    print("multisine window record:", multisine_window_failure(args.quality))
