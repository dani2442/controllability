"""Conditioning of the theoretical harmonic PE input against practical inputs.

Theorem ``thm:analytic-universal-sufficiency`` makes the harmonic signal of
``eq:harmonic-pe-example`` sufficient for *every* approximately controllable
analytic system, and the accumulation of its frequencies is what drives the
identity-theorem argument.  On a finite observation window and in finite
precision, that same accumulation is a liability, and this experiment measures
the price.  The heat equation is analytic and is the system the theorem covers;
the wave equation is run alongside it because its group is not analytic, so the
theorem says nothing there --- and the measured spectra look correspondingly
different.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.heat import heat_system
from ddinf.informativity import gramian_spectrum
from ddinf.plotting import configure, family_colors, savefig, write_table
from ddinf.signals import Prbs, harmonic_pe, multisine
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import parser, tex_num

TOL = 1e-10


def run(quality: str = "quick") -> dict:
    configure()
    dt = .004 if quality == "quick" else .002
    horizon = 10.0
    t = uniform_grid(horizon, dt)
    ne = 16 if quality == "quick" else 24

    systems = {
        "heat": (heat_system("neumann", n_elems=ne, nu=.02), "X"),
        "wave": (wave_system("dirichlet", n_elems=ne // 2, speed=.5), "W"),
    }
    signals = {
        "harmonic PE": harmonic_pe(10, sigma=.1),
        "multisine": multisine(np.geomspace(.2, 25.0, 20),
                               amps=np.geomspace(1.0, .2, 20), seed=4),
        "PRBS": Prbs(.04, seed=4, horizon=horizon),
    }

    spectra = {}
    for sys_name, (sys, space) in systems.items():
        for name, signal in signals.items():
            rec = simulate(sys, signal, t, np.zeros(sys.n), theta=.5)
            spectra[sys_name, name] = gramian_spectrum(rec, sys, space=space, tol=TOL)

    markers = dict(zip(signals, ("o", "s", "^")))
    fig = plt.figure(figsize=(7.0, 4.0))
    grid = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.8), hspace=.62, wspace=1.4)

    # Top row: the three probing inputs themselves, over the first quarter of
    # the observation window, where their shapes are still legible.
    t_show = t[t <= horizon / 2.5]
    for i, (name, signal) in enumerate(signals.items()):
        a = fig.add_subplot(grid[0, 2 * i:2 * i + 2])
        a.plot(t_show, np.atleast_2d(signal(t_show))[0], color=".25", lw=.8)
        a.set(xlabel=r"$t$", title=name)
        a.title.set_fontsize(8)
        a.grid(True, alpha=.25)
        if i == 0:
            a.set_ylabel(r"$u(t)$")

    # Bottom row: what each input resolves, one panel per system, in the shades
    # of that system's colormap.
    ax = [fig.add_subplot(grid[1, :3])]
    ax.append(fig.add_subplot(grid[1, 3:], sharey=ax[0]))
    for a, sys_name in zip(ax, systems):
        colors = family_colors(sys_name, len(signals))
        for color, name in zip(colors, signals):
            spec = spectra[sys_name, name]
            a.semilogy(np.arange(1, spec.dimension + 1),
                       spec.eigenvalues / spec.eigenvalues[0],
                       markers[name] + "-", ms=3, color=color, label=name)
        a.axhline(TOL, color="k", ls=":", lw=1)
        a.set(xlabel="Gramian eigenvalue index", title=sys_name)
        a.title.set_fontsize(8)
        a.grid(True, which="both", alpha=.25)
        a.legend(fontsize=6.5)
    ax[0].set(ylabel="normalized eigenvalue")
    # Below the line and at the left edge: the only corner no spectrum reaches.
    ax[0].annotate(r"threshold $10^{-10}$", (1, TOL), textcoords="offset points",
                   xytext=(2, -4), va="top", fontsize=6.5)
    fig_path = savefig(fig, "conditioning.pdf")

    rows = []
    for sys_name in systems:
        for name in signals:
            spec = spectra[sys_name, name]
            r = spec.numerical_rank
            # The smallest eigenvalue is at round-off for every input, so it
            # carries no information; what distinguishes the inputs is how far
            # the spectrum reaches before it falls through the threshold.
            discarded = ("---" if r == spec.dimension
                         else tex_num(spec.eigenvalues[r] / spec.eigenvalues[0]))
            rows.append(f"{sys_name} & {name} & {r}/{spec.dimension} & "
                        f"{tex_num(spec.eigenvalues[r - 1] / spec.eigenvalues[0])} & "
                        f"{discarded} \\\\")
        rows.append(r"\addlinespace")
    table = write_table("conditioning.tex", r"""\begin{tabular}{llrrr}
\toprule
Example & input & numerical rank & last resolved & first discarded \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "spectra": spectra}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for key, spec in result["spectra"].items():
        print(key, spec.numerical_rank, "/", spec.dimension)
