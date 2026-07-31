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
from ddinf.plotting import configure, savefig, write_table
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
        "heat": (heat_system("dirichlet", n_elems=ne, nu=.02), "X"),
        "wave": (wave_system("dirichlet", n_elems=ne // 2, speed=.5), "W"),
    }
    signals = {
        "harmonic PE": harmonic_pe(10),
        "multisine": multisine(np.geomspace(.2, 25.0, 20),
                               amps=np.geomspace(1.0, .2, 20), seed=4),
        "PRBS": Prbs(.04, seed=4, horizon=horizon),
    }

    spectra = {}
    for sys_name, (sys, space) in systems.items():
        for name, signal in signals.items():
            rec = simulate(sys, signal, t, np.zeros(sys.n), theta=.5)
            spectra[sys_name, name] = gramian_spectrum(rec, sys, space=space, tol=TOL)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for a, sys_name in zip(ax, systems):
        for name, marker in zip(signals, ("o", "s", "^")):
            spec = spectra[sys_name, name]
            a.semilogy(np.arange(1, spec.dimension + 1),
                       spec.eigenvalues / spec.eigenvalues[0], marker + "-", ms=3,
                       label=name)
        a.axhline(TOL, color="k", ls="--", lw=1)
        a.set(xlabel="Gramian eigenvalue index", title=sys_name)
        a.grid(True, which="both", alpha=.25)
    ax[0].set(ylabel="normalized eigenvalue")
    ax[0].legend(fontsize=7)
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
