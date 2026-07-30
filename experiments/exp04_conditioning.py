"""Conditioning of theoretical harmonic PE versus practical inputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.heat import heat_system
from ddinf.informativity import gramian_spectrum
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import Prbs, harmonic_pe, multisine
from ddinf.timestepping import simulate, uniform_grid
from experiments.common import parser, tex_num


def run(quality: str = "quick") -> dict:
    configure()
    dt = .004 if quality == "quick" else .002
    sys = heat_system("dirichlet", n_elems=16, nu=.02)
    t = uniform_grid(10.0, dt)
    signals = {
        "harmonic PE": harmonic_pe(10),
        "multisine": multisine(np.geomspace(.2, 25.0, 20),
                               amps=np.geomspace(1.0, .2, 20), seed=4),
        "PRBS": Prbs(.04, seed=4, horizon=10.0),
    }
    spectra = {}
    for name, signal in signals.items():
        rec = simulate(sys, signal, t, np.zeros(sys.n), theta=.5)
        spectra[name] = gramian_spectrum(rec, sys, tol=1e-10)

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for name, spec in spectra.items():
        ax.semilogy(np.arange(1, spec.dimension + 1),
                    spec.eigenvalues / spec.eigenvalues[0], "o-", ms=3, label=name)
    ax.axhline(1e-10, color="k", ls="--", lw=1)
    ax.set(xlabel="Gramian eigenvalue index", ylabel="normalized eigenvalue")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig_path = savefig(fig, "conditioning.pdf")

    rows = "\n".join(
        f"{name} & {spec.numerical_rank}/{spec.dimension} & "
        f"{tex_num(spec.eigenvalues[-1] / spec.eigenvalues[0])} \\\\"
        for name, spec in spectra.items()
    )
    table = write_table("conditioning.tex", rf"""\begin{{tabular}}{{lrr}}
\toprule
Input & numerical rank & normalized smallest eigenvalue \\
\midrule
{rows}
\bottomrule
\end{{tabular}}""")
    return {"figure": fig_path, "table": table, "spectra": spectra}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    print({k: v.numerical_rank for k, v in result["spectra"].items()})
