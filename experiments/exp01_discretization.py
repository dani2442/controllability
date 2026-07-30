"""Spatial convergence and discrete boundary-control growth."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.heat import fem_eigenvalues, heat_system
from ddinf.plotting import configure, savefig, write_table
from experiments.common import parser, tex_num


def run(quality: str = "quick") -> dict:
    configure()
    meshes = np.array([8, 16, 32, 64] if quality == "quick" else [8, 16, 32, 64, 128])
    kinds = ("dirichlet", "neumann")
    errors: dict[str, list[float]] = {k: [] for k in kinds}
    norms: dict[str, list[float]] = {k: [] for k in kinds}
    for kind in kinds:
        exact = 0.0 if kind == "neumann" else -np.pi**2
        mode = 0
        for ne in meshes:
            sys = heat_system(kind, n_elems=int(ne))
            errors[kind].append(abs(fem_eigenvalues(sys, mode + 1)[mode].real - exact)
                                if exact else abs(fem_eigenvalues(sys, 2)[1].real + np.pi**2))
            norms[kind].append(sys.control_operator_norm())

    h = 1.0 / meshes
    rates = {k: float(np.polyfit(np.log(h), np.log(errors[k]), 1)[0]) for k in kinds}
    growth = {k: float(-np.polyfit(np.log(h), np.log(norms[k]), 1)[0]) for k in kinds}

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for kind, marker in zip(kinds, ("o", "s")):
        ax[0].loglog(h, errors[kind], marker=marker, label=kind)
        ax[1].loglog(h, norms[kind], marker=marker, label=kind)
    ax[0].loglog(h, errors["dirichlet"][-1] * (h / h[-1]) ** 2, "k--", label=r"$h^2$")
    ax[0].set(xlabel=r"$h$", ylabel=r"first nonzero eigenvalue error")
    ax[1].set(xlabel=r"$h$", ylabel=r"$\|B_h\|_{\mathcal{L}(U,X_h)}$")
    for a in ax:
        a.grid(True, which="both", alpha=.25)
        a.legend()
    fig_path = savefig(fig, "discretization.pdf")

    rows = "\n".join(
        f"{kind.capitalize()} & {rates[kind]:.2f} & {growth[kind]:.2f} & "
        f"{tex_num(errors[kind][-1])} \\\\"
        for kind in kinds
    )
    table = write_table("discretization.tex", rf"""\begin{{tabular}}{{lrrr}}
\toprule
Boundary condition & eig. order & $B_h$ exponent & finest eig. error \\
\midrule
{rows}
\bottomrule
\end{{tabular}}""")
    return {"figure": fig_path, "table": table, "rates": rates, "growth": growth}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    print(run(args.quality))
