"""Finite-horizon LQR reconstructed from shifted windows of one record."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.heat import heat_system
from ddinf.lqr_data import shift_library, solve_data_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid
from experiments.common import parser, tex_num


def run(quality: str = "quick") -> dict:
    configure()
    dt = .002 if quality == "quick" else .001
    horizon = .3
    sys = heat_system("dirichlet", n_elems=4 if quality == "quick" else 6, nu=.02)
    t = uniform_grid(horizon, dt)
    weights = LqrWeights.make(sys, terminal=.1, control=.5)
    xi = sys.meta["mesh"].nodes[sys.meta["free"]]
    x0 = np.sin(np.pi * xi)
    ref = riccati_hamiltonian(sys, weights, t)
    optimal = ref.closed_loop(x0)
    optimum = ref.optimal_cost(x0)

    max_n = 512 if quality == "quick" else 800
    long_t = uniform_grid(3.0, dt)
    record = simulate(sys, Prbs(5 * dt, seed=3, horizon=3.0), long_t,
                      np.zeros(sys.n), theta=.5)
    sizes = [160, 256, max_n]
    solutions = []
    for n_library in sizes:
        spread = (n_library - 1) * dt
        library = shift_library(record, horizon, n_library, spread=spread)
        solutions.append(solve_data_lqr(library, sys, weights, x0, rho=1e-8))

    rel_cost = np.array([abs(s.cost - optimum) / optimum for s in solutions])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].plot(t, optimal.u[0], "k--", label="Riccati")
    ax[0].plot(t, solutions[-1].record.u[0], label="data-driven")
    ax[0].set(xlabel=r"$t$", ylabel=r"$u(t)$")
    ax[0].legend()
    ax[0].grid(True, alpha=.25)
    ax[1].loglog(sizes, rel_cost, "o-", label="relative cost error")
    ax[1].loglog(sizes, [s.initial_defect for s in solutions], "s-",
                 label=r"$W$-initial defect")
    ax[1].set(xlabel="library size")
    ax[1].grid(True, which="both", alpha=.25)
    ax[1].legend()
    fig_path = savefig(fig, "lqr.pdf")

    rows = "\n".join(
        f"{n} & {tex_num(sol.cost)} & {tex_num(err)} & "
        f"{tex_num(sol.initial_defect)} \\\\"
        for n, sol, err in zip(sizes, solutions, rel_cost)
    )
    table = write_table("lqr.tex", rf"""\begin{{tabular}}{{rrrr}}
\toprule
Library size & $J_{{\rm dd}}$ & relative cost error & initial defect \\
\midrule
{rows}
\bottomrule
\end{{tabular}}""")
    return {"figure": fig_path, "table": table, "solutions": solutions,
            "relative_cost": rel_cost, "optimal_cost": optimum}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    print("relative cost errors:", result["relative_cost"])
