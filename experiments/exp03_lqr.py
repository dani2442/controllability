"""Finite-horizon LQR reconstructed from shifted windows of one record.

Run on all three examples.  The library, the cost and the initial-state penalty
are assembled from measured signals only (Remark ``rmk:lqr-numerics``); the
Riccati solution is computed separately, purely as the reference the
data-driven regulator is scored against.

The library size is reported as ``N / n_T``, where ``n_T = T/dt + 1`` is
the number of samples in the control window.  That is the scale that matters: the sampled
behaviour restricted to a window has dimension ``m*n_T + n``, so a library of
fewer than ``n_T`` shifts cannot span it and the constrained minimum is taken
over a strictly smaller set than the true one.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from ddinf.delay import controllable_pair, delay_system
from ddinf.heat import heat_system
from ddinf.lqr_data import shift_library, solve_data_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import parser, tex_num

RHO = 1e-8


def _cases(quality: str) -> dict:
    """``label -> (system, x0, horizon, record length, PRBS dwell in samples)``."""
    fine = quality == "paper"
    heat = heat_system("neumann", n_elems=6 if fine else 4, nu=.02)
    xi_h = heat.meta["mesh"].nodes[heat.meta["free"]]

    wave = wave_system("dirichlet", n_elems=4 if fine else 3, speed=.5)
    xi_w = wave.meta["mesh"].nodes[wave.meta["free"]]

    data = controllable_pair()
    delay = delay_system(data["A0"], data["A1"], data["B0"], h=data["h"],
                         n_tau=4 if fine else 3)

    return {
        "heat": (heat, 1.0 + .2 * np.cos(np.pi * xi_h), .3, 3.0, 3),
        "wave": (wave, np.concatenate([np.sin(np.pi * xi_w), np.zeros(xi_w.size)]),
                 1.0, 8.0, 4),
        "delay": (delay, np.ones(delay.n) / np.sqrt(delay.n), 1.0, 8.0, 4),
    }


def run(quality: str = "quick") -> dict:
    configure()
    dt = .005 if quality == "quick" else .0025
    results = {}

    for label, (sys, x0, horizon, length, dwell) in _cases(quality).items():
        t = uniform_grid(horizon, dt)
        weights = LqrWeights.make(sys, terminal=.1, control=.5)
        ref = riccati_hamiltonian(sys, weights, t)
        optimal = ref.closed_loop(x0)
        optimum = ref.optimal_cost(x0)

        record = simulate(sys, Prbs(dwell * dt, seed=3, horizon=length),
                          uniform_grid(length, dt), np.zeros(sys.n), theta=.5)
        n_T = t.size
        sizes = [n_T // 2, n_T, 2 * n_T]
        solutions = [solve_data_lqr(shift_library(record, horizon, n), sys, weights,
                                    x0, rho=RHO) for n in sizes]
        results[label] = {
            "t": t, "n_T": n_T, "sizes": sizes, "optimal": optimal,
            "optimum": optimum, "solutions": solutions,
            "rel_cost": np.array([abs(s.cost - optimum) / abs(optimum)
                                  for s in solutions]),
        }

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    heat = results["heat"]
    best = heat["solutions"][1]  # N = n_T
    for a, signal, name in ((ax[0], "u", r"$u(t)$"), (ax[1], "y", r"$y(t)$")):
        a.plot(heat["t"], getattr(heat["optimal"], signal)[0], "k--", label="Riccati")
        a.plot(heat["t"], getattr(best.record, signal)[0], lw=.9, label="data-driven")
        a.set(xlabel=r"$t$", ylabel=name)
        a.grid(True, alpha=.25)
    ax[0].set_title("heat: input", fontsize=8)
    ax[1].set_title("heat: output", fontsize=8)
    ax[0].legend(fontsize=7)
    for label, marker in zip(results, ("o", "s", "^")):
        res = results[label]
        ratio = np.array(res["sizes"]) / res["n_T"]
        ax[2].loglog(ratio, res["rel_cost"], marker + "-", ms=4, label=label)
    ax[2].set(xlabel=r"$N/n_T$", ylabel="relative cost error")
    ax[2].set_xticks([.5, 1.0, 2.0], [r"$1/2$", r"$1$", r"$2$"])
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].set_title("convergence", fontsize=8)
    ax[2].grid(True, which="both", alpha=.25)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig_path = savefig(fig, "lqr.pdf")

    rows = []
    for label, res in results.items():
        for n, sol, err in zip(res["sizes"], res["solutions"], res["rel_cost"]):
            rows.append(f"{label} & {n} & {n / res['n_T']:.1f} & "
                        f"{tex_num(sol.cost)} & {tex_num(err)} & "
                        f"{tex_num(sol.initial_defect)} \\\\")
        rows.append(r"\addlinespace")
    table = write_table("lqr.tex", r"""\begin{tabular}{lrrrrr}
\toprule
Example & $N$ & $N/n_T$ & $J_{\rm dd}$ & relative cost error &
initial defect \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "results": results}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, res in result["results"].items():
        print(label, "J* =", res["optimum"], "rel errors:", res["rel_cost"])
