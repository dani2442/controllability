"""Finite-horizon LQR reconstructed from shifted windows of one record.

Run on all three examples.  The library, the cost and the initial-state penalty
are assembled from measured signals only (Remark ``rmk:lqr-numerics``); the
Riccati solution is computed separately, purely as the reference the
data-driven regulator is scored against.

The library size is reported as ``N / n_T``, where ``n_T = T/dt + 1`` is the
number of samples in the control window.  That is the scale that matters: the
sampled behaviour restricted to a window has dimension ``m*n_T + n``, so a
library of fewer than ``m*n_T + n`` shifts cannot span it and the constrained
minimum is taken over a strictly smaller set than the true one.

Two traps the shift construction has to avoid, both of which collapse the
library well below its nominal size:

* if the shift step is an integer multiple of the PRBS dwell, every window sees
  the probing input on the *same* dwell grid and the span of library inputs
  drops to ``n_T/dwell`` directions.  :func:`_dwell_safe` picks the record
  length so that this does not happen at any of the reported sizes;
* reaching full rank is necessary but not sufficient: at exactly ``m*n_T + n``
  the library is a square basis and the penalised solve is badly conditioned.
  Comfortable redundancy -- in practice ``2 n_T`` -- is what makes it accurate.
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
SHOWCASE = "wave"  # the example whose input and output are plotted


def _dwell_safe(length: float, horizon: float, dt: float, dwell: int,
                sizes) -> float:
    """Lengthen the record until no reported library size is dwell-aligned.

    ``shift_library`` spreads ``N`` starts over ``[0, n_rec - n_T]``; when every
    start falls in the same residue class modulo ``dwell`` all windows carry the
    probing input on the same dwell grid, the library inputs are piecewise
    constant on that grid, and their span collapses to ``n_T/dwell`` directions.
    Individual steps may be multiples of ``dwell`` without harm -- what matters
    is that the starts are not all congruent.
    """
    n_T = int(round(horizon / dt)) + 1

    def aliased(n_rec: int, N: int) -> bool:
        starts = np.unique(np.round(np.linspace(0, n_rec - n_T, N)).astype(int))
        return np.unique(starts % dwell).size == 1

    for extra in range(400):
        n_rec = int(round(length / dt)) + 1 + extra
        if not any(aliased(n_rec, N) for N in sizes):
            return (n_rec - 1) * dt
    raise RuntimeError("no dwell-safe record length found")


def _cases(quality: str) -> dict:
    """``label -> (system, x0, horizon, dt, record length, PRBS dwell in samples)``.

    Each example carries its own step: the three have very different natural
    time scales, and what has to be resolved is the control window, not a common
    clock.  ``n_T`` is kept near 400 everywhere so that the ``2 n_T`` library is
    the same size in all three and the convergence panel compares like with
    like.
    """
    fine = quality == "paper"
    # nu = 1 here, unlike exp02 and exp04.  Those need the probing input to
    # excite the leading modes, which is what nu = 0.02 buys; this experiment
    # instead needs the library to span the windowed behaviour, and that is
    # independent of nu.  At nu = 0.02 the Neumann flux nu*u gives the control so
    # little authority over [0, T] that the optimal input is essentially zero --
    # a regulator that does nothing is a poor test of a regulator.
    heat = heat_system("neumann", n_elems=6 if fine else 4, nu=1.0)
    xi_h = heat.meta["mesh"].nodes[heat.meta["free"]]

    # The showcase panels need a regulator that visibly does something.  With
    # the leading frequency omega_1 = pi c = pi/2, a horizon of 4 is one full
    # period of the mode the initial state excites: the optimal control
    # completes a sign change and the output is damped, while the undriven
    # output swings the whole way back.
    wave = wave_system("dirichlet", n_elems=4 if fine else 3, speed=.5)
    xi_w = wave.meta["mesh"].nodes[wave.meta["free"]]

    data = controllable_pair()
    delay = delay_system(data["A0"], data["A1"], data["B0"], h=data["h"],
                         n_tau=4 if fine else 3)

    step = 1.0 if fine else 2.0  # coarsen every clock together for --quality quick
    return {
        "heat": (heat, 1.0 + .2 * np.cos(np.pi * xi_h), 1.0,
                 .0025 * step, 8.0, 4),
        "wave": (wave, np.concatenate([np.sin(np.pi * xi_w), np.zeros(xi_w.size)]),
                 4.0, .01 * step, 24.0, 4),
        "delay": (delay, np.ones(delay.n) / np.sqrt(delay.n), 1.0,
                  .0025 * step, 8.0, 4),
    }


def run(quality: str = "quick") -> dict:
    configure()
    results = {}

    for label, (sys, x0, horizon, dt, length, dwell) in _cases(quality).items():
        t = uniform_grid(horizon, dt)
        weights = LqrWeights.make(sys, terminal=1.0, control=.5)
        ref = riccati_hamiltonian(sys, weights, t)
        optimal = ref.closed_loop(x0)
        optimum = ref.optimal_cost(x0)

        n_T = t.size
        sizes = [n_T // 2, n_T, 2 * n_T]
        length = _dwell_safe(length, horizon, dt, dwell, sizes)
        record = simulate(sys, Prbs(dwell * dt, seed=3, horizon=length + dt),
                          uniform_grid(length, dt), np.zeros(sys.n), theta=.5)
        # The undriven response, for contrast in the showcase panels: it is a
        # model-based reference like the Riccati one and enters no computation.
        free = simulate(sys, lambda tt: np.zeros((sys.m, np.size(tt))), t, x0,
                        theta=.5)
        solutions = [solve_data_lqr(shift_library(record, horizon, n), sys, weights,
                                    x0, rho=RHO) for n in sizes]
        results[label] = {
            "t": t, "n_T": n_T, "sizes": sizes, "optimal": optimal, "free": free,
            "optimum": optimum, "solutions": solutions, "dt": dt,
            "rel_cost": np.array([abs(s.cost - optimum) / abs(optimum)
                                  for s in solutions]),
        }

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    show = results[SHOWCASE]
    converged = show["solutions"][-1]  # N = 2 n_T, the size the table converges at
    for a, signal, name in ((ax[0], "u", r"$u(t)$"), (ax[1], "y", r"$y(t)$")):
        a.plot(show["t"], getattr(show["optimal"], signal)[0], "k--", label="Riccati")
        a.plot(show["t"], getattr(converged.record, signal)[0], lw=.9,
               label="data-driven")
        a.set(xlabel=r"$t$", ylabel=name)
        a.grid(True, alpha=.25)
    ax[1].plot(show["t"], show["free"].y[0], color=".6", lw=.8, zorder=0,
               label="undriven")
    ax[0].set_title(f"{SHOWCASE}: input", fontsize=8)
    ax[1].set_title(f"{SHOWCASE}: output", fontsize=8)
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
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
