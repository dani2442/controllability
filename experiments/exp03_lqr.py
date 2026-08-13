"""Compare two data-driven finite-horizon LQR discretisations.

``graph`` is the main-paper method: weak moments of one record identify a
basis of the finite-dimensional system graph, and a sparse QP constrains every
theta-method stage to that graph.  ``window`` is the shifted-trajectory
surrogate retained and analysed in ``paper_wfl2/window-informativity.tex``.
The Riccati solution is computed separately and used only as a reference.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from ddinf.delay import controllable_pair, delay_system
from ddinf.heat import heat_system
from ddinf.lqr_data import estimate_graph, solve_graph_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian
from ddinf.lqr_window import shift_library, solve_window_lqr
from ddinf.moments import hat_tests
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import parser, tex_num

RANK_TOL = 1e-10
RHO = 1e-8
SHOWCASE = "wave"


def _dwell_safe(length: float, horizon: float, dt: float, dwell: int,
                sizes: list[int]) -> float:
    """Lengthen the record until no shifted library is dwell-aliased."""
    n_t = int(round(horizon / dt)) + 1

    def aliased(n_rec: int, size: int) -> bool:
        starts = np.unique(np.round(np.linspace(0, n_rec - n_t, size)).astype(int))
        return np.unique(starts % dwell).size == 1

    for extra in range(400):
        n_rec = int(round(length / dt)) + 1 + extra
        if not any(aliased(n_rec, size) for size in sizes):
            return (n_rec - 1) * dt
    raise RuntimeError("no dwell-safe record length found")


def _cases(quality: str) -> dict:
    """Return systems and clocks with about 401 control-window samples."""
    fine = quality == "paper"
    heat = heat_system("neumann", n_elems=6 if fine else 4, nu=1.0)
    xi_h = heat.meta["mesh"].nodes[heat.meta["free"]]

    wave = wave_system("dirichlet", n_elems=4 if fine else 3, speed=.5)
    xi_w = wave.meta["mesh"].nodes[wave.meta["free"]]

    data = controllable_pair()
    delay = delay_system(data["A0"], data["A1"], data["B0"], h=data["h"],
                         n_tau=4 if fine else 3)

    step = 1.0 if fine else 2.0
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
        reference = riccati_hamiltonian(sys, weights, t)
        optimal = reference.closed_loop(x0)
        optimum = reference.optimal_cost(x0)

        graph_dim = sys.m + sys.n
        graph_sizes = [graph_dim, 2 * graph_dim, 4 * graph_dim]
        behavior_dim = sys.m * t.size + sys.n
        window_sizes = [behavior_dim // 2, behavior_dim, 2 * behavior_dim]
        length = _dwell_safe(length, horizon, dt, dwell, window_sizes)
        record = simulate(sys, Prbs(dwell * dt, seed=3, horizon=length + dt),
                          uniform_grid(length, dt), np.zeros(sys.n), theta=.5)
        record_t = record.window(0.0, horizon)

        graph_solutions = []
        for q in graph_sizes:
            graph = estimate_graph(
                record_t,
                hat_tests(record_t.t, q),
                sys.MW,
                derivative_metric=sys.MX,
                rank_tol=RANK_TOL,
                theta=.5,
            )
            graph_solutions.append(solve_graph_lqr(graph, t, weights, x0, theta=.5))

        window_solutions = [
            solve_window_lqr(shift_library(record, horizon, size), sys, weights,
                             x0, rho=RHO)
            for size in window_sizes
        ]
        free = simulate(sys, lambda tt: np.zeros((sys.m, np.size(tt))), t, x0,
                        theta=.5)
        graph_error = np.array([
            abs(solution.cost - optimum) / abs(optimum)
            for solution in graph_solutions
        ])
        window_error = np.array([
            abs(solution.cost - optimum) / abs(optimum)
            for solution in window_solutions
        ])
        results[label] = {
            "t": t,
            "optimal": optimal,
            "free": free,
            "optimum": optimum,
            "graph_sizes": graph_sizes,
            "graph_dim": graph_dim,
            "graph_solutions": graph_solutions,
            "graph_error": graph_error,
            "window_sizes": window_sizes,
            "behavior_dim": behavior_dim,
            "window_solutions": window_solutions,
            "window_error": window_error,
        }

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    show = results[SHOWCASE]
    graph_final = show["graph_solutions"][-1]
    window_final = show["window_solutions"][-1]
    for axis, signal, name in ((ax[0], "u", r"$u(t)$"),
                               (ax[1], "y", r"$y(t)$")):
        axis.plot(show["t"], getattr(show["optimal"], signal)[0], "k--",
                  label="Riccati")
        axis.plot(show["t"], getattr(graph_final.record, signal)[0], lw=1.0,
                  label="graph")
        axis.plot(show["t"], getattr(window_final.record, signal)[0], lw=.9,
                  label="window", alpha=.8)
        axis.set(xlabel=r"$t$", ylabel=name)
        axis.grid(True, alpha=.25)
    ax[1].plot(show["t"], show["free"].y[0], color=".65", lw=.8, zorder=0,
               label="undriven")
    ax[0].set_title(f"{SHOWCASE}: input", fontsize=8)
    ax[1].set_title(f"{SHOWCASE}: output", fontsize=8)
    ax[0].legend(fontsize=6.5)
    ax[1].legend(fontsize=6.5)

    colors = dict(zip(results, ("C0", "C1", "C2")))
    for label, result in results.items():
        ax[2].loglog(
            np.array(result["graph_sizes"]) / result["graph_dim"],
            result["graph_error"], "o-", ms=3.5, color=colors[label],
        )
        ax[2].loglog(
            np.array(result["window_sizes"]) / result["behavior_dim"],
            result["window_error"], "s--", ms=3.2, color=colors[label],
        )
    ax[2].set(xlabel="relative trial size", ylabel="relative cost error")
    ax[2].set_xticks([.5, 1.0, 2.0, 4.0],
                     [r"$1/2$", r"$1$", r"$2$", r"$4$"])
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].set_title("convergence", fontsize=8)
    ax[2].grid(True, which="both", alpha=.25)
    legend = [Line2D([0], [0], color=colors[label], label=label)
              for label in results]
    legend += [Line2D([0], [0], color=".3", marker="o", label="graph"),
               Line2D([0], [0], color=".3", ls="--", marker="s", label="window")]
    ax[2].legend(handles=legend, fontsize=6.2, ncol=2)
    fig.tight_layout()
    fig_path = savefig(fig, "lqr.pdf")

    rows = []
    for label, result in results.items():
        graph = result["graph_solutions"][-1]
        window = result["window_solutions"][-1]
        rows.append(
            f"{label} & graph & {result['graph_sizes'][-1]} & "
            f"{graph.graph.rank}/{graph.graph.domain_dimension} & "
            f"{tex_num(graph.cost)} & {tex_num(result['graph_error'][-1])} & "
            f"{tex_num(graph.initial_defect)} \\\\"
        )
        rows.append(
            f"{label} & window & {result['window_sizes'][-1]} & -- & "
            f"{tex_num(window.cost)} & {tex_num(result['window_error'][-1])} & "
            f"{tex_num(window.initial_defect)} \\\\"
        )
        rows.append(r"\addlinespace")
    table = write_table("lqr.tex", r"""\begin{tabular}{llrrrrr}
\toprule
Example & method & size & rank & $J_{\rm dd}$ & relative error &
initial defect \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "results": results}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, data in result["results"].items():
        print(label, "J* =", data["optimum"],
              "graph:", data["graph_error"],
              "window:", data["window_error"])
