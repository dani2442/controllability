"""Compare the two data-driven finite-horizon LQR discretisations of the paper.

``graph`` is the state-record method: weak moments of one record identify a
basis of the finite-dimensional system graph (``thm:willems-synthesis``,
``rmk:lqr-numerics``), and a sparse QP constrains every theta-method stage to
that graph.  ``window`` is the input--output method: shifted windows of the
same record span the finite-horizon behaviour
(``thm:windowed-io-fundamental-lemma``) and the regulator is a combination of
them, conditioned on the measured past of the plant instead of on its state.

The terminal weight is zero throughout, because ``<x(T), G_T x(T)>`` is not a
functional of the measured input and output and the two methods have to
minimise the same cost.  The Riccati solution is computed separately and used
only as a reference; each recovered input is also replayed on the plant, so the
cost a method predicts can be compared with the cost it actually pays.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from ddinf.delay import controllable_pair, delay_system
from ddinf.heat import heat_system
from ddinf.lqr_data import estimate_graph, solve_graph_lqr
from ddinf.lqr_io import behaviour_basis, io_shift_library, solve_io_lqr
from ddinf.lqr_model import LqrWeights, riccati_hamiltonian, trajectory_cost
from ddinf.moments import hat_tests
from ddinf.plotting import configure, family_colors, savefig, write_table
from ddinf.signals import Prbs
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import parser, tex_num

RANK_TOL = 1e-10
RHO = 1e-8
CONTROL = .05
SHOWCASE = "wave"
# Both methods are run at the same sizes relative to their own dimension: the
# graph needs q >= m + n test functions, the behaviour on the window needs
# N >= m n_w + n windows.  Ratio 1 is the smallest size at which either can be
# exact; below it the corresponding space is not spanned at all.
RATIOS = (1., 2., 4.)


def _dwell_safe(length: float, window: float, dt: float, dwell: int,
                sizes: list[int]) -> float:
    """Lengthen the record until no shifted library is dwell-aliased.

    If every window start falls in the same residue class modulo the dwell of
    the probing signal, all windows carry the input on one dwell grid and the
    span collapses by the dwell factor with no change in the nominal size.
    """
    n_w = int(round(window / dt)) + 1

    def aliased(n_rec: int, size: int) -> bool:
        starts = np.unique(np.round(np.linspace(0, n_rec - n_w, size)).astype(int))
        return np.unique(starts % dwell).size == 1

    for extra in range(400):
        n_rec = int(round(length / dt)) + 1 + extra
        if not any(aliased(n_rec, size) for size in sizes):
            return (n_rec - 1) * dt
    raise RuntimeError("no dwell-safe record length found")


def _cases(quality: str) -> dict:
    """Systems, clocks and conditioning windows; 401 samples per horizon.

    Each plant is released from a disturbed state and probed over the
    conditioning window; the state it reaches there is the initial state of the
    control problem.  The window has to be long enough for the output to
    determine that state.  The retarded system is the binding case: its output
    is the delayed first coordinate, so it is approximately observable only for
    ``T > 2h``, the same delay keeps the control from reaching the output
    before ``t = h``, and a history disturbance would sit in that unreachable
    first window.  Hence a horizon of ``2h``, a conditioning window of
    ``2.5 h``, and a plant that starts at rest and is disturbed by the probing
    input itself.
    """
    fine = quality == "paper"
    heat = heat_system("neumann", n_elems=6 if fine else 4, nu=1.0)
    xi_h = heat.meta["mesh"].nodes[heat.meta["free"]]
    wave = wave_system("dirichlet", n_elems=4 if fine else 3, speed=.5)
    xi_w = wave.meta["mesh"].nodes[wave.meta["free"]]
    data = controllable_pair()
    delay = delay_system(data["A0"], data["A1"], data["B0"], h=data["h"],
                         n_tau=4 if fine else 3)

    step = 1. if fine else 2.
    return {
        "heat": (heat, 1.0 + .2 * np.cos(np.pi * xi_h), 1.0, .25,
                 .0025 * step, 12.0, 4),
        "wave": (wave, np.concatenate([np.sin(np.pi * xi_w), np.zeros(xi_w.size)]),
                 4.0, 2.0, .01 * step, 40.0, 4),
        "delay": (delay, np.zeros(delay.n), 2.0, 2.5, .005 * step, 30.0, 4),
    }


def _replayed_cost(sys, weights, t, x0, solution) -> float:
    """Cost actually paid when the recovered input drives the plant from ``x0``.

    A model-based score: the data-driven routines never see this simulation.
    """
    record = solution.record if hasattr(solution, "record") else solution
    t_u, u = record.t, record.u
    replay = simulate(
        sys,
        lambda s: np.stack([np.interp(np.asarray(s, dtype=float), t_u, ui)
                            for ui in u]),
        t, x0, theta=.5,
    )
    return trajectory_cost(replay, sys, weights)


def run(quality: str = "quick") -> dict:
    configure()
    results = {}

    for label, (sys, start, horizon, past, dt, length, dwell) in _cases(quality).items():
        t = uniform_grid(horizon, dt)
        weights = LqrWeights.make(sys, terminal=0.0, control=CONTROL)

        # The plant runs before the regulator takes over.  Its state at the
        # junction is the initial state of the control problem; the graph
        # method is given that state, the window method only this record.
        conditioning = simulate(sys, Prbs(dwell * dt, seed=11, horizon=past + dt),
                                uniform_grid(past, dt), start, theta=.5)
        x0 = conditioning.x[:, -1]

        reference = riccati_hamiltonian(sys, weights, t)
        optimal = reference.closed_loop(x0)
        optimum = reference.optimal_cost(x0)

        graph_dim = sys.m + sys.n
        graph_sizes = [int(round(r * graph_dim)) for r in RATIOS]
        n_w = int(round(past / dt)) + int(round(horizon / dt)) + 1
        behaviour_dim = sys.m * n_w + sys.n
        window_sizes = [int(round(r * behaviour_dim)) for r in RATIOS]
        length = _dwell_safe(length, past + horizon, dt, dwell, window_sizes)
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
            # Raises if the moments do not resolve the whole graph, which at
            # these sizes would mean the record itself is uninformative.
            graph_solutions.append(solve_graph_lqr(graph, t, weights, x0, theta=.5))

        window_solutions = []
        for size in window_sizes:
            library = io_shift_library(record, past=past, horizon=horizon,
                                       n_windows=size)
            behaviour = behaviour_basis(library, rank_tol=RANK_TOL)
            window_solutions.append(solve_io_lqr(behaviour, weights,
                                                 conditioning.u, conditioning.y,
                                                 rho=RHO))

        free = simulate(sys, lambda tt: np.zeros((sys.m, np.size(tt))), t, x0,
                        theta=.5)
        def relative(value: float, optimum: float = optimum) -> float:
            return abs(value - optimum) / abs(optimum)

        results[label] = {
            "t": t,
            "optimal": optimal,
            "free": free,
            "optimum": optimum,
            # How much the regulator is worth on this horizon: with a delayed
            # or smoothing observation an optimal input can be close to zero,
            # and then a cost error says nothing about the method.
            "undriven_ratio": trajectory_cost(free, sys, weights) / optimum,
            "graph_sizes": graph_sizes,
            "graph_dim": graph_dim,
            "graph_solutions": graph_solutions,
            "graph_error": np.array([relative(s.cost) for s in graph_solutions]),
            "graph_replayed": np.array([
                relative(_replayed_cost(sys, weights, t, x0, s))
                for s in graph_solutions
            ]),
            "window_sizes": window_sizes,
            "behaviour_dim": behaviour_dim,
            "window_solutions": window_solutions,
            "window_error": np.array([relative(s.cost) for s in window_solutions]),
            "window_replayed": np.array([
                relative(_replayed_cost(sys, weights, t, x0, s))
                for s in window_solutions
            ]),
        }

    colors = {label: family_colors(label)[0] for label in results}
    dark, light = family_colors(SHOWCASE, 2)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    show = results[SHOWCASE]
    graph_final = show["graph_solutions"][-1]
    window_final = show["window_solutions"][-1]
    for axis, signal, name in ((ax[0], "u", r"$u(t)$"),
                               (ax[1], "y", r"$y(t)$")):
        axis.plot(show["t"], getattr(show["optimal"], signal)[0], "k--",
                  label="Riccati")
        axis.plot(show["t"], getattr(graph_final.record, signal)[0], lw=1.0,
                  color=dark, label="graph")
        axis.plot(window_final.t, getattr(window_final, signal)[0], lw=.9,
                  color=light, label="window", alpha=.9)
        axis.set(xlabel=r"$t$", ylabel=name)
        axis.grid(True, alpha=.25)
    ax[1].plot(show["t"], show["free"].y[0], color=".65", lw=.8, zorder=0,
               label="undriven")
    ax[0].set_title(f"{SHOWCASE}: input", fontsize=8)
    ax[1].set_title(f"{SHOWCASE}: output", fontsize=8)
    ax[0].legend(fontsize=6.5)
    ax[1].legend(fontsize=6.5)

    for label, result in results.items():
        ax[2].loglog(
            np.array(result["graph_sizes"]) / result["graph_dim"],
            result["graph_error"], "o-", ms=3.5, color=colors[label],
        )
        ax[2].loglog(
            np.array(result["window_sizes"]) / result["behaviour_dim"],
            result["window_error"], "s--", ms=3.2, color=colors[label],
        )
    ax[2].set(xlabel="trial size / dimension", ylabel="relative cost error")
    ax[2].set_xticks(list(RATIOS), [rf"${r:g}$" for r in RATIOS])
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].set_title("convergence", fontsize=8)
    ax[2].grid(True, which="both", alpha=.25)
    legend = [Line2D([0], [0], color=colors[label], label=label)
              for label in results]
    legend += [Line2D([0], [0], color=".3", marker="o", label="graph"),
               Line2D([0], [0], color=".3", ls="--", marker="s", label="window")]
    # The curves leave one free band, between the window errors and the graph
    # errors; the legend is anchored into it rather than left to overlap them.
    ax[2].legend(handles=legend, fontsize=6.2, ncol=2, loc="center left",
                 bbox_to_anchor=(0., .47), framealpha=.9)
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
            f"{tex_num(result['graph_replayed'][-1])} \\\\"
        )
        rows.append(
            f"{label} & window & {result['window_sizes'][-1]} & "
            f"{window.behaviour.rank}/{result['behaviour_dim']} & "
            f"{tex_num(window.cost)} & {tex_num(result['window_error'][-1])} & "
            f"{tex_num(result['window_replayed'][-1])} \\\\"
        )
        rows.append(r"\addlinespace")
    table = write_table("lqr.tex", r"""\begin{tabular}{llrrrrr}
\toprule
Example & method & size & rank & $J_{\rm dd}$ & $e_J$ & $e_J^{\rm plant}$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "results": results}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, data in result["results"].items():
        print(label, "J* =", data["optimum"], "J0/J* =", data["undriven_ratio"],
              "\n  graph:   ", data["graph_error"], data["graph_replayed"],
              "\n  window:  ", data["window_error"], data["window_replayed"],
              "\n  defects: ", data["graph_solutions"][-1].initial_defect,
              data["window_solutions"][-1].past_defect,
              "\n  ranks:   ", [s.behaviour.rank for s in data["window_solutions"]],
              "/", data["behaviour_dim"])
