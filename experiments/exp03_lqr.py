"""Compare the two data-driven finite-horizon LQR discretisations of the paper.

``graph`` is the state-record method: weak moments of one record identify a
basis of the finite-dimensional system graph (characterization
``item:wfl-synthesis`` of ``thm:willems-gramian``, whose proof is given in
Appendix ``app:willems-gramian``).  A sparse QP constrains every theta-method
stage to that graph.  ``window`` is the input--output method: shifted windows
of the same record span the finite-horizon behaviour
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
# Once a library spans its target space, enlarging it does not define a
# convergence process.  Use twice the nominal dimension throughout and expose
# the actual numerical rate by refining the time grid instead.
TRIAL_RATIO = 2.0


def _dwell_safe(length: float, window: float, dt: float, dwell: int,
                sizes: list[int]) -> float:
    """Lengthen the record until no shifted library is dwell-aliased.

    If every window start falls in the same residue class modulo the dwell of
    the probing signal, all windows carry the input on one dwell grid and the
    span collapses by the dwell factor with no change in the nominal size.
    """
    n_w = int(round(window / dt)) + 1

    def aliased(n_rec: int, size: int) -> bool:
        if dwell <= 1:
            return False
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


def _solve_case(case: tuple, *, dt: float | None = None,
                dwell_time: float | None = None) -> dict:
    """Solve both regulators at one time resolution and a spanning trial size."""
    sys, start, horizon, past, base_dt, length, dwell = case
    dt = base_dt if dt is None else dt
    dwell_time = dwell * base_dt if dwell_time is None else dwell_time
    dwell_samples = int(round(dwell_time / dt))
    if not np.isclose(dwell_samples * dt, dwell_time):
        raise ValueError("the PRBS dwell must be an integer number of time steps")

    t = uniform_grid(horizon, dt)
    weights = LqrWeights.make(sys, terminal=0.0, control=CONTROL)

    # Keep the physical conditioning input fixed across time refinements.  Its
    # terminal state is the initial state of the control problem; the graph
    # method receives that state, while the window method receives only (u,y).
    conditioning = simulate(
        sys, Prbs(dwell_time, seed=11, horizon=past + dt),
        uniform_grid(past, dt), start, theta=.5,
    )
    x0 = conditioning.x[:, -1]
    reference = riccati_hamiltonian(sys, weights, t)
    optimal = reference.closed_loop(x0)
    optimum = reference.optimal_cost(x0)

    graph_dim = sys.m + sys.n
    graph_size = int(round(TRIAL_RATIO * graph_dim))
    n_w = int(round(past / dt)) + int(round(horizon / dt)) + 1
    behaviour_dim = sys.m * n_w + sys.n
    window_size = int(round(TRIAL_RATIO * behaviour_dim))
    length = _dwell_safe(length, past + horizon, dt, dwell_samples, [window_size])
    record = simulate(
        sys, Prbs(dwell_time, seed=3, horizon=length + dt),
        uniform_grid(length, dt), np.zeros(sys.n), theta=.5,
    )
    record_t = record.window(0.0, horizon)
    graph = estimate_graph(
        record_t,
        hat_tests(record_t.t, graph_size),
        sys.MW,
        derivative_metric=sys.MX,
        rank_tol=RANK_TOL,
        theta=.5,
    )
    graph_solution = solve_graph_lqr(graph, t, weights, x0, theta=.5)
    library = io_shift_library(record, past=past, horizon=horizon,
                               n_windows=window_size)
    behaviour = behaviour_basis(library, rank_tol=RANK_TOL)
    window_solution = solve_io_lqr(
        behaviour, weights, conditioning.u, conditioning.y, rho=RHO,
    )
    free = simulate(sys, lambda tt: np.zeros((sys.m, np.size(tt))), t, x0,
                    theta=.5)

    def relative(value: float) -> float:
        return abs(value - optimum) / abs(optimum)

    return {
        "t": t,
        "optimal": optimal,
        "free": free,
        "optimum": optimum,
        # How much the regulator is worth on this horizon: with a delayed or
        # smoothing observation an optimal input can be close to zero, and then
        # a cost error says nothing about the method.
        "undriven_ratio": trajectory_cost(free, sys, weights) / optimum,
        "graph_size": graph_size,
        "graph_dim": graph_dim,
        "graph_solution": graph_solution,
        "graph_error": relative(graph_solution.cost),
        "graph_replayed": relative(
            _replayed_cost(sys, weights, t, x0, graph_solution)
        ),
        "window_size": window_size,
        "behaviour_dim": behaviour_dim,
        "window_solution": window_solution,
        "window_error": relative(window_solution.cost),
        "window_replayed": relative(
            _replayed_cost(sys, weights, t, x0, window_solution)
        ),
    }


def run(quality: str = "quick") -> dict:
    configure()
    cases = _cases(quality)
    results = {label: _solve_case(case) for label, case in cases.items()}

    # Refine only the heat case: its error is set by the time discretisation.
    # The wave and delay errors instead sit on unresolved-behaviour floors, so
    # a time-grid slope for them would have no convergence interpretation.
    heat_case = cases["heat"]
    base_dt = heat_case[4]
    heat_dts = base_dt * np.array((2.0, 1.0) if quality == "quick"
                                  else (4.0, 2.0, 1.0))
    heat_dwell_time = heat_case[6] * base_dt
    heat_runs = []
    for dt in heat_dts:
        if np.isclose(dt, base_dt):
            heat_runs.append(results["heat"])
        else:
            heat_runs.append(_solve_case(heat_case, dt=float(dt),
                                         dwell_time=heat_dwell_time))
    time_steps = heat_case[2] / heat_dts
    graph_errors = np.array([item["graph_error"] for item in heat_runs])
    window_errors = np.array([item["window_error"] for item in heat_runs])
    graph_order = float(
        -np.polyfit(np.log(time_steps), np.log(graph_errors), 1)[0]
    )
    window_order = float(
        -np.polyfit(np.log(time_steps), np.log(window_errors), 1)[0]
    )
    refinement = {
        "dt": heat_dts,
        "time_steps": time_steps,
        "graph_error": graph_errors,
        "window_error": window_errors,
        "graph_order": graph_order,
        "window_order": window_order,
    }

    dark, light = family_colors(SHOWCASE, 2)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    show = results[SHOWCASE]
    graph_final = show["graph_solution"]
    window_final = show["window_solution"]
    for axis, signal, name in ((ax[0], "u", r"$u(t)$"),
                               (ax[1], "y", r"$y(t)$")):
        axis.plot(show["t"], getattr(show["optimal"], signal)[0], "k--",
                  label="Riccati")
        axis.plot(show["t"], getattr(graph_final.record, signal)[0], lw=1.0,
                  color=dark, label="i-s-o")
        axis.plot(window_final.t, getattr(window_final, signal)[0], lw=.9,
                  color=light, label="i-o", alpha=.9)
        axis.set(xlabel=r"$t$", ylabel=name, xlim=(0.0, show["t"][-1]))
        axis.grid(True, alpha=.25)
    ax[1].plot(show["t"], show["free"].y[0], color=".65", lw=.8, zorder=0,
               label="undriven")
    ax[0].set_title(f"{SHOWCASE}: input", fontsize=8)
    ax[1].set_title(f"{SHOWCASE}: output", fontsize=8)
    ax[0].legend(fontsize=6.5)
    ax[1].legend(fontsize=6.5)

    heat_dark, heat_light = family_colors("heat", 2)
    ax[2].loglog(
        time_steps, graph_errors, "o-", ms=3.5, color=heat_dark,
        label=rf"i-s-o (order {graph_order:.2f})",
    )
    ax[2].loglog(
        time_steps, window_errors, "s--", ms=3.2, color=heat_light,
        label=rf"i-o (order {window_order:.2f})",
    )
    ax[2].set(xlabel=r"time steps per horizon ($T/\Delta t$)",
              ylabel="relative cost error")
    ax[2].set_xticks(time_steps, [rf"${int(round(n))}$" for n in time_steps])
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].set_title("heat: time refinement", fontsize=8)
    ax[2].grid(True, which="both", alpha=.25)
    ax[2].legend(fontsize=6.2, loc="center left", borderpad=.4,
                 labelspacing=.3, handlelength=1.6, framealpha=.9)
    fig.tight_layout()
    fig_path = savefig(fig, "lqr.pdf")

    rows = []
    for label, result in results.items():
        graph = result["graph_solution"]
        window = result["window_solution"]
        rows.append(
            f"{label} & i-s-o & {result['graph_size']} & "
            f"{graph.graph.rank}/{graph.graph.domain_dimension} & "
            f"{tex_num(graph.cost)} & {tex_num(result['graph_error'])} & "
            f"{tex_num(result['graph_replayed'])} \\\\"
        )
        rows.append(
            f"{label} & i-o & {result['window_size']} & "
            f"{window.behaviour.rank}/{result['behaviour_dim']} & "
            f"{tex_num(window.cost)} & {tex_num(result['window_error'])} & "
            f"{tex_num(result['window_replayed'])} \\\\"
        )
        rows.append(r"\addlinespace")
    table = write_table("lqr.tex", r"""\begin{tabular}{llrrrrr}
\toprule
Example & data & size & rank & $J_{\rm dd}$ & $e_J$ & $e_J^{\rm plant}$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "results": results,
            "refinement": refinement}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, data in result["results"].items():
        print(label, "J* =", data["optimum"], "J0/J* =", data["undriven_ratio"],
              "\n  graph:   ", data["graph_error"], data["graph_replayed"],
              "\n  window:  ", data["window_error"], data["window_replayed"],
              "\n  defects: ", data["graph_solution"].initial_defect,
              data["window_solution"].past_defect,
              "\n  rank:    ", data["window_solution"].behaviour.rank,
              "/", data["behaviour_dim"])
    print("heat time orders:", result["refinement"]["graph_order"],
          result["refinement"]["window_order"])
