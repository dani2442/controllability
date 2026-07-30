"""Data-driven controllability test on two heat-equation configurations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.controllability import data_driven_controllability
from ddinf.delay import delay_system, uncontrollable_pair
from ddinf.heat import heat_system
from ddinf.moments import moments, sine_tests
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import multisine
from ddinf.timestepping import simulate, uniform_grid
from experiments.common import parser, tex_num


def run(quality: str = "quick") -> dict:
    configure()
    ne = 12 if quality == "quick" else 20
    dt = .002 if quality == "quick" else .001
    t = uniform_grid(8.0, dt)
    signal = multisine(np.geomspace(.2, 20.0, 20), amps=np.geomspace(1.0, .2, 20), seed=2)
    reports = {}
    residuals = {}
    for kind in ("dirichlet", "dirichlet_sym"):
        sys = heat_system(kind, n_elems=ne, nu=.02)
        xi = sys.meta["mesh"].nodes[sys.meta["free"]]
        x0 = np.sin(np.pi * xi) + .2 * np.sin(2 * np.pi * xi)
        rec = simulate(sys, signal, t, x0, theta=.5)
        mom = moments(rec, sine_tests(t, max(30, sys.n + 4)))
        residuals[kind] = mom.dynamics_residual(sys.A, sys.B)
        reports[kind] = data_driven_controllability(
            mom, rec, sys, rank_tol=1e-10, residual_tol=1e-3, kappa_tol=1e-10)

    delay_data = uncontrollable_pair()
    delay = delay_system(delay_data["A0"], delay_data["A1"], delay_data["B0"],
                         h=delay_data["h"], n_tau=6 if quality == "quick" else 10)
    delay_rec = simulate(delay, signal, uniform_grid(10.0, dt),
                         np.random.default_rng(1).normal(size=delay.n), theta=.5)
    delay_mom = moments(delay_rec, sine_tests(delay_rec.t, max(36, delay.n + 8)))
    residuals["delay"] = delay_mom.dynamics_residual(delay.A, delay.B)
    reports["delay"] = data_driven_controllability(
        delay_mom, delay_rec, delay, rank_tol=1e-10, residual_tol=3e-3,
        kappa_tol=1e-10, space="W")

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for kind, report in reports.items():
        s = report.singular_values
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", ms=3, label=kind)
    ax.axhline(1e-10, color="k", ls="--", lw=1, label="rank threshold")
    ax.set(xlabel="moment direction", ylabel="normalized squared singular value")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig_path = savefig(fig, "controllability.pdf")

    rows = []
    for kind, report in reports.items():
        obs = report.obstructions
        lam = obs[0].lam.real if obs else np.nan
        rows.append(
            f"{kind.replace('_', '-')} & {report.numerical_rank}/{report.dimension} & "
            f"{len(obs)} & {tex_num(lam) if obs else '--'} & "
            f"{tex_num(residuals[kind])} \\\\"
        )
    table = write_table("controllability.tex", r"""\begin{tabular}{lrrrr}
\toprule
System & numerical rank & obstructions & leading $\lambda$ & identity residual \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "reports": reports,
            "residuals": residuals}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    print({k: str(v) for k, v in result["reports"].items()})
