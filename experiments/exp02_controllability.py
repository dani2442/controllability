"""The data-driven controllability test on all three examples of the paper.

Each family is run twice: once in a configuration that is approximately
controllable, where the test must reject every pencil candidate, and once in a
configuration whose obstruction is known in closed form, where the test must
recover it.  Nothing but the record and the two Gram matrices enters
:func:`ddinf.controllability.data_driven_controllability`; the closed-form
values and the model-based Hautus modes are computed only to score it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.controllability import data_driven_controllability, hautus_uncontrollable
from ddinf.delay import controllable_pair, delay_system, uncontrollable_pair
from ddinf.heat import heat_system
from ddinf.moments import moments, sine_tests
from ddinf.plotting import configure, savefig, write_table
from ddinf.signals import multisine
from ddinf.timestepping import simulate, uniform_grid
from ddinf.wave import wave_system
from experiments.common import nearest, parser, tex_complex, tex_num

NU = .02  # diffusivity; see the module docstring of ddinf.heat
SPEED = .5  # wave speed, chosen so the excited band covers the leading modes


def _rng_state(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n) / np.sqrt(n)


def _cases(quality: str) -> dict:
    """``label -> (system, initial state, horizon, reference obstruction, space)``."""
    ne = 12 if quality == "quick" else 20
    n_tau = 6 if quality == "quick" else 12

    heat_xi = heat_system("dirichlet", n_elems=ne, nu=NU)
    xi = heat_xi.meta["mesh"].nodes[heat_xi.meta["free"]]
    x0_heat = np.sin(np.pi * xi) + .2 * np.sin(2 * np.pi * xi)

    delay_u, delay_c = uncontrollable_pair(), controllable_pair()
    lambert = delay_u["obstruction"]
    delay_ref = complex(lambert[np.argmax(lambert.real)])

    def delay(data: dict):
        return delay_system(data["A0"], data["A1"], data["B0"], h=data["h"], n_tau=n_tau)

    wave_c = wave_system("dirichlet", n_elems=ne, speed=SPEED)
    wave_u = wave_system("dirichlet_sym", n_elems=ne, speed=SPEED)
    # The predicate of prop:data-fattorini-hautus needs kappa != 0, so the
    # initial state has to put energy into the mode that the input cannot
    # reach; a record in which the unreachable mode is simply absent carries no
    # evidence either way.
    x0_wave = np.concatenate([np.sin(np.pi * xi) + .2 * np.sin(2 * np.pi * xi),
                              np.zeros(xi.size)])

    return {
        "heat, one-sided": (heat_xi, x0_heat, 8.0, None, "X"),
        "heat, symmetric": (heat_system("dirichlet_sym", n_elems=ne, nu=NU),
                            x0_heat, 8.0, -NU * (2 * np.pi) ** 2, "X"),
        "wave, one-sided": (wave_c, x0_wave, 8.0, None, "W"),
        "wave, symmetric": (wave_u, x0_wave, 8.0, 2j * np.pi * SPEED, "W"),
        "delay, coupled": (delay(delay_c), _rng_state(delay(delay_c).n, 1), 10.0,
                           None, "W"),
        "delay, decoupled": (delay(delay_u), _rng_state(delay(delay_u).n, 1), 10.0,
                             delay_ref, "W"),
    }


def run(quality: str = "quick") -> dict:
    configure()
    dt = .002 if quality == "quick" else .001
    signal = multisine(np.geomspace(.2, 20.0, 20), amps=np.geomspace(1.0, .2, 20), seed=2)

    reports, rows, diagnostics = {}, [], {}
    for label, (sys, x0, horizon, reference, space) in _cases(quality).items():
        rec = simulate(sys, signal, uniform_grid(horizon, dt), x0, theta=.5)
        mom = moments(rec, sine_tests(rec.t, max(30, sys.n + 8)))
        report = data_driven_controllability(
            mom, rec, sys, rank_tol=1e-10, residual_tol=3e-3, kappa_tol=1e-10,
            space=space)
        reports[label] = report

        found = [c.lam for c in report.obstructions]
        model = [lam for lam, _ in hautus_uncontrollable(sys)]
        residual = mom.dynamics_residual(sys.A, sys.B)
        if reference is None:
            # Controllable configuration: success means no candidate survives.
            rows.append(f"{label} & {report.numerical_rank}/{report.dimension} & "
                        f"{len(found)} & --- & --- & --- & {tex_num(residual)} \\\\")
            diagnostics[label] = {"n_obstructions": len(found), "residual": residual}
            continue

        lam_found = nearest(found, reference)
        lam_model = nearest(model, reference)
        # What the data-driven test controls is the distance to the obstruction
        # of the *discretised* system; the distance from that to the closed form
        # is the discretisation error, and converges at the rate of exp01.
        e_data = abs(lam_found - lam_model)
        e_disc = abs(lam_model - reference)
        rows.append(f"{label} & {report.numerical_rank}/{report.dimension} & "
                    f"{len(found)} & {tex_complex(lam_found)} & "
                    f"{tex_complex(reference)} & {tex_num(e_data)} & "
                    f"{tex_num(residual)} \\\\")
        diagnostics[label] = {"n_obstructions": len(found), "lam_found": lam_found,
                              "lam_model": lam_model, "reference": reference,
                              "error_vs_model": e_data, "error_vs_closed_form": e_disc,
                              "residual": residual}

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for label, report in reports.items():
        s = report.singular_values
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", ms=3, label=label)
    ax.axhline(1e-10, color="k", ls="--", lw=1, label="rank threshold")
    ax.set(xlabel="moment direction", ylabel="normalized squared singular value")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=6.5)
    fig_path = savefig(fig, "controllability.pdf")

    table = write_table("controllability.tex", r"""\begin{tabular}{lrrllrr}
\toprule
Configuration & rank & obstr. & recovered $\lambda$ & closed form &
$|\lambda-\lambda_h|$ & residual \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
    return {"figure": fig_path, "table": table, "reports": reports,
            "diagnostics": diagnostics}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    for label, d in result["diagnostics"].items():
        print(label, d)
