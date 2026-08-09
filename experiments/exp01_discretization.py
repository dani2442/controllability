"""Spatial convergence and discrete boundary-control growth, for all three examples.

The source-aligned heat and wave configurations represent the parabolic and
hyperbolic families of Pritchard--Salamon Section 4. The retarded example is
the ``M = 0`` specialization of its neutral family and is discretised in the
delay variable rather than in space.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ddinf.delay import delay_system, uncontrollable_pair
from ddinf.heat import fem_eigenvalues, heat_system
from ddinf.plotting import configure, savefig, write_table
from ddinf.wave import fem_frequencies, wave_system
from experiments.common import parser, tex_num


def _heat_error(kind: str, exact: float, mode: int = 0):
    def err(n: int) -> tuple[float, float]:
        sys = heat_system(kind, n_elems=n)
        return (abs(fem_eigenvalues(sys, mode + 1)[mode].real - exact),
                sys.control_operator_norm())
    return err


def _wave_error(exact: float):
    def err(n: int) -> tuple[float, float]:
        sys = wave_system("dirichlet", n_elems=n)
        return abs(fem_frequencies(sys, 1)[0] - exact), sys.control_operator_norm()
    return err


def _delay_error(data: dict, exact: complex):
    def err(n: int) -> tuple[float, float]:
        sys = delay_system(data["A0"], data["A1"], data["B0"], h=data["h"], n_tau=n)
        lam = np.linalg.eigvals(sys.A)
        return float(np.min(np.abs(lam - exact))), sys.control_operator_norm()
    return err


def run(quality: str = "quick") -> dict:
    configure()
    sizes = np.array([8, 16, 32, 64] if quality == "quick" else [8, 16, 32, 64, 128])
    delay_data = uncontrollable_pair()
    # Closed form: the leading root of lambda = d + q e^{-lambda h}.
    lambert = delay_data["obstruction"]
    root = complex(lambert[np.argmax(lambert.real)])

    cases = {
        # label: (error/norm map, symbol of the converged quantity)
        "heat, Dirichlet control": (_heat_error("dirichlet", -np.pi**2), r"$\lambda_1$"),
        "heat, Neumann--Neumann": (_heat_error("neumann", -np.pi**2, mode=1),
                                      r"$\lambda_1$ (nonzero)"),
        "wave, Dirichlet control": (_wave_error(np.pi), r"$\omega_1$"),
        "delay, leading Lambert root": (_delay_error(delay_data, root), r"$\lambda_1$"),
    }

    errors, norms = {}, {}
    for label, (err, _) in cases.items():
        pairs = [err(int(n)) for n in sizes]
        errors[label] = np.array([p[0] for p in pairs])
        norms[label] = np.array([p[1] for p in pairs])

    h = 1.0 / sizes
    order = {k: float(np.polyfit(np.log(h), np.log(errors[k]), 1)[0]) for k in cases}
    growth = {k: float(-np.polyfit(np.log(h), np.log(norms[k]), 1)[0]) for k in cases}

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for label, marker in zip(cases, ("o", "s", "^", "d")):
        ax[0].loglog(h, errors[label], marker=marker, ms=4, label=label)
        ax[1].loglog(h, norms[label], marker=marker, ms=4, label=label)
    ref = errors["heat, Dirichlet control"][-1]
    ax[0].loglog(h, ref * (h / h[-1]) ** 2, "k--", lw=1, label=r"$h^2$")
    ax[0].set(xlabel="mesh width", ylabel="leading spectral error")
    ax[1].set(xlabel="mesh width", ylabel=r"$\|B_h\|_{\mathcal{L}(U,X_h)}$")
    for a in ax:
        a.grid(True, which="both", alpha=.25)
        a.legend(fontsize=6.5)
        # Label only the meshes actually used; the default decade ticks collide.
        a.set_xticks(h, [rf"$\frac{{1}}{{{n}}}$" for n in sizes], minor=False)
        a.set_xticks([], minor=True)
    fig.tight_layout()
    fig_path = savefig(fig, "discretization.pdf")

    rows = "\n".join(
        f"{label} & {cases[label][1]} & {order[label]:.2f} & "
        # the retarded equation has a bounded control operator: no growth at all
        f"{'0 (bounded)' if abs(growth[label]) < .05 else f'{growth[label]:.2f}'} & "
        f"{tex_num(errors[label][-1])} \\\\"
        for label in cases
    )
    table = write_table("discretization.tex", rf"""\begin{{tabular}}{{llrrr}}
\toprule
Example & quantity & observed order & $\|B_h\|$ exponent & finest error \\
\midrule
{rows}
\bottomrule
\end{{tabular}}""")
    return {"figure": fig_path, "table": table, "order": order, "growth": growth,
            "errors": errors, "norms": norms, "sizes": sizes}


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    result = run(args.quality)
    print("orders:", result["order"])
    print("B growth:", result["growth"])
