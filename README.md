# Numerical code for data-driven control in infinite-dimensional spaces

Numerical experiments supporting `paper`, *Data-Driven Control in
Infinite-Dimensional Spaces: Fundamental Lemma and Applications*
(D. López-Montero, E. Zuazua). Running the suite regenerates every figure and
every table of Section~*Numerical experiments* directly into the paper source
tree.

The scope of this repository is exactly what `paper/main.tex` consumes:
three experiments, three figures, two tables. Anything the paper does not use
has been removed.

The three examples are concrete instances of the three families of
Pritchard--Salamon, *The linear quadratic control problem for infinite
dimensional systems with unbounded input and output operators*, §4:

| example | module | control | observation | reference |
| --- | --- | --- | --- | --- |
| heat equation | `ddinf.systems.heat` | Neumann–Neumann boundary | interior point value | §4.2, Ex. 4.6 |
| wave equation | `ddinf.systems.wave` | Dirichlet boundary | smooth distributed | §4.3, Ex. 4.10 |
| retarded equation | `ddinf.systems.delay` | bounded, in-domain | delayed state component | §4.1, `M = 0` |

They differ in exactly the ways the theory predicts should matter: the heat
semigroup is analytic and smoothing, the wave semigroup is a nonanalytic group,
and the retarded system has a bounded control operator but an unbounded
observation. Each example also carries a symmetric-control variant
(`dirichlet_sym`, and the decoupled delay pair) whose obstruction is known in
closed form; those are the uncontrollable comparisons of `experiments/controllability.py`.

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). Everything is
dense NumPy/SciPy; there is no GPU or MPI path.

```bash
uv sync                                     # pinned by uv.lock
uv run pytest                               # 40 tests, ~25 s
uv run python -m experiments.run_all        # quick defaults, ~1 min
```

Individual experiments take the same `--quality` flag:

```bash
uv run python -m experiments.lqr --quality paper
```

The artifacts committed under `paper/` are produced by the publication
run, which takes about a minute and a half -- almost all of it the input–output
window libraries of `experiments/lqr.py`, whose largest behavior basis is an SVD of a few
thousand windows:

```bash
uv run python -m experiments.run_all --quality paper
```

`--quality quick` uses coarser meshes and time steps so the whole suite can be
rerun while editing the paper; `--quality paper` uses the finer grids that the
committed figures and tables were generated from. Only `paper` reproduces the
committed numbers.

The experiments must be run from a source checkout: `ddinf.paper` locates
the paper tree as `<repo>/paper/` relative to its own file, so an
installed copy of the package would write elsewhere.

## What each experiment produces

| experiment | figure | table | paper |
| --- | --- | --- | --- |
| `experiments/controllability.py` | `figures/controllability.pdf` | `tables/controllability.tex` | Fig. 1, Tab. 1 |
| `experiments/lqr.py` | `figures/lqr.pdf` | `tables/lqr.tex` | Fig. 2, Tab. 2 |
| `experiments/conditioning.py` | `figures/conditioning.pdf` | `tables/conditioning.tex` | Fig. 3 |

`run_all.py` runs the three in that order.

- **controllability.** Both data-driven Fattorini–Hautus tests, on one
  record per configuration and data class. Each example is run twice: once
  approximately controllable (every candidate must be rejected) and once with an
  obstruction known in closed form (it must be recovered). The `i-s` rows read
  `(u, x)` and look for a state functional; the `i-o` rows read `(u, y)` and look
  for a functional of one length-`T` window. The two use different probing
  inputs on purpose: the window test needs the shifted windows to span the
  behavior, and on the multisine of the state test a length-`T` kernel can
  isolate a single line, which the test then reports as an obstruction of a
  controllable system (`multisine_window_failure`). Also reports the numerical
  rank of the weak moment map and of the window library.
- **lqr.** Comparison of the two fundamental lemmas of the paper on the
  same regulator: the state-record synthesis method (a weak-moment basis of the
  system graph followed by a sparse constrained QP) and the input–output
  windowed method of the *Input–output behavior* subsection of
  `paper/sections/fundamental-lemma.tex` (shifted windows of the measured
  `(u, y)` span the finite-horizon behavior, and the initial condition enters as
  the plant's measured past instead of as a state). Both are scored against a
  separately computed Riccati solution, and each recovered input is replayed on
  the plant so that the cost a method predicts can be compared with the cost it
  actually pays.
- **conditioning.** Gramian spectra for three probing inputs — the
  paper's harmonic PE signal, a well-separated multisine, and a PRBS — on the
  heat equation (analytic, covered by the sufficiency theorem) and the wave
  equation (nonanalytic, not covered). Its table is the one artifact `main.tex`
  does not `\input`: the paper quotes those numbers (`9/26`, `11/26`, and the
  ratio of the weakest resolved directions) in the prose of §*Excitation and
  conditioning*, and `tables/conditioning.tex` is where they come from.

## Package layout

`src/ddinf/` is the library; nothing in it knows which PDE it is looking at
beyond the `ddinf.systems.LinearSystem` interface. Modules are grouped by what
they read, which is also how the paper distinguishes its results: `i-s`
(input–state), `i-s-o` (input–state–output), `i-o` (input–output).

```
src/ddinf/
├── systems/          the plants
│   ├── base.py       LinearSystem: (A, B, C) + the Gram matrices MX, MW
│   ├── fem.py        P1 mass/stiffness matrices, point evaluation, C^∞ bumps
│   ├── heat.py       Neumann/Dirichlet heat, and the symmetric variant
│   ├── wave.py       Dirichlet-controlled wave, and the symmetric variant
│   ├── delay.py      the retarded equation, Banks–Kappel P1 discretization
│   └── modal.py      closed-form modal data, used only to score
├── data/             the record and what is read off it
│   ├── signals.py    probing inputs: harmonic PE, multisine, PRBS
│   ├── records.py    theta-method integration and the sampled Record
│   ├── moments.py    the synthesis operator: X0, X1, U0, Y0 from a record
│   └── informativity.py   Gramian/moment spectra, numerical rank at a threshold
├── controllability/  the data-driven Fattorini–Hautus tests
│   ├── state.py      i-s: a state functional that is a pure exponential
│   └── window.py     i-o: the same predicate through window functionals
├── lqr/              the finite-horizon regulator
│   ├── riccati.py    the model-based reference (see below)
│   ├── graph.py      i-s-o: weak-moment graph basis + constrained QP
│   └── window.py     i-o: shifted window library, behavior basis, past match
└── paper.py          figure and table output into `<repo>/paper/`
```

Each subpackage re-exports its main entry points, so
`from ddinf.lqr import solve_io_lqr` and
`from ddinf.lqr.window import solve_io_lqr` are equivalent.

Every docstring cross-reference of the form ``thm:...``, ``eq:...`` names a
`\label` that exists in `paper/sections/`; they are kept in sync
deliberately, so a renamed label in the paper should be renamed here too.

## The model-free boundary

No model quantity enters a data-driven routine. `ddinf.controllability.state` and
`ddinf.lqr.graph` use sampled input–state–output records, prescribed LQR
weights, and matrices representing the relevant discrete Hilbert inner
products; `ddinf.lqr.window` uses even less — the sampled input and output only,
with no state and no state metric anywhere in its signature.
Model-based quantities — Riccati solutions, Hautus modes, closed-form
eigenvalues, the dynamics residual `‖X1 - A X0 - B U0‖` — are computed
separately and used only to score the results. The `sys` argument passed to a
controllability routine is used for `MX`/`MW` (and for reshaping
`η`) and never for `A`, `B` or `C`; `Moments.dynamics_residual` is the one function that takes
`A`, `B` deliberately, and is a diagnostic only. `experiments/lqr.py` replays each recovered
input on the plant for the same reason: to score, never to construct.

## How the reference quantities are computed

- **Riccati (`ddinf.lqr.riccati.riccati_hamiltonian`).** Closed form, not time
  marching. With the Hamiltonian `H = [[A, -B R⁻¹ Bᵀ], [-CᵀC, -Aᵀ]]`, the
  solution of the differential Riccati equation is the Riccati transform of the
  matrix exponential, `P(T-s) = (Φ₂₁ + Φ₂₂ G)(Φ₁₁ + Φ₁₂ G)⁻¹` with
  `Φ = exp(-Hs)`; the grid is walked by repeated multiplication by the single
  precomputed step `exp(-H Δt)`. This is exact for the semi-discrete system up
  to the accuracy of `scipy.linalg.expm`. `riccati_ivp` re-derives the same
  object by stiff backward integration (Radau, `rtol = 1e-11`) as an
  independent check; the two agree to ~10⁻¹³ in absolute terms on the
  paper's heat case.
  The optimal cost is `J* = ⟨x₀, P(0) x₀⟩`, again exact — no trajectory is
  simulated to obtain it.
- **Spectral references.** Heat and wave eigenvalues are closed form
  (`ddinf.systems.modal.heat_modal` carries the modal data, including the vanishing
  control coefficients that make `dirichlet_sym` uncontrollable); the retarded
  roots come from the Lambert `W` function (`ddinf.systems.delay.lambert_roots`) for the
  block-triangular example, and `ddinf.systems.delay.hautus_delay_defect` scores them
  against `σ_min([Δ(λ), B₀])`.

## Quadrature conventions

The numerical weak form is matched to its downstream discretization:

- `ddinf.data.moments.quadrature_weights` — composite **Simpson**. The moments are
  used for the continuous weak-moment calculations. `hat_tests` snaps its
  knots to even sample indices so no Simpson panel straddles a kink.
- `ddinf.data.moments.theta_moments` — the **theta-consistent discrete weak form**
  used to learn the LQR graph. For Crank–Nicolson it tests midpoint values and
  forms `X1 = sum phi_mid (x[k+1]-x[k])`; hence
  `X1 = A X0 + B U0` holds to the time-stepper's linear-solver tolerance
  without differentiating the measured state pointwise.
- `ddinf.lqr.graph.estimate_sampled_graph` — the **direct sampled graph**
  counterpart. It forms the theta-stage difference quotient
  `(x[k+1] - x[k]) / dt` and learns the same finite-dimensional graph without
  test functions. It is useful for exact simulated records; the weak-moment
  estimator is preferable when differentiating measured noise would be
  unstable.
- `ddinf.data.informativity.gramian_spectrum` — **trapezoid**, since only the
  spectrum's decay matters there.
- `ddinf.data.moments.trapezoid_weights` — **trapezoid**, used for the control term
  in `ddinf.lqr.window` and `ddinf.lqr.riccati.trajectory_cost`. Crank–Nicolson drives the state with
  `(u_k + u_{k+1})/2`, so the odd–even component of a sampled input never
  reaches the state; charging it with the *alternating* Simpson weights
  `4Δt/3, 2Δt/3` makes a free direction cheap on even samples and expensive on
  odd ones, and the discrete minimizer splits a given effective input between
  neighbours in the inverse ratio 2:1. That is a spurious sample-Nyquist ripple
  in an input the continuous problem wants smooth, and it costs two to four
  orders of magnitude of achieved cost. A uniform weight prices every sample
  alike; it is also the lumped form of the exact `∫|u|²` of the piecewise-linear
  interpolant the scheme implicitly integrates, so it is the consistent choice
  and not merely the safe one. The graph LQR instead uses one control and
  output per interval, so this nodal ambiguity does not arise.

`tests/test_lqr_window.py::test_reconstructed_input_carries_no_sample_nyquist_ripple`
guards this; swapping the weights back makes it and the Riccati-agreement test
fail.

## Determinism and reproducibility

Every random draw is seeded (`np.random.default_rng(seed)` in
`ddinf.data.signals.Prbs`, `multisine`, and `experiments.controllability`),
and no routine uses unseeded randomness or threading-dependent reductions.
Rerunning `--quality paper` reproduces the three `tables/*.tex` **byte for
byte** and the three `figures/*.pdf` byte for byte apart from the embedded
`CreationDate`/`ModDate`/`/ID` metadata. To verify:

```bash
uv run python -m experiments.run_all --quality paper
cd paper && git diff --stat -- tables      # must be empty
```

Environment used for the committed artifacts: Python 3.11.14, NumPy 2.4.1,
SciPy 1.17.0, Matplotlib 3.10.8, pinned in `uv.lock`.

## The two LQR discretizations

The main-paper routine first forms `Z = [U0; X0]` and
`D = [U0; X0; X1; Y0]` from theta-consistent weak moments. A weighted SVD of
`Z` selects the resolved directions and the corresponding columns of `D` span
the learned system graph. The LQR then imposes graph membership at every time
stage and fixes the initial state exactly. It requires no shifted record,
behavior library, or penalty parameter.

The input–output routine (`ddinf.lqr.window`) never sees a state. Shifted windows
of the measured `(u, y)` on `[-T_ini, T]` span the sampled behavior, a
metric-weighted SVD at the same `1e-10` threshold keeps the resolved part of
that span, and the regulator is the combination of those windows whose past
segment matches the plant's measured past. Four properties are easy to lose
without any visible symptom; the tests and `experiments/lqr.py` guard
them:

- **Spanning, not just rank.** The sampled behavior on the window has dimension
  `m·n_w + n`, and reaching that nominal size does not imply that an
  independent trajectory lies in the span.
  `tests/test_lqr_window.py::test_shifted_windows_span_the_sampled_behavior`
  checks the span itself. Its parabolic counterpart records the opposite fact:
  the heat library is one direction short at every size, because the fastest
  semi-discrete mode has decayed below rounding by the end of a window. That is
  the closure form of the theorem showing up numerically.
- **Shifts must not be commensurate with the probing signal.** If every window
  start falls in the same residue class modulo the PRBS dwell, all windows carry
  the input on the same dwell grid, the library inputs are piecewise constant on
  it, and the span collapses by the dwell factor while the nominal library size
  is unchanged. `_dwell_safe` picks the record length so this does not happen at
  any reported size.
- **The junction sample belongs to the past.** Under Crank–Nicolson the state
  at the junction is driven by the two-point average straddling it, so leaving
  that sample to the regulator lets it move the initial state and report a cost
  below the true optimum. Including it keeps the reconstruction a genuine
  trajectory — the predicted and replayed costs then agree to three digits — at
  the price of one inherited input sample, which is why this method is
  first-order in `Δt` where the graph method is second-order.
- **The conditioning window must be observable.** It has to be long enough for
  the output to determine the state at the junction; for the retarded example
  that means `T_ini > 2h`, the bound derived in the paper's observability
  subsection.
