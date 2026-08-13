# Numerical code for data-driven control in infinite-dimensional spaces

Numerical experiments supporting `paper_wfl2`, *Data-Driven Control in
Infinite-Dimensional Spaces: Fundamental Lemma and Applications*
(D. López-Montero, E. Zuazua). Running the suite regenerates every figure and
every table of Section~*Numerical experiments* directly into the paper source
tree.

The three examples are concrete instances of the three families of
Pritchard--Salamon, *The linear quadratic control problem for infinite
dimensional systems with unbounded input and output operators*, §4:

| example | module | control | observation | reference |
| --- | --- | --- | --- | --- |
| heat equation | `ddinf.heat` | Neumann–Neumann boundary | interior point value | §4.2, Ex. 4.6 |
| wave equation | `ddinf.wave` | Dirichlet boundary | smooth distributed | §4.3, Ex. 4.10 |
| retarded equation | `ddinf.delay` | bounded, in-domain | delayed state component | §4.1, `M = 0` |

They differ in exactly the ways the theory predicts should matter: the heat
semigroup is analytic and smoothing, the wave semigroup is a nonanalytic group,
and the retarded system has a bounded control operator but an unbounded
observation. The heat discretization also carries Dirichlet-control variants,
used as an independent convergence check (`dirichlet`) and as a closed-form
uncontrollable comparison (`dirichlet_sym`).

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). Everything is
dense NumPy/SciPy; there is no GPU or MPI path.

```bash
uv sync                                     # pinned by uv.lock
uv run pytest                               # 20 tests, ~2 s
uv run python -m experiments.run_all        # quick defaults, ~8 s
```

Individual experiments take the same `--quality` flag:

```bash
uv run python -m experiments.exp03_lqr --quality paper
```

The artifacts committed under `paper_wfl2/` are produced by the
publication run, which takes about ten seconds:

```bash
uv run python -m experiments.run_all --quality paper
```

`--quality quick` uses coarser meshes and time steps so the whole suite can be
rerun while editing the paper; `--quality paper` uses the finer grids that the
committed figures and tables were generated from. Only `paper` reproduces the
committed numbers.

The experiments must be run from a source checkout: `ddinf.plotting` locates
the paper tree as `<repo>/paper_wfl2/` relative to its own file, so an
installed copy of the package would write elsewhere.

## What each experiment produces

| experiment | figure | table | paper |
| --- | --- | --- | --- |
| `exp01_discretization` | `figures/discretization.pdf` | `tables/discretization.tex` | Fig. 1, Tab. 1 |
| `exp02_controllability` | `figures/controllability.pdf` | `tables/controllability.tex` | Fig. 2, Tab. 2 |
| `exp03_lqr` | `figures/lqr.pdf` | `tables/lqr.tex` | Fig. 3, Tab. 3 |
| `exp04_conditioning` | `figures/conditioning.pdf` | `tables/conditioning.tex` | Fig. 4, Tab. 4 |

- **exp01 — discretization.** Spatial convergence of the leading spectral
  quantity of each example (first Dirichlet eigenvalue `-π²`; first nonzero
  Neumann–Neumann eigenvalue; first wave frequency `ω₁ = π`; leading Lambert
  root of `λ = d + q e^{-λ}`), and the mesh growth of `‖B_h‖_{L(U,X_h)}`. All
  four converge at order 2; the control-operator norm grows like `h^{-3/2}`
  (heat, Dirichlet), `h^{-1/2}` (heat Neumann and wave), and is bounded for the
  retarded equation.
- **exp02 — controllability.** The data-driven Fattorini–Hautus test on one
  record per configuration. Each example is run twice: once approximately
  controllable (every pencil candidate must be rejected) and once with an
  obstruction known in closed form (it must be recovered). Also reports the
  numerical rank of the weak moment map.
- **exp03 — LQR.** The finite-horizon regulator reconstructed from
  time-shifted windows of a single record, for library sizes `N/n_T ∈
  {1/2, 1, 2}` where `n_T = T/Δt + 1`. Scored against a separately computed
  Riccati solution.
- **exp04 — conditioning.** Gramian spectra for three probing inputs — the
  paper's harmonic PE signal, a well-separated multisine, and a PRBS — on the
  heat equation (analytic, covered by the sufficiency theorem) and the wave
  equation (nonanalytic, not covered).

## Package layout

`src/ddinf/` is the library; nothing in it knows which PDE it is looking at
beyond the `ddinf.systems.LinearSystem` interface.

| module | contents |
| --- | --- |
| `systems` | `LinearSystem`: `(A, B, C)` plus the Gram matrices `MX`, `MW` of the state space `X` and the finer space `W` |
| `fem` | P1 mass/stiffness matrices, point evaluation, `C^∞` bump kernels on `(0,1)` |
| `heat`, `wave`, `delay` | the three semi-discrete examples and their closed-form spectral references |
| `timestepping` | theta-method integration and the sampled `Record` |
| `signals` | probing inputs: harmonic PE, multisine, PRBS |
| `moments` | the dynamic synthesis operator: `X0`, `X1`, `U0`, `Y0` from a record |
| `informativity` | Gramian and moment-map spectra, numerical rank at a threshold |
| `controllability` | the data-driven Fattorini–Hautus test, plus a model-based Hautus baseline |
| `lqr_data` | the data-driven regulator: shift/kernel libraries, `H`, `E`, `e`, and the penalized solve |
| `lqr_model` | the Riccati reference (see below) |
| `spectral` | closed-form modal series used to validate the discretizations |

Older finite-dimensional control modules live directly under `src/` and are not
used by these experiments.

## The model-free boundary

No model quantity enters a data-driven routine. `ddinf.controllability` and
`ddinf.lqr_data` receive only sampled input–state–output records and the two
Gram matrices representing the discrete Hilbert inner products of `X` and `W`.
Model-based quantities — Riccati solutions, Hautus modes, closed-form
eigenvalues, the dynamics residual `‖X1 - A X0 - B U0‖` — are computed
separately and used only to score the results. The `sys` argument passed to a
data-driven routine is used for `MX`/`MW` (and for reshaping `η`) and never for
`A`, `B` or `C`; `Moments.dynamics_residual` is the one function that takes
`A`, `B` deliberately, and is a diagnostic only.

## How the reference quantities are computed

- **Riccati (`ddinf.lqr_model.riccati_hamiltonian`).** Closed form, not time
  marching. With the Hamiltonian `H = [[A, -B R⁻¹ Bᵀ], [-CᵀC, -Aᵀ]]`, the
  solution of the differential Riccati equation is the Riccati transform of the
  matrix exponential, `P(T-s) = (Φ₂₁ + Φ₂₂ G)(Φ₁₁ + Φ₁₂ G)⁻¹` with
  `Φ = exp(-Hs)`; the grid is walked by repeated multiplication by the single
  precomputed step `exp(-H Δt)`. This is exact for the semi-discrete system up
  to the accuracy of `scipy.linalg.expm`. `riccati_ivp` re-derives the same
  object by stiff backward integration (Radau, `rtol = 1e-11`) as an
  independent check; the two agree to ~6·10⁻¹⁴ on the paper's heat case.
  The optimal cost is `J* = ⟨x₀, P(0) x₀⟩`, again exact — no trajectory is
  simulated to obtain it.
- **Spectral references.** Heat and wave eigenvalues are closed form; the
  retarded roots come from the Lambert `W` function
  (`ddinf.delay.lambert_roots`) for the block-triangular example, and from
  Newton-polished eigenvalues of a fine discretization
  (`characteristic_roots`) in general.

## Quadrature conventions

Two different rules are in use, deliberately:

- `ddinf.moments.quadrature_weights` — composite **Simpson**. The moments are
  the only place the record is integrated for the fundamental-lemma identities,
  and Simpson puts the quadrature error of `X1 = A X0 + B U0` below the error of
  the time stepping itself. `hat_tests` snaps its knots to even sample indices
  so no Simpson panel straddles the kink of a hat function.
- `ddinf.informativity.gramian_spectrum` — **trapezoid**, since only the
  spectrum's decay matters there.

- `ddinf.moments.trapezoid_weights` — **trapezoid**, and required for the
  control term of the LQR cost in `ddinf.lqr_data.assemble` and
  `ddinf.lqr_model.trajectory_cost`. Crank–Nicolson drives the state with
  `(u_k + u_{k+1})/2`, so the odd–even component of a sampled input never
  reaches the state; charging it with the *alternating* Simpson weights
  `4Δt/3, 2Δt/3` makes a free direction cheap on even samples and expensive on
  odd ones, and the discrete minimizer splits a given effective input between
  neighbours in the inverse ratio 2:1. That is a spurious sample-Nyquist ripple
  in an input the continuous problem wants smooth, and it costs two to four
  orders of magnitude of achieved cost. A uniform weight prices every sample
  alike; it is also the lumped form of the exact `∫|u|²` of the piecewise-linear
  interpolant the scheme implicitly integrates, so it is the consistent choice
  and not merely the safe one. The output term keeps Simpson.

`tests/test_lqr_data.py::test_reconstructed_input_carries_no_sample_nyquist_ripple`
guards this; swapping the weights back makes it and the Riccati-agreement test
fail.

## Determinism and reproducibility

Every random draw is seeded (`np.random.default_rng(seed)` in
`ddinf.signals.Prbs`, `multisine`, and `experiments.exp02_controllability`),
and no routine uses unseeded randomness or threading-dependent reductions.
Rerunning `--quality paper` reproduces the four `tables/*.tex` **byte for
byte** and the four `figures/*.pdf` byte for byte apart from the embedded
`CreationDate`/`ModDate`/`/ID` metadata. To verify:

```bash
uv run python -m experiments.run_all --quality paper
cd paper_wfl2 && git diff --stat -- tables      # must be empty
```

Environment used for the committed artifacts: Python 3.11.14, NumPy 2.4.1,
SciPy 1.17.0, Matplotlib 3.10.8, pinned in `uv.lock`.

## Building the LQR library

Two properties of the shift library are easy to lose without any visible
symptom, and `experiments/exp03_lqr.py` guards both:

- **Redundancy, not just rank.** The sampled behavior on the window has
  dimension `m·n_T + n`. Reaching that rank exactly makes the library a square
  basis and the penalized solve badly conditioned; the reported sizes go up to
  `2 n_T` because redundancy is what makes the reconstruction accurate rather
  than merely determined.
- **Shifts must not be commensurate with the probing signal.** If every window
  start falls in the same residue class modulo the PRBS dwell, all windows carry
  the input on the same dwell grid, the library inputs are piecewise constant on
  it, and the span collapses by the dwell factor while the nominal library size
  is unchanged. `_dwell_safe` picks the record length so this does not happen at
  any reported size. With the original settings the heat library at `N = n_T`
  had rank 48 instead of 128.
