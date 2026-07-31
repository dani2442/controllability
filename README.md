# Numerical code for data-driven control

This repository contains the numerical experiments supporting
`paper_wfl2`, *Data-Driven Control in Infinite-Dimensional Spaces*, alongside
the earlier finite-dimensional control modules in `src/`.

The experiments run on the three examples of the paper's introduction, which
are the concrete instances of the three families of Pritchard--Salamon, *The
linear quadratic control problem for infinite dimensional systems with
unbounded input and output operators*, Section 4:

| example | module | control | observation | reference |
| --- | --- | --- | --- | --- |
| heat equation | `ddinf.heat` | Dirichlet or Neumann boundary | mollified point | §4.2, Ex. 4.6 |
| wave equation | `ddinf.wave` | Dirichlet boundary | mollified point | §4.3, Ex. 4.10 |
| retarded equation | `ddinf.delay` | bounded, in-domain | delayed state | §4.1 |

The `ddinf` package implements:

- P1 finite-element heat, wave and delay-equation discretizations, together
  with the discrete Gram matrices of `X` and of the finer space `W`;
- theta-method trajectory generation and closed-form spectral references
  (modal series, Lambert `W` roots);
- weak moment synthesis and numerical informativity diagnostics;
- the data-driven controllability test;
- model-based and data-driven finite-horizon LQR references.

## Quick start

```bash
uv sync
uv run pytest
uv run python -m experiments.run_all
```

Individual experiments are available as
`experiments.exp01_discretization`, `exp02_controllability`,
`exp03_lqr`, and `exp04_conditioning`. They write reproducible PDF figures
and LaTeX tables directly to `paper_wfl2/figures/` and
`paper_wfl2/tables/`, both of which are tracked in git.

All experiment defaults are deliberately modest so the complete suite can be
rerun while editing the paper. Use `--quality paper` for the higher-resolution
publication run; the artifacts committed under `paper_wfl2/` are produced by

```bash
uv run python -m experiments.run_all --quality paper
```

No model quantity enters a data-driven routine. Riccati solutions, Hautus
modes and closed-form eigenvalues are computed separately and used only to
score the results.
