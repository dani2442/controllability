# Numerical code for data-driven control

This repository contains the numerical experiments supporting
`paper_wfl2`, *Data-Driven Control in Infinite-Dimensional Spaces*, alongside
the earlier finite-dimensional control modules in `src/`.

The `ddinf` package implements:

- P1 finite-element heat and delay-equation discretizations;
- theta-method trajectory generation and closed-form spectral references;
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
`paper_wfl2/tables/`.

All experiment defaults are deliberately modest so the complete suite can be
rerun while editing the paper. Use each command's `--quality paper` option for
the higher-resolution publication run.
