# Align the z_end and grid defaults across interfaces

## Status

Accepted, 2026-09-26 (EB).

Amends the "Every other default stays as it is" list of
[ADR 0001](0001-default-cli-solve-z-start-to-burst-window.md): the lines on
`SolverConfig::default` z_end and on the grid defaults no longer hold.

## Context

The three interfaces stop the PDE at different redshifts and use different frequency grids when
the user does not choose:

| Interface | z_end | Grid |
|---|---|---|
| Python `solve()`, `run_sweep()`, photon sweeps | 500 | production: N = 4000, x ∈ [1e-5, 60], x_t = 0.5, 35% log |
| Python Green's-function table builders | 0.0 (rejected by the binary) | N = 2000 on the coarse range |
| CLI `solve` and sweeps | 500 | coarse: N = 2000, x ∈ [1e-4, 50], x_t = 0.1, 30% log |
| Rust `SolverConfig::default`, `GridConfig::default` | 1.0 | coarse |

The ADR 0001 addendum measured the cost of the grid split: for `single-burst --z-h 2e5
--delta-rho 1e-5`, CLI and Python at their own defaults differ by 1.1% in y and 3.2e-4 in μ, and
agree to every printed digit on the same grid. The paper's figures and timing table use the
Python defaults, so a CLI or Rust user does not reproduce them without extra flags.

The Green's-function table builders (`build_greens_table`, `build_photon_greens_table`) default
to z_end = 0.0. The solver rejects z_end ≤ 0 ("z_end must be positive"), so a table build with
default arguments fails.

The end redshift matters little for the distortion. For a burst at z_h = 2e5 on the 2000-point
grid (measured 2026-09-25), ending at z = 10 instead of 500 leaves μ unchanged to 2e-9 relative,
lowers y by 0.45% (the gas, colder than the photons after decoupling, draws energy from them),
and adds 16% to the run time (0.75 s to 0.87 s). The solver does not model reionization, so a run
to z = 10 does not include the reionization y of order 1e-6 (Hill et al. 2015).

## Decision

Every interface ends at z = 10 and uses the production grid unless the user says otherwise:

- `SolverConfig::default().z_end` = 10. CLI `--z-end` defaults to 10. Python `solve()`,
  `run_sweep()`, and the photon sweeps default to `z_end=10.0`, and the table builders to
  `z_end=10.0, n_points=4000`.
- `GridConfig::default()` returns `GridConfig::production()`. The CLI builds every grid from it,
  and `--n-points` overrides only the point count.
- `--production-grid` stays a valid CLI flag with no effect, so that existing scripts and the
  Python wrapper, which still passes it, keep working. `SolverOpts::production_grid` is removed.

## Consequences

- CLI, Rust, and Python give the same answer at their defaults, and the paper's numbers are
  reproducible from any interface without extra flags.
- CLI and Rust runs at default settings take about twice as long (1.0 s to 2.1 s at z_h = 2e5,
  ADR 0001 addendum), and every run takes longer to reach z = 10. The full release test suite
  took 379 s with `-j 2` after the change.
- The old default survives as `GridConfig::coarse()`, a Rust-only preset with no CLI flag.
  `--n-points 2000` gives 2000 points on the production range, not the old grid.
- Five tests that build `GridConfig { n_points, ..GridConfig::default() }` were tuned on the coarse
  range and failed on the production range with the same point count: the energy-closure
  preconditions in `solver::tests::test_energy_closure_skip_rules` and `tests/coverage_gaps.rs`
  (100 and 300 points no longer fail closure), the joint Richardson order in
  `tests/convergence_order.rs` (0.65, band [1.0, 2.5]), and the sign of a 1e-10 y in
  `test_dcbr_thermalizes_mu_distortion`. They now pin `GridConfig::coarse()`. The CLI sweep test
  checked that Δρ/ρ differs across rows by more than 0.5%; that spread is energy-closure error,
  0.06% on the production grid, so the test now checks the μ spread (16%).
- Python's `debug=True` preset (n_points = 1000, `production_grid=False`) now gets 1000 points on
  the production range instead of the coarse range.
- Default Green's-function table builds work again. Tables built before this change used 2000
  points and, if they ran at all, a non-default z_end; cached tables are not rebuilt.
- The paper's timing table (`dev/scripts/benchmark_paper_table.py`) uses the Python defaults and
  must be re-measured: runs that ended at z = 500 now continue to z = 10.
