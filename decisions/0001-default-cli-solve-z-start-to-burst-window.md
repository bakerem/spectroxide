# Default CLI `solve` z_start to the burst window

## Status

Accepted, 2026-09-22.

## Context

The three front ends start the PDE at different redshifts when the user does not pass `--z-start`
(`dev/REVIEW_2026-09-22.md`, finding A-2):

- CLI `solve` starts at z = 5e6 for every non-resonance scenario (`src/cli.rs:1496-1499`).
- CLI sweeps start each point at z_h + 7σ_z, with σ_z = max(0.04 z_h, 100) (`src/cli.rs:1696-1697`,
  `:1812-1813`, `:1932-1933`).
- Python `solve()` starts single-burst and monochromatic-photon runs at z_h + 7σ, with the same σ,
  and continuous heat scenarios at 3e6 (`python/spectroxide/solver.py:679-696`).

For a burst, the steps between 5e6 and z_h + 7σ evolve only the adiabatic-cooling baseline. At
z_h = 1e5 they are 97.6% of the steps: the CLI takes 33.45 s against 0.52 s from z_h + 7σ, and μ
changes by 2.4e-4 relative. At z_h = 2e5 the CLI and Python results at their defaults differ in y
by 1.1% (6.497e-7 against 6.424e-7). So one physical question gets two answers, depending on the
front end, and the CLI answer costs up to 64 times more. The 30 s-per-solve benchmark in
`dev/audit/PERF_AUDIT_2026-08-18.md` mostly measures this coasting.

The burst and photon-line scenarios put all their energy within a few σ_z of z_h. A Gaussian
carries a fraction of about 1e-12 of its weight beyond 7σ.

## Decision

When the user gives no `--z-start`, CLI `solve` starts the single-burst and monochromatic-photon
scenarios at z_h + 7σ_z. σ_z is the value the run uses: `--sigma-z` if given, otherwise
max(0.04 z_h, 100). This matches the sweeps and Python.

Every other default stays as it is:

- Resonance scenarios keep z_start = z_res.
- Continuous scenarios in CLI `solve` keep 5e6. Python keeps 3e6 for them.
- Rust `SolverConfig::default` keeps z_start = 3e6 and z_end = 1.0.
- Grid defaults do not change.

`--help` and `docs/cli.rst` state the per-scenario default.

## Consequences

- CLI `solve`, CLI sweeps, and Python `solve()` give the same answer for a burst or photon line at
  their defaults, and CLI burst solves at low z_h run up to 64 times faster.
- A CLI burst result no longer includes the adiabatic-cooling baseline built up between 5e6 and
  z_h + 7σ. At z_h = 1e5 this changes μ by 2.4e-4 relative. Scripts that relied on the old default
  must pass `--z-start 5e6`.
- The continuous-scenario default still differs between CLI (5e6) and Python (3e6). This record
  leaves that difference in place on purpose.
- The perf audit's 30 s benchmark no longer describes a default CLI burst solve. Record that in
  `dev/audit/PERF_AUDIT_2026-08-18.md`.

## Addendum

2026-09-23.

The 1.1% difference in y at z_h = 2e5 comes from the grid, not from z_start. CLI `solve` defaults
to 2000 grid points and Python `solve()` to 4000. Measured on the implementing branch, for
`single-burst --z-h 2e5 --delta-rho 1e-5`:

| Run | Grid points | z_start | μ | y | Time |
|---|---|---|---|---|---|
| CLI, old default | 2000 | 5e6 | 1.388861e-5 | 6.496948e-7 | 14.9 s |
| CLI, new default | 2000 | 2.56e5 | 1.389005e-5 | 6.497011e-7 | 1.0 s |
| CLI, `--production-grid --n-points 4000` | 4000 | 2.56e5 | 1.389447e-5 | 6.423960e-7 | |
| Python `solve()` default | 4000 | 2.56e5 | 1.389447e-5 | 6.423960e-7 | 2.1 s |

The start redshift moves μ by 1.0e-4 relative and y by 1e-5. At the same grid, CLI and Python
now agree to every printed digit. At their own defaults they still differ by 1.1% in y and
3.2e-4 in μ, because this record keeps the grid defaults. So the first consequence above holds for
z_start and for run time, not for the full result.
