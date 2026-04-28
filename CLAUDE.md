# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a Rust PDE solver (spectroxide) with Python bindings and Jupyter notebooks. The primary focus is always the PDE solver — not Green's functions — unless I explicitly say otherwise.

## Build & Test

```bash
cargo build --release          # Build optimized binary
cargo test                     # Run all tests (167 unit + 264 integration + 3 doc)
cargo test --release           # Run tests with optimizations (some solver tests are slow in debug)
cargo test test_name           # Run a single test by name
cargo run --release --bin spectroxide -- sweep  # Run PDE sweep over default z_h grid
cargo run --release --bin spectroxide          # Print help / list subcommands
cargo run --release --bin check_adiabatic      # Utility: adiabatic cooling check
```

**Python package** (wraps Rust binary + pure-Python Green's function):
```bash
cd python && pip install -e ".[plot]"    # Install with matplotlib
cd python && pip install -e ".[notebook]" # Install with jupyter too
```

**Key constraints**: Zero production Rust dependencies (pure std library). Only dev-dependency is `approx` for float comparison in tests.

## Scope

This fork handles **heat injection** and **monochromatic photon injection** into the CMB photon-baryon plasma from post-recombination (z ~ 100) through the thermalization era (z ~ few × 10⁶). At z > 1100, injected energy is Comptonized into μ/y-type spectral distortions. At z < 1100, Compton scattering is inefficient (X_e ~ 10⁻⁴), so distortions are "locked-in" at their injection frequency with no μ/y redistribution. The heat-injection Green's function (`greens_function`, `distortion_from_heating`) uses the Chluba 2013 visibility fits and is **not** cosmology-aware. For photon injection (`greens_function_photon`), a `cosmo=` keyword feeds the photon survival probability `P_s` (DC+BR optical-depth integral) and the Compton-y broadening `y_γ = ∫ θ_e σ_T n_e c / H dz` of the surviving bump. It computes the resulting spectral distortions (μ, y, temperature shift) via both the PDE solver and the Green's function. Electron injection is not handled.

## Architecture

CMB spectral distortion solver: evolves photon occupation number n(x, z) through the coupled photon-electron Boltzmann equation. Two modes: full PDE solver and fast Green's function approximation.

### Rust modules (src/)

**Physics layer** — each module owns one physical process:
- `kompaneets.rs` — Compton scattering via Fokker-Planck equation. IMEX solver: Crank-Nicolson for Kompaneets + backward Euler for DC/BR, with nonlinear Newton iteration. Largest and most numerically delicate module.
- `double_compton.rs` — DC emission (γe → γγe), photon-number changing. Semi-implicit backward Euler.
- `bremsstrahlung.rs` — BR emission (e+ion → e+ion+γ). Non-relativistic Gaunt factor: Born approximation (Brussaard & van de Hulst 1962) with softplus interpolation (Chluba, Ravenni & Bolliet 2020, MNRAS 492, 177).
- `electron_temp.rs` — Electron temperature T_e. Perturbative quasi-stationary equilibrium.
- `recombination.rs` — Ionization fraction X_e(z). Peebles 3-level atom ODE (z<1500), Saha (z>1500). Cached with O(log N) lookup.
**Infrastructure layer**:
- `constants.rs` — CODATA 2018 constants, spectral integrals (G₁, G₂, G₃), β_μ, κ_c.
- `cosmology.rs` — Flat ΛCDM background: H(z), densities, Thomson time. Default params: Y_p=0.24, Ω_b=0.044, h=0.71.
- `spectrum.rs` — Planck/Bose-Einstein distributions, spectral shapes M(x), Y_SZ(x), G_bb(x).
- `grid.rs` — Non-uniform frequency grid: log-spaced at low x (where DC/BR diverge), linear at high x. Supports `RefinementZone` for adaptive local refinement near injection features.
- `energy_injection.rs` — Injection scenarios: SingleBurst, DecayingParticle, AnnihilatingDM, AnnihilatingDMPWave, MonochromaticPhotonInjection, DecayingParticlePhoton, DarkPhotonResonance, TabulatedHeating, TabulatedPhotonSource, Custom.
- `dark_photon.rs` — NWA helpers for γ↔A' (plasma_frequency_ev, resonance_redshift, gamma_con). Used by `InjectionScenario::DarkPhotonResonance`, which installs the impulsive Δn IC at z_res; the solver auto-sets z_start = z_res.

**Solver layer**:
- `solver.rs` — Main PDE integrator. Couples Kompaneets + DC/BR in a joint Newton iteration with adaptive redshift stepping. The core type is `ThermalizationSolver`. `full_te=true` (full quasi-stationary T_e with DC/BR heating integrals) is the default and should stay on; `set_injection()` does not auto-disable it. The simple mode (ρ_e = ρ_eq + δρ_inj without H_dcbr) remains available via `set_full_te(false)` but is treated as an unnecessary approximation given the < 0.5% overhead.
- `greens.rs` — Fast approximate mode: Green's function G_th with visibility functions J_bb*, J_μ, J_y (Chluba 2013).
- `distortion.rs` — Decompose Δn into (μ, y, ΔT/T) via joint least-squares. Convert to intensity (MJy/sr).

**Entry points**:
- `lib.rs` — Library root. `prelude` module re-exports `Cosmology`, `ThermalizationSolver`, `SolverConfig`, `GridConfig`, `FrequencyGrid`, `InjectionScenario`.
- `main.rs` — CLI binary entry. Subcommands: `solve`, `sweep`, `greens`, `info`, `help`.
- `cli.rs` — CLI argument parsing and dispatch. Handles JSON output, diagnostic flags (`--no-dcbr`, `--number-conserving`, `--split-dcbr`).
- `output.rs` — JSON serialization of `SolverResult` / `SolverSnapshot`.
- `bin/check_adiabatic.rs` — Utility for adiabatic cooling validation (only maintained binary).

### Python package (python/spectroxide/)

- `__init__.py` — Public API. `strip_gbb*` and `apply_style/C/SINGLE_COL/DOUBLE_COL` are re-exported at top level. Additional utilities via submodule import (`from spectroxide.cosmotherm import ...`, `from spectroxide.plot_params import ...`).
- `greens.py` — Pure Python port of Rust Green's function (NumPy vectorized). All visibility/spectral functions.
- `solver.py` — `run_sweep()` calls the Rust binary via subprocess; `run_single()` uses pure-Python Green's function. Shared helpers: `_build_common_solver_args()`, `_run_rust_binary()`.
- `cosmotherm.py` — CosmoTherm data loaders: DI files, Green's function database. Not re-exported at top level.
- `dark_photon.py` — Pure-Python NWA helpers (γ_con, z_res, ω_pl). Mirrors `src/dark_photon.rs`.
- `firas.py` — FIRAS monopole + 43×43 covariance matrix; χ² fitting utilities for spectral distortions.
- `greens_table.py` — Precomputed Green's function tables for fast convolution.
- `plot_params.py` — Plot parameter constants. Not re-exported at top level.
- `style.py` — Matplotlib style helpers (`apply_style`, `C`, `SINGLE_COL`, `DOUBLE_COL`).
- `_validation.py` — Input validation: errors for nonsensical inputs, warnings for untested regimes.

### Integration tests (tests/)

- `heat_injection.rs` — 201 integration tests: mathematical identities, Green's function constraints, PDE vs GF cross-validation, physical scenarios, literature benchmarks, dark sector, advanced PDE, BR/DC regression, recombination, T_e coupling, decomposition, solver robustness, photon injection.
- `adversarial_inputs.rs` — 17 tests: edge cases, invalid inputs, boundary conditions.
- `coverage_gaps.rs` — 14 tests: closes coverage gaps flagged during audit (energy conservation, warning thresholds, table I/O, boundary conditions, grid refinement).
- `cosmotherm_comparison.rs` — 7 tests: cross-validation against CosmoTherm reference data (DI_cooling, DI_damping, adiabatic μ).
- `greens_function_checks.rs` — 7 tests: Chluba 2013 Green's function limits (μ-era, y-era, pure temperature shift) and Gaunt-factor spot checks.
- `convergence_order.rs` — 8 tests + 1 ignored: grid and timestep convergence with two-sided Richardson-order bounds.
- `cli_integration.rs` — 4 tests: CLI end-to-end.
- `science_suite.rs` — 5 tests: end-to-end physics validation.

### Notebooks (notebooks/)

**`tutorials/`** — User-facing tutorial sequence:
- `01_getting_started.ipynb` — Quick start guide
- `02_energy_injection.ipynb` — Energy injection scenarios
- `03_new_physics.ipynb` — Dark photon, photon injection
- `04_custom_scenarios.ipynb` — Custom scenarios, tabulated sources
- `05_observational_constraints.ipynb` — FIRAS/PIXIE constraints
- `06_greens_table.ipynb` — Precomputed Green's function tables

**`physics/`** — Specific physics topics:
- `adiabatic_cooling.ipynb` — Adiabatic-cooling sanity checks
- `photon_injection.ipynb` — Monochromatic photon injection (Chluba 2015)
- `photon_injection_validation.ipynb` — Photon-injection validation against literature

**`observational/`** — FIRAS/PIXIE constraints:
- `firas_photon_limits.ipynb` — FIRAS photon-injection limits

**`paper_figures/`** — Self-contained notebooks, one per paper figure (10 notebooks). Generated by `_generate_notebooks.py` from source scripts/notebooks.

**`figures/`** — Generated PDF figures consumed by the paper.

### Development artifacts (dev/)

- `dev/scripts/` — 10 validation and diagnostic scripts (build_gf_table, build_visibility_table, convergence_figure, dm_cosmotherm_compare, fit_visibility_from_table, photon_energy_conservation, plot_visibility_comparison, remake_firas_photon_limits, benchmark_paper_table, check_refs)
- `dev/notebooks/` — 4 notebooks: cosmology_background, mu_y_vs_zh, pde_greens_function, pde_validation

## Critical Numerical Pitfalls

These are the hard-won lessons from development. **Violating any of these will silently produce wrong results:**

1. **Kompaneets cancellation**: The Planck identity dn_pl/dx + n_pl(1+n_pl) = 0 MUST be used analytically. Finite-difference error O(dx²) ≈ 0.003 is ~1000× the physical signal O(ρ_e−1) ≈ 10⁻⁵. The flux is split as: `F = x⁴[(φ−1)n_pl(1+n_pl) + dΔn/dx + φ(2n_pl+1)Δn + φΔn²]`, where **φ ≡ T_z/T_e = 1/ρ_e** (note: the code's convention, opposite of the intuitive T_e/T_z because x is normalised by T_z). When T_e = T_z the (φ−1) term vanishes and the Planck-subtracted flux contains only terms linear and quadratic in Δn.

2. **CFL instability**: Explicit Kompaneets at low x requires dt < dx²/(2θ_e x²) ~ 3, but steps are ~50. Must use implicit (Crank-Nicolson).

3. **DC/BR divergence at low x**: Emission rate ∝ 1/x³ → ~10⁶. Operator-split with semi-implicit backward Euler, not Crank-Nicolson (which gets negative diagonals).

4. **T_e feedback cancellation**: Full I₄/(4G₃) has 0.1% numerical error swamping the O(10⁻⁵) physical correction. Must use perturbative: Δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃ from Δn only.

5. **DC/BR source near-cancellation**: n_pl(x/ρ_e) − n_pl(x) subtracts nearly-equal numbers. Use analytical expansion x(ρ_e−1)/ρ_e × n_pl(1+n_pl) when |ρ_e−1| < 0.01.

6. **NaN hiding**: `f64::max(NaN, x)` returns x. Always use `.filter(|x| x.is_finite())` before fold-based max.

7. **Grid extent**: x_max must be ≥ 30 for accurate G₃ integrals. Log spacing needs many points at low x.

8. **Dimensional analysis as first-line defense**: Every physical coefficient must be checked for correct dimensions BEFORE trusting numerical output. Example: BR emission coefficient K_BR must be dimensionless (rate per Thomson time). BR_PREFACTOR [m³] × Σ Z²N_i [1/m³] = dimensionless ✓. An extra /n_e made it [m³] and suppressed BR by ~10¹¹×, but this was invisible for heat injection (where DC dominates) and went undetected through 375 tests. **Always verify dimensions of rate coefficients, especially when adapting formulas between per-volume and per-Thomson-time conventions.** Two-body processes (BR: e+ion) keep one density factor after Thomson normalization; one-body processes (DC: γ+e) cancel completely.

9. **Tests calibrated to code output cannot catch systematic errors**: If a test asserts `DC/BR > 1e10` because the code produces that value, the test passes even though the physical ratio should be ~17. Tests for physical quantities must derive targets from independent sources (analytic formulas, literature values, dimensional arguments), not from the code itself. When a computed quantity seems extreme, ask: "Is this physically reasonable?"

10. **Unsafe indexing in hot loops**: `kompaneets.rs` uses `get_unchecked` in the Thomas solver, K_old precompute, and Newton inner loop for performance (~15-20% speedup). Safety is guaranteed by `assert!` guards at function entry that verify all slice lengths. **When modifying workspace fields, adding new arrays to the Newton loop, or changing grid sizes, you must update the corresponding asserts.** Debug-mode `debug_assert!` checks also validate inputs (NaN, physical ranges) — these run in `cargo test` but are stripped in release. Run tests in debug mode after any change to these functions.

## Dimensionless Variables

- x = hν/(kT_z): frequency
- θ_e = kT_e/(m_ec²): electron temperature
- θ_z = kT_z/(m_ec²) ≈ 4.60×10⁻¹⁰(1+z)
- Δn = n − n_pl: the distortion (the thing being solved for)
- ρ_e = T_e/T_z: electron-photon temperature ratio

## Validation Targets

- μ ≈ 1.401 × Δρ/ρ for injection in deep μ-era (z > 2×10⁵)
- y = Δρ/(4ρ) for injection in y-era (z < 10⁴)
- PDE vs Green's function: 2–5% agreement for μ, ~5% for y
- Energy conservation < ±5% across all redshifts
- Photon number conservation under pure Compton scattering

## Working Style

- When I ask for scaffolding or a plan, provide ONLY scaffolding/TODOs — do not implement the actual logic unless I explicitly ask you to implement it.
- When I exit plan mode or ask you to implement, stop planning and start writing code immediately. Do not continue writing plan files.
- Keep tone professional. Bluntness and directness are good; vulgar shorthand or crude abbreviations (e.g. "idgaf") are not.

## Environment

Before running Jupyter notebooks, verify the correct Python/conda environment path. Use the miniforge installation and check `which python` and `which jupyter` to avoid PATH shadowing issues. If jupyter fails, fall back to running cells as standalone Python scripts.

## Debugging Philosophy

- Before proposing a fix for a numerical discrepancy, first check whether the underlying solver has a mode or configuration that handles the physics correctly (e.g., number_conserving mode). Don't filter or patch outputs — fix the root cause.
- When debugging numerical discrepancies against reference papers, do NOT filter out or mask problematic data points as a first approach. Instead, investigate the underlying physics or solver configuration (e.g., number_conserving mode, correct reference values) to match the reference methodology.
- When I reference a paper's formula or parameter value (e.g., Qh_ref = 1e3), use that exact value. Do not substitute your own assumptions.
- **After any physics code change, run the solver and compare outputs to reference data (CosmoTherm, DarkHistory, literature values) before making any claims about whether the change affects results.** Do not use theoretical reasoning to dismiss potential impacts — check numerically. "This shouldn't matter because..." is a red flag for the exact reasoning pattern that produces systematic errors.

## Rust Development

After making changes to Rust struct definitions (adding/removing fields), immediately grep for all existing struct literal instantiations in tests and examples and update them too.

## Testing & CI

After any multi-file edit that touches Rust code, always run `cargo clippy -- -D warnings` and `cargo test` before committing. Clippy warnings are treated as errors in CI.

## Git Conventions

Always include `[skip ci]` on commits that only touch documentation, paper (.tex), or non-code files.
