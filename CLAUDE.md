# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a Rust PDE solver (spectroxide) with Python bindings and Jupyter notebooks. The primary focus is always the PDE solver — not Green's functions — unless I explicitly say otherwise.

## Build & Test

```bash
cargo build --release          # Build optimized binary
cargo test --release           # Run all tests (209 unit + 258 integration + 3 doc pass; +5 ignored). Never run tests in debug mode.
cargo test --release test_name # Run a single test by name
CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true cargo test --release --lib  # Release build with debug_assert! checks on
cargo run --release --bin spectroxide -- sweep  # Run PDE sweep over default z_h grid
cargo run --release --bin spectroxide          # Print help / list subcommands
cargo run --release --bin check_adiabatic      # Utility: adiabatic cooling check
```

**Python package** (wraps Rust binary + pure-Python Green's function):
```bash
cd python && pip install -e ".[plot]"    # Install with matplotlib
cd python && pip install -e ".[notebook]" # Install with jupyter too
```

**Key constraints**: Zero production Rust dependencies (pure std library). Dev-dependencies are `approx` (float comparison in tests) and `criterion` (benchmarks).

**Cargo features**: `axion` (off by default) gates resonant axion–photon conversion — `src/axion.rs`, `InjectionScenario::AxionResonance`, the `solve axion-resonance` subcommand, and five tests in `tests/dark_sector.rs`. It is experimental and excluded from the release. The user-facing docs (README, Sphinx, public docstrings) omit it on purpose; instead each axion run pushes `axion::EXPERIMENTAL_WARNING`, and the Python helpers in `spectroxide.axion` warn once per process. Build/test it with `--features axion`. **Both configurations must build, test and pass clippy** — check the `not(feature)` arm when touching `InjectionScenario` matches, `axion_params`, or `warn_axion_range` (the latter two are defined in both configurations, returning `None`/empty when off, so call sites need no `cfg`).

## Scope

This fork handles **heat injection** and **monochromatic photon injection** into the CMB photon-baryon plasma from post-recombination (z ~ 100) through the thermalization era (z ~ few × 10⁶). At z > 1100, injected energy is Comptonized into μ/y-type spectral distortions. At z < 1100, Compton scattering is inefficient (X_e ~ 10⁻⁴), so distortions are "locked-in" at their injection frequency with no μ/y redistribution. The heat-injection Green's function (`greens_function`, `distortion_from_heating`) uses the Chluba 2013 visibility fits and is **not** cosmology-aware. For photon injection (`greens_function_photon`), a `cosmo=` keyword feeds the photon survival probability `P_s` (DC+BR optical-depth integral) and the Compton-y broadening `y_γ = ∫ θ_e σ_T n_e c / H dz` of the surviving bump. It computes the resulting spectral distortions (μ, y, temperature shift) via both the PDE solver and the Green's function. Electron injection is not handled.

## Architecture

CMB spectral distortion solver: evolves photon occupation number n(x, z) through the coupled photon-electron Boltzmann equation. Two modes: full PDE solver and fast Green's function approximation.

### Rust modules (src/)

**Physics layer** — each module owns one physical process:
- `kompaneets.rs` — Compton scattering via Fokker-Planck equation. IMEX solver: Crank-Nicolson for Kompaneets + backward Euler for DC/BR, with nonlinear Newton iteration. Largest and most numerically delicate module.
- `double_compton.rs` — DC emission (γe → γγe), photon-number changing. Semi-implicit backward Euler.
- `bremsstrahlung.rs` — BR emission (e+ion → e+ion+γ). Non-relativistic Gaunt factor: Born approximation (Brussaard & van de Hulst 1962) with softplus interpolation (Draine 2011, *Physics of the Interstellar and Intergalactic Medium*, Ch. 10).
- `electron_temp.rs` — Electron temperature T_e. Perturbative quasi-stationary equilibrium. The solver evolves X_H alongside T_e below z ≈ 1575 by a lagged split with an X_H corrector pass, and caps the step there at `XE_COUPLED_DZ_FRAC` = 0.005 z (ADR 0009); `--fixed-ionization` / `.fixed_ionization()` restores the standard table.
- `recombination.rs` — Ionization fraction X_e(z). Peebles 3-level atom ODE (z<1575) with RECFAST 1.5.2's fudge factor F = 1.125 and Lyman-α escape correction (ADR 0011), Saha (z>1575). The cached standard table (gas at T_z, O(log N) lookup) feeds the Green's function and the fixed-ionization mode; `advance_x_h` evolves X_H at the gas temperature for the PDE solver (ADR 0009). Helium stays Saha at T_z.
**Infrastructure layer**:
- `constants.rs` — CODATA 2018 constants, spectral integrals (G₁, G₂, G₃), β_μ, κ_c.
- `cosmology.rs` — Flat ΛCDM background: H(z), densities, Thomson time. Default params: Y_p=0.24, Ω_b=0.044, h=0.71.
- `spectrum.rs` — Planck/Bose-Einstein distributions, spectral shapes M(x), Y_SZ(x), G_bb(x).
- `grid.rs` — Non-uniform frequency grid: log-spaced at low x (where DC/BR diverge), linear at high x. Supports `RefinementZone` for adaptive local refinement near injection features.
- `energy_injection.rs` — Injection scenarios: SingleBurst, DecayingParticle, AnnihilatingDM, AnnihilatingDMPWave, MonochromaticPhotonInjection, DecayingParticlePhoton, DarkPhotonResonance, TabulatedHeating, TabulatedPhotonSource, Custom, plus AxionResonance behind the `axion` feature.
- `dark_photon.rs` — NWA helpers for γ↔A' (plasma_frequency_ev, resonance_redshift, gamma_con). Used by `InjectionScenario::DarkPhotonResonance`, which installs the impulsive Δn IC at z_res; the solver auto-sets z_start = z_res.
- `axion.rs` — **behind the off-by-default `axion` feature.** NWA helpers for γ↔a (Cyr, Chluba & Manoj 2024): Wien-tail depletion, P(x) = 1 − exp(−γ_con x), κ = g_aγγ B_rms. Reuses `dark_photon::{resonance_redshift, dln_omega_pl_sq_dlna}`. Monopole (m_γ² ≈ ω_pl²) treatment only.

**Solver layer**:
- `solver.rs` — Main PDE integrator. Couples Kompaneets + DC/BR in a joint Newton iteration with adaptive redshift stepping. The core type is `ThermalizationSolver`. T_e always uses the full quasi-stationary solution with the DC/BR heating integrals; the older simple mode (ρ_e = ρ_eq + δρ_inj without H_dcbr) and its `set_full_te` switch have been removed.
- `greens.rs` — Fast approximate mode: Green's function G_th with visibility functions J_bb*, J_μ, J_y (Chluba 2013).
- `distortion.rs` — Decompose Δn into (μ, y, ΔT/T) via joint least-squares. Convert to intensity (MJy/sr).

**Entry points**:
- `lib.rs` — Library root. `prelude` module re-exports `Cosmology`, `ThermalizationSolver`, `SolverConfig`, `GridConfig`, `FrequencyGrid`, `InjectionScenario`.
- `main.rs` — CLI binary entry. Subcommands: `solve`, `sweep`, `photon-sweep`, `photon-sweep-batch`, `greens`, `info`, `physics-hash`, `help`.
- `cli.rs` — CLI argument parsing and dispatch. Handles JSON output, diagnostic flags (`--no-dcbr`, `--no-number-conserving`, `--split-dcbr`, `--fixed-ionization`).
- `output.rs` — JSON serialization of `SolverResult` / `SolverSnapshot`.
- `bin/check_adiabatic.rs` — Utility for adiabatic cooling validation (only maintained binary).

### Python package (python/spectroxide/)

- `__init__.py` — Public API. `strip_gbb*` and `apply_style/C/SINGLE_COL/DOUBLE_COL` are re-exported at top level. Additional utilities via submodule import (`from spectroxide.cosmotherm import ...`, `from spectroxide.plot_params import ...`).
- `cosmology.py` — Flat ΛCDM background (`Cosmology` dataclass, three presets, `hubble`/`cosmic_time`/`ionization_fraction`). Mirrors `src/cosmology.rs`. Source of 11 top-level exports.
- `greens.py` — Pure Python port of Rust Green's function (NumPy vectorized). All visibility/spectral functions.
- `solver.py` — `run_sweep()` calls the Rust binary via subprocess; `run_single()` uses pure-Python Green's function. Shared helpers: `_build_common_solver_args()`, `_run_rust_binary()`.
- `cosmotherm.py` — CosmoTherm data loaders: DI files, Green's function database. Not re-exported at top level.
- `dark_photon.py` — Pure-Python NWA helpers (γ_con, z_res, ω_pl). Mirrors `src/dark_photon.rs`.
- `axion.py` — Pure-Python NWA helpers for γ↔a. Mirrors `src/axion.rs`. Importable regardless (calls no Rust), but its PDE path needs the binary built `--features axion`; marked experimental and not re-exported at top level.
- `firas.py` — FIRAS monopole + 43×43 covariance matrix; χ² fitting utilities for spectral distortions.
- `greens_table.py` — Precomputed Green's function tables for fast convolution.
- `plot_params.py` — Plot parameter constants. Not re-exported at top level.
- `style.py` — Matplotlib style helpers (`apply_style`, `C`, `SINGLE_COL`, `DOUBLE_COL`).
- `_validation.py` — Input validation: errors for nonsensical inputs, warnings for untested regimes.

### Integration tests (tests/)

Shared setup lives in `tests/common/mod.rs`: burst and photon-injection builders, the energy and number integrals, and `memo`, which runs each distinct PDE configuration once per test binary (keyed on every input; the solver is deterministic). Hand-written physics oracles stay in each test file.

- `pde_heat.rs` — 42 tests + 1 ignored: heat-injection PDE runs (bursts, decaying particles, DM annihilation, custom and tabulated heating): energy conservation, μ and y against era targets and the Green's function, spectral shapes and golden references, linearity, the μ–y transition.
- `pde_photon.rs` — 26 tests: photon-injection PDE runs (Gaussian initial spectra, monochromatic and decaying-particle photon scenarios, post-recombination): μ(x_inj) against Chluba (2015), energy and number bookkeeping, Green's-function comparisons.
- `solver_numerics.rs` — 15 tests: stability at large Δn and large steps, snapshot landing, adaptive stepping, number-conserving mode, free streaming, timestep convergence, null tests (no injection, adiabatic cooling).
- `gf_visibility.rs` — 20 tests, no PDE runs: visibility functions and their limits, heat and photon Green's functions, the (μ, y, ΔT/T) decomposition, FIRAS helpers.
- `components.rs` — 23 tests, no PDE runs: cosmology background, grid, DC/BR rates, Kompaneets kernel, injection-rate functions, table I/O, quasi-stationary T_e.
- `dark_sector.rs` — 5 tests in the default build (dark-photon depletion, including the install redshift and the off-by-default neutral-hydrogen switch of ADR 0008); 5 more behind `--features axion`, which also enables 4 unit tests in `src/axion.rs`, so the feature adds 9 tests in total.
- `adversarial_inputs.rs` — 19 tests: edge cases, invalid inputs, boundary conditions, rejected solver tolerances (R-2), refinement zones that overlap the grid (N-2).
- `coverage_gaps.rs` — 22 tests: closes coverage gaps flagged during audit (energy conservation, warning thresholds, table I/O, boundary conditions, grid refinement), plus the post-run energy-closure and small-grid warnings (R-1), and full heat delivery from narrow bursts that drive T_e far above T_z (ADR 0007). `GridConfig::validate` rejects `n_points < 100` (review decision 2026-09-23; a sanity floor, not an accuracy bound). A 50-point run — built directly via `ThermalizationSolver::new` + `set_injection`, bypassing `validate`, since the builder can no longer construct it — must stop with the NaN error, never panic; a run on the 100-point validation floor must be finite and warn, or fail cleanly, never return silently.
- `cosmotherm_comparison.rs` — 7 tests + 1 ignored: cross-validation against CosmoTherm reference data (DI_cooling, DI_damping, adiabatic μ), plus a μ-era decay against the CosmoTherm GF database (ignored by default; needs `Greens_data.dat` and `SPECTROXIDE_GREENS_DB`).
- `greens_function_checks.rs` — 10 tests: Chluba 2013 Green's function limits (μ-era, y-era, pure temperature shift), energy conservation, and PDE cross-validation, plus four anchors of the BR Gaunt factor and coefficient against Draine (2011).
- `convergence_order.rs` — 8 tests + 1 ignored: grid and timestep convergence with two-sided Richardson-order bounds.
- `cli_integration.rs` — 5 tests: CLI end-to-end.
- `science_suite.rs` — 6 tests: end-to-end physics validation.
- `physics_identities.rs` — 12 tests + 1 ignored: closed-form and published identities added by the physics-check audit (`dev/audit/PHYSICS_CHECKS_STATUS_2026-07-26.md`): Thomson depth vs Planck z_*, exact moments of the G_bb/M/Y shapes, Kompaneets first/second moment identities and H-theorem, quasi-stationary T_e energy return, DC/BR crossover redshift, grid-boundary independence, T_e Compton/adiabatic balance, α_th = 5/2 (ignored, ~7 min), plus the sensitivity-directed photon anchors T-PS-1/2/3 (P_s(x_c) = 1/e, x_c vs Chluba 2015 Eq. 25, μ at x_inj = x_c), and the reduced-mass hydrogen ionization energy from typed CODATA values.
- `mms_convergence.rs` — 8 tests: method of manufactured solutions on the Kompaneets kernel and the coupled path, plus the photon-number ledger identity. **Verifies the discretization, not the equation** — see Pitfall #11.
- `conservation_fuzz.rs` — 3 tests: randomized energy/number-closure fuzzing across scenarios and grids.
- `ionization_coupling.rs` — 5 tests: X_H evolved with T_e (ADR 0009). A no-injection run matches HyRec-2 X_e and T_m after freeze-out (z = 200, 100, 50), where the fixed history is off by 2–10% and fails the same bands; the fixed mode reproduces the standard table exactly; runs ending above recombination are bit-identical in both modes; smooth heating through recombination (a decay) gives X_e within 1% of small steps; a burst at z_h = 600 matches X_e(200) and the coupling's change in delivered heat from the independent `dev/scripts/heatloss/coupled_xe_expectation.py`.
- `heat_delivery.rs` — 2 tests: fraction of injected heat that reaches the photons after recombination, pinned to an independent gas-temperature integration (`dev/scripts/heatloss/`, `dev/audit/fix_a_cn_old_half_ab.md`). The burst at z_h = 1000 must deliver 0.999856 ± 3e-5 (`coupled_xe_expectation.py`, which evolves X_H with T_m as the solver does since ADR 0009; 0.999863 with a fixed X_e); the tolerance is a fifth of the 1.44e-4 physical loss, so full delivery (1.0) fails. The decay with lifetime at z = 1000 must deliver 0.99418 ± 1e-3. Both fail without ADR 0004 (0.938 and 0.849).

**The moment-hierarchy suite** (`dev/audit/KOMPANEETS_VERIFICATION_RESULTS.md`, plan `dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md`). These exist to pin the *formulation* against targets derived outside the code, closing the Pitfall #11 gap that MMS cannot reach. Coverage is tracked per physical term in `dev/audit/term_coverage_matrix.md`.
- `kompaneets_moments.rs` — 11 tests: the exact moment hierarchy `dM_k/dy = (k−2)(k+1)M_k − (k−2)M_{k+1}` (derived by integration by parts on the *published* Kompaneets equation, coefficients not taken from the code) at k = 3,4,5; the Zel'dovich–Sunyaev energy law; the (φ−1) heating branch against the analytic Y_SZ shape *and* amplitude — the only test that exercises that branch, since every other kernel test runs at φ = 1 where it vanishes; a Δn² linearity diagnostic; and the H-theorem at Δn ~ n_pl, the only fully nonlinear check in the repo. Two tiers: tier-a carries the independent physics, tier-b adds the measured stimulated/quadratic term `C_k` to separate regime contamination from real failure.
- `compton_equilibrium_analytic.rs` — 4 tests: the perturbative Δρ_eq coefficients (Pitfall #4) against mpmath quadrature at dps = 40 (`dev/scripts/compton_equilibrium_coefficients.py`, which imports nothing from spectroxide). Uses the difference method, since reading the absolute ratio would re-commit Pitfall #4.
- `rate_coefficients_first_principles.rs` — 3 tests: DC and BR coefficient magnitudes from CODATA constants **typed literally into the test file**, nothing imported from `constants.rs`, checked for z-independence across three redshifts. This is the test class that would have caught the historical 10¹¹× BR bug (Pitfall #8).
- `mu_photosphere_profile.rs` — 2 tests (~41 s): fitted μ-photosphere x_c(z) vs Chluba (2015) Eq. 25 at z = 2×10⁶ (DC-dominated) and 3×10⁵ (BR-significant). The only test of the *coupled* DC/BR + Compton balance against an analytic target rather than CosmoTherm's 2–5% envelope.

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
- `injection_width_resolution.ipynb` — Sensitivity of μ, y, and the spectrum to the burst's temporal and spectral widths (referee 2, comment 4)
- `photon_injection.ipynb` — Monochromatic photon injection (Chluba 2015)
- `photon_injection_validation.ipynb` — Photon-injection validation against literature

**`observational/`** — FIRAS/PIXIE constraints:
- `firas_photon_limits.ipynb` — FIRAS photon-injection limits

**`paper_figures/`** — Self-contained notebooks, one per paper figure (11 notebooks). Edit them directly; there is no generator.

**`figures/`** — Generated PDF figures consumed by the paper.

### Development artifacts (dev/)

- `dev/scripts/` — 24 validation and diagnostic scripts (build_gf_table, build_visibility_table, build_baseline_table, fit_visibility_conservation, convergence_figure, mms_convergence_figure, dm_cosmotherm_compare, fit_visibility_from_table, photon_energy_conservation, plot_visibility_comparison, remake_firas_photon_limits, benchmark_paper_table, check_refs, class_sd_compare, class_sd_case_b, compton_equilibrium_coefficients, gamma_con_landau_zener, highprec_oracle, extract_test_assertions, error_budget, build_test_provenance, bryce2411_red_sensitivity, docs_style_lint, nb_md_replace, plus the ccj24_gap/, dm_residual_diagnostics/, heatloss/, visibility_diagnostics/, and y_estimator/ subdirectories). `build_visibility_table` -> `build_baseline_table` -> `fit_visibility_conservation` is the paper's Table 1 (visibility-function fit) pipeline.
- `dev/audit/` — validation records. Two coverage matrices, deliberately: `coverage_matrix.md` is indexed by *published result* (one row per paper figure, R0), `term_coverage_matrix.md` by *physical term* in the code. Do not merge them; do not rename `term_coverage_matrix.md` back to `COVERAGE_MATRIX.md` (case-insensitive collision breaks macOS/Windows checkouts).
- `dev/verify_refs.html` — BibCheck: open in a browser to step through the paper's `refs.bib` (in `../cosmoxide/paper/`) and compare each entry with CrossRef, DataCite, and INSPIRE (for arXiv numbers). Keep it identical to `../cosmoxide/dev/verify_refs.html`; the refresh snippet is in its header comment.
- `dev/notebooks/` — 6 notebooks: cosmology_background, mu_y_vs_zh, pde_greens_function, pde_validation, remake_pathological_figure, xe_darkhistory_comparison (coupled X_e/T_e against DarkHistory's three-level atom, paper Appendix `app:xe`; needs `pip install darkhistory`, and uses `examples/xe_history.rs` for snapshot histories)

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

10. **Unsafe indexing in hot loops**: `kompaneets.rs` uses `get_unchecked` in the Thomas solver, K_old precompute, and Newton inner loop for performance (~15-20% speedup). Safety is guaranteed by `assert!` guards at function entry that verify all slice lengths. **When modifying workspace fields, adding new arrays to the Newton loop, or changing grid sizes, you must update the corresponding asserts.** `debug_assert!` checks also validate inputs (NaN, physical ranges); they are stripped from a normal release build. After any change to these functions, run the tests in release mode with `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true`, which keeps them on. Never run tests in debug mode.

11. **MMS verifies the scheme, not the equation**: the manufactured residual `S = ∂_τΔn_m − L[Δn_m]` is built from an operator `L` transcribed from the code's own flux form. Any coefficient error in that form — recoil `2Δn` instead of `Δn`, flux weighted `x³` instead of `x⁴`, the wrong θ normalizing the Comptonization variable — appears in *both* the code and the residual, cancels exactly, and MMS still reports clean convergence at p = 2.00. **A term whose only strong test is MMS or Richardson order is unverified physics.** Pin the formulation separately against targets derived outside the code: exact moment identities, mpmath quadrature, literal CODATA constants, literature fits (`tests/kompaneets_moments.rs` and siblings; `dev/audit/term_coverage_matrix.md` tracks which terms still lack such an anchor). This is the same failure mode mutation testing found from the other end — there a coefficient was written twice and the literature anchor tested the unused copy (F-R2-3). Test *construction* and test *placement* fail independently, and a coverage percentage detects neither.

12. **Cross-code source-term hand-offs: audit term content, not conventions**: when another code hands us a source term (a heating table, an injection history), matching sign, units and variable is necessary and not close to sufficient. Ask *which physical processes are inside that number*, and whether this solver already models any of them. Anything the PDE models unconditionally — adiabatic cooling via Λρ_e above all — is a double-count hazard by construction. This bit us for real: CLASS's `_sd_heating.dat` includes first-order photon-baryon cooling, feeding it verbatim to the PDE counted the cooling twice, and it produced a plausible μ discrepancy that was written up as physics (finding R1-A, since retracted; ~51% of the effect was bookkeeping). **It was invisible to inspection** because acoustic dissipation dominates the column at every z, so no entry ever turned negative. See `dev/audit/class_sd_comparison.md` and the `--subtract-cooling` path in `dev/scripts/class_sd_compare.py`.

## Dimensionless Variables

- x = hν/(kT_z): frequency
- θ_e = kT_e/(m_ec²): electron temperature
- θ_z = kT_z/(m_ec²) ≈ 4.60×10⁻¹⁰(1+z)
- Δn = n − n_pl: the distortion (the thing being solved for)
- ρ_e = T_e/T_z: electron-photon temperature ratio

## Validation Targets

- μ ≈ 1.401 × Δρ/ρ for injection in deep μ-era (z > 3×10⁵)
- y = Δρ/(4ρ) for injection in y-era (z < 10⁴)
- PDE vs Green's function: 2–5% agreement for μ, ~5% for y
- Energy conservation < ±5% across all redshifts
- Photon number conservation under pure Compton scattering

## Working Style

- When I ask for scaffolding or a plan, provide ONLY scaffolding/TODOs — do not implement the actual logic unless I explicitly ask you to implement it.
- When I exit plan mode or ask you to implement, stop planning and start writing code immediately. Do not continue writing plan files.
- Keep tone professional. Bluntness and directness are good; vulgar shorthand or crude abbreviations are not.

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
