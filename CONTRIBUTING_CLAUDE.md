# LLM context file for spectroxide contributors

This file is context for a large language model (LLM), such as Claude, GPT, or Gemini, that
helps you contribute to spectroxide. Give it to the model before work starts:

- In Claude Code, `CLAUDE.md` loads automatically but this file does not. Start the session by
  asking the model to read `CONTRIBUTING_CLAUDE.md`, or mention it with `@CONTRIBUTING_CLAUDE.md`.
- In other tools, paste this file into the system prompt or the project instructions.

`CLAUDE.md` is the canonical reference for architecture, numerical pitfalls, and validation
targets. If the two files disagree, `CLAUDE.md` wins, and the disagreement is a bug in this
file. This file adds the contribution workflow and the rules for tests and review.

## Why these rules exist

spectroxide is a numerical physics code, and plausible code in it can be silently wrong. Five
physics bugs passed the full automated test suite during development. A human caught each one
by physical reasoning. Section 8 of the paper (Baker, Liu & Mishra-Sharma 2026) tells the
story. The bugs, and the rule each one taught, were these:

1. The bremsstrahlung (BR) emission coefficient had a spurious 1/n_e, which suppressed BR by
   about 10¹¹. It passed 375 tests because every BR test target had been read off the code's
   own output. Rule: derive targets independently, and check the dimensions of every rate
   coefficient.
2. The double Compton (DC) and BR absorption term dropped the detailed-balance factor, which
   gave μ/(Δρ/ρ) = 0.66 at z = 10⁶ instead of about 1.16. Rule: check equilibrium limits.
3. An energy-leak correction was added along an ad hoc spectral shape, which put a spurious y
   into the output. Rule: no ad hoc numerical fixes; the correct shape is G_bb, a temperature
   shift.
4. Energy that DC and BR absorbed from injected soft photons was put back as a temperature
   shift instead of heating the electrons. Energy was conserved, but the spectral shape was
   wrong. Rule: energy tests alone cannot validate a shape.
5. The adiabatic cooling term in the electron-temperature equation was written as −Λ(ρ_e − 1)
   instead of −Λρ_e, which removed adiabatic cooling. The signal (μ ≈ −3×10⁻⁹) was far below
   every test amplitude. Rule: test each term where its effect is large.

The same paper section describes the behavior that made these bugs hard to find. When output
was wrong, the model explained why it was right instead of investigating it. It wrote tests
that checked the code against itself. It drifted from the partial differential equation (PDE)
solver toward the Green's function (GF) approximation. The rules in the next section target
those habits.

## How to work on this code

Follow these rules in every session:

- **Treat surprising output as a bug until proven otherwise.** If a result disagrees with an
  analytic limit, a published value, or CosmoTherm, investigate it. Do not write an
  explanation for why the result is correct. If you cannot find the cause, say so.
- **Work on the PDE solver.** It is the product. The GF (`src/greens.rs`,
  `python/spectroxide/greens.py`) is a fast approximation used for cross-checks and tables.
  Do not move work to the GF unless the person you work with asks for it.
- **Propose a plan before you change physics or numerics.** State what you will change, why,
  and which test will show that the change is correct. Wait for approval.
- **Fix root causes.** Do not filter, mask, or clip output, and do not special-case a source
  term, to make a comparison pass. If the solver cannot handle a source term, fix the solver.
- **Never weaken a test to make it pass.** Do not relax a tolerance, narrow a comparison range,
  add `#[ignore]`, or delete a case without finding out why it fails and saying so.
- **Run the numbers.** After any physics change, run the solver and compare with reference
  data before you claim the change does or does not affect results. "This should not matter
  because..." is not evidence.
- **Read the architecture decision records (ADRs) first.** Before you change a numerical
  method, a default, a data format, or a public API, read `decisions/` for a record on that
  subject. Do not reopen an accepted decision silently. If you think one is wrong, say so and
  propose a new ADR that supersedes it.

## Build and test

Run all Rust tests in release mode. Some physics tests take minutes even there, and debug mode
makes them unusable.

```bash
cargo build --release
cargo test --release                      # full suite, including doc tests
cargo test --release TEST_NAME            # one test, or every test whose name contains TEST_NAME
cargo clippy --all-targets -- -D warnings # CI treats warnings as errors
cargo fmt --check
```

Replace `TEST_NAME` with a test name or part of one.

The experimental `axion` feature is off by default. Both configurations must build, pass their
tests, and pass Clippy. Check the feature build with these commands:

```bash
cargo test --release --features axion --lib --test dark_sector
cargo clippy --all-targets --features axion -- -D warnings
```

If you change the `get_unchecked` loops in `src/kompaneets.rs`, also run the tests with the
`debug_assert!` checks on. A normal release build strips them:

```bash
CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true cargo test --release --lib
```

The Python package calls the Rust binary. Install it and run its checks from `python/`:

```bash
cd python && pip install -e ".[plot]"
pytest tests/
black --check spectroxide/                # CI pins black==26.5.1
```

Continuous integration (CI) runs these checks, except that it runs the debug-assertion build
only on the kernel tests (`miri_kernel` and `thomas_solve`). It also runs a Miri check of the
unsafe kernel, a Rust–Python parity check, and the Sphinx documentation build
(`make -C docs html`). CI uses Rust 1.98.0. `Cargo.toml` declares 1.85 as the minimum
version, but CI does not test it.

## Architecture in brief

Each physical process has its own module. Keep processes in their own files.

`src/kompaneets.rs`
: Compton scattering (Kompaneets equation). Crank–Nicolson with Newton iteration. The most
  delicate module.

`src/double_compton.rs`, `src/bremsstrahlung.rs`
: Photon-number-changing emission and absorption.

`src/electron_temp.rs`
: Quasi-stationary electron temperature, computed perturbatively.

`src/recombination.rs`
: Ionization history. Below z ≈ 1575 the PDE solver evolves X_H together with T_e (ADR 0009).

`src/solver.rs`
: The integrator, `ThermalizationSolver`, built with `ThermalizationSolver::builder`. It couples
  Kompaneets with DC and BR in one Newton iteration and chooses the redshift step adaptively.

`src/energy_injection.rs`
: Every injection scenario, as a variant of `InjectionScenario`.

`src/distortion.rs`
: Decomposition of Δn into μ, y, and ΔT/T.

`src/greens.rs`
: The GF approximation (Chluba 2013).

`src/cli.rs`, `src/main.rs`
: The command-line interface (CLI): `solve`, `sweep`, `photon-sweep`, `photon-sweep-batch`,
  `greens`, `info`, `physics-hash`, and `help`.

`python/spectroxide/`
: `solver.py` runs the Rust binary. `greens.py`, `cosmology.py`, and `dark_photon.py` are
  pure-Python mirrors of the Rust modules of the same names.

`tests/`
: Rust integration tests. Shared setup, including `memo`, which runs each PDE configuration
  once per test binary, lives in `tests/common/mod.rs`. `CLAUDE.md` describes each file.

Production Rust code has no dependencies outside the standard library. Do not add a crate. The
only dev-dependencies are `approx` and `criterion`.

## Add an energy injection scenario

A new injection scenario is the most common contribution. First check whether you need one. A
heating history given as a function or a table needs no new variant: in Python, pass `dq_dz=`
or `photon_source=` to `spectroxide.solve`; in Rust, use `InjectionScenario::Custom`,
`TabulatedHeating`, or `TabulatedPhotonSource`. Add a variant only for a physical model with
its own parameters.

To add a scenario, follow these steps:

1. Add the variant to `InjectionScenario` in `src/energy_injection.rs`. Document the scenario,
   its source paper, and the meaning and units of each field.
2. Add a match arm to `name`, `validate`, and `heating_rate`. These matches have no wildcard,
   so the compiler lists every one you miss. `validate` must reject non-finite and unphysical
   values with an `Err` that names the parameter.
3. Decide which matches with a wildcard arm your scenario needs. They silently fall back to a
   default if you forget them:
   - Photon injection: add the variant to `has_photon_source`, add an arm to
     `photon_source_rate_with_step_factor` (and to `photon_source_step_factor` if the source
     has a factor that depends only on z), and write an explicit `0.0` arm in `heating_rate`.
     Add the injection frequency to the grid-coverage warning in
     `ThermalizationSolver::set_injection` (`src/solver.rs`).
   - A narrow spectral feature: `refinement_zones` and `suggested_x_min`.
   - A Gaussian time profile: `gaussian_burst`, which `characteristic_redshift` uses and the
     CLI uses to pick `z_start` (ADR 0001).
   - An impulsive change to the spectrum, as for the dark photon: `is_impulsive_resonance`,
     `initial_delta_n`, and a parameter method such as `dark_photon_params`, which
     `resonance_params` calls.
   - Heat whose integral is known in closed form: the private `integrate_heat_between`, which
     `injected_delta_rho_between` calls for the solver's energy-closure check.
   - Warnings for untested regimes: `warn_strong_distortion` and its neighbors.
4. Wire the scenario into the CLI in `src/cli.rs`. Add its flags to `injection_param_keys`
   (and to `BOOL_FLAGS` for a flag that takes no value), construct it in
   `build_injection_scenario`, add its name to the "Unknown injection type" error there,
   document it in the `solve --help` text, and add its name to the tests
   `test_known_flags_accepted` and `test_build_injection_all_types`. New scenarios run through
   `solve <injection-type>`; `sweep` handles single bursts only.
5. Add Python support in `python/spectroxide/solver.py`. Map each new parameter name to its
   CLI flag in `_INJECTION_PARAM_MAP`, list the type in the `injection` documentation of
   `solve`, and give it a `z_start` default in `_run_pde_single_solve` if it needs one. For a
   photon scenario, add the type to `photon_types` in
   `_validation.warn_grid_resolution_photon`.
6. Write tests with independent targets, as the next section describes. Heat scenarios go in
   `tests/pde_heat.rs`, photon scenarios in `tests/pde_photon.rs`.
7. Document the scenario in `README.md`, `docs/cli.rst`, and `docs/api/solver.rst`. A
   tutorial notebook in `notebooks/tutorials/` is welcome.

If the scenario introduces a new physical process rather than a new source, it needs its own
module and a discussion first. Open an issue before you write code.

## Write tests with independent targets

Every rule in this section exists because a bug in this project's history broke it.

### Derive every target outside the code

A target read off the code's output passes whether the code is right or wrong. Take each target
from an analytic formula, a published value, a dimensional argument, or an independent
calculation such as an mpmath quadrature. The test's doc comment must say where the number
comes from and why the tolerance has the value it has.

The following test sketch shows the pattern. It follows
`test_decaying_particle_y_era_energy` in `tests/pde_heat.rs`:

```rust
/// Energy delivered by MY_SCENARIO in the y-era against the injected total.
///
/// Target: the injected Δρ/ρ, integrated from the heating rate by `injected_drho`.
/// Tolerance: 2%, the measured closure of this configuration plus the time-step error
/// for continuous heating (cite the record that measured it).
#[test]
fn test_my_scenario_y_era_energy() {
    let cosmo = Cosmology::default();
    let scenario = InjectionScenario::MyScenario { /* ... */ };
    let inj = injected_drho(&scenario, &cosmo, 1e3, 3e6);
    let mut solver = ThermalizationSolver::new(cosmo.clone(), fast_grid());
    solver.set_injection(scenario).unwrap();
    solver.set_config(SolverConfig { z_start: 3.0e6, z_end: 1.0e3, ..SolverConfig::default() });
    solver.run_with_snapshots(&[1.0e3]);
    let last = solver.snapshots.last().unwrap();
    let e_rel = (last.delta_rho_over_rho - inj).abs() / inj;
    assert!(e_rel < 0.02, "energy closure {:.2}%", 100.0 * e_rel);
}
```

In the sketch, replace `MY_SCENARIO` and `MyScenario` with your scenario. The target never
comes from a previous run of the solver.

### Put the test where it is sensitive

A tight tolerance on an insensitive observable constrains nothing. Before you set a tolerance,
estimate ∂ln(observable)/∂ln(parameter) for the parameter the test should pin, and place the
test where it is of order one. For example, a GF comparison at z_h = 10⁶ moves by only 4% when
the DC coefficient is off by 53%, because the thermalization depth there is small. At
z_h = 3×10⁶ the same test fails. Exponential and threshold regimes, such as the
thermalization tail and freeze-out, are where tests have power.

### Check dimensions first

Before you trust a number, check that every rate coefficient has the right dimensions. Rates
are normalized per Thomson time, 1/(n_e σ_T c):

- A two-body process, such as BR (electron and ion), keeps one density factor, so K_BR is
  (ion density) × (length³), which is dimensionless.
- A one-body process, such as DC (photon and electron), cancels all density factors.

Type physical constants into rate tests literally from CODATA values instead of importing them
from `src/constants.rs`, as `tests/rate_coefficients_first_principles.rs` does.

### Use the known limits

Use these targets, and state which one a test uses:

- Deep μ-era (3×10⁵ ≲ z_h ≲ 5×10⁵ for a burst): μ ≈ 1.401 × Δρ/ρ. At higher z_h, multiply
  by the thermalization visibility J_bb*, which is about 0.81 at z_h = 10⁶, so μ ≈ 1.14 Δρ/ρ
  there.
- y-era: y = Δρ/(4ρ), but only while Comptonization of the distortion is negligible. Near
  z_h ≈ 10⁴ a fitted y already differs from Δρ/(4ρ) by a few percent, with a sign that depends
  on the fit weighting. For energy tests, compare the spectrum's Δρ/ρ (`delta_rho_over_rho`)
  with the injected energy instead of using 4y.
- Energy: the Δρ/ρ of the final spectrum equals the injected Δρ/ρ. The deficit is a
  time-step error: about 0.1% to 0.5% for bursts and about 1% for continuous heating at the
  default step size. Quote the closure your configuration achieves.
- Photon number: pure Compton scattering conserves ∫x²Δn dx.
- Photon injection: μ(x_inj) against Chluba (2015), and the μ-photosphere x_c(z), Chluba
  (2015) Eq. 25.

### Compare codes through the same estimator

μ and y are fitted numbers, and the fit changes them. Since ADR 0006 the default decomposition
weights the fit by intensity. Tests against visibility-function targets use
`decompose_number_conserving`. When you compare with another code, pass both spectra through
the same decomposition.

For PDE–GF comparisons between the μ-era and the y-era, use the CosmoTherm GF database, not
the Chluba (2013) fitting formulas. For decays with lifetimes at z = 2×10⁵ and 5×10⁵, the PDE
matched the database to 0.6% in μ, while the fitting formulas were off by several percent in μ
and by a factor of several in y. The test that checks this,
`test_decaying_particle_vs_cosmotherm_gf_database` in `tests/cosmotherm_comparison.rs`, is
ignored by default because it needs the database file. Elsewhere, the PDE and the GF agree to
2% to 5% in μ and about 5% in y.

### Know what each kind of test can prove

The method of manufactured solutions (MMS) and Richardson convergence tests verify the
discretization, not the equation. The manufactured residual uses the code's own operator, so a
wrong coefficient cancels and the test still reports second-order convergence. A term whose
only strong test is MMS is unverified physics. Pin the formulation separately, for example with
the exact moment hierarchy in `tests/kompaneets_moments.rs`. `dev/audit/term_coverage_matrix.md`
lists which terms still lack such an anchor.

### Audit the content of source terms from other codes

If you feed the solver a table from another code, such as a heating history from CLASS, check
which physical processes the table contains. Anything the PDE already models, above all
adiabatic cooling, is counted twice. Matching sign, units, and variable is not enough. See
`dev/audit/class_sd_comparison.md`.

## Numerical pitfalls

`CLAUDE.md` lists twelve numerical pitfalls with their algebra; it is the canonical version.
Each of the following summaries names the pitfall and the rule:

1. **Kompaneets cancellation.** Use the Planck identity dn_pl/dx + n_pl(1 + n_pl) = 0
   analytically. A finite-difference error of about 0.003 is 1000 times the physical signal.
   The code defines φ = T_z/T_e = 1/ρ_e, the reverse of the intuitive ratio.
2. **CFL instability.** Explicit Kompaneets steps are unstable at low x; the step must be
   implicit.
3. **DC and BR at low x.** The emission rate grows as 1/x³. Use backward Euler; Crank–Nicolson
   gives negative diagonals.
4. **Electron-temperature cancellation.** The full integral ratio has a numerical error of 0.1%,
   which swamps the 10⁻⁵ correction. Compute the correction perturbatively from Δn.
5. **DC and BR source cancellation.** n_pl(x/ρ_e) − n_pl(x) subtracts nearly equal numbers. Use
   the series expansion when |ρ_e − 1| < 0.01.
6. **Hidden NaN.** `f64::max(NaN, x)` returns x. Filter with `is_finite()` before a fold.
7. **Grid extent.** x_max must be at least 30 for accurate energy integrals.
8. **Dimensions.** Check every rate coefficient's dimensions before you trust its output.
9. **Self-calibrated tests.** A target taken from the code cannot catch a systematic error.
10. **Unsafe indexing.** The hot loops in `kompaneets.rs` use `get_unchecked`, guarded by
    `assert!` checks of slice lengths at function entry. If you add a workspace array or change
    the grid size, update the asserts.
11. **MMS verifies the scheme, not the equation.** See the previous section.
12. **Source terms from other codes.** Audit their physical content. See the previous section.

Energy routing adds one more rule. Energy that DC and BR absorb from injected photons must heat
the electrons and return to the photons through Compton scattering. Do not route photon
injection through `heating_rate`, and do not restore absorbed energy as a G_bb correction.

## Conventions

Dimensionless variables:

x
: hν/(kT_z), the frequency, with T_z = T_CMB(1 + z).

θ_e, θ_z
: kT_e/(m_e c²) and kT_z/(m_e c²); θ_z ≈ 4.60×10⁻¹⁰ (1 + z).

Δn
: n − n_pl, the distortion, which the PDE solves for.

ρ_e
: T_e/T_z.

Heating rates are d(Δρ/ρ)/dt in 1/s. `heating_rate_per_redshift` returns d(Δρ/ρ)/dz, which is
negative for heating; the GF routines expect a positive value.

The default cosmology (`Cosmology::default()`) follows Chluba (2013) on purpose, so the solver
can be checked against published CosmoTherm results: h = 0.71, Ω_b = 0.044, Ω_m = 0.26,
Y_p = 0.24, T_CMB = 2.726 K, N_eff = 3.046. CosmoTherm's GF code hard-codes N_eff = 3.04, a
negligible difference. The Rust presets are `planck2015`, `planck2015_cosmotherm`
(T_CMB = 2.726 K, for CosmoTherm comparisons), and `planck2018`. The CLI accepts `default`,
`planck2015`, and `planck2018`; Python exposes the CosmoTherm variant as `PLANCK2015_COSMO`.

## Keep Rust and Python in step

Several pure-Python modules mirror Rust ones. If you change a mirrored formula, change both
sides, then regenerate the parity fixture:

```bash
cargo run --release --example generate_parity_fixtures -- python/tests/data/parity_fixtures.json
```

CI fails if the committed fixture differs from fresh Rust output.

The binary embeds a hash of its physics source files (`build.rs`), and Python warns when the
binary is older than the source. After a physics change, rebuild with
`cargo build --release` before you run Python.

If you add or remove a field of a Rust struct, search `src/`, `tests/`, `examples/`, and
`benches/` for every literal construction of that struct and update each one.

## Documentation and commits

Documentation follows the Google developer documentation style guide, with the exceptions
listed in `CONTRIBUTING.md`. Rust doc comments start with a third-person verb ("Returns"),
state the units of every physical quantity, and document every `Err`, `None`, and panic.
Python docstrings use the NumPy format with an imperative summary. Check your changes with
the style checker:

```bash
python dev/scripts/docs_style_lint.py
```

Commit messages use a short `area: subject` line, such as `solver: ...` or `docs: ...`, and
name the ADR if one motivated the change. Add `[skip ci]` to commits that touch only
documentation, the paper, or other files that are not code.

## Before you hand the work back

Check each item:

- [ ] `cargo test --release`, both Clippy configurations, and `cargo fmt --check` pass.
- [ ] If you changed Python, `pytest tests/` and `black --check spectroxide/` pass.
- [ ] Every new test says where its target comes from and why its tolerance has that value.
- [ ] Every new rate coefficient has a dimensional check in a comment or test.
- [ ] A new scenario conserves energy to its measured closure.
- [ ] You ran the solver after any physics change and compared the output with a reference.
- [ ] You reported every failure, skipped step, and unexplained discrepancy.

## References

- Chluba & Sunyaev (2012), MNRAS 419, 1294: CosmoTherm and the primary reference for the
  equations.
- Chluba (2013), MNRAS 434, 352: the GF formalism and the visibility fits.
- Chluba (2015), MNRAS 454, 4182 (arXiv:1506.06582): photon injection and the
  μ-photosphere.
- Chluba, Ravenni & Bolliet (2020), MNRAS 492, 177: exact BR Gaunt factors (BRpack).
- Draine (2011), *Physics of the Interstellar and Intergalactic Medium*, Ch. 10: the
  softplus Gaunt-factor interpolation that `bremsstrahlung.rs` uses.
