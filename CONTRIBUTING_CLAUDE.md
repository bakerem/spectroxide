# LLM Context File for spectroxide Contributors

This file is designed to be included as context when using an LLM (Claude, GPT, etc.) to develop features for spectroxide. Copy its contents into your LLM's system prompt or project instructions before starting work.

> **Why this file exists.** spectroxide is a numerical physics code where plausible-looking implementations can be silently wrong. During development, four serious physics bugs passed a large automated test suite because the LLM wrote tests calibrated to its own (incorrect) output. This file encodes the hard-won lessons so you don't repeat them. See Sec. 6 of the paper (Baker, Liu & Mishra-Sharma, 2026) for the full story.

> **Canonical sources.** `CLAUDE.md` (loaded automatically by Claude Code) is the authoritative reference for architecture, numerical pitfalls, validation targets, and working-style rules. On any conflict, defer to `CLAUDE.md`. This file is a workflow-focused complement, not a replacement.

---

## Project overview

spectroxide is a Rust PDE solver for CMB spectral distortions. It evolves the photon occupation number n(x, z) through the coupled Kompaneets (Compton scattering), double Compton, and bremsstrahlung equations from z ~ 10^6 to z ~ 100. It computes mu-, y-, and intermediate-type distortions from arbitrary energy injection histories. A fast Green's function mode is also available.

**Key constraint:** Zero *production* Rust dependencies (pure std). Dev-dependencies are limited to `approx` (test float comparison) and `criterion` (benchmarks). Do not add new production crates.

## Build and test

```bash
cargo build --release
cargo test --release                      # Always use --release; some tests are slow in debug
cargo test test_name                      # Run a single test
cargo clippy --all-targets -- -D warnings # CI rejects warnings
cargo fmt --check                         # CI rejects unformatted code
```

Python package:
```bash
cd python && pip install -e ".[plot]"
```

## Architecture (what you need to know)

- **Physics modules** (`src/`): Each physical process has its own file — `kompaneets.rs`, `double_compton.rs`, `bremsstrahlung.rs`, `electron_temp.rs`, `recombination.rs`. Do not mix processes across files.
- **Solver** (`src/solver.rs`): IMEX integrator coupling all processes. Crank-Nicolson for Kompaneets, backward Euler for DC/BR, Newton iteration.
- **Energy injection** (`src/energy_injection.rs`): All injection scenarios live here as variants of `InjectionScenario`. New scenarios go here.
- **Grid** (`src/grid.rs`): Non-uniform frequency grid with optional `RefinementZone` for adaptive resolution near spectral features.
- **Distortion decomposition** (`src/distortion.rs`): Extracts (mu, y, DeltaT/T) from the solved spectrum.
- **Green's function** (`src/greens.rs`): Fast approximate mode. Visibility functions J_bb*, J_mu, J_y.
- **Python** (`python/spectroxide/`): Wraps Rust binary + pure-Python Green's function.
- **Tests** (`tests/`): 430+ Rust tests across 8 files (`heat_injection.rs` is the main integration file; others cover adversarial inputs, coverage gaps, CosmoTherm comparison, Green's-function checks, convergence order, CLI integration, and the science suite). A separate Python test suite lives under `python/tests/`.

## How to add a new energy injection scenario

This is the most common contribution. Follow these steps exactly:

### Step 1: Add the variant to `InjectionScenario` in `src/energy_injection.rs`

```rust
/// Your scenario description with references.
MyScenario {
    /// Parameter description with units
    param1: f64,
    /// ...
    param2: f64,
},
```

### Step 2: Implement required methods

You must add match arms to ALL of these methods on `InjectionScenario`:

- `name(&self)` — string identifier for CLI/output
- `validate(&self)` — parameter validation, return `Err(String)` for invalid inputs
- `heating_rate(&self, z, cosmo)` — energy injection rate d(Deltarho/rho)/dt [1/s]. Return 0.0 if your scenario injects photons rather than heat.
- `photon_source_rate(&self, x, z, cosmo)` — frequency-dependent photon source dn/dt [1/s]. Return 0.0 for pure heat injection.
- `has_photon_source(&self)` — return true if `photon_source_rate` is non-zero
- `refinement_zones(&self)` — return `Vec<RefinementZone>` for grid adaptation near spectral features. Return empty vec if not needed.
- `depletion_rate(&self, x, z, cosmo)` — photon removal rate [1/s]. Return 0.0 unless your scenario removes photons.
- `heating_rate_per_redshift(&self, z, cosmo)` — rate per unit redshift (used by Green's function). Usually `heating_rate / (H(z) * (1+z))`.

### Step 3: Wire into CLI (`src/cli.rs`)

CLI subcommands are `solve`, `sweep`, `greens`, `photon-sweep`, `photon-sweep-batch`, `info`, `help` (entry point in `src/main.rs`, dispatch in `src/cli.rs`). New scenarios usually plug into `solve` and `sweep` via the existing `--scenario` / `--params` machinery; photon-source scenarios additionally go through the photon-sweep subcommands.

### Step 4: Write tests with INDEPENDENT targets

This is the most important step. See the next section.

### Step 5: Add Python support

Update `python/spectroxide/solver.py` to accept your scenario's parameters and pass them to the Rust binary.

## Critical rules for writing tests

These rules exist because every major bug in spectroxide's history was missed by tests that violated them.

### 1. NEVER calibrate test targets from the code itself

```rust
// BAD: Running the code, seeing it outputs 42.7, then writing:
assert!((result - 42.7).abs() < 0.1);

// GOOD: Deriving the target from physics:
// In the y-era (z < 1e4), y = Deltarho/(4*rho), so for drho=1e-4:
let expected_y = 1e-4 / 4.0;
assert!((result.y - expected_y).abs() / expected_y < 0.05);
```

### 2. Always check dimensions FIRST

Before trusting any numerical output, verify that every rate coefficient has the correct dimensions. Thomson time normalization (dividing by n_e * sigma_T * c) cancels:
- One density factor for two-body processes (BR: e + ion), leaving N_ion * lambda_e^3 (dimensionless)
- All density factors for one-body processes (DC: gamma + e), leaving a pure number

### 3. Use known analytic limits

- **Deep mu-era** (z > 2e5): mu/Deltarho = 1.401
- **y-era** (z < 1e4): y = Deltarho/(4*rho)
- **Energy conservation**: mu/1.401 + 4y + 4*DeltaT/T = Deltarho/rho
- **Photon conservation** under pure Compton: integral of x^2 * Deltan * dx = const

### 4. Cross-validate PDE against Green's function

For simple injection histories, the PDE and GF should agree within ~5%. Large discrepancies indicate a bug.

### 5. Never weaken a test to make it pass

If a test fails, investigate the root cause. Do not relax tolerances, restrict comparison ranges, or delete failing cases without understanding WHY they fail.

## Numerical pitfalls you MUST know about

`CLAUDE.md` enumerates **10 critical numerical pitfalls** in full, with the algebraic expansions and code-level requirements. That list is mandatory reading and is the canonical source — do not paraphrase it from memory. The summaries below are the workflow-relevant tips; consult `CLAUDE.md` for the rest (CFL/implicit Kompaneets, NaN-hiding via `f64::max`, x_max ≥ 30 for G₃, the BR dimensional-analysis war story, the unsafe-indexing/`assert!` invariants in `kompaneets.rs`, and the "tests calibrated to code output" failure mode).

1. **Kompaneets cancellation**: The flux must use the Planck identity dn_pl/dx + n_pl(1+n_pl) = 0 analytically. Finite-difference error (~0.003) is 1000x the physical signal (~1e-5).

2. **DC/BR divergence at low x**: Emission rate ~ 1/x^3. Must use backward Euler (not Crank-Nicolson, which gets negative diagonals and oscillates).

3. **T_e feedback cancellation**: Full integral ratio has 0.1% numerical error swamping the 1e-5 physical correction. Must use perturbative expansion from Deltan only.

4. **DC/BR source near-cancellation**: n_pl(x/rho_e) - n_pl(x) subtracts nearly-equal numbers. Use analytical expansion when |rho_e - 1| < 0.01.

5. **Energy routing**: Injected photon energy absorbed by DC/BR must flow through T_e -> Kompaneets -> mu/y. Do NOT shortcut this by adding G_bb corrections or routing through heating_rate(). The solver must handle arbitrary source terms without special-casing.

## Dimensionless variables

- x = h*nu/(k*T_z): frequency
- theta_e = k*T_e/(m_e*c^2): electron temperature
- theta_z = k*T_z/(m_e*c^2) ~ 4.6e-10 * (1+z)
- Deltan = n - n_pl: the distortion (what the PDE solves for)
- rho_e = T_e/T_z: electron-photon temperature ratio

## Cosmology defaults

h=0.71, Omega_b=0.044, Omega_m=0.26, Y_p=0.24, T_cmb=2.726, N_eff=3.046 (Chluba 2013 conventions). Planck 2015 and 2018 presets also available.

## What NOT to do

- **Do not add external crate dependencies.** The zero-dependency constraint is deliberate.
- **Do not route soft photons through `heating_rate()`.** This is a scenario-specific hack. Photon injection must go through `photon_source_rate()` -> DC/BR absorption -> T_e -> Kompaneets.
- **Do not use ad hoc numerical fixes.** If the solver can't handle a source term, fix the solver, not the source.
- **Do not compare only Green's function results.** Always validate with the full PDE solver.
- **Do not weaken or delete tests** to make the code pass. Fix the code.
- **Do not modify Rust struct fields without grepping for every literal instantiation** in `src/`, `tests/`, `examples/`, and `benches/` and updating each one. CI failures from missed call sites waste a review cycle.
- **Do not modify the `kompaneets.rs` Newton/Thomas hot loops without re-checking the `assert!` slice-length guards.** Those guards are the safety contract for `get_unchecked` indexing; if you add a workspace field or change grid sizing, update the asserts in lock-step and re-run `cargo test` (debug mode runs the `debug_assert!` checks that release strips).
- **Do not skip `cargo clippy --all-targets -- -D warnings` or `cargo fmt --check` before pushing.** CI runs both and treats warnings as errors.

## Useful references

- Chluba & Sunyaev (2012), MNRAS 419, 1294 — CosmoTherm paper, primary reference for equations
- Chluba (2013), MNRAS 436, 2232 — Green's function formalism
- Chluba (2015), arXiv:1506.06582 — Photon injection
- Chluba, Ravenni & Bolliet (2020), MNRAS 492, 177 — BR Gaunt factor (BRpack)

## Example: adding a simple scenario

Here's a minimal example of adding a constant heating rate scenario. Use this as a template.

```rust
// In energy_injection.rs, add to the enum:
/// Constant heating rate (for testing/pedagogical purposes).
ConstantHeating {
    /// d(Deltarho/rho)/dt [1/s]
    rate: f64,
},

// Then add match arms to ALL methods:
// name() -> "constant-heating"
// validate() -> check rate > 0 and rate.is_finite()
// heating_rate() -> self.rate
// photon_source_rate() -> 0.0
// has_photon_source() -> false
// refinement_zones() -> vec![]
// depletion_rate() -> 0.0
// heating_rate_per_redshift() -> rate / (cosmo.hubble(z) * (1.0 + z))

// Test with INDEPENDENT target:
// At z_obs = 5000 (deep y-era), injecting for time dt at rate R:
// y = integral of R/(4*H*(1+z)) dz, computed analytically
```
