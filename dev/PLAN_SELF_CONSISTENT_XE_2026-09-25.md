# Plan: evolve X_e with the three-level atom, coupled to T_e (2026-09-25)

Status: plan only, nothing implemented. ADR 0009 must be drafted and accepted
before step 2.

## What is wrong now

- `RecombinationHistory::new` integrates the Peebles three-level atom (TLA) once,
  before the run, with α_B evaluated at T_rad = T_z (`src/recombination.rs`,
  `peebles_rhs`). X_e(z) is then a fixed table.
- The solver already evolves T_e with the RECFAST matter-temperature equation
  plus the distortion terms (`update_temperatures`, `src/solver.rs`):
  dρ_e/dτ = R[(ρ_eq + δρ_inj − H_dcbr) − ρ_e] − H t_C ρ_e.
  It reads X_e from the fixed table, but X_e never sees T_e.
- Consequence: below z_switch ≈ 1575, a run that heats the gas (ρ_e > 1)
  recombines as if it were cold. Hot gas has a smaller α_B (α_B ∝ T^-0.62 at
  low T), so the true X_e stays higher, t_C shorter, and the gas hands its heat
  to the photons faster. ADR 0007 already warns at ρ_e > 10 for z < 1500
  instead of modeling this.
- For runs with no injection the error is small: T_m tracks T_γ to about 1% in
  the recombination era. The fix matters for heating below z ≈ 1600.

## Scope

In: hydrogen X_H from the Peebles TLA with α_B(T_m), T_m = ρ_e T_z, advanced
alongside ρ_e inside the PDE run.

Out (per the request, "just the TLA"): helium stays Saha at T_z; no
collisional ionization; no recombination cooling or photoionization heating in
the T_e equation; no direct ionization by injected energy; the Green's
function, `ionization_fraction`, dark-photon z_res (ADR 0008), and the Python
port keep the fixed standard history.

Known limitation to state in the ADR: without collisional ionization the TLA
under-ionizes gas at ρ_e ≳ 10 near z ≈ 850 (T ≈ 2e4 K). The ADR 0007 hot-gas
warning stays, reworded to name collisional ionization, not the fixed history.
Also check the temperature range over which the Péquignot et al. (1991) α_B fit
is valid; narrow bursts reach T_e ~ 1e6 K (ADR 0007).

## Design

### Recombination module (`src/recombination.rs`)

1. `alpha_recomb` unchanged. `peebles_rhs` and `peebles_step` gain `rho_m`
   (T_m / T_rad). β_B and X_S stay at T_rad (photoionization by the CMB).
2. Keep the Saha-subtracted form and add the temperature correction as a
   separate term, so ρ_m = 1 reproduces the current table bit for bit:
   dX_H/dz_up = C n_H / [H(1+z)] ·
     { α_B(T_rad) [X_H² − X_S²(1−X_H)/(1−X_S)] + [α_B(T_m) − α_B(T_rad)] X_H² }.
   Evaluate C with the same T_rad β_B. The Δα term is identically 0.0 at ρ_m = 1.
3. New `pub fn advance_x_h(z_from, z_to, x_h, rho_m, cosmo) -> f64`: sub-cycles
   `peebles_step` with dz_sub ≤ 0.5 (the step the table already uses stably).
   Above z_switch it returns Saha at T_rad (hydrogen is ionized; T_m does not
   matter there).

### Solver (`src/solver.rs`)

4. State: `x_h: f64` on `ThermalizationSolver`. Initialize at `z_start` from the
   existing table (`recomb.x_e(z_start) − helium`), which assumes ρ_e = 1 before
   the run starts. Reset it in `reset()`.
5. Coupling: lagged (Lie) split, first order in the X_e–T_e coupling.
   At the start of `step_with_dz`, advance a *local* copy of X_H from z to z_mid
   and on to z_new with ρ_e held at the start-of-step value. Use X_e(z_mid) for
   t_C, Δτ, R, n_e, and BR in that step. Commit `self.x_h` only after the
   finite-Δn check passes, so a failed step leaves the state untouched.
   Justification: X_e changes on a Hubble time; one step moves z by at most
   5% (`adaptive_dz` cap). Step 7 below measures the splitting error; if it is
   not negligible, add one corrector pass with ρ_e averaged over the step.
6. `x_e_at(z)` is used by `adaptive_dz` and `update_temperatures`. Replace
   both uses with the state value. Keep the table for initialization and for
   the fixed-history mode.
7. Config: `SolverConfig::fixed_ionization_history: bool`, default false, plus
   a CLI flag `--fixed-ionization` next to `--no-dcbr`. The default change goes
   in ADR 0009. The flag gives an A/B comparison and reproduces pre-change runs.
8. Output: add `x_e` to `SolverSnapshot` and its JSON in `output.rs`. Grep and
   update every struct literal (tests, examples, `generate_parity_fixtures.rs`).
   Python reads the field if present.
9. Reword `HOT_GAS_RHO_E` / `HOT_GAS_Z_MAX` docs and the warning text.

Build both feature configurations (`--features axion` and default): step 5
touches `step_with_dz`, which the axion scenario uses.

## Verification

Targets come from outside the code (CLAUDE.md pitfall #9).

V1. Identity: `advance_x_h` with ρ_m = 1 matches `RecombinationHistory` to
    1e-15 over z = 1575 to 1 (unit test).
V2. Invariance: any run with z_end > z_switch gives bit-identical Δn, μ, y
    (compare the full test suite and a sweep before and after).
V3. Null run (no injection) to z = 200: X_e(z) against RECFAST milestones in
    `test_xe_vs_recfast_milestones`. It must not get worse; the low-z residual
    should shrink because T_m < T_γ below z ≈ 500 is now included.
V4. Heated run: independent Python integration of the coupled (X_H, T_m) ODEs
    with the same q(z), written from Peebles (1968) and Seager et al. (1999),
    importing nothing from spectroxide. Compare X_e(z) and T_e(z) for a burst
    at z_h = 1000 and z_h = 800. This extends
    `dev/scripts/heatloss/heatloss_common.py`, which today takes X_e from the
    solver's table.
V5. Heat delivery: `tests/heat_delivery.rs` targets (0.999862, 0.99417) were
    computed with the fixed X_e. Recompute them with the V4 oracle and update
    the tests with the provenance in the comments. Expect delivery to rise.
V6. Splitting error: halve `dtau_max` and `dz` cap; X_e(z_end) and the
    delivered fraction must converge at first order or better.
V7. Physics-inquisitor on the recombination diff; production-code-reviewer on
    the full diff; `cargo clippy -- -D warnings` in both feature configurations;
    `cargo test --release`; `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true cargo
    test --release --lib`.

## Expected fallout

- Tests with z_end < 1575 may move: `heat_delivery.rs` (certain),
  `pde_photon.rs` post-recombination cases, `cosmotherm_comparison.rs` if its
  runs end below z_switch, golden references in `pde_heat.rs`. Every changed
  target needs a reason from V3 to V5, not a re-read of the new output.
- Paper text that says X_e comes from a fixed history, and any figure with
  heating below z ≈ 1600 (Fig. 3 low-z, pathological heating), need a check.
- CLAUDE.md: update the `recombination.rs` and `electron_temp.rs` bullets and
  the `heat_delivery.rs` targets.

## Order of work

1. Draft ADR 0009 (context above, alternatives: keep the fixed table and warn,
   the lagged split chosen here, a fully implicit joint (X_H, ρ_e, Δn) Newton
   solve, a full multilevel code such as HyRec). EB accepts or rejects.
2. Recombination module changes plus V1.
3. Solver state, coupling, config flag, snapshot field. V2.
4. V3, V4 oracle, V6.
5. V5 test retargets, fallout sweep, docs.
6. V7 reviews, then commit.

## Status (2026-09-25, end of session)

Implemented, uncommitted. All verification steps done except the final
claim-verifier pass on the step cap (running at time of writing). Full
release suite: 468 pass, 0 fail, 5 ignored; clippy clean in both feature
configurations; debug-assertion lib run and Python suite pass (before the
step cap; rerun before commit).

Files: `src/recombination.rs`, `src/solver.rs` (X_H state, corrector,
`XE_COUPLED_DZ_FRAC` = 0.005 step cap in coupled mode below z_switch,
`fixed_ionization`, `SolverSnapshot::x_e`), `src/cli.rs`, `src/output.rs`,
`python/spectroxide/solver.py`, `docs/cli.rst`, `tests/ionization_coupling.rs`
(5 tests), `tests/heat_delivery.rs` (burst target 0.999855),
`tests/coverage_gaps.rs` (hot-gas case moved to 1e-3), `tests/gf_visibility.rs`,
`dev/scripts/heatloss/coupled_xe_expectation.py` (oracle, with `--table-xe`),
ADR 0009 + addendum, ADR 0007 back-link, CLAUDE.md.

Measured numbers: see the ADR 0009 addendum. Plan corrections: V5 targets did
not leave tolerance; the main numerical error was the first-order T_e step
at the old 0.05 z cap, not the split, fixed by the step cap.

Open for EB: commit; paper text on the ionization history; collisional
ionization (out of scope); α_B fit range (unresolved).

## DarkHistory comparison (2026-09-25, later session)

`dev/notebooks/xe_darkhistory_comparison.ipynb` (+ `examples/xe_history.rs`, output
`dev/data/xe_darkhistory_comparison.json`, figure `notebooks/figures/xe_darkhistory_comparison.pdf`)
compares against DarkHistory 1.1.2's TLA with background, binding energy and Lyα escape matched.
DarkHistory shares the α_B fit and F = 1.125, so this tests the coupling, not the atomic physics.
Results: default steps agree to 0.7% (X_e) and 1.2% (ρ_e) for T_e < 5.4e3 K; the difference
halves with every halving of the step (Δz = 2 to 0.125) and extrapolates to 2e-5, i.e. it is
the first-order T_e step error at the 0.005 z cap. Paper: new Appendix `app:xe` in
~/cosmoxide/paper/paper.tex (revision markup), forward reference in Sec. 2, pointer in
app:stepping, step-count sentence fixed.

Open: `Liu:2019bbm` missing from refs.bib (in Zotero, "Dark Photons" collection only);
\rtodo citations for Péquignot 1991, the F = 1.125 source, HyRec-2, collisional rates;
performance table (2026-09-24) predates the 0.005 z cap (y-era burst CLI 0.06 s -> 0.30 s,
182 -> 1009 steps) and needs a re-benchmark; the hot-gas warning (ρ_e > 10) fires well above
the collisional threshold (T_e ~ 1e4 K), consider an absolute-T_e guard.
