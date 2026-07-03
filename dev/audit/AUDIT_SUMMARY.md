# Validation Audit — Findings Log

Running log for the correctness audit (dev/PLAN_VALIDATION_AUDIT_2026-07-02.md,
Part B). Protocol per B5: every confirmed bug is fixed, its impact on published
figures quantified numerically (no theoretical dismissals), and recorded here.

Finding IDs: `P<phase>-<n>`. Status: fixed / open / documented.

---

## Phase 0 (2026-07-02): parity harness + known-bug fixes

Infrastructure delivered in this phase:

- **Rust↔Python parity harness** (`examples/generate_parity_fixtures.rs` →
  `python/tests/data/parity_fixtures.json` → `python/tests/test_parity.py`;
  CI job `parity` in `.github/workflows/ci.yml` regenerates the fixture from
  current Rust and re-runs the comparison, so drift on either side fails CI).
  40 groups / 957 evaluation points covering: visibility functions, heat and
  photon Green's functions, critical frequencies, analytic + numerical P_s,
  spectral shapes, background cosmology, recombination history, Compton y(z),
  and all dark-photon NWA helpers, for both the default and Planck-2018
  cosmologies. Per-group tolerances are declared in the generator with a
  written justification (`note` field); everything closed-form is held to
  1e-11, table/quadrature-mediated groups to 1e-5–5e-3 with the cause stated.
- **Assertion inventory** (`dev/scripts/extract_test_assertions.py` →
  `dev/audit/test_assertions.json`): 1343 numeric assertions in 619 tests,
  input to the B0 provenance census (`TEST_PROVENANCE.md`).

Findings P0-1 … P0-3 were discovered (P0-1 by the referee, P0-2/P0-3 by the
parity harness on its first run) as Rust↔Python divergences where the Rust
side had already been corrected in an earlier audit but the Python mirror had
not been updated — exactly the failure mode B2 exists to eliminate.

### P0-1 — Python numerical P_s used raw Saha ionization (fixed)

- **Where:** `python/spectroxide/greens.py::_photon_survival_probability_numerical`
- **Defect:** the τ_ff integrand used raw Saha hydrogen ionization at all z:
  exponential underestimate of X_e below z ≈ 1500 (no Peebles freeze-out) and
  no helium electrons for 1500 < z < 6000. Rust `tau_ff_survival` uses the
  full Peebles+He `ionization_fraction`. Referee-flagged (SciPost Report 1).
- **Fix:** Python now calls `ionization_fraction` on the same z grid,
  mirroring Rust. Parity: agrees with Rust to < 1e-5 on a 42-point
  (x_inj, z_h) grid.
- **Impact (old→new, default cosmology):** P_s shifts up to ~50% where
  P_s ≲ 0.1 (e.g. x=3e-3, z_h=5e4: 1.25e-4 → 6.4e-5) and by orders of
  magnitude in deep-absorption tails (x=1e-3, z_h=1e4: 1.16e-10 → 1.8e-11);
  ≲ 5% where P_s ≳ 0.5. Figure-level: see "Affected figures" below.

### P0-2 — Python He Saha used the superseded He-only quadratic (fixed)

- **Where:** `python/spectroxide/cosmology.py::_saha_he_i/_saha_he_ii`
- **Defect:** solved y²-type Saha with n_e = y·n_He. The Rust side was
  corrected in an earlier audit to the RECFAST/Seager+1999 total-free-electron
  linear form y/(1−y) = K(T)/n_e with n_e dominated by ionized H; the Python
  mirror kept the old form. X_e wrong by up to 4.5% at z ≈ 2000–6500
  (He recombination), with knock-on errors in P_s, ω_pl(z), and γ_con.
- **Fix:** ported the Rust form (incl. `_solve_saha_linear`).
- **Impact:** X_e max rel change 4.5% at z ≈ 2400; γ_con/ε² up to 9.4% at
  m ≈ 1.8e-9 eV (resonance in the He-recombination range); z_res up to 1%.

### P0-3 — Python Peebles step was forward Euler; Rust is Heun (fixed)

- **Where:** `python/spectroxide/cosmology.py::_peebles_step`
- **Defect:** Rust `peebles_step` was upgraded to trapezoidal/Heun (2nd
  order) in audit M1; Python kept forward Euler. 0.24% X_e offset at
  freeze-out (z ≲ 100), constant down to z = 1.
- **Fix:** ported the Heun step (`_peebles_rhs` + predictor-corrector).
  Residual Rust↔Python difference ≤ 7e-6, attributable to Python's dz = 0.5
  table interpolation vs Rust's direct-to-z integration (declared tolerance
  1e-5 in the fixture).

### P0-4 — Rust cosmic_time quadrature unconverged (fixed)

- **Where:** `src/cosmology.rs::cosmic_time` (and Python default `n_points`)
- **Defect:** 64 midpoint points in ln(1+z) over [z, 1e9] → up to 0.8%
  error at z = 10 (verified by n-refinement: 64→8192 changes the result by
  7.7e-3 at z=10). Python used 128 (0.2% error). Enters decaying-particle
  vacuum survival exp(−Γ_X t(z)).
- **Fix:** both sides now use 2048 points (< 1e-5 relative error at all z,
  identical algorithm; parity < 1e-11... declared 1e-4 in fixture for slack).

### P0-5 — Planck-2018 presets disagreed between Rust and Python (fixed)

- **Where:** `Cosmology::planck2018` (both languages)
- **Defect:** Python used the paper's Ω_m = 0.3153, which includes the
  Σm_ν = 0.06 eV neutrino contribution (ω_ν ≈ 0.00064); Rust used the
  paper's ω_cdm = 0.1200, which excludes it. Both cannot hold in a ν-less
  code: the presets differed by 0.5% in Ω_m (plus a 3e-5 rounding offset in
  Ω_b). `ec26_likelihood.py` had already discovered this and carried a local
  workaround dict (Ω_m = 0.31377) instead of fixing the preset.
- **Fix:** Python presets (2015 and 2018) now derive density fractions from
  the papers' physical densities ω_b, ω_cdm — the CMB-calibrated quantities
  that matter for distortion physics — matching Rust exactly. The ν-omission
  is documented in the docstring.

### P0-6 — planck2015 T_CMB convention differs Rust vs Python (open, documented)

- Rust `planck2015` uses T_CMB = 2.726 K (CosmoTherm DI-file convention);
  Python keeps the paper value 2.7255 K, with the CT convention available as
  `PLANCK2015_COSMO`. Not silently changed because Rust-side CosmoTherm
  comparison tests may depend on it. **Decision needed (EB):** unify on the
  paper value and route CT comparisons through an explicit CT preset on both
  sides. Not in the parity fixture until resolved.

### P0-7 — Python Compton y(z) ignored matter-temperature decoupling (fixed)

- **Where:** `python/spectroxide/greens.py::_y_compton`
- **Defect:** used θ_e ∝ T_γ at all z, where Rust applies T_m ∝ (1+z)²
  below z_dec = 200 (audit M1); additionally 32-point Gauss–Legendre
  under-resolved the recombination X_e drop (2.5% error in y at z = 1e3).
- **Fix:** mirrors the Rust 128-point midpoint rule with the T_m switch.
  Parity ≤ 2e-3 (declared; dominated by X_e table interpolation).

### Affected published figures (regeneration required before resubmission)

Quantified old→new on the figures' own ingredient functions (default
cosmology; script: scratchpad `figure_impact.py`, HEAD vs fixed package):

| Figure | Ingredient | Max change | Where | Median |
|---|---|---|---|---|
| `paper_figures/firas_photon_limits` | y from GF (least-squares, as in remake script) | 14% / 16% / 34% (z_h = 1e4 / 3e4 / 5e4) | x_inj ≈ 0.02–0.03 | 0.15–0.8% |
| `paper_figures/dark_photon_constraints` | γ_con/ε² | 9.4% | m ≈ 1.8e-9 eV | 0.19% |
| (same) | ε limit (∝ √γ_con) | 4.6% | same masses | ≲ 0.1% |
| `paper_figures/photon_injection_spectra` | G_ph amplitude | ≲ few % (soft x_inj) | low x_inj, y-era | ≲ 0.3% |

μ-era photon-injection results (`mu_from_photon_injection`) use the analytic
P_s and are unchanged. PDE-based curves (Rust binary) are unaffected by
P0-1/2/3/7 (Rust side already correct); PDE decaying-particle results move by
≤ 0.8% in t(z) via P0-4 — impact on μ(z_h) sweeps to be spot-checked when the
figures are regenerated (Phase 4 / B4).

---

## B0 test-provenance census (complete)

`dev/audit/TEST_PROVENANCE.md` classifies all 624 tests carrying numeric
assertions (1343 assertions; inventory in `test_assertions.json`, raw
classification fragments in `census/`):

| Class | Tests |
|---|---|
| analytic | 288 |
| literature | 61 |
| cross-method | 49 |
| dimensional | 107 |
| regression-pin (all explicitly labelled) | 4 |
| structural (no physics target) | 115 |

One flagged gap: `tests/heat_injection.rs::test_photon_injection_spectral_decomposition_residual`
asserts a 12% residual bound while its docstring claims "< 5%", with no stated
origin for the 12% — to be re-derived or relabelled in Phase 1. Two
tautological assertions were also noted (they can never fail):
`test_greens.py::test_gf_linearity` and the energy-conservation check inside
`test_pde_negative_injection_produces_negative_distortion` — to be replaced
with meaningful versions in Phase 1.

## Test status after Phase 0 fixes

- Python: 311 existing tests + 41 parity tests pass.
- Rust: all `cargo test --release` suites pass (lib 212, adversarial 17,
  cli 4, cosmotherm 7, coverage_gaps 14, greens_function_checks 7,
  heat_injection 198, science_suite 5, convergence_order 8+1 ignored,
  fh/CLASS suites, doc tests); clippy clean with `-D warnings`.
