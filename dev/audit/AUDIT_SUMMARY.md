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

---

## Phase 1 (2026-07-03): B1 module audits

One independent physics-inquisitor pass per module (fresh context each), per
the B1 protocol: primary-reference re-derivation before reading code output,
equation↔code mapping table, per-finding triage. Memos:
`dev/audit/<module>_audit.md`.

| Module | Memo | Confirmed production bugs | Notable findings |
|---|---|---|---|
| kompaneets.rs | kompaneets_audit.md | none | P1-1, P1-2 |
| electron_temp.rs (+full_te) | electron_temp_audit.md | none | P1-3 |
| dark_photon.rs/.py | dark_photon_audit.md | none | P1-4 |
| firas.py | firas_audit.md | none | **P1-5**, P1-6 |
| distortion.rs | distortion_audit.md | none | P1-7 |
| double_compton.rs + bremsstrahlung.rs | double_compton_bremsstrahlung_audit.md | none | P1-8 |
| greens.rs/.py | greens_audit.md | none | P1-9 |
| recombination.rs | recombination_audit.md | none | P1-10 |

Independence spot-check: the distortion audit brief deliberately quoted
`β_M ≈ 0.4561` (the reciprocal of the true β_μ = 3ζ(3)/G₁ = 2.19229); the
auditor derived the correct value from scratch and refuted the brief rather
than repeating it (distortion_audit.md §0, §4).

### P1-1 — kompaneets.rs stale docstring (FIXED 2026-07-03, doc-only)

Docstring near `kompaneets.rs:454` claims the ρ_e Newton update is "capped at
1e8"; the variable is literally `uncapped` (is_finite() guard only). Fix text.

### P1-2 — coverage gap: no in-module ΔN/N test for the coupled path (open)

Kompaneets flux form is conservative and the pure-Compton number-conservation
tests pass, but there is no unit-level photon-number ledger assertion on the
coupled Kompaneets+DC/BR Newton path. Feeds B3 (MMS candidate: the
flux-conservative operator was assessed as a good manufactured-solutions
target).

### P1-3 — test-validity bug: `test_equilibrium_for_bose_einstein` (FIXED 2026-07-03)

`electron_temp.rs:78-146`. For any Bose-Einstein spectrum, n(1+n) = −dn/dx
⇒ I₄ = 4G₃ identically ⇒ ρ_eq = 1 exactly, independent of μ. The test's
"μ>0 ⇒ ρ_e>1" and "larger μ ⇒ larger ρ_e" assertions pass only on a
positive O(dx²) discretization artifact (numerically confirmed:
ρ_eq−1 = 2.15e-6 for μ = 0, 1e-4, 1e-3, 5e-3 — identical, μ-independent,
→0 as O(dx²)). No production impact (`compton_equilibrium_ratio` is
off-path; the solver uses the perturbative Δρ_eq, which the audit verified
returns O(μ²) for BE inputs as it must). Fix: assert |ρ_e−1| < tol(N),
correct the docstring physics. Exactly the pitfall-#9 class the census
exists to catch.

### P1-4 — dark photon: formula verified; 22% cross-code thread narrowed

γ_con matches an independent Landau–Zener derivation of CCJ24 Eq. 6 exactly
(including the ε²m² numerator after the ω_pl²=m² cancellation and the kT_γ
normalisation, which is *not* a thermal average — blackbody weighting is
carried spectrally by P(x) in the IC). The unresolved ~22% γ_con offset vs
the reference figure (see memory/axion-dp work) is therefore not a defect in
our formula; remaining candidates are the reference's cosmology parameters or
d-factor evaluation for resonances near recombination, where d is
finite-difference-sensitive on X_e (only region where tens-of-% cross-code
spread is plausible). NWA validity is not explicitly enforced (z_res-range
warning only) — acceptable for ε ≪ 1, documented.

### P1-5 — firas.py: default marginalisation inflates quoted limits ~1.8× (RESOLVED 2026-07-03: no figure impact; docs + anchor tests added)

`upper_limit_mu()` defaults to `marginalise_y=True` (and vice versa), but
Fixsen 1996 §6.2 fit μ and y *separately* ("too similar to fit them
simultaneously"). The μ–y degeneracy over the FIRAS band inflates σ by ~82%,
so the default-call 95% limits are μ < 1.61e-4 / y < 1.68e-5 vs the
literature anchors 9e-5 / 1.5e-5. With `marginalise_y=False` /
`marginalise_mu=False` the code reproduces Fixsen to ≲8% (μ̂ = −1.23e-5 ±
3.59e-5 vs paper −1e-5 ± 4e-5). Not an algebra bug (GLS profiling verified
correct); a convention default + doc gap. Required follow-up per B5: audit
every call site (notebooks, paper figures) for which convention was assumed;
document; add anchor tests (current tests allow 3 orders of magnitude —
finding 2.8).

### P1-6 — firas.py: two coexisting CL conventions undocumented (FIXED 2026-07-03: module-level note)

`upper_limit_*` (two-sided z=1.96, Fixsen-style) vs
`profile_limit_floating_T` (one-sided Δχ²=2.71, CCJ24-style) are both
individually correct but mutually inconsistent and not distinguished at
module level. Document.

### P1-7 — distortion.rs: doc-only issues; latent band_weights edge case (open, low)

Decomposition algebra, B&F(2022) nonlinear path, δ_BF = δ_GS + μ/β_μ offset,
and the CODATA intensity prefactor all verified independently; 13/13 module
tests non-circular. Two low-priority items: stale docstring variable
definitions (lines 74-81, extra norm factors vs code); `band_weights`
half-weight rule keys off the parent-array edges, not the band edges —
benign at every current call site (all grids extend past [0.5, 18]) but a
footgun for pre-trimmed grids. Plus: `decompose_distortion` docstring should
warn that out-of-span spectra (frozen z<1100 bumps) yield an L² best-fit
triple, not a physical decomposition (inspect `residual`).

**P1-3 fix:** the test now asserts the correct analytic anchor (|ρ_eq − 1| <
1e-5 for any BE spectrum, from the I₄ = 4G₃ integration-by-parts identity)
plus a grid-refinement check that the residual shrinks under N-doubling
(pinning it as discretization error, not physics).

**P1-5 resolution detail:** call-site audit found exactly one consumer of
`upper_limit_mu/y` outside firas.py itself —
`notebooks/paper_figures/dark_photon_constraints.ipynb` — and it already
passes `marginalise_y=False`. **No published figure used the inflated
default.** Fixes applied: `.. warning::` blocks on both methods quantifying
the ~1.8× effect and naming the Fixsen-matching flag; module-level
"Statistical conventions" section distinguishing the two-sided Fixsen recipe
from the one-sided CCJ24 profile recipe (also closes P1-6); three anchor
tests in `test_firas.py` (μ limit within 15% of 9e-5 under the Fixsen
recipe; y within 45% of the statistical-only anchor 1.28e-5 and below the
published 1.5e-5; the default/Fixsen ratio pinned to 1.4–2.6 so a silent
convention change fails CI). Closes census finding 2.8 (previous tests
allowed 3 orders of magnitude).

### P1-8 — DC/BR: comment-only finding; one unverifiable literature input

No production bugs. Detailed balance, near-cancellation expansion (pitfall
#5), DC prefactor vs CS2012 Eq. 13, BR prefactor αλ_e³/(2π√(6π)) and
dimensional structure (two-body: one density survives Thomson
normalisation), and the independently derived μ-era DC/BR ratio (17.06 at
z=10⁶, x=0.1; crossover z≈3–4×10⁵, matching Danese & de Zotti) all verified.
F1 (FIXED 2026-07-03): the hand-calc comment on
`test_br_emission_coefficient_magnitude` had compensating errors
(BR_PREFACTOR 6.1e-40 → 3.82e-39 m³, θ_z^{-7/2} 2.1e16 → 1.5e15, g_ff 3 →
1.9, K_BR 6.6e-9 → 1.9e-9); assertion itself was valid. F4 (documented,
open): the softplus Gaunt offset 1.425 is a CosmoTherm
private-communication fit not printed in CRB2020 — limits verified, the
transition-region coefficient is unverifiable against published equations.

### P1-9 — greens: all coefficients match Chluba 2013/2015 raw text; one provenance comment wrong

No physics bugs; every fit coefficient (J_bb* 0.983/0.0381/2.29, J_μ
5.8e4/1.88, J_y 6.0e4/2.58, μ-amplitude 1.401 = 3/κ_c, x_c DC/BR
coefficients, x₀ = 3.6016) reproduces the papers exactly, and all limiting
cases verified analytically. M-2 (FIXED 2026-07-03): `greens.rs` comment
falsely attributed J_y's 6.0×10⁴ to an Arsenadze+2025 refit of an "original
5.9×10⁴" — raw Chluba 2013 Eq. 5 has 6.0×10⁴; comment corrected.
Methodology note: WebFetch of ar5iv garbled Chluba 2015 Eq. 25 coefficients;
raw-HTML reads were required (recorded for future audits). Open flags:
Arsenadze-2025 bump-broadening helpers not re-derived (dedicated pass
suggested); Rust +∞ vs Python 1e200 overflow sentinels on a dead x>500
branch (latent parity wart).

### P1-10 — recombination: coefficients exact; two documented conventions

No bugs. Pequignot α_B fit digits, Λ_2s1s = 8.22458 s⁻¹, C-factor, Sobolev
K, He Saha weights/energies all exact vs Seager+1999 / Chluba & Thomas 2011.
The classic 3.4-vs-10.2 eV β-exponential bug is absent — the code's
Saha-subtracted ODE form was proven algebraically identical to the raw TLA
(the two exponentials recombine into the full-Rydberg Saha relation).
Documented conventions: α_B/β at T_radiation rather than SSS99's T_matter
(≲1%, already flagged in code comments); fudge F = 1.125 vs CT2011's printed
best fit 1.126 (0.09% — comment annotated 2026-07-03). Open: no
CLASS/HyRec grid comparison possible in this environment (defers to B4
X_e-swap); literature-anchor test tolerances are 2–5× wide (tighten once an
external X_e table is available).

## Phase 2 (2026-07-03): B3 adversarial numerics — MMS, fuzzing, error budget

Gate: error budget quantified. Artifacts:

- **`tests/mms_convergence.rs`** (8 tests) — method of manufactured solutions.
  A smooth Δn_m(x, τ) = a(τ)·g(x) is made an exact solution by injecting the
  analytic residual S = ∂Δn_m/∂τ − L[Δn_m] through the *production* source
  path (`DcbrCoupling::photon_source` at the kernel level; a dense
  `TabulatedPhotonSource` at the solver level — **no production code changes
  were needed**). Measured **true errors** (not self-convergence) confirm the
  design orders exactly:
  - Kompaneets CN + Newton (incl. nonlinear Δn²): spatial p = 2.00 (log grid
    and production mixed log/linear grid), temporal p = 2.01.
  - Coupled DC/BR backward-Euler relaxation in the same Newton solve:
    spatial p ≈ 2.0–2.3, temporal p = 0.95–0.98 → 1 (BE design order).
  - End-to-end `ThermalizationSolver` (adaptive stepping, T_e coupling,
    recombination t_C, source splitting under `disable_dcbr`): reproduces the
    analytic solution to 2.1×10⁻⁴ (rel. x³-weighted L2) at N=2000,
    dtau_max=2; splitting order 0.96. The T_e feedback is neutralised
    analytically by building g(x) orthogonal to the two linear functionals of
    the perturbative Δρ_eq (∫x³g dx = 0 and ∫x⁴(2n_pl+1)g dx = 0).
- **`tests/conservation_fuzz.rs`** (3 tests, 14 randomized cases,
  deterministic splitmix64 seeds) — property-based ledger closure:
  - Energy: final Δρ/ρ vs independent Simpson quadrature of the heating rate;
    all 6 randomized heat scenarios (SingleBurst/Decaying/s-wave/p-wave,
    N=800–1600) close to 0.2–0.7% (10% budget).
  - Photon number, pure Compton (random ICs, full production solver):
    conserved to ≲3×10⁻⁸ over 10³–2×10³ adaptive steps — the residual is the
    Newton stopping tolerance (1e-8·max|Δn|), 4+ orders below truncation.
  - Photon number, monochromatic injection: closes to ≤0.1% (2% budget).
- **P1-2 closed** — discrete photon-number ledger on the coupled Newton path:
  pure-Compton conservation at machine precision (1.2×10⁻¹⁵ over 200 kernel
  steps with the tightest Newton tolerance), and the per-step balance
  ΔN = Σ w_i[dτ·em_i(neq_i − Δn_i) + S_i] holds to <10⁻⁹ with DC/BR + source
  active (`photon_number_ledger_identity_with_dcbr_and_source`).
- **Error budget** (`dev/scripts/error_budget.py` →
  `dev/output/error_budget.md` + `.pdf`; inputs regenerable from the test
  suites and `examples/temporal_error_check.rs`):

  | Source | Setting | Rel. error | Method |
  |---|---|---|---|
  | Spatial, μ (full physics) | N=2000 | 1.8×10⁻³ | Richardson p=1.97 |
  | Spatial, μ (full physics) | N=4000 | 4.5×10⁻⁴ | Richardson p=1.97 |
  | Temporal, μ (production defaults) | dtau_max=10 | 2.9×10⁻³ | direct dtau refinement, p=1.00 |
  | Temporal, y (production defaults) | dtau_max=10 | 1.4×10⁻³ | direct dtau refinement, p=1.00 |
  | Spectrum, MMS true error | N=2000 | 9.1×10⁻⁵ | exact manufactured solution |
  | Spectrum, MMS true error | N=4000 | 2.3×10⁻⁵ | exact manufactured solution |

  Bottom line: **total discretization error on μ at production defaults
  ≈ 0.3%**, dominated by the first-order dtau_max=10 temporal cap (halving
  dtau_max halves it); spatial error is subdominant at N≥2000. Well inside
  the 2–5% PDE↔Green's-function band and the ±5% energy-conservation target;
  benchmark-pack (Part A) tolerances should be set no tighter than ~0.5% on μ
  at default settings.

### P2-1 — dy_max is not the binding temporal control at defaults (documented)

The convergence-order tests drive refinement through `dy_max` with
`dtau_max=200` (20× the production default 10). Extrapolating that sweep to
the default dy_max=0.02 would suggest ~5% temporal error on μ — but at the
actual defaults `dtau_max=10` binds (10 096 steps vs 581 if dy alone
controlled, z_h=2×10⁵ burst), and the directly measured error is 0.29%.
The dtau_max refinement is cleanly first-order (diffs halve exactly:
3.96→1.99→1.00 ×10⁻⁸), Richardson-consistent with the dy-sweep limit
(μ_∞ = 1.3956×10⁻⁵ from both). No code change; recorded so future
convergence claims cite the correct control parameter.

### P2-2 — number-conservation must be measured in the kernel's own weights (documented)

The discrete invariant of the conservative flux form is Σ x_i²Δx_cell,i·Δn_i
(trapezoidal cell weights, half cells at the boundaries). Measuring it with
any other quadrature (e.g. `spectrum::delta_n_over_n`'s midpoint rule) shows
an apparent O(dx²) "drift" (~10⁻⁴ relative at N=1200) as the spectral shape
evolves, which is quadrature mismatch, not a conservation violation. The fuzz
test documents this; relevant for anyone auditing conservation externally.

## Phase 2b (2026-07-05): open-thread closure

Remaining Part-B threads from Phases 0–2, closed this pass:

- **P1-10 closed — external X_e anchor (HyRec-2).** Built HyRec-2 from
  source and ran it on the exact default cosmology; full comparison and
  observable-impact table in `dev/audit/xe_hyrec_comparison.md`, table
  archived in `dev/output/hyrec2_xe_default_cosmo.dat`. Peebles+Saha agrees
  with HyRec-2 to ≤1.9% for 200 ≤ z ≤ 1600, 5.7% in the He-recombination
  region (Saha vs non-equilibrium He), 33% in the z ≲ 50 tail (documented
  α_B(T_rad) convention). Ingredient-level X_e-swap (B4): P_s ≤0.9% where it
  matters, y_γ ≤1.6%, but **γ_con/ε² moves up to +25% (ε limit −10.5%) for
  dark-photon masses with z_res ≈ 1800–2500 (m ≈ 1.2–2.5×10⁻⁹ eV)** — the
  Saha-He-kink d-factor sensitivity predicted by P1-4, now quantified; must
  be stated as a validity bound on the dark-photon figure (Sect. 7).
  `test_xe_vs_recfast_milestones` tightened from order-of-magnitude bands to
  ±6% around exact HyRec-2 values (closes the "anchor tolerances 2–5× wide"
  item).
- **P1-7 closed (doc-only fixes applied).** Stale `M_y/G_y/G_μ` docstring at
  `distortion.rs` corrected to the code's plain inner products; `band_weights`
  precondition (grid must extend past the band) documented;
  `decompose_distortion` now warns that out-of-span spectra (frozen z<1100
  bumps) yield an in-band L² best fit, not a physical decomposition.
- **P1-9 sentinel wart closed.** Python `_photon_survival_probability_numerical`
  now uses `np.inf` (was `1e200`) for the dead x>500 bose-factor branch,
  mirroring Rust's `f64::INFINITY` with the same rationale comment.
- Python suite after changes: 327 passed (incl. parity + FIRAS anchors).

- **P1-9 M-4 closed — Arsenadze bump-broadening audit**
  (`dev/audit/arsenadze_broadening_audit.md`). All five helpers (`f_cs`,
  `alpha_cs`, `beta_cs`, `broadened_bump`, `f_int`) verified verbatim against
  the raw arXiv LaTeX source of Arsenadze et al. 2024 (arXiv:2409.12940,
  App. green_func_y), with the log-normal parameters independently re-derived
  from the paper's F exponent. The intensity(paper)↔occupation(code)
  convention explains the G₂/x² prefactor; absolute normalization confirmed
  factor-exact numerically (∫x²G/G₂ = 0.99997 at P_s=1). y_γ ≤ 0.11 across
  the allowed y-era, inside the paper's small-y regime. Findings all LOW:
  A-1 median-vs-mode comment (fixed in greens.py; Rust had no such comment),
  A-2 benign pointwise jump at the y_γ=1e-6 fallback threshold (integrated
  quantities continuous), A-3 x'y_γ ≈ 1.1 at the gated band edge where the
  code's exact log-normal is strictly more accurate than the paper's
  Gaussian-in-x. No published-figure impact.
- **B4-3 complete — reference-cosmology audit**
  (`dev/audit/reference_cosmology_audit.md`, 19 comparison sites, findings
  RC-1…RC-5). **No wrong-cosmology comparison at a level affecting published
  numbers.** Verified: Rust `planck2015` matches the CosmoTherm DI-file
  headers on all seven parameters (the headers' "Om"=0.264737 is Ω_cdm, not
  total matter); `Cosmology::default()` matches Chluba 2013's stated
  parameters exactly, covering all ~136 literature benchmarks. RC-2: CCJ24
  (arXiv:2409.12115) publishes no numeric cosmology; running its comparison
  on the Chluba-2013 default vs planck2018 moves γ_con by ≤3% (≤1.5% in ε) —
  documented, not fixable further. Doc fixes applied: RC-3 (default-cosmology
  docstring now notes CosmoTherm's N_eff=3.04 vs the paper's 3.046), RC-4
  (`firas.py` _T_CMB=2.726 attribution corrected — Fixsen & Mather 2002 give
  2.725; offset absorbed by floating-T fit parameters), RC-5
  (`data/cosmotherm/README.md` Ω_cdm mislabel). RC-1 open with P0-6:
  `adiabatic_cooling.ipynb` uses paper-convention `planck2015` (2.7255 K)
  against CT-convention DI data (sub-0.1% effect).

Still open / blocked on EB: P0-6 (planck2015 T_CMB convention decision),
Phase 1 memo spot-check (the Phase 1 gate), P1-8 F4 (softplus Gaunt 1.425 —
unverifiable against published equations; documented, nothing further
possible).

## Phase 1 conclusion

All eight module audits complete (memos in dev/audit/). **Zero confirmed
production physics bugs.** Confirmed defects were confined to: one
wrong-physics test (P1-3, fixed), stale/wrong comments (P1-1, P1-8 F1,
P1-9 M-2, fixed), and statistical-convention documentation gaps (P1-5/P1-6,
fixed; no figure impact). Independence checks: the planted β_μ reciprocal in
the distortion brief was caught and refuted; the firas and electron_temp
audits each independently rediscovered the pitfall-#9 failure mode in
existing tests. Remaining open threads feed later phases: coupled-path ΔN/N
unit test + MMS candidate (P1-2 → B3), softplus Gaunt 1.425 provenance
(P1-8 F4), Arsenadze bump-broadening pass (P1-9), CLASS X_e comparison
(P1-10 → B4). EB spot-check of the memos is the Phase 1 gate.

## Python API & documentation audit (2026-07-08)

Report-only pass over `python/spectroxide/` (usability + doc accuracy): every doc
code example executed, scripted signature↔docstring diff over the full public
surface, three no-context fresh-eyes agents run against public docs only. Full
findings in `dev/audit/python_api_audit.md`. Headlines: README's quick-start
decaying-particle example is physically dead (`gamma_x=5e4` s⁻¹ → lifetime
2×10⁻⁵ s; returns noise); `docs/api/greens.rst` unit comment wrong by 10⁶
(Jy/sr vs MJy/sr — Rust and Python converters use different units); its worked
heating example yields μ = 0.23 (nonlinear); `solver.rst` required-keys table
wrong in two rows; `solve()` docstring names a nonexistent `intensity` property;
`run_sweep(cosmo_params=Cosmology(...))` TypeErrors while `solve(cosmo=...)`
accepts the dataclass. Docstring coverage otherwise excellent (2 gaps in ~120
callables checked); zero dead API names in docs/notebooks except one stale
executed warning in `cosmotherm_comparison.ipynb`. No fixes applied.
