# Plan: Kompaneets moment-hierarchy verification + ironclad physics suite

**Author target:** hand this to Opus to implement.
**Date:** 2026-07-07 (Validation Round 2 follow-up). Revised same day after
review against `tests/mms_convergence.rs` and `src/kompaneets.rs` (see §0 —
the original draft's coverage claim was wrong, and §3's regime numbers were
off by ~10×; both fixed below). Part II added same day: broader
independent-anchor tests from the coverage review.
**Branch:** everything in this plan is done directly on `main`.
**Part I test file:** `tests/kompaneets_moments.rs` (§0–§9).
**Part II:** separate files per item (§II.0–§II.7).

---

## 0. What this is, is NOT, and what already exists (read first)

This verifies the **bare Kompaneets operator in isolation** against **exact
analytic moment identities** whose coefficients come from the literature
Kompaneets equation, not from the code. It is deliberately DC/BR-off,
expansion-off, fixed-θ_e.

**This is NOT the Chluba photon Green's function** (`greens_function_photon`,
`greens.rs`). That object is the end-to-end cosmological thermalization
response built on *fitted* visibility functions and validated against
CosmoTherm at ~2–5%. Comparing the PDE to it cannot catch a wrong
drift/recoil coefficient, because the fit's tolerance absorbs it.

**Relation to existing coverage — be precise, this determines the test's
entire value:**

- `tests/mms_convergence.rs` ALREADY tests the Kompaneets **dynamics**: it
  injects a source built from the analytic operator applied to a manufactured
  solution and shows the kernel converges to that solution at design order
  (O(Δx²) spatial, O(Δτ²) temporal; φ = 1, with `(2n_pl+1)` and `Δn²`
  included). Do not claim "no current test checks the dynamics" — false.
- **What MMS cannot catch:** the MMS operator (`MmsCase::kompaneets_op`) was
  transcribed from the code's own flux form. If the flux form itself carried
  a wrong coefficient (recoil `2Δn` instead of `Δn`, `x³` instead of `x⁴`),
  MMS verifies the code against the same wrong equation and passes. MMS is
  *discretization* verification against the code's *formulation*.
- **This test pins the formulation** against coefficients derived
  independently from the physics: `(k−2)(k+1)` and `(k−2)` in (★), the
  Zeldovich–Sunyaev energy-shift law `4 − x₀`, and (T5) the analytic
  Y_SZ source shape with its **amplitude**, which no existing test pins
  (`test_heat_spectral_shape_y_era`, `tests/heat_injection.rs:7844`, fits
  the amplitude away before asserting shape).
- `photon_number_conserved_coupled_path_pure_compton`
  (`tests/mms_convergence.rs:397`) already covers T1's content at rel drift
  < 1e-11 over 200 steps. T1 below is a cheap in-harness re-anchor, not new
  coverage.

Put this section (condensed) in the module doc comment so a future dev does
not delete the file as "redundant with MMS."

---

## 1. Physics: exact moment ODEs

The non-relativistic Kompaneets equation in the Comptonization variable
`y = ∫ θ_e σ_T n_e c dt`, with `x = hν/kT_e`, in the test-particle limit
(stimulated and quadratic terms dropped — handled exactly in §5 tier (b)):

    ∂n/∂y = (1/x²) ∂/∂x [ x⁴ ( ∂n/∂x + n ) ]          (the +n term is recoil)

Note the code's `x = hν/kT_z`; at φ ≡ T_z/T_e = 1 (all moment tests) the two
normalizations coincide, so no conversion is needed.

Define the moments `M_k(y) = ∫ x^k n(x,y) dx`. Multiply by `x^{k-2}`,
integrate over x, integrate by parts twice (zero-flux boundaries kill the
surface terms). Result — **exact**, recoil included:

    dM_k/dy = (k-2)(k+1) M_k  −  (k-2) M_{k+1}          (★)

Coefficients (all analytic, no fitted content):

| k | (k-2)(k+1) | (k-2) | meaning |
|---|-----------|-------|---------|
| 2 | 0         | 0     | photon number conserved: dM_2/dy = 0 |
| 3 | 4         | 1     | energy: dM_3/dy = 4 M_3 − M_4 |
| 4 | 10        | 2     | dM_4/dy = 10 M_4 − 2 M_5 |
| 5 | 18        | 3     | dM_5/dy = 18 M_5 − 3 M_6 |

The "4" in the k=3 row is Compton up-scattering (the y-distortion energy
gain); the `−M_{k+1}` term is recoil down-scattering. For a **narrow line at
x_0**, `M_{k+1}/M_k → x_0`, so the leading energy law is

    d ln M_3/dy |_{y→0} = 4 − x_0                        (★★)

**Derivation check for the implementer:** re-derive (★) yourself before
coding; do not trust the table blind. The double integration by parts is the
step where a sign error hides. (This revision re-derived it: the table is
correct.)

**Exact identity for the code's φ=1 flux (basis of §5 tier (b)):** the code
evolves Δn with flux `x⁴[dΔn/dx + (2n_pl+1)Δn + Δn²]` at φ = 1. The same
two integrations by parts give, with NO regime restriction,

    dM_k/dy = (k-2)(k+1) M_k − (k-2) [ M_{k+1} + C_k ]   (★′)
    C_k = ∫ x^{k+1} ( 2 n_pl Δn + Δn² ) dx

where `M_k = ∫x^k Δn dx` and `C_k` is computed by the same quadrature from
the evolved Δn and the analytic `n_pl`. (★′) is exact at continuum level;
(★) is (★′) with `C_k → 0` in the test-particle regime.

---

## 2. Why (★) is a real test, not a self-consistency tautology

`M_4`, `M_5` on the RHS are measured from the code, but the **coefficients**
`(k-2)(k+1)` and `(k-2)` are analytic and literature-derived. A bug in the
formulation or stencil — wrong flux power, wrong recoil coefficient,
misplaced factor — makes the code's discrete moments obey a *different*
relation, so `FD_y(M_k)` vs the analytic-coefficient RHS breaks. Asserting
k=3,4,5 simultaneously over-determines the operator structure: three
independent coefficient pairs pin the x⁴ weighting and the recoil term hard.

Caveat on (★′): its `C_k` correction is transcribed from the code's flux
form, so tier (b) alone would inherit MMS's tautology. The independent
physics content lives in tier (a)/(★) and in T2/T5; tier (b) is the
diagnostic that separates regime contamination from real failures.

---

## 3. Regime selection (original draft was quantitatively wrong here)

Two constraints on `(x_0, σ_0, A, y_total)`:

**(i) Test-particle contamination.** The stimulated correction is
`2n_pl(x) ≈ 2e^{−x}` and the quadratic one is `Δn²/Δn ≈ A`.
NB the original draft claimed `A = 1e-3 ⇒ Δn² ≲ 1e-6 of Δn` — wrong
arithmetic: `Δn²/Δn = Δn ≈ 1e-3`, i.e. a ~0.1% relative contamination of
the recoil moment term. Both contaminations are *measured exactly* as `C_k`
in (★′), so they set the tier-(a) tolerance rather than a hidden systematic.
Estimate at line center: relative contamination of the k=3 RHS
`≈ 2e^{−x_0+σ_f²/2}·M_4/|4M_3−M_4| + A·(same)`. Target ≲ 1%.

**(ii) Line broadening — the binding constraint the original draft missed.**
Kompaneets diffusion broadens a narrow line at rate `dσ²/dy = 2x_0²`, which
dwarfs the mean drift `d⟨x⟩/dy = (4−x_0)x_0`. The draft's `y ∈ [0.02, 0.08]`
is 5–10× too large: at `x_0 = 6, y = 0.08` the width grows by
`Δσ² = 2·36·0.08 ≈ 5.8` ⇒ `σ_f ≈ 2.4`, flooding the stimulated region and
approaching both boundaries. Budget: demand
`σ_f = √(σ_0² + 2x_0² y) ≲ 1`, i.e. `y ≲ (1 − σ_0²)/(2x_0²)`.

**Concrete defaults (re-derive if you change any of them):**

- `x_0 = 7`, `σ_0 = 0.5`, `A = 1e-3`.
- `y_total ≈ 0.008` ⇒ `σ_f ≈ 1.1`, `⟨x⟩` drift `≈ −0.17`, contamination of
  the k=3 RHS ≈ 0.8% (from (i) with `2e^{−7+0.6} ≈ 3e-3`, `M_4/M_3 ≈ 7`,
  RHS ≈ −3M_3).
- `M_3` still moves by `|4−x_0|·y ≈ 2.4%` over the run — measurable; the
  per-interval FD in T3 is exact-arithmetic clean (change per step ≫ f64
  noise on `M_3`).
- Temporal resolution: per-step variance growth must resolve the profile,
  `2x_0² θ_e dτ ≲ σ_0²/10` ⇒ `θ_e·dτ ≲ 2.5e-4`. With the knob
  `θ_e = 1e-2`, `dτ = 2.5e-2`, that is ~32 steps for `y_total = 0.008`;
  use ~64–128 steps for margin.
- Amplitude floor: check `M_3 / (ε_f64 · Σ w_i x_i³)` stays ≥ ~1e8 so the
  moments aren't roundoff-dominated. Same check for `M_6` before enabling
  k=5 (stumbling point 6).

---

## 4. Test setup (kernel-level, mirrors `tests/mms_convergence.rs`)

Reuse the MMS driver pattern, with these corrections to the original sketch
(verified against `src/kompaneets.rs:508`):

- `dcbr` is `Option<&DcbrCoupling>` — pass **`None`**. Do not build a
  zero-filled `DcbrCoupling` (MMS only builds one because it needs
  `photon_source`).
- The 9th argument `max_dn_abs` is NOT a tolerance you pick: the kernel sets
  its adaptive Newton tolerance as `tol = 1e-8·max_dn_abs + 1e-14`
  (`kompaneets.rs:931`). Passing `0.0` gives the tightest (1e-14 absolute)
  tolerance; MMS does exactly this and converges within 30 iterations.
- **Grid:** use `FrequencyGrid::log_uniform(0.2, 30.0, n)` for the primary
  runs (as MMS does) — no log/linear transition to keep the line away from.
  Optionally add one production-grid (`GridConfig::default()`) repeat as a
  robustness check, mirroring `mms_kernel_spatial_order_production_grid`.
- **Weights:** there is no weight accessor in `grid.rs`. Copy (or factor
  into a shared test util) `number_weights()` from
  `tests/mms_convergence.rs:385`: `w_i = x_i²·Δx_cell,i` with half cells at
  the boundaries — this is the kernel's exact discrete number invariant.
  For higher moments use the SAME cell widths: `M_k = Σ_i w_i x_i^{k−2} Δn_i`,
  and the same rule for `C_k`. With any other quadrature T1 will NOT sit at
  1e-11 (stumbling point 8).

```rust
// φ = 1: pass theta_e == theta_z. θ_e is a free knob that only sets
// y-per-step (y = θ_e·Σdτ); the operator is θ_e-independent in y. It need
// NOT be physical — document as a knob.
let theta = 1.0e-2;
let dtau  = 2.5e-2;              // θ·dτ = 2.5e-4, see §3
let n_steps = 32;                // y_total = 8e-3 (×2,×4 for T4-light)

let grid = FrequencyGrid::log_uniform(0.2, 30.0, 2000);
let w = number_weights(&grid);   // copied from mms_convergence.rs
let mut delta_n = gaussian_line(&grid, 7.0, 0.5, 1e-3);
let mut ws = KompaneetsWorkspace::new(&grid);

for k in 0..n_steps {
    let (converged, _, _) = kompaneets_step_coupled_inplace(
        &grid, &mut delta_n, theta, theta, dtau,
        None,          // DC/BR off — no zeros struct needed
        None,          // T_e fixed
        &mut ws, 0.0, 30);
    assert!(converged, "Newton diverged at step {k}");
    // record M_2..M_6 and C_2..C_5 at y = theta*(k+1)*dtau
}
```

---

## 5. Assertions

**T1 — photon number (anchor, existing coverage re-run in-harness).**
`|M_2(y) − M_2(0)| / Σw_i|Δn_i(0)| < 1e-9` at every recorded step. Cite
`photon_number_conserved_coupled_path_pure_compton` in the comment; if T1
fails here but that test passes, the weights were rolled wrong (§4).

**T2 — energy law (fully analytic RHS, the ZS anchor).**
Assert `|d ln M_3/dy|_{y→0} − (4 − M_4(0)/M_3(0))|` below the tier-(b)
tolerance — this is the k=3 identity with a measured ratio, needing no
width-series derivation. Separately assert `|M_4(0)/M_3(0) − x_0| ≤`
`3σ_0²/x_0 + quadrature slop` (for a narrow Gaussian
`M_4/M_3 = x_0(1 + 3σ²/x_0² + …)` — note the coefficient is 3, not the 1
in the original draft; just measure it rather than trusting the series).
Print `4 − x_0` vs the measured slope as the human-readable ZS number.

**T3 — moment hierarchy, k=3,4,5 — two tiers (core test).**
For each recorded interval, central `FD_y(M_k)` at the interval midpoint vs:

- **Tier (a) — physics:** RHS `(k-2)(k+1)M_k − (k-2)M_{k+1}` (★), analytic
  coefficients only. Tolerance = truncation floor (from T4-light) + the
  measured contamination bound `(k-2)·C_k / |RHS|`. This tier carries the
  independent-physics content.
- **Tier (b) — exact:** RHS from (★′) including the measured `C_k`.
  Tolerance = truncation floor only. Exact for the code's φ=1 flux at
  continuum level — no regime fragility.

Failure triage: (a) fails, (b) passes ⇒ regime contamination — shrink `y`
or raise `x_0`, do NOT loosen the tolerance. Both fail ⇒ real
formulation/stencil bug (or y-vs-τ factor, stumbling point 1).

**T4-light — truncation floor (do NOT rerun a full order study).**
MMS already established O(Δx²)/O(Δτ²) for this exact operator at φ = 1.
NB the original draft claimed "temporal ~O(dtau) coupled" — wrong for this
configuration: DC/BR is off, the step is pure Crank–Nicolson, order 2
(measured 1.7–2.4 in `mms_kernel_temporal_order_pure_kompaneets`). Here,
two refinements suffice: halve `dx` (hold `y`, `dτ`) and separately halve
`dτ`; assert the tier-(b) residual drops by ≳3× under the spatial halving
(or is already at the FD/roundoff floor). This proves the T2/T3 tolerances
are truncation-dominated, i.e. derived rather than tuned.

**T5 — (φ−1) source term: pointwise Zeldovich–Sunyaev shape AND amplitude
(NEW — closes a genuine coverage hole).**
Every kernel-level test in the repo (all of MMS, T1–T4 above) runs φ = 1,
where the `(φ−1)n_pl(1+n_pl)` flux branch — the term that converts heating
into y-distortions — vanishes identically. Solver-level coverage
(`test_heat_spectral_shape_y_era`) checks shape *correlation with a
best-fit amplitude*, so a wrong prefactor (factor 2, θ_e↔θ_z swap in the
y normalization) passes today.

Setup: `Δn = 0` initially, `φ = 1 ± ε` with `ε ≈ 1e-3` (kernel args: the
code's `φ = T_z/T_e = theta_z/theta_e`; hold both fixed, `rho_coupling =
None`). Take a few steps, `Δy = θ_e·Σdτ ≈ 1e-3`. Continuum prediction,
exact to O(ε², Δy²):

    Δn(x)/Δy = (φ−1)·(1/x²) d/dx[ x⁴ n_pl(1+n_pl) ] = (1−φ)·Y_SZ(x),
    Y_SZ(x) = [x e^x/(e^x−1)²]·[x coth(x/2) − 4]

(derivation: `d/dx[x⁴ e^x/(e^x−1)²] = x³ e^x/(e^x−1)²·[4 − x coth(x/2)]`;
re-derive before coding). Assert **pointwise** over `x ∈ [0.5, 15]` at a
tolerance derived from two step sizes (Richardson) plus O(ε) — expect
sub-percent. Sign anchors: T_e > T_z ⇒ φ < 1 ⇒ positive y-distortion;
zero crossing at x ≈ 3.830. **Hardcode the analytic shape in the test**;
then separately assert it matches `spectrum::y_shape` (catches convention
drift between test and library, and validates the library shape's
normalization for free). Also assert `M_2` conservation within this run —
the (φ−1) term is a pure flux divergence, so number conservation must hold
on this branch too (currently untested there).

This is the highest-value addition: analytic, pointwise, amplitude-pinning,
and it uniquely determines which θ normalizes the code's Comptonization
variable.

**T6 — linearity diagnostic (cheap).** Rerun the T3 harness at `A/2` and
`−A`. The tier-(a) residual per unit amplitude must be amplitude-independent
to O(A) and sign-symmetric; a residual scaling ∝ A isolates the `Δn²` term,
an asymmetry under `A → −A` flags something worse. Two extra runs, no new
machinery.

---

## 6. Tolerances — derive, do not tune to pass

- `M_2` conservation is machine-precision (flux telescoping): 1e-9 is safe
  (existing test achieves < 1e-11 over 200 steps).
- Tier (b) tolerance = truncation floor measured by T4-light, with margin
  ×3. Tier (a) tolerance = tier (b) + measured `(k-2)C_k/|RHS|` per
  interval (computed, not guessed). T5 tolerance from two-step-size
  Richardson + O(ε).
- Rule unchanged: the tolerance must be looser than the observed error at
  the finest resolution and tighter than the error a plausible-magnitude
  coefficient bug would produce (e.g. recoil coefficient off by one unit
  shifts the k=3 RHS by `M_4/|RHS| ≈ 7/3` — enormous; a swapped θ in T5
  shifts the amplitude by `θ_z/θ_e`). If the bounds cross, refine.
- **Do not** read a tolerance off the passing run and hard-code it
  (CLAUDE.md Pitfall #9). T4-light plus the measured `C_k` are the
  anti-tuning safeguards.

---

## 7. Stumbling points (anticipated)

1. **y vs τ confusion.** The kernel steps in `dτ` and multiplies by `θ_e`
   internally; the physical Comptonization variable is `y = θ_e·Σdτ`. All
   moment ODEs are in `y`. Getting this wrong rescales every RHS by `θ_e`
   and T3 fails by exactly that factor — a useful diagnostic. T5 pins it
   independently.

2. **Regime contamination is now measured, not assumed.** If tier (a) drifts
   out of tolerance while tier (b) holds, the regime is contaminated —
   shrink `y` or raise `x_0`, never the tolerance.

3. **Broadening, not drift, is the budget.** `dσ²/dy = 2x_0²` (§3). Assert
   at the end of the run: `⟨x⟩ = M_3/M_2 > x_0 − 1` and measured
   `σ² = M_4/M_2·(M_2/M_3)²…` — simpler: assert `|Δn|` at `x = 2` and at
   `x_max` stays below 1e-6 of the peak throughout (boundary/low-x guard).

4. **Boundary contamination.** With `log_uniform(0.2, 30, n)` there is no
   log/linear transition; only the `x_max` guard from point 3 applies. On
   the optional production-grid repeat, also check the transition region.

5. **Δn² not negligible.** Covered exactly by `C_k` (tier b) and diagnosed
   by T6. Remember the correct scaling: contamination is O(A) relative,
   not O(A²).

6. **Roundoff floor in high moments.** `M_6` at `x_0 = 7` with `A = 1e-3`
   can hit f64 cancellation. Check the moment-to-roundoff ratio (§3); if
   `M_6` is noise, restrict T3 to k=3,4 and note why.

7. **T_e must stay fixed.** `rho_coupling = None` (as MMS does). Do not
   route through `ThermalizationSolver` for the primary test. A solver-level
   end-to-end variant is optional and secondary; if added, back out the
   realized `y` from the accumulated `∫θ_e dτ` diagnostic.

8. **Quadrature-weight mismatch.** Use `number_weights()`-style weights
   (§4) for ALL moments including `M_2(0)` and `C_k`. A different rule for
   the initial vs evolved moments introduces a spurious offset into T1.

9. **Newton at large θ_e·dτ.** `θ_e·dτ = 2.5e-4` per §3 is far below any
   stress regime; keep `assert!(converged)` every step anyway.

10. **T5 sign conventions.** The code's `φ = T_z/T_e` (CLAUDE.md Pitfall #1
    — opposite the intuitive ratio). Get the sign of `(1−φ)` vs the
    y-distortion direction right by asserting the physical anchor (T_e > T_z
    ⇒ Δn > 0 at high x, < 0 at low x, crossing at 3.830), not just an
    overall magnitude.

---

## 8. Acceptance criteria

- `tests/kompaneets_moments.rs` added; module doc includes the condensed §0
  "relation to MMS / not the Chluba GF" paragraphs.
- T1 at 1e-9; T2 at tier-(b) tolerance; T3 tiers (a) and (b) for k=3,4
  (k=5 if roundoff permits); T4-light shows the residual is
  truncation-dominated; T5 pointwise with amplitude; T6 linearity.
- `cargo test --release` green (never debug; memory `release-tests-only`).
- `cargo clippy -- -D warnings` clean.
- This adds test code, so CI should run — no `[skip ci]`.

---

## 9. Out of scope (do not gold-plate)

- No ln-x Gaussian profile-shape test: with recoil the ln-x operator has
  x-dependent coefficients, so the clean constant-coefficient Gaussian has
  no valid window. The moment hierarchy supersedes it and is exact.
- No relativistic-correction comparison: the code's Kompaneets is
  non-relativistic; test it against its own non-relativistic moments.
- No electron-side energy cross-check here (dM_3/dy vs the code's Compton
  heating / `Δρ_eq` path, CLAUDE.md Pitfall #4): T_e is fixed in this
  harness. That check is now Part II item §II.2, which uses `−dM_3/dy`
  from this harness as the photon-side reference.

---
---

# Part II: Broader ironclad-verification suite

Motivation: the audits keep finding bugs where a physical term has no test
that pins its **amplitude** against an anchor independent of the code
(shape-correlation with fitted amplitude, self-consistency of a stepper
against the code's own coefficient, and tests calibrated to code output all
failed to catch past bugs — CLAUDE.md Pitfalls #8, #9). Part II fills the
amplitude-pinning holes found in the 2026-07-07 coverage review.

Verified-absent before writing this (do not re-litigate): μ zero-crossing at
x_inj ≈ 3.602 EXISTS (`heat_injection.rs:1601`); burst superposition EXISTS;
DC backward-Euler-vs-exponential EXISTS but uses the code's own coefficient
(stepper check only); P&B 2009 BE-equilibrium T_e benchmark EXISTS;
`test_compton_equilibrium_mu_distortion` is sign-and-order-of-magnitude only;
the `x_c` test at `heat_injection.rs:5424` compares `greens::x_c_dc` to
`greens::x_c_br` — self-referential, no PDE or literature anchor.

Recommended order: II.7 (coverage matrix) first — it is cheap and scopes the
rest; then II.2, II.3 (quick, pure-quadrature); then II.1 (highest physics
value, medium effort); II.4 and II.5 independent of the Rust suite; II.6
optional last.

---

## II.1 Low-frequency μ(x) profile vs analytic x_c(z)

**New test file:** `tests/mu_photosphere_profile.rs`. Highest-value item:
the only test of the *coupled* DC/BR + Compton balance — the core μ-era
thermalization physics — against an analytic target rather than CosmoTherm's
2–5% envelope.

**Physics.** In the μ-era the quasi-stationary balance between photon
production/absorption (rate ∝ 1/x² at x ≪ 1 per unit y) and Compton
redistribution gives the classic frequency-dependent chemical potential

    μ(x) ≈ μ_∞ · exp(−x_c(z)/x),      x_c ≪ x ≪ 1

(Danese & de Zotti 1982; Chluba & Sunyaev 2012; Chluba 2014). The
implementer must derive/transcribe x_c(z) from the **literature** (Chluba &
Sunyaev 2012 give the DC and BR pieces; combined x_c ≈ (x_c,DC² + x_c,BR²)^½
form — verify against the paper, do not trust this line). The code's
`greens::x_c_dc` / `x_c_br` may be *cross-checked* against the literature
formula as a bonus assertion, but must NOT be the test target.

**Setup.** PDE with steady small heating (`TabulatedHeating` or a long
`DecayingParticle` plateau) deep in the μ-era; run at fixed z-windows around
z ≈ 2×10⁶ (DC-dominated) and z ≈ 3×10⁵ (BR contribution significant) long
enough to reach quasi-stationarity at low x (the relaxation time at x is
~1/Λ(x) Compton times — estimate before choosing the window). Extract

    μ(x) = −Δn(x) / [n_pl(x)(1 + n_pl(x))]

(from n = 1/(e^{x+μ(x)}−1) linearized), after removing the temperature-shift
component (fit and subtract the G_bb piece first — decomposition machinery in
`distortion.rs`; a residual T-shift contaminates the low-x tail as μ_eff ∝ x·
stuff and biases the fit). Fit ln μ(x) vs 1/x over a window
[~3x_c, ~0.3]; the slope is −x_c.

**Assertions.** Fitted x_c matches literature x_c(z) at both redshifts;
tolerance from (a) fit-window sensitivity (vary window bounds ×2, spread
sets the floor) and (b) grid refinement at low x. Bonus: `greens::x_c_*`
agree with the literature formula.

**Stumbling points.**
1. Quasi-stationarity: at x ~ x_c the DC/BR rate is fast, but the *approach*
   to the exp(−x_c/x) shape in the intermediate window is Compton-limited.
   Check convergence by doubling the run window and requiring the fitted
   x_c to move < tolerance/3.
2. Grid: need log spacing dense at x ~ 1e-3–1e-1 (production grid was built
   for this — but check the refinement is adequate; `RefinementZone` if
   not). x_min must sit well below x_c.
3. The window must satisfy x_c ≪ x ≪ 1 on BOTH sides; at z = 3×10⁵ x_c is
   smaller and the window wider — derive per-z windows, don't reuse.
4. Expansion redshifts x during the run; keep the z-window narrow or account
   for the drift of x_c across it (evaluate x_c at an effective z̄ and bound
   the variation).
5. μ_∞ itself is NOT the target (it depends on injection history); only the
   exponential slope is. Do not assert μ_∞.

## II.2 Compton-equilibrium temperature: analytic O(y) and O(μ) coefficients

**New test file:** `tests/compton_equilibrium_analytic.rs` + derivation
script `dev/scripts/compton_equilibrium_coefficients.py`.

**Gap.** `test_compton_equilibrium_mu_distortion` asserts only ρ_e > 1 and
ρ_e − 1 < 0.01. The perturbative path the solver actually uses,
`Δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃` (CLAUDE.md Pitfall #4 — the delicate one), has
no independent amplitude anchor.

**Anchor.** For Δn = y·Y_SZ(x) and Δn = μ·M(x):

    Δρ_eq = [∫x⁴(1+2n_pl)Δn dx]/(4G₃) − [∫x³Δn dx]/G₃,   G₃ = π⁴/15

Both integrals are analytic-shape integrals reducible to ζ/polylog values.
Evaluate them in `dev/scripts/compton_equilibrium_coefficients.py` with
mpmath quadrature to ≥ 12 digits from the analytic shapes ONLY (hardcode
Y_SZ, M(x), n_pl in the script — import nothing from spectroxide), commit
the script and paste the constants with derivation into the test. Sanity
identities the script must verify: ∫x²Y_SZ dx = 0 (number-conserving),
∫x³Y_SZ dx = 4G₃ (Δρ/ρ = 4y). Do NOT use a remembered literature value for
the O(y) coefficient — this review did not verify one; the quadrature is the
anchor.

**Assertions.**
1. `spectrum::compton_equilibrium_ratio` on n_pl + y·Y_SZ (and + μ·M)
   reproduces 1 + coeff·y (resp. μ) to quadrature+grid tolerance, for
   y, μ ∈ {1e-6, 1e-5, 1e-4} (linearity check included).
2. **The perturbative solver path itself** (grep `electron_temp.rs` for the
   function computing Δρ_eq from Δn; test whatever the solver calls, not a
   convenience wrapper) reproduces the same coefficients.
3. Cross-link to Part I: with the §4 harness, `−dM_3/dy` (photon energy loss)
   must equal the code's Compton heating rate diagnostic if one is exposed;
   if none is exposed, note that and skip — do not add plumbing for it.

**Stumbling points.** Shape-normalization conventions: verify the test's
hardcoded Y_SZ/M(x) match `spectrum.rs` conventions by asserting pointwise
agreement at 3 x-values first (catches a β_μ or G_bb convention mismatch
before it poisons the coefficients). Grid extent: x_max ≥ 50 for the
x⁴(1+2n_pl)Δn integrand with M(x) (check tail convergence explicitly).

## II.3 First-principles DC/BR coefficient magnitudes

**New test file:** `tests/rate_coefficients_first_principles.rs`.

**Gap.** The 10¹¹× BR bug (Pitfall #8) survived 375 tests because every test
either used the code's own coefficient or a regime where DC dominates. The
existing exponential-decay test verifies the stepper, not the magnitude.

**Anchor.** Recompute the rate coefficients in the test from literature
formulas with CODATA constants **typed literally in the test file — import
nothing from `constants.rs`** (that is the point of the test):

- DC: K_DC in the x → 0 limit for a Planck spectrum:
  K = (4α/3π)·θ_z²·I_pl with I_pl = ∫x⁴n_pl(1+n_pl)dx = 4π⁴/15 ≈ 25.9757.
  Compare against `dc_emission_coefficient*` at x ≪ 1 (the code includes a
  Gaunt-factor x-dependence; compare in the limit, or include the known
  first-order correction from the code's cited source — Chluba 2005/2014 —
  transcribed from the paper, not from `double_compton.rs`).
- BR: K_BR at a few (x, z) from the Born-approximation Gaunt factor
  (Brussaard & van de Hulst 1962, per the module doc) with n_e, n_ion from
  Y_p and Ω_b h² computed in-test. Tolerance ~ few % to cover the code's
  softplus interpolation vs raw Born.

**Assertions.** Code/first-principles ratio ∈ [1−tol, 1+tol] at each point.
A conventions bug (per-volume vs per-Thomson-time, wrong density factor,
θ_e↔θ_z) shows up as orders of magnitude or as a systematic (1+z)-dependent
drift — also assert the *ratio is z-independent* across 3 redshifts, which
isolates density-factor errors from Gaunt slop.

**Stumbling points.** Which densities enter: BR keeps ΣZ²N_i after Thomson
normalization (two-body), DC keeps none (one-body) — re-derive this before
comparing. X_e at the chosen z (deep thermalization era: X_e = 1 + f(Y_p);
match the code's helium treatment or pick z high enough that it is fully
ionized and compute the He contribution explicitly). θ_z vs θ_e in the DC
prefactor: at these z they differ at 1e-5 — irrelevant, but note it.

## II.4 Dark-photon γ_con: direct mixing-equation integration

**Deliverables:** `dev/scripts/gamma_con_landau_zener.py` + findings note
`dev/audit/gamma_con_lz_check.md` + (if agreement) a regression test pinning
`gamma_con` against 3–5 tabulated LZ values.

**Motivation.** The NWA `gamma_con` (both `src/dark_photon.rs` and
`python/spectroxide/dark_photon.py`) is implicated in the unresolved ~22%
discrepancy against Bryce's frozen-absorption curve (memory:
`axion-dp-distortion`). No test checks the NWA against the underlying
mixing dynamics.

**Method.** Integrate the 2-level γ–A′ system through the resonance with the
actual ω_p(z) profile (from `plasma_frequency_ev`), extract the conversion
probability, compare to the NWA result P ≈ πε²m_{A'}²/|d ln ω_p²/dt| (verify
this expression against the standard reference — e.g. Mirizzi, Redondo &
Sigl 2009 — before coding).

**Stumbling points.**
1. Stiff oscillation: the phase evolves fast away from resonance; integrate
   in a variable centered on the crossing (LZ variable) and average over the
   residual fast oscillation when extracting P (fit the asymptotic plateau,
   don't read a single endpoint).
2. Pick benchmark points where the NWA should be exact (narrow, adiabaticity
   parameter ≪ 1 AND ≫ 1 cases) and one near the validity boundary — the
   boundary case is where a 22%-class discrepancy would live.
3. In-medium damping/width is ignored by vacuum LZ; if the NWA formula in
   the code includes it, compare like with like and document.
4. This is diagnosis, not gold-plating: if LZ confirms `gamma_con`, the
   Bryce discrepancy is elsewhere (write that down); if not, file the bug
   before writing any regression test.

## II.5 FIRAS closed-loop coverage calibration

**Deliverable:** `python/tests/test_firas_coverage.py` (pytest, seeded RNG;
mark the full-N run `slow` and keep a small-N smoke version fast).

**Gap.** The paper's limits rest on `firas.py`'s fitting machinery. The
SciPost referee work reproduced the CCJ24 statistic to ~3% (memory:
`scipost-referee-fig8-validation`) — a comparison, not a calibration of our
own pipeline's statistical coverage.

**Method.** Monte Carlo, N ≈ 10⁴: mock monopole = Planck(T₀) + known
(ΔT, μ, y) injected at several amplitudes (0, ~1σ, ~5σ) + noise drawn from
the 43×43 covariance (multivariate normal, Cholesky). Fit with the
production pipeline (same band cuts and floating-T treatment as the paper —
the Fig. 8 lesson: floating-T profiling, not covariance, drove that offset).

**Assertions.**
1. Bias on each recovered parameter ≪ σ_param/√N·(a few).
2. 95% upper limits cover at 95% within binomial MC error.
3. χ² GOF at zero injection follows χ²_dof (KS test, loose p-threshold).

**Stumbling points.** Covariance conditioning — check how `firas.py` handles
the 43×43 matrix before Cholesky (regularization? band cuts first?); use the
identical path. Degeneracy: μ/y vs ΔT correlations make marginal vs profiled
limits differ — calibrate the SAME limit definition the paper quotes.

## II.6 H-theorem monotonicity (optional — nonlinear-regime coverage)

**Location:** add to `tests/kompaneets_moments.rs` (shares the harness).

Everything else in this plan is linearized. For fixed T_e the continuum
Kompaneets flow monotonically decreases the free-energy functional (code
units, φ = T_z/T_e, n = n_pl + Δn the full occupation):

    F[n] = ∫x²[ φ·x·n + n ln n − (1+n)ln(1+n) ] dx
    dF/dy = −∫x⁴ n(1+n) [ ∂_x( ln(n/(1+n)) + φx ) ]² dx  ≤ 0

(re-derive: the flux is J = x⁴n(1+n)·∂_x[ln(n/(1+n)) + φx]; one integration
by parts, zero-flux boundaries). This is a structural inequality valid at
ANY amplitude — the only proposed check of the solver's nonlinear regime
(dark-photon-like impulsive ICs, large Δn).

**Test.** Large-amplitude interior bump (max|Δn| ~ 0.1·n_pl(x₀), and one run
with Δn comparable to n_pl), pure Kompaneets (dcbr `None`), assert F
non-increasing every step within a truncation allowance measured by halving
dτ (CN preserves the inequality only to O(Δτ²) — bound observed positive
increments by the measured truncation, do not hand-tune).

**Stumbling points.** n must stay > 0 for the logs — reject absorption-line
ICs that drive n < 0, or clamp the integrand where n < ε and bound the
clamped mass. Compute n ln n − (1+n)ln(1+n) via a stable form at n ≪ 1
(n(ln n − 1) − n²/2 + … or just guard the x-range). Use the §4 weights.

## II.7 Coverage matrix (do this FIRST)

**Deliverable:** `dev/audit/COVERAGE_MATRIX.md`, maintained from now on.

Rows = physical terms/paths: Kompaneets drift, recoil, (φ−1) source, Δn²
term, DC magnitude, BR magnitude, Gaunt factors, perturbative Δρ_eq, T_e
full path, recombination X_e, expansion/redshifting, P_s, y_γ broadening,
γ_con, μ/y/G_bb decomposition, FIRAS χ², injection scenarios' bookkeeping.
Columns = test class: (1) exact structural identity, (2) analytic anchor
**pinning amplitude**, (3) design-order convergence (MMS-class), (4)
cross-code, (5) regression-only. Fill from the existing suites (cite test
names), mark Part I/II items as planned, and let empty (1)/(2) cells drive
any future work. The definition of "ironclad" this plan works to: every row
has at least one entry in class (1) or (2).

---

## II.8 Acceptance criteria (Part II)

- II.7 matrix committed first; each subsequent item updates its row.
- II.1: fitted x_c matches literature x_c(z) at two redshifts, tolerance
  derived per the item; window- and grid-sensitivity checks included.
- II.2: derivation script committed; coefficients asserted on both the full
  ratio and the solver's perturbative path.
- II.3: in-test constants independent of `constants.rs`; z-independence of
  the code/analytic ratio asserted.
- II.4: findings note committed regardless of outcome; regression test only
  if the NWA is confirmed.
- II.5: seeded, `slow`-marked MC; coverage and bias assertions pass.
- II.6: optional; if included, truncation-bounded monotonicity.
- All Rust items: `cargo test --release` green, `cargo clippy -- -D
  warnings` clean. Test-code commits run CI (no `[skip ci]`); the matrix and
  audit notes alone are `[skip ci]`.
- Everything lands on `main` directly.
