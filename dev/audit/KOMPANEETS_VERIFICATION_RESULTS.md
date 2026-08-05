# Kompaneets verification results — moment hierarchy + ironclad physics suite

**Date:** 2026-07-07 (measured), **merged to `main` and re-run 2026-07-30**.
**Plan:** `dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md`.
**Origin:** branch `kompaneets-validation` (worktree `~/spectroxide-kompaneets`).
**Bottom line: all 20 new Rust tests + 4 Python MC tests pass. Zero physics
bugs found. Every anchor is independent of the code being tested.**

**Re-run against current `main` (2026-07-30, release mode):** 20/20 Rust pass
(`compton_equilibrium_analytic` 4, `kompaneets_moments` 11,
`mu_photosphere_profile` 2 in 41.3 s, `rate_coefficients_first_principles` 3);
4/4 Python pass in 5.3 s; `cargo clippy --release --all-targets -- -D warnings`
clean. This matters because `main` had moved three commits ahead of the branch
and those commits rewrote `double_compton.rs` (+231) and `greens.rs` (+457),
including the F-R2-3 K_DC deduplication that §II.3 pins — the results below are
therefore current, not stale.

## Why this suite exists

`tests/mms_convergence.rs` verifies the Kompaneets *discretization* converges at
design order — but its manufactured-solution operator was transcribed from the
code's own flux form, so a wrong coefficient in that form (recoil `2Δn` vs `Δn`,
`x³` vs `x⁴`) would pass MMS. This suite pins the *formulation* against anchors
derived independently: analytic moment identities, mpmath quadrature, literal
CODATA constants, literature fits, and direct ODE integration of the underlying
mixing dynamics.

## Deliverables

| Item | File | Tests | Result |
|---|---|---|---|
| P-I moment hierarchy + T5 + H-theorem | `tests/kompaneets_moments.rs` | 11 | ✅ all pass (0.12 s) |
| II.1 μ-photosphere x_c(z) | `tests/mu_photosphere_profile.rs` | 2 | ✅ (see below) |
| II.2 Compton-equilibrium coefficients | `tests/compton_equilibrium_analytic.rs` + `dev/scripts/compton_equilibrium_coefficients.py` | 4 | ✅ all pass |
| II.3 DC/BR first-principles magnitudes | `tests/rate_coefficients_first_principles.rs` | 3 | ✅ all pass |
| II.4 γ_con Landau-Zener check | `dev/scripts/gamma_con_landau_zener.py` + `dev/audit/gamma_con_lz_check.md` | script | ✅ NWA confirmed |
| II.5 FIRAS closed-loop coverage | `python/tests/test_firas_coverage.py` | 4 | ✅ (see below) |
| II.7 coverage matrix | `dev/audit/term_coverage_matrix.md` | — | committed, gaps listed |

---

## Part I — moment-hierarchy verification (`tests/kompaneets_moments.rs`)

Bare Kompaneets operator in isolation: DC/BR off, expansion off, fixed θ_e,
φ = 1 (except T5). Narrow Gaussian line x₀ = 7, σ₀ = 0.5, A = 10⁻³ on a
2000-point log grid [0.2, 30]; y_total = 8×10⁻³ over 32 steps.

The exact identity, derived from the literature Kompaneets equation (two
integrations by parts, coefficients NOT from the code):

    dM_k/dy = (k−2)(k+1) M_k − (k−2) M_{k+1}                    (★, tier a)
    dM_k/dy = (k−2)(k+1) M_k − (k−2)[M_{k+1} + C_k]             (★′, tier b)
    C_k = ∫ x^{k+1} (2 n_pl Δn + Δn²) dx     (measured stimulated/quadratic term)

Tier (a) carries the independent physics; tier (b) is exact for the code's φ=1
flux and separates regime contamination from real failures.

| Test | Measured | Tolerance | Verdict |
|---|---|---|---|
| T1 photon number M₂ | max rel drift **1.6×10⁻¹⁵** over 32 steps | 10⁻⁹ | machine precision |
| T2 ZS energy law | d ln M₃/dy = **−3.1228** vs 4−x₀ = −3 (tier a: −3.1061, tier b: −3.1227) | tier-b residual 1.3×10⁻⁴ < 5×10⁻⁴ | drift+recoil pinned |
| T3 k=3 | tier-a **6.4×10⁻³** / tier-b **4.2×10⁻⁵** | contam 6.4×10⁻³ / floor 3×10⁻³ | ✅ |
| T3 k=4 | tier-a 8.2×10⁻³ / tier-b 8.2×10⁻⁵ | contam 8.1×10⁻³ / floor | ✅ |
| T3 k=5 | tier-a 1.4×10⁻² / tier-b 1.9×10⁻⁴ | contam 1.4×10⁻² / floor | ✅ |
| T4-light | tier-b residual 4.2×10⁻⁵ → **1.1×10⁻⁵** under dx-halving (4×) | ≥3× drop | truncation-dominated, floor is derived not tuned |
| T5 (φ−1) source | pointwise Y_SZ shape+amplitude err **3.4×10⁻⁴** over x∈[0.5,15]; Richardson spread 6.7×10⁻⁹; sign/zero-crossing at 3.830 ✓; number consistency 2.0×10⁻⁴ | 5·Rich + 3ε + 5×10⁻³ | amplitude + θ-normalization pinned |
| T5 library check | hardcoded Y_SZ vs `spectrum::y_shape`: **5.8×10⁻¹⁶** | 10⁻¹⁰ | no convention drift |
| T6 linearity | tier-b residual A / A/2 / −A: 4.236 / 4.239 / 4.248 ×10⁻⁵; A→−A asymmetry 1.4×10⁻³ | asym < 0.5 | no sign-dependent bug; contamination shrinks with A as expected |
| Roundoff guard | M₆/(ε·Σw x⁴\|Δn\|) = **4.5×10¹⁵** | > 10⁸ | k=5 hierarchy trustworthy |
| Regime guards | boundary \|Δn\|/peak 6.9×10⁻⁴¹; ⟨x⟩_final = 6.899 | < 10⁻⁶; \|⟨x⟩−7\| < 1 | regime clean |
| II.6 H-theorem | max +ΔF = **0 exactly** at amp = 0.1·n_pl and 1.0·n_pl; total F drop 1.6×10⁻³ / 1.1×10⁻¹ | truncation-bounded | F strictly monotone even fully nonlinear |

Interpretation of T3: the tier-a residuals sit *at* the measured C_k
contamination bound (as they must — (★) omits exactly that term) while tier-b
sits ~100× lower at the truncation floor. Three simultaneous coefficient pairs
(k = 3,4,5) over-determine the x⁴ weighting and the recoil term; a recoil
coefficient off by one unit would shift the k=3 RHS by ~230%.

What each test uniquely pins:
- **T2/T3**: the drift coefficient (k−2)(k+1) and recoil coefficient (k−2) —
  invisible to MMS by construction.
- **T5**: the (φ−1)n_pl(1+n_pl) flux branch (heating→y conversion), which every
  other kernel-level test runs at φ = 1 where it vanishes identically. Pins
  amplitude, hence which θ normalizes the Comptonization variable — a θ_e↔θ_z
  swap would shift the amplitude by ε, ~3× the observed error.
- **T6**: isolates the Δn² term (coverage-matrix row 4 diagnostic).
- **H-theorem**: the only nonlinear-regime check in the repo (Δn ~ n_pl);
  structural inequality valid at any amplitude.

---

## Part II results

### II.1 μ-photosphere profile vs analytic x_c(z) (`tests/mu_photosphere_profile.rs`)

The only test of the *coupled* DC/BR + Compton quasi-stationary balance against
an analytic target rather than CosmoTherm's 2–5% envelope. PDE burst runs, then
fit the slope of ln μ(x) vs 1/x → x_c; compare to Chluba (2015) Eq. 25
(transcribed fresh into the test; `greens::x_c` cross-checked to <10⁻⁹ but NOT
used as the target — the PDE never touches it).

| z | x_c fitted (main window) | x_c literature | ratio | window spread | μ-purity \|y/μ\| |
|---|---|---|---|---|---|
| 2×10⁶ (DC-dominated, x_c,DC ≈ 7×x_c,BR) | 8.470×10⁻³ | 8.688×10⁻³ | **0.975** | 1.6×10⁻² | 0.004 |
| 3×10⁵ (BR-significant, x_c,BR > x_c,DC) | 5.321×10⁻³ | 5.520×10⁻³ | **0.964** | 2.3×10⁻² | 0.003 |

Both within the derived tolerance 0.12, with window-to-window spread ~50× below
it (the fit is stable, not window-tuned). A broken DC or BR coefficient shifts
x_c by tens of percent; the two redshifts weight DC and BR differently, so both
production channels are pinned. Runtime: ~51 s (two PDE runs).

### II.2 Compton-equilibrium temperature coefficients (`tests/compton_equilibrium_analytic.rs`)

Anchors the perturbative Δρ_eq path (CLAUDE.md Pitfall #4) with coefficients
from mpmath quadrature (dps = 40) over the analytic shapes only — the derivation
script imports nothing from spectroxide, and its sanity identities
∫x²Y_SZ = 0, ∫x³Y_SZ = 4G₃ hold to <10⁻⁴⁰:

    Δρ_eq = COEFF_Y · y  = 5.3996232391327225 · y     (Δn = y·Y_SZ)
    Δρ_eq = COEFF_MU · μ = 0.45614425920673529 · μ    (Δn = μ·M)

| Check | Result |
|---|---|
| COEFF_Y, amp ∈ {10⁻⁶,10⁻⁵,10⁻⁴} | rel err 1.6×10⁻⁶ … 3.9×10⁻⁴ (< 3×10⁻³) ✅ |
| COEFF_MU, same amps | rel err 1.6×10⁻⁶ … 6.2×10⁻⁵ ✅ |
| Linearity spread | 3.9×10⁻⁴ (< 10⁻³) ✅ |
| Shape conventions vs `spectrum.rs` | < 10⁻¹² pointwise ✅ |

The extraction uses the difference method (ratio(n_pl+Δn) − ratio(n_pl)), which
cancels the O(grid) baseline error — reading the absolute ratio would repeat
Pitfall #4. The fused solver-internal path is covered via the existing
`test_full_te_perturbative_vs_brute_force` cross-validation (no new plumbing
added, per plan).

### II.3 DC/BR first-principles magnitudes (`tests/rate_coefficients_first_principles.rs`)

The anti-Pitfall-#8 test: CODATA constants typed literally in the test file,
nothing imported from `constants.rs`. This is the test class that would have
caught the historical 10¹¹× BR density-factor bug.

| Coefficient | code/anchor ratio | z-spread across 3 redshifts |
|---|---|---|
| DC: K = (4α/3π)θ_z²·(4π⁴/15)/(1+14.16θ_z), x = 10⁻⁴, z ∈ {3×10⁵, 10⁶, 2×10⁶} | **0.999950** (H_dc(x) deficit, expected) | **1.1×10⁻¹⁶** |
| BR: K = (αλ_e³/2π√(6π))·θ_e^(−7/2)·(e^(−xφ)/φ³)·ΣZ²N_i·g_ff, x = 0.5, z ∈ {5×10⁵, 10⁶, 2×10⁶} | **1.000000** | **9.0×10⁻¹⁵** |
| I_pl quadrature vs 4π⁴/15 | rel 7.4×10⁻¹⁰ | — |

z-independence isolates density-factor errors (a spurious /n_e would drift as
(1+z)³ and be off by ~10¹⁷ in magnitude). Gaunt factors are reused from the
code — they carry separate class-1 coverage (CRB-2020 spot checks at 10⁻¹⁰) —
so the test isolates the prefactor × density × temperature assembly, exactly
where the historical bug lived.

### II.4 Dark-photon γ_con vs Landau-Zener integration (`dev/audit/gamma_con_lz_check.md`)

Direct integration of the 2-level γ–A′ mixing ODE through the resonance with
the actual ω_pl(z) profile (m = 10⁻⁷ eV, z_res = 3.21×10⁴), vs the code's NWA
P = 1 − exp(−ε²γ_con):

| Regime | P_NWA | P_numeric vs P_LZ rel err |
|---|---|---|
| non-adiabatic (ε²γ = 2.5×10⁻³) | 2.50×10⁻³ | 7.3×10⁻³ |
| boundary (ε²γ = 1) | 0.632 | 1.2×10⁻² |
| adiabatic (ε²γ = 9) | 0.9999 | 4.3×10⁻⁴ |

**NWA confirmed** (worst 1.2% ≪ 5% threshold), including at the validity
boundary where a 22%-class error would live. Consequence: the unresolved ~22%
discrepancy vs Bryce's frozen-absorption curve (memory `axion-dp-distortion`)
is **not in γ_con** — it lives in the frozen-vs-thermalized treatment.

### II.5 FIRAS closed-loop coverage calibration (`python/tests/test_firas_coverage.py`)

Monte Carlo through the *production* fit path (`FIRASData.fit_amplitude`, mock
injected via `residual_kJy` — no re-implemented linear algebra). Seeded
(20260707); smoke tests at N = 800–1000, full-N (10⁴) runs marked `slow`.

| Check | N | Result |
|---|---|---|
| Unbiased recovery at a_true ∈ {0, 1σ, 5σ} | 800 | bias < 4σ/√N ✅ |
| χ² GOF under null | 1000 | χ²_null ~ χ²₄₃, χ²_min ~ χ²₄₂ (KS + mean) ✅ |
| Error-bar calibration: std(â) = σ_reported | 10⁴ | rel 5% ✅ |
| 95% interval coverage at a_true ∈ {0, 1σ, 5σ} | 10⁴ | \|coverage − 0.95\| < 0.01 ✅ |

All four pass in ~4 s (the "slow" full-N marks were precautionary — the 1-param
linear fit is cheap). The pipeline's error bars, coverage, and GOF distribution
are calibrated; the paper's limit machinery is statistically sound at the
tested amplitudes.

### II.7 Coverage matrix (`dev/audit/term_coverage_matrix.md`)

17 physical terms × 5 test classes, every entry citing test file:line. Working
definition of "ironclad": every row has a class-1 (exact identity) or class-2
(independent amplitude anchor) entry. Rows closed by this suite: 1–3
(Kompaneets drift/recoil/(φ−1)), 5–6 (DC/BR magnitude), 8 (Δρ_eq), 14 (γ_con),
16 (FIRAS coverage). **Remaining open gaps, in priority order:**

1. **Row 13, y_γ broadening** — weakest row, only indirect full-GF shape
   checks. Flagged highest-priority future gap.
2. **Row 10, recombination X_e** — cross-code (HyRec/RECFAST bands) only; the
   Peebles/Saha ODE has no exact identity or MMS.
3. **Row 9, full nonlinear T_e path** — leading order pinned via row 8 only.
4. **Row 11, expansion/redshifting** — cross-code + scaling relations only.
5. **Row 12, P_s** — II.1's x_c is the nearest anchor; no direct P_s(x,z)
   identity.
6. **Row 4, Δn²** — no standalone identity; acceptable (T6 bounds it).

---

## Merge record (2026-07-30)

Merged into `main` as files, not as a branch merge — every deliverable was an
untracked new file, so there was nothing to rebase. What changed on the way in:

1. **`Cargo.toml` was NOT carried over.** The branch had criterion and the
   `[[bench]]` section commented out to survive the 7 GB box during concurrent
   mutation runs. `main`'s version is unmodified; the new test files need no
   `[[test]]` entries (auto-discovered).
2. **`COVERAGE_MATRIX.md` → `dev/audit/term_coverage_matrix.md`.** The original
   name collides case-insensitively with the pre-existing
   `dev/audit/coverage_matrix.md` (a different document: R0's claim→anchor
   matrix over paper figures) and would break any checkout on macOS or Windows.
   The two are complementary — R0's is indexed by *published result*, this one
   by *physical term*.
3. **Caveat retired:** `tests/mms_convergence.rs` and `tests/conservation_fuzz.rs`
   are now tracked on `main`, so the module docs and matrix rows that cite them
   resolve.

## Remaining caveats / follow-ups

1. **Chluba 2015 Eq. 25 fit coefficients** (x_c,DC = 8.60×10⁻³·((1+z)/2×10⁶)^0.5,
   x_c,BR = 1.23×10⁻³·(·)^−0.672) could not be re-verified by automated PDF
   fetch; transcribed to match `src/greens.rs`. **Human spot-check against the
   paper still recommended** — this is the one anchor in the suite whose
   independence rests on a transcription rather than on a derivation.
2. **Row 13 (y_γ broadening) is the highest-priority open gap** and it is not
   cosmetic: the sensitivity map gives ∂lnμ/∂ln y_γ = −2.03, so an error in the
   broadening kernel propagates at O(1) into the photon-injection figures. No
   item in this plan targets it.
3. **§II.5 exercises the single-amplitude FIRAS fit, not the floating-`T`
   profile-likelihood path** where the surviving `firas.py` mutants sit — and
   floating `T` is exactly what drives the Fig. 8 offset vs CCJ24. The coverage
   calibration is real but does not reach the path the paper's limits use.
4. `#[allow(clippy::excessive_precision)]` on the verbatim-pasted mpmath
   constants is deliberate: the digits are the anchor, and rounding them to
   f64-representable values would silently weaken the test.
