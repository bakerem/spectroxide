# B1 Audit: `src/recombination.rs`

**Scope:** Peebles three-level-atom (TLA) hydrogen recombination + Saha helium.
**Primary sources used (fetched before reading code output):**
- Seager, Sasselov & Scott 1999, ApJL 523, L1 (astro-ph/9909275) — RECFAST TLA ODE.
- Chluba & Thomas 2011, MNRAS 412, 748 (arXiv:1011.3758) — fudge-factor recalibration.
- Standard Saha/Boltzmann detailed-balance derivation (independent, done by hand below).

## 1. Derivation from primary sources (pre-registered, before reading code numerics)

SSS99 hydrogen ODE (their eq. 1, matter temperature T_M, θ ≡ hν₂ₛ,₁ₛ = 10.2 eV Lyα energy):

```
dx_p/dz = C_r · [x_e x_p n_H α_B − β_B(1−x_p) e^{−hν_{2s,1s}/kT_M}] / [H(z)(1+z)]

C_r = [1 + K Λ_2s1s n_H(1−x_p)] / [1 + K(Λ_2s1s+β_B) n_H(1−x_p)]
K   = λ_Lyα³/(8πH(z)),  λ_Lyα = 1215.67 Å
β_B = α_B(T_M)·(2π m_e k T_M/h²)^{3/2}·e^{−E_2s/kT_M},   E_2s = 13.6/4 = 3.4 eV   [NOT 10.2 eV]
Λ_2s1s = 8.22458 s⁻¹
α_B(T) = F·10⁻¹⁹·a t^b/(1+c t^d) m³/s,  t=T/10⁴K,  a=4.309,b=−0.6166,c=0.6703,d=0.5300,  F=1.14 (RECFAST)
```

SSS99 explicitly warns: *"T_M and ν_2s are used here... incorrectly using T_R or ν_2p will cause a small but important difference"* — i.e. (a) rates go at **matter** temperature, not radiation temperature, and (b) β's Boltzmann factor uses **E_2s = 3.4 eV**, while the ODE's `(1−x_p)` term additionally carries a **separate** factor e^{−hν_2s,1s/kT_M} with hν_2s,1s = **10.2 eV** (Lyα, the 1s→2 transition energy) to convert the n=2 population into an n_1s-normalized quantity. This is the classic "3.4 eV vs 10.2 eV" trap named in the task brief.

**Independent check that the two exponentials are consistent:** multiplying them,
exp(−E_2s/kT)·exp(−E_Lyα/kT) = exp(−(3.4+10.2)/kT) = exp(−13.6 eV/kT) = exp(−E_H/kT),
i.e. the full Rydberg energy. So

```
β_B(1−x_p)e^{−hν_2s,1s/kT_M} = α_B n_H · (n_Q/n_H) e^{−E_H/kT_M} · (1−x_p) = α_B n_H · X_S²/(1−X_S) · (1−x_p)
```

using the ground-state (full-Rydberg) Saha relation X_S²/(1−X_S) = (n_Q/n_H)e^{−E_H/kT}. This is exactly the algebraic identity behind the "Saha-subtracted" rewrite
`dX/dz = C α_B n_H/(H(1+z))·[X² − X_S²(1−X)/(1−X_S)]` used for cancellation control (CLAUDE.md pitfall #5) — **the two forms are provably identical**, not just numerically similar.

Chluba & Thomas 2011 (§III.2.2): RECFAST's F=1.14; *"the best fitting fudge factor would be F=1.126, with relative errors reaching 0.2%."*

## 2. Equation ↔ code mapping

| Item | Primary-source value/form | Code (`src/recombination.rs`) | Verdict |
|---|---|---|---|
| α_B fit form/coeffs | a=4.309,b=−0.6166,c=0.6703,d=0.5300 | `alpha_recomb`: identical (l.178-182) | ✅ correct |
| Fudge factor F | RECFAST 1.14; Chluba&Thomas best-fit **1.126** | `f = 1.125` (l.180), cites C&T 2011 | ⚠️ LOW: off by 0.001 (0.09%), sub-% w.r.t. α_B; effectively a rounding slip in the cited digit, not a physics error |
| β_B exponent energy | E_2s = 13.6/4 = 3.4 eV (in the *standalone* β formula) | `E_ION_N2 = E_RYDBERG_EV/4.0` used in `beta_ion` (l.194-199) | ✅ correct — uses 3.4 eV, not the 10.2 eV Lyα energy; the classic bug is **absent** |
| "Extra" 10.2 eV Boltzmann bridge factor | separate e^{−hν_Lyα/kT} multiplying β(1−x) in the raw ODE | Never appears explicitly — instead the code jumps straight to the Saha-subtracted form `X_h²−X_S²(1−X_h)/(1−X_S)` (l.264-278) | ✅ correct by the identity derived in §1; the missing explicit factor is not an omission, it's folded into the (correct) full-Rydberg H Saha term |
| Λ_2s1s | 8.22458 s⁻¹ (Labzowsky et al./RECFAST value) | `LAMBDA_2S1S = 8.2245809` (constants.rs) | ✅ correct to 7 sig figs |
| K Sobolev factor | λ_Lyα³/(8πH) | `k_h = LAMBDA_LYA.powi(3)/(8π·h)` (l.230) | ✅ correct; λ_Lyα=1215.670 Å matches `LAMBDA_LYA=1.21567e-7` m |
| Peebles C factor | (K Λ + esc)/(K Λ + esc + Kβ) form ⇒ (rate_esc+Λ)/(rate_esc+Λ+β) after dividing by K n_1s | `peebles_c`: `rate_down=(esc+Λ)`, `C=rate_down/(rate_down+rate_ion)` (l.224-249) | ✅ algebraically identical to SSS99 form |
| **T in α_B, β** | **T_M (matter)**, explicitly warned against T_R | Uses `t = cosmo.t_cmb*(1+z)` = **T_radiation** everywhere (`peebles_rhs` l.265, `beta_ion` docstring l.192-193 self-admits this) | ⚠️ **CONVENTION MISMATCH vs. primary source**, but disclosed in-code (l.258-263) and bounded: Compton coupling keeps T_m≈T_γ to ≲1% for z≳800 where the TLA rates matter; the code accepts 1-5% deviation from RECFAST as its stated accuracy target. Not a silent bug — flag as accepted/documented approximation, and one of the two "confusable" issues the task brief pre-warned about. |
| Saha H (statistical weight) | g_e g_p/g_H ≈ 1 (standard convention, hyperfine ignored) | No prefactor in `saha_hydrogen` (l.161) | ✅ correct, matches RECFAST/Peebles convention |
| Saha He II (54.4 eV) | g ratio = g(He²⁺)g(e)/g(He⁺) = 1·2/2 = 1 | no prefactor (l.106) | ✅ correct |
| Saha He I (24.6 eV) | g ratio = g(He⁺)g(e)/g(He) = 2·2/1 = 4 | `4.0 *` prefactor (l.133) | ✅ correct |
| He ionization energies | 54.4178 eV, 24.5874 eV (NIST) | `E_HE_II_ION_EV=54.4178`, `E_HE_I_ION_EV=24.5874` | ✅ correct (real atomic values, not naive 4×13.6 scaling) |
| n_e for He Saha | RECFAST total-free-electron form n_e = n_H+2n_He (He²⁺) / n_H+n_He (He⁺), not n_e=y·n_He | matches (l.103, l.130), with explicitly quantified ≲7%/≲4% bias vs the fully self-consistent y_II=0 case | ✅ correct convention, error bound documented and justified |
| Saha switch criterion | X_e crosses ~0.99 (standard TLA-activation threshold) | `find_saha_switch`: X<0.99 (l.312) | ✅ standard choice |
| Integration scheme | (SSS99 doesn't specify a numerical scheme) | Heun (RK2) predictor-corrector, O(dz²) (l.280-303), upgraded from forward Euler per in-code note | ✅ reasonable, no accuracy concern flagged |
| Cache/interpolation | — | Binary-search-free direct indexing on a uniform table + linear interpolation (l.445-469) | ✅ verified: `test_recombination_history_matches_uncached` (rel_err<1%), `..._monotonic`, `..._interpolation_smooth` all present and non-trivial |
| Y_p convention | Y_p mass fraction ⇒ n_He/n_H = Y_p/(4(1−Y_p)) | `F_HE = Y_p/(4(1−Y_p))` (constants.rs:83) | ✅ matches CLAUDE.md-documented convention |

## 3. External-anchor comparison (literature milestones, no CLASS available in this environment — not importable; noted as a gap, see §5)

Independent literature values (RECFAST-family codes, ΛCDM-like params): X_e(1100)≈0.14–0.15, X_e(800)≈3×10⁻³, freeze-out X_e(z≲200)≈2–4×10⁻⁴ scaling ∝(Ωbh)⁻¹.

Code's own `Cosmology::default()` (T0=2.726, h=0.71, Ωb=0.044, Yp=0.24) reproduces, per existing tests (`test_xe_vs_recfast_milestones`, run and inspected — not re-derived from scratch here since these are genuine external literature anchors, not code-derived):
- X_e(1100) ∈ (0.10, 0.20) — consistent with ~0.14 anchor.
- X_e(800) ∈ (5e-4, 0.01) — consistent with ~3e-3 anchor.
- X_e(200) ∈ (1e-4, 2e-3) — consistent with ~2-4e-4 freeze-out anchor, band is generous (±5x) but centered correctly.

These test tolerances are wide (factor ~2-5), appropriate given the disclosed T_rad-vs-T_matter and no-fudge-refinement approximations — but they are wide enough that a genuine ~30% bug would slip through undetected. This is a **test-strength gap**, not a physics bug (see §5 recommendation).

## 4. Limiting cases checked

- z→∞ (z>8000): X_e→1+2f_He exactly (He fully double-ionized); `test_fully_ionized_high_z` confirms to <1%. ✅
- z→0 asymptote: freeze-out plateau X_e~few×10⁻⁴, monotonic decrease enforced by both direct and cached paths (`test_recombination_history_monotonic`, `test_recombination_physical_values`). ✅
- Saha↔Peebles matching at switch: `find_saha_switch` returns z+1 the first time Saha X<0.99 scanning downward from 1800; continuity at the switch not explicitly unit-tested by itself, but `test_recombination_history_interpolation_smooth` at z=1200 (well past the switch) and the monotonicity test spanning both regimes provide indirect coverage.
- No hard jump at z=200 (frozen-regime boundary of the table's use elsewhere in the solver): explicitly tested (`test_recombination_physical_values`, ratio∈(0.5,2) at z=199/201). ✅

## 5. Triage summary

**Confirmed bugs:** none.

**Convention mismatches (documented, bounded):**
1. α_B, β_B evaluated at T_radiation instead of T_matter (SSS99 explicitly requires T_M). Justified in-code by Compton-coupling argument (≲1% at z≳800); consistent with the code's stated 1-5% RECFAST-agreement target. Recommend: if the "X_e-swap experiment" in Plan B4 is executed, use a CLASS/HyRec table (which correctly separates T_m) as the anchor rather than re-deriving T_m internally here — that will directly quantify how much of the residual is due to this convention vs. the missing fudge refinement.

**False alarms refuted:**
2. The 3.4 eV vs 10.2 eV substitution "bug" the task brief warns about is **not present**: `E_ION_N2=3.4 eV` is correctly used only in the standalone β_B Boltzmann factor. The code never separately multiplies by an extra Lyα-energy exponential — instead it uses the algebraically-equivalent Saha-subtracted form, which folds in the missing exp(−10.2 eV/kT) via the full-Rydberg hydrogen Saha relation. Verified by hand in §1.
3. Fudge factor F=1.125 vs. Chluba & Thomas's actual best-fit 1.126: negligible (0.09% on F, propagates to <0.1% on α_B). Cosmetic citation-precision nit only.

**Gaps (not bugs, recommend closing under B4/B0):**
4. No CLASS/HyRec import available in this environment to do a numeric X_e(z) grid comparison as the plan requested (`import classy` unavailable) — deferred to Plan B4's "X_e-swap experiment," which should use a pre-tabulated CLASS or HyRec/CosmoRec output file rather than requiring the library at audit time.
5. `test_xe_vs_recfast_milestones` / `test_freeze_out` tolerances are wide enough (2-5×) that they would not catch a real order-unity-fraction bug in `beta_ion` or `peebles_c`; tighten once a CLASS/HyRec reference table is available (item 4).

## 6. Verified correct (full list)

α_B fit and coefficients; F fudge-factor citation (to 0.1%); β_B exponent (3.4 eV, not 10.2 eV); Saha-subtraction identity (independently re-derived, provably equivalent to the primary-source raw ODE); Λ_2s1s value; Sobolev K factor; Peebles C-factor algebra; H-Saha statistical weight (=1); He II/He I Saha statistical weights (1 and 4); He ionization energies (real NIST values); n_e convention for He Saha (RECFAST total-electron form, with quantified small bias); Saha switch threshold; Heun (RK2) integration scheme; cache/interpolation against uncached ODE; Y_p mass-fraction convention; z→∞ and freeze-out limiting cases; no discontinuity at internal regime boundaries.
