# Module Audit B1-dcbr: `double_compton.rs` + `bremsstrahlung.rs`

**Auditor:** physics-inquisitor (adversarial, anchored outside code)
**Date:** 2026-07-03
**Commit:** a46baba (branch `main`)
**Verdict:** NO CONFIRMED BUGS. One stale test-comment (LOW). Three documented conventions/assumptions.

Protocol: emission coefficients re-derived from primary references *before* reading
code numeric output; numerical spot-checks only afterward. Source files NOT edited.

---

## 1. References used

- **DC:** Lightman (1981) ApJ 244, 392; Thorne (1981) MNRAS 194, 439; Chluba, Sazonov
  & Sunyaev (2007) A&A 468, 785 [arXiv:0705.3033] (θ_e relativistic corrections);
  Chluba & Sunyaev (2012) MNRAS 419, 1294, Eqs. 8, 10–13.
- **BR:** Brussaard & van de Hulst (1962) Rev.Mod.Phys. 34, 507 (Born-approx Gaunt);
  Chluba & Sunyaev (2012) Eq. 14; Chluba, Ravenni & Bolliet (2020) MNRAS 492, 177 (BRpack).
- **Near-cancellation:** CLAUDE.md pitfall #5; Planck identity dn_pl/dx = −n_pl(1+n_pl).

---

## 2. Independent derivations

### 2.1 Detailed-balance / source-term structure (both processes)
Emission ∝ (1+n) [stimulated], absorption ∝ n, equilibrium at n = n_pl(x_e),
x_e = x·T_z/T_e = x/ρ_e. Net rate:

  dn/dτ = K_c/x³ · e^{−x_e}[(1+n) − n e^{x_e}]
        = K_c/x³ · [e^{−x_e} − n(1 − e^{−x_e})].

Code writes `K/x³ · [1 − n(e^{x_e}−1)]` with e^{−x_e} folded into K (BR: explicit
`exp(-x*phi)`; DC: carried by H_dc). Algebraically identical:
K_c e^{−x_e}/x³·[1 − n(e^{x_e}−1)] = K_c/x³[e^{−x_e} − n(1−e^{−x_e})]. ✓
Equilibrium fixed point 1 − n(e^{x_e}−1)=0 ⇒ n = 1/(e^{x_e}−1) = n_pl(x_e). ✓ **VERIFIED.**

### 2.2 Near-cancellation expansion coefficient (pitfall #5)
Let ρ_e = 1+ε. n_pl(x/ρ_e) − n_pl(x) ≈ n_pl'(x)·(x/ρ_e − x)
= n_pl'(x)·x(1−ρ_e)/ρ_e = −n_pl'(x)·x(ρ_e−1)/ρ_e.
Planck identity n_pl'(x) = −n_pl(1+n_pl) ⇒
  n_pl(x/ρ_e) − n_pl(x) ≈ **x(ρ_e−1)/ρ_e · n_pl(1+n_pl)**. ✓
Matches the coefficient quoted in both module docstrings (`dc_rhs`, `br_rhs`) and used
in `solver.rs::compute_emission_rates` when |ρ_e−1| < 0.01. **VERIFIED.**

### 2.3 DC normalization I4_pl
g_dc(x→0) → I4_pl = ∫₀^∞ x⁴ n_pl(1+n_pl) dx. Integrate by parts with n(1+n)=−dn/dx:
∫x⁴(−dn/dx)dx = 4∫x³n dx = 4 G₃ = 4π⁴/15 ≈ 25.98. Code: `I4_PLANCK = 4.0*G3_PLANCK`. ✓

---

## 3. Equation ↔ code mapping

### double_compton.rs
| Item | Reference | Code | Verdict |
|---|---|---|---|
| Prefactor 4α/3π · θ_z² | Lightman81 / CS2012 | L63,70 | ✓ |
| Normalization I4_pl = 4G₃ = 4π⁴/15 | ∫x⁴n(1+n)dx | L33,102 | ✓ (25.98) |
| Rel. correction 1/(1+14.16 θ_z) | CSS2007 | L34 | see F2 |
| H_dc = e^{−2x}[1+3x/2+29x²/24+11x³/16+5x⁴/12] | CS2012 Eq.13 | L51 | ✓ Horner form correct |
| H_dc(0)=1, H_dc→0 (x>100 guard) | limits | L48,143 | ✓ |
| Source K_DC/x³·[1−n(e^{x_e}−1)] | detailed balance §2.1 | L105 | ✓ |
| Dimensionless (one-body, no density factor) | pitfall #8 | — | ✓ |
| Heating integral /(4G₃θ_z) | consistent w/ BR | L133 | ✓ |

### bremsstrahlung.rs
| Item | Reference | Code | Verdict |
|---|---|---|---|
| Prefactor αλ_e³/(2π√(6π)), λ_e=h/mc | CS2012 Eq.14 | L23 | ✓ = 3.821e-39 m³ |
| temp_factor θ_e^{−7/2} e^{−xφ}/φ³ | CS2012/Burigana | L137 | ✓ |
| Species sum Z²={1,4,1} for {H⁺,He²⁺,He⁺} | — | L170 | ✓ |
| He⁺ = (y_he_i − y_he_ii) | Saha layers | L168 | ✓ |
| Gaunt 1+softplus((√3/π)(ln(2.25/(xZ))+½lnθ_e)+1.425) | BRpack/CosmoTherm | L64 | see F4 |
| Gaunt Z linear in log, Z² in sum | Coulomb η_Z | L44,170 | ✓ |
| Low-x limit g_ff ≈ (√3/π)ln(2.25θ_e^{1/2}/x) | Born (BvdH62) | L64 | ✓ diverges log |
| High-x floor g_ff→1 | softplus→0 | L64,90 | ✓ |
| Dimensionless (two-body, one N_e cancels) | pitfall #8 | L109 | ✓ [m³]·[1/m³] |
| Source K_BR/x³·[1−n(e^{x_e}−1)] | detailed balance §2.1 | L358 | ✓ |
| Hardcoded consts (√6π, √3/π, ln2.25, ln1.125) | test | L644 | ✓ |

---

## 4. Numerical spot-checks (after derivation)

- **BR_PREFACTOR** = α·(h/mc)³/(2π√6π) = **3.821e-39 m³** (matches `test_br_hardcoded_constants`).
- **DC/BR ratio** (T_e=T_z, ΛCDM n_H0=0.189/m³): z=4e5 x=0.1 → 1.73 (crossover z≈3–4×10⁵,
  matches Danese & de Zotti); **z=1e6 x=0.1 → 17.06** (matches expected μ-era ratio).
- K_DC(x=1,θ_z=4.6e-5) ≈ 9.5e-11; K_BR(x=0.1,z=1e5) ≈ 1.9e-9. Both physically sane.

---

## 5. Findings triage

**F1 (LOW — false alarm on code, stale test comment).** `bremsstrahlung.rs` L692 comment
claims "BR_PREFACTOR ≈ 6.1e-40 m³"; actual value is 3.82e-39 m³ (~6.3× off). The
companion component estimates in the same block are also wrong: θ_z^{−7/2}(4.6e-5) is
1.5e15 not "2.1e16", g_ff≈1.9 not 3, and true K_BR(x=0.1,z=1e5)≈1.9e-9 not "6.6e-9"
(compensating errors). The **assertion itself is fine** (OOM band 1e-12…1e-4 and ratio
0.01…100×), so the test still guards against the historical /n_e bug. Only the explanatory
comment is misleading. Recommend correcting the comment; no code change.

**F2 (documented convention).** DC relativistic correction 1/(1+14.16θ_z) evaluated at
θ_z, not θ_e. Docstring (L26–31) acknowledges θ_e is cleaner; effect <1% for θ_z≲1e-3
even at |ρ_e−1|~0.1. The 14.16 coefficient is from CSS2007 and cannot be re-derived
here; accepted as literature input. Not a bug.

**F3 (assumption, benign).** DC emission uses the Planck value I4_pl for the photon-number
integral rather than ∫x⁴ n(1+n)dx over the *distorted* spectrum. Correction is O(Δρ/ρ)
~ 1e-8 for ΛCDM; entirely negligible. Standard CS2012 soft-photon approximation.

**F4 (unverifiable vs primary — WARNING).** The softplus Gaunt form with offset **1.425**
and the 0.5·ln(θ_e) Born↔classical interpolation are a CosmoTherm private-communication
fit, NOT printed in CRB2020. Limits are correct (low-x → Born log, high-x → 1) but the
transition-region coefficient cannot be checked against a published equation. Only
validated by hand-calc magnitude tests. Carried forward from 2026-05-28 audit.

---

## 6. Cross-check with prior audits
Consistent with 2026-05-28 per-file notes and 2026-07-03 kompaneets B1 audit (which
verified the DC/BR IMEX coupling, backward-Euler diagonal positivity, and the coupled
production source term in `solver.rs`/`kompaneets.rs` — those live outside these two
modules and are not re-audited here).
