# Module audit B1: `electron_temp.rs` (+ `full_te` coupling in `solver.rs`)

**Date:** 2026-07-03
**Auditor:** physics-inquisitor (independent context)
**Scope:** `src/electron_temp.rs`, the production T_e update in
`solver.rs::update_temperatures` (lines 860-1065), and the equilibrium-ratio
helper `spectrum.rs::compton_equilibrium_ratio` (lines 121-146).
**Primary references:** Zeldovich & Levich (1969); Weymann (1965); Chluba &
Sunyaev (2012) MNRAS 419, 1294; Seager, Sasselov & Scott (1999).

Method: every coefficient re-derived from scratch **before** reading numeric
output; numerical spot-checks run only to confirm the derivation.

---

## 1. Independent derivation

### 1.1 Compton-equilibrium electron temperature (Zeldovich-Levich)

The electron temperature at which net Compton energy exchange with the photon
field vanishes:

    T_e^eq = (h/4k) · ∫ν⁴ n(1+n) dν / ∫ν³ n dν.

In the code's dimensionless frequency x = hν/kT_z (ν = (kT_z/h)x):

    ∫ν⁴ n(1+n)dν = (kT_z/h)⁵ I₄,   I₄ ≡ ∫x⁴ n(1+n)dx
    ∫ν³ n dν     = (kT_z/h)⁴ G₃,   G₃ ≡ ∫x³ n dx

  ⇒  ρ_eq ≡ T_e^eq/T_z = I₄/(4 G₃).      ✓ matches `spectrum.rs:124`, docstring.

**Planck fixed point.** Using dn_pl/dx = −n_pl(1+n_pl):
I₄^pl = −∫x⁴ dn_pl/dx dx = 4∫x³ n_pl dx = 4 G₃^pl (boundary term vanishes),
so ρ_eq = 1 for a Planck spectrum. ✓

### 1.2 Perturbative expansion (pitfall #4)

Write n = n_pl + Δn.

    G₃ = G₃^pl + ΔG₃,        ΔG₃ = ∫x³ Δn dx
    I₄ = I₄^pl + ΔI₄ + ∫x⁴Δn² dx,   ΔI₄ ≡ ∫x⁴(1+2n_pl)Δn dx

The linearisation n(1+n) → n_pl(1+n_pl) + (1+2n_pl)Δn + Δn² is exact; only Δn²
is dropped. With I₄^pl = 4G₃^pl:

    ρ_eq = [4G₃^pl + ΔI₄ + O(Δn²)] / [4G₃^pl(1 + ΔG₃/G₃^pl)]
         ≈ [1 + ΔI₄/(4G₃^pl)]·[1 − ΔG₃/G₃^pl]
         ≈ 1 + ΔI₄/(4G₃^pl) − ΔG₃/G₃^pl.

  ⇒  **Δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃.**   The coefficient of the −ΔG₃/G₃ term is
  exactly **1**, because the baseline ρ_eq^pl ≡ 1. The docstring's
  "×(I₄/(4G₃))" factor (electron_temp.rs:8) equals 1 to the working order, and
  the code correctly uses coefficient 1.

**Dropped terms:** (i) ∫x⁴Δn²/(4G₃) and (ii) the cross term ΔI₄·ΔG₃/G₃², both
O(Δn²) ~ 10⁻¹⁰ for ΛCDM (Δn ~ 10⁻⁵) vs the O(10⁻⁵) linear signal → 5 orders
down. Justified. The switch to the exact form at |ΔG₃/G₃| > 0.1 covers the
regime where Δn² matters (strong dark-photon depletion). Correct threshold.

### 1.3 Injection source normalisation

`heating_rate` returns q_rel = d(Δρ_γ/ρ_γ)/dt [1/s] (verified
energy_injection.rs:708-711). Compton energy balance in steady state — the
extra electron temperature the injection sustains transfers energy to photons
at exactly the injection rate:

    q_rel = (1/ρ_γ)dρ_γ/dt|_C = 4 θ_z (ρ_e − ρ_eq)/t_c
  ⇒ δρ_inj ≡ (ρ_e − ρ_eq) = q_rel · t_c / (4 θ_z).   ✓ solver.rs:939.

Dimensions: [1/s]·[s]·1 = dimensionless. ✓ The "4" is the 4 in Δρ/ρ = 4Δθ/θ.

### 1.4 Compton coupling rate R

t_C/t_γ with 1/t_γ = (8σ_T ρ_γ)/(3 m_e c)·n_e/(n_e+n_H+n_He) (Weymann 1965):

    R = t_C/t_γ = (8/3)·ρ_γ/(m_e c² n_tot) = (8/3)·ρ̃_γ/α_h,
    ρ̃_γ = ρ_γ/(m_e c² n_e),  α_h = n_tot/n_e = (n_e+n_H+n_He)/n_e.

Verified ρ̃_γ = KAPPA_GAMMA·θ_z⁴·G₃/n_e algebraically:
KAPPA_GAMMA = 8π/λ_e³ (λ_e = h/m_e c) gives
KAPPA_GAMMA·θ_z⁴·G₃/n_e = 8π(kT_z)⁴G₃/(h³ m_e c⁵ n_e) = ρ_γ/(m_e c² n_e). ✓
(ρ_γ = (8π/c³h³)(kT_z)⁴G₃.) Matches solver.rs:968-970, constants.rs:124.

### 1.5 Adiabatic cooling

Non-relativistic matter: T_m ∝ (1+z)² ⇒ dlnT_m/dt = −2H; radiation
dlnT_z/dt = −H. Hence dρ_e/dt|_ad = ρ_e(dlnT_m − dlnT_z)/dt = −H ρ_e, i.e.
in Thomson time dρ_e/dτ = −(H t_C)ρ_e. **Coefficient 1, not 2.** ✓ (lambda_htc,
solver.rs:971, 1036).

### 1.6 Quasi-stationarity

R ≈ (8/3)ρ̃_γ/α_h. ρ̃_γ = ρ_γ/(n_e m_e c²) ~ θ_z·(n_γ/n_b)/x_e ~ 0.46(1+z)/x_e,
so R ~ 10⁵–10⁶ at z ≳ 10⁵. t_γ = t_C/R ≪ t_C ≪ H⁻¹ ⇒ ρ_e relaxes to
ρ_source = ρ_eq + δρ_inj within a step (R·Δτ ≫ 1). Justifies quasi-stationary
treatment in the μ/y eras; the backward-Euler form degrades smoothly to
adiabatic ρ_e ∝ (1+z) post-recombination (R·Δτ ≪ 1). ✓

### 1.7 Backward Euler discretisation

ODE: dρ_e/dτ = R[(ρ_source − H_dcbr(ρ_e)) − ρ_e] − λρ_e, with
H_dcbr = H₀ + H'(ρ_e − ρ^n). Implicit solve:

    ρ^{n+1} = [ρ^n + Δτ·R(ρ_source − H₀ + H'ρ^n)] / [1 + Δτ(R(1+H') + λ)].

Matches solver.rs:1034-1036 (numerator/denominator) term-for-term. ✓
The bordered-Newton path (kompaneets.rs, via RhoECache) uses the same frozen
coefficients — consistent (previously verified 2026-04-11).

---

## 2. Equation ↔ code map

| Quantity | Code | Verdict |
|---|---|---|
| ρ_eq = I₄/(4G₃) | spectrum.rs:145 | ✓ correct |
| ΔG₃ = ∫x³Δn | solver.rs:906 (`x3·dn_mid·dx`) | ✓ |
| ΔI₄ = ∫x⁴(1+2n_pl)Δn | solver.rs:907 (`x3·x_half·(2n_pl+1)·dn_mid·dx`) | ✓ |
| Δρ_eq = ΔI₄/(4G₃)−ΔG₃/G₃ | solver.rs:927 (coeff 1) | ✓ |
| exact ρ_eq at \|ΔG₃/G₃\|>0.1 | solver.rs:924-925 | ✓ |
| δρ_inj = q_rel·t_c/(4θ_z) | solver.rs:937-939 | ✓ |
| R = (8/3)ρ̃_γ/α_h | solver.rs:968-970 | ✓ |
| ρ̃_γ = κ_γθ_z⁴G₃/n_e | solver.rs:969 | ✓ |
| λ = H·t_C (adiab.) | solver.rs:971, 1036 | ✓ coeff 1 |
| Backward Euler | solver.rs:1034-1036 | ✓ |
| Analytic G₃^pl norm | uses `G3_PLANCK`=π⁴/15 (6.4939…) | ✓ (1 ULP) |
| Grid quadrature | midpoint-x + averaged integrand, both integrals same scheme; dx[i-1],x_half[i-1],x_half_cubed[i-1] (grid.rs:178-179) | ✓ consistent |

**Key correctness point (pitfall #4).** Production uses the *analytic* constant
`G3_PLANCK` in the denominators and subtracts the Planck baseline analytically
(I₄^pl ≡ 4G₃^pl), so the O(dx²) quadrature error cancels in the physical signal.
Verified numerically: for a pure BE-μ input the perturbative Δρ_eq scales as
O(μ²) (−8.8×10⁻⁸ at μ=10⁻³, −8.7×10⁻⁶·μ leakage), i.e. it correctly returns ≈0
(analytic truth: ρ_eq=1 for any BE-μ). No spurious first-order feedback in the
μ-era.

---

## 3. Findings

### F1 — TEST-VALIDITY BUG (non-production): `test_equilibrium_for_bose_einstein`
`electron_temp.rs:78-146`, assertions at **lines 122-127 and 130-144**, and the
docstring reasoning at **lines 78-83**.

The test asserts `ρ_e > 1.0` for μ>0, with the docstring rationale "μ>0 →
spectrum harder than Planck → ρ_e>1." **This is physically wrong.** For any
Bose-Einstein spectrum n = 1/(e^{x+μ}−1), n(1+n) = −dn/dx, so
I₄ = −∫x⁴ dn/dx dx = 4G₃ **identically** ⇒ ρ_eq = 1 **exactly, for all μ**. The
Compton-equilibrium temperature of a BE spectrum is T_z regardless of μ (a BE
distribution is the Kompaneets stationary state).

Numerical confirmation (code's own quadrature, log grid 1e-4..50, N=10000):
ρ_eq−1 = **2.15×10⁻⁶ for μ = 0, 1e-4, 1e-3, 5e-3 — identical to that precision,
i.e. completely μ-independent**, and it converges to 0 as O(dx²)
(5.4e-5→2.2e-6→8.6e-8→5.4e-9 for N=2000→200000). The "ρ_e>1" assertion passes
only because the discretization artifact happens to be positive; it measures
grid error, not physics. The "larger μ → larger ρ_e" check (lines 130-144)
likewise passes on noise (values equal to 2e-6, compared with `>=` and 1e-9
slop).

The primary assertion (lines 116-120, `rel < 1e-6`) compares `te.rho_e` to a
re-implementation of the *same* trapezoidal integral in the test body →
tautological self-consistency check, not an independent target.

**Impact:** none on production (`update_equilibrium`/`compton_equilibrium_ratio`
are off-path; the solver uses the perturbative Δρ_eq). But the test encodes a
physics misconception and is exactly the kind of code-/artifact-calibrated test
the B0 provenance census must flag. **Recommend:** replace the direction
assertions with the correct anchor — assert |ρ_e − 1| < (discretization
tolerance) and shrinking with N — and fix the docstring to state ρ_eq = 1 for
any BE.

### F2 — MINOR (off-path): `compton_equilibrium_ratio` discretization
`spectrum.rs:126-146`. ~2×10⁻⁶ error at N=10000 from directly forming I₄/(4G₃)
without analytic baseline subtraction. This is exactly the 0.1% -class
cancellation the docstring warns against; the function is (correctly) confined
to tests/off-path verification. No action needed beyond the F1 test fix.

---

## 4. Verdicts summary

- ρ_eq = I₄/(4G₃) (Zeldovich-Levich): **CORRECT**
- Perturbative Δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃, coeff 1, Δn² dropped: **CORRECT**
- ΔI₄ linearisation (1+2n_pl)Δn: **CORRECT**
- Grid quadrature consistency + analytic G₃^pl normalization: **CORRECT**
- δρ_inj = q_rel·t_c/(4θ_z): **CORRECT** (derived from Compton energy balance)
- R = (8/3)ρ̃_γ/α_h and ρ̃_γ identity: **CORRECT**
- Adiabatic −H·t_C·ρ_e (coeff 1): **CORRECT**
- Backward-Euler discretisation: **CORRECT**
- exact-form switch at |ΔG₃/G₃|>0.1: **CORRECT**
- `test_equilibrium_for_bose_einstein` direction assertions + docstring: **CONFIRMED TEST-VALIDITY BUG** (F1)
- `compton_equilibrium_ratio` off-path discretization: **MINOR** (F2)

**No production physics bug found.** One confirmed test-validity bug (F1).
