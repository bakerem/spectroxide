# Module audit B1-kompaneets — `src/kompaneets.rs`

**Auditor:** physics-inquisitor (adversarial, anchored outside the code)
**Date:** 2026-07-03
**Commit:** 1f45d64 (branch `main` per plan; audited on a local working tree, file unchanged)
**Verdict:** NO CONFIRMED BUGS. Two documented convention/accuracy trade-offs, one
stale docstring (LOW), one test-coverage gap. All 13 module unit tests + 7
integration Kompaneets tests pass.

---

## 1. References used

- Kompaneets (1957), Sov. Phys. JETP 4, 730 — original Fokker–Planck form.
- Chluba & Sunyaev (2012), MNRAS 419, 1294, Eq. 2/4 — modern statement of the
  Kompaneets operator with `n(1+n)` induced-scattering term.
- Burigana, Danese & de Zotti (1991) A&A 246, 49 — thermalization solver
  conventions (flux form, boundary treatment).

Derivation below is done **independently from first principles**; the code's
numeric output was not consulted until after the equation↔code mapping was
fixed. Confidence in the derivation: high (standard, well-established operator).

---

## 2. Independent derivation

### 2.1 Change of variables (x normalised by T_z, not T_e)

Standard Kompaneets in electron-frame frequency x_e = hν/(kT_e):

    ∂n/∂τ = (θ_e/x_e²) ∂_xe [ x_e⁴ ( ∂n/∂x_e + n(1+n) ) ],   τ = ∫ n_e σ_T c dt.

The code uses x = hν/(kT_z), so x_e = x·(T_z/T_e) = φ x with **φ ≡ T_z/T_e = 1/ρ_e**.
Substituting ∂_xe = (1/φ)∂_x, x_e⁴ = φ⁴x⁴, x_e² = φ²x², and using that φ is
x-independent (can be pulled through ∂_x):

    ∂n/∂τ = (θ_e/x²) ∂_x [ x⁴ ( ∂n/∂x + φ n(1+n) ) ].              (★)

- Prefactor is **θ_e** (NOT θ_e·φ).
- **φ multiplies only the n(1+n) drift**, not the ∂n/∂x diffusion.
- Matches module docstring lines 6–8 exactly. ✓
- Cross-check on the drift coefficient: after the θ_e prefactor, the drift term
  carries θ_e·φ = θ_z, i.e. the induced/recoil drift depends only on T_z; the
  diffusion term carries θ_e. This is used below to bound the H5 error.

### 2.2 Equilibrium (zero-flux stationary point)

F=0 ⇒ ∂n/∂x + φ n(1+n) = 0 ⇒ n = 1/(e^{φx − μ'} − 1) = BE at T_e (since φx = x_e).
For φ=1, μ'=0: n = n_pl(x). So Planck-at-T_z is stationary iff T_e=T_z. ✓

### 2.3 Planck-subtracted flux split

With n = n_pl+Δn and dn_pl/dx = −n_pl(1+n_pl):

    ∂n/∂x + φ n(1+n)
      = (φ−1) n_pl(1+n_pl) + dΔn/dx + φ(2n_pl+1)Δn + φΔn².        (†)

Verified by direct expansion of (n_pl+Δn)(1+n_pl+Δn) = n_pl(1+n_pl) +
(2n_pl+1)Δn + Δn². **Algebraically exact** (no truncation). Matches code
comment lines 42–44 and the flux built at lines 60–68 (test) / 734–743 (prod). ✓

### 2.4 Conservation properties of (★)

- Photon number: d/dτ ∫x²n dx = θ_e[F]_boundaries. F ∝ x⁴(...) → 0 at both
  x→0 (x⁴ beats the 1/x² of n_pl(1+n_pl)) and x→∞ (e^{−x}). So exact number
  conservation iff BCs enforce F=0 at ends. ✓
- Energy: d/dτ ∫x³n dx = −θ_e ∫F dx ≠ 0 in general → net Compton energy
  exchange. Correct: Compton conserves number, redistributes energy. ✓

---

## 3. Equation ↔ code mapping table

| # | Physics (derived) | Code location | Verdict |
|---|---|---|---|
| 1 | Operator form (★), prefactor θ_e, φ=1/ρ_e | docstring 6–8; phi=θ_z/θ_e L36,96 | ✓ exact |
| 2 | Flux split (†) | test L60–68; prod K_old L604–614; Newton L734–743 | ✓ exact, all three copies identical |
| 3 | Divergence (θ_e/x²)(F_R−F_L)/dx_cell, dx_cell=½(dx_{i-1}+dx_i) | L73–76; geom L403–408 | ✓ FV-consistent |
| 4 | Interface diffusion dΔn/dx uses dx[i] (right), dx[i-1] (left) | L57,601–602,730–731 | ✓ no L/R asymmetry |
| 5 | Interface drift uses arithmetic ½(Δn_i+Δn_{i+1}) at midpoint x_half | L54,599–600 | ✓ 2nd-order at midpoint |
| 6 | Equilibrium = BE at T_e; Planck stationary iff φ=1 | test_..._preserves_planck; rhs cancellation test | ✓ passes to 1e-20 |
| 7 | CN residual: Δn−Δn_old = ½dτ(K_new+K_old)+DCBR+src | L790–794 | ✓ |
| 8 | Newton Jacobian of flux incl. Δn²: n_{l,r}=n_pl+dn_half ⇒ φ(2n+1) | L799–808 | ✓ re-derived all 3 stencil coeffs (below) |
| 9 | c_vec = ∂R/∂ρ_e = −½dτ·inv_x2dc·(dfdr_R−dfdr_L), dfdr=−x⁴ n(1+n)/ρ² | L820–825 | ✓ = −∂F/∂φ·(1/ρ²) |
| 10 | Bordered block elimination δρ=(r_ρ−b'u)/(d−b'v), δΔn=u−vδρ | L830–917 | ✓ standard Schur complement |
| 11 | ρ_e ODE BE + Jacobian d_rho, b'_j=−dτ·R·h_norm·wem_j | L845–876 | ✓ signs re-derived |
| 12 | DC/BR IMEX (BE): +dτ·em·(neq−Δn_new), Jac +dτ·em; CN option | L755–772, 692–716 | ✓ relaxes Δn→neq, correct sign |
| 13 | Thomas get_unchecked guarded by len asserts | L187–195; entry asserts L537–556, 618–630 | ✓ complete |
| 14 | Boundary rows: zero Kompaneets flux, DC/BR+src only | L692–716 (prod); rhs[0]=rhs[n-1]=0 (test) | ✓ absorbing (see F2) |

### Jacobian re-derivation (item 8, the delicate one)

f_r = x⁴_r[(φ−1)np1_r + (Δn_{i+1}−Δn_i)/dx_r + φ·twonp1_r·dn_half_r + φ·dn_half_r²]
with dn_half_r=½(Δn_i+Δn_{i+1}). Then
∂f_r/∂Δn_i = x⁴_r[−1/dx_r + φ·½(twonp1_r+2dn_half_r)] = −a_r+b_r,
∂f_r/∂Δn_{i+1} = a_r+b_r, where a_r=x⁴_r/dx_r, b_r=x⁴_r·φ·(2n_r+1)·½ and
n_r=n_pl_half+dn_half_r ⇒ 2n_r+1 = twonp1_r+2dn_half_r. The +2dn_half_r piece is
exactly the derivative of the φΔn² term ⇒ **the Jacobian is the true Newton
Jacobian, not a Picard freeze.** Propagating through K_i=inv_x2dc(f_r−f_l) and
R_i = Δn_i − … − ½dτ k_i reproduces j_lower=−hdc(a_l−b_l),
j_diag=1−hdc(−a_r+b_r−a_l−b_l)+dcbr_jac, j_upper=−hdc(a_r+b_r). Matches L806–808. ✓

---

## 4. Findings (triaged)

### F1 — θ_e time-centering frozen in diffusion prefactor (documented; NOT a bug)
`inv_x2_dx_cell` (∝θ_e) is frozen at θ_e_old across the Newton loop while φ=1/ρ_e
iterates (L586–589, 676–690). Since the drift coefficient is θ_e·φ = θ_z·(ρ_e_old/ρ_e_new)
and the true value is θ_z, the mismatch is O(Δρ_e per step) ~ O(dτ) — same order as
the CN time-truncation floor, non-secular. Freezing keeps the Jacobian (incl. c_vec,
which omits ∂θ_e/∂ρ_e consistently) self-consistent so Newton converges to the fixed
point of the discrete equations as written. **Triage: convention/accuracy trade-off,
correctly documented as audit H5.** My independent bound confirms the comment's
O(Δρ_e·θ_e)~1e-7 estimate.

### F2 — Boundary node photon leak (intended absorbing BC; acceptable)
Boundary rows (L692–716) carry no Kompaneets flux term, so the interface flux
F_{1/2} that appears in the i=1 equation is not balanced by an equal-and-opposite
term at node 0 → photon number carried across the 1/2 interface into the boundary
is not conserved. This is the standard absorbing low-x boundary: at x_min~1e-4
DC/BR pins n to equilibrium, so the physical leak is negligible. The integration
test `test_photon_injection_number_conservation_pure_kompaneets` (passes) exercises
the production path and bounds the total leak. **Triage: acceptable, documented.**

### F3 — Stale docstring "capped at 1e8" (LOW)
`DcbrCoupling::emission_rates` docstring (L454) says rates are "capped at 1e8".
Production (`solver.rs` L1214–1245, 1287) uses a variable literally named
`uncapped` and applies only an `is_finite()` rejection — **no 1e8 cap exists**.
Backward-Euler DC/BR is unconditionally stable so the absence of a cap is not a
bug, but the docstring is wrong and could mislead. **Triage: doc fix. Same finding
as 2026-05-28; still unfixed.**

### F4 — Mixed quadrature in `quad_weights_x3` (benign)
H_dcbr integral weights (L410–417) use midpoint x³ (`x_half_cubed`) with a
trapezoidal average of f — a hybrid O(dx²) rule, not pure trapezoid. Consistent
and second-order; identical style to the `compton_equilibrium_ratio` note.
**Triage: benign.**

### F5 — Coverage gap: no in-module number-conservation test for the coupled path
The three number/energy-conservation unit tests (L994, 1042) exercise only the
test-only linear `kompaneets_step`. `test_coupled_inplace_preserves_planck` only
checks Δn stays 0, not conservation under a finite perturbation. The production
coupled solver IS covered by an integration test, but a direct in-module
perturbation→ΔN/N test would harden the flux-conservative claim. **Triage:
recommendation, not a defect.**

---

## 5. Verified correct (explicit)

- Change of variables x_e=φx → operator (★): prefactor θ_e, φ on drift only. ✓
- Flux split (†) algebraically exact; all three code copies identical. ✓
- Equilibrium = BE(T_e); Planck stationary iff T_e=T_z (1e-20 cancellation test). ✓
- Non-uniform-grid FD stencils: correct L/R spacing assignment, no sign asymmetry,
  finite-volume conservative in the interior. ✓
- Newton Jacobian consistent with residual **including the φΔn² term** (full
  re-derivation of j_lower/j_diag/j_upper + c_vec). ✓
- Bordered Newton = correct Schur-complement elimination; b'/d Jacobian signs
  re-derived and matched. ✓
- DC/BR IMEX backward-Euler sign and detailed-balance target (neq→relaxation). ✓
- Thomas `get_unchecked` fully guarded by entry asserts (L187–195, 537–556, 618–630);
  Thomas correctness tests (2×2, 3×3, diagonal) pass. ✓
- y-distortion magnitude test uses independent analytic target Δρ/ρ=4y (not
  code-derived); passes at <5%. ✓

## 6. Recommendations

1. Fix the L454 docstring (remove "capped at 1e8" or state "is_finite rejection only").
2. Add an in-module ΔN/N test for `kompaneets_step_coupled_inplace` with a finite
   Gaussian perturbation, T_e=T_z, no DC/BR — target the boundary-leak bound.
3. (Optional, feeds B3 MMS) The flux-conservative structure makes this operator a
   good candidate for a manufactured-solution convergence test to pin the CN order.

## 7. What would convince a skeptic

- (†) is an identity, verifiable by hand in two lines; §2.3 shows the expansion.
- The Planck-cancellation test asserts rhs=0 to **1e-20** and linearity in (φ−1)
  to 1e-3 — this is impossible unless dn_pl/dx is used analytically (pitfall #1).
- The Jacobian re-derivation in §3 reproduces every stencil coefficient from the
  residual symbolically; the φΔn² term appears in both residual and Jacobian via
  the same n=n_pl+dn_half construction, so a reviewer can check consistency without
  running code.
- The y-magnitude test's target (4y) is an external analytic anchor, so a passing
  run is not circular calibration.
