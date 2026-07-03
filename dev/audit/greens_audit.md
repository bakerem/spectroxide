# B1 Module Audit — `greens.rs` / `greens.py`

**Date:** 2026-07-03
**Auditor:** physics-inquisitor (independent context)
**Scope:** heat-injection Green's function `G_th`, visibility fits, photon-injection
GF `G_ph`, photon survival probability `P_s`, critical frequencies `x_c`, Compton
broadening of the surviving bump.
**Method:** every coefficient re-derived from the **primary reference raw text**
(ar5iv HTML of arXiv:1304.6120 and arXiv:1506.06582), not the paper PDF's rendered
equations and not the code's output. Rust↔Python parity is separately pinned by
`python/tests/test_parity.py`; this memo audits **physics vs literature**.
Out of scope / not re-reported: P0-1…P0-7 (already fixed, see `AUDIT_SUMMARY.md`).

---

## Equation ↔ code mapping (verified against raw paper text)

| Quantity | Code (rs / py) | Primary source (raw text) | Verdict |
|---|---|---|---|
| μ amplitude | `3/KAPPA_C = 1.40066` | Chluba 2013 Eq. 6: `α ≈ 1.401` | ✅ match |
| J_μ | `1 − exp(−((1+z)/5.8e4)^1.88)` | C13 Eq. 5: `1 − exp(−[(1+z)/5.8×10⁴]^1.88)` | ✅ exact |
| J_y | `1/(1 + ((1+z)/6.0e4)^2.58)` | C13 Eq. 5: `(1 + [(1+z)/6.0×10⁴]^2.58)^{−1}` | ✅ exact |
| J_bb (base) | `exp(−(z/1.98e6)^2.5)` | C13 `J = exp(−(z/z_μ)^{5/2})`, C15 Eq. 13 `z_μ ≈ 1.98×10⁶` | ✅ match |
| J_bb* | `0.983·J_bb·(1 − 0.0381(z/z_μ)^2.29)` | C15 Eq. 13: `0.983 e^{−(z/z_μ)^2.5}[1 − 0.0381(z/z_μ)^2.29]` | ✅ exact |
| y-weight | `J_y/4` | C13 Eq. 6: `J_y/4 · Y_SZ` | ✅ exact |
| T-weight | `(1 − J_bb*)/4` | C13 Eq. 6: `(1 − J)/4 · G` | ✅ (J→J_bb*, see M-1) |
| x_c,DC | `8.60e-3·((1+z)/2e6)^{1/2}` | C15 Eq. 25a: `8.60×10⁻³[(1+z)/2×10⁶]^{1/2}` | ✅ exact |
| x_c,BR | `1.23e-3·((1+z)/2e6)^{−0.672}` | C15 Eq. 25b: `1.23×10⁻³[(1+z)/2×10⁶]^{−0.672}` | ✅ exact |
| x_c combine | `x_c² = x_DC² + x_BR²` | C15 Eq. 25: `x_c² ≈ (x_c^DC)² + (x_c^BR)²` | ✅ exact (quadrature) |
| P_s | `exp(−x_c/x)` | C15 Eq. 24: `P_s ≈ e^{−x_c(z)/x}` | ✅ exact |
| x₀ (balanced) | `X_BALANCED = 4/(3α_ρ) = 3.6016` | C15 Eq. 7: `x₀ = (4/3)/α_ρ ≈ 3.6016` | ✅ exact |
| μ-era short-circuit | `J_μ > 1 − 1e-12` (z ≳ 3.5×10⁵) | overflow guard only | ✅ benign |
| Compton f, α, β, f_int, bump | Arsenadze 2025 Eq. D13–D16 | not re-derived here (2025 preprint) | ⚠ see M-4 |

**Constants** (`constants.rs`): `BETA_MU = 3ζ(3)/ζ(2)`, `KAPPA_C = 12/β_μ − 9G₂/G₃ =
2.14185`, `ALPHA_RHO = G₂/G₃ = 0.37021`, `X_BALANCED = 4/(3α_ρ) = 3.6016`,
`3/KAPPA_C = 1.40066`. All reproduce the paper values to full f64 precision.

---

## Limiting cases (checked analytically)

- **Deep μ-era** (z_h ≳ 3×10⁵): J_μ→1, J_y→0 ⇒ `G_th → 1.401·J_bb*·M(x)`. ✅
  (test `test_greens_mu_era`, `test_mu_from_delta_injection`.)
- **y-era** (z_h ≲ 10⁴): J_μ→0, J_y→1, J_bb*→1 ⇒ `G_th → ¼ Y_SZ(x)`. ✅
- **Deep thermalization** (z_h ≫ z_μ): J_bb*→0 ⇒ `G_th → ¼ G_bb(x)` = pure ΔT. ✅
  (test `test_greens_high_z_is_temperature_shift`.) *Note:* pure ΔT is the
  **high-z** limit, not z→0. At z→0 the GF → pure y (J_y→1), which is physically
  correct; the plan's "z→0 pure ΔT" phrasing is loose.
- **P_s → 1** at x ≫ x_c ✅; **P_s → 0** at x ≪ x_c ✅.
- **Soft-photon limit** (P_s→0) of `greens_function_photon`: `mu_factor→1`,
  reduces to `α_ρ·x_inj·G_th(x,z_h)`. Verified: injecting ΔN/N at x_inj deposits
  Δρ/ρ = α_ρ·x_inj·(ΔN/N) (energy/photon × N/ρ = x_inj·G₂/G₃), so fully-absorbed
  photon injection ≡ heat injection of that energy. ✅ docstring claim holds.
- **Balanced injection** x_inj = x₀, P_s≈1 ⇒ `mu_factor = 1 − x₀/x₀ = 0` ⇒ μ=0,
  with sign flip below/above x₀. Matches C15 Eq. 7. ✅

---

## Findings

### No confirmed physics bugs.
Every numeric coefficient in both files reproduces the primary literature exactly.
This is consistent with the module's prior audit history (multiple passes, no
criticals) and the fact that the real defects here (P0-1…P0-7) were Rust↔Python
mirror divergences, all already fixed.

### Documented convention mismatches / doc defects (not bugs)

- **M-1 (convention, correct):** `G_th` uses `J_bb*` (C15 Eq. 13) in *both* the
  μ-part and the T-part where C13 Eq. 6 printed the base `J`. This is the standard
  post-2015 improvement; the base J_bb and z_μ are unchanged. Consistent and
  documented in the module header.

- **M-2 (DOC DEFECT — recommend fix):** `greens.rs:78-79` and `greens.py:347-353`
  claim *"the original Chluba 2013 value was 5.9×10⁴, updated to 6.0×10⁴ from
  Arsenadze 2025."* **This provenance is false.** Raw Chluba 2013 Eq. 5 reads
  `J_y = (1 + [(1+z)/6.0×10⁴]^2.58)^{−1}` — i.e. 6.0×10⁴ is the *original* C13
  value; "5.9×10⁴" appears nowhere in the J_y context of the paper. The **numeric
  value used by the code (6.0e4) is correct and matches the primary source**; only
  the comment's history is wrong. Recommend correcting the comment to avoid a
  reviewer concluding a coefficient was silently swapped. The 2.58 exponent
  (flagged "unverified" in prior notes) is now **verified correct** against C13.

- **M-3 (energy non-closure, documented):** independent J_y fit ⇒
  `J_μ·J_bb* + J_y + (1−J_bb*) ≠ 1`; the ≤16–17% residual (peaking z~7–8×10⁴)
  is absorbed into the unobservable ΔT. This is Chluba 2013 §3's own ansatz;
  callers needing strict closure use the PDE. Correctly documented; the `gf_fit`
  decomposition path uses a strictly-closed T-weight `(1−J_μJ_bb*−J_y)/4`
  (calibration-only, not production).

- **M-4 (secondary reference, not re-derived):** the photon-bump broadening
  helpers (`f_cs`, `alpha_cs`, `beta_cs`, `broadened_bump`, `f_int`) cite
  Arsenadze et al. 2025 Eq. D13–D16, a preprint not audited against a peer-reviewed
  primary source here. The log-normal is normalized to unit integral (one surviving
  photon) with mean `x_inj·f_int`; structure is internally consistent and the
  y_γ→0 fallback (narrow Gaussian) is sane. Flag for a dedicated Arsenadze pass.

### Warnings (regime-dependent)

- **W-1 (P_s branch stitch):** `photon_survival_probability_numerical` uses the
  τ_ff integral for z_h ≤ 5×10⁴ and analytic `exp(−x_c/x)` for z_h > 5×10⁴. The
  two need not agree exactly at z_h = 5×10⁴, so P_s can be mildly discontinuous
  there. Benign in practice: 5×10⁴ is the lower edge of the forbidden μ-y band
  `(5e4, 2e5)` in which the photon GF is gated off entirely, so the stitch is never
  exercised as a smooth curve by production callers. Worth a continuity spot-check
  if the band gating is ever relaxed.

- **W-2 (P_s consistency across entry points):** `mu_from_photon_injection` uses
  the **analytic** P_s while `greens_function_photon` uses the **numerical** one.
  They coincide in the μ-era (z_h ≥ 2×10⁵ ⇒ numerical falls back to analytic), the
  only regime where both are called for photon injection, so the μ values are
  consistent. No action needed; noted for future maintainers.

- **W-3 (τ_ff x>500 overflow handling):** `bose_factor` is set to `+∞` (rs) /
  `1e200` (py) for x>500, relying on downstream `tau>500 → 0` saturation. At x>500
  P_s should be ≈1 (photon far above x_c survives), and indeed the integrand blows
  up only via `(e^x−1)/x³` in the *absorption* rate — but at high x the physical
  P_s→1, so forcing P_s→0 via saturation would be **wrong** if ever reached.
  In practice x>500 combined with z in the y-era is unphysical for injection
  studies and the branch is dead; still, the asymmetry (`+∞` rs vs `1e200` py)
  is a latent parity/robustness wart. Low priority.

---

## Recommendations

1. Fix the M-2 provenance comment in both `greens.rs` and `greens.py` (state that
   6.0×10⁴ / 2.58 are the original Chluba 2013 Eq. 5 values).
2. Add a P_s branch-continuity test at z_h = 5×10⁴ (W-1) to pin the stitch size,
   even though the band gating hides it.
3. Schedule a dedicated audit of Arsenadze 2025 App. C/D against a primary source
   for the photon-bump broadening (M-4).
4. Harmonize the x>500 overflow sentinel between Rust (`+∞`) and Python (`1e200`)
   and add an assertion that this branch implies P_s→1 not P_s→0 (W-3).

**Bottom line:** `greens.rs`/`greens.py` are physically faithful to Chluba 2013
Eq. 5/6 and Chluba 2015 Eq. 13/24/25 to the last digit. One false provenance
comment (M-2) is the only actionable defect; no numerical coefficient is wrong.
