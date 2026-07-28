# B1 Module Audit — photon-bump Compton-broadening helpers

**Date:** 2026-07-05
**Auditor:** physics-inquisitor (independent context)
**Scope:** `f_cs`, `alpha_cs`, `beta_cs`, `broadened_bump`, `f_int` in
`src/greens.rs` and their mirrors `_f_cs`, `_alpha_cs`, `_beta_cs`,
`_broadened_bump` in `python/spectroxide/greens.py`; the surviving-bump term
of `greens_function_photon`; and the `y_γ` integrand
(`Cosmology::compton_y_parameter` / `_y_compton`).
**Trigger:** finding **M-4** of `dev/audit/greens_audit.md` — these helpers cite
Arsenadze et al. 2025, App. D, and were *not re-derived* in the earlier pass.

## Primary source obtained (verbatim)

Downloaded the arXiv LaTeX **source** (not ar5iv, which garbled Chluba 2015 in a
prior audit) of **arXiv:2409.12940**, *"Shaping Dark Photon Spectral
Distortions"*, Arsenadze, Caputo, Gan, Liu & Ruderman (2025), Appendix
`green_func_y`. All equations below are quoted from `Draft.tex`:

```
f(x') = e^{-x'} (1 + x'^2/2)
α(x',z') = (3 − 2 f(x')) / sqrt(1 + x' y_γ)
β(x',z') = 1 / (1 + x' y_γ [1 − f(x')])
F(x;x',z') = exp{ −(1/(4 β y_γ)) [ log(x(1/x' + y_γ)) − α y_γ ]^2 } / (x' sqrt(4π β y_γ))
∫₀^∞ F dx = e^{(α+β) y_γ} / (1 + x' y_γ)                                       (= f_int)
G_y ⊃ α_ρ x' (ρ̄/(2T)) P_s F(x)                          (surviving free-streaming bump)
G_y ⊃ α_ρ x' (1 − P_s e^{(α+β)y_γ}/(1+x'y_γ)) Y(x)/4                     (broadened y-term)
y_γ(z) = ∫₀^z dz' (T/m_e)(σ_T n_e)/(H(1+z'))                                  (Eq. 607)
```

**Provenance/labeling notes:** (1) the paper's Green's function is for the
**intensity ΔI_γ** (Eq. 557/560), not the occupation number Δn that
`greens.rs` works in — this matters for the shape mapping below. (2) In the
`Draft.tex` source these are unnumbered `\bea` blocks in App. `green_func_y`;
the code's "Eq. D13–D16" labels are the *published JHEP* numbering, which I
could not verbatim-confirm from source — but the **content** is confirmed
verbatim. (3) This is a **different** paper from `2409.12115` (Chluba, Cyr &
Johnson, *"Revisiting Dark Photon Constraints…"*, = CCJ24 in
`dark_photon_audit.md`); that paper contains no log-normal broadening. The
citation `arXiv:2409.12940` (authors/title) is correct.

## Equation ↔ code mapping (verdicts)

| Quantity | Code (rs / py) | Arsenadze App. D (raw source) | Verdict |
|---|---|---|---|
| f(x') | `(-x).exp()*(1+x*x/2)` / `np.exp(-x)*(1+0.5*x**2)` | `e^{-x'}(1+x'^2/2)` | ✅ exact |
| α | `(3-2f)/√(1+x·yg)` | `(3−2f)/√(1+x'y_γ)` | ✅ exact |
| β | `1/(1+x·yg·(1-f))` | `1/(1+x'y_γ[1−f])` | ✅ exact |
| μ_ln | `ln x' + α·yg − ln(1+x'·yg)` | matches F exponent (see derivation) | ✅ exact |
| σ²_ln | `2·β·yg` | matches F exponent `−(·)²/(4βy_γ)` ⇒ 2σ²=4βy_γ | ✅ exact |
| f_int | `exp((α+β)yg)/(1+x'yg)` | `e^{(α+β)y_γ}/(1+x'y_γ)` | ✅ exact |
| bump shape (Δn) | unit log-normal `L(x)` × `G2/x²` | `F=(x/x')L` in intensity ÷ x³ ⇒ `L/x²` | ✅ exact (see below) |
| surviving coeff | `p_s·(1−j_μ)·G2/x²` | `α_ρ x'(ρ̄/2T)P_s = P_s·(x' n̄/2)` → `P_s·G2/x²` | ✅ factor-exact |
| coeff_y | `1 − p_s·f_int` | `1 − P_s e^{(α+β)y_γ}/(1+x'y_γ)` | ✅ exact |
| μ_factor | `1 − p_s·x₀/x'` | `1 − P_s x_0/x'` | ✅ exact |
| y_γ integrand | `θ_e σ_T c n_e / H` in `d ln(1+z)` | `(T/m_e)(σ_T n_e)/(H(1+z')) dz'` | ✅ exact (= ∫θ_e dτ, 0→z) |

### Derivation anchoring the log-normal (done before trusting output)
Rewriting the F exponent: `log(x(1/x'+y_γ)) − α y_γ = log x − [log x' + α y_γ −
log(1+x'y_γ)] = log x − μ_ln`, so `F ∝ exp[−(log x − μ_ln)²/(4β y_γ)]` ⇒ a
log-normal with `μ_ln` exactly as coded and `σ²_ln = 2β y_γ` exactly as coded.
Integrating `F dx` (Gaussian in `u=ln x`, `dx=e^u du`) reproduces
`∫F = e^{μ_ln+β y_γ}/x' = e^{(α+β)y_γ}/(1+x'y_γ) = f_int` — matching the paper's
own normalization identity. Also `F = (x/x')·L(x)` where `L` is the
**unit-normalized** log-normal the code uses; converting the paper's *intensity*
bump (`∝ x L`) to *occupation* (÷ x³ ∝ 2hν³/c²) gives `∝ L/x²`, exactly the
code's `G2/x²·L`. The code therefore uses the **exact** log-normal (Eq. F),
not the secondary Gaussian-in-x approximation (paper Eq. `sigma_xpeak_Appx`,
valid only for x'y,αy,βy ≲ 1) — i.e. equal-or-better than the paper's own plots.

## Numerical verification (after derivation)

- **Absolute normalization is factor-exact.** With `P_s=1` (x'=2, z_h=3×10³),
  `∫x² G_ph dx = 2.404044`, and the closed decomposition
  `surviving(=P_s(1−J_μ)G₂=2.39495) + t_part(=α_ρx'·J_μ·(λ/4)·∫x²g_bb=0.00910)
  = 2.404044` matches to 6 digits; `num/G₂ = 0.999971`. This rules out a
  factor-2 or missing-G₂ error in the `G2/x²` prefactor (and confirms the
  surviving term correctly carries **no** `α_ρ x'`, since that factor is
  absorbed by the intensity→occupation + per-ΔN/N normalization). Injecting
  ΔN/N=1 with P_s=1 conserves photon number (∫x²Δn = G₂).
- **Unit log-normal:** `∫L dx = 1.00000`, `mean/x' = f_int` to 5 digits for
  (x',y_γ) = (2,0.01),(2,0.3),(0.5,0.3).
- **f_int sign** flips at x'≈4 (Kompaneets energy balance): >1 for x'<4 (net
  up-scatter/heating), <1 for x'>4 (recoil loss). Physical.
- **Operating regime:** y_γ(z_h) = 1.3×10⁻⁷ (10³), 3.7×10⁻³ (10⁴),
  3.9×10⁻² (3×10⁴), 0.113 (5×10⁴). So y_γ ≪ 1 across the *entire* allowed
  y-era band (z_h ≤ 5×10⁴; the (5×10⁴,2×10⁵) transition band is gated off),
  inside the paper's stated validity. The `f_int` overflow clamp at 700 is
  therefore dead code (defensive only).
- **Fallback continuity:** across the `y_γ = 1e-6` threshold the integral
  (=1) and mean (=x'·f_int→x') are continuous; only the *pointwise* peak height
  jumps ~3.5× (fixed 0.5%·x' linear Gaussian below vs the narrower log-normal
  σ_ln=√(2y_γ)≈1.4×10⁻³ above). Occurs only near z_h~1.5–2×10³.
- **Rust↔Python parity:** regenerated `parity_fixtures.json` from
  `cargo run --release --example generate_parity_fixtures`;
  `test_parity.py -k "greens_function_photon or compton_y"` → 4 passed
  (agreement ≤1e-12). Every branch of the helpers is formula-identical by read.

## Triage

**CONFIRMED BUGS: none.** Every coefficient and functional form reproduces
Arsenadze 2409.12940 App. D verbatim; absolute normalization is factor-exact.

**Doc/robustness notes (not bugs):**

- **A-1 (DOC, low):** `greens.rs:510` / `greens.py:1052-1054` comment calls
  `exp(μ_ln) = x'e^{αy}/(1+x'y)` the "mode (peak)" of the distribution. For a
  log-normal that is the **median**; the mode is `exp(μ_ln−σ²)`. The paper's
  `x_peak = exp(μ_ln)` is the peak of its *Gaussian-in-x* approximation (which
  the code does not use). Numerically negligible (σ²=2βy_γ ≤ 0.23 in-regime).
  Recommend correcting the comment to "median" to avoid confusion.

- **A-2 (benign discontinuity, low):** pointwise peak Δn of the surviving bump
  jumps ~3.5× across the `y_γ = 1e-6` fallback threshold (z_h~1.5–2×10³).
  Integrated μ, y, photon number, and energy are all continuous, so no
  observable (FIRAS/PIXIE band-integrated) is affected. Cosmetic; same class as
  greens_audit W-1/W-3.

- **A-3 (edge of validity, low):** for x'≳10 at the top of the y-era band
  (z_h=5×10⁴) `x'y_γ ≈ 1.1` mildly exceeds the paper's `x'y_γ ≲ 1` window, but
  (i) the code uses the exact log-normal F, not the Gaussian approx, and
  (ii) x'≳5 contributes negligibly to FIRAS per the paper (§ App.), and
  (iii) this sits exactly at the gated band edge. Benign.

**Published-figure impact: none credible.** The helpers faithfully implement
the cited primary source, and the one place the code departs from the paper's
own text — using the exact log-normal instead of the Gaussian-in-x
approximation — is strictly *more* accurate and agrees with it to O(y_γ²). No
finding here can shift a published figure.

## Closes
greens_audit.md **M-4** (Arsenadze App. C/D now audited against primary source).
