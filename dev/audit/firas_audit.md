# Audit B1-firas: `python/spectroxide/firas.py`

Auditor: physics-inquisitor pass, 2026-07-03. Scope: `firas.py` only, per
`dev/PLAN_VALIDATION_AUDIT_2026-07-02.md` Part B1. No source files edited.

## 1. Primary-source recipe (established before reading code output)

Source: Fixsen, Cheng, Gales, Mather, Shafer, Wright 1996, ApJ 473, 576
(astro-ph/9605054), full text via ar5iv.

- **Fit model**: `I(ν) = B_ν(T₀) + ΔT ∂B_ν/∂T + G₀ g(ν) + p·∂S_c/∂p`, p = μ
  or y. Table 4 residuals are `I_obs − B_ν(T₀) − ΔT ∂B_ν/∂T − G₀ g(ν)`,
  i.e. the temperature offset and Galactic model are **already profiled
  out** in constructing the tabulated residual — units kJy/sr, confirmed
  by spot-checking 4 rows (2.27, 2.72, 11.34, 21.33 cm⁻¹) against the
  bundled `data/firas_monopole_spec_v1.txt`; all match exactly.
- **Galactic model**: paper explicitly adopts `G(ν) = G₀ ν² B_ν(T=9 K)`
  over the empirical alternative ("produces a lower χ²... we use this
  model").
- **μ and y were fit separately, not jointly**: "We fit either the
  Kompaneets parameter or the chemical potential, but the two are too
  similar to fit them simultaneously." This is the single most important
  methodological fact for reproducing the quoted numbers.
- **Quoted central values** (Sect. 6.2–6.3): `μ = (−1 ± 4)×10⁻⁵`
  (statistical), `y = (−1 ± 6)×10⁻⁶` (statistical). Final 95% CL:
  `|μ| < 9×10⁻⁵`, `|y| < 15×10⁻⁶` (the y limit explicitly adds a
  4×10⁻⁶ systematic).
- **Reconstructed 95% CL recipe**: `limit = |Â| + z·σ_total`,
  `z = 1.96` (two-sided 95%), `σ_total` from stat+sys in quadrature.
  Check: y → `1 + 1.96·√(6²+4²) ≈ 1 + 14.1 = 15.1×10⁻⁶` ≈ quoted 15.
  μ → `1 + 1.96·4 ≈ 8.8×10⁻⁵` ≈ quoted 9 (systematic folded into the
  quoted 4×10⁻⁵ already). This confirms `firas.py`'s
  `upper_limit(template, cl) = |Â| + z_cl·σ` formula and its docstring's
  claim that this is "the literature convention" — **verified correct**.
- T₀ was **fixed** at 2.728 K in the 1996 fit (not floated jointly with
  μ/y); `firas.py`'s default `_T_CMB = 2.726` K is the later Fixsen &
  Mather (2002) value — a documented, intentional convention update, not
  an error.

## 2. Code audit

### 2.1 Data and units — VERIFIED CORRECT
- Table 4 residual/σ/Galaxy columns spot-checked against ar5iv full text
  for 4 rows spanning the band: exact match.
- `_freq_cm_to_ghz`: `ν[GHz] = freq_cm · c[m/s] · 1e-7`. Dimensionally
  `[cm⁻¹]·[m/s]·[10⁻⁷] = [cm⁻¹]·[cm/s]·[10⁻⁹] → Hz·10⁻⁹`, correct; numeric
  check `2.27 cm⁻¹ → 68.05 GHz` vs header comment `68.02 GHz` (rounding
  of the nominal channel spacing) — fine.
- `_freq_cm_to_x`, `_dn_to_dI_kJy`: round-trip exactly (`x → ν → x`
  identity to 1e-6, tested numerically); `ΔI = 2hν³/c² · Δn`, division by
  `1e-23` (1 kJy = 10⁻²³ W/m²/Hz) — dimensionally correct.
- Covariance construction `C_ij = σ_i σ_j corr_ij`: checked numerically
  — `cov` is symmetric to machine precision, positive-definite (min
  eigenvalue 14.5, max 81165, condition number ~5600 — well-conditioned),
  `cov @ cov_inv = I` to 1e-15. **No numerical pathology.**

### 2.2 μ-template null crossing — FALSE ALARM (initially flagged, then refuted)
`mu_shape` docstring claims the null is at `x = β_μ ≈ 2.19`. This is the
correct, well-known μ-distortion intensity null (≈124 GHz), distinct from
the y-distortion null at x≈3.83 (≈217 GHz) and from κ_c≈3.6 (the DC/BR
photon-production frequency scale used elsewhere in the codebase for a
different purpose). Confusing these three would be an easy error; they
are not confused here.

### 2.3 Nuisance-parameter profiling algebra — VERIFIED CORRECT
`fit_amplitude`, `fit_amplitude_marginalised`, `fit_distortion` all solve
the standard weighted normal equations `(AᵀC⁻¹A)θ = AᵀC⁻¹r` via
`np.linalg.solve`/`inv`; `param_cov = (AᵀC⁻¹A)⁻¹` is the correct GLS
covariance. `chi2_min = rCr − (AᵀC⁻¹r)ᵀ(AᵀC⁻¹A)⁻¹(AᵀC⁻¹r)` for the
single-template case (`fit_amplitude`) is the standard profiled-χ²
identity (Schur complement), matches the general multi-template formula
`resid = r − Aθ̂; χ² = residᵀC⁻¹resid`. Verified `chi2_min ≤ chi2_null`
holds numerically for all fits tried.

### 2.4 CRITICAL FINDING: default `upper_limit_mu()`/`upper_limit_y()` do not reproduce the Fixsen 1996 anchors — CONFIRMED, convention mismatch (not a numerical bug)

Numerically (full pipeline, `FIRASData()` defaults):

| Quantity | Code (defaults, marginalise_y/mu=True) | Code (no cross-marginalisation) | Fixsen 1996 |
|---|---|---|---|
| μ̂ ± σ | −3.28e-5 ± 6.55e-5 | −1.23e-5 ± 3.59e-5 | −1e-5 ± 4e-5 |
| μ 95% UL | **1.61e-4** | 8.27e-5 | 9e-5 |
| ŷ ± σ | 2.70e-6 ± 7.22e-6 | −0.32e-6 ± 3.96e-6 | −1e-6 ± 6e-6 (stat) |
| y 95% UL | **1.68e-5** | 8.08e-6 | 1.5e-5 (incl. 4e-6 sys) |

Root cause: `upper_limit_mu()` defaults to `marginalise_y=True`, and
`upper_limit_y()` defaults to `marginalise_mu=True`. But §6.2 of Fixsen
1996 states explicitly that μ and y were fit *separately* ("too similar
to fit them simultaneously") — i.e. the historical 9e-5/1.5e-5 numbers
come from a fit that does **not** marginalise μ over y or vice versa.
Marginalising jointly is statistically more conservative but the μ–y
shape degeneracy over the FIRAS band is severe enough that it inflates
σ_μ by ~82% (3.59e-5→6.55e-5) and σ_y by ~82% as well, blowing the
default upper limits ~1.8× (μ) and ~1.1–2× (y, depending on whether the
paper's systematic is added) above the textbook anchors. Setting
`marginalise_y=False`/`marginalise_mu=False` reproduces μ 95% UL to 8%
and central values (μ̂=−1.23e-5 vs −1e-5, σ=3.59e-5 vs 4e-5) essentially
exactly — this *is* the Fixsen-matching recipe, but it is not the
package default and its correctness relative to the literature anchor is
not stated anywhere in the docstrings.

This is not a coding bug — the joint-marginalisation code path is
algebraically correct (§2.3) — but the docstrings advertise these
functions as computing "the FIRAS upper limit," which readers will
reasonably take to mean "reproduces 9e-5/1.5e-5." As shipped, the
*default* call does not, by a factor of ~1.8. Any use of
`upper_limit_mu()`/`upper_limit_y()` with default arguments elsewhere in
the codebase or in notebooks (paper figures, FIRAS constraint plots)
should be checked for whether this joint-marginalisation choice was
intended or is inflating quoted limits.

**Action recommended**: (a) add an explicit docstring note quantifying
this ~1.8× effect and stating which flag combination reproduces the
literature numbers; (b) audit call sites (`notebooks/observational/`,
Sect. 5/7 figures) for which convention they assume.

### 2.5 `profile_limit_floating_T` / CCJ24 path — VERIFIED CONSISTENT, documented as separate convention
This one-sided, `Δχ²=2.71` (Wilks, `z=norm.ppf(0.95)=1.645`), floating-T,
single-signal-template method matches the `dp_firas_method_comparison.ipynb`
CCJ24-statistic notebook (already cross-validated to ~3% per prior audit
memory) — it does **not** marginalise over the competing μ/y template by
default, consistent with the modern (Chluba, Cyr & Johnson 2024)
practice of deprojecting only ΔT and dust. This is the correct point of
comparison for that notebook, not `upper_limit_mu()`. The two code paths
use genuinely different — and inconsistent with each other — marginal­
isation conventions; this should be documented at the module level so a
reader doesn't assume `upper_limit_mu()` and `profile_limit_floating_T`
are interchangeable.

`g_kJy` inside `_joint_fit_floating_T` (line ~923-925) is manually
recomputed as `prefactor·G_bb(x)/T` rather than calling `_g_bb` — this
is `∂I/∂T` (per-Kelvin, not per ΔT/T as in the main class's `_G_kJy`
template built from `_g_bb`). Verified algebraically equivalent to
`greens.g_bb` (`x·n_pl·(1+n_pl) = x e^x/(e^x-1)²`) and dimensionally
distinct on purpose (Kelvin vs fractional) — internal-only use, not
exposed, **false alarm**, but worth a one-line comment since it silently
diverges from the reusable `_g_bb` template convention used everywhere
else.

### 2.6 Galactic dust template units — MINOR, documented false alarm
`_galactic_dust_template_kJy` computes `ν²·B_ν(T_dust)` and divides by
`1e-23`, labelling the result "kJy/sr." Dimensionally this is
`W·Hz/m²/sr / 1e-23`, not kJy/sr (extra factor of ν² relative to a true
intensity). This is intentional and stated in the docstring — G₀ is a
free-floating nuisance amplitude with no physical normalisation
requirement, so the "kJy/sr" label is just informal shorthand for "same
numeric scale as the other templates before fitting." No numerical
consequence since it only enters as `G₀ · template` with `G₀` free.
Confirmed harmless; recommend renaming the docstring's units claim to
avoid future confusion.

### 2.7 χ² threshold / CL conventions — VERIFIED, two distinct (correct) conventions coexist
- `upper_limit()` family: two-sided `z_cl = Φ⁻¹(0.5+cl/2)` (1.96 at 95%)
  — matches Fixsen 1996 (§1 above).
- `profile_limit_floating_T`: one-sided `z_cl = Φ⁻¹(cl)` (1.645 at 95%),
  equivalent to `Δχ²=2.71` for 1 dof — standard Wilks' theorem profile
  likelihood, matches CCJ24 usage.
Both are internally correct for their respective stated conventions;
flagged only so a reader doesn't conflate the two 95% numbers.

### 2.8 Test-validity audit — CONFIRMED WEAKNESS (per audit protocol item 7/CLAUDE.md pitfall #9)
`python/tests/test_firas.py`: `test_upper_limit_mu_order_of_magnitude`
and `test_upper_limit_y_order_of_magnitude` assert
`1e-6 < mu_lim < 1e-3` and `1e-7 < y_lim < 1e-3` — a 3-order-of-magnitude
window. These tests would not have caught the ~1.8× discrepancy found in
§2.4, nor would they catch a factor-of-5 normalization bug in `mu_shape`.
`MU_FIRAS_95`/`Y_FIRAS_95` module constants are tested only for equality
to their own hardcoded literals (`test_mu_95_value` etc.) — tautological,
not an anchor check against the live `FIRASData` pipeline. **No test in
the file asserts that `upper_limit_mu()`/`upper_limit_y()` agree with
`MU_FIRAS_95`/`Y_FIRAS_95` to any quantitative tolerance.**
Recommend: add a test asserting `upper_limit_mu(marginalise_y=False)`
agrees with `MU_FIRAS_95` to ~15% (documenting the recipe from §1), and
a second test documenting (not silently passing) that the
joint-marginalised default is a factor ~1.5–2× looser, so a future
change to the marginalisation default is caught.

## 3. Triage summary

| # | Item | Verdict |
|---|---|---|
| 2.1 | Data/units/covariance construction | Verified correct |
| 2.2 | μ-null at β_μ≈2.19 vs y-null at 3.83 | False alarm (correct) |
| 2.3 | GLS normal-equations / profiling algebra | Verified correct |
| 2.4 | Default `upper_limit_mu/y` vs Fixsen 1996 anchors | **Confirmed convention mismatch, ~1.8× loose vs literature under default marginalisation; recipe to match literature exists but isn't the default and isn't documented as such** |
| 2.5 | `profile_limit_floating_T` vs `upper_limit_mu` conventions | Both correct, but mutually inconsistent conventions coexist undocumented at module level |
| 2.6 | Galactic template pseudo-units | Documented false alarm |
| 2.7 | Two-sided vs one-sided CL | Both correct for stated use |
| 2.8 | Test tolerance too loose to catch 2.4 | **Confirmed test-quality gap** |

No arithmetic bugs, sign errors, or dimensional errors found in
`firas.py`. The one substantive finding (2.4) is a statistical-convention
choice whose magnitude (~80% inflation of the quoted limit) is large
enough to matter for any paper figure or claim that cites
`upper_limit_mu()`/`upper_limit_y()` defaults as "the FIRAS 95% CL
limit."
