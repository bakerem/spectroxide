# Fig. 2 visibility-parameter differences: degeneracy analysis

**Date:** 2026-07-30
**Trigger:** Referee 1, section B — "in each of the cases (e.g. Fig. 2, Fig. 3, Fig. 4, Fig. 6)
there are small differences in the results." EB's actionable note: *"probably should have more
discussion about fig 2, what parameters control the discrepancy, do these make a difference."*
**Scripts:** `dev/scripts/visibility_diagnostics/visibility_likelihood_flatness.py` (Result 1, 2),
`dev/scripts/visibility_diagnostics/visibility_firas_indistinguishability.py` (Result 3),
`dev/scripts/visibility_diagnostics/visibility_variant_spread.py` (Result 4),
`dev/scripts/visibility_diagnostics/visibility_zy_convention_test.py` (the z_y residual)
**Response draft:** `/home/bakerem/cosmoxide/paper/fig2_response_draft.tex`
**Data:** `dev/data/visibility_table.npz` (118 z_h × 4000 x, PDE single bursts),
`dev/data/visibility_flatness.json` (output)

## Framing correction

Fig. 2 (`fig:visibility`, paper.tex:692) is **not** a spectroxide-vs-CosmoTherm comparison. It
compares visibility functions fitted to spectroxide PDE spectra against the Chluba (2013, 2015)
*fitting formulas*. The CosmoTherm spectral comparison is Fig. 3. Any response text must say
"literature fitting formulas," not "CosmoTherm." The distinction matters: Chluba's coefficients
are themselves the output of a fit, with its own choice of weighting, x-range and z-range, none
of which are published in enough detail to reproduce.

## Result 1 — the cost difference is real but small, and mostly irreducible

Cost is the x³-weighted spectral residual over x ∈ [0.5, 20] summed over 118 redshifts
(111,864 points), the same functional minimised in `fit_visibility_from_table.py`.

| model | cost | free params |
|---|---|---|
| null (Δn = 0) | 44856.3 | 0 |
| Chluba 2013/2015 parameters | 464.685 | 7 |
| our fit (`spectral_05_20`, as quoted in Table 1) | 447.723 | 7 |
| global minimum, 7 free (differential evolution) | 447.605 | 7 |
| global minimum, 9 free (z_th, α_th released) | 447.598 | 9 |
| **per-redshift free amplitudes (hard floor)** | **445.171** | 236 |

The floor is the key number. Under NC-stripping the g_bb basis vector vanishes **exactly** — not
merely to numerical precision: the strip computes `G - [∫x²G/∫x²G]·G`, the same expression in
numerator and denominator, so `G_nc` is bitwise 0.0. The three-component Ansatz therefore retains
exactly **two** free amplitudes per redshift,

    a_μ(z) = (3/κ_c) J_μ(z) J_bb*(z),    a_y(z) = ¼ J_y(z),

and an unconstrained per-redshift least-squares fit of those two amplitudes is a rigorous lower
bound on the cost of *any* visibility parameterisation. (It is a genuine bound over any functional
form, not just this Ansatz: the parametric model per redshift lies in span{M_nc, Y_nc}, and the
least-squares fit minimises over that whole span independently at each z. It is attained partly
with amplitudes the parametric family cannot reach — a_μ < 0 at 2/118 redshifts, a_y < 0 at
12/118 — which only makes it looser, hence still valid.)

> **Scope warning, added after Result 5.** Everything in this section is a property of the
> *NC-stripped* metric, which is the metric the paper's Fig. 2 uses. It does **not** generalise:
> without the strip only 51.8% of the residual is irreducible, not 95.8%, and the parameters
> control 48.2% rather than 4.2%. See Result 5. Do not quote "95.8% irreducible" as a statement
> about the Ansatz; quote it as a statement about Fig. 2's fit metric.

Consequences:

- `C_floor / C_lit = 0.9580` — **95.8% of the residual at the literature parameters cannot be
  removed by any choice of parameters, on this metric.** It is Ansatz misfit (the three-component
  form cannot represent the μ–y transition), not a parameter error.
- Only 4.20% of `C_lit` is reachable by the parameters, and our fit realises 3.65% of that.
- As a weighted RMS residual relative to the weighted signal: **10.178% (literature) → 9.991%
  (our fit) → 9.962% (floor)**. The entire Fig. 2 parameter difference moves the misfit by
  0.19 percentage points on a ~10% misfit.
- Better framing than "95.8% irreducible", which is arithmetically right but understates the
  point: **7 parameters get within 0.6% of a 236-parameter fit.**

Two caveats on this metric, both important:

1. **This cost function is structurally blind to the G_bb / temperature-shift component of the
   Ansatz.** Since `G_nc ≡ 0` identically, the `¼(1 − J_bb*) G_bb` term contributes nothing to the
   cost (verified: cost with and without it agrees to 12 digits). Agreement or disagreement
   measured on this cost says nothing about that third component, and nothing about z_th or α_th
   except through their appearance in J_bb* inside the μ term.
2. `C_lit/C_fit = 1.038`. This is *not* a flat likelihood: a 3.8% cost reduction from 7 parameters
   over 10⁵ points would be overwhelming if the residual were noise. It is insignificant because
   the residual is systematic Ansatz error, not noise — which is why the observational test in
   Result 3 is the load-bearing one, and why the flatness framing should not be used on its own.

Incidental: the parameters quoted in Table 1 are **not** the minimum of the stated cost.
Differential evolution over the 7 free parameters reaches 447.605 (0.118 lower) and over all 9
reaches 447.598 at z_th = 1.943×10⁶, α_th = 2.484. Table 1's fit is 0.026% above the global
optimum. Immaterial to every conclusion here, but do not describe it as "the best fit".

## Result 2 — which parameters control it, and which are unconstrained

Setting one parameter back to its literature value, holding the other six at the fitted values:

| param | fit | lit | Δ | ΔC | flat range (ΔC ≤ 16.96) |
|---|---|---|---|---|---|
| z_y | 63128 | 60000 | −4.96% | **14.996** | [5.99e4, 6.70e4] |
| A | 0.99182 | 0.983 | −0.89% | 1.743 | [0.964, 1.019] |
| α_μ | 1.9466 | 1.88 | −3.42% | 0.648 | [1.635, 2.336] |
| α_y | 2.6534 | 2.58 | −2.77% | 0.620 | [2.300, 3.077] |
| z_μ | 58429 | 58000 | −0.73% | 0.112 | [5.46e4, 6.22e4] |
| B | 0.04828 | 0.0381 | −21.1% | 0.075 | [0, 0.145] |
| β | 2.0719 | 2.29 | +10.5% | **0.016** | [0.62, 6.22] |

**z_y carries 88% of the entire cost difference** (15.0 of the joint 17.1). Everything else is
≤ 1.75. Note the single-parameter ΔC values sum to 18.21 while the joint LIT−FIT gap is 16.96, so
the decomposition is non-additive: z_y is 88% of the *joint* gap, not 100% of the effect. Note also
that A at 1.743 sits only 3% below the 1.8 bound; do not restate this as "all well under 2".

B and β — the two parameters with the largest percentage deviations quoted in Table 1, and
therefore the ones most likely to draw referee attention — are together worth ΔC = 0.09, i.e.
0.02% of the cost. Hessian eigenvalues in ln p span 1.51 to 4.8×10⁴ (condition number 3.2×10⁴);
the two softest directions are dominated by β and B, requiring 475% and 356% excursions in ln p
to change the cost by the literature-vs-fit amount.

## Result 3 — the difference is not observable (the load-bearing test)

The cost function has no noise model, so its absolute value is meaningless. Rebuilding the
question against the FIRAS 43×43 covariance: for the same energy release, take the difference
between the two predicted spectra, convert to kJy/sr, and compute the significance marginalised
over the blackbody temperature-shift template (FIRAS cannot measure the absolute CMB
temperature, so any G_bb component is unobservable by construction) and the galactic dust
template. Δρ/ρ limits are computed self-consistently per z_h with the Green's function as the
signal template and the same marginalisation.

| z_h | FIRAS 95% limit on Δρ/ρ | difference at that limit | Δχ² | + amplitude profiled | **shift in the limit** |
|---|---|---|---|---|---|
| 3e3 | 3.23e-5 | 0.0007 σ | 0.000000 | 1.5e-4 σ | +0.02% |
| 1e4 | 3.21e-5 | 0.0028 σ | 0.000008 | 8.6e-4 σ | −0.05% |
| 3e4 | 3.34e-5 | 0.0348 σ | 0.001210 | 3.0e-3 σ | −1.69% |
| 5e4 | 3.75e-5 | 0.0770 σ | 0.005923 | 4.8e-3 σ | −3.56% |
| **8e4** | 4.38e-5 | **0.0841 σ** | **0.007065** | 4.8e-3 σ | **−3.73%** |
| 1.5e5 | 5.32e-5 | 0.0361 σ | 0.001301 | 1.4e-3 σ | −1.58% |
| 3e5 | 5.93e-5 | 0.0199 σ | 0.000397 | 4.3e-5 σ | −0.86% |
| 1e6 | 7.25e-5 | 0.0117 σ | 0.000136 | 1.8e-5 σ | −0.50% |
| 2e6 | 1.74e-4 | 0.0044 σ | 0.000019 | 4.6e-6 σ | +0.19% |

**Worst case Δχ² = 0.0071 at z_h = 8×10⁴.** FIRAS would need an energy release ~12× its own 95%
upper limit to distinguish the two parameterisations at 1σ.

z_h ≥ 3×10⁶ is excluded: the Green's function is essentially zero there (μ ~ 4×10⁻¹⁰ at 5×10⁶),
the fitted "limit" on Δρ/ρ runs to 1.3, and the reported σ is a ratio of two vanishing numbers. The
template there is nearly pure G_bb, degenerate with the marginalised temperature shift, giving
σ(Δρ/ρ) = 0.60. Quoting z_h = 5×10⁶ as the "worst case" (0.169 σ) would wrongly imply FIRAS
constrains something there; the physically relevant worst case is z_h = 8×10⁴.

**Part of the difference is a pure amplitude rescaling, and therefore doubly unobservable.** Δρ/ρ
is not known a priori, so any component of the LIT−FIT difference along the template itself is
degenerate with the injected energy. Profiling it out (column 5) drops the significance by a factor
3.5–4.7 in the μ era and ~80–100× at z ≥ 3×10⁵; the best-fit rescaling is k ≈ 1.6–3.7% of Δρ/ρ.
Worst Δχ² falls to 0.0027. The quoted numbers are therefore conservative.

**Margin is a factor ~1.8, not a factor of 10.** Under looser limit conventions the numbers move:
at fixed Δρ/ρ = μ_lim/1.401 = 1.15×10⁻⁴ the worst case rises to 0.250 σ, and a 3σ rather than
1.96σ limit convention puts the μ-era point near 0.26 σ. Still far below 1σ, but do not oversell
the headroom.

### The number a referee will actually ask for

"Does it change the constraint?" is answered by how much the derived FIRAS 95% limit on Δρ/ρ moves
between the two parameterisations, not by Δχ² (last column above). Dense scan, 120 points,
z_h ≤ 2×10⁶:

**Worst shift −3.97% at z_h = 6.40×10⁴** (limit 4.07×10⁻⁵), confined to the μ/y transition:
≤1.7% outside 3×10⁴ ≲ z_h ≲ 1.5×10⁵, ≤0.9% for z_h ≥ 3×10⁵, ≤0.05% in the y era.

So both of these are true and both belong in the response:
- FIRAS cannot distinguish the two spectra (Δχ² < 0.008).
- The choice of visibility parameters moves the published Δρ/ρ limit by up to ~4% in the μ/y
  transition region.

Only the first is a "no difference" statement. The second is the honest caveat and is well below
the ~10% systematics already acknowledged elsewhere in the paper, but it should be stated, not
omitted.

### Implementation notes verified

- `_dn_to_dI_kJy` and `firas.x` use a consistent T_CMB (ν round-trips to 2.2×10⁻¹⁶).
- The profiled-χ² formula S² = dᵀC⁻¹d − (dᵀC⁻¹T)(TᵀC⁻¹T)⁻¹(TᵀC⁻¹d) is the correct expected Δχ²
  between two models with linear nuisances. The G_bb part of the difference lies exactly along
  T_dT and is removed identically, consistent with Result 1's `G_nc ≡ 0`.
- Latent fragility, unrelated to this analysis: `_galactic_dust_template_kJy` returns an arbitrary
  normalisation, |T_gal| ≈ 5.4×10³⁰ kJy/sr, so `np.linalg.cond` on the Fisher matrix overflows to
  `inf`. The scale-invariant condition number is 2.25 and rescaling T_gal to unit maximum changes
  the worst-case σ in the 5th digit. Harmless here; worth normalising in the module.

## Result 4 — the literature values lie inside the spread of our own equally-good fits

All eight stored fit variants, re-evaluated on one common cost function:

| variant | z_y | α_y | z_μ | α_μ | A | B | β | cost |
|---|---|---|---|---|---|---|---|---|
| literature | 60000 | 2.58 | 58000 | 1.88 | 0.983 | 0.0381 | 2.29 | 464.685 |
| diff_evolution | 63504 | 2.653 | 58508 | 1.947 | 0.9906 | **3.5e-06** | 2.482 | 447.598 |
| basin-hopping | 63471 | 2.653 | 58505 | 1.947 | 0.9908 | **0.0466** | 2.444 | 447.602 |
| l-bfgs-b (warm) | 63421 | 2.650 | 58364 | 1.949 | 0.9902 | 0.0459 | 2.445 | 447.611 |
| **paper fit** | 63128 | 2.653 | 58429 | 1.947 | 0.9918 | 0.0483 | 2.072 | 447.723 |
| extended_x_03_25 | 62890 | 2.613 | 57338 | 1.964 | 0.9833 | 0.0377 | 2.498 | 448.426 |
| l-bfgs-b (std) | 62650 | 2.621 | 57055 | 1.969 | 0.9864 | 0.0416 | 2.476 | 448.626 |
| spectral_1_15 | 62757 | 2.614 | 56794 | 1.938 | 0.9846 | 0.0400 | 2.687 | 449.174 |
| narrow_x_1_15 | 62767 | 2.613 | 56783 | 1.939 | 0.9842 | 0.0405 | 2.477 | 449.176 |

Every fitted variant lies within **0.352%** in cost. Across them:

- **B spans 3.5×10⁻⁶ to 0.0483 — a factor 10⁴ — over a cost range of 0.001%.** Two independently
  converged global optimisers (differential evolution, basin hopping) differ by that factor at
  costs 447.598 vs 447.602. B is not a measured quantity.
  **Mechanism: B is degenerate with z_th, not flat on its own.** At fixed z_th = 1.98×10⁶,
  setting B = 0 costs ΔC = +1.67 (0.37%), which is not negligible. Differential evolution reaches
  B ≈ 0 only by simultaneously moving z_th to 1.943×10⁶ (−1.9%). Do not claim B is unconstrained
  at fixed z_th; the defensible statements are (i) the literature value costs ΔC = 0.075 at fixed
  everything else, and (ii) a 1.9% move in z_th buys B → 0 at equal total cost.
- **β spans 2.07 to 2.69** (26%). At fixed everything else, the literature β = 2.29 costs
  ΔC = 0.016; β = 1.5 costs 0.199 and β = 4.0 costs 0.678, so the basin is genuinely wide here.
- The literature values of **B (0.0381) and β (2.29) both fall inside these ranges.** The +27% and
  −9.5% deviations quoted in Table 1 are smaller than the scatter of our own fits.
- The five primary parameters are stable across our fits (0.9–3.0% spread), and the literature
  values fall just outside — consistent with Result 2, where z_y is the one parameter carrying a
  systematic offset.
- Free fits recover the analytically fixed values: α_th → 2.482–2.536 (analytic 5/2) and
  z_th → 1.94–1.98×10⁶ (analytic 1.98×10⁶). Independent support for the thermalisation physics.

## Honest residual: z_y

The z_y offset (+5.0%) is **not** explained away by degeneracy. It is stable across our own fits
(spread 1.35%, all clustered at 62.6–63.5×10³, none near 6.0×10⁴) and it carries 88% of the cost
difference. There is a real, small difference between the y-era visibility fitted to spectroxide
PDE spectra and Chluba's published J_y.

**Tested and rejected: the temperature-shift convention.** The leading candidate was that the
offset comes from how the unobservable blackbody direction is removed before the μ/y split.
`dev/scripts/visibility_diagnostics/visibility_zy_convention_test.py` refits (z_y, α_y) alone under three residual
definitions:

| residual definition | z_y | α_y | z_y vs lit | ΔC to lit |
|---|---|---|---|---|
| NC-stripped (paper fit) | 63396 | 2.655 | +5.66% | 15.03 |
| raw Δn, free G_bb amplitude profiled per z (what an observer sees) | 65065 | 2.619 | **+8.44%** | 30.13 |
| raw Δn, G_bb fixed by the Ansatz | 63225 | 2.656 | +5.37% | 13.62 |

z_y lands at 6.32–6.51×10⁴ regardless, and the observationally correct `freeT` definition pushes
it *further* from Chluba's 6.0×10⁴, not closer. **The offset is not a stripping artefact.**

Remaining untested candidates:

1. Chluba's J_y (C13 Eq. 5) may have been fitted to a different observable (integrated μ, y
   rather than the full spectral shape), a different x-range, z-range, or weighting. None of
   these are specified in enough detail in C13 to reproduce.
2. A genuine difference in the low-z Compton-y buildup between the two solvers. Note Fig. 3
   shows sub-percent spectral agreement with the CosmoTherm Green's function table over most of
   x, which bounds how large any such difference can be.

The defensible statement for the paper is that the offset is real, robust to the fit convention,
localised to z_y, and observationally irrelevant (Result 3) — **not** that it is a fitting
degeneracy. Do not claim the likelihood is flat in z_y; it is not.

## Recommended paper changes

1. Table 1 caption / surrounding text: state that B and β lie within the spread of equally-good
   fits, quoting the B range 3.5×10⁻⁶–0.048 at fixed cost. This is stronger than the current
   hand-wave that they "are less well constrained."
2. Add the Result 3 sentence: at the FIRAS 95%-limit energy release the two parameterisations
   differ by Δχ² < 0.008 for all z_h, so the Fig. 2 differences do not propagate to constraints.
3. Add the Result 1 floor: ~96% of the Fig. 2 residual is the three-component Ansatz, not the
   parameters. This also strengthens the existing text about the Ansatz being "an inherently
   limited approximation" in the μ–y transition.
4. Attribute the residual difference to z_y specifically, and say so rather than leaving the
   reader to infer that all seven parameters disagree.

## Verification status

All four load-bearing claims were re-derived by a fresh-context verifier with independent
implementations (own `G_bb`/`M`/`Y`/`J` closed forms, own FIRAS covariance construction and
inverse, own profiled-χ²), not by re-running these scripts:

- Costs 464.685 / 447.723 reproduced to 6 significant figures. β_μ re-derived from
  ∫x²M dx = 0 by quadrature as 3G₂/(2ζ(2)) = 2.192288908205, matching `constants` to 3×10⁻¹³, and
  3/κ_c = 1.400657 against the 1.401 validation target.
- `G_nc` confirmed bitwise zero; the floor 445.171 confirmed and confirmed to be a valid bound
  (30-start L-BFGS-B plus differential evolution over all 9 parameters bottom out at 447.598,
  above the floor as required).
- Single-parameter ΔC table reproduced exactly.
- FIRAS worst case confirmed by an 801-point dense log scan on [3×10³, 5×10⁶], so no
  between-node maximum was missed. T_CMB consistency between `firas.x` and `_dn_to_dI_kJy`
  confirmed to 2.2×10⁻¹⁶; independent G_bb template matches `gbb_template_kJy()` to 4.4×10⁻¹⁶.
- The −3.97% limit shift was found independently by the verifier (−3.96% at z_h ≈ 6.3×10⁴) before
  it was added to `visibility_firas_indistinguishability.py`.

The B–z_th degeneracy in Result 4 was found afterwards, when the claim "B is unconstrained" was
checked at fixed z_th; the original wording was wrong and is corrected above.

## Result 5 — refitting without the NC strip

**Scripts:** `dev/scripts/visibility_diagnostics/visibility_fit_no_ncstrip.py`,
`dev/scripts/visibility_diagnostics/visibility_raw_fit_diagnosis.py`, `dev/scripts/visibility_diagnostics/visibility_fit_jmu_jy_raw.py`
**Data:** `dev/data/visibility_fit_no_ncstrip.json`, `dev/data/visibility_fit_jmu_jy_raw.json`

Result 1 is a property of the stripped metric. Dropping the strip changes the accounting
substantially, so the "95.8% irreducible" statement must be scoped to Fig. 2's fit and not
presented as a property of the Ansatz.

| metric | C_lit | C_paper | floor | params in floor | reducible by params | C_lit/C_paper |
|---|---|---|---|---|---|---|
| NC-stripped (Fig. 2) | 464.685 | 447.723 | 445.171 | 236 | **4.2%** | 1.038 |
| raw | 494.290 | 461.958 | 256.011 | 354 | **48.2%** | 1.070 |
| freeT (G_bb profiled per z) | 419.125 | 385.739 | 256.011 | 236 | 38.9% | 1.087 |

The `raw` and `freeT` floors are identical (256.011) as they must be: projecting out G and fitting
M⊥, Y⊥ leaves the same residual as fitting span{M, Y, G}. Independent check that both are right.

### 5a. Fitting all 7–9 parameters to un-stripped spectra fails

| fit | cost | above floor | pathology |
|---|---|---|---|
| raw, 7 free | 375.00 | +46.5% | A = 1.097 (**unphysical**), β = 1.0 at bound |
| raw, 9 free | 370.77 | +44.8% | A = 1.100 at bound, α_th = 1.610 (−36%) |
| freeT, 7 free | 270.57 | +5.7% | A = 1.090, β = 1.0 at bound |
| freeT, 9 free | 268.21 | +4.8% | A = 1.100, B = 0.300 both at bounds, α_th = 1.535 |

A > 1 is inadmissible: A = J_bb*(z→0) is a thermalised energy fraction. Constraining A ≤ 1 pins it
at exactly 1.000 and the raw 9-free cost rises to 446.50, 74.4% above the floor. So on un-stripped
spectra the Ansatz is a poor fit and the optimiser only reaches its nominal minimum by leaving the
physical region.

### 5b. Root cause: an M↔G degeneracy the Ansatz cannot follow

Comparing the amplitudes a free per-redshift 3-component fit prefers against the Ansatz prediction
(a_μ = (3/κ_c)J_μJ_bb*, a_y = ¼J_y, a_G = ¼(1−J_bb*)), the misfit is confined to the μ–y
transition and is a correlated trade-off, not two independent failures:

- In 2×10⁴ < z_h < 3×10⁵ the free fit wants **a_μ larger by +0.239** and **a_G smaller by −0.053**,
  and these two deviations are anticorrelated at **r = 0.9918**.
- M and G_bb are **75% correlated** in the x³-weighted metric (Gram matrix off-diagonal 0.7516;
  M·Y = 0.6664, Y·G = 0.2308). The basis is far from orthogonal, so the split between "μ amplitude"
  and "temperature shift" is poorly determined in the transition.
- The Ansatz's only lever along that direction is A, which is why A runs monotonically upward: the
  floor with a_G fixed falls from 521.4 at A = 0.96 to 473.9 at A = 0.983 to 443.8 at A = 1.000 and
  keeps falling to 352.4 at A = 1.10, the bound. **A is not measured by this metric; it runs away.**
  Do not report "the data prefers A = 1" — the data prefers A > 1, which is unphysical, and A = 1 is
  merely the best admissible value.

Free a_G by regime, against the Ansatz at literature parameters:

| regime | z_h | free a_G | Ansatz (A=0.983) |
|---|---|---|---|
| y era | < 2×10⁴ | −0.00289 | +0.00425 |
| transition | 2×10⁴–3×10⁵ | **−0.04815** | +0.00458 |
| μ era | 3×10⁵–2×10⁶ | +0.02653 | +0.03247 |
| thermalisation | 2×10⁶–6×10⁶ | **+0.22014** | +0.22063 |

The sign flip in the transition is the degeneracy above, not evidence that the PDE lacks a
temperature shift. Where the thermalisation branch dominates it is confirmed to 0.2% (and to 0.02%
at z_h = 4.2×10⁶: 0.24973 vs 0.24968), which is a genuine validation of ¼(1−J_bb*) that the
stripped metric cannot perform.

### 5c. Fixing J_bb* and fitting only J_μ, J_y — the well-posed version

J_bb* is the one visibility function with an analytic backbone (z_th, α_th from the DC opacity
scaling), so it should be held fixed rather than fitted. Freeing only z_y, α_y, z_μ, α_μ:

| configuration | z_y | α_y | z_μ | α_μ | cost | above floor | C_lit/C_fit |
|---|---|---|---|---|---|---|---|
| raw, J_bb* lit (A=0.983) | 64218 | 2.6687 | 59593 | 1.9797 | 478.41 | +0.95% | 1.0332 |
| raw, J_bb* lit but A=1 | 63409 | 2.6921 | 59118 | 1.9375 | 446.65 | +0.64% | 1.0238 |
| NC-stripped, J_bb* lit | 63590 | 2.6186 | 58016 | 1.9524 | 448.26 | +0.69% | 1.0366 |
| literature | 60000 | 2.58 | 58000 | 1.88 | — | — | — |

No parameter hits a bound, and every configuration lands within 1% of its 236-parameter floor. The
earlier failure was entirely in the J_bb* sector.

**Main conclusions, all of which strengthen the referee response:**

1. **The z_y offset is confirmed without the strip.** z_y = 63.4–64.2×10³ across all three
   configurations, i.e. **+5.7% to +7.0% above Chluba's 6.0×10⁴**, versus +5.2% for the published
   NC fit. Dropping the strip does not move it toward the literature value. Combined with the
   convention test above, the offset is robust to every residual definition tried.
2. α_y comes out +1.5% to +4.4%, α_μ +3.1% to +5.3%, z_μ +0.03% to +2.8%. Same sign and rough
   magnitude as Table 1.
3. **The near-degeneracy of J_μ and J_y survives dropping the strip**: C_lit/C_fit = 1.024–1.037,
   comparable to the 1.038 of the published metric. The Fig. 2 argument does not depend on the
   strip.
4. Four parameters reach within 0.64% of a 236-parameter per-redshift fit.

### 5d. What this changes in the paper

- Scope the "most of the residual is irreducible" statement to the stripped metric, or drop it and
  lead with 5c-4 (four parameters within 0.64% of a per-redshift free fit), which holds on both.
- The z_y offset should now be reported as robust across three independent residual definitions,
  which is a stronger and more honest statement than the earlier single-metric result.
- Consider stating the M↔G non-orthogonality (0.75) as the reason the μ/ΔT split is ill-determined
  in the transition. It is a cleaner explanation of the Fig. 2 residual structure than "the
  three-component Ansatz is an inherently limited approximation", and it is quantitative.
- Do **not** add a claim that the data prefers A = 1. A runs to its bound.

---

## Result 6 — the fit cost was grid-dependent; correcting it removes most of the Fig. 2 discrepancy

**This supersedes the "Honest residual: z_y" section above.** That section concluded the z_y offset
was a property of the spectra because it survived three different residual definitions. All three
shared the same defective cost, so the test was blind to the actual cause.

### 6a. The defect

`dev/scripts/fit_visibility_from_table.py`, `make_cost()`, builds `w = x_m ** weight_power` and
minimises the sum of `(w * residual)**2` over frequency grid nodes. There is no `dx`. The cost is
therefore not a functional of the spectra: its effective weight is `x^3` times the local node
density of whatever grid the spectra happen to live on. Our PDE grid is log-spaced at low x and
linear at high x for solver reasons (`src/grid.rs`), so its density profile differs sharply from
CosmoTherm's database grid:

| x range | our node share | CosmoTherm share | ratio |
|---|---|---|---|
| [0.5, 1) | 7.2% | 3.4% | 2.14 |
| [1, 2) | 8.6% | 6.7% | 1.29 |
| [2, 5) | 16.1% | 20.1% | 0.80 |
| [5, 10) | 22.7% | 33.5% | 0.68 |
| [10, 20) | 45.4% | 36.4% | 1.25 |

### 6b. Isolation (`dev/scripts/visibility_diagnostics/visibility_zy_sampling_isolation.py`)

Our spectra, held fixed, fitted for (z_y, α_y, z_μ, α_μ) with J_bb* fixed at the literature form:

| grid swapped | z_y | vs 6.0×10⁴ |
|---|---|---|
| native x, native z | 64218 | +7.03% |
| **CT x, native z** | **60754** | **+1.26%** |
| native x, CT z | 64137 | +6.90% |
| CT x, CT z | 60626 | +1.04% |

The x-node distribution moves z_y by 5.4%; the redshift sampling moves it by 0.13%. Reweighting
the redshift sum trapezoidally in ln z (64144) or normalising each redshift by its own signal power
(64552) changes nothing, confirming this is not a z-weighting effect.

### 6c. The fix (`dev/scripts/visibility_diagnostics/visibility_zy_quadrature_fix.py`)

Replacing the node sum with a quadrature restores grid independence:

| measure | ours (native x) | ours (CT x) | CosmoTherm | grid dependence |
|---|---|---|---|---|
| node sum (paper) | 64218 | 60626 | 60091 | 5.9% |
| ∫ · dx | 60745 | 60620 | 60085 | 0.2% |
| ∫ · d ln x | 73761 | 73772 | 72813 | 0.01% |

Both quadratures are grid-independent, so grid-independence alone does not select one. `dx` is the
physically motivated choice: FIRAS channels are uniform in ν and the observable is ΔI ∝ x³Δn, so
∫(x³Δn)² dx is the natural least-squares. It is also the one that reproduces Chluba's values.

### 6d. Fitting CosmoTherm's own Green's function

`dev/scripts/visibility_diagnostics/visibility_fit_cosmotherm_gf.py` runs CosmoTherm's `Greens_data.dat` through the same
pipeline. The two codes track each other far more closely than the metric ambiguity does:

| metric | CosmoTherm z_y | spectroxide z_y | code-to-code |
|---|---|---|---|
| x⁰, [0.5,20] | 151300 | 154100 | +1.9% |
| x², [0.5,20] | 90100 | 91800 | +1.9% |
| x³, [0.5,20] | 60091 | 60626 | +0.9% |
| x⁴, [0.5,20] | 45554 | 45631 | +0.2% |
| x³, [1,10] | 61316 | 61907 | +1.0% |
| x³, [0.1,30] | 60310 | 60841 | +0.9% |

z_y spans a factor 3.4 across weighting choices; the code-to-code difference is ≤2% throughout.
The spectra themselves agree to 1.14% x³-weighted RMS on the common grid, of which ~0.4 points is
a pure amplitude offset, leaving 0.6–1.1% in shape.

### 6e. Table 1 recomputed (`dev/scripts/visibility_diagnostics/visibility_refit_quadrature_table1.py`)

Same configuration as the paper (NC-stripped, 7 free, z_th and α_th fixed analytic), `dx` measure:

| param | quadrature fit | literature | new dev | published dev |
|---|---|---|---|---|
| z_y | 60150 | 60000 | **+0.25%** | +5.21% |
| α_y | 2.6265 | 2.58 | +1.80% | +2.84% |
| z_μ | 56425 | 58000 | −2.72% | +0.74% |
| α_μ | 1.9306 | 1.88 | +2.69% | +3.54% |
| A | 0.98699 | 0.983 | +0.41% | +0.91% |
| B | 0.044161 | 0.0381 | +15.91% | +26.73% |
| β | 2.4153 | 2.29 | +5.47% | −9.50% |

C_lit/C_fit falls from 1.0382 to 1.0117. Every parameter except B (degenerate with z_th, Result 4)
now agrees to ≤5.5%. z_μ is the only one that gets worse. Re-running the paper's node-sum fit
reproduces 63505 against the published 63128, so the published fit sat at its own minimum: the
defect is the metric, not the optimiser.

### 6f. Downstream FIRAS numbers, recomputed

`visibility_firas_indistinguishability.py` updated to the corrected parameters (the node-sum values
are retained as `FIT_NODESUM`). Everything shrinks:

| quantity | published fit | corrected fit |
|---|---|---|
| worst FIRAS 95%-limit shift | −3.97% at z_h = 6.4×10⁴ | **−1.60% at z_h = 6.4×10⁴** |
| worst Δχ² (constrained range) | 0.0071 at z_h = 8×10⁴ | **0.0013 at z_h = 8×10⁴** |
| worst significance | 0.084σ | **0.036σ** |
| energy release for 1σ separation | ~10× FIRAS limit | **28× FIRAS limit** |

### 6g. What this changes in the paper

1. **Fix the cost function** in `fit_visibility_from_table.py` (add the `dx` quadrature) and
   regenerate Fig. 2 and Table 1. This is a defect, not a convention choice: the published numbers
   partly measure our own frequency grid.
2. Table 1's deviation column becomes ≤2.7% for everything except B (+15.9%) and β (+5.5%).
3. State the metric explicitly in the caption, along with the sensitivity: z_y ranges over
   45.6–154×10³ across weighting exponents x⁰ to x⁴, so the metric must be quoted for the
   comparison to mean anything. Chluba (2013) does not specify it.
4. The strongest referee-facing statement is now available: running CosmoTherm's own Green's
   function database through our fit reproduces Chluba's published z_y to 0.14%, and our spectra
   to 0.25%.
5. Retract the "z_y offset is real and robust" line from the earlier draft.

### 6h. Verification status

Sub-claims 6a–6d were sent to a fresh-context claim-verifier with an independent reimplementation
(claim, data-file spec, and Ansatz only — no reasoning). Record the outcome here when it returns.

---

## Result 7 — reconstructing Chluba's fit procedure

EB's observation: in Chluba's original treatment `J_therm` carries **no free parameters**. If so,
the fit that produced (z_y, α_y, z_μ, α_μ) had exactly four, with J_bb* fixed at the analytic
exp(−(z/1.98×10⁶)^{5/2}). He also describes it as "a least squares fit".

### 7a. The procedure that reproduces his four published numbers

`dev/scripts/visibility_diagnostics/visibility_scalar_leastsq_chluba.py`. Decompose each Green's function into free
per-redshift amplitudes (a_M, a_Y), convert to visibilities via the Ansatz,
`J_y(z_k) = 4 a_Y`, `J_μ(z_k) = a_M κ_c / (3 J_bb*)`, then fit each visibility **separately** by
ordinary least squares to its own O(1) sequence. Weighting x³, i.e. least squares on the intensity
ΔI ∝ x³Δn.

| dataset | J_bb* | z_y | α_y | z_μ | α_μ | RMS dev |
|---|---|---|---|---|---|---|
| CosmoTherm | analytic | 60147 (+0.25%) | 2.573 (−0.27%) | 57560 (−0.76%) | 1.892 (+0.65%) | **0.53%** |
| CosmoTherm | analytic, NC | 60147 (+0.24%) | 2.570 (−0.39%) | 57511 (−0.84%) | 1.900 (+1.09%) | 0.73% |
| spectroxide | analytic, NC | 60222 (+0.37%) | 2.588 (+0.30%) | 57522 (−0.82%) | 1.849 (−1.64%) | **0.95%** |
| spectroxide | analytic | 60210 (+0.35%) | 2.593 (+0.50%) | 57508 (−0.85%) | 1.849 (−1.65%) | 0.98% |
| Chluba (2013) | — | 60000 | 2.58 | 58000 | 1.88 | — |

All four parameters land simultaneously within 1.7%. This is a reproduction of his procedure, not
a tuning: the scan covered 2 datasets × 2 J_bb* forms × 2 residual treatments × 4 weightings and
the winner is the physically motivated corner of it.

### 7b. Analytic J_bb* beats the literature form in every pairing

| dataset / residual | analytic RMS | literature RMS |
|---|---|---|
| CosmoTherm, sub | 0.53% | 2.91% |
| CosmoTherm, NC | 0.73% | 2.28% |
| spectroxide, sub | 0.98% | 1.27% |
| spectroxide, NC | 0.95% | 1.63% |

There is a mechanism, and it connects to Result 5. J_μ is bounded above by 1, and the μ-era plateau
of the extracted a_M is (3/κ_c)·J_μ·A. With A = 0.983 the extraction returns J_μ → 1.017 > 1 in the
μ era, which the Ansatz cannot represent, so z_μ and α_μ distort to compensate. With A = 1 it
returns J_μ → 1 correctly. This is the same conclusion the un-stripped 9-parameter fit reached from
the other direction, where A ran to its upper bound: **the data wants A ≥ 1, and A = 1 is the
physical value.** A = 0.983 is not a property of the spectra.

### 7c. The weighting is x³, unambiguously

| p | best RMS dev over all configurations |
|---|---|
| 0 (least squares on Δn) | 95.0% |
| 2 | 28.6% |
| **3 (least squares on ΔI ∝ x³Δn)** | **0.53%** |
| 4 | 16.3% |

### 7d. Under his procedure the two codes agree

z_y 60222 (ours) vs 60147 (CosmoTherm), 0.12%. z_μ 57522 vs 57511, 0.02%. Largest code-to-code
difference is α_μ, 1.849 vs 1.900 (2.7%).

### 7e. NC strip vs analytic subtraction

Parameters barely differ (RMS 0.53% vs 0.73% for CosmoTherm). With J_bb* fixed analytically both
treatments handle the temperature shift consistently. The strip fits the J_μ *sequence* noticeably
worse (residual RMS 0.031 vs 0.009 for CosmoTherm, 0.050 vs 0.013 for ours), so prefer the analytic
subtraction if we adopt this procedure.

### 7f. Why the earlier repo scalar fits failed

`dev/data/visibility_scalar_fit.json` (z_y = 1.38×10⁵) and `visibility_scalar_ec_fit.json`
(z_y = 6.6×10⁴) minimised **relative** error, which puts almost all weight on the tails where μ and
y vanish. They do not test this reading of "least squares".

### 7g. What this means for the paper

Our published Table 1 compares a 7-parameter joint spectral fit against parameters obtained from a
4-parameter scalar fit with J_therm fixed. That is not like-for-like, and it is the larger part of
the apparent discrepancy: the joint fit lets A, B and β absorb structure that Chluba's procedure
assigns to J_μ and J_y, on top of the grid-dependent cost of Result 6. Recommend reporting Table 1
under Chluba's procedure, with the joint fit as a secondary column if wanted.

### 7h. Robustness to the redshift range

`dev/scripts/visibility_diagnostics/visibility_scalar_zrange_robustness.py`. The redshift range was the one lever the
Result 7 scan never varied, and it is not innocuous: J_μ and J_y are sigmoids, so how much of the
flat tails is included changes the leverage on the transition.

| dataset | z range | z_y | α_y | z_μ | α_μ | RMS dev |
|---|---|---|---|---|---|---|
| CosmoTherm | [10⁴, 5×10⁶] | 60150 | 2.574 | 57580 | 1.883 | **0.41%** |
| CosmoTherm | full table | 60147 | 2.573 | 57560 | 1.892 | 0.53% |
| CosmoTherm | [3×10⁴, 3×10⁵] | 60272 | 2.599 | 57474 | 1.779 | 2.75% |
| spectroxide | full table | 60210 | 2.593 | 57508 | 1.849 | 0.98% |
| spectroxide | [10⁴, 5×10⁶] | 60211 | 2.593 | 57515 | 1.843 | 1.12% |
| spectroxide | [3×10⁴, 3×10⁵] | 60268 | 2.603 | 57438 | 1.781 | 2.71% |

Spread across all wide ranges: z_y 0.00–0.01%, α_y 0.03–0.06%, z_μ 0.01–0.04%, α_μ 0.34–0.47%.
Only the narrow transition-only window degrades the agreement, and then only to 2.7%. The
reconstruction is robust, not tuned. Best case: all four CosmoTherm parameters within **0.72%** of
the published values.

### 7i. The scalar route is required, not merely sufficient

`dev/scripts/visibility_diagnostics/visibility_reverse_engineer_chluba.py` runs the same procedure space (2 datasets ×
2 J_bb* forms × 2 residual treatments × 2 measures × 5 weightings = 80 fits) but fits the
**spectra** directly rather than the extracted visibility sequences. The best it reaches is:

| dataset | J_bb* | resid | meas | p | z_y | α_y | z_μ | α_μ | RMS |
|---|---|---|---|---|---|---|---|---|---|
| spectroxide | literature | NC | dx | 3 | +0.32% | +1.06% | −3.15% | +2.79% | 2.17% |
| spectroxide | analytic | sub | dx | 3 | +0.07% | +3.99% | −1.30% | +2.13% | 2.36% |
| CosmoTherm | analytic | sub | dx | 3 | −0.21% | +3.59% | −1.45% | +4.61% | 3.01% |

against 0.41–0.53% for the scalar route. The direct spectral fit systematically misses α_y by
+3.2% to +4.2%, where the scalar fit gets −0.27%. Mechanism: in a joint spectral cost the μ term
carries a_M ≈ 1.4 against a_Y ≈ 0.25, so μ has ~5.6× the leverage and α_y is weakly determined.
Fitting each visibility on its own O(1) scale removes that imbalance.

Both routes independently select p = 3 and the `dx` quadrature, which corroborates Result 6.

### 7j. Figure

`dev/scripts/visibility_diagnostics/visibility_figure_chluba_procedure.py` produces
`notebooks/figures/pde_visibility_fit_chluba.pdf`. Applying Chluba's procedure to our spectra:

    z_y = 60210 (+0.35%)   α_y = 2.5928 (+0.50%)
    z_μ = 57508 (−0.85%)   α_μ = 1.8489 (−1.65%)

Peak curve residuals vs Chluba (2013): ΔJ_μ = 0.0084, ΔJ_y = 0.0028. Unlike the published figure
this plots the extracted J_y points as well as J_μ; the old version fixed J_y from the formula and
never showed it.

**Caption caveat.** The visible ΔJ_bb* = 0.0224 is *not* a disagreement. It is the difference
between the analytic J_bb* we hold fixed (A = 1, B = 0) and Chluba's separately fitted refinement
(A = 0.983, B = 0.0381, β = 2.29), and it is there by construction. Say so in the caption or a
reader will misread it.

**Confidence.** The reconstruction of the procedure is an inference from EB's recollection that
J_therm carried no free parameters plus the statement that the fit was least squares. It is
strongly supported (four parameters recovered simultaneously to <1% from CosmoTherm's own database,
robust to redshift range, and the analytic-J_bb* corner wins every pairing) but it is not a
citation. Do not state in the paper that this *is* Chluba's procedure; state that a fit performed
this way reproduces his published parameters.
