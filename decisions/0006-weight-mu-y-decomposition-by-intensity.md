# Weight the μ, y, ΔT/T decomposition by intensity (x³)

## Status

Accepted, 2026-09-24 (EB).

## Context

The code reduces a spectrum Δn(x) to (μ, y, ΔT/T) with two least-squares fits, each in Rust and
Python:

- `decompose_nonlinear_be` (Rust `src/distortion.rs`, Python `greens._decompose_nonlinear_be`):
  the paper's Appendix fit, after Bianchini & Fabbian (2022), with μ inside the Bose–Einstein
  exponential. It is the default `decompose_distortion`, so it sets the μ and y that the solver
  reports (`src/solver.rs`, `decompose_distortion` on the final Δn) and that every figure built
  from solver μ and y shows.
- `decompose_gram_schmidt` (Rust and `greens._decompose_gram_schmidt`): documented as Chluba &
  Jeong (2014, arXiv:1306.5751) Appendix A. It also seeds the nonlinear fit.

Both minimize ∫ (Δn − model)² dx over x ∈ [0.5, 18], with no weight.

Three findings from 2026-09-24 (scripts in `dev/scripts/y_estimator/`):

1. **The Gram–Schmidt fit is not the Chluba & Jeong estimator.** Their vectors are intensities
   in PIXIE-like channels, 30–1000 GHz, uniform in ν, summed without weight. Their Appendix A
   gives the basis norms |Y_SZ|, |M⊥|, |G⊥| ≃ {73.3, 7.99, 21.4} × 10⁻¹⁸ W m⁻² Hz⁻¹ sr⁻¹. Our
   shapes reproduce these to 0.3% under the x³ (intensity) inner product. Under the Δn inner
   product, |M⊥|/|Y_SZ| = 0.230, against their 0.109.
2. **In the μ–y transition, the weighting decides the answer.** The spectra there are not in
   span{M, Y_SZ, G}. Up to 17% of the x³-weighted norm lies outside it near z_h ≈ 6×10⁴, and
   each metric assigns that part differently. On the 118 Table 1 PDE bursts, the unweighted fit
   gives 4y/(Δρ/ρ) = 2.10 at z_h = 6.9×10⁴, and J_μ = J_y at z_h ≈ 1.4×10⁵. The x³-weighted
   Appendix fit and the Chluba & Jeong recipe agree with each other to 0.021 in J_μ and 0.008
   in J_y, put J_μ = J_y at z_h ≈ 4.0×10⁴, and reproduce Chluba & Jeong (2014) Fig. 1 (J_μ peak
   1.23 near 10⁵, J_T minimum −0.41 near 7×10⁴, J_R minimum −0.042 near 4×10⁴). CosmoTherm's
   Green's-function database gives the same numbers as the PDE through each fit, so the
   difference is the estimator, not the solver.
3. **The Appendix's stated reason for the nonlinear form does not hold at small μ.** It says
   keeping μ inside the exponential breaks the μ–temperature degeneracy of the linear fit. At
   μ ∼ 10⁻⁵ the nonlinear fit equals the linear {M, G, Y_SZ} fit to 0.5%.

Instruments measure intensity, ΔI ∝ x³Δn. FIRAS channels and the PIXIE-like band are close to
uniform in ν. An unweighted Δn fit puts most of its weight at the low-x end of the band, where
Δn ∝ 1/x.

## Decision

1. Both fits minimize the intensity residual, ∫ [x³(Δn − model)]² dx, over the same band
   x ∈ [0.5, 18], and keep ΔT/T as a free parameter. In code, the trapezoid weight dx at each
   band point becomes x⁶ dx. This applies to `decompose_nonlinear_be` (the default
   `decompose_distortion`, hence the μ and y the solver reports) and `decompose_gram_schmidt`,
   in Rust and Python. Function names, signatures, the band, and the nonlinear μ form stay the
   same.
2. A second estimator, `decompose_number_conserving` (Rust) and `method="nc"` (Python), fixes
   ΔT/T by photon-number conservation, ΔT/T = ∫x²Δn dx / ∫x²G dx over the whole grid, removes
   that G component, and fits μ and y linearly with the same weight and band. Tests whose
   targets come from the Chluba (2013, 2015) visibility functions, and paper Fig. 1, use it.
   In those functions M and Y_SZ carry no photon number, so the temperature term holds all
   of it. On the 118 Table 1 bursts this estimator matches J_bb*·J_μ to 0.031 (worst at
   z_h = 5.4×10⁴; 0.012 for z_h ≥ 3×10⁵) and J_y to 0.006. The free-ΔT fit gives
   J_μ > 1 for 5.7×10⁴ ≲ z_h ≲ 3.6×10⁵, up to 1.25 near 10⁵.

EB decided both points on 2026-09-24: first the x³ weight, then free ΔT/T for the default and
the number-conserving split for the tests and Fig. 1.

## Alternatives considered

- **Keep the unweighted fit and document it.** The code would keep a "Chluba & Jeong" function
  that does not reproduce their basis, and the reported y would reach twice the injected energy
  in the transition.
- **Add a weighting argument, unweighted by default.** Two estimators that disagree by a factor
  of 2 in the transition would stay one keyword apart, and the default would be the one that
  matches neither the literature nor the observable.
- **Use the visibility-function fit (`method="gf_fit"`) as the default.** It needs the
  injection redshift and assumes a single burst, so it cannot decompose an arbitrary spectrum.
- **Weight by the FIRAS or PIXIE noise covariance.** That belongs in the likelihood code
  (`firas.py`), not in a general decomposition.

## Consequences

- Solver-reported μ and y change wherever the spectrum is not a pure μ, y, or temperature
  shape. The μ–y transition moves most: J_μ now peaks at 1.25 near z_h = 10⁵, and ΔT/T absorbs
  the balance, as in Chluba & Jeong (2014) Fig. 1. At z_h = 2×10⁵ the reported μ is about 10%
  above (3/κ_c)·J_bb*·J_μ·Δρ/ρ. Pure μ, y, and G_bb inputs are recovered in either metric.
- Nine tests failed under the free-ΔT x³ fit, eight of them comparing solver μ or y with
  visibility-function or era targets: `solver::tests::test_solver_builder_disable_dcbr`,
  `golden_mu_era_spectral_shape`, `golden_y_era_spectral_shape`,
  `test_y_era_burst_spectral_purity` (`pde_heat.rs`), `test_pde_vs_gf_photon_injection_high_x`,
  `test_decaying_particle_photon_soft_pde` (`pde_photon.rs`), and
  `science_mu_era_coefficient_pde`, `science_y_era_coefficient_pde` (`science_suite.rs`).
  They now use the number-conserving estimator, as does the low-x sibling of the photon test,
  which shares its Chluba (2015) target; targets and tolerances are unchanged. All pass.
  The ninth, `test_solver_respects_cosmology_parameters` (`pde_heat.rs`), required two
  cosmologies to change μ at z_h = 5×10⁴ by more than 10⁻³, a threshold never derived; they
  changed it by 8.4×10⁻⁴. EB had it deleted (2026-09-24).
  The other μ/y assertions pass with the new default and are unchanged.
- New unit tests anchor the metric outside the code: the Chluba & Jeong (2014) basis norms
  from SI constants typed into the test (0.3%), and `decompose_gram_schmidt` against an
  independent least-squares fit to intensities in 1 GHz channels on a spectrum with an
  out-of-span term. Under the old metric that fit misses by 23 times the tolerance in μ.
- The x⁶ weights span 2×10⁹ over the band, but conditioning improves: the condition number of
  the normalized {M, G, Y_SZ} basis on [0.5, 18] falls from 20 to 6.1 (singular-value ratio of the weighted design matrix). The linear and
  nonlinear fits agree to 3×10⁻⁵ in J_μ on the PDE and CosmoTherm tables at Δρ/ρ = 10⁻⁵.
- Paper: the Appendix describes the x³ weight, drops the claim that the nonlinear μ form
  breaks the μ–T degeneracy, and describes the number-conserving variant; Fig. 1 is
  regenerated with it.
