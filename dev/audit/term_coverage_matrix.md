# Test coverage matrix — by physical term

Maintained per `dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md` §II.7.
**Rows** = physical terms/paths. **Columns** = test class. Update the relevant
row whenever a test is added.

**Not the same document as `coverage_matrix.md`.** That one (Workstream R0) is
indexed by *published result* — one row per paper figure/table — and asks "is
every published number anchored, and how independently." This one is indexed by
*physical term* in the code and asks "is every term in the equations pinned by
something outside the code." A term can be ironclad here while a figure that
uses it is still a gap there, and vice versa. Both are needed; keep them in
sync when a test lands.

**Definition of "ironclad" this project works to: every row has at least one
entry in class (1) or (2).**

## Column key

1. **Exact structural identity** — an identity that holds to machine/quadrature
   precision (conservation law, flux cancellation, analytic moment identity).
2. **Analytic anchor pinning amplitude** — target derived independently of the
   code (literature formula, first-principles constants, mpmath quadrature),
   pinning the *magnitude*, not merely a shape correlation with a fitted
   amplitude or a sign/order-of-magnitude bound.
3. **Design-order convergence** — MMS / Richardson at the scheme's design order.
4. **Cross-code / literature numeric** — CosmoTherm, DarkHistory, published values.
5. **Regression / shape / sign / order-of-magnitude** — asserts against code
   output or a loose physical bound; catches regressions but not systematic
   errors (CLAUDE.md Pitfall #9).

`P-I` / `P-II.x` = added/strengthened by this plan (Part I file
`tests/kompaneets_moments.rs`; Part II per-item files).

## Matrix

| # | Physical term / path | (1) identity | (2) amplitude anchor | (3) design-order | (4) cross-code | (5) regression |
|---|----------------------|--------------|----------------------|------------------|----------------|----------------|
| 1 | Kompaneets drift (up-scatter) | `test_kompaneets_rhs_planck_cancellation` (kompaneets.rs:1195); **P-I** T3 tier-a k=3,4,5 (★) | `test_kompaneets_y_distortion_magnitude` (kompaneets.rs:1097, Δρ/ρ=4y 5%); **P-I** T2 ZS law `4−x₀` | `mms_kernel_*` (mms_convergence.rs:208,232,352); `convergence_order_*_pure_kompaneets` (convergence_order.rs:393,504) | — | `test_kompaneets_yields_ysz_shape` (kompaneets.rs:1547); `test_kompaneets_te_gt_tz_positive_drho_all_solvers` (:1137) |
| 2 | Kompaneets recoil (`+n`/`(2n_pl+1)`) | `test_kompaneets_rhs_planck_cancellation` (kompaneets.rs:1195); **P-I** T3 tier-a `(k−2)` recoil coeff over k=3,4,5 | **P-I** T2/T3 (the `−M_{k+1}` term is pinned by the analytic `(k−2)` coefficient) | MMS op carries recoil terms (mms_convergence.rs:104–112) | — | — |
| 3 | (φ−1) source (heating→y flux branch) | `test_kompaneets_rhs_planck_cancellation` probe b (∝(φ−1), rel 1e-3); **P-I** T5 number conservation on branch | `test_kompaneets_y_distortion_magnitude` (5%); **P-I** T5 pointwise Y_SZ **shape+amplitude** (θ-normalization pinned) | — (MMS sets φ=1) | — | `miri_kernel_coupled_driven_no_dcbr` (kompaneets.rs:1463) |
| 4 | Δn² (quadratic) term | **GAP** (only bundled in MMS op) | **P-I** T6 linearity diagnostic isolates the Δn² contribution (relative-∝A) | MMS op includes `a²g²`,`2a²gg'` (mms_convergence.rs:110) | — | `test_pde_linearity_double_injection` (pde_heat.rs); `test_kompaneets_large_perturbation_stability` (:2671) |
| 5 | DC emission magnitude | `test_dc_polynomial_coefficients_cs2012` (double_compton.rs:324, 1e-10); `test_dc_temperature_scaling` (θ²) | `test_dc_emission_coefficient_magnitude` (double_compton.rs:278, 10%); **P-II.3** first-principles K_DC; `test_dc_emission_coefficient_absolute_value_z1e6` (double_compton.rs, K_DC(x=1, z=1e6) at 5e-4) | — | `test_dc_br_ratio_at_p18_reference_point` (components.rs; DC/BR = 17.06 at z = 1e6, x = 0.1, hand derivation checked against Danese & de Zotti 1982) | `test_dc_high_freq_suppression_extremes` (double_compton.rs) |
| 6 | BR emission magnitude | `test_br_hardcoded_constants` (bremsstrahlung.rs:644); `test_br_temperature_scaling` (θ^−7/2) | `br_coefficient_first_principles` (rate_coefficients_first_principles.rs; literal CODATA prefactor, 3%, z-independent across three redshifts) | — | `test_dc_br_ratio_at_p18_reference_point` (components.rs) | `test_br_hii_capped_at_nh` (bremsstrahlung.rs; Saha-He path) |
| 7 | Gaunt factors | BR: `br_coefficient_invariant_under_theta_z_at_fixed_nu_and_te` (greens_function_checks.rs:486; K_BR φ³e^{x_e}θ_e^{7/2} fixed under θ_z at fixed ν, T_e, rel 1e-12, all five K_BR paths; P-1); `test_br_drho_matches_finite_difference` (bremsstrahlung.rs:731; analytic dK_BR/dρ_e incl. the (3/2)φ Gaunt term vs central FD); DC Gaunt (double_compton.rs:169,220,324) | BR: `gaunt_ff_classical_limit_matches_draine` (greens_function_checks.rs:350) and `br_coefficient_gaunt_uses_electron_frequency_draine_anchor` (:443, through K_BR at ρ_e = 0.5–1.6 with a literal CODATA prefactor): low-x_e limit vs Draine (2011) Eq. 10.9, (√3/π) ln(C θ_e^{1/2}/(Z x_e)), C = 183.107 literal (mpmath), 1e-3 absolute. Pins only the classical limit; the fit's transition and high-x_e shape (softplus, +1 floor) have no independent anchor (BRpack/Itoh tables would be one); `gaunt_ff_nr_matches_draine_ch10_physical_units` (:398, physical units up to x_e ≈ 15, 2e-4) checks the same interpolation, not an independent table; DC Gaunt covered by row 5 | — | — | — |
| 8 | Perturbative Δρ_eq (Compton-eq. T_e) | `test_full_te_perturbative_vs_brute_force` (components.rs); `test_full_te_rho_e_for_mu_distortion` (components.rs; a Bose–Einstein spectrum has ρ_eq = 1 exactly, checked as ρ_eq(BE) − ρ_eq(Planck) < 1e-8 on one grid); `test_alpha_rho` (constants.rs, 0.37020884 ± 1e-6) | `test_equilibrium_recovers_shifted_temperature` (electron_temp.rs:139); **P-II.2** analytic COEFF_Y/COEFF_MU (mpmath) on the full ratio **and** the solver path | — | — | — |
| 9 | T_e full quasi-stationary path | (leading order via row 8) | **GAP** for the full nonlinear path | **GAP** | — | `convergence_quasi_stationary_te_consistency` (convergence_order.rs:781); `test_pde_electron_temperature_feedback` (pde_heat.rs) |
| 10 | Recombination X_e(z) | **GAP** | **GAP** | **GAP** | `test_xe_vs_recfast_milestones` and `test_xe_vs_hyrec_helium_epoch` (recombination.rs; HyRec-2 on the default cosmology, ±6% at z = 1100, 800, 200 and ±1–8% over 1800–5000) | `recombination_x_e_sanity_checks` (convergence_order.rs); `test_helium_saha_transitions`, `test_xe_bounds_between_anchors` (recombination.rs) |
| 11 | Expansion / redshifting | **GAP** | **GAP** | — | `test_adiabatic_cooling_mu_vs_cosmotherm` (cosmotherm_comparison.rs:608); `test_cosmotherm_cooling_sign_convention` (:87) | `test_adiabatic_cooling_no_injection` (solver_numerics.rs); `test_density_scaling_relations` (components.rs) |
| 12 | P_s (photon survival) | **GAP** | **GAP** (**P-II.1** μ-photosphere x_c is the nearest analytic anchor; cross-checks `greens::x_c_*`) | — | — | `test_photon_survival_regime_structure` (gf_visibility.rs); greens.rs `test_x_c_*`, `test_tau_ff_*` |
| 13 | y_γ broadening | **GAP** (weakest row) | **GAP** | **GAP** | — | indirect only via full photon GF: `test_photon_gf_soft_photon_absorbed` (gf_visibility.rs) |
| 14 | γ_con (dark-photon conversion) | `gamma_con_scales_as_epsilon_squared` (dark_photon.rs:146); `resonance_round_trip` (:136) | `plasma_frequency_matches_first_principles` (dark_photon.rs:99, 1e-12) | — | `gamma_con_matches_chluba_cyr` (dark_photon.rs:154, ±20%); **P-II.4** Landau-Zener integration (diagnosis) | — |
| 15 | μ/y/G_bb decomposition | `test_gram_schmidt_pure_*` (distortion.rs); `test_firas_check_values` (:583) | `test_decompose_pure_mu/y/delta_t` (distortion.rs:453,483,534, 1%); `test_literature_mu_y_conversion_coefficients` (gf_visibility.rs) | — | — | `test_decomposition_comprehensive` (gf_visibility.rs) |
| 16 | FIRAS χ² | `test_firas_check_values` (distortion.rs:583, fraction-of-limit) | **GAP** (no χ²-vs-data amplitude; LM χ² in `fit_distortion` tested only indirectly) → **P-II.5** closed-loop MC coverage calibration | — | — | `test_firas_limits_consistency` (gf_visibility.rs) |
| 17 | Injection scenarios' energy bookkeeping | `photon_number_ledger_identity_with_dcbr_and_source` (mms_convergence.rs:442, 1e-9); `photon_number_conserved_coupled_path_pure_compton` (:397) | `test_single_burst_energy_normalization` (components.rs); `test_photon_injection_energy_conservation_tight` (:6000) | — | — | `conservation_fuzz.rs` (`fuzz_energy_closure_*`); coverage_gaps.rs (`energy_conservation_*`) |

## Remaining class-1/2 gaps after this plan

- **Row 4 (Δn²)**: no isolated class-1 identity. Structurally pinned by MMS (3)
  and diagnosed by P-I T6, but no exact standalone identity. *Acceptable* — the
  term is quadratic and small; T6 bounds its contribution.
- **Row 9 (T_e full path)**: leading order pinned via row 8; the full nonlinear
  quasi-stationary path has no design-order or amplitude anchor. *Open.*
- **Row 10 (Recombination X_e)**: cross-code (4) + sanity (5) only; no exact
  identity, analytic anchor, or MMS. *Open* — Peebles/Saha ODE is not
  MMS-verified.
- **Row 11 (Expansion/redshifting)**: cross-code (4) + scaling (5) only. *Open.*
- **Row 12 (P_s)**: P-II.1 provides the nearest analytic anchor (x_c(z)); a
  direct P_s(x,z) identity remains open.
- **Row 13 (y_γ broadening)**: weakest row — only indirect full-GF shape checks.
  No item in this plan targets the broadening kernel directly. *Open — flagged
  as the highest-priority future gap.* Priority is set by sensitivity, not by
  aesthetics: the R2 sensitivity map measures **∂lnμ/∂ln y_γ = −2.03**, so a
  10% error in the broadening kernel is a 20% error in the photon-injection μ
  that Figs. 6–8 rest on. This is the largest unanchored O(1) lever in the code.
- **Row 16 (FIRAS χ²)**: P-II.5 calibrates coverage; a per-spectrum χ² identity
  vs FIRAS data remains only indirectly tested. **Narrower than it looks:**
  P-II.5 drives the *single-amplitude* fit (`FIRASData.fit_amplitude`), whereas
  the surviving `firas.py` mutants and the paper's published limits both go
  through the **floating-`T` profile likelihood** — the very path that accounts
  for the whole Fig. 8 offset against CCJ24. Coverage is calibrated on the path
  that is not the one in question. *Open.*

## Why the class (3) column is not enough on its own

MMS (column 3) builds its manufactured residual `S = ∂_τΔn_m − L[Δn_m]` from an
operator `L` transcribed from the code's own flux form. A wrong coefficient in
that form — recoil `2Δn` instead of `Δn`, flux weighted `x³` instead of `x⁴` —
appears in *both* the code and the residual, cancels, and MMS still converges at
p = 2.00. So a row whose only strong entry is class (3) is pinned to the
scheme, not to the physics. This is the structural blind spot the moment
hierarchy (rows 1–3) exists to close, and it is the reason the "ironclad"
definition above demands class (1) or (2) rather than accepting (3).

It is the same failure mode mutation testing found from the other end (see
`mutation_audit.md`, F-R2-3): there, a coefficient was written twice and the
literature anchor tested the unused copy. Test *construction* and test
*placement* fail independently; a high coverage number detects neither.

## Change log

- 2026-07-07: matrix created (§II.7). Existing coverage mapped; Part I/II items
  marked planned.
- 2026-07-30: merged to `main` (renamed from `COVERAGE_MATRIX.md` to avoid a
  case-insensitive collision with R0's `coverage_matrix.md`). All P-I/P-II rows
  re-verified against current `main` in release mode — see
  `KOMPANEETS_VERIFICATION_RESULTS.md`. Added the class-(3) caveat above and the
  cross-reference to R0's matrix. Row 13 (y_γ) restated as the top open gap with
  its measured sensitivity ∂lnμ/∂ln y_γ = −2.03.
