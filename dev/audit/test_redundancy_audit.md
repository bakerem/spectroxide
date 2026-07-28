# Test-suite redundancy audit

**Date:** 2026-07-07. **Scope:** full Rust test suite (integration + src unit
tests), analyzed in the `kompaneets-validation` worktree
(`~/spectroxide-kompaneets`) so the four new verification files count as
subsuming tests. **Line numbers refer to the worktree copies** (main's
`heat_injection.rs` differs by a few lines).

Definition of redundant (strict, per Pitfall #9 history): same physical
quantity vs the same kind of anchor at comparable-or-weaker tolerance, on the
same code path. Kernel vs solver, PDE vs GF, and declared regression pins are
NOT redundant. Prior triage passes already removed ~8 tests; these candidates
survived those passes. No candidate is a sole class-1/2 `COVERAGE_MATRIX.md`
entry for its row.

## A. High-confidence removals (12 tests, ~14–16 slow PDE runs saved)

| # | Test (heat_injection.rs unless noted) | Subsumed by | Saves |
|---|---|---|---|
| A1 | `test_heat_mu_first_principles_ratio` (:7139) — μ=1.401·J·Δρ @ 12%, z_h=2e5 | `golden_mu_era_spectral_shape` (:9868, 10% + shape + energy) and `science_mu_era_coefficient_pde` (science_suite.rs:56, 10%) | 1 PDE run |
| A2 | `test_heat_y_era_pure_y_parameter` (:7193) — y=Δρ/4 @ 5%, z_h=5000 | `test_pure_y_analytical_convergence` (:11548, **1%**) + `golden_y_era_spectral_shape` (:9959, 3%) | 1 PDE run |
| A3 | `test_photon_injection_negative_mu_chluba2015` (:1615) — Eq. 30 w/o visibility @ 20% | `test_pde_vs_gf_photon_injection_low_x` (:5660, with visibility, 15%), `_balanced` (:5735), `_high_x` (:5587), `test_photon_injection_mu_y_systematics` (:6444) | **~8 PDE runs** (re-runs baseline per call) |
| A4 | `test_greens_function_energy_accounting` (:691) | `chluba2013_energy_conservation` (greens_function_checks.rs:189) — same integral, same 3%/22% split, denser z. Fold the z=5e6 point in when removing | seconds |
| A5 | `test_br_absolute_value_z1e6_x1` (:11825) — 4-decade bound | `br_coefficient_first_principles` (rate_coefficients_first_principles.rs:106) — 3% vs literal CODATA + z-independence. Matrix row 6 names this replacement | instant |
| A6 | `test_compton_equilibrium_mu_distortion` (:3062) — sign/OoM only | `compton_equilibrium_coeff_mu` (compton_equilibrium_analytic.rs) — 0.3% mpmath anchor, same function, same amp | instant |
| A7 | `test_dc_high_freq_suppression_decay` (:2913) | `test_dc_suppression_monotonicity` (:3973) — same normalization, x=0.5…100. Extend its range to x>0 on removal (one line) | instant |
| A8 | `test_gaunt_ff_z_dependence` (:2727) — Z-ordering | `brpack_gaunt_factor_spot_checks` (greens_function_checks.rs:336) — 1e-10 pointwise CRB-2020 pin incl. Z=2 + ordering; `gaunt_ff_nr` has no Z/θ branches | instant |
| A9 | `test_solver_config_validation` (:9305) — 2 cases | `test_solver_config_rejects_bad_params` (adversarial_inputs.rs:215) — strict superset, 10 cases | instant |
| A10 | `test_grid_convergence_rate` (:9499) — order ∈ (0.6,3.3), μ 10% | `convergence_order_spatial_full_physics` (convergence_order.rs:432) — μ <1% over 500–4000 pts, two-sided order fence | 3 PDE runs |
| A11 | `science_high_z_thermalization_is_temperature_shift` (science_suite.rs:121) | `chluba2013_limit_pure_temperature_shift` (greens_function_checks.rs:88) — same call, 0.5% vs 1%, more points. Keep instead if science_suite must stay standalone | instant |
| A12 | `test_pde_y_to_mu_conversion` (:2335) — name lies: only asserts Newton/signs/energy 15%; the μ/y-evolution claim is eprintln-only | `test_heat_energy_conservation_sweep_tight` (:7252, 2%) + golden μ tests | 1 PDE run |

## B. Medium-confidence candidates (~12, up to ~14 more PDE runs)

- **B1** `test_photon_injection_analytic_match` (:1544) — chain-subsumed via
  PDE-vs-GF @ x=5 (:6376, 10%) + GF-vs-Eq.30 algebraic identities (:6521).
  Saves 2 PDE runs.
- **B2** `test_timestep_convergence_order` (:10826) — weaker statistic than
  `convergence_order_temporal_full_physics` (convergence_order.rs:526); also
  runs its 4 dy-solves twice (defect, see below). Saves up to 8 PDE runs.
- **B3** `test_grid_transition_artifact` (:11754) — μ/energy claims covered by
  spatial-order + sweep-tight + golden. Saves 2 runs (one at 4000 pts).
- **B4** `test_compton_equilibrium_mu_distortion_deviation` (:3937) — OoM band
  vs `test_full_te_rho_e_for_mu_distortion` (:10715, 1e-6 quadrature) + new
  COEFF anchors.
- **B5** `test_gaunt_ff_limiting_behavior` (:2752) + `test_gaunt_ff_cross_validation`
  (:9059) — both re-derive the same single-branch formula pinned at 1e-10 by
  the spot checks; cross_validation is not genuinely cross-code. If keeping
  one, keep cross_validation (classical-limit point). Matrix row 7 edit needed.
- **B6** `test_dc_temperature_scaling` (:2887) + `test_br_temperature_scaling`
  (:2849) — θ-power now pinned catastrophically harder by II.3; only increment
  is a lower θ decade on a branch-free formula. Matrix rows 5/6 edit needed.
- **B7** `test_literature_mu_y_conversion_coefficients` (:1078) — GF-path
  1.401/Δρ4 @ 10% vs `test_mu_efficiency_deep_mu_era` (:367, 5%) +
  `test_y_efficiency_y_era` (:426, 5%) + `test_literature_regime_boundaries` (:1148).
- **B8** `test_pde_planck_is_stable_equilibrium` (:1202) — null run subsumed by
  `test_pde_no_injection_full_range` (:4169, superset z) +
  `test_adiabatic_cooling_no_injection` (:12044, literature-pinned). Saves 1 run.
- **B9** Recombination pair (:2499 vs :9440) — neither strictly subsumes;
  **merge** (fold z=3000 + tighter z=1100 band into `ionization_history`,
  delete `physical_values`).
- **B10** Linearity pair `test_pde_linearity_double_injection` (:4211) vs
  `test_heat_pde_amplitude_linearity` (:8436) — same structural property, same
  anchor; drop :4211 (its Δρ leg is linear by construction). Saves 2 runs.
- **B11** Negative-injection pair (:3847 vs :11116) — identical scenario;
  merge signs + energy magnitude into one run. Saves 1 run.
- **B12** `test_gf_energy_sum_rule` (:11283) — analytic skeleton of A4's
  integral; keep as the cheap version (keeping it makes A4's removal safer).

## Incidental defects found (fix regardless of removals)

1. **Tautological assertion** in
   `test_pde_negative_injection_produces_negative_distortion` (:~11152):
   `dt_t` is defined as `Δρ/4 − μ/(4·1.401) − y`, then
   `energy_sum = μ/1.401 + 4y + 4·dt_t` reconstructs Δρ identically —
   `energy_err < 0.01` can never fail.
2. **Doubled solve loop** in `test_timestep_convergence_order` (:10886–10920):
   the 4 dy-solves run twice, the second pass only for step counts — 8 PDE
   runs where 4 suffice.

## Investigated and kept (so the pairs aren't re-litigated)

- Kernel unit tests (`test_kompaneets_yields_ysz_shape`, `_y_distortion_magnitude`,
  `_te_gt_tz_positive_drho_all_solvers`) vs new T5: different kernel entry
  points (`kompaneets_step`/`_nonlinear` vs `_coupled_inplace`). Keep.
- `t1_photon_number_conserved` (kompaneets_moments.rs): validates that file's
  own quadrature weights; also mms_convergence.rs is absent from the worktree.
- `test_dc_br_ratio_pinned_z1e6` (:10787): the named /n_e-bug regression pin.
- `test_dcbr_dimensional_scaling_vs_z` (:10641): uniquely covers z=1e4–1e5
  (He-Saha regime) which II.3 deliberately avoids.
- Spectral-shape RMS tests (:7783, :7856): stronger than the decomposition
  residual sweep; not implied by goldens. (:7783 shares its solver config with
  A1 — natural merge target for A1's μ assertion.)
- `test_y_era_burst_spectral_purity` (:3187): only y-amplitude anchor at z_h=1e4.
- Thermalization-suppression pair (:479, :1339): endpoint tightness vs
  monotonicity — complementary.
- Visibility constraint/literature pair (:580, :11978): model-independent vs
  pointwise bands — complementary.
- Transition-region trio (z=1e4/3e4/8e4): different physics per z.
- `test_heat_pde_vs_gf_multi_z_sweep` (:8153): z=1e5 leg unique; optionally
  trim the duplicated z=2e5 and y@5000 legs (saves 2 runs).
- `test_greens_function_asymptotic_limits` (:3446): only assembled G_th→Y/4
  check at z=1e3; its μ-era assertion is near-tautological — tighten, don't drop.
- `test_high_z_dtau_convergence` (:4325): coarse-dtau stress (dtau_max=3),
  not covered by convergence_order.rs (dtau_max=200).
- `test_single_burst_energy_normalization` (:882) and coverage_gaps
  `energy_conservation_*`: scenario-level quadrature / per-scenario, distinct.

## Bookkeeping on removal

Update `dev/audit/COVERAGE_MATRIX.md` rows 5, 6, 7 and
`dev/audit/TEST_PROVENANCE.md` if any of A5/A8/B5/B6 land. CLAUDE.md test
counts (`tests/` section) also need adjusting.
