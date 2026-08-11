# Test-Provenance Census (audit B0)

Generated 2026-07-03 by `dev/scripts/build_test_provenance.py` from the assertion inventory (`dev/scripts/extract_test_assertions.py`, 1343 numeric assertions in 619 tests).

Every test with a numeric assertion is classified by the origin of its target value:

- **analytic** — closed-form result verifiable independently of the code
- **literature** — number from a cited paper or public dataset
- **cross-method** — agreement between two independent implementations
- **dimensional** — sign/scaling/order-of-magnitude from a physical argument
- **regression-pin** — the code's own historical output, explicitly labelled
- **structural** — numeric literal is incidental (sizes, finiteness); no physics target

## Summary

| Class | Tests |
|---|---|
| analytic | 288 |
| literature | 61 |
| cross-method | 49 |
| dimensional | 107 |
| regression-pin | 4 |
| structural | 115 |
| **total** | **624** |

Flagged audit gaps (code-derived target without a pin label): **1**

## Flagged gaps

- `tests/heat_injection.rs::test_photon_injection_spectral_decomposition_residual` — 3-component fit RMS residual < 12% of peak (docstring states residual 'should be < 5%' but assertion allows 12% - the 12% has no stated independent origin (looks widened to code output))

## Declared regression pins

- `tests/heat_injection.rs::test_photon_survival_post_recombination` — x_c(500)<1 and P_s(3,500)>0.5 — explicitly documented as fitting-formula artifact accommodation; harmless because J_Compton ~ 0
- `tests/heat_injection.rs::test_dc_br_ratio_pinned_z1e6` — DC/BR at z=1e6, x=1 in (8, 50) — test name and comment label it as pinned; failure messages diagnose known bug modes
- `src/dark_photon.rs::gamma_con_matches_chluba_cyr` — z_res ~ 3.21e4 (5%) and gamma_con/eps^2 ~ 9.3e10 (20%) — test name claims literature but the amplitude target is measured code output guarding drift; note the ~22% discrepancy vs an external comparison (memory: axion_DP_distortion) — as of 2026-07-07/30 that discrepancy is NOT in gamma_con: direct Landau-Zener ODE integration reproduces the NWA to 1.2% at the adiabaticity boundary (dev/audit/gamma_con_lz_check.md), so it lives in the downstream frozen-vs-thermalized treatment. The 20% band on this test's amplitude target is still measured code output, not a literature number, and is unaffected by that result
- `python/tests/test_fh_basis.py::test_wp3_stationary_y_audit` — mu-overlay residual norm < 5% (computation basis); CSV finite — y-residual is reported but deliberately not asserted; test skips if CSV absent

## Full table

| File | Test | Class | Target | Source |
|---|---|---|---|---|
| python/tests/test_anisotropy.py | `test_boost_identity_g_shape` | analytic | boost identity residual -xG' = Y_SZ + 3G < 1e-3 | exact identity (superposition of blackbodies), docstring: 'pure math' |
| python/tests/test_anisotropy.py | `test_boost_linearity` | analytic | amplitudes scale x10 with input to 1e-6 | linearity of boost operator (exact) |
| python/tests/test_anisotropy.py | `test_boost_of_temperature_shift_decomposes_to_y` | analytic | boosted dT*G gives y=dT, dT'=3dT, mu=0 to 2% | same exact boost identity |
| python/tests/test_anisotropy.py | `test_e_g_contributes_only_to_dt` | analytic | include_e_g adds exactly +1 to b_dT, mu unchanged (=0) | definition of e_G term in boost operator |
| python/tests/test_anisotropy.py | `test_load_class_cls_fixture` | literature | CLASS TT acoustic peak at 180<=l<=260; >=2000 multipoles | CLASS fixture (data/class_fixtures); Planck TT first peak l~220 |
| python/tests/test_cosmotherm.py | `test_mixed_signal` | analytic | recovers alpha_in=5e-4 to 5% and mu component to 10% of peak | synthesis-recovery of known injected amplitudes |
| python/tests/test_cosmotherm.py | `test_mu_shape_mostly_preserved` | analytic | stripping M(x) removes alpha<10% of amplitude, spectrum within 10% of peak | M(x) approximately photon-number-neutral (analytic property) |
| python/tests/test_cosmotherm.py | `test_number_conservation` | analytic | int x^2 dn_nc dx < 1e-8 after stripping | number conservation ~0 by construction of strip_gbb |
| python/tests/test_cosmotherm.py | `test_peaks_around_x4` | analytic | G_bb(x) in Jy/sr peaks at 3<x<5 | closed form x^4 e^-x/(1-e^-x)^2 peak near x~3.9 |
| python/tests/test_cosmotherm.py | `test_pure_gbb_fully_stripped` | analytic | recovers injected alpha=1e-4 to 0.1%; residual photon number ~0 | synthesis-recovery: stripping a pure G_bb must return injected amplitude |
| python/tests/test_cosmotherm.py | `test_pwave_vs_swave_ratio` | analytic | rp/rs = 1+z exactly (1e-10) | <sigma v> ~ v^2 ~ T ~ (1+z), exact ratio |
| python/tests/test_cosmotherm.py | `test_round_trip` | analytic | Jy/sr -> dn -> Jy/sr round trip to 1e-10 | exact unit-conversion inverse |
| python/tests/test_cosmotherm.py | `test_round_trip_identity` | analytic | dn -> DI[Jy/sr] -> dn round trip to 1e-10 | exact unit-conversion inverse (2hnu^3/c^2 factor) |
| python/tests/test_cosmotherm.py | `test_swave_scales_with_fann` | analytic | heating rate doubles with f_ann x2 to 1e-10 | linearity in f_ann (exact) |
| python/tests/test_cosmotherm.py | `test_y_shape_mostly_preserved` | analytic | stripping Y_SZ removes little (same 10% bounds) | Y_SZ approximately photon-number-neutral |
| python/tests/test_cosmotherm.py | `test_zero_di_gives_zero_dn` | structural | zero DI gives zero dn | trivial zero map |
| python/tests/test_dm_baryon.py | `test_ah21_fig1_benchmark` | literature | z_dec in (3e6,8e6); mu in (-2e-6,-5e-7) | AH21 Fig. 1 benchmark (z_dec~5e6, \|mu\|~1e-6) |
| python/tests/test_dm_baryon.py | `test_f_chi_linearity` | analytic | heating rate ratio 0.5 for f_chi halved, to 1% | linearity in f_chi (exact) |
| python/tests/test_dm_baryon.py | `test_gamma_chi_p_spot_value` | analytic | Gamma_chi_p = 1.2472e-12 s^-1 at z=1e5 to 2e-3 | hand calculation from AH21 momentum-exchange rate formula |
| python/tests/test_dm_baryon.py | `test_mu_max_tight_coupling` | literature | \|mu_max\| = 8.6e-5 for m_chi=0.1 MeV to 20%; q>0.99 | AH21 Eq. 2.20 (\|mu_max\|=8.6e-6 f_chi MeV/m_chi); tol covers Chluba-2013 vs AH21 visibility difference |
| python/tests/test_dm_baryon.py | `test_n_indices_and_electron_scatterer` | dimensional | q finite and >=0 for n in {-2,2,4,6}; electron-scatterer rate finite and <=0 | sign/finiteness from physics (DM-baryon scattering cools photons) |
| python/tests/test_dm_baryon.py | `test_prefactor_matches_ah21` | literature | -dQ/dz/H = 1.7095e-6 f_chi(MeV/m_chi) to 1%; q>0.999 | Ali-Haimoud 2021 (arXiv:2101.04070) Eq. 2.11, hand-derived constant |
| python/tests/test_dm_baryon.py | `test_q_bounded_and_dm_colder` | dimensional | 0<=q<=1 and T_chi<=T_gamma for 3 parameter sets | physical bounds: DM never hotter than photons, deposited fraction bounded |
| python/tests/test_dm_baryon.py | `test_rust_python_parity` | analytic | prefactor/q z-independent to 1e-12 across 3 redshifts | exact z-independence of (3/2)n_chi k T_gamma/rho_gamma |
| python/tests/test_fh_basis.py | `test_amplitude_spectrum_linearity` | analytic | unit vectors synthesize exactly G(x), M(x), Y_0(x) | basis definition (exact) |
| python/tests/test_fh_basis.py | `test_b0_mu_era_computation_50_50` | literature | energy split f_y, f_mu each in [0.40,0.60] | EC26 Fig. 10 anchor: O_x M ~ 1.9M+0.4Y splits energy nearly evenly |
| python/tests/test_fh_basis.py | `test_dim_constants` | structural | DIM=18, index positions, N_MOMENTS=15 | basis layout constants |
| python/tests/test_fh_basis.py | `test_g_bb_at_known_point` | analytic | G(2) = 2e^2/(e^2-1)^2 to 1e-12 | closed form |
| python/tests/test_fh_basis.py | `test_load_basis` | analytic | matrix shapes 18x18; eps_Theta=eps_Y0=4; eps_mu=1/1.4007; beta_m=3G2/(2G1) | exact energy weights (4 for T/y) and kappa_c=1.4007 mu constant; G-integral formula |
| python/tests/test_fh_basis.py | `test_ox_m_projection` | literature | O_x M = 0.3736 Y0 + 1.9069 M to 2% via constrained Gram projection | SS-II / WP-0 acceptance values (quoted in paper); mirrors constrainedRepOn in fh_matrices.wl |
| python/tests/test_fh_basis.py | `test_spectral_channels` | structural | channel grid shape (970,), positive, increasing | basis file channel count |
| python/tests/test_fh_basis.py | `test_table1_N0` | literature | y=-0.02070, mu=0.8385 to 0.5% | EC26 Table 1, N=0 row |
| python/tests/test_fh_basis.py | `test_table1_N15` | literature | y=-0.02060, mu=0.8374 to 0.5% | EC26 Table 1, N=15 row |
| python/tests/test_fh_basis.py | `test_wp3_stationary_y_audit` | regression-pin | mu-overlay residual norm < 5% (computation basis); CSV finite | WP-3 status-log finding ('stationary overlay tracks full solution to 0.1% in mu-slot') = code's own prior output; explicitly labelled as a re-audit |
| python/tests/test_fh_basis.py | `test_yk_recursion` | analytic | -x dY_{k-1}/dx = 4 Y_k to 1e-4 (finite diff) | exact boost recursion O_x Y_{k-1} = 4 Y_k |
| python/tests/test_fh_basis.py | `test_yk_zero_matches_y0` | analytic | Y_0 equals SZ shape G(x)[x coth(x/2)-4] to 1e-10 | closed-form SZ shape |
| python/tests/test_firas.py | `test_chi2_null_good_fit` | analytic | 0 < chi2_null < 200 for 43 channels | chi2 ~ ndof +/- sqrt(2 ndof) statistics; FIRAS consistent with blackbody |
| python/tests/test_firas.py | `test_chi2_perfect_fit_zero` | analytic | chi2(model=residuals) = 0 | chi2 definition identity |
| python/tests/test_firas.py | `test_correlation_diagonal_near_ones` | analytic | diag(corr) = 1 to 0.01 | definition of correlation matrix |
| python/tests/test_firas.py | `test_cov_inv_is_inverse` | analytic | C @ C^-1 = I to 1e-8 | matrix-inverse identity |
| python/tests/test_firas.py | `test_covariance_shape` | structural | cov and cov_inv are 43x43 | dataset dimension |
| python/tests/test_firas.py | `test_covariance_symmetric` | structural | cov == cov.T | matrix property of loaded data |
| python/tests/test_firas.py | `test_custom_t_cmb` | structural | t_cmb attribute round-trips; x grid shifts | API passthrough |
| python/tests/test_firas.py | `test_default_t_dust_9k` | literature | default t_dust == 9.0 K | Fixsen 1996 Sect. 6.1 (cited in docstring) |
| python/tests/test_firas.py | `test_dn_to_dI_kJy_zero` | structural | zero dn gives zero dI | trivial zero map |
| python/tests/test_firas.py | `test_fisher_matrix_symmetric` | analytic | F == F.T | Fisher matrix symmetric by definition |
| python/tests/test_firas.py | `test_fisher_single_template` | analytic | F[0,0] = t^T C^-1 t to 1e-12 | Fisher definition identity |
| python/tests/test_firas.py | `test_fit_amplitude_best_fit_reduces_chi2` | analytic | chi2_min <= chi2_null | least-squares property |
| python/tests/test_firas.py | `test_fit_distortion_null` | structural | ndof == 40 (43-3) and finite params | channel count minus 3 fitted params |
| python/tests/test_firas.py | `test_freq_cm_to_ghz_known_value` | analytic | 1 cm^-1 = 29.9792 GHz to 0.01 | c = 2.998e10 cm/s (exact conversion) |
| python/tests/test_firas.py | `test_freq_cm_to_x_identity` | analytic | x=1 at nu=kT/h to 1e-6 | definition x = h nu/(kT) |
| python/tests/test_firas.py | `test_freq_range_ghz` | literature | freq range brackets ~68-640 GHz | Fixsen et al. 1996 (cited in docstring) |
| python/tests/test_firas.py | `test_helper_matches_method` | structural | module helper matches cached method to 1e-12 | same code path consistency (API), not independent implementations |
| python/tests/test_firas.py | `test_limit_on_model_relaxes_or_equal` | analytic | limit_on_model with galactic marg >= without | same inequality |
| python/tests/test_firas.py | `test_marginalised_chi2_min_leq_null` | analytic | chi2_min <= chi2_null | least-squares property |
| python/tests/test_firas.py | `test_marginalised_sigma_larger` | analytic | marginalised sigma >= raw sigma | profiling over nuisance cannot shrink errors |
| python/tests/test_firas.py | `test_mu_68_value` | literature | MU_FIRAS_68 == 4.5e-5 | Fixsen 1996 95% limit / 2 (Gaussian 1-sigma); no explicit citation in test |
| python/tests/test_firas.py | `test_mu_95_value` | literature | MU_FIRAS_95 == 9e-5 | Fixsen et al. 1996 (\|mu\| < 9e-5, 95% CL) |
| python/tests/test_firas.py | `test_mu_limit_relaxes_or_equal` | analytic | limit with galactic marg >= without | adding nuisance freedom cannot tighten limit |
| python/tests/test_firas.py | `test_mu_limit_still_order_of_magnitude` | dimensional | 1e-6 < mu limit < 1e-3 with galactic marg | order-of-magnitude vs FIRAS literature |
| python/tests/test_firas.py | `test_mu_template_nonzero` | structural | template shape (43,), nonzero | API sanity |
| python/tests/test_firas.py | `test_n_freq` | literature | 43 frequency channels | Fixsen et al. 1996 FIRAS monopole dataset |
| python/tests/test_firas.py | `test_predict_linearity` | analytic | prediction linear in mu to 1e-12 | linear model by construction |
| python/tests/test_firas.py | `test_predict_null` | structural | zero params predict zero | trivial |
| python/tests/test_firas.py | `test_predict_with_extra_dn_array` | structural | zero extra dn gives zero prediction | trivial |
| python/tests/test_firas.py | `test_predict_with_extra_dn_callable` | structural | zero extra dn gives zero prediction | trivial |
| python/tests/test_firas.py | `test_profile_limit_floating_T_relaxes` | analytic | floating-T limit with galactic marg >= without | nuisance-marginalisation inequality |
| python/tests/test_firas.py | `test_profile_limit_floating_T_with_galactic` | literature | finite limit; t_best in [2.720, 2.732] K | FIRAS T_CMB = 2.725 +/- a few mK (Fixsen 2009) |
| python/tests/test_firas.py | `test_repr` | structural | repr contains '43' and 'GHz' | API string check |
| python/tests/test_firas.py | `test_template_shape` | structural | galactic template shape (43,) | dataset dimension |
| python/tests/test_firas.py | `test_upper_limit_mu_no_y_marginalisation` | analytic | limit without y-marg <= with | fewer nuisance parameters -> tighter limit |
| python/tests/test_firas.py | `test_upper_limit_mu_order_of_magnitude` | dimensional | 1e-6 < mu_95 < 1e-3 | order-of-magnitude vs literature O(1e-5) FIRAS limit |
| python/tests/test_firas.py | `test_upper_limit_y_order_of_magnitude` | dimensional | 1e-7 < y_95 < 1e-3 | order-of-magnitude vs literature O(1e-5) |
| python/tests/test_firas.py | `test_x_range` | dimensional | x spans roughly 1-11 (brackets 0.5-2 and 8-15) | follows from FIRAS 68-640 GHz and T=2.726 K |
| python/tests/test_firas.py | `test_y_68_value` | literature | Y_FIRAS_68 == 7.5e-6 | Fixsen 1996 95% limit / 2; no explicit citation in test |
| python/tests/test_firas.py | `test_y_95_value` | literature | Y_FIRAS_95 == 1.5e-5 | Fixsen et al. 1996 (\|y\| < 1.5e-5, 95% CL) |
| python/tests/test_firas.py | `test_y_limit_relaxes_or_equal` | analytic | limit with galactic marg >= without | same inequality |
| python/tests/test_firas.py | `test_y_limit_still_order_of_magnitude` | dimensional | 1e-7 < y limit < 1e-3 with galactic marg | order-of-magnitude vs FIRAS literature |
| python/tests/test_firas.py | `test_y_template_nonzero` | structural | template shape (43,), nonzero | API sanity |
| python/tests/test_greens.py | `test_baryon_photon_ratio_scaling` | analytic | R(z1)/R(z2) = (1+z2)/(1+z1) to 1e-6 | R ~ 1/(1+z) exact scaling |
| python/tests/test_greens.py | `test_default` | literature | h=0.71, omega_b=0.044, Y_p=0.24 | Chluba 2013 default cosmology (also project CLAUDE.md defaults) |
| python/tests/test_greens.py | `test_drho_matches_direct_integral` | analytic | returned drho = int x^3 dn dx / G3 to 1e-3 | drho definition, independent trapezoid quadrature in-test |
| python/tests/test_greens.py | `test_fit_residual_small` | analytic | BF fit recovers injected mu and y to 5% | synthesis-recovery; BE and Chluba-M mu coincide in linear regime |
| python/tests/test_greens.py | `test_g_bb_integral_gives_g3` | analytic | int x^3 G_bb dx = 4 G3 = 4pi^4/15 to 0.5% | exact energy integral |
| python/tests/test_greens.py | `test_gf_deep_mu_era` | analytic | decomposed mu = 1.401 J_bb* J_mu drho to 5% | mu = 1.401 drho/rho (deep mu-era), Chluba 2013 visibility scaling |
| python/tests/test_greens.py | `test_gf_energy_conservation` | analytic | int x^3 G_th dx = G3 to 20% | energy conservation; 20% slack documented from Chluba J_T=(1-J_bb*)/4 approximation |
| python/tests/test_greens.py | `test_gf_linearity` | structural | 2*g1 == g*2 to 1e-15 | tautological: both sides are the same call doubled |
| python/tests/test_greens.py | `test_gf_y_era` | analytic | y = drho/4 to 5%; mu < 0.1 y | y = drho/(4 rho) exact y-era result |
| python/tests/test_greens.py | `test_hubble_today` | analytic | H(0) = 100h km/s/Mpc to 1% | H0 definition with h=0.71 |
| python/tests/test_greens.py | `test_ionization_fraction_post_recombination` | dimensional | 1e-5 < X_e(500) < 0.01 | standard recombination freeze-out X_e ~ O(1e-4) |
| python/tests/test_greens.py | `test_ionization_fraction_pre_recombination` | dimensional | X_e(1e4) > 1 | full ionization + helium implies X_e>1 |
| python/tests/test_greens.py | `test_j_bb_high_z_thermalization` | analytic | J_bb(1e7) < 1e-10 | thermalization limit J_bb->0 at z>>z_mu |
| python/tests/test_greens.py | `test_j_bb_low_z_no_thermalization` | analytic | J_bb(1e3) = 1 to 1e-6 | J_bb->1 limit at z<<z_mu |
| python/tests/test_greens.py | `test_j_bb_star_non_negative` | dimensional | J_bb* in [0,1] at 5 redshifts | visibility is a fraction; clamping requirement |
| python/tests/test_greens.py | `test_j_mu_high_z` | analytic | J_mu(1e7) = 1 to 1% | pure mu-era limit |
| python/tests/test_greens.py | `test_j_mu_low_z` | analytic | J_mu(1e3) < 0.01 | no-mu limit |
| python/tests/test_greens.py | `test_j_y_complements_j_mu` | dimensional | \|J_y - (1-J_mu)\| < 0.3 | approximate energy conservation between independently-fitted Chluba 2013 visibilities |
| python/tests/test_greens.py | `test_mu_and_y_energy_neutral_basis` | dimensional | \|cos angle\| of energy-neutral f_mu, f_y < 0.95 | Chluba & Jeong 2014 energy-neutral basis reduces M-Y correlation (raw cos~0.88) |
| python/tests/test_greens.py | `test_mu_from_single_burst` | analytic | mu = 1.401 J_bb* J_mu drho to 5% for Gaussian burst | mu=1.401 drho/rho with Chluba 2013 visibilities |
| python/tests/test_greens.py | `test_mu_shape_zero_crossing` | analytic | M(x) crosses zero at beta_mu ~ 2.19; sign pattern | beta_mu from photon-number/energy balance (Chluba) |
| python/tests/test_greens.py | `test_mu_sign_flip_at_x_balanced` | dimensional | mu>0 at x_inj=10, mu<0 at x_inj=2 | sign flip across balance point x0 ~ 3.60 (Chluba 2015 photon injection) |
| python/tests/test_greens.py | `test_n_electron` | analytic | n_e = X_e n_H to 1% | definition |
| python/tests/test_greens.py | `test_n_electron_custom_x_e` | analytic | n_e = 0.5 n_H with x_e=0.5 to 1e-6 | definition |
| python/tests/test_greens.py | `test_n_hydrogen` | analytic | n_H ~ (1+z)^3 to 1e-6 | exact scaling |
| python/tests/test_greens.py | `test_omega_gamma` | dimensional | 1e-5 < Omega_gamma < 1e-4 | known Omega_gamma ~ 5e-5 order of magnitude |
| python/tests/test_greens.py | `test_photon_gf_high_x_dominated_by_survival` | dimensional | \|G(x_inj=10)\| > 10x \|G(x_inj=0.01)\| | P_s -> 1 vs -> 0 limits (order-of-magnitude) |
| python/tests/test_greens.py | `test_photon_survival_limits` | dimensional | P_s(10)>0.99, P_s(1e-5)<1e-10 | survival probability limiting behavior |
| python/tests/test_greens.py | `test_planck2015` | literature | h = 0.6727 | Planck 2015 parameters |
| python/tests/test_greens.py | `test_planck2018` | literature | h = 0.6736, T_cmb = 2.7255 | Planck 2018 parameters |
| python/tests/test_greens.py | `test_planck_identity_derivative` | analytic | dn/dx + n(1+n) = 0 residual < 1% of scale | Planck identity |
| python/tests/test_greens.py | `test_planck_large_x_wien` | analytic | n_pl -> e^-x at x>>1 to 1e-3 | Wien limit |
| python/tests/test_greens.py | `test_planck_low_x_rayleigh_jeans` | analytic | n_pl -> 1/x at x<<1 to 1% | Rayleigh-Jeans limit |
| python/tests/test_greens.py | `test_pure_mu_distortion` | analytic | decompose(mu M(x)) recovers mu to 2%, y<1% mu | synthesis-recovery |
| python/tests/test_greens.py | `test_pure_y_distortion` | analytic | decompose(y Y(x)) recovers y to 2%, mu<5% y | synthesis-recovery; M-Y correlation leakage noted |
| python/tests/test_greens.py | `test_rho_gamma_scaling` | analytic | rho_gamma ~ (1+z)^4 to 1e-6 | exact scaling |
| python/tests/test_greens.py | `test_to_dict` | structural | to_dict round-trips h and omega_b | API round-trip |
| python/tests/test_greens.py | `test_x_c_br_dominates_at_low_z` | dimensional | x_c_br > x_c_dc at z=1e4 | BR dominates at low z |
| python/tests/test_greens.py | `test_x_c_dc_dominates_at_high_z` | dimensional | x_c_dc > x_c_br at z=2e6 | DC dominates absorption at high z (physical ordering) |
| python/tests/test_greens.py | `test_y_from_single_burst_y_era` | analytic | y = drho/4 to 5% | exact y-era result |
| python/tests/test_greens.py | `test_y_shape_zero_crossing` | analytic | Y_SZ zero at x coth(x/2)=4 (x~3.831), found by bisection | transcendental equation solved independently in-test |
| python/tests/test_greens_table.py | `test_at_grid_point` | cross-method | table interpolation at grid points reproduces stored values (rtol 1e-6 / 1e-10) | table vs direct stored evaluation (spline passes through training points); two classes share this name |
| python/tests/test_greens_table.py | `test_distortion_from_heating_single_burst` | cross-method | convolved narrow burst matches point GF evaluation to 15% | table convolution vs direct evaluation |
| python/tests/test_greens_table.py | `test_distortion_from_heating_zero_rate` | structural | zero heating gives zero distortion | trivial |
| python/tests/test_greens_table.py | `test_energy_conservation` | analytic | int x^3 G_th dx / G3 = 1 to 20% | energy conservation; slack documented from Chluba J_T approximation |
| python/tests/test_greens_table.py | `test_greens_function_agrees_with_analytic_in_limits` | cross-method | NC-stripped table GF matches analytic GF to 5% in deep mu and y eras | table interpolation vs analytic greens_function |
| python/tests/test_greens_table.py | `test_metadata_preserved` | structural | metadata dict values survive construction | API |
| python/tests/test_greens_table.py | `test_mu_decreases_toward_low_z` | dimensional | mu(highest z) >= 0.5 mu(lowest z) in mu-era | monotonicity of thermalization efficiency |
| python/tests/test_greens_table.py | `test_mu_from_heating_deep_mu_era` | analytic | mu = 1.401 J_bb* J_mu drho to 20% | mu = 1.401 drho/rho with Chluba 2013 visibilities |
| python/tests/test_greens_table.py | `test_round_trip` | structural | save/load .npz round-trips all arrays and metadata exactly | serialization round-trip (two classes share this name) |
| python/tests/test_greens_table.py | `test_shape` | structural | g_th (50,7) / g_ph (80,3,5) and companion array shapes | construction dimensions (two classes share this name) |
| python/tests/test_greens_table.py | `test_single_burst` | structural | solve(method='table') returns finite mu, y, matching z_h and array lengths | API integration, finiteness only |
| python/tests/test_greens_table.py | `test_single_z_table` | structural | single-z table returns raw stored values to 1e-10 | degenerate-grid round-trip |
| python/tests/test_greens_table.py | `test_y_increases_toward_low_z` | analytic | y -> 0.25 at z<3e3 to 0.02 | y = drho/4 pure y-era limit |
| python/tests/test_greens_table.py | `test_zero_rate` | structural | zero photon injection rate gives zero | trivial |
| python/tests/test_parity.py | `test_parity` | cross-method | Python mirror matches Rust-generated fixture values per-group rtol (declared in fixture) | tests/data/parity_fixtures.json generated by cargo example generate_parity_fixtures; Rust vs Python implementations |
| python/tests/test_solver.py | `test_apply_settings_debug` | structural | merged n_points == 1000 | preset merge logic |
| python/tests/test_solver.py | `test_apply_settings_default_production` | structural | merged n_points == 4000 | preset merge logic |
| python/tests/test_solver.py | `test_apply_settings_none_values_ignored` | structural | None does not override; n_points == 4000 | preset merge logic |
| python/tests/test_solver.py | `test_apply_settings_override` | structural | override n_points == 2000 | preset merge logic |
| python/tests/test_solver.py | `test_custom_x_grid` | structural | len(result['x']) == 100 for run_single custom grid | API passthrough |
| python/tests/test_solver.py | `test_debug_n_points` | structural | DEBUG n_points == 1000 | preset constant |
| python/tests/test_solver.py | `test_delta_I_property` | structural | delta_I returns tuples of length 50, finite, positive for positive dn | API property |
| python/tests/test_solver.py | `test_h_passthrough` | structural | --h arg equals '0.6736' | CLI arg round-trip |
| python/tests/test_solver.py | `test_omega_b_passthrough` | structural | --omega-b arg equals '0.044' | CLI arg round-trip |
| python/tests/test_solver.py | `test_omega_m_passthrough` | structural | --omega-m arg equals '0.26'; no --omega-cdm | CLI arg round-trip |
| python/tests/test_solver.py | `test_production_n_points` | structural | PRODUCTION n_points == 4000 | preset constant |
| python/tests/test_solver.py | `test_resolve_quality_settings` | structural | resolved (4000, True, ...) | preset constants |
| python/tests/test_solver.py | `test_resolve_quality_settings_debug` | structural | resolved (1000, False, ...) | preset constants |
| python/tests/test_solver.py | `test_single_burst_mu_era` | analytic | mu/drho in (0.5, 1.5) at z=5e5 | mu = 1.401 drho/rho deep mu-era (broad window) |
| python/tests/test_solver.py | `test_single_burst_y_era` | literature | y/drho in (0.20, 0.23) at z=3e4 | Chluba 2013 GF: y = J_y(z_h)/4 with J_y(3e4) ~ 0.86 (docstring) |
| python/tests/test_solver.py | `test_solve_custom_x_grid` | structural | len(result.x) == 200 and equals input grid | API passthrough |
| python/tests/test_solver.py | `test_z_h_set` | structural | z_h attribute == 5e4 after construction | dataclass round-trip |
| src/bessel.rs | `fixture_matches_scipy` | cross-method | j_l, j_l', j_l'' match SciPy fixture to <1e-10 rel; >200 rows | SciPy-generated fixture data/bessel_fixtures/jl_fixture.csv (plan WP-4 Stage 1 gate) |
| src/bessel.rs | `j0_j1_closed_forms` | analytic | j0=sin(x)/x, j1 closed forms + small-x limits | exact closed-form spherical Bessel expressions |
| src/bessel.rs | `ladder_matches_single` | cross-method | j_ladder vs single-l jl agree to 1e-12 rel | two internal algorithms (ladder recurrence vs per-l Miller) |
| src/bessel.rs | `upward_recurrence_holds` | analytic | j_{l+1}=(2l+1)/x j_l - j_{l-1} to 1e-12 | exact recurrence identity |
| src/bremsstrahlung.rs | `test_br_coefficient_positive` | dimensional | K_BR >= 0 over x grid | physical positivity |
| src/bremsstrahlung.rs | `test_br_emission_coefficient_magnitude` | dimensional | K_BR(x=0.1, z=1e5) in [1e-12, 1e-4], within 2 OOM of hand estimate 6.6e-9, decreasing with x | hand calculation from BR formula components documented in doc comment; guards against historical /n_e bug (~1e11) |
| src/bremsstrahlung.rs | `test_br_fast_variants_match_reference` | cross-method | fast/preln/with_he variants match reference br_emission_coefficient to rel 1e-10 | internal consistency of two implementations of same formula |
| src/bremsstrahlung.rs | `test_br_hardcoded_constants` | analytic | sqrt(6pi), sqrt(3)/pi, ln(2.25), ln(1.125), BR_PREFACTOR recomputed exactly | closed-form arithmetic recomputation of hardcoded constants |
| src/bremsstrahlung.rs | `test_br_heating_integral_zero_for_planck` | analytic | \|H_BR\| < 1e-10 for Planck equilibrium | detailed balance implies zero net heating |
| src/bremsstrahlung.rs | `test_br_hii_capped_at_nh` | dimensional | K_BR finite; x_e=1.16 within 20% of x_e=1.0 | physical argument: H+ contribution capped at n_H dominates; 20% tolerance stated in comment |
| src/bremsstrahlung.rs | `test_br_precompute_returns_none_for_tiny_inputs` | structural | br_precompute returns None for theta_e or n_e < 1e-30 | guard-branch coverage |
| src/bremsstrahlung.rs | `test_br_rhs_nonzero_for_te_ne_tz` | dimensional | BR RHS > 0 at low x when T_e > T_z; sign matches dc_rhs | detailed-balance sign argument |
| src/bremsstrahlung.rs | `test_br_rhs_zero_for_planck` | analytic | max\|BR RHS\| < 1e-20 for Planck with T_e=T_z | detailed balance: [1 - n_pl(e^x - 1)] = 0 exactly |
| src/bremsstrahlung.rs | `test_gaunt_ff_nr_guards` | structural | gaunt_ff_nr returns 1.0 for degenerate inputs, >= 1.0 otherwise | guard-branch coverage |
| src/bremsstrahlung.rs | `test_gaunt_ff_positive_and_physical` | dimensional | g_ff > 0, >= 1, Z-dependence, g(low x) > 3, decreasing with x | Born approximation g ~ (sqrt(3)/pi) ln(2.25 theta_e/x) classical-limit argument stated in test |
| src/bremsstrahlung.rs | `test_softplus_all_branches` | analytic | softplus(0)=ln2, softplus(25)~25, softplus(-25)~e^-25 | closed-form softplus limits |
| src/cli.rs | `test_build_cosmology_variants` | structural | preset h values (0.71/0.6736/0.6727) and Omega_b h^2 conversion arithmetic | echoes preset definitions; conversion checked by inline arithmetic (0.04*0.7^2) |
| src/cli.rs | `test_build_injection_all_types` | structural | built scenario variants and field round-trip (z_h=2e5, drho=1e-5) | CLI plumbing |
| src/cli.rs | `test_build_injection_errors` | structural | error paths for unknown/invalid/missing params | CLI plumbing |
| src/cli.rs | `test_parse_commands` | structural | parsed CLI values echo inputs (z_h=2e5 etc.) | CLI parsing plumbing |
| src/cli.rs | `test_parse_flat_args_boolean_flag` | structural | boolean flag parsed to empty string, value flags round-trip | CLI plumbing |
| src/cli.rs | `test_parse_float_list` | structural | parsed list length 3, first element 1e3 | parsing plumbing |
| src/cli.rs | `test_solver_opts_all_flags` | structural | flag values round-trip through parser | parsing plumbing |
| src/constants.rs | `test_alpha_rho` | analytic | ALPHA_RHO ~ 0.37020884 | 30*zeta(3)/pi^4 stated in comment (deliberately non-tautological transcription guard) |
| src/constants.rs | `test_beta_mu` | analytic | BETA_MU ~ 2.1923 and identity 3*zeta(3)/zeta(2) to 1e-14 | exact zeta-function identity |
| src/constants.rs | `test_f_he` | analytic | F_HE ~ 0.07895 | Y_p/(4(1-Y_p)) with Y_p=0.24 |
| src/constants.rs | `test_g1_g2_identities` | analytic | G1=pi^2/6, G2=2*zeta(3), I4=4*G3 exact | closed-form spectral integrals |
| src/constants.rs | `test_g3_planck` | analytic | G3 = pi^4/15 to 1e-14 | closed form |
| src/constants.rs | `test_kappa_c_analytical_and_numerical` | analytic | kappa_c ~ 2.1419 via 12/beta_mu - 9G2/G3 and independent quadrature | closed form + numerical integration of 3*int x^3 M(x) dx / G3 |
| src/constants.rs | `test_theta_z_value` | analytic | theta_z(0) ~ 4.60e-10 | kT_CMB/(m_e c^2) from CODATA constants |
| src/constants.rs | `test_x_balanced` | analytic | x0 ~ 3.602 and 4G3/(3G2) identity | closed form |
| src/cosmology.rs | `test_baryon_photon_ratio` | dimensional | R(1100) ~ 0.6 window, R proportional to 1/(1+z) | standard baryon loading scaling; R(1100)~0.6 is textbook |
| src/cosmology.rs | `test_cached_f_he` | analytic | cached f_He equals Y_p/(4(1-Y_p)) | first-principles recomputation |
| src/cosmology.rs | `test_compton_y_parameter_high_z` | dimensional | y_C(1e5)>0.1, y_C(1e6)>10 | thermalization-era order-of-magnitude argument |
| src/cosmology.rs | `test_compton_y_parameter_low_z` | dimensional | y_C(500)<0.1 post-recombination | inefficient Comptonization argument |
| src/cosmology.rs | `test_compton_y_parameter_monotonic` | dimensional | y_C monotonically increasing in z | integral positivity |
| src/cosmology.rs | `test_cosmic_time_order_and_magnitude` | literature | t(0) in [3e17,5e17] s (~13.7 Gyr) | standard age of the universe |
| src/cosmology.rs | `test_cosmology_new_custom_params` | analytic | derived f_He recomputed; doubling omega_b doubles n_H | closed-form derivations from custom params |
| src/cosmology.rs | `test_default_params` | structural | Omega_m=0.26, Omega_b=0.044 echo constructor inputs | constructor round-trip |
| src/cosmology.rs | `test_density_accessors` | analytic | n_gamma(0) ~ 4.1e8 /m^3, n_H(0) ~ 0.19 /m^3 | 2*zeta(3)/pi^2 (kT/hbar c)^3 and 3H0^2 Omega_b/(8 pi G m_p) closed forms |
| src/cosmology.rs | `test_e_of_z_today` | analytic | E(0)=1 | Friedmann normalization |
| src/cosmology.rs | `test_hubble_today` | analytic | H(0) = 100 h km/s/Mpc exactly | definition of h |
| src/cosmology.rs | `test_n_h_positive_and_scaling` | dimensional | n_H scales as (1+z)^3 | number-density dilution |
| src/cosmology.rs | `test_planck2015_preset` | literature | h=0.6727, omega_b=0.02225, Y_p=0.2467 | Planck 2015 parameter paper |
| src/cosmology.rs | `test_planck2018_preset` | literature | h=0.6736, omega_b=0.02237, omega_cdm=0.1200, T=2.7255, Y_p=0.2454, N_eff=3.044 | Planck 2018 parameter paper |
| src/cosmology.rs | `test_radiation_dominated` | dimensional | E(z=1e6) matches sqrt(Omega_r)(1+z)^2 to 1% | radiation-era scaling |
| src/cosmology.rs | `test_z_eq` | analytic | z_eq in [3000,3600] | z_eq = Omega_m/Omega_rel - 1 for the stated params (~3300) |
| src/dark_photon.rs | `gamma_con_matches_chluba_cyr` | regression-pin | z_res ~ 3.21e4 (5%) and gamma_con/eps^2 ~ 9.3e10 (20%) | z_res derived analytically from omega_pl scaling; 9.3e10 explicitly labelled '(measured)' i.e. code output, framed as consistent with Chluba & Cyr (2024) Eq. 6 |
| src/dark_photon.rs | `gamma_con_scales_as_epsilon_squared` | dimensional | gamma_con ratio = 4 when epsilon doubles | epsilon^2 scaling of mixing rate |
| src/dark_photon.rs | `plasma_frequency_matches_first_principles` | analytic | omega_pl at z=1e5 vs CODATA recomputation (1e-12 rel) and (1+z)^1.5 scaling | first-principles formula; Mirizzi, Redondo & Sigl (2009) JCAP 0903, 026 Eq. 2 cited |
| src/dark_photon.rs | `resonance_round_trip` | analytic | omega_pl(z_res(m)) = m to 1e-5 | inverse-function round trip |
| src/distortion.rs | `test_bf_pure_bose_einstein` | analytic | BF recovers mu_true=2e-5 from true BE distortion to 1e-3, dT/T ~ 0; GS dT/T = -mu/beta_mu | synthetic BE round-trip + analytic photon-number bookkeeping offset |
| src/distortion.rs | `test_bf_vs_gs_greens_function_spectrum` | cross-method | BF vs GS mu, y agree to 1e-4 on realistic GF spectrum; dT offset = mu/beta_mu | two independent decomposition methods |
| src/distortion.rs | `test_bf_vs_gs_mixed` | cross-method | both methods recover synthetic mu=3e-6, y=1e-6, dT=5e-7; BF-GS agreement 1e-4; dT offset mu/beta_mu | synthetic round-trip + two-method agreement |
| src/distortion.rs | `test_bf_vs_gs_pure_mu` | cross-method | BF vs GS mu agree to 1e-4; y < O(mu^2); dT offset = mu/beta_mu to 1e-3 | two independent decomposition methods + analytic offset prediction mu/beta_mu |
| src/distortion.rs | `test_decompose_mixed_mu_y` | analytic | recover mu=5e-6, y=2e-6 to 1% from mixed synthetic input | synthetic round-trip |
| src/distortion.rs | `test_decompose_pure_delta_t` | analytic | recover dT/T=1e-6 to 1% with negligible mu, y | synthetic round-trip; dT/T recovery from energy conservation shown in comment |
| src/distortion.rs | `test_decompose_pure_mu` | analytic | recover injected mu_true=1e-5 to 1%, spurious y < 1% mu | synthetic input with known parameters (round-trip) |
| src/distortion.rs | `test_decompose_pure_y` | analytic | recover injected y_true=1e-6 to 1%, spurious mu < 1% y | synthetic round-trip |
| src/distortion.rs | `test_delta_n_to_intensity_mjy` | dimensional | intensity positive/finite, linear in dn, sign flips with dn | sign and linearity arguments only; no absolute magnitude asserted |
| src/distortion.rs | `test_firas_check_values` | literature | mu_frac = y_frac = 0.5 for half-limit inputs | FIRAS 95% limits mu < 9e-5, y < 1.5e-5 (Fixsen et al. 1996); assertion itself is arithmetic on those constants |
| src/distortion.rs | `test_gram_schmidt_pure_delta_t` | analytic | GS recovers dT/T=1e-6 to 1e-4; mu, y ~ 0 | synthetic round-trip |
| src/distortion.rs | `test_gram_schmidt_pure_mu` | analytic | GS recovers mu_true=1e-5 to 1e-4; y ~ 0; dT/T = 0 by CJ2014 basis construction | synthetic round-trip; CJ2014 basis property |
| src/distortion.rs | `test_gram_schmidt_pure_y` | analytic | GS recovers y_true=1e-6 to 1e-4; mu, dT/T ~ 0 | synthetic round-trip |
| src/dm_baryon.rs | `test_ah21_fig1_benchmark` | literature | z_dec ~ 5e6 (window 3e6-8e6), \|mu\| ~ 1e-6 (window) | AH21 Fig. 1 quoted values |
| src/dm_baryon.rs | `test_f_chi_linearity` | dimensional | heating rate ratio ~ 0.5 when f_chi halved | linearity in f_chi (physical argument) |
| src/dm_baryon.rs | `test_gamma_chi_p_spot_value` | analytic | Gamma_chi_p = 1.2472e-12 s^-1 (0.1%) | hand evaluation of AH21 Eqs. 2.13+2.16 closed form |
| src/dm_baryon.rs | `test_mu_max_tight_coupling` | literature | mu = -8.6e-5 (20%) for m=0.1 MeV | AH21 Eq. 2.20 tight-coupling limit |
| src/dm_baryon.rs | `test_n_indices_and_electron_scatterer` | structural | finite non-negative Q for n in {-2,2,4,6}; electron rate finite and <= 0 | finiteness/sign checks |
| src/dm_baryon.rs | `test_prefactor_matches_ah21` | literature | heating prefactor 1.7095e-6 (1%) | AH21 Eq. 2.11, hand-computed from Planck 2018 params (comment: 'hand calculation, not code output') |
| src/dm_baryon.rs | `test_q_bounded_and_dm_colder` | dimensional | Q in [0,1], T_chi <= T_gamma | physical bounds |
| src/dm_baryon.rs | `test_validation_errors` | structural | constructor rejects out-of-range params | input validation |
| src/double_compton.rs | `test_dc_coefficient_positive` | dimensional | K_DC >= 0 over x grid | physical positivity |
| src/double_compton.rs | `test_dc_drives_to_equilibrium` | dimensional | DC RHS < 0 where positive distortion at low x (absorption) | detailed-balance sign argument |
| src/double_compton.rs | `test_dc_emission_coefficient_magnitude` | analytic | K_DC(x=1, theta=4.6e-5) in [1e-14, 1e-7] and within 10% of hand calc ~9.5e-11 | hand-recomputed (4a/3pi) theta^2 I4 H_dc(1)/(1+14.16 theta); formula from CS2012 |
| src/double_compton.rs | `test_dc_gaunt_factor_at_specific_x_values` | literature | g_dc(1,0) matches manual H_dc(1)=e^-2*4.8125 to 1e-12; g_dc(5,0) < 5% of g(0,0) | Chluba & Sunyaev 2012 Eq. 13 polynomial, hand-evaluated |
| src/double_compton.rs | `test_dc_gaunt_factor_literature_values` | literature | g_dc(0,0) = I4_pl = 4pi^4/15; g_dc(10,0) matches manually evaluated polynomial; strong suppression at x=10 | Chluba & Sunyaev 2012 Eq. 13 polynomial, manually recomputed in test |
| src/double_compton.rs | `test_dc_high_freq_suppression_extremes` | analytic | H_dc(200)=0 (guard), H_dc(0.01) in (0.99, 1.0) | closed-form evaluation exp(-0.02)*(1+0.015+...) ~ 0.995; x>100 guard is structural |
| src/double_compton.rs | `test_dc_polynomial_coefficients_cs2012` | literature | Horner form equals expanded polynomial with coefficients 1, 3/2, 29/24, 11/16, 5/12; H_dc(0)=1; H_dc(0.5) exact | Chluba & Sunyaev 2012 Eq. 13 |
| src/double_compton.rs | `test_dc_rhs_zero_for_planck` | analytic | max\|DC RHS\| < 1e-12 for Planck with T_e=T_z | detailed balance [1 - n_pl(e^x - 1)] = 0 |
| src/double_compton.rs | `test_dc_suppression_high_x` | analytic | H_dc(50) < 1e-20 | e^{-2x} suppression, e^{-100} bound |
| src/double_compton.rs | `test_dc_suppression_low_x` | analytic | H_dc(0) = 1 | polynomial limit at x=0 |
| src/electron_temp.rs | `test_equilibrium_for_bose_einstein` | cross-method | rho_e matches independent trapezoidal I4/(4G3) to rel 1e-6; rho_e > 1 for mu > 0; monotone in mu | independent numerical integration in test + physical direction argument |
| src/electron_temp.rs | `test_equilibrium_for_planck` | analytic | rho_e = 1 within 1e-3 for Planck | Compton equilibrium T_e = T_z for Planck spectrum (exact) |
| src/energy_injection.rs | `test_characteristic_redshift_all_variants` | structural | z_h round-trips; continuous scenarios return None | API plumbing |
| src/energy_injection.rs | `test_custom_injection_scenario` | structural | Custom closure returns 1e-15 constant | echoes the closure |
| src/energy_injection.rs | `test_heating_rate_all_scenarios` | analytic | SingleBurst peak rate = drho*Gauss(0)*H*(1+z) to 1e-10; signs/finiteness elsewhere | closed-form peak rate |
| src/energy_injection.rs | `test_interp_log_z_basic` | analytic | log-space midpoint interpolates to 1.5; endpoints exact; out-of-range 0 | interpolation arithmetic |
| src/energy_injection.rs | `test_load_heating_table` | structural | 3 rows loaded, z[0]=1e3 | file I/O round-trip |
| src/energy_injection.rs | `test_photon_source_rate_nonzero` | analytic | peak photon source rate matches closed form to 1e-10; exp(-50) off-peak suppression | documented oracle: Chluba 2015 / scenario definition, closed form |
| src/energy_injection.rs | `test_single_burst_normalization` | analytic | integrated d(drho)/dz equals drho to 5% | Gaussian normalization |
| src/energy_injection.rs | `test_tabulated_heating_matches_burst` | cross-method | tabulated rate matches closed-form SingleBurst to 1% | two paths through the same physics (table interpolation vs closed form) |
| src/energy_injection.rs | `test_tabulated_photon_source` | structural | positive finite peak rate, zero heating, name check | API plumbing |
| src/energy_injection.rs | `test_tabulated_photon_source_bilinear_interpolation` | analytic | bilinear midpoint value 1.0 (hand-derived in comment), corners exact, out-of-range 0 | bilinear interpolation arithmetic |
| src/energy_injection.rs | `test_warn_strong_distortion` | structural | warning presence/absence at thresholds (0.1 amplitude, x_inj>150) | warning-logic thresholds |
| src/fh_basis.rs | `b0_is_eg_plus_mb_y0` | structural | b0(e_mu) = e_G + mu-column of M_B | definition self-consistency of the b0 assembly |
| src/fh_basis.rs | `boost_of_g_identity` | analytic | M_B G-column = 3 e_G + e_Y0 exactly | exact SS-II identity O_x G = 3G + Y_0 |
| src/fh_basis.rs | `delta_n_synthesis_is_linear` | analytic | delta_n(e_G)=G(x), (e_M)=M(x), (e_Y0)=Y(x) to 1e-12 | basis-function definitions |
| src/fh_basis.rs | `loads_all_matrices_and_vectors` | analytic | eps=4 for G/Y_k, eps_mu=1/gamma_rho, beta_M=3G2/(2G1) | closed-form energy weights and identities |
| src/fh_basis.rs | `m_t_writes_emission_into_mu_column` | structural | M_T equals M_K except mu column = x_c*D_0 | matrix-assembly definition |
| src/fh_basis.rs | `mb_quadruples_energy` | analytic | eps^T M_B = 4 eps^T | boost energy-scaling identity |
| src/fh_basis.rs | `mk_conserves_energy` | analytic | eps^T M_K = 0 per column to 1e-12 rel | energy conservation identity |
| src/fh_basis.rs | `mk_zero_structure` | analytic | M_K Theta-row/col and mu-col identically zero | photon-number conservation + mu null direction (EC26/SS-I) |
| src/fh_basis.rs | `mok_is_commutator` | analytic | M_OK = [M_B, M_K] recomputed to 1e-9 rel | first-principles recomputation from loaded matrices |
| src/fh_basis.rs | `observer_ls_matches_python` | cross-method | Rust binned-LS observer conversion matches python fh_basis.py reference values (1e-4) | python/spectroxide/fh_basis.py output generated 2026-06-12 (independent implementation) |
| src/fh_basis.rs | `project_d_reproduces_ec26_table1` | literature | (y0, mu) = (-0.0207, 0.8385) within 5e-4 | EC26 Table 1 (N=0); comment states independent paper target, not code output |
| src/fh_basis.rs | `yk_recursion_holds` | analytic | O_x Y_{k-1} = 4 Y_k via centred finite difference | boost recursion identity |
| src/fh_basis.rs | `yk_zero_matches_y_shape` | analytic | combinatoric Y_0 equals closed-form Y(x) to 1e-12 | two closed forms of the same function |
| src/fh_los.rs | `grid_is_ascending_and_spans_window` | structural | grid length > 700, ascending, last <= eta_0 | grid construction |
| src/fh_los.rs | `no_source_distortion_transfer_vanishes` | cross-method | distortion slots exactly zero with source off; Theta channel matches standard LOS transfer to 5% L2 | exact zero-source argument + agreement with the standard-sector LOS implementation |
| src/fh_los.rs | `reionization_optical_depth_matches_class` | cross-method | tau_reio = 0.05597 within 3% | classy v3.2.5 (CLASS) value queried 2026-06-12 with matched cosmology |
| src/fh_perturbed.rs | `background_energy_closure` | analytic | eps.y0 = 0 to 1% of drho post-injection | energy conservation (eps^T M_T = 0); floor set by documented ~0.6% impulsive-IC projection residual |
| src/fh_perturbed.rs | `background_two_path_consistency` | cross-method | Gaussian-sourced background vs fh_solver impulsive-IC path agree to <1% of drho | two independent integration paths (plan S-3) |
| src/fh_perturbed.rs | `collapse_distortion_slots_vanish` | analytic | non-Theta RHS slots < 1e-10*scale in collapse state | structure of the FH operator (Comptonization off) |
| src/fh_perturbed.rs | `collapse_monopole_matches_standard` | analytic | FH RHS in collapse limit equals standard photon Boltzmann hierarchy expressions (1e-12 rel) | Ma & Bertschinger-style hierarchy equations recomputed inline |
| src/fh_perturbed.rs | `conservation_c3` | analytic | boost-covariance relations: Theta-slot/Theta = 1+3*Thetabar, Y0-slot/Theta = Thetabar, no mu leakage, energy ratio 4*Thetabar (1e-3) | exact conservation identities (plan C-3 / EC26 Sect. verified numerically) |
| src/fh_perturbed.rs | `dln_gamma_rd_limit` | analytic | d ln gamma_con/d ln(1+z) = -3 within 0.1 at z=5e5 | RD limit of EC26 Eq. 3.15 (H ~ (1+z)^2, xi_e ~ 1) |
| src/fh_perturbed.rs | `interpolation_grid_convergence` | structural | doubling n_pert_samples shifts monopole < 1e-2 rel | numerical-convergence self-consistency (same code, two resolutions) |
| src/fh_perturbed.rs | `source_gaussian_resolved` | analytic | net injected energy ~ 0 (2% of drho); Theta channel negative and O(drho) | energy split of the impulsive IC (conservation) |
| src/fh_solver.rs | `dense_solve_identity` | analytic | I x = b solves to x = b (1e-12) | trivial linear-algebra identity |
| src/fh_solver.rs | `xc_reproduces_mu_visibility` | literature | tau_FH(z) reproduces (z/z_dc)^2.5 with z_dc=1.98e6 within 10% | standard mu-visibility exponent, Chluba-2015 x_c fit; comment states target derived from independent visibility formula (pitfall #9) |
| src/greens.rs | `assert_photon_gf_regime` | structural | panics if z_h inside (5e4, 2e5) transition band | validity-window guard; not a test function (inventory picked up assert! in production helper) |
| src/greens.rs | `test_distortion_from_heating_spectrum` | analytic | spectrum energy = drho within factor 2; mu ~ 1.401 drho within factor 2 | energy conservation + mu = 1.401 drho/rho (= 3/kappa_c, standard BE result; CLAUDE.md validation target) |
| src/greens.rs | `test_distortion_from_photon_injection_spectrum` | dimensional | positive bump near x_inj=5, nonzero spectrum, mu > 0 for x_inj > x0 | sign and existence arguments; contradictory comment about mu sign resolved in favor of x_inj > x0 -> mu > 0 |
| src/greens.rs | `test_greens_high_z_is_temperature_shift` | analytic | G(x, z=5e6) = 0.25 G_bb(x) within 1% | asymptotic limit: J_bb*, J_y -> 0 leaves pure T-shift, coefficient 1/4 from drho/rho = 4 dT/T |
| src/greens.rs | `test_greens_mu_era` | dimensional | J_mu(3e5) > 0.5 and J_bb*(3e5) > 0.5 | mu-era dominance argument |
| src/greens.rs | `test_greens_y_era` | dimensional | J_mu(5e3) < 0.1 | y-era dominance argument |
| src/greens.rs | `test_heating_rate_per_redshift_sign_convention` | analytic | rate > 0, rate_per_z < 0, rate_per_z = -rate/(H(1+z)) to 1e-10; positive dq/dz gives mu > 0 | definition of dz = -H(1+z) dt sign convention |
| src/greens.rs | `test_mu_from_delta_injection` | cross-method | integrated mu matches delta-function limit (3/kappa_c) J_bb* J_mu drho within 5% | closed-form delta limit of the same GF integrand; visibilities shared with code path |
| src/greens.rs | `test_mu_from_photon_injection_balanced` | analytic | \|mu(x0)\| < 1% of \|mu(x=10)\|; P_s(x0) > 0.99 | X_BALANCED = 4/(3 alpha_rho) ~ 3.60, Chluba 2015 arXiv:1506.06582 Eq. 31 (cited at constant definition) |
| src/greens.rs | `test_mu_from_photon_injection_sign_flip` | dimensional | mu > 0 for x_inj=10 > x0, mu < 0 for x_inj=2 < x0 | sign argument from energy-vs-number balance at x0 (Chluba 2015 photon injection) |
| src/greens.rs | `test_mu_y_from_heating_consistency` | cross-method | mu_y_from_heating equals separate mu_from_heating/y_from_heating to 1e-10 | internal consistency of joint vs separate code paths |
| src/greens.rs | `test_photon_survival_limits` | dimensional | P_s(x=10) > 0.99, P_s(x=1e-5) < 1e-10 at z=2e5 | asymptotic limits of optical-depth integral |
| src/greens.rs | `test_tau_ff_fallback_to_analytic_at_high_z` | structural | numerical P_s equals analytic branch exactly (1e-15) above z=5e4 | branch-selection coverage: fallback path returns the analytic function |
| src/greens.rs | `test_tau_ff_limits` | dimensional | P_s = 1 at z=100; P_s(x=10, z=1e4) > 0.95; P_s(x=1e-5, z=1e5) < 0.01 | asymptotic limits |
| src/greens.rs | `test_visibility_functions_physical_bounds` | dimensional | J's in [0,1] (J_T in [-0.2,1]), asymptotic limits, monotonicity in z | physical bounds and asymptotics; J_T negative excursion explained in comment |
| src/greens.rs | `test_visibility_spot_checks_transition_region` | literature | J_mu, J_y, J_bb* at z=1e5 match hand-evaluated fitting formulas to 1e-10; sanity ranges at z=1e5 and 5e4 | Chluba 2013 arXiv:1304.6120 Eq. 5 (J_mu, J_y); Chluba 2015 arXiv:1506.06582 Eq. 13 (J_bb*) |
| src/greens.rs | `test_x_c_br_dominance_at_low_z` | dimensional | x_c_BR > x_c_DC at z=1e4 | BR dominates at low z (standard result) |
| src/greens.rs | `test_x_c_dc_dominance_at_high_z` | dimensional | x_c_DC > x_c_BR at z=2e6 | DC dominates thermalization era (standard result) |
| src/greens.rs | `test_x_c_physics_properties` | dimensional | x_c positive finite, < 1, >= max(x_c_dc, x_c_br) (quadrature), smooth in z | physical/structural arguments about critical frequency |
| src/greens.rs | `test_y_from_heating` | analytic | y = drho/(4 rho) within 5% for y-era burst | standard y-era relation y = drho/4rho |
| src/grid.rs | `test_grid_dx_consistency` | structural | dx[i] = x[i+1]-x[i] to 1e-14 | internal consistency |
| src/grid.rs | `test_grid_log_region` | structural | dx/x approximately constant in log region (10%) | grid-construction property |
| src/grid.rs | `test_refinement_zone_no_duplicates` | structural | relative spacing > 1e-10 everywhere | grid-construction property |
| src/kompaneets.rs | `test_coupled_inplace_preserves_planck` | analytic | converged, rho_e=1 to 1e-10, Newton delta < 1e-12, max\|dn\| < 1e-14 at equilibrium | Planck equilibrium fixed point |
| src/kompaneets.rs | `test_coupled_inplace_with_dcbr` | analytic | converged, rho_e=1 to 1e-10, Newton delta < 1e-10 with DC/BR at detailed balance | detailed-balance equilibrium fixed point |
| src/kompaneets.rs | `test_kompaneets_energy_conservation` | analytic | \|dE/E\| < 1e-4 over 10 steps with T_e=T_z | first-order energy conservation at T_e=T_z; O(dn^2) correction bounded in comment |
| src/kompaneets.rs | `test_kompaneets_photon_number_conservation` | analytic | \|dN/N\| < 1e-4 over 10 steps | Kompaneets is exactly number-conserving (divergence form) |
| src/kompaneets.rs | `test_kompaneets_preserves_planck` | analytic | max\|dn\| < 1e-12 after step with dn=0, T_e=T_z | Planck is exact Kompaneets equilibrium |
| src/kompaneets.rs | `test_kompaneets_rhs_planck_cancellation` | analytic | rhs = 0 to 1e-20 at equilibrium; rhs scales linearly with (phi-1) to 0.1% | analytic Planck identity cancellation (CLAUDE.md pitfall #1); linearity in epsilon |
| src/kompaneets.rs | `test_kompaneets_te_gt_tz_positive_drho_all_solvers` | dimensional | drho/rho > 0 for T_e > T_z in CN, backward-Euler, and nonlinear solvers | sign argument: upscattering transfers energy electrons to photons |
| src/kompaneets.rs | `test_kompaneets_y_distortion_magnitude` | analytic | drho/rho = 4y with y=(theta_e-theta_z)dtau, within 5% | standard y-distortion relation drho/rho = 4y (Zeldovich-Sunyaev) |
| src/kompaneets.rs | `test_kompaneets_yields_ysz_shape` | analytic | Pearson correlation of dn with Y_SZ(x) > 0.95 for small T_e > T_z | defining property of y-distortion: Kompaneets linear response is Y_SZ shape |
| src/kompaneets.rs | `test_nonlinear_preserves_planck` | analytic | max\|dn\| < 1e-12 for nonlinear solver at equilibrium | Planck equilibrium |
| src/kompaneets.rs | `test_thomas_solve_inplace_2x2` | analytic | 2x2 tridiagonal solution x=[1,1] | exact linear algebra |
| src/kompaneets.rs | `test_thomas_solve_inplace_3x3` | analytic | 3x3 tridiagonal solution x=[1,1,1] | exact linear algebra |
| src/kompaneets.rs | `test_thomas_solve_inplace_identity` | analytic | diagonal system solution rhs/diag | exact linear algebra |
| src/perturbations.rs | `adiabatic_relation_early` | analytic | delta_b/delta_g = 3/4 to 1e-3 | adiabatic initial-condition relation |
| src/perturbations.rs | `background_eta_radiation_limit` | analytic | eta(a) = a/((H0/c) sqrt(Omega_r)) in deep RD (1e-3) + round trip | RD closed form |
| src/perturbations.rs | `background_eta_today_magnitude` | dimensional | eta_0 in [1e4, 2e4] Mpc | standard-cosmology magnitude (~1.4e4 Mpc) |
| src/perturbations.rs | `background_tau_prime_scaling` | dimensional | tau' ratio = 4 for z doubling (a^-2 scaling) | n_e a scaling in the fully-ionized era |
| src/perturbations.rs | `comoving_curvature_conserved_superhorizon` | analytic | R = 1 conserved through equality to 5e-3 | comoving-curvature conservation, normalization R=1 |
| src/perturbations.rs | `delta_b_prime_matches_finite_difference` | cross-method | analytic delta_b' matches centred finite difference to 1e-3 | FD vs analytic derivative |
| src/perturbations.rs | `evolve_rejects_bad_input` | structural | error paths for bad k / a-lists | input validation |
| src/perturbations.rs | `initial_conditions_satisfy_momentum_constraint` | analytic | momentum-constraint residual < 1e-3 of scale at ICs | Einstein momentum constraint (independent of the energy constraint used to evolve) |
| src/perturbations.rs | `momentum_constraint_maintained` | analytic | unused momentum constraint residual < 2e-3 through evolution | Bianchi-identity consistency check |
| src/perturbations.rs | `superhorizon_phi_constant_in_rd` | analytic | phi drift < 2e-3; psi/phi = 1/(1+2R_nu/5) to 5e-3 | standard superhorizon RD result |
| src/perturbations.rs | `tight_coupling_slip_is_small` | analytic | \|theta_b - theta_g\| < 1e-3 * theta_b in tight coupling | tight-coupling limit |
| src/power_spectra.rs | `k_grid_resolves_acoustic` | structural | grid length > 1000, monotone, spans [k_min, k_max] | grid construction |
| src/power_spectra.rs | `output_multipoles_cover_range` | structural | multipole list starts at 2 and ends at l_max | list construction |
| src/power_spectra.rs | `primordial_pivot` | analytic | P_R(k_pivot) = A_s exactly | definition of the pivot amplitude |
| src/recombination.rs | `test_freeze_out` | literature | X_e(100) in [1e-4, 5e-3] (~2-4e-4) | RECFAST freeze-out value |
| src/recombination.rs | `test_fully_ionized_high_z` | analytic | X_e(1e6) = 1 + 2 f_He within 0.01 | Saha fully-ionized limit |
| src/recombination.rs | `test_helium_saha_transitions` | dimensional | He2+ > 0.99 at z=5e4, He+ > 0.5 at z=3000, monotone decrease | Saha-regime physical bounds |
| src/recombination.rs | `test_recombination_history_interpolation_smooth` | structural | interpolation monotone between adjacent points | interpolation smoothness |
| src/recombination.rs | `test_recombination_history_matches_uncached` | structural | cached vs direct ionization_fraction agree to 1% | cache correctness (same algorithm) |
| src/recombination.rs | `test_recombination_history_monotonic` | dimensional | cached X_e monotone in z over [100, 2000] | physical monotonicity |
| src/recombination.rs | `test_recombination_physical_values` | literature | X_e(1100)~0.14, X_e(800)<0.01, X_e(200) window, monotonicity, smoothness | RECFAST milestone values |
| src/recombination.rs | `test_xe_vs_recfast_milestones` | literature | X_e(1100)~0.14, X_e(800)~3e-3, X_e(200)~3e-4 windows | RECFAST (Seager, Sasselov & Scott 1999); Peebles fudge F=1.125 (Chluba & Thomas 2011) |
| src/solver.rs | `test_decomposition_pure_mu` | analytic | injected mu=1e-5 recovered to 5%, y < 1e-7 | round trip through the least-squares decomposition |
| src/solver.rs | `test_decomposition_pure_tshift` | analytic | pure G(x) input decomposes to mu, y < 5e-8 | orthogonality of the decomposition basis |
| src/solver.rs | `test_energy_conservation` | analytic | drho_out matches injected drho within 5% | energy conservation |
| src/solver.rs | `test_greens_function_consistency` | dimensional | mu scales linearly: mu2/mu1 = 2.0 within 1% | linearity of the Green's function in drho |
| src/solver.rs | `test_no_injection_stays_planck` | analytic | max\|dn\| < 1e-6 with no injection (adiabatic-cooling signal is O(1e-8)) | Planck spectrum stays Planck up to O(Lambda) expansion cooling |
| src/solver.rs | `test_pde_vs_greens_mu_era` | cross-method | PDE mu vs Green's function mu within 12%; energy within 10% | PDE solver vs Chluba 2013 Green's function |
| src/solver.rs | `test_pde_vs_greens_y_era` | cross-method | PDE y vs Green's function y within 10% | PDE solver vs Green's function |
| src/solver.rs | `test_pde_y_at_multiple_redshifts` | analytic | energy conservation within 10% at z_h = 1e4, 3e4, 5e4 | energy conservation (GF y only printed, not asserted) |
| src/solver.rs | `test_snapshots_at_requested_redshifts` | structural | snapshot z within 0.1% of requested | snapshot plumbing |
| src/solver.rs | `test_solver_builder_disable_dcbr` | analytic | mu = (3/kappa_c)*drho = 1.401*drho within 5%; energy within 2% | documented oracle: SZ 1970 / Chluba 2013 with J_bb* -> 1 (pure Kompaneets) |
| src/solver.rs | `test_solver_builder_split_dcbr` | dimensional | mu > 0 for heating; drho finite | sign argument only |
| src/solver.rs | `test_solver_reset` | structural | reset clears state; rerun reproduces mu to 1e-10 | reproducibility check |
| src/solver.rs | `test_solver_run_to_result` | structural | step_count > 0, snapshot z, JSON keys present | API plumbing |
| src/spectrum.rs | `test_bose_einstein_branches` | analytic | BE branch limits: ~Planck at low x mu=0, e^-(x+mu) at high x | closed-form limits |
| src/spectrum.rs | `test_bose_einstein_reduces_to_planck` | analytic | BE(x, mu=0) = Planck(x) | definition |
| src/spectrum.rs | `test_compton_equilibrium_planck` | analytic | T_e^eq/T_z = 1 within 1e-3 for Planck | Compton equilibrium exact for Planck |
| src/spectrum.rs | `test_delta_rho_over_rho_and_delta_n_over_n` | analytic | for dn = dT/T G_bb: drho/rho = 4 dT/T and dN/N = 3 dT/T within 1% | int x^3 G_bb = 4 G3, int x^2 G_bb = 3 G2 (exact integrals) |
| src/spectrum.rs | `test_g_bb_all_branches` | analytic | g_bb branch values: ~1/x at low x, x e^-x at high x, x e^x/(e^x-1)^2 mid | closed-form G_bb(x) = x e^x/(e^x-1)^2 and its limits |
| src/spectrum.rs | `test_mu_shape_sign_change_at_beta_mu` | analytic | M(x) flips sign across beta_mu (negative below, positive above) | algebraic factor (x/beta_mu - 1) in M(x); Chluba & Sunyaev 2012, MNRAS 419, 1294 cited |
| src/spectrum.rs | `test_planck_high_x_branch` | analytic | n_pl(600) = e^-600 (Wien limit) | Wien expansion |
| src/spectrum.rs | `test_planck_identity_analytical` | analytic | dn_pl/dx + n_pl(1+n_pl) = 0 to rel 1e-13 | exact Planck identity (derivation in doc comment) |
| src/spectrum.rs | `test_planck_low_x` | analytic | n_pl(1e-8) ~ 1/x (Rayleigh-Jeans limit) | RJ expansion |
| src/spectrum.rs | `test_planck_moderate_x` | analytic | n_pl(1) = 1/(e-1) | closed form |
| src/spectrum.rs | `test_spectral_integral_g2` | analytic | int x^2 n_pl dx = 2 zeta(3) to rel 1e-5 | exact Riemann zeta integral |
| src/spectrum.rs | `test_spectral_integral_g3` | analytic | int x^3 n_pl dx = pi^4/15 to rel 1e-6 | exact Riemann zeta integral |
| src/spectrum.rs | `test_y_shape_zero_crossing` | analytic | Y_SZ zero crossing at x = 3.83 +/- 0.01 | standard SZ null x0 ~ 3.830, root of x coth(x/2) = 4 (well-known analytic result) |
| tests/adversarial_inputs.rs | `test_builder_rejects_bad_inputs` | structural | builder/validate return is_err() for bad cosmology/grid/injection/z_range | API contract |
| tests/adversarial_inputs.rs | `test_grid_constructor_panics` | structural | grid constructors panic on empty/single-point/n=1 inputs | API contract |
| tests/adversarial_inputs.rs | `test_injection_characteristic_redshifts` | structural | characteristic_redshift returns (z_h, z_h+7sigma_z) matching input fields | API contract (7-sigma window is the code's own convention) |
| tests/adversarial_inputs.rs | `test_injection_near_z_start` | analytic | energy ratio tight/wide z_start > 0.90 (3-sigma Gaussian tail loses ~0.1%) | Gaussian tail integral (closed form); 10% tolerance is loose |
| tests/adversarial_inputs.rs | `test_two_near_simultaneous_bursts` | analytic | linearity mu(2*drho)/mu(drho) = 2 within 0.1 | linearity of the perturbed Boltzmann equation for small distortions |
| tests/adversarial_inputs.rs | `test_very_narrow_sigma_z` | structural | drho finite and \|drho\| > 1e-7 (nonzero) for sigma_z=10 burst | finiteness/nonzero check, no physics target |
| tests/cli_integration.rs | `test_cli_greens_json_output` | cross-method | CLI gf_mu at z_h=2e5 matches (3/kappa_c)*J_bb*J_mu*drho to 10%; mu/y > 20 | Chluba 2013 MNRAS 434, 352 Eq. 5, evaluated via the library's own visibility fits (binary vs library cross-check) |
| tests/cli_integration.rs | `test_cli_solve_single_burst_json` | analytic | energy sum mu/1.401 + 4y in [0.5, 2.0]e-5 for drho=1e-5 at z=5e4; mu,y > 0 | energy conservation identity mu/1.401 + 4y = drho/rho |
| tests/cli_integration.rs | `test_execute_sweep_parallel_rows_consistent` | cross-method | sweep rows: GF mu matches Chluba Eq.5 to 15%, PDE-vs-GF mu to 30%, row order 1e-12, drho spread > 0.5% | Chluba 2013 Eq. 5 (via library fns) + PDE-vs-GF agreement; tolerances justified in doc comment |
| tests/convergence_order.rs | `convergence_order_joint` | analytic | joint grid+timestep Richardson median order in [1.0, 2.5] | IMEX adaptive theoretical order 1.0-1.5 |
| tests/convergence_order.rs | `convergence_order_spatial_full_physics` | analytic | mu spread < 1%; L2 Richardson order in [0.8, 2.2] | mixed CN + backward-Euler IMEX theoretical order (1st-2nd) |
| tests/convergence_order.rs | `convergence_order_spatial_gaussian` | analytic | L2 Richardson order > 1.8 | pure CN on smooth IC: theoretical 2nd order |
| tests/convergence_order.rs | `convergence_order_spatial_pure_kompaneets` | analytic | mu spread across refinements < 1%; spectral L2 Richardson order > 1.5 | Crank-Nicolson theoretical O(dx^2) convergence |
| tests/convergence_order.rs | `convergence_order_temporal_full_physics` | analytic | temporal Richardson order in [0.8, 1.5] | mixed CN+BE with adaptive stepping: theoretical ~1st order |
| tests/convergence_order.rs | `convergence_order_temporal_pure_kompaneets` | analytic | temporal Richardson order in [0.8, 1.5] | adaptive dy_max stepping gives effective 1st-order temporal control (theoretical argument in comment) |
| tests/convergence_order.rs | `convergence_quasi_stationary_te_consistency` | dimensional | rho_e in (1, 1.001) shortly after drho=1e-5 injection; rho_e relaxes toward 1 at late time | perturbative quasi-stationary T_e: rho_e - 1 = O(drho) ~ 1e-5 for small injection |
| tests/convergence_order.rs | `recombination_x_e_sanity_checks` | dimensional | X_e in (0, 1.17], X_e(1400) > 0.5, X_e(200) < 0.01, monotone decrease; cached vs direct integration < 0.2%; cosmology sensitivity > 1% | physical range/monotonicity of recombination + internal table-vs-direct cross-check |
| tests/cosmotherm_comparison.rs | `test_adiabatic_cooling_mu_vs_cosmotherm` | literature | PDE adiabatic-cooling mu matches CosmoTherm decomposed mu to 5%; both in [-5e-9, -1e-9] | data/cosmotherm/DI_cooling.dat (Chluba & Sunyaev 2012, Chluba 2016 Fig. 1: mu_cool ~ -3e-9) |
| tests/cosmotherm_comparison.rs | `test_cosmotherm_cooling_pde_comparison` | literature | CT cooling mu < 0, \|mu\| in [1e-10, 1e-7]; PDE mu/drho in [0.8, 1.6]; scaled PDE/CT shape ratio in [0.1, 10] | data/cosmotherm/DI_cooling.dat; mu_cool ~ -(2-3)e-9 (Chluba 2016); mu=1.401*drho relation |
| tests/cosmotherm_comparison.rs | `test_cosmotherm_cooling_sign_convention` | dimensional | negated cooling dI < 0 at ~200 GHz; peak \|dI\| in 50-1000 GHz | sign/location argument on the CosmoTherm dataset (data/cosmotherm/DI_cooling.dat) |
| tests/cosmotherm_comparison.rs | `test_cosmotherm_damping_mu_y_comparison` | literature | CT damping mu in [1e-9, 1e-6], \|y\| < 1e-6; PDE mu/drho in (-0.1, 1.5) | data/cosmotherm/DI_damping.dat; LCDM mu ~ 2.0e-8 (Chluba 2016) |
| tests/cosmotherm_comparison.rs | `test_cosmotherm_data_loads` | structural | > 4000 rows, equal column lengths, sign pattern of DI files | file-structure check on data/cosmotherm/DI_cooling.dat and DI_damping.dat |
| tests/cosmotherm_comparison.rs | `test_cosmotherm_single_burst_spectral_shape` | analytic | mu/drho in [0.8, 1.6]; dI sign change between 100 and 150 GHz (beta_mu = 2.19); Wien-tail falloff; peak \|dI\| range | mu-distortion shape M(x) zero crossing at x = beta_mu and mu = 1.401*drho analytic relation |
| tests/cosmotherm_comparison.rs | `test_pde_intensity_conversion_sanity` | dimensional | \|dI(300 GHz)\| in [1e-5, 1e-1] MJy/sr for drho=1e-5 at z=5e4 | order-of-magnitude estimate dI ~ y*Y_SZ*I_0 ~ 2e-3 MJy/sr derived in comment |
| tests/coverage_gaps.rs | `boundary_conditions_delta_n_zero_at_edges` | structural | dn[n-1] < 1e-20 (Dirichlet BC), interior nonzero | scheme design contract (Kompaneets pins dn=0 at x_max where n_pl ~ e^-50) |
| tests/coverage_gaps.rs | `coupled_vs_split_dcbr_consistency` | cross-method | Newton-coupled vs operator-split DC/BR: mu to 25%, drho to 5% | two independent numerical schemes in the same solver; tolerance rationale in comment |
| tests/coverage_gaps.rs | `energy_conservation_annihilating_dm_pwave` | dimensional | p-wave drho > s-wave drho; p-wave mu/drho > s-wave mu/drho | scaling argument: p-wave rate prop (1+z)^3 vs s-wave (1+z)^2 |
| tests/coverage_gaps.rs | `energy_conservation_annihilating_dm_swave` | analytic | solver drho matches integrated heating rate to 15%; mu, y > 0 | energy conservation vs in-test quadrature of the analytic heating rate |
| tests/coverage_gaps.rs | `energy_conservation_decaying_particle` | analytic | solver drho matches independently integrated heating rate (int rate/[H(1+z)] dz) to 10% | energy conservation vs in-test quadrature of the analytic heating rate |
| tests/coverage_gaps.rs | `energy_conservation_photon_injection_x1` | analytic | drho = (dN/N)*G2*x_inj/G3 to 10% | closed-form energy integral of the narrow-Gaussian photon injection |
| tests/coverage_gaps.rs | `energy_conservation_tabulated_heating` | analytic | drho recovers the tabulated Gaussian's integrated 1e-5 to 5% | normalized Gaussian integrates to drho by construction |
| tests/coverage_gaps.rs | `extract_y_matches_joint` | structural | extract_mu_y_joint() equals snapshot mu/y to 1e-15; y > 0 | API self-consistency (same decomposition path) |
| tests/coverage_gaps.rs | `grid_find_index_above_max` | structural | find_index clamps to n-1 above x_max | API contract |
| tests/coverage_gaps.rs | `grid_find_index_below_min` | structural | find_index clamps to 0 below x_min | API contract |
| tests/coverage_gaps.rs | `grid_find_index_between_points` | structural | find_index returns nearest index | API contract |
| tests/coverage_gaps.rs | `grid_find_index_exact_match` | structural | find_index returns exact indices on a uniform grid | API contract |
| tests/coverage_gaps.rs | `heating_rate_per_redshift_sign_convention` | analytic | \|dq/dz\| = rate/(H(1+z)) to 1e-10; sign convention | exact chain-rule identity dt/dz = -1/[H(1+z)] |
| tests/coverage_gaps.rs | `newton_exhaustion_recorded_in_diagnostics` | structural | diag_newton_exhausted >= 1 with max_newton_iter=2; mu, y finite | diagnostic-plumbing check, no physics target |
| tests/fh_background.rs | `ac1_mu_era_z5e5` | cross-method | FH mu matches PDE mu to 5% at z_h=5e5; y subdominant (<5% of mu) | FhBackgroundSolver vs ThermalizationSolver PDE |
| tests/fh_background.rs | `ac1_transition_z5e4` | cross-method | FH mu and y each match PDE to 5% at z_h=5e4 | FhBackgroundSolver vs ThermalizationSolver PDE |
| tests/fh_background.rs | `ac1_y_era_z5e3` | cross-method | FH y matches PDE to 5% at z_h=5e3; mu subdominant | FhBackgroundSolver vs ThermalizationSolver PDE |
| tests/fh_background.rs | `dp_energy_closure` | analytic | \|eps.y\|/drho_d < 5e-2 for all z_con | conversion spectrum carries exactly zero net energy by construction (EC26 Eq. 3.2a identity) |
| tests/fh_background.rs | `dp_mu_vs_pde_lower_z` | cross-method | FH dark-photon mu matches PDE to 5% at z_con = 5e4, 5e3 | FH vs PDE cross-check |
| tests/fh_background.rs | `dp_mu_vs_pde_mu_era_z5e5` | cross-method | FH dark-photon mu matches PDE (same injected spectrum) to 5% | FH vs PDE with EC26 Eq. 3.2a conversion spectrum as IC |
| tests/fh_background.rs | `dp_thermalizes_at_z5e6` | analytic | \|mu\|, \|y\| < 1e-3 * drho after conversion at z_con=5e6 | full-thermalization limit (only T-shift survives above z_mu) |
| tests/fh_background.rs | `energy_closure_all_eras` | analytic | FH energy functional eps.y recovers injected drho to 1% | energy conservation |
| tests/fh_spectra.rs | `check_anchor_case` | literature | muT/TT and yT/TT match EC26 Fig. 9 anchors to 35%; band (1.0-4.5)e-5; yT > 0; stationary overlay b_X to 25%; muE/TE ~ muT/TT to 50% | EC26 Fig. 9 (right, stationary dashed), anchors read off the paper per plan section 4; +-30% cross-backend tolerance |
| tests/fh_spectra.rs | `s13_lmax_truncation_converged` | analytic | doubling FH l_max 15->30 changes C_l^muT by < 1% | truncation-convergence criterion (self-convergence on same k-grid) |
| tests/fh_spectra.rs | `thermalized_z_con_5e6_gives_no_mu_t` | dimensional | muT/TT < 2e-7 (100x below the mu-era anchor) at z_con=5e6 | thermalization suppression order-of-magnitude argument (threshold = anchor/100) |
| tests/greens_function_checks.rs | `brpack_gaunt_factor_spot_checks` | literature | gaunt_ff_nr matches CRB 2020 softplus formula (hand-implemented in test) to 1e-10; Z and x monotonicity | Chluba, Ravenni & Bolliet 2020 MNRAS 492, 177 published fit expression |
| tests/greens_function_checks.rs | `chluba2013_energy_conservation` | analytic | int x^3 G_th dx / G3 = 1 to 3% in pure mu/y eras; transition residual bounded < 22% | exact energy-conservation integral; visibility identity (Chluba 2013 Sect. 3; Arsenadze et al. 2025 J_y fit) |
| tests/greens_function_checks.rs | `chluba2013_limit_pure_mu` | analytic | J_mu(3e5) > 0.99, J_y(3e5) < 0.05; y/mu spectral fraction < 5% | Chluba 2013 Sect. 2.2 deep mu-era limit |
| tests/greens_function_checks.rs | `chluba2013_limit_pure_temperature_shift` | analytic | G_th(x, 5e6) = (1/4) G_bb(x) to 0.5% | Chluba 2013 Sect. 2.1 full-thermalization limit (closed form) |
| tests/greens_function_checks.rs | `chluba2013_limit_pure_y` | analytic | J_mu(2e3) < 0.01, J_y(2e3) > 0.99; y coefficient = 0.25 to 1% | Chluba 2013 Sect. 2.3 y-era limit; y = drho/4rho |
| tests/greens_function_checks.rs | `chluba2013_pde_spectral_shape_mu_era` | cross-method | PDE dn(x) matches GF dn(x) pointwise to 20% at z_h=2e5 | PDE solver vs Green's function (Chluba 2013 fits) |
| tests/greens_function_checks.rs | `chluba2013_visibility_pde_cross_validation` | cross-method | PDE mu matches GF visibility formula to 12% at z_h=2e5 | PDE (no fitting formulas) vs Chluba 2013 GF fits |
| tests/heat_injection.rs | `assert_rel` | structural | helper: generic relative-error assertion | test helper, not a test; inventory attributes its internal assert to this fn |
| tests/heat_injection.rs | `find_resonance_z` | structural | helper: asserts omega_pl bisection bracket contains resonance | test helper (bracket precondition), not a test |
| tests/heat_injection.rs | `golden_mu_era_spectral_shape` | literature | PDE mu = (3/kappa_c) J_bb* J_mu drho (~1.36e-5) to 10%; y/mu<8%; energy to 2%; M(x) sign structure | Chluba 2013 MNRAS 434, 352 Eq. 5 (docstring gives oracle+uncertainty; production grid) |
| tests/heat_injection.rs | `golden_transition_era_spectral_shape` | analytic | energy to 2%; mu,y>0; sum rule mu/1.401+4y within factor 2.5 of drho | energy conservation + basis-independent sum rule (docstring explains why mu/y not compared to GF) |
| tests/heat_injection.rs | `golden_y_era_spectral_shape` | analytic | y = drho/4 to 3%; mu/y<4%; energy to 1%; Y_SZ sign structure; step count bounds | exact y-era relation |
| tests/heat_injection.rs | `test_adaptive_dz_bounds` | structural | dz_min <= dz <= 0.05z | solver config contract |
| tests/heat_injection.rs | `test_adiabatic_cooling_no_injection` | literature | mu_ac = -2.9e-9 to 25%; y < 0.3\|mu\|; drho = -3.1e-9 to 25% | Chluba & Sunyaev 2012 MNRAS 419, 1294; Chluba 2016 Fig. 1 (docstring gives oracle+uncertainty) |
| tests/heat_injection.rs | `test_alpha_rho_from_integrals` | analytic | ALPHA_RHO = 2 zeta(3)/(pi^4/15) ~ 0.3702 to 1e-14 | G2/G3 exact values |
| tests/heat_injection.rs | `test_annihilation_mu_y_properties` | dimensional | mu,y>0; mu/y>0.5 s-wave; p-wave mu/y > s-wave mu/y | sign/ordering from redshift weighting |
| tests/heat_injection.rs | `test_annihilation_redshift_scaling` | analytic | s-wave rate ~ (1+z)^2, p-wave ~ (1+z)^3 to 0.1% | exact scaling of annihilation rate |
| tests/heat_injection.rs | `test_annihilation_swave_low_x_spectral_shape` | cross-method | PDE/GF spectral ratio at x=0.3-0.8 = 1 +/- 30% | PDE solver vs GF convolution |
| tests/heat_injection.rs | `test_beta_mu_from_zeta_functions` | analytic | BETA_MU = 3zeta(3)/zeta(2) ~ 2.19229 to 1e-4 | zeta-function identity; Abramowitz & Stegun Table 23.3, OEIS A002117 |
| tests/heat_injection.rs | `test_bose_factor_taylor_vs_exact` | analytic | Taylor error < 2 x drho^2 x(x+2)/2 + 1e-10; direction of Bose-factor shift | second-order Taylor remainder derived from f''(1)=x(x+2)exp(x) |
| tests/heat_injection.rs | `test_br_absolute_value_z1e6_x1` | dimensional | K_BR(z=1e6, x=1) in (1e-10, 1e-6), i.e. O(1e-8) | order-of-magnitude from BR formula; Rybicki & Lightman 1979 Eq. 5.14b, Chluba & Sunyaev 2012 Eq. 14; guards /n_e bug (~1e-19 failure mode) |
| tests/heat_injection.rs | `test_br_coefficient_saha_transition` | structural | K_BR finite and nonnegative across He recombination z | finiteness only |
| tests/heat_injection.rs | `test_br_heating_integral_planck_zero` | analytic | BR heating integral ~ 0 (<1e-10) for Planck at T_e=T_z | detailed balance |
| tests/heat_injection.rs | `test_br_temperature_scaling` | analytic | K_BR ratio = (theta2/theta1)^3.5 x Gaunt ratio to 5% | theta_e^{-7/2} scaling from BR emission formula |
| tests/heat_injection.rs | `test_brightness_temperature` | analytic | T_b deviation = 0 for Planck (<1e-10); sign change near beta_mu for mu-distortion | brightness-temperature definition + mu zero crossing |
| tests/heat_injection.rs | `test_compton_equilibrium_mu_distortion` | dimensional | rho_e>1 for mu>0 and rho_e - 1 < 0.01 | sign of harder spectrum, O(mu) magnitude |
| tests/heat_injection.rs | `test_compton_equilibrium_mu_distortion_deviation` | analytic | rho_e - 1 in (1e-5, 1e-3) for BE(mu=1e-4), i.e. O(mu) x factor-10 band | perturbative expansion of I4/(4G3) for BE distribution |
| tests/heat_injection.rs | `test_compton_equilibrium_planck_exact` | analytic | I4/(4G3)=1 for Planck spectrum to 1e-4 | integration-by-parts identity |
| tests/heat_injection.rs | `test_compton_y_parameter_convergence` | structural | 128-pt vs 1024-pt quadrature to 1%; y_C(1e5)>0.1, monotone | self-convergence plus order-of-magnitude sanity |
| tests/heat_injection.rs | `test_compton_y_parameter_post_recombination` | dimensional | y_C(500)<0.05, y_C(100)<0.01 | post-recombination X_e ~ 1e-4 suppression |
| tests/heat_injection.rs | `test_cosmic_time_milestones` | literature | t0 in 12-15 Gyr; t(1100) in 200-500 kyr; t(1e6) 3-30 months | standard LCDM age ~13.8 Gyr, recombination ~380 kyr |
| tests/heat_injection.rs | `test_cosmology_presets` | literature | h, omega_b, omega_cdm, Y_p, T_cmb, N_eff match Planck 2015/2018 | Planck 2015/2018 parameter papers |
| tests/heat_injection.rs | `test_cosmology_self_consistency` | analytic | z_eq self-consistency to 1e-10 (and in [3000,4000]); Omega closure; E(z) asymptotics; t_C ~ (1+z)^-3 | Friedmann-equation identities from code's own parameters |
| tests/heat_injection.rs | `test_coupled_vs_split_z1e6` | cross-method | coupled IMEX vs operator-split mu ratio in [0.9,1.1]; both conserve energy to 15% | two integration schemes of same equations |
| tests/heat_injection.rs | `test_custom_injection_captures_data` | structural | Custom closure with captured Vec evaluates polynomial to 1e-10 | API contract |
| tests/heat_injection.rs | `test_custom_injection_closure` | structural | Custom closure returns specified rate | API contract |
| tests/heat_injection.rs | `test_dark_photon_conservation_sum_rule` | analytic | E_mu+E_y+E_temp = drho to 30%; algebraic identity to 1e-12 | GF energy branching sum rule |
| tests/heat_injection.rs | `test_dark_photon_mass_dependent_distortion_type` | dimensional | z_res regimes per mass; y vs mu dominance; efficiency suppression ordering | NWA + visibility regime structure |
| tests/heat_injection.rs | `test_dark_photon_nwa_gf_prediction` | analytic | z_res in [1e5,1e6]; mu>y; eps^2 scaling exact to 1% | NWA formula; Mirizzi, Redondo & Sigl 2009; Arias et al. 2012 |
| tests/heat_injection.rs | `test_dc_backward_euler_accuracy` | analytic | 1/(1+a) >= exp(-a) for all a>=0 (BE under-thermalizes) | exact inequality |
| tests/heat_injection.rs | `test_dc_br_ratio_pinned_z1e6` | regression-pin | DC/BR at z=1e6, x=1 in (8, 50) | band around observed ~15-20 (BRpack Gaunt), with first-principles order-of-magnitude backing; explicitly a pinned regression guard for the /n_e bug |
| tests/heat_injection.rs | `test_dc_heating_integral_planck_zero` | analytic | DC heating integral ~ 0 (<1e-10) for Planck at T_e=T_z | detailed balance |
| tests/heat_injection.rs | `test_dc_high_freq_suppression_decay` | analytic | H_dc(0)=1 to 1e-14; monotone decreasing | exact normalization + exp(-2x) decay |
| tests/heat_injection.rs | `test_dc_suppression_extreme_x` | structural | H_dc(100)<1e-50, H_dc(50)<1e-20 finite, H_dc(1000)=0 (cutoff) | implementation cutoff behavior |
| tests/heat_injection.rs | `test_dc_suppression_monotonicity` | dimensional | H_dc(0)=1; nonnegative; decreasing for x>2 | exact normalization + monotone decay |
| tests/heat_injection.rs | `test_dc_temperature_scaling` | analytic | K_DC ratio = (theta2/theta1)^2 to 1% | theta_z^2 scaling of DC emissivity |
| tests/heat_injection.rs | `test_dcbr_absorption_coefficients_at_soft_x` | dimensional | K_BR, K_DC > 0; absorption rate > 1e-4; DC/BR in (0.01,100) at z=2e5 | order-of-magnitude absorption efficiency at x=0.01 |
| tests/heat_injection.rs | `test_dcbr_affects_thermalization_depth` | dimensional | mu without DC/BR > mu with DC/BR at z_h=5e5 | thermalization direction |
| tests/heat_injection.rs | `test_dcbr_dimensional_scaling_vs_z` | dimensional | DC/BR ratio increases with z; <5 at z=1e4; positive finite | first-principles dimensional scaling (K_BR ~ n_ion lambda_e^3, K_DC ~ theta_z^2); guards historical /n_e bug |
| tests/heat_injection.rs | `test_dcbr_thermalizes_mu_distortion` | dimensional | mu decreases from z=2e5 to 5e3; mu/drho O(1) at z=2e5 | DC/BR thermalization direction |
| tests/heat_injection.rs | `test_decay_rate_non_monotonic` | dimensional | decay heating rate has interior maximum (rate(z>1e4) > rate(1e4)) | competing n_H/rho_g vs exp(-Gamma t) factors |
| tests/heat_injection.rs | `test_decaying_particle_extreme_lifetime` | dimensional | rate nonnegative, finite, nonzero for tau >> t_universe | exp(-Gamma t) ~ 1 limit |
| tests/heat_injection.rs | `test_decaying_particle_pde_vs_gf` | cross-method | PDE vs GF mu sign agreement; drho>0 | PDE solver vs Green's function |
| tests/heat_injection.rs | `test_decaying_particle_photon_hard_pde` | dimensional | drho>0; mu/drho in (-0.1, 1.6) physical range | physical bounds on mu/drho |
| tests/heat_injection.rs | `test_decaying_particle_photon_soft_pde` | literature | mu/drho in [1.15, 1.42] (decay-weighted 1.401 x visibility) | Chluba 2013 Eq. 5 decay-weighted; docstring gives oracle+uncertainty |
| tests/heat_injection.rs | `test_decaying_particle_photon_vacuum` | structural | photon_source_rate nonzero at x_inj, heating_rate zero, 1 refinement zone | scenario routing contract |
| tests/heat_injection.rs | `test_decaying_particle_time_dependence` | analytic | rate/(n_H/rho_g) follows exp(-Gamma dt) to 5%; rate >= 0 | exact exponential decay kinematics |
| tests/heat_injection.rs | `test_decomposition_comprehensive` | analytic | round-trip recovery of dT (10%), negative mu (15%), mixed mu+y (15%/30%) | synthetic input decomposition |
| tests/heat_injection.rs | `test_density_scaling_relations` | analytic | n_H ~ (1+z)^3, rho_g ~ (1+z)^4, n_He/n_H = Y_p/(4(1-Y_p)) to 1e-10 | exact scaling relations |
| tests/heat_injection.rs | `test_diag_warnings_collected` | structural | warnings populated and cleared by reset() | diagnostics plumbing |
| tests/heat_injection.rs | `test_dm_baryon_scattering_pde_vs_gf` | cross-method | PDE vs GF mu to 35%; \|mu\| in (5e-7, 2e-6) per AH21; mu<0; y subdominant | PDE vs GF; amplitude anchor Ali-Haimoud 2021 arXiv:2101.04070 Fig. 1 (\|mu\|~1e-6) |
| tests/heat_injection.rs | `test_extreme_large_injection` | analytic | mu_PDE/mu_linear(Chluba 2013) in [0.9,1.3] at drho=0.01; energy to 10%; Newton clean | Chluba 2013 Eq. 5 linear prediction; nonlinear band bounded |
| tests/heat_injection.rs | `test_extreme_small_injection` | analytic | mu/drho identical to 10% between drho=1e-5 and 1e-7 | linearity |
| tests/heat_injection.rs | `test_firas_check_and_energy_consistency` | analytic | FIRAS mu fraction ~0.5 at half-limit; drho = mu*kappa_c/3 to 15% | construction (mu=half FIRAS limit) + kappa_c energy relation |
| tests/heat_injection.rs | `test_firas_limits_consistency` | literature | FIRAS_MU_LIMIT=9e-5, FIRAS_Y_LIMIT=1.5e-5; LCDM mu << limit | Fixsen et al. 1996, ApJ 473, 576 (95% CL) |
| tests/heat_injection.rs | `test_full_te_perturbative_vs_brute_force` | analytic | perturbative rho_e (1 + dI4/(4G3) - dG3/G3) agrees with brute-force I4/(4G3) to 10% for mu=1e-4; both > 1 | first-order perturbative expansion vs exact ratio (O(dn^2) terms explain the 10% tolerance, documented) |
| tests/heat_injection.rs | `test_full_te_rho_e_for_mu_distortion` | analytic | compton_equilibrium_ratio matches independent I4/(4G3) quadrature to 1e-6; rho_e>1 for mu>0; =1 for Planck | independent numerical integration of same definition |
| tests/heat_injection.rs | `test_gaunt_ff_cross_validation` | analytic | g_ff in 1-10 range; low-x > high-x; Z ordering; classical limit matches CRB20 softplus formula to 1% | Chluba, Ravenni & Bolliet 2020 formula asymptotics (Karzas & Latter 1961 mentioned but not tabulated values) |
| tests/heat_injection.rs | `test_gaunt_ff_limiting_behavior` | dimensional | g_ff(x=50)~1 within 0.5; g_ff low-x > high-x | softplus asymptotics of Born-approximation Gaunt factor |
| tests/heat_injection.rs | `test_gaunt_ff_z_dependence` | dimensional | g_ff(Z=2) < g_ff(Z=1), both positive | ln(2.25/(xZ)) argument decreases with Z |
| tests/heat_injection.rs | `test_gf_energy_sum_rule` | literature | J_mu J_bb* + J_y + (1-J_bb*) = 1 to 20%, max deviation in transition region | Chluba 2013 Sect. 3: residual bounded ~16-17% near z~7-8e4 |
| tests/heat_injection.rs | `test_gf_mu_resolution_independence` | structural | mu converges with GF integration resolution (500/2000/5000 pts) | self-convergence, no physics target |
| tests/heat_injection.rs | `test_gf_photon_injection_post_recombination_locked_in` | dimensional | P_s(3,500)>0.5; smooth GF part <1e-3 far from x_inj; delta survives | locked-in physics; comment documents x_c fit extrapolation artifact at z<1e4 |
| tests/heat_injection.rs | `test_greens_function_asymptotic_limits` | analytic | G_th -> Y(x)/4 at z=1e3 within 50%; mu-shape sign structure at z=2e5 | GF asymptotic limits |
| tests/heat_injection.rs | `test_greens_function_decomposition_accuracy` | cross-method | decomposed mu/y of G_th matches visibility prediction to 5% (dominant component) | least-squares decomposition vs GF visibility construction |
| tests/heat_injection.rs | `test_greens_function_energy_accounting` | analytic | int G_th x^3 dx / G3 = 1 to 3% in pure regimes; <22% in transition | total energy conservation; Chluba 2013 Eq. 5 |
| tests/heat_injection.rs | `test_greens_function_linearity` | analytic | doubling amplitude doubles mu,y,spectrum to 1e-10/1e-8 | linearity of Green's function by construction |
| tests/heat_injection.rs | `test_greens_function_smooth_transition` | dimensional | G_th finite, continuous (<20% jump per 1% z-step), positive at x=5 | continuity/sign arguments |
| tests/heat_injection.rs | `test_grid_convergence_rate` | structural | Richardson ratio in (1.5,10) suggesting ~2nd order; 1000 vs 2000 pt mu to 10% | self-convergence |
| tests/heat_injection.rs | `test_grid_extreme_configurations` | structural | log grid dx/x constant to 1%; uniform grid dx constant | grid construction properties |
| tests/heat_injection.rs | `test_grid_find_index_boundaries` | structural | find_index boundary/exact-point behavior | API contract |
| tests/heat_injection.rs | `test_grid_transition_artifact` | structural | mu converges <5% between 2000 and 4000 pts; energy <2% both | self-convergence |
| tests/heat_injection.rs | `test_he_ionization_saha_regimes` | analytic | He2+ fraction >0.95 at z=1e4, He+ dominant at z=4000, bounds/monotonicity | Saha equilibrium regime limits (He ionization potentials) |
| tests/heat_injection.rs | `test_heat_custom_matches_single_burst` | structural | Custom closure replicating SingleBurst matches builtin to 1% | injection infrastructure identity (same physics code) |
| tests/heat_injection.rs | `test_heat_decay_lifetime_controls_mu_y` | dimensional | early decay \|mu/y\| > late decay; signs positive | decay epoch controls mu vs y weighting |
| tests/heat_injection.rs | `test_heat_decay_total_energy_deposited` | cross-method | drho>0; PDE y vs GF y to 16% | PDE vs GF for fast-decay scenario |
| tests/heat_injection.rs | `test_heat_dm_annihilation_energy_conservation` | analytic | PDE drho = time-integrated heating rate to 20% | energy conservation (independent quadrature of heating rate) |
| tests/heat_injection.rs | `test_heat_dm_annihilation_pde_vs_gf` | cross-method | PDE vs GF mu to 12% (mu-era-clipped injection window) | PDE vs GF with matched integration bounds (docstring documents oracle) |
| tests/heat_injection.rs | `test_heat_dm_fann_linear_scaling` | analytic | mu,y double with f_ann to 15% | linearity (tolerance widened for adiabatic-cooling offset, explained in comment) |
| tests/heat_injection.rs | `test_heat_energy_conservation_sweep_tight` | analytic | measured drho matches injected to 2% at 7 redshifts | energy conservation |
| tests/heat_injection.rs | `test_heat_heating_cooling_cancellation` | analytic | +/-drho bursts cancel: max\|dn diff from no-injection run\| < 1e-12 | exact algebraic cancellation of source; differenced against no-injection run |
| tests/heat_injection.rs | `test_heat_mu_first_principles_ratio` | analytic | PDE mu = 1.401 drho J_bb* J_mu to 12% at z=2e5 | 3/kappa_c coefficient + Chluba 2013 visibility |
| tests/heat_injection.rs | `test_heat_pde_amplitude_linearity` | analytic | mu,y ratios = 2.0 +/- 2% | PDE linearity |
| tests/heat_injection.rs | `test_heat_pde_vs_gf_multi_z_sweep` | cross-method | PDE vs 1.401 J_bb* J_mu GF mu to 20-40% (z-dependent); y to 5% at z=5000 | PDE B&F decomposition vs Chluba visibility convolution |
| tests/heat_injection.rs | `test_heat_spectral_decomposition_residual_sweep` | dimensional | 3-component reconstruction rel RMS < 20% at 3 redshifts | decomposition completeness |
| tests/heat_injection.rs | `test_heat_spectral_shape_mu_era` | analytic | dn shape matches M(x) with rel RMS < 10% in x=[1,15] | pure mu-shape in deep mu-era |
| tests/heat_injection.rs | `test_heat_spectral_shape_y_era` | analytic | dn shape matches Y_SZ(x) with rel RMS < 10% | pure y-shape in y-era |
| tests/heat_injection.rs | `test_heat_superposition_two_bursts` | analytic | combined mu = sum of individual to 3% (y 30%) | PDE linearity |
| tests/heat_injection.rs | `test_heat_swave_vs_pwave_mu_y_ratio` | dimensional | p-wave \|mu/y\| > s-wave; all mu,y positive | extra (1+z) weighting of p-wave |
| tests/heat_injection.rs | `test_heat_thermalization_suppression_net_decrease` | dimensional | mu/drho decreases from z=2e5 to 5e5; >1.0 at z=2e5 | J_bb* monotone suppression |
| tests/heat_injection.rs | `test_heat_transition_region_mixed_distortion` | analytic | mu,y>0; \|mu/y\| in (1e-4,100); energy to 2% at z=1e4 | energy conservation + mixed-mode positivity |
| tests/heat_injection.rs | `test_heat_y_era_pure_y_parameter` | analytic | y = drho/4 to 5%; \|mu/y\|<0.1 at z=5000 | exact y-parameter definition |
| tests/heat_injection.rs | `test_heating_rate_sign_convention` | dimensional | heating_rate>0, heating_rate_per_redshift<0 | dz/dt < 0 sign convention |
| tests/heat_injection.rs | `test_helium_electron_fraction_and_transition_continuity` | analytic | He e- fraction in [0, 2f_He], correct limits; Saha->Peebles continuity <5%/z-step | conservation bounds + continuity |
| tests/heat_injection.rs | `test_high_z_dtau_convergence` | cross-method | dtau_max=3 PDE mu vs GF mu to 15% at z_h=5e5 | PDE vs Green's function |
| tests/heat_injection.rs | `test_initial_perturbation_evolution` | dimensional | mu>0 for injection at x_i=5 > x0~3.6; nonzero distortion | sign prediction from Chluba 2015 photon-injection theory |
| tests/heat_injection.rs | `test_intermediate_photon_injection_x01` | analytic | drho matches injected to 5% at x_inj=0.1 | energy conservation |
| tests/heat_injection.rs | `test_kappa_c_from_numerical_integration` | analytic | KAPPA_C = 3*int x^3 M(x) dx / G3 matches hardcoded 2.1419 to 0.5% | energy conservation for pure mu-distortion, independent quadrature |
| tests/heat_injection.rs | `test_kompaneets_free_streaming` | analytic | energy conserved to 1% (y IC); mu preserved to 15% (mu IC) without DC/BR | conservation + Kompaneets equilibrium |
| tests/heat_injection.rs | `test_kompaneets_large_dtau_stability` | analytic | no NaN at dtau=100; energy conserved to 10% | implicit-scheme stability + energy conservation at T_e=T_z |
| tests/heat_injection.rs | `test_kompaneets_large_perturbation_stability` | analytic | finite mu,y; drho within factor 2 of injected 1e-3 | energy conservation (loose) + stability |
| tests/heat_injection.rs | `test_kompaneets_photon_number_hybrid_grid` | analytic | photon number conserved to 1% under pure Kompaneets | Kompaneets conserves photon number exactly |
| tests/heat_injection.rs | `test_kompaneets_preserves_bose_einstein` | analytic | BE spectrum preserved under pure Kompaneets to 5% | BE is stationary solution of Kompaneets equation |
| tests/heat_injection.rs | `test_lambda_expansion_small_at_high_z` | dimensional | Lambda < 1e-6 and > 0 at z=1e6 | strong Compton coupling estimate |
| tests/heat_injection.rs | `test_literature_mu_y_conversion_coefficients` | analytic | mu = 1.401 drho (10%), y = 0.25 drho (10%), dominance ordering | 1.401=3/kappa_c and 1/4 exact; SZ 1970, Hu & Silk 1993, Chluba 2013 |
| tests/heat_injection.rs | `test_literature_regime_boundaries` | dimensional | mode ordering at z=3e6/2e5/1e4; suppression <10%/1% of drho at z=3e6 | three-regime structure, SZ 1970 |
| tests/heat_injection.rs | `test_load_heating_table_roundtrip` | structural | CSV load roundtrip, sorted, interpolates, zero outside | I/O plumbing |
| tests/heat_injection.rs | `test_load_photon_source_table_roundtrip` | structural | 2D CSV load, interpolation inside, zero outside bounds | I/O plumbing |
| tests/heat_injection.rs | `test_mu_decay_eigenvalue` | dimensional | Kompaneets-only preserves mu (>0.85); DC/BR decays mu below that | mu is Kompaneets equilibrium; DC/BR thermalization direction |
| tests/heat_injection.rs | `test_mu_efficiency_deep_mu_era` | analytic | mu = (3/kappa_c) J_bb* J_mu drho to 5% at z=5e5; y << mu | Kompaneets + number conservation; SZ 1970, Hu & Silk 1993, Chluba 2013 (visibility factors evaluated via code's Chluba-2013 fits) |
| tests/heat_injection.rs | `test_mu_shape_zero_crossing_and_sign_structure` | analytic | M(beta_mu)=0 exactly; sign structure around beta_mu | definition M(x)=(x/beta_mu - 1) g_bb/x |
| tests/heat_injection.rs | `test_mu_y_shapes_independent` | dimensional | M(x)/Y(x) ratio varies by >0.1 across x | linear independence of basis shapes |
| tests/heat_injection.rs | `test_mu_y_transition_redshift` | literature | mu-y energy-weight crossing in [2e4, 1e5] | z_muy ~ 5e4 consensus: Hu & Silk 1993; Chluba & Sunyaev 2012; Khatri & Sunyaev 2012 |
| tests/heat_injection.rs | `test_nc_energy_y_era_and_high_z_mu` | analytic | NC energy conservation to 10%; y-era unchanged to 1%; NC error vs 1.401 not worse | energy conservation + mu=1.401 drho analytic target |
| tests/heat_injection.rs | `test_nc_planck_stable_and_photon_number` | analytic | NC no-injection: max\|dn\|<1e-6; with injection dN/N<1e-3 | null test + number-conservation by construction of NC mode |
| tests/heat_injection.rs | `test_nc_stripping_integral_zero` | analytic | dN/N < 1e-5 after NC run; mu preserved | number conservation enforced by NC construction |
| tests/heat_injection.rs | `test_negative_injection_cooling` | analytic | drho<0 and equals injected -1e-5 to 10% | energy conservation, sign |
| tests/heat_injection.rs | `test_negative_occupation_guard` | dimensional | min(n_pl + dn) > -1e-3 under gc=10 depletion; finite | physical n >= 0 constraint with numerical-undershoot allowance |
| tests/heat_injection.rs | `test_newton_convergence_indirect` | dimensional | mu,y>0; dn finite; rho_e in [0.5,1.01] | sanity bounds as Newton-convergence proxy |
| tests/heat_injection.rs | `test_output_format_parsing` | structural | OutputFormat::from_str parsing | plumbing |
| tests/heat_injection.rs | `test_pb2009_bose_einstein_temperature` | literature | BE decomposition recovers mu_0 to 10%; phi_BE = (1-1.11 mu_0)^{-1/4} slightly > 1 | Procopio & Burigana 2009 A&A 507, 1243 |
| tests/heat_injection.rs | `test_pb2009_energy_conservation` | analytic | energy conservation: 1.5% default grid, 0.5% production grid | exact target drho_out=drho_in; docstring documents why P&B's 0.05% is not the applicable oracle |
| tests/heat_injection.rs | `test_pb2009_grid_coverage` | structural | production grid covers x in [10^-4.3, 10^1.7] | P&B 2009 grid range (configuration check, no physics assertion) |
| tests/heat_injection.rs | `test_pde_electron_temperature_feedback` | dimensional | \|mu\|>1e-10; rho_e in (0.99,1.01) | nonzero response, near-equilibrium T_e |
| tests/heat_injection.rs | `test_pde_linearity_double_injection` | analytic | mu ratio 2.0 +/- 2.5%, drho ratio 2.0 +/- 1% | linearity of Boltzmann equation for small dn |
| tests/heat_injection.rs | `test_pde_negative_injection_produces_negative_distortion` | dimensional | mu<0, y<0, drho<0 for cooling burst | sign of distortion under cooling |
| tests/heat_injection.rs | `test_pde_no_injection_full_range` | analytic | max\|dn(x>0.1)\| < 5e-5 over z=[3e6,200] with no injection | null test; bound = 2.5x adiabatic-cooling estimate (tightened per CLAUDE.md Pitfall #9) |
| tests/heat_injection.rs | `test_pde_photon_depletion_post_recombination` | dimensional | depletion dip at x_inj, negative, concentrated | locked-in depletion |
| tests/heat_injection.rs | `test_pde_photon_injection_post_recombination` | dimensional | spectral peak stays within 3 sigma of x_inj; far-field <10% of peak | locked-in distortion at z<1100 (Compton inefficient) |
| tests/heat_injection.rs | `test_pde_planck_is_stable_equilibrium` | analytic | no injection: max\|dn(x>0.1)\|<5e-5, \|drho\|<1e-7 | Planck is equilibrium; bound = 2.5x physical adiabatic-cooling estimate |
| tests/heat_injection.rs | `test_pde_vs_gf_photon_injection_balanced` | analytic | \|mu_PDE(x0)\| < 5% of analytic mu_max(x=10) | mu(x_balanced)=0 by construction of Chluba 2015 coefficient |
| tests/heat_injection.rs | `test_pde_vs_gf_photon_injection_high_x` | literature | PDE mu at x_inj=10 matches Chluba 2015 Eq. 30 (~3.25e-5) to 10% | Chluba 2015 MNRAS 454, 4182 Eq. 30 (docstring gives oracle+uncertainty) |
| tests/heat_injection.rs | `test_pde_vs_gf_photon_injection_low_x` | literature | PDE mu at x_inj=2 negative, matches Chluba 2015 Eq. 30 (~-8.1e-6) to 15% | Chluba 2015 Eq. 30 with P_s -> 1 |
| tests/heat_injection.rs | `test_pde_vs_gf_photon_injection_y_era` | dimensional | J_mu<0.05 at z=5e3; \|y\|>1e-10; GF mu(y-era) << mu(mu-era) | regime structure |
| tests/heat_injection.rs | `test_pde_y_to_mu_conversion` | analytic | mu>0, drho>0, energy conservation to 15% | energy conservation + heating sign |
| tests/heat_injection.rs | `test_perturbative_te_small_mu_distortion` | analytic | rho_eq > 1 both methods; perturbative delta-rho O(eps) in (1e-8,1e-4) | perturbative dI4/(4G3) - dG3/G3 formula vs exact ratio |
| tests/heat_injection.rs | `test_photon_depletion_signs_and_magnitude` | analytic | drho<0; mu>0; mu = (3/kappa_c)(P/3) J_bb* J_mu to 15% for uniform depletion | entropy-corrected mu formula; Chluba & Cyr 2024 |
| tests/heat_injection.rs | `test_photon_gf_balanced_injection_zero_mu` | analytic | \|mu(x0)\| < 1% of mu(x=10); P_s(x0)>0.99 | exact cancellation at x0=4G3/(3G2) |
| tests/heat_injection.rs | `test_photon_gf_energy_only_limit` | dimensional | P_s<0.01 for x_inj=1e-5; photon GF same sign as alpha_rho x G_th | P_s->0 limit (approximate under Arsenadze T_mu) |
| tests/heat_injection.rs | `test_photon_gf_mu_linearity` | analytic | mu linear in dN/N to 1e-12 | linearity by construction |
| tests/heat_injection.rs | `test_photon_gf_sign_flip_negative_mu` | analytic | mu(2)<0, mu(10)>0; ratio matches (1-P_s x0/x) formula to 1% | Chluba 2015 photon-injection mu formula |
| tests/heat_injection.rs | `test_photon_gf_soft_photon_absorbed` | dimensional | mu>0 for absorbed soft photons (P_s~0) | absorbed photons act as pure energy injection |
| tests/heat_injection.rs | `test_photon_injection_analytic_match` | analytic | PDE mu matches (3/kappa_c)(x_i G2/G3 - 4/3) dN to 20% | Chluba 2015 MNRAS 454, 4182 deep-mu-era formula |
| tests/heat_injection.rs | `test_photon_injection_energy_conservation_tight` | analytic | drho = alpha_rho x_inj dN/N to 3% for 5 x_inj values | exact energy bookkeeping of injected photons |
| tests/heat_injection.rs | `test_photon_injection_energy_number_decomposition` | analytic | PDE mu matches (3/kappa_c) alpha_rho (x - x0 P_s) dN J_bb* J_mu to 15%, signs match | Chluba 2015 energy-number imbalance identity |
| tests/heat_injection.rs | `test_photon_injection_extreme_frequencies` | dimensional | mu(x=20)>0, mu(x=0.5)<0; \|ratio\| within factor ~3 of (x_h-x0)/(x0-x_l) | sign and approximate linear-in-x scaling |
| tests/heat_injection.rs | `test_photon_injection_gf_algebraic_identities` | analytic | P_s->0 sign identity; linearity to 1e-12; \|mu(x0)\| < 2% of mu(10) | exact algebraic identities of photon GF |
| tests/heat_injection.rs | `test_photon_injection_grid_convergence` | structural | Richardson: \|mu(2000)-mu(1000)\| < 1.5x\|mu(1000)-mu(500)\|; 5% agreement | self-convergence |
| tests/heat_injection.rs | `test_photon_injection_kompaneets_redistribution_y_era` | analytic | J_mu<0.01 at z=3000; y measurable; drho = alpha_rho x dN to 5% | energy conservation in y-era |
| tests/heat_injection.rs | `test_photon_injection_mu_y_systematics` | analytic | mu monotone in x_inj; zero crossing at x0 +/- 0.5 | Chluba 2015 sign-flip location x0=3.602 |
| tests/heat_injection.rs | `test_photon_injection_negative_mu_chluba2015` | analytic | mu<0 at x_inj=2, \|mu(x0)\|<15% scale, sign flip; 20% vs Eq. 30 | Chluba 2015 arXiv:1506.06582 Eqs. 30-31 |
| tests/heat_injection.rs | `test_photon_injection_number_conservation_pure_kompaneets` | analytic | dN/N conserved to 1% at z=2000 (DC/BR negligible) | Kompaneets photon-number conservation |
| tests/heat_injection.rs | `test_photon_injection_pde_vs_gf_tight_mu_era` | cross-method | PDE vs GF mu to 10% at 3 x_inj values, signs match | PDE solver vs GF photon-injection formula |
| tests/heat_injection.rs | `test_photon_injection_scenario_vs_initial_condition` | cross-method | scenario-source vs initial-condition mu agree to 30%, same sign | two injection code paths of same solver |
| tests/heat_injection.rs | `test_photon_injection_spectral_decomposition_residual` ⚠ | dimensional | 3-component fit RMS residual < 12% of peak | spectral completeness of mu+y+dT basis |
| tests/heat_injection.rs | `test_photon_injection_spectral_shape_match_mu_era` | cross-method | normalized PDE vs GF spectral shape RMS < 0.10 | PDE vs GF full spectral shape |
| tests/heat_injection.rs | `test_photon_injection_superposition` | analytic | mu and spectrum of A+B = A plus B to 3% | PDE linearity |
| tests/heat_injection.rs | `test_photon_survival_post_recombination` | regression-pin | x_c(500)<1 and P_s(3,500)>0.5 | code's x_c fitting-formula extrapolation behavior (physical x_c ~ 0 post-recombination) |
| tests/heat_injection.rs | `test_photon_survival_regime_structure` | dimensional | x_c_DC dominates at z=2e6, x_c_BR at z=1e4; x_c non-monotone | opposite z-scalings of DC and BR absorption |
| tests/heat_injection.rs | `test_planck_integral_accuracy` | analytic | numerical G3, G2, I4 match exact constants to 1e-4 | zeta-function integrals |
| tests/heat_injection.rs | `test_plasma_frequency_formula` | analytic | omega_pl API vs first-principles n_e to 1e-10; (1+z)^3 scaling to 1% | plasma frequency formula, self-consistent recomputation |
| tests/heat_injection.rs | `test_post_recombination_locked_in_distortion` | dimensional | photon drho < 10% of injected; mu,y < 1% of drho at z_h=800 | X_e ~ 1e-4 decoupling argument |
| tests/heat_injection.rs | `test_pure_y_analytical_convergence` | analytic | y = drho/4 to 1%; \|mu/y\|<5%; energy to 1% at z_h=5000 | exact y-era relation |
| tests/heat_injection.rs | `test_recombination_cache_properties` | cross-method | cached X_e vs direct computation to 1-2% | same implementation, cache vs direct path (consistency, not independent physics) |
| tests/heat_injection.rs | `test_recombination_ionization_history` | literature | X_e(1e4)~1+f_He, X_e(1100)~0.16, X_e(800)~1e-3, X_e(200)~2-4e-4; monotone | RECFAST; Seager, Sasselov & Scott 2000 ApJ 523, 1 |
| tests/heat_injection.rs | `test_recombination_physical_values` | literature | X_e(3000) in (1,1.2); X_e(1100)~0.14; X_e(800)~3e-3; X_e(200)~3e-4; monotone | RECFAST; Seager, Sasselov & Scott 1999; Peebles 1968 |
| tests/heat_injection.rs | `test_recombination_quantitative_milestones` | literature | X_e at z=1e4/1400/1100/800/200 within RECFAST bands | Seager, Sasselov & Scott 2000, ApJ 523, 1 |
| tests/heat_injection.rs | `test_recombination_saha_peebles_physics` | analytic | Saha limits (X_e->1 high T, ->0 low T), Peebles>Saha freeze-out, He stages | Saha equation regime limits |
| tests/heat_injection.rs | `test_refinement_grid_properties` | structural | refined grid monotone, no duplicates, n>2000 | grid construction |
| tests/heat_injection.rs | `test_relativistic_correction_magnitude` | analytic | 1+2.5 theta_e correction <0.2%, DC (1+14.16 theta)^-1 correction <1% at z=1e6 | relativistic correction formulas evaluated at theta_z(1e6) |
| tests/heat_injection.rs | `test_single_burst_energy_normalization` | analytic | integral of dQ/dz equals drho to 1% | Gaussian normalization |
| tests/heat_injection.rs | `test_snapshot_close_spacing` | structural | snapshots land at requested z to 1%, monotone | plumbing |
| tests/heat_injection.rs | `test_soft_photon_equivalence_multi_z` | analytic | drho matches alpha_rho x_inj dN to 40% at 4 redshifts | energy conservation through DC/BR pre-absorption path |
| tests/heat_injection.rs | `test_solver_config_validation` | structural | invalid configs rejected | validation plumbing |
| tests/heat_injection.rs | `test_solver_respects_cosmology_parameters` | dimensional | different cosmologies give different mu (>0.1%), same positive sign | parameter sensitivity |
| tests/heat_injection.rs | `test_solver_snapshot_consistency` | structural | snapshot count/order; nonzero distortion; rho_e near 1 | plumbing checks |
| tests/heat_injection.rs | `test_spectral_decomposition_mixed_mode` | analytic | decompose recovers input mu to 2%, y to 5% | round-trip of synthetic mu*M + y*Y spectrum |
| tests/heat_injection.rs | `test_spectral_integrals_exact_values` | analytic | G1=pi^2/6, G3=pi^4/15, I4=4G3 to 1e-14; numerical G3 to 1e-7 | Riemann zeta integral representation; A&S Ch. 23 |
| tests/heat_injection.rs | `test_spectral_shape_after_burst` | analytic | corr(dn,M)>0.95; fit residual <5%; mu/drho in [0.8,1.5] at z_h=1e6 | mu-shape dominance; range from 1.401 x J_bb*(1e6) J_mu ~ 1.23 |
| tests/heat_injection.rs | `test_spectral_shape_near_orthogonality` | dimensional | \|r(M,Y)\|<0.3; \|r(M,G)\|,\|r(Y,G)\|<0.95 | near-orthogonality of basis shapes under x^2 dx measure |
| tests/heat_injection.rs | `test_strong_depletion_scaling` | dimensional | mu(gc=1)/mu(gc=0.01) in (20, 105) - sublinear vs linear 100 | depletion saturation 1-exp(-gc/x) |
| tests/heat_injection.rs | `test_tabulated_heating_matches_single_burst` | cross-method | tabulated interpolation vs closed-form SingleBurst mu,y to 2% | two injection code paths of same solver (interpolation consistency) |
| tests/heat_injection.rs | `test_tabulated_heating_zero_outside_bounds` | structural | rate zero outside table, positive inside | API contract |
| tests/heat_injection.rs | `test_tabulated_photon_source_interpolation` | structural | bilinear interpolation positive inside, zero outside | API contract |
| tests/heat_injection.rs | `test_thermalization_era_pure_temperature_shift` | literature | mu/drho = (3/kappa_c) J_bb*(3e6) J_mu(3e6) ~ 0.08 to 10%; energy to 3% | Chluba 2013 MNRAS 434, 352 Eq. 5 (docstring gives oracle+uncertainty); #[ignore]d production-grid test |
| tests/heat_injection.rs | `test_thermalization_suppression_high_z` | dimensional | mu,y < 1e-3 * drho at z=5e6 | exponential J_bb* suppression argument (~exp(-(z/2e6)^2.5)) |
| tests/heat_injection.rs | `test_thermalization_suppression_monotonic` | dimensional | J_bb* and mu/drho monotone decreasing with z; mu/drho<0.01 at z=5e6 | monotonicity of thermalization efficiency |
| tests/heat_injection.rs | `test_thomas_algorithm_accuracy` | analytic | tridiagonal solve of -u''=sin(pi x) matches sin(pi x)/pi^2 to O(h^2) | exact PDE solution |
| tests/heat_injection.rs | `test_timestep_convergence_order` | structural | mu spread < 5% across dy_max in [0.005,0.05]; step count increases | temporal self-convergence |
| tests/heat_injection.rs | `test_transition_region_pde_z3e4` | dimensional | mu,y>0; mu/drho>1e-3, y/drho>0.1; energy to 5% | mixed-mode positivity + energy conservation |
| tests/heat_injection.rs | `test_transition_region_pde_z5e4_gf_comparison` | cross-method | PDE vs GF: mu within 100%, y within 300%, same signs | PDE B&F vs GF visibility convolution; loose by documented decomposition-basis mismatch |
| tests/heat_injection.rs | `test_transition_region_pde_z8e4` | dimensional | mu,y>0; mu>y at z=8e4; energy to 5% | mu-dominance in mu-side transition + energy conservation |
| tests/heat_injection.rs | `test_visibility_function_physical_constraints` | dimensional | J's in [0,1], monotone, correct limits; J_mu/J_y crossing in [3e4,1e5] | model-independent limits; crossing range from mu-y transition literature |
| tests/heat_injection.rs | `test_visibility_functions_literature_limits` | literature | J_bb*, J_mu, J_y limits and z~5e4 crossover; visibility sum within 25% of 1 | Chluba 2013 MNRAS 434, 352 Fig. 2 regime boundaries |
| tests/heat_injection.rs | `test_x_balanced_from_first_principles` | analytic | X_BALANCED = 4G3/(3G2) ~ 3.602 to 1e-14 | exact constant |
| tests/heat_injection.rs | `test_y_efficiency_y_era` | analytic | y = drho/4 (visibility-corrected) to 5% at z=3000; mu << y | exact definition of y-parameter; ZS 1969, Kompaneets 1957 |
| tests/heat_injection.rs | `test_y_era_burst_spectral_purity` | analytic | y/(drho/4) in [0.7,1.3]; \|mu/y\|<0.20 | y=drho/4 exact in y-era |
| tests/heat_injection.rs | `test_y_sz_zero_crossing_from_transcendental_equation` | analytic | Y_SZ zero at x0~3.8310 where x coth(x/2)=4 | transcendental equation solved by independent bisection; Zeldovich & Sunyaev 1969 |
| tests/perturbations_class.rs | `beta_and_delta_b_prime_vs_fixture_finite_differences` | literature | beta = theta_b/k within 1% of fixture; delta_b' within 5% of fixture finite differences | CLASS v3.2.5 fixture (finite-differenced); FD truncation sets the 5% tolerance |
| tests/perturbations_class.rs | `class_fixture_agreement_to_z500` | literature | delta_b, delta_cdm, phi, psi, delta_g, theta_g within 1% (envelope-normalized) of CLASS to z=500 | CLASS v3.2.5 fixture data/class_fixtures/perturbations_subset.csv; 1% tolerance from plan WP-2 |
| tests/perturbations_class.rs | `convergence_in_l_max_and_tolerance` | analytic | halving l_max shifts fields < 2e-3; rtol 100x tighter shifts < 1e-3 | self-convergence criterion in truncation and ODE tolerance |
| tests/perturbations_class.rs | `load_fixture` | structural | fixture has 12 columns per row and 9 k modes | fixture-file structure (data/class_fixtures/perturbations_subset.csv) |
| tests/power_spectra_class.rs | `standard_spectra_match_class_fixture` | literature | TT/EE/TE match CLASS per-band (2-9% at reduced resolution); first acoustic peak at l = 220 +- 5 | CLASS fixture data/class_fixtures/cls_unlensed.dat; band tolerances calibrated to measured reduced-resolution performance (documented in header) |
| tests/science_suite.rs | `science_deep_thermalization_pde` | cross-method | PDE mu at z_h=1e6 matches GF to 5%; J_bb* in (0.1, 0.95); mu/drho < 1.401*0.95 | PDE vs Chluba 2013 GF visibility fits (J_bb*(1e6) ~ 0.14) |
| tests/science_suite.rs | `science_high_z_thermalization_is_temperature_shift` | analytic | G(x, 5e6) = 0.25*G_bb(x) to 1% | Chluba 2013 Eq. 6 full-thermalization limit |
| tests/science_suite.rs | `science_mu_era_coefficient_pde` | cross-method | PDE mu matches GF (3/kappa_c)*J_bb*J_mu*drho to 10%; \|y/mu\| < 10% | PDE vs Chluba 2013 GF visibility formula |
| tests/science_suite.rs | `science_te_decoupling_post_recombination` | literature | rho_e(500) ~ 1, rho_e(200) = 0.86 +- 5%, rho_e(100) = 0.62 +- 5%, ratio = 0.73 +- 5% | comment cites Peebles TLA (Seager, Sasselov & Scott 1999/2000; Ali-Haimoud & Hirata 2011; Chluba & Thomas 2011) and cross-check vs DarkHistory (Liu+ 2020) TLA |
| tests/science_suite.rs | `science_y_era_coefficient_pde` | cross-method | PDE y matches 0.25*J_y*drho to 2%; \|mu/y\| < 4% | PDE vs GF; y = drho/4rho analytic relation with J_y visibility |
