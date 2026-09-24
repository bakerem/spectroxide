# Codebase size review and cleanup plan, 2026-09-24

This report asks where the codebase and the test suite can shrink without losing coverage or changing physics. Four read-only reviewers covered the Rust tests, the Rust production code, the Python package, and the rest of the repository (`dev/`, notebooks, data, git history). Nothing was edited during the review.

Line numbers refer to `main` at `492842a` plus the uncommitted edits in the working tree on 2026-09-24 (`src/solver.rs`, `tests/coverage_gaps.rs`). They drift as phases land; search by test or function name.

Each finding carries one of these status labels:

- **Checked:** the main session re-read the code and confirmed the claim.
- **Reviewer only:** the reviewer quoted evidence, but no second check ran.

## Summary

The bloat sits in one place. `tests/heat_injection.rs` (12,470 lines, 203 tests) duplicates itself, the `src/` unit tests, and the newer anchor suites; about 100 of the 527 Rust tests can go, and the file can shrink to about 6,500 lines. Production Rust has about 900 removable lines out of 7,580 code lines, mostly in `cli.rs`, `output.rs`, and `energy_injection.rs`. The Python package is lean (about 40 lines). The repository carries 11.9 MB of raw mutation-testing output.

The review found one gap larger than the bloat: CI runs neither `tests/heat_injection.rs` nor any of the anchor suites (`physics_identities`, `kompaneets_moments`, `rate_coefficients_first_principles`, `compton_equilibrium_analytic`, `mu_photosphere_profile`, `heat_delivery`, `mms_convergence`, `conservation_fuzz`, `greens_function_checks`, `cosmotherm_comparison`). Status: Checked (`.github/workflows/ci.yml`).

## Execution status (updated as phases land)

| Phase | Items | Status | Branch or commit |
|---|---|---|---|
| 0 | D-1 to D-4, D-6, H-4, H-5 | Done 2026-09-24; D-5 moved to phase 2 | see `git log` |
| 1 | X-1 | Done 2026-09-24 (see note under X-1) | see `git log` |
| 2 | T-1, T-3, T-11, T-12, T-15, D-5 | Done 2026-09-24 (see phase 2 notes) | see `git log` |
| 3 | T-2, T-6 to T-10, T-13 | Done 2026-09-24 (see phase 3 notes) | see `git log` |
| 4 | T-4, T-5, T-14, T-16 | Done 2026-09-24 (see phase 4 notes); T-11 find-index merge not done | see `git log` |
| 5 | C-1 | Not started | |
| 6 | P-3, P-5, P-6, P-10, P-11 | Not started | |
| 7 | P-1, P-2, P-4, P-7, P-8, P-9 | Not started | |
| 8 | Y-1, Y-2 | Not started | |

## Decisions from your review (2026-09-24)

| Item | Decision |
|---|---|
| CI coverage | CI runs `heat_injection` and the anchor suites (C-1) |
| `axion` feature | Keep the code. Remove every mention from user-facing docs. Warn at run time that it is not thoroughly tested (X-1) |
| Mutation-testing output | Remove (H-5) |
| Order | Stale docs first, then the test cleanup in phases with a full release run after each, then production code |

Interpretations made without asking, which you can overturn:

- "Docs" means the README, the Sphinx site (`docs/`), and public docstrings such as `solve()` in `python/spectroxide/solver.py`. CLAUDE.md keeps its axion notes, because developers still must build and test both configurations. The module docstrings of `src/axion.rs` and `python/spectroxide/axion.py` stay; with `docs/api/axion.rst` gone, nothing renders them.
- "The mutation stuff" means `dev/audit/mutation/` (112 files, 11.9 MB) and `dev/scripts/run_mutation_shards.sh`. `dev/audit/mutation_audit.md` stays as the written record, because code comments cite its finding IDs (for example F-R2-1 at `src/bremsstrahlung.rs:484`). Git history keeps the raw files.

## Findings

### Stale documentation (D)

- **D-1.** CLAUDE.md says `tests/greens_function_checks.rs` has 7 tests; it has 10 (four Draine Gaunt anchors were added). Checked. Correction (phase 0): the top-line count of 324 integration tests is right, since `cargo test --release -- --list` lists 328 integration tests, 4 of them ignored.
- **D-2.** CLAUDE.md describes `set_full_te(false)`, which no longer exists anywhere in `src/`. Checked.
- **D-3.** `src/cli.rs:165` says `--split-dcbr` splits DC from BR. It turns off the coupling between DC/BR and the Kompaneets solve (`solver.rs:1700`); the help text at `cli.rs:1033` is correct. Reviewer only.
- **D-4.** `src/kompaneets.rs:575` cites `kompaneets_step_nonlinear_coupled`, which does not exist. Reviewer only.
- **D-5.** About 15 docstrings in `tests/heat_injection.rs` belong to deleted tests and now sit on the next test, so they describe the wrong code (for example lines 219–221, 1252, 1560, 1975, 2885, 3411, 3498, 3667, 3713, 3989, 4161, 4977–4996, 10019). Reviewer only.
- **D-6.** `dev/audit/census/*.json` and `dev/audit/TEST_PROVENANCE.md` cite Python test files that no longer exist (`test_anisotropy.py`, `test_fh_basis.py`, `test_dm_baryon.py`). Checked.

### Rust tests (T)

All line numbers are in `tests/heat_injection.rs` unless another file is named. Estimated totals: about 95–105 tests removed, about 4,000 lines deleted, and about 1,400 lines saved by shared helpers, taking the file to about 6,500 lines and removing about 50 duplicate PDE runs. No test that CLAUDE.md names as load-bearing is touched. Rows 4, 8, 10, and 14 of `dev/audit/term_coverage_matrix.md` must be repointed.

- **T-1. Constants tests copied from `src/`.** Delete 8 tests, about 250 lines: `test_beta_mu_from_zeta_functions` (94), `test_x_balanced_from_first_principles` (5618), `test_spectral_integrals_exact_values` (184), `test_planck_integral_accuracy` (3472), `test_kappa_c_from_numerical_integration` (123; `physics_identities::test_distortion_shape_moment_identities` checks the same moment to 1e-10), `test_mu_shape_zero_crossing_and_sign_structure` (318), `test_compton_equilibrium_planck_exact` (229). First move two stronger checks into `src/`: the transcendental Y_SZ root (261, 1e-6, beats `spectrum::test_y_shape_zero_crossing` at 0.01) and ALPHA_RHO = 2ζ(3)/(π⁴/15) (5604, 1e-14, beats `test_alpha_rho` at 1e-6). Reviewer only.
- **T-2. Green's-function self-consistency.** Delete about 8 tests, about 400 lines. `test_greens_function_energy_accounting` (689) repeats `greens_function_checks::chluba2013_energy_conservation` with the same thresholds. `test_thermalization_suppression_high_z` (477) is weaker than `chluba2013_limit_pure_temperature_shift`. `test_mu_efficiency_deep_mu_era` (365) and `test_y_efficiency_y_era` (424) compare the Green's function with its own visibility functions. Also `test_greens_function_asymptotic_limits` (3675, 50%), `test_visibility_functions_literature_limits` (12355), `test_greens_function_linearity` (512). Reviewer only.
- **T-3. Photon Green's-function duplicates.** Delete 4 tests, about 110 lines: `test_photon_gf_balanced_injection_zero_mu` (5696) repeats `greens::test_mu_from_photon_injection_balanced`; `test_photon_gf_mu_linearity` (6064) and `test_photon_gf_energy_only_limit` (5672) repeat identities in `test_photon_injection_gf_algebraic_identities` (6772); `test_photon_gf_sign_flip_negative_mu` (5716) retypes the formula in `greens.rs:713-722` (pitfall #9). Reviewer only.
- **T-4. Photon-injection μ(x_inj) in the PDE.** About 9 tests become one or two table-driven tests, saving about 650 lines and about 25 PDE runs. All of them test μ = (3/κ_c) α_ρ (x − x₀ P_s) J_bb* J_μ ΔN/N; `mu_from_photon_injection` computes exactly that, so "against the Green's function" and "against the formula" are the same check. Tests: `analytic_match` (1645), `negative_mu_chluba2015` (1716, runs x = 5 twice), `initial_perturbation_evolution` (1272), `pde_vs_gf_photon_injection_high_x`, `_low_x`, `_balanced` (5790, 5863, 5938), `energy_number_decomposition` (6446), `pde_vs_gf_tight_mu_era` (6627), `mu_y_systematics` (6695), `extreme_frequencies` (7226). Keep each row's tolerance (5% absolute at x₀, 10% at x = 10). Reviewer only.
- **T-5. Heat bursts with an identical config.** Fold about 8 tests into one z_h table, saving about 550 lines and about 12 PDE runs. Shared config: 2000 points, σ = 0.01 z_h, z_start = 1.5 z_h, z_end = 500. Tests: `energy_conservation_sweep_tight` (7461), `pde_vs_gf_multi_z_sweep` (8362), `thermalization_suppression_net_decrease` (7922), `spectral_shape_mu_era` (7992), `spectral_shape_y_era` (8065), `spectral_decomposition_residual_sweep` (8291), `transition_region_mixed_distortion` (8578), `heat_pde_amplitude_linearity` (8645). Reviewer only.
- **T-6. One era target asserted many times.** Delete about 6 tests, about 350 lines. y = Δρ/4 at z = 5000: keep `test_pure_y_analytical_convergence` (11925, 1%) and `science_y_era_coefficient_pde` (2%); drop `golden_y_era` (10186), `test_heat_y_era_pure_y_parameter` (7390), `test_y_era_burst_spectral_purity` (3416). μ = 1.401 J at z = 2e5: keep the science-suite test and the multi-z table from T-5; drop `golden_mu_era` (10096). `test_thermalization_era_pure_temperature_shift` (4642, ignored) is superseded by `science_deep_thermalization_pde_z3e6`. Reviewer only.
- **T-7. Linearity.** Delete 3 tests, about 200 lines: `test_pde_linearity_double_injection` (4440), `test_extreme_small_injection` (11988), `test_heat_dm_fann_linear_scaling` (8223) are weaker than `test_heat_pde_amplitude_linearity`. `kompaneets_moments` T6 anchors the Δn² term; repoint matrix row 4. Reviewer only.
- **T-8. Recombination.** Delete 4 tests and merge 3 into 1, about 300 lines. `test_recombination_ionization_history` (2728), `test_recombination_physical_values` (9651), and `test_recombination_quantitative_milestones` (12239) band z = 1100, 800, and 200 at 2–10×; `recombination::test_xe_vs_recfast_milestones` pins the same points to ±6% against HyRec-2. `test_recombination_cache_properties` (2846) repeats `test_recombination_history_matches_uncached`. The three helium Saha tests (2801, 3010, 3509) overlap each other and `test_helium_saha_transitions`. Repoint matrix row 10. Reviewer only.
- **T-9. DC, BR, and Gaunt tests superseded by the anchors.** Delete about 9 tests, about 350 lines. Gaunt tests at 2956, 2981, and 9268 are covered by `gaunt_ff_classical_limit_matches_draine`. `test_dc_br_ratio_pinned_z1e6` (11112) says in its own comment that its center "has never been independently derived". `test_br_absolute_value_z1e6_x1` (12202) is a four-decade bound; `test_dcbr_absorption_coefficients_at_soft_x` (11443) and `test_br_coefficient_saha_transition` (4249) check only sign and finiteness; `test_dcbr_dimensional_scaling_vs_z` (10966) is weaker than `rate_coefficients_first_principles`. The three DC-suppression tests (3142, 4202, 4224) assert H(0) = 1, which `double_compton.rs` also asserts. The literal-CODATA test keeps the pitfall #8 guard. Reviewer only.
- **T-10. Compton equilibrium.** Delete 3 tests, about 130 lines: `test_compton_equilibrium_mu_distortion` (3291), `test_compton_equilibrium_deviation` (4166), `test_perturbative_te_small_mu_distortion` (12288). Stronger: `test_full_te_rho_e_for_mu_distortion` (11040, 1e-6), `test_full_te_perturbative_vs_brute_force` (11571, matrix row 8), `compton_equilibrium_analytic`. Repoint matrix row 8. Reviewer only.
- **T-11. Infrastructure duplicates.** Delete about 15 tests, about 350 lines. `test_output_format_parsing` (9546) is identical to `output::test_output_format_from_str` (Checked). `test_cosmology_presets` (4510) is looser than the `cosmology.rs` tests. `test_solver_config_validation` (9516) is inside `adversarial_inputs::test_solver_config_rejects_bad_params`. `test_custom_injection_closure` and `test_custom_injection_captures_data` (4038, 4054) test Rust closures. `test_tabulated_photon_source_interpolation` (9461) checks only positivity. The heating-rate sign convention appears here (4117), in `coverage_gaps` (384), and in `greens.rs`; keep `greens.rs`. The find-index tests (3721, and four in `coverage_gaps` at 417–448) become one table. Also `test_compton_y_parameter_post_recombination` (8837), `test_photon_survival_post_recombination` (8889, same assertion as 8864), `test_mu_y_shapes_independent` (4369). Reviewer only except where marked.
- **T-12. Tautologies and leftovers.** Delete about 13 tests, about 700 lines. These assert only arithmetic done inside the test: `test_relativistic_correction_magnitude` (5006), `test_dc_backward_euler_accuracy` (5192), `test_bose_factor_taylor_vs_exact` (9570, never calls solver code), `test_pb2009_bose_einstein_temperature` (9786), `test_lambda_expansion_small_at_high_z` (4944). The three dark-photon NWA tests near 1822, 2088, and 2218 never call the `dark_photon` module or `DarkPhotonResonance` (Checked: the file does not import `dark_photon`). `test_plasma_frequency_formula` (2414) is weaker than `plasma_frequency_matches_first_principles`. `test_high_z_dtau_convergence` (4554) runs a dtau = 10 case and never asserts on it. `test_newton_convergence_indirect` (10031) repeats the run in `test_pb2009_energy_conservation` with weaker checks. Keep `find_resonance_z` because an axion test uses it; gate it with `#[cfg(feature = "axion")]` or the default build fails clippy's dead-code check. Reviewer only except where marked.
- **T-13. Other near-duplicate pairs.** Delete about 10 tests, about 450 lines. `test_pde_planck_is_stable_equilibrium` (1200) is inside `test_pde_no_injection_full_range` (4398). `test_heat_dm_annihilation_energy_conservation` (8517, 20%) is weaker than `coverage_gaps::energy_conservation_annihilating_dm_swave` (15%). `coverage_gaps::coupled_vs_split_dcbr_consistency` (25%) is weaker than `test_coupled_vs_split_z1e6` (10%). `test_kompaneets_photon_number_hybrid_grid` (3951) and `test_photon_injection_number_conservation_pure_kompaneets` (6832), both 1%, are weaker than `conservation_fuzz` (1e-9). `test_grid_convergence_rate` (9710) is weaker than `convergence_order`. Three sign-only "DC/BR lowers μ" tests (5057, 10431, 10831) sit beside the quantitative `mu_photosphere_profile`. Reviewer only.
- **T-14. Duplicated setup code.** Refactor to save about 1,000 lines. The file has 116 `ThermalizationSolver::new` calls, 105 literal `SolverConfig` blocks, and 69 `SingleBurst` setups. `science_suite::run_single_burst` and `coverage_gaps::burst_run` are the same helper. The photon-injection starting spectrum is typed out 8 times instead of calling `photon_injection_ic` (6148), and the Gaussian heating closure 7 times instead of calling `gaussian_heating` (1057). Add `tests/common/mod.rs` with burst and photon builders. Hand-written physics oracles (`g_bb`, `ysz`, `planck_n`, CODATA literals) stay in each file under the independent-oracle policy. Reviewer only.
- **T-15. Comments.** Delete or move about 450 lines. The file is 21% comments (2,674 lines). It holds 31 "test_X removed" notes and 28 "previous version / tightened from / R2 audit" histories (for example 685, 7841, 9844, 10090, 11130); git history already records them. Reviewer only.
- **T-16. Split the file.** After T-1 to T-15, split the remaining 6,000 or so lines by topic: `gf_visibility.rs`, `pde_heat.rs`, `pde_photon.rs` (photon injection, post-recombination, `DecayingParticlePhoton`), `dark_sector.rs` (including the four axion tests). Update CLAUDE.md, which says the axion tests live in `heat_injection.rs`. More test crates mean more link jobs; keep `-j 2` on this machine. Reviewer only.

### CI (C)

- **C-1.** Add `heat_injection` (or its T-16 successors) and every anchor suite to `.github/workflows/ci.yml`. Keep `#[ignore]` tests ignored (α_th = 5/2 takes about 7 minutes). Run the dark-sector file with `--features axion` so the four axion integration tests run in CI; update the comment in `ci.yml` that explains why integration suites skip the feature. `cosmotherm_comparison` needs reference data; check which of its tests run without `SPECTROXIDE_GREENS_DB`. Measure the total CI time after the test trim, and split the job if it exceeds the runner limit. Checked (current state of `ci.yml`).

### Axion (X)

- **X-1.** Remove axion from user-facing docs: `README.md:306`, `docs/api/axion.rst` (delete), the axion card and toctree entry in `docs/api/index.rst` (93–100, 128), and the `"axion_resonance"` text in the `solve()` docstring (`python/spectroxide/solver.py:1530-1546`). The CLI help text should not list `axion-resonance`. Add a run-time warning, emitted once per run, in both places a user can reach it:
  - Rust: when a solve uses `InjectionScenario::AxionResonance`, push a warning into the run's warnings (the same channel as the existing validation warnings) stating that axion support is experimental and not thoroughly tested.
  - Python: `warnings.warn(...)` when `solve()` receives `"axion_resonance"`, and on the first call into `spectroxide.axion` helpers.
  - Add one test per language that asserts the warning appears.
  - Check that notebooks under `notebooks/tutorials/` and `docs/` do not mention axion (the current mentions are in `notebooks/observational/` and `paper_figures/`, which are analysis notebooks, not docs; leave them).
  - Done (phase 1), with one change: `solve()` has no Python-side warning of its own. The Rust warning reaches Python through `_emit_solver_warnings`, which re-emits every warning in the binary's JSON, so a second Python warning would duplicate it. The Rust test is `test_axion_experimental_warning` (feature build); the Python test is `TestAxionExperimentalWarning` in `python/tests/test_solver.py`. In the default build, `spectroxide help` no longer mentions axion; the feature build lists `axion-resonance` marked experimental.

### Rust production code (P)

Production code lines (before `mod tests`, not blank, not comment): `cli.rs` 1751, `solver.rs` 1533, `energy_injection.rs` 1014, `kompaneets.rs` 645, `output.rs` 574, `greens.rs` 348, others 1715; total 7580. `kompaneets.rs` and most of `solver.rs` are large because the numerics need them.

- **P-1.** `execute_photon_sweep` (`cli.rs:1918-2029`) is `execute_photon_sweep_batch` (`2035-2186`) with one `x_inj`, and their parse arms (592–632, 633–680) match. Keep both subcommand names, since Python calls both; parse `photon-sweep` into the batch options and delete `PhotonSweepOpts`. About 150 lines. Reviewer only.
- **P-2.** The per-task solve code appears three times (`cli.rs:1837-1880`, `1956-1998`, `2076-2120`), the auto-refine block four times (`cli.rs:1661`, `1972`, `2095`, `solver.rs:2408`), the `n_threads` default three times, and the σ_z default `(z_h*0.04).max(100.0)` six times. Add a shared `run_one_solve`, `GridConfig::apply_injection`, and `InjectionScenario::default_sigma_z`. About 90 lines. Reviewer only.
- **P-3.** `execute_solve` builds `SolverConfig` inline twice (`cli.rs:1669-1682`, `1709-1720`) instead of calling `build_solver_config` (1439); the comment at 1706 cites "special is_continuous logic" that does not exist. About 30 lines. Reviewer only.
- **P-4.** `validate_and_collect_warnings` (`cli.rs:1552-1593`) re-implements `SolverBuilder::build` (`solver.rs:2401-2500`), and the two have diverged: only the CLI rejects `z_end > z_upper`, only the builder warns when `z_start` differs from `z_res`. `injection.validate()` also runs twice. Route the CLI through the builder. About 45 lines; some warning text changes. Reviewer only.
- **P-5.** `cn_dcbr` / `--cn-dcbr` has no caller that turns it on (Checked: outside `src/`, only `tests/mms_convergence.rs` sets it, to `false`). It sits in the unsafe Newton loop (`kompaneets.rs:756-759`, `832-842`, `917-925`, buffers at 431, 525). Its docs contradict each other (`solver.rs:169-171` against `cli.rs:168`), and pitfall #3 rules the method out. ADR 0004 mentions it in a consequence; add a dated addendum there. About 70 lines.
- **P-6.** `nc_stride` / `--nc-stride` has no users. Checked: outside `src/`, only mutation output mentions it. About 20 lines.
- **P-7.** `disable_dcbr`, `coupled_dcbr`, `number_conserving`, and `nc_stride` live on the solver instead of in `SolverConfig`, which forces save-and-restore code (`solver.rs:1786-1802`, 900–903, 2269–2271, 2503–2505). `SolverBuilder` mirrors every config field as an `Option`. Move the flags into `SolverConfig`; the builder then holds one config plus a `z_start_explicit` flag. About 60 lines; breaks direct field access in five test files and `examples/`. Reviewer only.
- **P-8.** In `output.rs`, five `Serializable` impls only forward to inherent methods (573–632); warning headers and footers repeat five times; the `SweepResult` and `PhotonSweepResult` writers (196–297, 322–420) differ only in the Green's-function columns. About 90 lines; `cli_integration` pins the JSON. Reviewer only.
- **P-9.** `InjectionScenario::validate` (`energy_injection.rs:502-802`) writes "must be positive and finite" 17 times and the ascending-order loop 3 times. Add `require_pos`, `require_finite_all`, and `require_ascending` that keep the same messages. About 120 lines. Reviewer only.
- **P-10.** `is_photon_source` (`solver.rs:954-958`) re-implements `has_photon_source()` (`energy_injection.rs:1033`); `burst_params` (`solver.rs:947-952`) copies the match in `characteristic_redshift`; the Gaussian dq/dz appears at `cli.rs:1370` and `1873`. About 20 lines. Reviewer only.
- **P-11.** History inside production doc comments: removal notes at `bremsstrahlung.rs:477-484` and `double_compton.rs:18-19`, `76-80`, `129-135`; the data table and pre-ADR-0004 story on `ENERGY_CHECK_Z_LATE` (`solver.rs:46-65`); 34 comment lines citing audit labels. About 75 lines. Keep the long blocks that state API contracts (`kompaneets.rs:572-634`, `greens.rs:573-615`, `bremsstrahlung.rs:40-73`). Reviewer only.

### Python (Y)

- **Y-1.** `GreensTable.save`/`load` (`greens_table.py:384-433`) and `PhotonGreensTable.save`/`load` (623–673) are near copies. Extract shared helpers. About 35 lines. Reviewer only.
- **Y-2.** `cosmotherm.py` (573, 617, 657), `axion.py` (50), and `dark_photon.py` (29) import cosmology names through the shim in `greens.py:664-672`, which is marked deprecated. Import from `.cosmology` directly. Reviewer only.

### Repository files (H)

- **H-4.** Delete `dev/audit/census/*.json` (5 files, 163 KB). They cite deleted test files (D-6). Correction (phase 0): `dev/audit/AUDIT_SUMMARY.md` and `dev/scripts/build_test_provenance.py`, which reads them as input, also mention them; both now point to git history.
- **H-5.** Remove `dev/audit/mutation/` (112 files, 11.9 MB, of which 11.3 MB is `outcomes.json`) and `dev/scripts/run_mutation_shards.sh`. Update the references in CLAUDE.md (script list), `dev/audit/mutation_audit.md`, `dev/audit/R2_WRAPUP_TODO.md`, and `dev/PLAN_VALIDATION_ROUND2_2026-07-06.md` to say the raw output lives in git history. Checked (reference list).

## Fix plan

Rules for every phase:

1. Run one heavy job at a time on this machine (`-j 2`, at most two test threads).
2. After each phase: `cargo fmt`, `cargo clippy --all-targets -- -D warnings` in both configurations, and the full release test suite in both configurations.
3. Before deleting any test (phases 2–4), give a fresh claim-verifier only the deletion list, each deleted test's named stronger replacement, and the two test bodies. It must confirm the replacement asserts the same property at an equal or tighter tolerance. Drop any deletion it rejects.
4. Production phases (6–7) must leave physics output bit-identical: compare a CLI `sweep`, a `photon-sweep-batch`, and a `solve` for each injection type before and after.
5. Update the execution status table when a phase lands. Commit each phase separately; documentation-only phases carry `[skip ci]`.

**Phase 0: stale docs and repository files.** D-1 to D-6, H-4, H-5. No code changes except comments.

**Phase 1: axion.** X-1. Code change (warnings) plus docs; run both configurations.

**Phase 2: test cleanup, duplicates and tautologies.** T-1 (move the two stronger checks into `src/` first), T-3, T-11, T-12, T-15, and the D-5 docstrings.

Phase 2 notes (2026-09-24). A fresh claim-verifier checked 39 deletions: 29 confirmed, 8 partial, 2 refuted. Outcome:

- Deleted 32 tests from `heat_injection.rs` and `heating_rate_per_redshift_sign_convention` from `coverage_gaps.rs`.
- Refuted: `test_compton_equilibrium_planck_exact` was tighter than `spectrum::test_compton_equilibrium_planck` (1e-4 on 10k points against 1e-3 on 5k), so the src test was tightened to match before deleting. `test_alpha_rho_from_integrals` was *not* moved: the src `test_alpha_rho` (0.37020884 ± 1e-6) is 1000× tighter on the literal, and the 1e-14 identity uses the code's own constants. It was deleted.
- Partial rows, assertions moved into src first: the Y_SZ transcendental root (1e-6) into `spectrum::test_y_shape_zero_crossing`; G₃ quadrature at 200k points and 1e-7 into `test_spectral_integral_g3`; Planck 2015 ω_cdm and both presets' Ω_m bands into the cosmology preset tests; y_C(500) < 0.05 and y_C(100) < 0.01 into `test_compton_y_parameter_low_z`; n_e against a retyped n_H formula at 1e-10 into the new `cosmology::test_n_e_from_first_principles`.
- Partial rows kept and trimmed instead: `test_pb2009_bose_einstein_temperature` lost its φ_BE tautology and became `test_decompose_bose_einstein_recovers_mu`; `test_high_z_dtau_convergence` lost its never-asserted dtau = 10 run and became `test_high_z_mu_vs_gf_dtau3` (one PDE run instead of two).
- Deferred to phase 4: merging the five find-index tests into one table (T-11), since `coverage_gaps.rs` also carries unrelated uncommitted work.
- T-15 and D-5: a comments-only pass removed 238 lines (30 removal notes, about 24 history notes, 7 empty section banners) and fixed 4 misplaced docstrings; the comment-stripped code was verified token-identical.
- Term-matrix rows 5, 8, and 14 repointed to surviving tests.

**Phase 3: test cleanup, weaker copies.** T-2, T-6 to T-10, T-13. Repoint term-matrix rows 4, 8, 10, and 14.

Phase 3 notes (2026-09-24). Two fresh claim-verifiers checked 45 deletions: 26 confirmed, 19 partial, none refuted outright. Outcome:

- Deleted 35 tests from `heat_injection.rs` (1,757 lines, including 27 section banners left empty): 14 from the Green's-function, era, linearity, and near-duplicate groups (T-2, T-6, T-7, T-13), and all 21 from the recombination, DC/BR/Gaunt, and Compton-equilibrium groups (T-8 to T-10).
- Kept, because the verifier found an assertion the replacement lacks: `golden_mu_era_spectral_shape` and `golden_y_era_spectral_shape` (with the transition golden test, the only `diag_newton_exhausted == 0` checks in the suite), `test_y_era_burst_spectral_purity` (z_h = 1e4), `test_thermalization_era_pure_temperature_shift` (the only energy check above z = 5e5), `test_pde_linearity_double_injection` (Δρ ratio, z = 5e4), `test_extreme_small_injection` (100× amplitude range), `coverage_gaps::coupled_vs_split_dcbr_consistency` (5% Δρ agreement at z_h = 2e5), `test_photon_injection_number_conservation_pure_kompaneets` (the only number check with DC/BR on), `test_mu_decay_eigenvalue` (not sign-only: μ/μ₀ > 0.85 under Kompaneets alone), and `test_dcbr_thermalizes_mu_distortion`. Phase 4 can fold some into the T-5 table.
- Merged into `src/recombination.rs` before deleting: the three helium and Saha tests into `test_helium_saha_transitions`; X_e(1e4) = 1.16 ± 0.10, X_e(3000) in (1.0, 1.2), X_e(1500) > 0.9, X_e(1400) in (0.60, 1.05), and monotonicity over 2000 to 1500 into the new `test_xe_bounds_between_anchors`; the cache test's extra points (1% at z = 50, 1e4, 5e5; 2% across the Saha to Peebles switch) into `test_recombination_history_matches_uncached`. One deleted test had X_e(1e4) centered at 1 + f_He = 1.079; the right value is 1 + 2f_He ≈ 1.16, because helium is still doubly ionized at z = 1e4.
- Physics fix. `test_full_te_rho_e_for_mu_distortion` asserted ρ_eq > 1 for a Bose–Einstein spectrum. Every BE spectrum obeys n(1 + n) = −dn/dx, so I₄ = 4G₃ and ρ_eq = 1 exactly. The assertion passed only because of the 2.2e-6 quadrature error (pitfall #9). It now asserts ρ_eq(BE) − ρ_eq(Planck) < 1e-8 on one grid (measured: 2e-14). The deleted `test_compton_equilibrium_mu_distortion_deviation` pinned that same quadrature error inside (1e-5, 1e-3).
- Removed the now-unused `assert_rel` helper. Term-matrix rows 5, 6, 7, 8, and 10 repointed; row 4 needed no change, because its test was kept.
- Suite after phase 3: 450 pass and 4 ignored in the default build, 459 and 4 with `--features axion`. The CLAUDE.md counts are updated in phase 4.
- Dated audit records (`TEST_PROVENANCE.md`, `test_assertions.json`, `test_redundancy_audit.md`, `R2_WRAPUP_TODO.md`, the 2026-07 plan files) still name deleted tests. They are records of their dates and were left as written.

**Phase 4: test structure.** T-14 shared helpers, then the T-4 and T-5 tables (each row keeps its original tolerance), then the T-16 split. Update the test counts in CLAUDE.md.

Phase 4 notes (2026-09-24).

- T-16 first: a script moved every item of `tests/heat_injection.rs` verbatim into `pde_heat.rs` (44 tests + 1 ignored), `pde_photon.rs` (26), `solver_numerics.rs` (15), `gf_visibility.rs` (20), `components.rs` (23), and `dark_sector.rs` (3, plus 5 behind `--features axion`), with shared helpers in `tests/common/mod.rs`. Numbered section banners were dropped; comment blocks that explain a test stay above it.
- T-4 and T-5 changed approach. Folding the tests into tables would have renamed tests that CLAUDE.md and the term-coverage matrix cite, and forced a tolerance decision per row. Instead every test keeps its name and assertions, and `common::memo` runs each distinct PDE configuration once per test binary (`standard_burst`, `photon_run`, `baseline_run`, keyed on every input). The eight T-5 tests went from 22 runs to 9; the T-4 photon tests from about 36 to about 22.
- T-14: 26 repeated burst setups now call `burst_solver`; inline photon initial spectra (6) and Gaussian heating closures (3) call the shared helpers. `science_suite::run_single_burst` and `coverage_gaps::burst_run` were left alone: they differ in σ, grid, and return type, and `coverage_gaps.rs` carries unrelated uncommitted work.
- A fresh claim-verifier confirmed that the 55 changed tests build bit-identical solver inputs, keep every assertion and tolerance, and that the other 82 moved byte for byte.
- The verifier found a pre-existing bug. In `test_nc_energy_y_era_and_high_z_mu`, the reference runs for parts 2 and 3 never set `number_conserving = false`, and `ThermalizationSolver::new` defaults it to `true`, so each part compared a run with itself. Both reference runs now turn NC off. Part 2 (y-era, below `nc_z_min` = 5e4) still agrees bit for bit, as it must. Part 3 (z_h = 2e5) now differs by 4e-9 in μ/Δρ, with NC slightly further from 1.401; the test passes on its 1% slack, so its comment no longer claims that NC improves μ.
- Not done: merging the find-index tests (T-11), since four of them sit in `coverage_gaps.rs`.
- Lines: 9,095 in `heat_injection.rs` became 8,254 across the seven files. Suite: 450 pass and 4 ignored in the default build, 459 and 4 with `--features axion`, unchanged from phase 3.

**Phase 5: CI.** C-1, with a measured runtime.

**Phase 6: production removals.** P-5 (with the ADR 0004 addendum), P-6, P-3, P-10, P-11. Rerun the `debug_assertions` release suite, since P-5 touches the unsafe Newton loop (pitfall #10).

**Phase 7: production refactors.** P-1, P-2, P-8, P-9, then P-4 (warning text changes), then P-7 (struct literals across tests and examples).

**Phase 8: Python.** Y-1, Y-2.

## Open items for you

These need a decision and are not in the plan:

- `notebooks/observational/firas_photon_limits.ipynb` (883 KB): superseded by the paper-figure pipeline, which builds its copy from `dev/scripts/remake_firas_photon_limits.py`. Delete or keep.
- Finished plan files (`dev/PLAN_VALIDATION_ROUND2_2026-07-06.md`, `dev/PLAN_VALIDATION_AUDIT_2026-07-02.md`) and `dev/audit/docs_style_rereview_2026-09-21/`: keep as records or remove.
- Tutorial notebook outputs (about 2.4 MB): stripping them breaks the Sphinx site if it renders stored output; check `nbsphinx_execute` first.
- `--no-auto-refine` (no caller in code) and `greens::distortion_from_photon_injection` (63 lines, no Rust caller outside its own test; Python has its own port).
- Git history holds about 10.8 MB of blobs no longer in `HEAD`. Removing them means rewriting history, which is destructive; this review does not recommend it.
