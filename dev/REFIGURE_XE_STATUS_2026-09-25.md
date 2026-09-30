# Refigure after the X_e change (ADR 0009, ADR 0011), 2026-09-25

Chain: `<scratchpad>/chain.sh`, log `<scratchpad>/chain.log`, per-notebook logs `nb_<name>.log`.
Baseline PDFs (working tree before rerun) copied to `<scratchpad>/baseline/`.

## Affected (run ends below z = 1575, or uses X_e through the photon GF)
- cosmotherm_comparison (z_end 500)
- photon_injection_spectra (z_end 50, photon GF with cosmo)
- firas_photon_limits (z_end 500, photon GF)
- energy_conservation (run_sweep default z_end 10)
- mu_y_vs_injection_redshift (run_sweep default z_end 10)
- pathological_heating (z_end 100; HQ GF table z_end 1001, moved aside to
  `~/.spectroxide/greens_table_hq.pre_xe_2026-09-25.npz` so it rebuilds)
- dm_scenario_comparison (Z_END 1001; caches keyed on physics hash, rebuild automatically;
  8000-point GF table, slowest)

## Not rerun
- convergence_study: runs end at z = 1e4 or 5e4.
- mms_convergence: Kompaneets kernel only.
- dark_photon_constraints (Fig. 8): regenerated 2026-09-25 17:18 by another session, after the X_e code.
- visibility_functions (Fig. 2): three table rows (z_h 3e3, 5.1e4, 2.7e6) rerun at z_end 500 change
  5.8% of peak at x = 1e-5, ≤ 4e-8 above x = 0.01, ≤ 2e-12 above x = 0.1. The fit uses x in [0.5, 20]:
  no rebuild needed.

## GF table spot check (HQ table, 6 columns rebuilt)
max|ΔG|/peak: z_h 1.0e3 17%, 1.53e3 2.2e-3, 2.9e3 1.5e-4, 8.5e3 1.4e-4, 7.2e4 2.1e-5, 6.1e5 5.9e-5.

## Result (2026-09-26 00:18, all seven notebooks exit 0, all computed fresh)
Rendered diff at 150 dpi against the pre-rerun working-tree PDFs:
- unchanged (0 px): photon_injection_spectra, firas_photon_limits_paper
- sub-pixel only: pde_cosmotherm_comparison (31 px), pde_mu_y_vs_zh (10 px),
  pathological_heating_validation (163 px), pde_gf_dm_comparison (241 px)
- visible: pde_energy_conservation and _photon: uniform offset of about -0.004% (heat), -0.02%
  (photon x_inj = 0.5). Cause is the z_end default (500 -> 10, ADR 0010), not X_e: burst at z_h = 1e4
  gives drho 9.997440e-6 (z_end 500), 9.997090e-6 (z_end 10), 9.997073e-6 (z_end 10, fixed X_e).
  Heat panel also gained a zero line (notebook code change).
Old tables kept: `~/.spectroxide/greens_table_hq.pre_xe_2026-09-25.npz`,
`greens_table_dm_7c3bb703b2e74025.npz`.
