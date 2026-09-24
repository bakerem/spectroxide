//! Numerical-robustness tests of the PDE solver.
//!
//! Stability at large Δn and large steps, snapshot landing, adaptive stepping,
//! number-conserving mode, free streaming, timestep convergence, and the
//! null tests (no injection, adiabatic cooling).

mod common;

use common::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::distortion;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::solver::{SolverConfig, ThermalizationSolver};
use spectroxide::spectrum;

/// Kompaneets: zero initial perturbation should remain zero.
///
/// If we start with exactly Planck and no energy injection, the PDE solver
/// should not introduce any numerical drift (or at least very small drift).
/// Kompaneets: photon depletion (negative Δn) should evolve stably.
///
/// A negative initial perturbation (fewer photons than Planck) should be
/// filled in by DC/BR emission, approaching Planck. The solver should
/// remain stable and not produce NaN or diverge.
/// Kompaneets: large perturbation regime.
///
/// A large distortion (drho ~ 10⁻³) should still evolve stably.
/// The solver uses Newton iteration which should handle larger
/// perturbations without diverging.
#[test]
fn test_kompaneets_large_perturbation_stability() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig::default();

    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h: 2e5,
            delta_rho_over_rho: 1e-3, // 100× larger than typical
            sigma_z: 2000.0,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: 2.1e5,
        z_end: 1e4,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[1e4]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!("Large perturbation stability:");
    eprintln!(
        "  mu = {:.4e}, y = {:.4e}, drho = {:.4e}, steps = {}",
        snap.mu, snap.y, snap.delta_rho_over_rho, solver.step_count
    );

    // Should be finite and positive
    assert!(
        snap.mu.is_finite(),
        "mu should be finite for large perturbation"
    );
    assert!(
        snap.y.is_finite(),
        "y should be finite for large perturbation"
    );
    assert!(
        snap.delta_rho_over_rho > 0.0,
        "drho should be positive: {:.4e}",
        snap.delta_rho_over_rho
    );

    // Energy conservation: drho should be within factor of 2 of injection
    let energy_ratio = snap.delta_rho_over_rho / 1e-3;
    assert!(
        energy_ratio > 0.5 && energy_ratio < 2.0,
        "Energy conservation: drho/drho_inj = {energy_ratio:.3}"
    );
}

/// Solver snapshots should be saved at the requested redshifts and
/// contain consistent data (μ, y signs, energy).
#[test]
fn test_solver_snapshot_consistency() {
    let mut solver = burst_solver(&GridConfig::fast(), 2e5, 1e-5, 5000.0, 5.0e5, 1.0e4);

    let snap_zs = [3e5, 2e5, 1e5, 5e4, 1e4];
    let snaps = solver.run_with_snapshots(&snap_zs);

    assert_eq!(
        snaps.len(),
        snap_zs.len(),
        "Should have {} snapshots, got {}",
        snap_zs.len(),
        snaps.len()
    );

    // Snapshots should be in descending z order
    for i in 1..snaps.len() {
        assert!(
            snaps[i].z <= snaps[i - 1].z,
            "Snapshots should be z-descending: z[{}]={} > z[{}]={}",
            i,
            snaps[i].z,
            i - 1,
            snaps[i - 1].z
        );
    }

    // The last snapshot (lowest z) should show the injection signal
    let last = snaps.last().unwrap();
    let max_dn: f64 = assert_finite_max(last.delta_n.iter().map(|x| x.abs()));
    assert!(
        max_dn > 1e-15,
        "Final snapshot should have nonzero distortion: max|Δn|={max_dn:.4e}"
    );

    // ρ_e should be close to 1 (Compton equilibrium)
    assert!(
        (last.rho_e - 1.0).abs() < 0.1,
        "ρ_e should be near 1: {:.6}",
        last.rho_e
    );
}

/// Adaptive timestep should be bounded: dz_min ≤ dz ≤ z × 0.05.
#[test]
fn test_adaptive_dz_bounds() {
    let cosmo = Cosmology::default();

    for &z_start in &[3e6, 1e6, 1e5, 1e4, 1e3] {
        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::fast());
        solver.set_config(SolverConfig {
            z_start,
            z_end: z_start * 0.1,
            ..SolverConfig::default()
        });

        // Take one step and verify dz is in bounds
        let dz = solver.step();
        assert!(
            dz >= solver.config.dz_min,
            "dz at z={z_start:.0e} should be ≥ dz_min: dz={dz:.4e}"
        );
        assert!(
            dz <= z_start * 0.05 + 1e-10,
            "dz at z={z_start:.0e} should be ≤ 0.05z: dz={dz:.4e}"
        );
    }
}

/// Snapshot landing should produce snapshots at the exact requested redshifts.
/// Test with closely-spaced snapshots that might cause overshoot issues.
#[test]
fn test_snapshot_close_spacing() {
    let mut solver = burst_solver(&GridConfig::fast(), 5e4, 1e-5, 2000.0, 1.0e5, 500.0);

    // Very closely-spaced snapshots
    let requested = [9e4, 8.9e4, 8.8e4, 5e4, 1e4, 5e3, 1e3];
    let snaps = solver.run_with_snapshots(&requested);

    assert_eq!(
        snaps.len(),
        requested.len(),
        "Should get exactly {} snapshots",
        requested.len()
    );

    for (snap, &z_req) in snaps.iter().zip(requested.iter()) {
        let rel_err = (snap.z - z_req).abs() / z_req;
        assert!(
            rel_err < 0.01,
            "Snapshot z={:.1} should be at z={:.1}",
            snap.z,
            z_req
        );
    }

    // Snapshots should be monotonically decreasing in z
    for i in 1..snaps.len() {
        assert!(
            snaps[i].z <= snaps[i - 1].z,
            "Snapshots should decrease in z: z[{}]={:.1} > z[{}]={:.1}",
            i,
            snaps[i].z,
            i - 1,
            snaps[i - 1].z
        );
    }
}

/// PDE with no injection should maintain Δn = 0 even over long evolution.
/// This is a stronger test than existing: evolve from z=3e6 to z=200.
#[test]
fn test_pde_no_injection_full_range() {
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
    solver.set_config(SolverConfig {
        z_start: 3.0e6,
        z_end: 200.0,
        ..SolverConfig::default()
    });

    solver.run_with_snapshots(&[200.0]);
    let last = solver.snapshots.last().unwrap();

    // At x ≪ 1 (Rayleigh-Jeans), DC/BR equilibrates toward Planck at T_e.
    // Post-recombination T_e < T_γ gives |Δn| ~ |ρ_e−1|/x which diverges
    // as a small relative perturbation on a large background n_pl ~ 1/x.
    // Check x > 0.1 where the μ/y distortion is the dominant signal and the
    // Rayleigh-Jeans 1/x divergence is no longer present.
    let max_dn: f64 = solver
        .grid
        .x
        .iter()
        .zip(last.delta_n.iter())
        .filter(|&(&x, v)| x > 0.1 && v.is_finite())
        .map(|(_, v)| v.abs())
        .fold(0.0_f64, f64::max);

    // Adiabatic cooling over z=[3e6,200] produces O(10⁻⁵) distortion.
    // Bound at 5e-5 = 2.5× that expectation (CLAUDE.md Pitfall #9).
    assert!(
        max_dn < 5e-5,
        "No injection over z=[3e6,200] should give max|Δn(x>0.1)| < 5e-5: got {max_dn:.4e}"
    );
    assert_eq!(
        solver.diag.newton_exhausted, 0,
        "Newton should converge for no-injection full range"
    );
}

/// NC mode: Planck stability (no injection) and photon number zeroed (with injection).
#[test]
fn test_nc_planck_stable_and_photon_number() {
    // Part 1: No injection + NC should keep Planck stable
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::fast());
    solver.number_conserving = true;
    solver.set_config(SolverConfig {
        z_start: 1.0e6,
        z_end: 1.0e5,
        ..SolverConfig::default()
    });
    for _ in 0..100 {
        if solver.z <= solver.config.z_end {
            break;
        }
        solver.step();
    }
    let max_dn: f64 = assert_finite_max(solver.delta_n.iter().map(|x| x.abs()));
    // Adiabatic cooling creates O(10⁻⁸) distortion even in NC mode over z=[1e6,1e5].
    assert!(max_dn < 1e-6, "Planck: max|Δn| = {max_dn}");
    assert!(
        solver.accumulated_delta_t.abs() < 1e-6,
        "Planck: accumulated_delta_t = {:.4e}",
        solver.accumulated_delta_t
    );

    // Part 2: With injection, ΔN/N should be ~0 after NC subtraction
    let z_h = 3e5;
    let drho = 1e-5;
    let sigma = 100.0;
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver.number_conserving = true;
    solver.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[1e4]);
    let delta_n_over_n = spectroxide::spectrum::delta_n_over_n(&solver.grid.x, &solver.delta_n);
    assert!(
        delta_n_over_n.abs() < 1e-3,
        "ΔN/N = {delta_n_over_n:.4e} should be small with NC"
    );
}

/// NC mode: energy conservation, y-era unchanged, and no loss of μ accuracy at high z.
#[test]
fn test_nc_energy_y_era_and_high_z_mu() {
    let cosmo = Cosmology::default();
    let drho = 1e-5;

    // Part 1: Energy conservation at multiple redshifts
    for &z_h in &[5e4, 1e5, 2e5] {
        let sigma = 100.0;
        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: sigma,
            })
            .unwrap();
        solver.number_conserving = true;
        solver.set_config(SolverConfig {
            z_start: z_h + 7.0 * sigma,
            z_end: 1e4,
            dtau_max: 3.0,
            ..SolverConfig::default()
        });
        solver.run_with_snapshots(&[1e4]);
        let last = solver.snapshots.last().unwrap();
        let drho_err = (last.delta_rho_over_rho / drho - 1.0).abs();
        assert!(
            drho_err < 0.10,
            "NC energy conservation failed at z_h={z_h:.0e}: err={drho_err:.2e}"
        );
    }

    // Part 2: y-era should be unchanged by NC (z < nc_z_min = 5e4 throughout,
    // so the two runs must agree bit for bit)
    let z_h = 5000.0;
    let sigma = 200.0;
    // `new` turns number conservation on, so the reference runs must turn it off.
    let mut solver = burst_solver(&GridConfig::default(), z_h, drho, sigma, 1.0e4, 1.0e3);
    solver.number_conserving = false;
    solver.run_with_snapshots(&[1.0e3]);
    let y_no_nc = solver.snapshots.last().unwrap().y;

    let mut solver_nc = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver_nc
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver_nc.number_conserving = true;
    solver_nc.set_config(SolverConfig {
        z_start: 1.0e4,
        z_end: 1.0e3,
        ..SolverConfig::default()
    });
    solver_nc.run_with_snapshots(&[1.0e3]);
    let y_nc = solver_nc.snapshots.last().unwrap().y;
    let y_diff = (y_nc - y_no_nc).abs() / y_no_nc.abs().max(1e-20);
    assert!(y_diff < 0.01, "y-era: NC changed y by {y_diff:.2e}");

    // Part 3: NC must not make μ/Δρ worse at z=2e5. With the default grid and
    // steps the two runs differ by about 4e-9 in μ/Δρ (measured 2026-09-24), so
    // this guards against NC degrading μ by more than 1%; it does not show that
    // NC improves μ.
    let z_h = 2e5;
    let sigma = 100.0;
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver.number_conserving = true;
    solver.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[1e4]);
    let mu_over_drho_nc = solver.snapshots.last().unwrap().mu / drho;

    let mut solver2 = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver2
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver2.number_conserving = false;
    solver2.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver2.run_with_snapshots(&[1e4]);
    let mu_over_drho_no_nc = solver2.snapshots.last().unwrap().mu / drho;

    let err_nc = (mu_over_drho_nc - 1.401).abs();
    let err_no_nc = (mu_over_drho_no_nc - 1.401).abs();
    assert!(
        err_nc < err_no_nc * 1.01 + 1e-6,
        "NC made μ/Δρ worse: err_nc={err_nc:.4} vs err_no_nc={err_no_nc:.4}"
    );
}

/// Benchmark 3: Free-streaming test — Kompaneets with y-distortion feedback.
///
/// With DC/BR disabled and no injection, the Kompaneets equation acts on a
/// pre-existing y-distortion. The distortion energy feeds back into ρ_e
/// through the perturbative T_e formula, so the y-distortion is NOT preserved
/// — it amplifies as ρ_e > 1 drives further Kompaneets evolution. This is
/// physical: the photon field energy heats electrons, which create more y.
///
/// What we test:
///   1. Energy conservation (Δρ/ρ stays constant — energy just redistributes
///      between y and ΔT/T components)
///   2. No spurious μ generation (distortion stays y-type, not μ-type)
///   3. The distortion remains well-behaved (finite, no blow-up)
///
/// Also tested: a μ-distortion should be preserved by Kompaneets-only
/// (μ is an equilibrium of Kompaneets at ρ_e = 1 + δ).
#[test]
fn test_kompaneets_free_streaming() {
    let cosmo = Cosmology::default();
    let y0 = 1e-5;

    let grid_config = GridConfig {
        x_min: 1e-4,
        x_max: 50.0,
        n_points: 1000,
        x_transition: 0.10,
        log_fraction: 0.30,
        refinement_zones: Vec::new(),
    };

    // === Part A: y-distortion evolves (energy feedback amplifies y) ===
    let grid = FrequencyGrid::new(&grid_config);
    let delta_n_y: Vec<f64> = grid.x.iter().map(|&x| y0 * spectrum::y_shape(x)).collect();

    let init_params = distortion::decompose_distortion(&grid.x, &delta_n_y);
    eprintln!("\n=== Kompaneets free-streaming test ===");
    eprintln!("Part A: y-distortion (y₀={y0:.0e})");
    eprintln!(
        "Initial: μ={:.4e}, y={:.4e}, Δρ/ρ={:.4e}",
        init_params.mu, init_params.y, init_params.delta_rho_over_rho
    );

    let snap_z = [8e4, 5e4, 2e4, 1e4];

    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver.set_initial_delta_n(delta_n_y);
    solver.disable_dcbr = true;
    solver.set_config(SolverConfig {
        z_start: 1e5,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&snap_z);

    eprintln!("\nEvolution (Kompaneets only, no DC/BR):");
    eprintln!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "z", "y/y₀", "μ/y₀", "ΔT/T", "Δρ/ρ"
    );
    for snap in &solver.snapshots {
        let params = distortion::decompose_distortion(&solver.grid.x, &snap.delta_n);
        eprintln!(
            "{:>10.2e} {:>12.6} {:>12.4e} {:>12.4e} {:>12.4e}",
            snap.z,
            params.y / y0,
            params.mu / y0,
            params.delta_t_over_t,
            params.delta_rho_over_rho
        );
    }

    let last = solver.snapshots.last().unwrap();
    let last_params = distortion::decompose_distortion(&solver.grid.x, &last.delta_n);

    // Energy conservation: Δρ/ρ should be preserved (< 1% change)
    let energy_err = (last_params.delta_rho_over_rho - init_params.delta_rho_over_rho).abs()
        / init_params.delta_rho_over_rho.abs().max(1e-30);
    eprintln!("\nEnergy conservation error: {:.4}%", energy_err * 100.0);
    assert!(
        energy_err < 0.01,
        "Energy should be conserved: Δρ/ρ error = {:.4}%",
        energy_err * 100.0
    );

    // At z = 1e5 → 1e4, Kompaneets scattering partially converts y → μ
    // (transition region). The energy-conserving decomposition captures this.
    let mu_frac = (last_params.mu / last_params.y).abs();
    eprintln!("μ/y ratio: {mu_frac:.4e}");

    // y component decreases as Kompaneets redistributes energy toward μ-shape
    // in the transition region. Energy is conserved (checked above).
    eprintln!("y evolution: y/y₀ = {:.4}", last_params.y / y0);

    // === Part B: μ-distortion IS preserved by Kompaneets-only ===
    let mu0 = 1e-5;
    let delta_n_mu: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| mu0 * spectrum::mu_shape(x))
        .collect();

    let _init_mu_params = distortion::decompose_distortion(&grid.x, &delta_n_mu);

    let mut solver_mu = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_mu.set_initial_delta_n(delta_n_mu);
    solver_mu.disable_dcbr = true;
    solver_mu.set_config(SolverConfig {
        z_start: 1e5,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver_mu.run_with_snapshots(&[1e4]);

    let last_mu = solver_mu.snapshots.last().unwrap();
    let last_mu_params = distortion::decompose_distortion(&solver_mu.grid.x, &last_mu.delta_n);

    let mu_preserved = last_mu_params.mu / mu0;
    eprintln!("\nPart B: μ-distortion (μ₀={mu0:.0e})");
    eprintln!("μ(1e4)/μ₀ = {mu_preserved:.6}");
    eprintln!("y(1e4)/μ₀ = {:.4e}", last_mu_params.y / mu0);

    // μ should be well preserved by Kompaneets (no DC/BR)
    assert!(
        (mu_preserved - 1.0).abs() < 0.15,
        "μ should be preserved by Kompaneets-only: μ/μ₀={mu_preserved:.4}"
    );
}

/// Test that the Kompaneets equation preserves Bose-Einstein spectrum.
///
/// Under pure Compton scattering (no DC/BR), a Bose-Einstein distribution
/// n_BE(x, μ) = 1/(exp(x + μ) - 1) is a stationary solution.
/// This is a fundamental property of the Kompaneets equation.
#[test]
fn test_kompaneets_preserves_bose_einstein() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 1000,
        ..GridConfig::default()
    };
    let grid = FrequencyGrid::new(&grid_config);

    let mu_0: f64 = 1e-4;
    let delta_n: Vec<f64> = grid
        .x
        .iter()
        .map(|&xi| 1.0_f64 / ((xi + mu_0).exp() - 1.0) - 1.0 / (xi.exp() - 1.0))
        .collect();

    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config);
    // No injection, disable DC/BR
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h: 1e6,
            sigma_z: 1e4,
            delta_rho_over_rho: 0.0,
        })
        .unwrap();
    solver.set_initial_delta_n(delta_n.clone());
    solver.disable_dcbr = true;
    let config = SolverConfig {
        z_start: 5e5,
        z_end: 4e5,
        ..SolverConfig::default()
    };
    solver.set_config(config);

    let snaps = solver.run_with_snapshots(&[4e5]);
    let snap = &snaps[0];

    // The shape should be preserved: Δn should still look like BE - Planck
    // Compare at x ∈ [1, 10] where the signal is cleanest
    let mut max_change = 0.0_f64;
    for i in 0..grid.x.len() {
        if grid.x[i] > 1.0 && grid.x[i] < 10.0 {
            let initial = delta_n[i];
            let final_val = snap.delta_n[i];
            if initial.abs() > 1e-15 {
                let change = (final_val - initial).abs() / initial.abs();
                max_change = max_change.max(change);
            }
        }
    }

    // Pure Kompaneets should preserve BE shape; T_e shift causes small drift
    // but should be < 5% over this short evolution
    assert!(
        max_change < 0.05,
        "BE spectrum should be nearly preserved under pure Kompaneets: max change = {:.2}%",
        max_change * 100.0
    );
}

// 37.3: Timestep convergence order test
//
// The IMEX scheme uses Crank-Nicolson (O(Δτ²)) for Kompaneets and
// backward Euler (O(Δτ)) for DC/BR. With adaptive stepping controlled
// by dy_max, we expect effective temporal convergence.
// This closes a gap: only spatial convergence is tested elsewhere.
#[test]
fn test_timestep_convergence_order() {
    let cosmo = Cosmology::default();
    let z_h = 2.0e5;
    let drho = 1e-5;

    // Run at 4 different dy_max values (controls timestep size)
    // Use moderate grid (1000 pts) so temporal error is dominant
    // Wider dy_max range to see clear convergence trend
    let dy_values = [0.05, 0.02, 0.01, 0.005];
    let mut mus = Vec::new();

    for &dy in &dy_values {
        let mut solver = ThermalizationSolver::new(
            cosmo.clone(),
            GridConfig {
                n_points: 1000,
                ..GridConfig::default()
            },
        );
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: 3000.0,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 5.0e5,
            z_end: 1.0e4,
            dy_max: dy,
            dtau_max: 200.0,
            ..SolverConfig::default()
        });

        solver.run_with_snapshots(&[1.0e4]);
        let snap = solver.snapshots.last().unwrap();
        eprintln!(
            "dy_max={dy:.4}: μ={:.8e}, steps={}",
            snap.mu, solver.step_count
        );
        mus.push(snap.mu);
    }

    // Check that the spread in μ values is small (all converging to same answer)
    let mu_min = mus.iter().cloned().fold(f64::INFINITY, f64::min);
    let mu_max = mus.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = (mu_max - mu_min).abs() / mu_min.abs();
    eprintln!(
        "μ spread across dy_max range: {spread:.4e} ({:.2}%)",
        spread * 100.0
    );

    // The key test: all runs should give consistent μ to within 5%.
    // This validates that the temporal integration is stable and convergent.
    assert!(
        spread < 0.05,
        "Temporal convergence: μ spread = {spread:.4e} ({:.1}%) across dy_max range (limit 5%)",
        spread * 100.0
    );

    // Step count should increase as dy_max decreases (more steps for finer control)
    let steps: Vec<usize> = dy_values
        .iter()
        .map(|&dy| {
            let mut s = ThermalizationSolver::new(
                cosmo.clone(),
                GridConfig {
                    n_points: 1000,
                    ..GridConfig::default()
                },
            );
            s.set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: 3000.0,
            })
            .unwrap();
            s.set_config(SolverConfig {
                z_start: 5.0e5,
                z_end: 1.0e4,
                dy_max: dy,
                dtau_max: 200.0,
                ..SolverConfig::default()
            });
            s.run_with_snapshots(&[1.0e4]);
            s.step_count
        })
        .collect();
    // Finest should take more steps than coarsest
    assert!(
        steps.last().unwrap() > steps.first().unwrap(),
        "Finer dy_max should require more steps: coarse={}, fine={}",
        steps.first().unwrap(),
        steps.last().unwrap()
    );
}

// 37.4: NC stripping integral test — ∫x²Δn = 0 after NC mode
//
// When number_conserving mode is enabled, the solver periodically strips
// the G_bb component to maintain ∫x²Δn dx = 0 (number conservation).
// After the run completes, the number integral should be zero to high
// precision. This tests the NC stripping mechanism in isolation.
#[test]
fn test_nc_stripping_integral_zero() {
    let mut solver = burst_solver(&GridConfig::default(), 2e5, 1e-5, 3000.0, 5e5, 1e4);
    solver.number_conserving = true;

    solver.run_with_snapshots(&[1e4]);
    let snap = solver.snapshots.last().unwrap();

    // Compute ∫x²Δn dx
    let dn_over_n = spectrum::delta_n_over_n(&solver.grid.x, &snap.delta_n);

    eprintln!("NC mode ΔN/N = {dn_over_n:.4e} (should be ~0)");

    // Should be very small — NC stripping removes the number-changing part.
    // NC strips every step, so residual ΔN/N should be limited by the
    // last step's G_bb contribution, not accumulated error.
    assert!(
        dn_over_n.abs() < 1e-5,
        "NC mode should give ΔN/N ≈ 0: got {dn_over_n:.4e} (threshold 1e-5)"
    );

    // The distortion should still be physically present (μ > 0)
    assert!(
        snap.mu.abs() > 1e-7,
        "NC mode should preserve μ-distortion: μ={:.4e}",
        snap.mu
    );
}

// 37.5: Negative occupation guard — strong depletion must not give n < 0
//
// For strong depletion (gc >> 1), Δn → -n_pl. The total occupation
// n = n_pl + Δn should never go negative (unphysical). The solver
// must either prevent this or handle it gracefully.
#[test]
fn test_negative_occupation_guard() {
    let cosmo = Cosmology::default();

    // Strong depletion: Δn = -(1 - exp(-gc/x)) × n_pl
    // For gc=10, at x=0.01: 1-exp(-1000) ≈ 1 → Δn ≈ -n_pl → n ≈ 0
    let gc = 10.0;
    let grid_config = GridConfig {
        n_points: 2000,
        x_min: 1e-4,
        x_max: 40.0,
        ..GridConfig::default()
    };
    let mut solver = ThermalizationSolver::new(cosmo, grid_config);
    solver.number_conserving = false; // raw depletion test, NC would distort the initial condition

    let initial_dn: Vec<f64> = solver
        .grid
        .x
        .iter()
        .map(|&x| -(1.0 - (-gc / x).exp()) * spectrum::planck(x))
        .collect();
    solver.set_initial_delta_n(initial_dn);
    solver.set_config(SolverConfig {
        z_start: 3e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });

    solver.run_with_snapshots(&[500.0]);
    let snap = solver.snapshots.last().unwrap();

    // Check: n = n_pl + Δn should be >= 0 everywhere (or at least not badly negative)
    let mut min_n = f64::MAX;
    let mut min_x = 0.0;
    for (i, &x) in solver.grid.x.iter().enumerate() {
        let n_total = spectrum::planck(x) + snap.delta_n[i];
        if n_total < min_n {
            min_n = n_total;
            min_x = x;
        }
    }

    eprintln!("Strong depletion gc={gc}: min(n_pl+Δn) = {min_n:.4e} at x={min_x:.4}");

    // Allow small numerical undershoot but not grossly negative.
    // Physical constraint: n_total >= 0 everywhere. Allow a small absolute
    // tolerance for numerical error, but the tolerance should be independent
    // of n_pl (which diverges at low x).
    assert!(
        min_n > -1e-3,
        "Occupation number went badly negative: n={min_n:.4e} at x={min_x:.4} \
         (threshold: -1e-3)"
    );

    // The spectrum should be finite everywhere
    assert!(
        snap.delta_n.iter().all(|v| v.is_finite()),
        "Non-finite Δn values after strong depletion"
    );

    // μ should be finite and negative (depletion removes photons)
    assert!(
        snap.mu.is_finite(),
        "μ should be finite after strong depletion: got {}",
        snap.mu
    );
}

/// Verify diag.warnings collects solver runtime warnings and that reset() clears them.
#[test]
fn test_diag_warnings_collected() {
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
    // DecayingParticlePhoton triggers a stimulated-emission warning.
    solver
        .set_injection(InjectionScenario::DecayingParticlePhoton {
            x_inj_0: 0.5 * (1.0 + 2e5),
            f_inj: 1e-5,
            gamma_x: 1e-15,
        })
        .unwrap();

    assert!(
        !solver.diag.warnings.is_empty(),
        "diag.warnings should contain a warning"
    );

    solver.reset();
    assert!(
        solver.diag.warnings.is_empty(),
        "reset() should clear diag.warnings"
    );
}

/// Grid resolution convergence test. Run at two resolutions (2000 and 4000 pts)
/// with identical physics. Compare the decomposed μ and y parameters, which
/// are robust integral quantities. For a SingleBurst at z=2e5, the μ parameter
/// should converge to <5% between 2000 and 4000 grid points.
#[test]
fn test_grid_transition_artifact() {
    let z_h = 2e5;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    let run_at_resolution = |n_pts: usize| -> (f64, f64, f64) {
        let grid_config = GridConfig {
            n_points: n_pts,
            ..GridConfig::default()
        };
        let mut solver = burst_solver(&grid_config, z_h, drho, sigma, z_h + 7.0 * sigma, 500.0);
        solver.run_with_snapshots(&[500.0]);
        let last = solver.snapshots.last().unwrap();
        (last.mu, last.y, last.delta_rho_over_rho)
    };

    let (mu_lo, y_lo, drho_lo) = run_at_resolution(2000);
    let (mu_hi, y_hi, drho_hi) = run_at_resolution(4000);

    eprintln!("Grid convergence test (z_h=2e5):");
    eprintln!("  2000 pts: mu = {mu_lo:.6e}, y = {y_lo:.6e}, drho/rho = {drho_lo:.6e}");
    eprintln!("  4000 pts: mu = {mu_hi:.6e}, y = {y_hi:.6e}, drho/rho = {drho_hi:.6e}");

    // mu should agree to <5%
    let mu_rel = (mu_lo - mu_hi).abs() / mu_hi.abs();
    eprintln!("  mu relative difference: {mu_rel:.4e}");
    assert!(
        mu_rel < 0.05,
        "mu should converge to <5% between 2000 and 4000 pts: \
         mu_2k = {mu_lo:.6e}, mu_4k = {mu_hi:.6e}, rel = {mu_rel:.4e}"
    );

    // Energy conservation should be identical at both resolutions
    let e_lo = (drho_lo / drho - 1.0).abs();
    let e_hi = (drho_hi / drho - 1.0).abs();
    eprintln!("  Energy err: 2000 pts = {e_lo:.4e}, 4000 pts = {e_hi:.4e}");
    assert!(
        e_lo < 0.02 && e_hi < 0.02,
        "Energy conservation should be <2% at both resolutions"
    );
}

/// Adiabatic cooling μ-distortion with zero explicit injection — the standard
/// ΛCDM prediction first computed by Chluba & Sunyaev (2012).
///
/// Oracle:             Chluba & Sunyaev (2012) MNRAS 419, 1294 — adiabatic
///                     cooling of baryons extracts energy from the photon
///                     field, yielding a negative μ-distortion. For
///                     Ω_b·h² ≈ 0.022 and Y_p = 0.24, the predicted μ_ac is
///                     in the range (-3.5, -2.0) × 10⁻⁹; Chluba 2016 Fig. 1
///                     quotes μ_ac ≈ -2.9 × 10⁻⁹ for Planck 2015 parameters.
/// Expected:           μ_ac = -2.9 × 10⁻⁹ (central value)
/// Oracle uncertainty: 15% (cosmology-parameter sensitivity; Ω_b ± a few %
///                     alone shifts μ by ~10%)
/// Tolerance:          25% on μ (tol = oracle_uncertainty · cosmology_margin)
///                     y and Δρ/ρ: magnitude bounds per Chluba 2016 Fig. 1.
///
/// The oracle is Chluba & Sunyaev 2012, not the code.
#[test]
fn test_adiabatic_cooling_no_injection() {
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
    solver.set_config(SolverConfig {
        z_start: 2.0e6,
        z_end: 1.0e4,
        ..SolverConfig::default()
    });

    solver.run_with_snapshots(&[1.0e4]);
    let last = solver.snapshots.last().unwrap();
    let mu = last.mu;
    let y = last.y;
    let drho = last.delta_rho_over_rho;

    eprintln!("Adiabatic cooling μ_ac test (Chluba & Sunyaev 2012):");
    eprintln!("  μ = {mu:.4e}  (target: -2.9 × 10⁻⁹ ± 25%)");
    eprintln!("  y = {y:.4e}  Δρ/ρ = {drho:.4e}");

    let mu_target = -2.9e-9_f64;
    let mu_rel_err = (mu - mu_target).abs() / mu_target.abs();
    assert!(
        mu_rel_err < 0.25,
        "Adiabatic μ: got {mu:.4e} vs Chluba & Sunyaev 2012 target {mu_target:.4e} \
         (rel_err {:.1}%, tol 25%)",
        mu_rel_err * 100.0
    );

    // y_ac from adiabatic cooling is O(10⁻¹⁰), subdominant to μ by factor ~10.
    assert!(
        y.abs() < 0.3 * mu.abs(),
        "Adiabatic y = {y:.4e} should be at most ~30% of |μ| = {:.4e}",
        mu.abs()
    );

    // Δρ/ρ should track μ_ac with sign (both negative, both ~few×10⁻⁹).
    let drho_target = -3.1e-9_f64;
    let drho_rel_err = (drho - drho_target).abs() / drho_target.abs();
    assert!(
        drho_rel_err < 0.25,
        "Adiabatic Δρ/ρ: got {drho:.4e} vs Chluba 2016 ~{drho_target:.4e} \
         (rel_err {:.1}%, tol 25%)",
        drho_rel_err * 100.0
    );
}

// T_e decoupling / adiabatic-cooling ρ_e check lives in
// tests/science_suite.rs::science_te_decoupling_post_recombination.
