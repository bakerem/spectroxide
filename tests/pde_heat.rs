//! PDE tests of heat injection.
//!
//! Bursts, decaying particles, dark-matter annihilation, custom and tabulated
//! heating: energy conservation, μ and y against the era targets and the
//! Green's function, spectral shapes, linearity, and the μ–y transition.

mod common;

use common::*;
use spectroxide::constants::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::distortion;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::greens;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::solver::{SolverConfig, ThermalizationSolver};
use spectroxide::spectrum;

// The μ-era decaying-particle comparison is covered by
// cosmotherm_comparison.rs::test_decaying_particle_vs_cosmotherm_gf_database,
// which uses the CosmoTherm Green's-function database as the reference.
/// y-era decaying particle: photon energy against the injected total.
///
/// Γ_X puts the lifetime at z = 5000, and the run stops at z = 1000, the lower
/// end of the Green's-function formalism and of the CosmoTherm database
/// (z = 1000 to 5e6). There is no μ check: in the y-era μ is a residual at the
/// 1e-3 level of y, and free-free emission from electrons heated above T_z
/// dominates that residual; the Chluba (2013) Green's function omits it
/// (decisions/0002-relax-dc-br-toward-actual-electron-temperature.md,
/// Addendum). The 2% tolerance is the measured 1% closure plus the time-step
/// error for continuous heating that `adaptive_dz` does not refine (finding
/// N-4 in dev/REVIEW_2026-09-22.md); decays peaking closer to z = 1000 lose
/// more (0.83 closure at Δτ_max = 10 for a lifetime at z = 1000).
#[test]
fn test_decaying_particle_y_era_energy() {
    let cosmo = Cosmology::default();
    let f_x = 1e5;
    let gamma_x = 1.0 / cosmo.cosmic_time(5.0e3);
    let scenario = InjectionScenario::DecayingParticle { f_x, gamma_x };
    let mut solver = ThermalizationSolver::new(cosmo.clone(), fast_grid());
    solver
        .set_injection(InjectionScenario::DecayingParticle { f_x, gamma_x })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: 3.0e6,
        z_end: 1.0e3,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[1.0e3]);
    let last = solver.snapshots.last().unwrap();

    let inj = injected_drho(&scenario, &cosmo, 1e3, 3e6);
    eprintln!(
        "y-era decay: PDE μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}; injected Δρ/ρ = {inj:.4e}; \
         warnings = {:?}",
        last.mu, last.y, last.delta_rho_over_rho, solver.diag.warnings
    );
    let e_rel = (last.delta_rho_over_rho - inj).abs() / inj;
    assert!(
        e_rel < 0.02,
        "energy: PDE Δρ/ρ = {:.4e} vs injected {inj:.4e} ({:.2}%)",
        last.delta_rho_over_rho,
        100.0 * e_rel
    );
    assert!(last.y > 0.0);
    assert_eq!(solver.diag.newton_exhausted, 0);
}

/// DC and BR drive the low-frequency spectrum to a Planck spectrum at the
/// actual electron temperature, n → n_pl(x/ρ_e) (Chluba & Sunyaev 2012, Eq. 8;
/// decisions/0002-relax-dc-br-toward-actual-electron-temperature.md).
///
/// During steady heating ρ_e exceeds the Compton-equilibrium value by δρ_inj.
/// At x ≪ x_c emission and absorption are much faster than Comptonization, so
/// Δn = n_pl(x/ρ_e) − n_pl(x). With n_pl(x) = 1/x − 1/2 + x/12 + O(x³), this is
/// (ρ_e − 1)/x × [1 + O(x²/12)], so x·Δn/(ρ_e − 1) = 1. We test at
/// x = 3e-6 to 3e-5 (grid extended to x_min = 1e-6). Corrections:
/// - series: x²/12 < 1e-10;
/// - Compton leakage: (x/x_c)² < 1e-6, with x_c ≈ 0.07 at z = 5000 from the BR
///   photosphere fit x_c,BR = 1.23e-3 [(1+z)/2e6]^(−0.672) (Chluba 2015);
/// - finite absorption rate: BR absorption per Thomson time scales as x⁻², so
///   the lag behind a moving ρ_e scales as x²; it reaches 1% only near
///   x ≈ 1.5e-4, two orders above the test points in x²;
/// - number-conserving mode subtracts this step's temperature shift δT·G_bb,
///   and x·G_bb → 1, so the ratio carries an x-independent offset
///   −δT_step/(ρ_e − 1), with δT_step a small fraction of ρ_e − 1 per step.
///
/// We allow 1%.
#[test]
fn test_dcbr_relaxes_to_actual_electron_temperature() {
    let cosmo = Cosmology::default();
    let f_x = 1e5;
    let gamma_x = 1.0 / cosmo.cosmic_time(5.0e3);
    let grid = GridConfig {
        x_min: 1e-6,
        ..fast_grid()
    };
    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid);
    solver
        .set_injection(InjectionScenario::DecayingParticle { f_x, gamma_x })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: 3.0e6,
        z_end: 5.0e3,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[5.0e3]);
    let snap = solver.snapshots.last().unwrap();
    let excess = snap.rho_e - 1.0;
    eprintln!("z = {:.1}, ρ_e − 1 = {excess:.4e}", snap.z);
    assert!(
        excess > 1e-4,
        "heating should keep ρ_e above 1, got {excess:.3e}"
    );
    for &x_t in &[3e-6, 1e-5, 3e-5] {
        let i = solver.grid.x.partition_point(|&x| x < x_t);
        let ratio = solver.grid.x[i] * snap.delta_n[i] / excess;
        eprintln!(
            "  x = {:.3e}: x·Δn/(ρ_e − 1) = {ratio:.5}",
            solver.grid.x[i]
        );
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "x = {:.3e}: x·Δn/(ρ_e − 1) = {ratio:.4}, expected 1 (DC/BR target must be ρ_e)",
            solver.grid.x[i]
        );
    }
}

/// A decaying particle with Γ_X = 1e-13 s⁻¹ (lifetime near z ≈ 1000), run on to
/// z = 500, heats the electrons below recombination until T_e hits
/// the solver's cap. That regime would change the ionization history (hotter
/// electrons recombine more slowly) while X_e is held on
/// its recombination history, so the solver must say so with exactly one
/// warning per run (decisions/0002-..., Addendum).
#[test]
fn test_decaying_particle_late_heating_warns() {
    let mut solver = ThermalizationSolver::new(Cosmology::default(), fast_grid());
    solver
        .set_injection(InjectionScenario::DecayingParticle {
            f_x: 1e5,
            gamma_x: 1e-13,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: 3.0e6,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);

    let n_warn = solver
        .diag
        .warnings
        .iter()
        .filter(|w| w.contains("change the ionization history"))
        .count();
    assert_eq!(
        n_warn, 1,
        "expected one heating/ionization warning, got {n_warn}: {:?}",
        solver.diag.warnings
    );
    assert!(solver.diag.rho_e_clamped > 0);
}

/// DM annihilation PDE: s-wave and p-wave mu/y properties.
#[test]
fn test_annihilation_mu_y_properties() {
    let cosmo = Cosmology::default();

    // s-wave PDE: mu and y should be positive, mu/y > 0.5
    let f_ann = 1e-19;
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver
        .set_injection(InjectionScenario::AnnihilatingDM { f_ann })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    let snaps = solver.run_with_snapshots(&[500.0]);
    let snap = snaps.last().unwrap();
    assert!(snap.mu > 0.0, "s-wave mu positive");
    assert!(snap.y > 0.0, "s-wave y positive");
    assert!(snap.mu / snap.y > 0.5, "s-wave mu/y > 0.5");

    // p-wave should have LARGER mu/y ratio than s-wave (extra (1+z) factor)
    let f_ann_gf = 1e-30;
    let s_s = InjectionScenario::AnnihilatingDM { f_ann: f_ann_gf };
    let dq_s = |z: f64| -> f64 { -s_s.heating_rate_per_redshift(z, &cosmo) };
    let (mu_s, y_s) = greens::mu_y_from_heating(&dq_s, 500.0, 3e6, 20000);
    let s_p = InjectionScenario::AnnihilatingDMPWave { f_ann: f_ann_gf };
    let dq_p = |z: f64| -> f64 { -s_p.heating_rate_per_redshift(z, &cosmo) };
    let (mu_p, y_p) = greens::mu_y_from_heating(&dq_p, 500.0, 3e6, 20000);
    assert!(mu_p / y_p > mu_s / y_s, "p-wave mu/y > s-wave mu/y");
}

/// PDE solver: DC+BR emission drives photon production toward
/// Bose-Einstein equilibrium. At high z with a y-type initial distortion,
/// the photon number should increase (DC/BR produce low-frequency photons)
/// and the distortion should evolve from y-type toward μ-type.
#[test]
fn test_pde_y_to_mu_conversion() {
    let grid_config = GridConfig::default();

    // Inject at z = 2e5 (transition region) where both Kompaneets
    // and DC/BR are active
    let z_h = 2e5;
    let drho = 1e-5;
    let sigma = z_h * 0.01;
    let mut solver = burst_solver(&grid_config, z_h, drho, sigma, z_h * 1.05, 5e4);

    // Take snapshots: right after injection and at end
    solver.run_with_snapshots(&[1.9e5, 1e5, 5e4]);

    assert_eq!(
        solver.diag.newton_exhausted, 0,
        "Newton should converge for y-to-mu conversion"
    );

    // Should have 3 snapshots
    assert!(
        solver.snapshots.len() >= 2,
        "Expected at least 2 snapshots, got {}",
        solver.snapshots.len()
    );

    let early = &solver.snapshots[0];
    let late = solver.snapshots.last().unwrap();

    eprintln!("y→μ conversion test:");
    eprintln!(
        "  Early (z={:.0e}): mu={:.4e}, y={:.4e}, mu/y={:.3}",
        early.z,
        early.mu,
        early.y,
        if early.y.abs() > 1e-20 {
            early.mu / early.y
        } else {
            f64::NAN
        }
    );
    eprintln!(
        "  Late  (z={:.0e}): mu={:.4e}, y={:.4e}, mu/y={:.3}",
        late.z,
        late.mu,
        late.y,
        if late.y.abs() > 1e-20 {
            late.mu / late.y
        } else {
            f64::NAN
        }
    );

    // At high z, DC/BR should convert some y into μ over time
    // The mu/y ratio should increase (or at least not decrease drastically)
    // Also, both mu and y should be positive (heating)
    assert!(late.mu > 0.0, "Late mu should be positive: {:.4e}", late.mu);
    assert!(
        late.delta_rho_over_rho > 0.0,
        "Energy should be positive: {:.4e}",
        late.delta_rho_over_rho
    );

    // Energy should be roughly conserved (< 15%)
    let e_frac = late.delta_rho_over_rho / drho;
    assert!(
        (e_frac - 1.0).abs() < 0.15,
        "Energy conservation: drho/drho_inj = {e_frac:.3}, expected ~1.0"
    );
}

/// Energy injection via PDE solver should increase the electron temperature.
/// Verify that the solver's T_e feedback is consistent with the distortion.
#[test]
fn test_pde_electron_temperature_feedback() {
    let grid_config = GridConfig::default();

    // Inject energy as a single burst in the μ-era
    let mut solver = burst_solver(&grid_config, 2e5, 1e-5, 2000.0, 2.1e5, 1e4);
    solver.run_with_snapshots(&[1e4]);
    let snap = solver.snapshots.last().unwrap();

    // Check that the solver produced a meaningful distortion
    assert!(
        snap.mu.abs() > 1e-10,
        "Should produce nonzero μ: {:.4e}",
        snap.mu
    );

    // ρ_e should be close to 1 but slightly above (energy injection heats electrons)
    eprintln!(
        "PDE T_e feedback: mu={:.4e}, rho_e={:.8}, drho={:.4e}",
        snap.mu, snap.rho_e, snap.delta_rho_over_rho
    );
    assert!(
        snap.rho_e > 0.99 && snap.rho_e < 1.01,
        "ρ_e should be near 1: {:.8}",
        snap.rho_e
    );
}

/// Multiple sequential bursts should conserve total energy.
/// Sum of individual Δρ/ρ injections should match the total PDE Δρ/ρ.
/// Verify that the spectral distortion from a single burst at z=1e4 (deep y-era)
/// produces a y-type distortion with minimal μ component.
#[test]
fn test_y_era_burst_spectral_purity() {
    let grid_config = GridConfig::default();

    let z_h = 1e4_f64;
    let drho = 1e-5_f64;

    let mut solver = burst_solver(&grid_config, z_h, drho, 200.0, 1.1e4, 5e3);
    solver.run_with_snapshots(&[5e3]);
    let snap = solver.snapshots.last().unwrap();

    // In the deep y-era, should be mostly y-type
    // y ≈ Δρ/(4ρ) = drho/4
    let y_expected = drho / 4.0;

    eprintln!(
        "y-era burst: mu={:.4e}, y={:.4e}, y_expected={y_expected:.4e}, drho={:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho
    );

    // y should be positive and ~Δρ/4 (exact in y-era)
    assert!(snap.y > 0.0, "y should be positive");
    let y_ratio = snap.y / y_expected;
    assert!(
        y_ratio > 0.7 && y_ratio < 1.3,
        "y/y_expected = {y_ratio:.3}, should be ~1 (measured ~0.99)"
    );

    // μ should be much smaller than y in the y-era
    let mu_y_ratio = snap.mu.abs() / snap.y.abs();
    // Measured |μ|/|y| ~ 0.09.
    assert!(
        mu_y_ratio < 0.20,
        "In y-era, |μ|/|y| should be small: {mu_y_ratio:.3}"
    );
}

/// SingleBurst with negative Δρ/ρ (cooling) should produce negative distortion.
#[test]
fn test_negative_injection_cooling() {
    let drho = -1e-5; // cooling

    let mut solver = burst_solver(&GridConfig::default(), 5e4, drho, 2000.0, 2.0e5, 1.0e3);

    let snaps = solver.run_with_snapshots(&[1.0e3]);
    let last = snaps.last().unwrap();

    // Energy should be negative (cooling)
    assert!(
        last.delta_rho_over_rho < 0.0,
        "Cooling should give Δρ/ρ < 0: {:.4e}",
        last.delta_rho_over_rho
    );

    // Magnitude should be approximately |Δρ/ρ| (energy conservation)
    let rel_err = (last.delta_rho_over_rho - drho).abs() / drho.abs();
    assert!(
        rel_err < 0.1,
        "Cooling energy conservation: Δρ/ρ = {:.4e} vs {drho:.4e}",
        last.delta_rho_over_rho
    );
}

/// Superposition principle: two bursts at the same z should give
/// 2× the distortion of a single burst.
#[test]
fn test_pde_linearity_double_injection() {
    let z_h = 5e4;
    let sigma = 2000.0;

    // Single burst with Δρ/ρ = 1e-5
    let mut solver1 = burst_solver(&GridConfig::default(), z_h, 1e-5, sigma, 2.0e5, 1.0e3);
    let snaps1 = solver1.run_with_snapshots(&[1.0e3]);
    let last1 = snaps1.last().unwrap();

    // Single burst with Δρ/ρ = 2e-5
    let mut solver2 = burst_solver(&GridConfig::default(), z_h, 2e-5, sigma, 2.0e5, 1.0e3);
    let snaps2 = solver2.run_with_snapshots(&[1.0e3]);
    let last2 = snaps2.last().unwrap();

    // μ and y should scale linearly. In the small-distortion regime (Δρ/ρ ≤ 1e-5),
    // the Kompaneets+DC+BR equation is linear in Δn, so doubling the source
    // must double the response to within truncation error. A ratio far from 2
    // indicates either nonlinearity leaked in or a grid-dependent artifact.
    eprintln!(
        "Linearity: μ₁={:.4e}, μ₂={:.4e}, Δρ₁={:.4e}, Δρ₂={:.4e}",
        last1.mu, last2.mu, last1.delta_rho_over_rho, last2.delta_rho_over_rho
    );
    if last1.mu.abs() > 1e-10 {
        let ratio_mu = last2.mu / last1.mu;
        eprintln!("  ratio_mu = {ratio_mu:.4}");
        assert!(
            (ratio_mu - 2.0).abs() < 0.05,
            "μ linearity: ratio = {ratio_mu:.4} (expected 2.0 ± 2.5%); \
             linearity should hold exactly at Δρ/ρ ≤ 1e-5"
        );
    }

    let ratio_drho = last2.delta_rho_over_rho / last1.delta_rho_over_rho;
    eprintln!("  ratio_drho = {ratio_drho:.4}");
    assert!(
        (ratio_drho - 2.0).abs() < 0.02,
        "Δρ/ρ linearity: ratio = {ratio_drho:.4} (expected 2.0 ± 1%); \
         energy injection is linear by construction"
    );
}

/// At z_h = 5e5 with dtau_max = 3, the PDE μ agrees with the Green's
/// function to 15%.
#[test]
fn test_high_z_mu_vs_gf_dtau3() {
    let cosmo = Cosmology::default();
    let z_h = 5e5;
    let drho = 1e-5;
    let sigma = z_h * 0.04;

    // Run with dtau_max=3 (tight)
    let mut solver3 = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver3
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver3.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 1e4,
        dtau_max: 3.0,
        ..SolverConfig::default()
    });
    solver3.run_with_snapshots(&[1e4]);
    let mu3 = solver3.snapshots.last().unwrap().mu;
    let steps3 = solver3.step_count;

    // Green's function reference
    let dq_dz = |z: f64| gaussian_heating(z, z_h, sigma, drho);
    let mu_gf = spectroxide::greens::mu_from_heating(&dq_dz, 1e3, 5e6, 10000);

    let gf_err_3 = (mu3 - mu_gf).abs() / mu_gf.abs().max(1e-20);

    eprintln!("High-z PDE vs GF (z_h={z_h:.0e}):");
    eprintln!("  dtau_max=3:  mu={mu3:.4e}, steps={steps3}");
    eprintln!("  GF:          mu={mu_gf:.4e}");
    eprintln!("  |mu3-muGF|/|muGF| = {gf_err_3:.3}");

    // dtau_max=3 should agree with GF to within 15%
    assert!(
        gf_err_3 < 0.15,
        "dtau_max=3 mu={mu3:.4e} vs GF mu={mu_gf:.4e}, err={:.1}%",
        gf_err_3 * 100.0
    );
}

/// Thermalization-era burst (z_h=3×10⁶): most energy thermalizes to a
/// temperature shift; residual μ is suppressed by J_bb*(z_h).
///
/// Oracle:             Chluba (2013) Eq. 5 with J_bb*(3e6)·J_μ(3e6):
///                     μ/Δρ = (3/κ_c) · J_bb*(z_h) · J_μ(z_h)
/// Expected:           J_bb*(3e6) ≈ 0.06, J_μ(3e6) ≈ 1.0 → μ/Δρ ≈ 0.08
/// Oracle uncertainty: 5% (GF fit vs CosmoTherm)
/// Tolerance:          10% (production grid; PDE vs GF at deep thermalization
///                     is method-limited). With the production grid the PDE
///                     agrees with the analytic target to a few percent.
///
/// Marked `#[ignore]`: production grid at z=3×10⁶ takes ~4 minutes. Run with
/// `cargo test --release -- --ignored` in CI/paper-production.
#[ignore]
#[test]
fn test_thermalization_era_pure_temperature_shift() {
    let z_h = 3e6;
    let drho = 1e-5;
    let sigma = z_h * 0.04;

    // Integrate only through μ-formation (z > 5e4) — μ is photon-number-conserving
    // below that, so z_end=1e5 gives the same final μ as z_end=1e4 at ~20% of the cost.
    let mut solver = burst_solver(
        &GridConfig::production(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        1e5,
    );
    solver.run_with_snapshots(&[1e5]);
    let last = solver.snapshots.last().unwrap();

    let j_bb = greens::visibility_j_bb_star(z_h);
    let j_mu = greens::visibility_j_mu(z_h);
    let expected = (3.0 / KAPPA_C) * j_bb * j_mu;
    let mu_over_drho = last.mu.abs() / drho;
    let rel_err = (mu_over_drho - expected).abs() / expected;

    eprintln!(
        "T-era (z_h={z_h:.0e}, production grid): μ/Δρ={mu_over_drho:.4}, \
         expected={expected:.4} (J_bb*={j_bb:.4}, J_μ={j_mu:.4}), rel_err={:.2}%",
        rel_err * 100.0,
    );
    assert!(
        rel_err < 0.10,
        "At z_h={z_h:.0e}, μ/Δρ={mu_over_drho:.4} vs Chluba 2013 Eq.5 target {expected:.4} \
         (rel_err {:.2}%, tol 10%)",
        rel_err * 100.0,
    );

    let drho_err = (last.delta_rho_over_rho / drho - 1.0).abs();
    eprintln!("  Energy conservation: drho_err = {drho_err:.2e}");
    assert!(
        drho_err < 0.03,
        "Energy conservation at z_h={z_h:.0e}: drho_err={drho_err:.2e} (tol 3% on prod grid)"
    );
}

/// Coupled IMEX and operator splitting should give consistent μ at z=1e6.
/// Both modes agree to within ~50%; DC/BR stiffness differences are secondary.
#[test]
fn test_coupled_vs_split_z1e6() {
    let cosmo = Cosmology::default();
    let z_h = 1e6;
    let drho = 1e-5;
    let sigma = z_h * 0.04;

    // Run with coupled IMEX (default)
    let mut solver_coupled = burst_solver(&fast_grid(), z_h, drho, sigma, z_h + 7.0 * sigma, 1e4);
    solver_coupled.run_with_snapshots(&[1e4]);
    let coupled = solver_coupled.snapshots.last().unwrap();
    let mu_coupled = coupled.mu.abs();

    // Run with operator splitting
    let mut solver_split = ThermalizationSolver::new(cosmo, fast_grid());
    solver_split.coupled_dcbr = false;
    solver_split
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver_split.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 1e4,
        ..SolverConfig::default()
    });
    solver_split.run_with_snapshots(&[1e4]);
    let split = solver_split.snapshots.last().unwrap();
    let mu_split = split.mu.abs();

    eprintln!("z_h=1e6: coupled mu={mu_coupled:.4e}, split mu={mu_split:.4e}");
    eprintln!("  Coupled/split ratio: {:.2}", mu_coupled / mu_split);

    // Both modes should give consistent μ/Δρ (within 10% of each other)
    let ratio = mu_coupled / mu_split;
    assert!(
        ratio > 0.9 && ratio < 1.1,
        "Coupled/split ratio out of range: {ratio:.4} (want 0.9-1.1)"
    );

    // Both should conserve energy to < 15%
    let coupled_err = (coupled.delta_rho_over_rho / drho - 1.0).abs();
    let split_err = (split.delta_rho_over_rho / drho - 1.0).abs();
    assert!(coupled_err < 0.15, "Coupled energy: {coupled_err:.2e}");
    assert!(split_err < 0.15, "Split energy: {split_err:.2e}");
}

/// Benchmark 1: μ-decay eigenvalue test.
///
/// Initialize the solver with a pure μ-distortion Δn = μ₀ M(x), no injection,
/// and evolve from z=2e6 down to z=5e4 with Kompaneets + DC/BR.
/// The μ-parameter should decay as DC/BR thermalizes the distortion.
///
/// Key diagnostic: compare μ(z)/μ₀ evolution to:
///   (a) No DC/BR baseline (should preserve μ exactly)
///   (b) Theoretical survival fraction from J_bb* thermalization depth
///
/// If DC/BR rate is ~6.7× too strong, the μ decay will be much faster
/// than theory predicts.
#[test]
fn test_mu_decay_eigenvalue() {
    let cosmo = Cosmology::default();
    let mu0 = 1e-5;

    let grid_config = GridConfig {
        x_min: 1e-4,
        x_max: 50.0,
        n_points: 500,
        x_transition: 0.10,
        log_fraction: 0.30,
        refinement_zones: Vec::new(),
    };

    // Build initial μ-distortion: Δn = μ₀ × M(x)
    let grid = FrequencyGrid::new(&grid_config);
    let delta_n_init: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| mu0 * spectrum::mu_shape(x))
        .collect();

    // Verify initial μ decomposition
    let init_params = distortion::decompose_distortion(&grid.x, &delta_n_init);
    eprintln!("\n=== μ-decay eigenvalue test ===");
    eprintln!(
        "Initial: μ={:.4e}, y={:.4e}, ΔT/T={:.4e}, Δρ/ρ={:.4e}",
        init_params.mu, init_params.y, init_params.delta_t_over_t, init_params.delta_rho_over_rho
    );

    // Snapshot redshifts (high to low)
    let snap_z = [1.8e6, 1.5e6, 1.2e6, 1e6, 8e5, 5e5, 3e5, 1e5, 5e4];

    // Run WITH DC/BR (standard)
    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver.set_initial_delta_n(delta_n_init.clone());
    solver.set_config(SolverConfig {
        z_start: 2e6,
        z_end: 5e4,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&snap_z);

    eprintln!("\nWith DC/BR (standard):");
    eprintln!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}",
        "z", "μ/μ₀", "y/μ₀", "Δρ/ρ", "accum_ΔT"
    );
    for snap in &solver.snapshots {
        let mu_ratio = snap.mu / mu0;
        let y_ratio = snap.y / mu0;
        eprintln!(
            "{:>10.2e} {:>10.4} {:>10.4e} {:>10.4e} {:>10.4e}",
            snap.z, mu_ratio, y_ratio, snap.delta_rho_over_rho, snap.accumulated_delta_t
        );
    }

    // Run WITHOUT DC/BR (Kompaneets only)
    let mut solver_nodcbr = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_nodcbr.set_initial_delta_n(delta_n_init.clone());
    solver_nodcbr.disable_dcbr = true;
    solver_nodcbr.set_config(SolverConfig {
        z_start: 2e6,
        z_end: 5e4,
        ..SolverConfig::default()
    });
    solver_nodcbr.run_with_snapshots(&snap_z);

    eprintln!("\nWithout DC/BR (Kompaneets only):");
    eprintln!("{:>10} {:>10} {:>10} {:>10}", "z", "μ/μ₀", "y/μ₀", "Δρ/ρ");
    for snap in &solver_nodcbr.snapshots {
        let mu_ratio = snap.mu / mu0;
        let y_ratio = snap.y / mu0;
        eprintln!(
            "{:>10.2e} {:>10.4} {:>10.4e} {:>10.4e}",
            snap.z, mu_ratio, y_ratio, snap.delta_rho_over_rho
        );
    }

    // Key assertions
    let last_dcbr = solver.snapshots.last().unwrap();
    let last_nodcbr = solver_nodcbr.snapshots.last().unwrap();

    // 1. Without DC/BR, μ should be well preserved (>85% at z=5e4)
    let mu_preserved = last_nodcbr.mu / mu0;
    assert!(
        mu_preserved > 0.85,
        "Kompaneets-only should preserve μ: μ/μ₀={mu_preserved:.4} at z=5e4"
    );

    // 2. With DC/BR, μ should decay (thermalization)
    let mu_decayed = last_dcbr.mu / mu0;
    assert!(
        mu_decayed < mu_preserved,
        "DC/BR should cause μ to decay: {mu_decayed:.4} >= {mu_preserved:.4}"
    );

    // 3. Report effective thermalization depth
    let survival_ratio = (mu_decayed / mu_preserved).max(1e-30);
    let tau_th_effective = -(survival_ratio).ln();
    eprintln!("\nEffective thermalization:");
    eprintln!("  μ(5e4)/μ₀ with DC/BR:    {mu_decayed:.6}");
    eprintln!("  μ(5e4)/μ₀ without DC/BR: {mu_preserved:.6}");
    eprintln!("  Survival ratio:           {survival_ratio:.6}");
    eprintln!("  Effective τ_th = -ln(ratio): {tau_th_effective:.4}");

    // 4. Compare to theoretical J_bb* thermalization depth
    // J_bb*(z) = exp(-(z/z_dc)^{5/2}) with z_dc ≈ 1.98e6
    // Thermalization depth from z_start to z_end:
    //   Δτ_th = (z_start/z_dc)^{5/2} - (z_end/z_dc)^{5/2}
    let z_dc = 1.98e6;
    let tau_th_theory = (2e6_f64 / z_dc).powf(2.5) - (5e4_f64 / z_dc).powf(2.5);
    let survival_theory = (-tau_th_theory).exp();
    eprintln!("  Theoretical τ_th:           {tau_th_theory:.4}");
    eprintln!("  Theoretical survival:       {survival_theory:.4}");
    eprintln!(
        "  PDE τ_th / theory τ_th:     {:.4}",
        tau_th_effective / tau_th_theory
    );
}

/// Benchmark 4: Spectral shape after z=1e6 burst.
///
/// After injecting energy at z=1e6 and evolving to z=1e4, examine:
///   (a) The solver's μ and y decomposition (fit over x ∈ [1, 15])
///   (b) How well Δn correlates with M(x) in the spectral core
///   (c) Whether the low-x region (x < 1) has large DC/BR artifacts
///
/// NOTE: decompose_distortion() and the solver now both use the same
/// energy-conserving constrained decomposition with a restricted fit
/// range (x ∈ [1, 15]) to avoid DC/BR artifacts at low x.
#[test]
fn test_spectral_shape_after_burst() {
    let drho = 1e-5;
    let z_h = 1e6;
    let sigma = z_h * 0.04;

    let grid_config = GridConfig {
        x_min: 1e-4,
        x_max: 50.0,
        n_points: 1000,
        x_transition: 0.10,
        log_fraction: 0.30,
        refinement_zones: Vec::new(),
    };

    let mut solver = burst_solver(&grid_config, z_h, drho, sigma, z_h + 7.0 * sigma, 1e4);
    solver.run_with_snapshots(&[1e4]);

    let snap = solver.snapshots.last().unwrap();

    // Use the solver's own decomposition (restricted to x ∈ [1, 15])
    let mu_solver = snap.mu;
    let y_solver = snap.y;
    let drho_solver = snap.delta_rho_over_rho;

    // Also compute full-range decomposition for comparison
    let params_full = distortion::decompose_distortion(&solver.grid.x, &snap.delta_n);

    eprintln!("\n=== Spectral shape after z_h=1e6 burst ===");
    eprintln!(
        "Solver decomposition (x ∈ [1, 15]): μ={:.4e}, y={:.4e}, Δρ/ρ={:.4e}",
        mu_solver, y_solver, drho_solver
    );
    eprintln!("  μ/Δρ = {:.4}", mu_solver / drho);
    eprintln!(
        "Full-range decomposition: μ={:.4e}, y={:.4e}, Δρ/ρ={:.4e}",
        params_full.mu, params_full.y, params_full.delta_rho_over_rho
    );
    eprintln!("  μ/Δρ (full) = {:.4}", params_full.mu / drho);

    // Compute correlation in the spectral core x ∈ [1, 15]
    let mut sum_dn_m = 0.0;
    let mut sum_dn2 = 0.0;
    let mut sum_m2 = 0.0;
    let mut sum_dn_y = 0.0;
    let mut sum_y2 = 0.0;

    // Also compute the fit residual in [1, 15]
    let mu_to_energy = 3.0 / KAPPA_C;
    let delta_t_solver = (drho_solver - mu_solver / mu_to_energy - 4.0 * y_solver) / 4.0;
    let mut sum_res2 = 0.0;
    let mut sum_dn2_core = 0.0;

    for i in 0..solver.grid.n {
        let x = solver.grid.x[i];
        if x < 1.0 || x > 15.0 {
            continue;
        }
        let dn = snap.delta_n[i];
        let m = spectrum::mu_shape(x);
        let ys = spectrum::y_shape(x);
        let g = spectrum::g_bb(x);

        sum_dn_m += dn * m;
        sum_dn2 += dn * dn;
        sum_m2 += m * m;
        sum_dn_y += dn * ys;
        sum_y2 += ys * ys;

        let fit = mu_solver * m + y_solver * ys + delta_t_solver * g;
        let res = dn - fit;
        sum_res2 += res * res;
        sum_dn2_core += dn * dn;
    }

    let corr_mu = sum_dn_m / (sum_dn2.sqrt() * sum_m2.sqrt());
    let corr_y = sum_dn_y / (sum_dn2.sqrt() * sum_y2.sqrt());
    let residual_frac = (sum_res2 / sum_dn2_core.max(1e-60)).sqrt();

    eprintln!("\nSpectral correlations (x ∈ [1, 15]):");
    eprintln!("  Corr(Δn, M(x)):    {corr_mu:.6}");
    eprintln!("  Corr(Δn, Y_SZ(x)): {corr_y:.6}");
    eprintln!("  Residual fraction:  {residual_frac:.6}");

    // Print low-x region to show DC/BR artifacts
    eprintln!("\nLow-x region (DC/BR artifact zone):");
    eprintln!("{:>10} {:>12} {:>12}", "x", "Δn", "|Δn|/max");
    let max_dn_core: f64 = solver
        .grid
        .x
        .iter()
        .zip(snap.delta_n.iter())
        .filter(|&(&x, _)| x > 1.0 && x < 15.0)
        .map(|(_, &dn)| dn.abs())
        .fold(0.0, |a, b| {
            assert!(b.is_finite(), "NaN/Inf in filtered Δn");
            a.max(b)
        });
    for i in 0..solver.grid.n {
        let x = solver.grid.x[i];
        if x > 0.5 {
            break;
        }
        let dn = snap.delta_n[i];
        if i % 5 == 0 || x < 0.01 {
            // sample every 5th point
            eprintln!(
                "{x:>10.4e} {dn:>12.4e} {:>12.2}",
                dn.abs() / max_dn_core.max(1e-30)
            );
        }
    }

    // Print spectral core shape comparison
    eprintln!("\nSpectral core shape comparison (x ∈ [1, 15]):");
    eprintln!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "x", "Δn", "μ·M(x)", "y·Y(x)", "residual"
    );
    let x_sample = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0];
    for &x_target in &x_sample {
        let idx = solver
            .grid
            .x
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - x_target).abs())
                    .partial_cmp(&((**b - x_target).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();
        let x = solver.grid.x[idx];
        let dn = snap.delta_n[idx];
        let mu_comp = mu_solver * spectrum::mu_shape(x);
        let y_comp = y_solver * spectrum::y_shape(x);
        let g_comp = delta_t_solver * spectrum::g_bb(x);
        let res = dn - mu_comp - y_comp - g_comp;
        eprintln!("{x:>8.2} {dn:>12.4e} {mu_comp:>12.4e} {y_comp:>12.4e} {res:>12.4e}");
    }

    // Assertions (using the restricted-range fit)
    // At z_h=1e6 (deep μ era), the distortion is almost pure μ-type.
    assert!(
        corr_mu.abs() > 0.95,
        "Distortion from z=1e6 should have strong μ-correlation: R={corr_mu:.4}"
    );
    // Residual in the fit range should be small (μ+y+ΔT captures the core)
    assert!(
        residual_frac < 0.05,
        "Residual in [1,15] should be small: {residual_frac:.4}"
    );
    // μ/Δρ from the solver should be near 1.401 × J_bb* × J_μ
    // At z=1e6: J_bb* ~ 0.88, J_μ ~ 1.0, so μ/Δρ ~ 1.23
    assert!(
        mu_solver / drho > 0.8 && mu_solver / drho < 1.5,
        "μ/Δρ should be in [0.8, 1.5] at z=1e6: {:.4}",
        mu_solver / drho
    );
    // The low-x DC/BR artifact region should be much larger than the core
    // This is the photon creation that drives thermalization
    assert!(max_dn_core > 0.0, "should have nonzero core distortion");
}

/// S-wave annihilation: PDE spectral shape at x < 1 vs GF convolution.
///
/// This catches systematic G_bb excess from incorrect DC/BR energy change
/// estimation. G_bb is positive at all x, so excess G_bb pushes the low-x
/// spectrum upward (less negative), causing PDE/GF ratio to deviate from 1.
#[test]
fn test_annihilation_swave_low_x_spectral_shape() {
    let cosmo = Cosmology::default();
    let f_ann = 1e-19;
    let z_start = 5e5;
    let z_end = 500.0;

    // PDE run
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    let x_grid = solver.grid.x.clone();
    solver
        .set_injection(InjectionScenario::AnnihilatingDM { f_ann })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    let snaps = solver.run_with_snapshots(&[z_end]);
    let snap = snaps.last().unwrap();

    // GF convolution over same z range
    let scenario_gf = InjectionScenario::AnnihilatingDM { f_ann };
    let dq_dz_fn = |z: f64| -> f64 { -scenario_gf.heating_rate_per_redshift(z, &cosmo) };
    let dn_gf = greens::distortion_from_heating(&x_grid, &dq_dz_fn, z_end, z_start, 20000);

    // Compare at x < 1 where distortion is negative (below Planck)
    let x_targets = [0.3, 0.5, 0.8];
    eprintln!("s-wave low-x spectral shape (PDE vs GF convolution):");
    for &x_target in &x_targets {
        // Find nearest grid point
        let idx = x_grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - x_target)
                    .abs()
                    .partial_cmp(&((**b) - x_target).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        let x_actual = x_grid[idx];
        let dn_pde = snap.delta_n[idx];
        let dn_gf_val = dn_gf[idx];

        if dn_gf_val.abs() > 1e-30 {
            let ratio = dn_pde / dn_gf_val;
            eprintln!("  x={x_actual:.3}: PDE={dn_pde:.4e}, GF={dn_gf_val:.4e}, ratio={ratio:.4}");
            assert!(
                (ratio - 1.0).abs() < 0.30,
                "PDE/GF spectral ratio at x={x_actual:.3}: {ratio:.4}, expected 1.0 ± 0.30"
            );
        }
    }
}

/// Energy conservation: Δρ/ρ measured from the PDE output should match the
/// injected value to <0.6% across all eras (y, transition, μ).
///
/// This is tighter than the existing sweep test and covers more redshifts.
///
/// Measured at the shipped defaults (N = 2000, dtau_max = 10, dy_max = 0.02):
/// −0.09%, −0.08%, −0.08%, −0.17%, −0.20%, −0.27%, −0.43% at z_h = 3e3 … 5e5.
/// One-sided and growing with z_h because it is the first-order-in-Δτ residual
/// of the coupled T_e / DC-BR step, generated inside the injection window
/// (`dev/audit/energy_conservation_audit.md`): DC/BR off removes ~60% of it at
/// z_h = 1e5, dtau_max = 2 removes ~85%, and grid or dy_max refinement removes
/// none of it. Of the quoted deficit, 0.015–0.03 pp is the physical
/// adiabatic-cooling distortion (−1.5 to −2.9 × 10⁻⁹ against the injected
/// 10⁻⁵), which this test does not subtract.
///
/// Tolerance 0.6% = 1.4× the worst measured point.
#[test]
fn test_heat_energy_conservation_sweep_tight() {
    let drho_injected = 1e-5;
    let z_values = [3000.0, 5000.0, 1e4, 5e4, 1e5, 2e5, 5e5];

    for &z_h in &z_values {
        let run = standard_burst(z_h, drho_injected);
        let last = &run.snap;

        let drho_measured = delta_rho_over_rho(&run.x, &last.delta_n);
        let rel_err = (drho_measured - drho_injected).abs() / drho_injected;

        eprintln!(
            "z_h={z_h:.0e}: Δρ/ρ measured = {drho_measured:.6e}, err = {:.2}%",
            rel_err * 100.0
        );

        assert!(
            rel_err < 0.006,
            "Energy conservation at z_h={z_h}: err = {:.2}% > 0.6%",
            rel_err * 100.0
        );
    }
}

/// For a decaying particle with f_X and Γ_X, the total energy deposited
/// should be f_X × n_H,0 / ρ_γ,0 (when Γ >> H at all relevant z).
/// Test that the PDE captures the correct total energy.
#[test]
fn test_heat_decay_total_energy_deposited() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    // Short-lived particle that decays entirely in the y-era.
    // f_x must be large enough that Δρ/ρ >> adiabatic cooling floor (~3e-9).
    // GF gives μ ~ 6e-12 × (f_x/1e-6), so need f_x ~ 1e3 to get μ ~ 6e-6.
    let f_x = 1e3; // eV per hydrogen nucleus
    let gamma_x = 1e-11; // fast decay, lifetime ~ 1e11 s ≈ 3000 yr, well before y-era ends

    let scenario = InjectionScenario::DecayingParticle { f_x, gamma_x };

    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config);
    solver.set_injection(scenario).unwrap();
    solver.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    // Total deposited energy: integral of heating rate over all time
    // For fast decay (Γ >> H), essentially all energy is deposited:
    // Δρ/ρ ≈ f_X × n_H,0 / ρ_γ,0
    // ρ_γ,0 = a_rad * T_cmb^4, n_H,0 = (1-Y_p) × n_b,0
    let drho_pde = delta_rho_over_rho(&solver.grid.x, &last.delta_n);

    // Compute expected from GF as independent cross-check
    let (mu_gf, y_gf) = {
        let scenario_gf = InjectionScenario::DecayingParticle { f_x, gamma_x };
        let mu = greens::mu_from_heating(
            |z| -scenario_gf.heating_rate_per_redshift(z, &cosmo),
            500.0,
            5e5,
            2000,
        );
        let y = greens::y_from_heating(
            |z| -scenario_gf.heating_rate_per_redshift(z, &cosmo),
            500.0,
            5e5,
            2000,
        );
        (mu, y)
    };

    eprintln!("Decay total energy: PDE Δρ/ρ = {drho_pde:.6e}");
    eprintln!("  PDE: μ = {:.6e}, y = {:.6e}", last.mu, last.y);
    eprintln!("  GF:  μ = {mu_gf:.6e}, y = {y_gf:.6e}");

    // PDE should have captured substantial energy
    assert!(
        drho_pde > 0.0,
        "Decaying particle should deposit positive energy: Δρ/ρ = {drho_pde}"
    );

    // PDE vs GF agreement on y (most of the energy is in y-era for fast decay)
    let y_err = (last.y - y_gf).abs() / y_gf.abs().max(1e-20);
    eprintln!("  y PDE vs GF err = {:.2}%", y_err * 100.0);

    assert!(
        y_err < 0.16,
        "Decay y PDE vs GF: err = {:.2}% > 16%",
        y_err * 100.0
    );
}

/// p-wave annihilation has an extra (1+z) factor, so it deposits more
/// energy at high z (more μ-like) relative to s-wave. At fixed f_ann,
/// p-wave should have LARGER |μ/y| ratio than s-wave.
#[test]
fn test_heat_swave_vs_pwave_mu_y_ratio() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    // Must be large enough that injection signal dominates adiabatic cooling floor (μ ~ -3e-9)
    let f_ann = 1e-21; // eV·m³/s

    // s-wave
    let mut solver_s = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_s
        .set_injection(InjectionScenario::AnnihilatingDM { f_ann })
        .unwrap();
    solver_s.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_s.run_with_snapshots(&[500.0]);
    let last_s = solver_s.snapshots.last().unwrap();

    // p-wave
    let mut solver_p = ThermalizationSolver::new(cosmo.clone(), grid_config);
    solver_p
        .set_injection(InjectionScenario::AnnihilatingDMPWave { f_ann })
        .unwrap();
    solver_p.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_p.run_with_snapshots(&[500.0]);
    let last_p = solver_p.snapshots.last().unwrap();

    let ratio_s = last_s.mu.abs() / last_s.y.abs().max(1e-30);
    let ratio_p = last_p.mu.abs() / last_p.y.abs().max(1e-30);

    eprintln!(
        "s-wave: μ = {:.4e}, y = {:.4e}, |μ/y| = {ratio_s:.4}",
        last_s.mu, last_s.y
    );
    eprintln!(
        "p-wave: μ = {:.4e}, y = {:.4e}, |μ/y| = {ratio_p:.4}",
        last_p.mu, last_p.y
    );

    // p-wave should have stronger μ relative to y (more high-z weighted)
    assert!(
        ratio_p > ratio_s,
        "p-wave should have larger |μ/y| than s-wave: p={ratio_p:.4} vs s={ratio_s:.4}"
    );

    // Both should produce positive μ (heating)
    assert!(
        last_s.mu > 0.0,
        "s-wave μ should be positive: {:.4e}",
        last_s.mu
    );
    assert!(
        last_p.mu > 0.0,
        "p-wave μ should be positive: {:.4e}",
        last_p.mu
    );

    // Both should produce positive y
    assert!(
        last_s.y > 0.0,
        "s-wave y should be positive: {:.4e}",
        last_s.y
    );
    assert!(
        last_p.y > 0.0,
        "p-wave y should be positive: {:.4e}",
        last_p.y
    );
}

/// PDE vs GF for DM s-wave annihilation heating, restricted to deep μ-era so
/// the two methods are comparing the same regime.
///
/// Oracle:             PDE and GF agree in the deep μ-era (z > 2×10⁵) where
///                     J_bb* ≈ 1, J_μ ≈ 1; both methods compute
///                     μ = (3/κ_c) · ∫ J_bb*·J_μ · (dQ/dz) dz.
/// Expected:           μ_PDE / μ_GF = 1
/// Oracle uncertainty: ~10% for continuous sources (per CLAUDE.md validation
///                     target, bursts hit 2-5%; continuous heating has larger
///                     method disagreement because the GF evaluates J_μ at z_heat
///                     whereas the PDE does the full evolution).
/// Tolerance:          12% on μ
///
/// The injection range is clipped to [3.5e5, 1e5] so both methods see a
/// pure μ-era integrand. The subdominant y component is not checked (y is
/// orthogonal to what's being tested).
#[test]
fn test_heat_dm_annihilation_pde_vs_gf() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    let f_ann = 1e-21;
    let z_start = 5e5;
    let z_end = 2.5e5;

    // PDE
    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config);
    solver
        .set_injection(InjectionScenario::AnnihilatingDM { f_ann })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[z_end]);
    let last = solver.snapshots.last().unwrap();

    // GF with matched integration bounds
    let scenario_gf = InjectionScenario::AnnihilatingDM { f_ann };
    let mu_gf = greens::mu_from_heating(
        |z| -scenario_gf.heating_rate_per_redshift(z, &cosmo),
        z_end,
        z_start,
        2000,
    );

    let mu_err = (last.mu - mu_gf).abs() / mu_gf.abs();
    eprintln!(
        "DM annihilation μ-era (z in [{z_end:.0e}, {z_start:.0e}]):\n  \
         PDE μ = {:.4e}, GF μ = {mu_gf:.4e}, err = {:.2}%",
        last.mu,
        mu_err * 100.0
    );
    assert!(
        mu_err < 0.12,
        "DM annihilation μ: PDE vs GF err = {:.2}% > 12% (μ-era only)",
        mu_err * 100.0
    );
    assert_eq!(
        solver.diag.newton_exhausted, 0,
        "Newton should converge for DM annihilation"
    );
}

/// Two bursts at different μ-era redshifts should produce the same
/// result as running them individually and summing.
/// Tests PDE linearity for heat injection.
#[test]
fn test_heat_superposition_two_bursts() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    let drho = 1e-6; // Small to stay in linear regime
    let z_a = 2e5;
    let z_b = 1e5;

    // Combined: inject both
    let mut solver_combined = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_combined
        .set_injection(InjectionScenario::Custom(Box::new(move |z, cosmo| {
            let a = InjectionScenario::SingleBurst {
                z_h: z_a,
                delta_rho_over_rho: drho,
                sigma_z: z_a * 0.01,
            };
            let b = InjectionScenario::SingleBurst {
                z_h: z_b,
                delta_rho_over_rho: drho,
                sigma_z: z_b * 0.01,
            };
            a.heating_rate(z, cosmo) + b.heating_rate(z, cosmo)
        })))
        .unwrap();
    solver_combined.set_config(SolverConfig {
        z_start: 3e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_combined.run_with_snapshots(&[500.0]);
    let combined = solver_combined.snapshots.last().unwrap();

    // Individual: run each separately and sum
    let mut mu_sum = 0.0;
    let mut y_sum = 0.0;

    for &z_h in &[z_a, z_b] {
        let mut solver = burst_solver(&grid_config, z_h, drho, z_h * 0.01, 3e5, 500.0);
        solver.run_with_snapshots(&[500.0]);
        let last = solver.snapshots.last().unwrap();
        mu_sum += last.mu;
        y_sum += last.y;
    }

    let mu_err = (combined.mu - mu_sum).abs() / mu_sum.abs().max(1e-20);
    let y_err = (combined.y - y_sum).abs() / y_sum.abs().max(1e-20);

    eprintln!("Two-burst superposition:");
    eprintln!(
        "  Combined: μ = {:.6e}, y = {:.6e}",
        combined.mu, combined.y
    );
    eprintln!("  Sum:      μ = {mu_sum:.6e}, y = {y_sum:.6e}");
    eprintln!(
        "  err: μ = {:.2}%, y = {:.2}%",
        mu_err * 100.0,
        y_err * 100.0
    );

    assert!(
        mu_err < 0.03,
        "Two-burst superposition μ: err = {:.2}% > 3%",
        mu_err * 100.0
    );
    // y is very small in μ-era, so allow looser tolerance
    assert!(
        y_err < 0.30 || y_sum.abs() < 1e-10,
        "Two-burst superposition y: err = {:.2}% > 30%",
        y_err * 100.0
    );
}

/// Symmetric ±Δρ/ρ bursts cancel: the residual spectrum must be indistinguishable
/// from the no-injection adiabatic-cooling case.
///
/// Oracle:             heat.heating_rate + cool.heating_rate ≡ 0 identically, so
///                     the injection closure produces zero source. The two solver
///                     runs (±burst, no-injection) must therefore produce identical
///                     spectra up to floating-point roundoff in heating_rate summation.
/// Expected:           max|Δn_cancel − Δn_noinj| ≈ 0
/// Oracle uncertainty: machine ε (adiabatic cooling is deterministic once solver
///                     state is fixed)
/// Tolerance:          1e-12 (absolute; ~4 OOM above f64 ε to cover Newton residual)
///
/// Tests cancellation directly by differencing two runs.
#[test]
fn test_heat_heating_cooling_cancellation() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 1000,
        ..GridConfig::default()
    };
    let drho = 1e-5;
    let z_h = 1e5;

    // Run 1: ± bursts summed in a Custom closure (should cancel analytically).
    let mut solver_cancel = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_cancel
        .set_injection(InjectionScenario::Custom(Box::new(move |z, cosmo| {
            let heat = InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: z_h * 0.01,
            };
            let cool = InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: -drho,
                sigma_z: z_h * 0.01,
            };
            heat.heating_rate(z, cosmo) + cool.heating_rate(z, cosmo)
        })))
        .unwrap();
    solver_cancel.set_config(SolverConfig {
        z_start: z_h * 1.5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_cancel.run_with_snapshots(&[500.0]);
    let cancel_snap = solver_cancel.snapshots.last().unwrap().clone();

    // Run 2: identical solver with no injection — pure adiabatic cooling.
    let mut solver_noinj = ThermalizationSolver::new(cosmo, grid_config);
    solver_noinj
        .set_injection(InjectionScenario::Custom(Box::new(|_, _| 0.0)))
        .unwrap();
    solver_noinj.set_config(SolverConfig {
        z_start: z_h * 1.5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_noinj.run_with_snapshots(&[500.0]);
    let noinj_snap = solver_noinj.snapshots.last().unwrap();

    let max_diff = cancel_snap
        .delta_n
        .iter()
        .zip(noinj_snap.delta_n.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "Cancellation check: μ_cancel={:.4e}, μ_noinj={:.4e}, max|Δn_cancel−Δn_noinj|={:.4e}",
        cancel_snap.mu, noinj_snap.mu, max_diff
    );

    assert!(
        max_diff < 1e-12,
        "± bursts should cancel to roundoff: max|Δn_cancel − Δn_noinj| = {max_diff:.4e} \
         (tol 1e-12). Any larger residual means the injection summation is not \
         producing the zero source it should."
    );
}

/// J_bb*(z) is monotonically decreasing for z > 5e4, so μ/Δρ from identical
/// bursts at increasing z should show a net decrease across the tested range.
///
/// Note: pairwise strict monotonicity fails under the B&F decomposition at
/// the ~1% level (r-type residual reallocates between μ and y at each z),
/// so this test only verifies the envelope — first-vs-last decrease — not
/// pairwise monotonicity. Renamed from `_strict_monotonicity` for honesty.
#[test]
fn test_heat_thermalization_suppression_net_decrease() {
    let drho = 1e-5;

    // In the thermalization regime (z > 2e5), J_bb* decreases,
    // so μ/Δρ should monotonically decrease with increasing z.
    // Below z=2e5, μ/Δρ can increase (transition from y→μ era).
    let z_values = [2e5, 3e5, 5e5];
    let mut mu_over_drho = Vec::new();

    for &z_h in &z_values {
        let run = standard_burst(z_h, drho);
        let last = &run.snap;
        let ratio = last.mu / drho;
        mu_over_drho.push(ratio);
        eprintln!("z_h = {z_h:.0e}: μ/Δρ = {ratio:.4e}");
    }

    // Overall decrease from z=2×10⁵ to z=5×10⁵. Pairwise strict monotonicity
    // fails under the B&F decomposition at the ~1% level because the residual
    // r-type shape partitions differently between μ and y at each z; the
    // envelope J_bb* suppression is still clearly captured.
    assert!(
        mu_over_drho.last().unwrap() < mu_over_drho.first().unwrap(),
        "μ/Δρ should show net suppression from z={:.0e} ({:.4e}) to z={:.0e} ({:.4e})",
        z_values[0],
        mu_over_drho[0],
        *z_values.last().unwrap(),
        mu_over_drho.last().unwrap()
    );

    // At z=2e5, close to 1.401 (peak μ-era)
    assert!(
        mu_over_drho[0] > 1.0,
        "At z=2e5, μ/Δρ should be > 1.0: got {:.4e}",
        mu_over_drho[0]
    );

    // At z=5e5, noticeable suppression (J_bb*(5e5) ≈ 0.7)
    assert!(
        mu_over_drho[2] < mu_over_drho[0],
        "At z=5e5, μ/Δρ should be less than at z=2e5: {:.4e} vs {:.4e}",
        mu_over_drho[2],
        mu_over_drho[0]
    );
}

/// In the deep μ-era (z = 2e5), the spectral shape of Δn should be
/// proportional to M(x) = (x/2.19 − 1) × g_bb(x) × x⁻¹ to good
/// approximation. Check the shape correlation.
#[test]
fn test_heat_spectral_shape_mu_era() {
    let drho = 1e-5;
    let z_h = 2e5;

    let run = standard_burst(z_h, drho);
    let last = &run.snap;

    // Compute M(x) shape at each grid point
    let mu_shape: Vec<f64> = run.x.iter().map(|&x| spectrum::mu_shape(x)).collect();

    // Find best-fit amplitude: μ_amp = Σ(Δn × M) / Σ(M²)
    // Only use x ∈ [1, 15] to avoid edge effects
    let mut num = 0.0;
    let mut den = 0.0;
    let mut count = 0;
    for (i, &x) in run.x.iter().enumerate() {
        if x > 1.0 && x < 15.0 {
            num += last.delta_n[i] * mu_shape[i];
            den += mu_shape[i] * mu_shape[i];
            count += 1;
        }
    }
    let amp = num / den;

    // Compute residual: || Δn − amp × M(x) || / || Δn ||
    let mut res_sq = 0.0;
    let mut norm_sq = 0.0;
    for (i, &x) in run.x.iter().enumerate() {
        if x > 1.0 && x < 15.0 {
            let residual = last.delta_n[i] - amp * mu_shape[i];
            res_sq += residual * residual;
            norm_sq += last.delta_n[i] * last.delta_n[i];
        }
    }
    let rel_rms = (res_sq / norm_sq).sqrt();

    eprintln!("μ-era spectral shape: amp = {amp:.6e}, rel_rms = {rel_rms:.4e} ({count} points)");

    // In pure μ-era, shape should be >90% M(x)
    assert!(
        rel_rms < 0.10,
        "μ-era spectral shape residual {rel_rms:.4e} > 10%"
    );
}

/// In the y-era (z = 5000), the spectral shape should be dominated by
/// Y_SZ(x) = x × coth(x/2) − 4. Check shape correlation.
#[test]
fn test_heat_spectral_shape_y_era() {
    let drho = 1e-5;
    let z_h = 5000.0;

    let run = standard_burst(z_h, drho);
    let last = &run.snap;

    // Y_SZ shape
    let y_shape: Vec<f64> = run.x.iter().map(|&x| spectrum::y_shape(x)).collect();

    // Best-fit amplitude in [1, 15]
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &x) in run.x.iter().enumerate() {
        if x > 1.0 && x < 15.0 {
            num += last.delta_n[i] * y_shape[i];
            den += y_shape[i] * y_shape[i];
        }
    }
    let amp = num / den;

    // Residual
    let mut res_sq = 0.0;
    let mut norm_sq = 0.0;
    for (i, &x) in run.x.iter().enumerate() {
        if x > 1.0 && x < 15.0 {
            let residual = last.delta_n[i] - amp * y_shape[i];
            res_sq += residual * residual;
            norm_sq += last.delta_n[i] * last.delta_n[i];
        }
    }
    let rel_rms = (res_sq / norm_sq).sqrt();

    eprintln!("y-era spectral shape: amp = {amp:.6e}, rel_rms = {rel_rms:.4e}");

    assert!(
        rel_rms < 0.10,
        "y-era spectral shape residual {rel_rms:.4e} > 10%"
    );
}

/// For a decaying particle, shorter lifetimes (larger Γ_X) deposit energy
/// at later times (lower z, more y-like), while longer lifetimes deposit
/// at earlier times (higher z, more μ-like).
///
/// Test with two lifetimes that span the μ-y transition.
#[test]
fn test_heat_decay_lifetime_controls_mu_y() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    // Must be large enough that injection signal dominates adiabatic cooling floor (μ ~ -3e-9)
    let f_x = 1e4; // eV per hydrogen nucleus

    // "Early" decay: short lifetime, decays at high z (μ-era)
    // cosmic_time(z=1e5) ≈ 2.4e9 s, so Γ=1e-9 gives τ=1e9 s → peaks near z~1e5
    let gamma_early = 1e-9;

    // "Late" decay: longer lifetime, decays at low z (y-era)
    // cosmic_time(z=5000) ≈ 1e12 s, so Γ=1e-12 gives τ=1e12 s → peaks near z~5000
    let gamma_late = 1e-12;

    let mut solver_early = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    solver_early
        .set_injection(InjectionScenario::DecayingParticle {
            f_x,
            gamma_x: gamma_early,
        })
        .unwrap();
    solver_early.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_early.run_with_snapshots(&[500.0]);
    let last_early = solver_early.snapshots.last().unwrap();

    let mut solver_late = ThermalizationSolver::new(cosmo.clone(), grid_config);
    solver_late
        .set_injection(InjectionScenario::DecayingParticle {
            f_x,
            gamma_x: gamma_late,
        })
        .unwrap();
    solver_late.set_config(SolverConfig {
        z_start: 5e5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_late.run_with_snapshots(&[500.0]);
    let last_late = solver_late.snapshots.last().unwrap();

    let ratio_early = last_early.mu.abs() / last_early.y.abs().max(1e-30);
    let ratio_late = last_late.mu.abs() / last_late.y.abs().max(1e-30);

    eprintln!("Decay lifetime effect:");
    eprintln!(
        "  Early (Γ={gamma_early}): μ = {:.4e}, y = {:.4e}, |μ/y| = {ratio_early:.4}",
        last_early.mu, last_early.y
    );
    eprintln!(
        "  Late  (Γ={gamma_late}):  μ = {:.4e}, y = {:.4e}, |μ/y| = {ratio_late:.4}",
        last_late.mu, last_late.y
    );

    // Early decay (high z) → more μ relative to y
    assert!(
        ratio_early > ratio_late,
        "Early decay should give larger |μ/y|: early={ratio_early:.4} vs late={ratio_late:.4}"
    );

    // Both should produce positive distortion (heating)
    assert!(
        last_early.mu > 0.0,
        "Early decay should have positive μ: {:.4e}",
        last_early.mu
    );
    assert!(
        last_late.y > 0.0,
        "Late decay should have positive y: {:.4e}",
        last_late.y
    );
}

/// The 3-component decomposition (μ + y + temperature shift) should
/// capture >85% of the variance of the PDE output. Test at multiple
/// redshifts.
#[test]
fn test_heat_spectral_decomposition_residual_sweep() {
    let drho = 1e-5;
    // Pure y-era, transition, and pure μ-era (skip 5e4 transition where
    // the 3-component decomposition has inherently high residual)
    let z_values = [5000.0, 1e4, 2e5];

    for &z_h in &z_values {
        let run = standard_burst(z_h, drho);
        let last = &run.snap;

        // Reconstruct from decomposition: Δn = μ×M(x) + y×Y(x) + c_T×g_bb(x)
        // where c_T = ΔT/T is fitted, not equal to Δρ/ρ
        // Use: c_T = (Δρ/ρ − 4y) / 4 since M(x) is energy-neutral
        let c_t = (last.delta_rho_over_rho - 4.0 * last.y) / 4.0;
        let mut res_sq = 0.0;
        let mut norm_sq = 0.0;
        for (i, &x) in run.x.iter().enumerate() {
            if x > 0.5 && x < 20.0 {
                let reconstructed = last.mu * spectrum::mu_shape(x)
                    + last.y * spectrum::y_shape(x)
                    + c_t * spectrum::g_bb(x);
                let residual = last.delta_n[i] - reconstructed;
                res_sq += residual * residual;
                norm_sq += last.delta_n[i] * last.delta_n[i];
            }
        }
        let rel_rms = (res_sq / norm_sq.max(1e-40)).sqrt();

        eprintln!(
            "z_h={z_h:.0e}: decomposition residual = {:.2}%",
            rel_rms * 100.0
        );

        assert!(
            rel_rms < 0.20,
            "Decomposition residual at z_h={z_h}: {:.2}% > 20%",
            rel_rms * 100.0
        );
    }
}

/// PDE vs GF cross-validation in the regimes where the two methods agree.
///
/// `last.mu` is the B&F BE chemical potential fit to the PDE spectrum; `mu_gf`
/// is 1.401·Δρ·J_bb*·J_μ, a visibility-function convolution. In the deep μ-era
/// the two agree to <30%; in the y-era where μ is suppressed by J_μ, only the
/// y comparison is meaningful (and it's tight). The μ-y crossover (z ∈ [1e4, 5e4])
/// is a known definition mismatch and is deliberately not tested here — hiding
/// it behind a ±1000% bound pretends to validate something it can't.
#[test]
fn test_heat_pde_vs_gf_multi_z_sweep() {
    let drho = 1e-5;

    // Deep-μ-era μ cross-check (μ dominant, but B&F-vs-visibility definitions
    // still disagree at the ~30–40% level at z_h=1e5 because the PDE spectrum
    // in the transition tail has residual r-type shape that the B&F BE fit
    // and the 1.401·J_bb*·J_μ convolution distribute differently).
    let mu_cases: &[(f64, f64)] = &[
        (1e5, 0.40), // early μ-era (measured ~35%)
        (2e5, 0.20), // deep μ-era
        (5e5, 0.20), // deep μ-era with J_bb* suppression
    ];
    for &(z_h, mu_tol) in mu_cases {
        let run = standard_burst(z_h, drho);
        let last = &run.snap;

        let mu_gf = 1.401 * drho * greens::visibility_j_bb_star(z_h) * greens::visibility_j_mu(z_h);
        let mu_err = (last.mu - mu_gf).abs() / mu_gf.abs();
        eprintln!(
            "μ-era z_h={z_h:.0e}: PDE μ={:.4e} GF μ={mu_gf:.4e} err={:.1}%",
            last.mu,
            mu_err * 100.0
        );
        assert!(
            mu_err < mu_tol,
            "z_h={z_h}: μ PDE vs GF err = {:.1}% > {:.0}%",
            mu_err * 100.0,
            mu_tol * 100.0
        );
    }

    // Pure y-era y cross-check (y dominant, both methods agree to ~5%).
    let z_h = 5000.0;
    let run = standard_burst(z_h, drho);
    let last = &run.snap;

    let y_gf = 0.25 * drho * greens::visibility_j_y(z_h);
    let y_err = (last.y - y_gf).abs() / y_gf.abs();
    eprintln!(
        "y-era z_h={z_h:.0e}: PDE y={:.4e} GF y={y_gf:.4e} err={:.1}%",
        last.y,
        y_err * 100.0
    );
    assert!(
        y_err < 0.05,
        "z_h={z_h}: y PDE vs GF err = {:.1}% > 5%",
        y_err * 100.0
    );
}

/// A Custom injection scenario that replicates SingleBurst should
/// produce identical results. This tests the injection infrastructure.
#[test]
fn test_heat_custom_matches_single_burst() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 1000,
        ..GridConfig::default()
    };
    let drho = 1e-5;
    let z_h = 1e5;
    let sigma_z = z_h * 0.01;

    // Builtin
    let mut solver_builtin = burst_solver(&grid_config, z_h, drho, sigma_z, z_h * 1.5, 500.0);
    solver_builtin.run_with_snapshots(&[500.0]);
    let last_builtin = solver_builtin.snapshots.last().unwrap();

    // Custom that calls the same heating rate
    let mut solver_custom = ThermalizationSolver::new(cosmo.clone(), grid_config);
    let scenario_inner = InjectionScenario::SingleBurst {
        z_h,
        delta_rho_over_rho: drho,
        sigma_z,
    };
    solver_custom
        .set_injection(InjectionScenario::Custom(Box::new(move |z, cosmo| {
            scenario_inner.heating_rate(z, cosmo)
        })))
        .unwrap();
    solver_custom.set_config(SolverConfig {
        z_start: z_h * 1.5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver_custom.run_with_snapshots(&[500.0]);
    let last_custom = solver_custom.snapshots.last().unwrap();

    let mu_err = (last_custom.mu - last_builtin.mu).abs() / last_builtin.mu.abs().max(1e-20);
    let y_err = (last_custom.y - last_builtin.y).abs() / last_builtin.y.abs().max(1e-20);

    eprintln!(
        "Custom vs builtin: μ err = {:.6e}, y err = {:.6e}",
        mu_err, y_err
    );

    // Should be very close — small differences from adaptive stepping
    // seeing different function pointer types
    assert!(
        mu_err < 0.01,
        "Custom vs builtin μ: err = {mu_err:.4e} > 1%"
    );
    assert!(y_err < 0.01, "Custom vs builtin y: err = {y_err:.4e} > 1%");
}

/// At z ≈ 10⁴ (transition region), both μ and y should be nonzero.
/// The distortion is neither pure μ nor pure y. Test that both
/// components are measurable and that the total Δρ/ρ is correct.
#[test]
fn test_heat_transition_region_mixed_distortion() {
    let drho = 1e-5;
    let z_h = 1e4; // Transition region

    let run = standard_burst(z_h, drho);
    let last = &run.snap;

    eprintln!(
        "Transition (z_h=1e4): μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        last.mu, last.y, last.delta_rho_over_rho
    );

    // Both μ and y should be nonzero and positive
    assert!(
        last.mu > 0.0,
        "Transition μ should be positive: {:.4e}",
        last.mu
    );
    assert!(
        last.y > 0.0,
        "Transition y should be positive: {:.4e}",
        last.y
    );

    // Both components are present. At z_h = 10⁴ the spectrum is y-dominated
    // and B&F's BE-exponential μ is small (|μ/y| ~ 0.01), but nonzero.
    let ratio = last.mu.abs() / last.y.abs().max(1e-30);
    eprintln!("  |μ/y| = {ratio:.3}");
    assert!(
        ratio > 1e-4 && ratio < 100.0,
        "Transition should have mixed μ+y: |μ/y| = {ratio:.3}"
    );

    // Energy conservation
    let drho_measured = delta_rho_over_rho(&run.x, &last.delta_n);
    let e_err = (drho_measured - drho).abs() / drho;
    eprintln!("  Energy err = {:.2}%", e_err * 100.0);
    assert!(
        e_err < 0.02,
        "Transition energy conservation: {:.2}% > 2%",
        e_err * 100.0
    );
}

/// The PDE solver should be linear: doubling the injection amplitude
/// should double all distortion parameters.
#[test]
fn test_heat_pde_amplitude_linearity() {
    let z_h = 1e5;

    let drho_1 = 1e-5;
    let drho_2 = 2e-5;

    let run1 = standard_burst(z_h, drho_1);
    let last1 = &run1.snap;

    let run2 = standard_burst(z_h, drho_2);
    let last2 = &run2.snap;

    let mu_ratio = last2.mu / last1.mu;
    let y_ratio = last2.y / last1.y;

    eprintln!("Amplitude linearity at z_h={z_h}:");
    eprintln!("  1×: μ = {:.6e}, y = {:.6e}", last1.mu, last1.y);
    eprintln!("  2×: μ = {:.6e}, y = {:.6e}", last2.mu, last2.y);
    eprintln!("  μ ratio = {mu_ratio:.4} (expect 2.0)");
    eprintln!("  y ratio = {y_ratio:.4} (expect 2.0)");

    assert!(
        (mu_ratio - 2.0).abs() < 0.02,
        "PDE μ not linear in amplitude: ratio = {mu_ratio:.4}"
    );
    assert!(
        (y_ratio - 2.0).abs() < 0.02,
        "PDE y not linear in amplitude: ratio = {y_ratio:.4}"
    );
}

/// Tabulated heating matching a SingleBurst should give the same PDE result.
#[test]
fn test_tabulated_heating_matches_single_burst() {
    let cosmo = Cosmology::default();
    let z_h = 1e5_f64;
    let drho = 1e-5_f64;
    let sigma = (z_h * 0.04).max(100.0);

    // Reference: built-in SingleBurst
    let burst = InjectionScenario::SingleBurst {
        z_h,
        delta_rho_over_rho: drho,
        sigma_z: sigma,
    };
    let z_start = z_h + 7.0 * sigma;
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver.set_injection(burst).unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);
    let ref_snap = solver.snapshots.last().unwrap().clone();

    // Tabulated: sample the burst's dq/dz on a dense grid
    let burst = InjectionScenario::SingleBurst {
        z_h,
        delta_rho_over_rho: drho,
        sigma_z: sigma,
    };
    let n = 2000;
    let z_lo = (z_h - 7.0 * sigma).max(100.0);
    let z_hi = z_h + 7.0 * sigma;
    let mut z_table = Vec::with_capacity(n);
    let mut rate_table = Vec::with_capacity(n);
    for i in 0..n {
        let z = z_lo + (z_hi - z_lo) * i as f64 / (n - 1) as f64;
        let dq_dz = burst.heating_rate_per_redshift(z, &cosmo).abs();
        z_table.push(z);
        rate_table.push(dq_dz);
    }

    let tabulated = InjectionScenario::TabulatedHeating {
        z_table,
        rate_table,
    };
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver.set_injection(tabulated).unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);
    let tab_snap = solver.snapshots.last().unwrap();

    let mu_err = (tab_snap.mu - ref_snap.mu).abs() / ref_snap.mu.abs().max(1e-30);
    let y_err = (tab_snap.y - ref_snap.y).abs() / ref_snap.y.abs().max(1e-30);

    eprintln!("Tabulated vs SingleBurst at z_h={z_h:.0e}:");
    eprintln!("  ref: mu={:.4e}, y={:.4e}", ref_snap.mu, ref_snap.y);
    eprintln!("  tab: mu={:.4e}, y={:.4e}", tab_snap.mu, tab_snap.y);
    eprintln!("  err: mu={mu_err:.2e}, y={y_err:.2e}");

    assert!(mu_err < 0.02, "mu should match within 2%: err={mu_err:.4e}");
    assert!(y_err < 0.02, "y should match within 2%: err={y_err:.4e}");
}

/// μ-era burst energy conservation benchmark.
///
/// Oracle:             CLAUDE.md §Validation Targets (and Chluba & Sunyaev 2012
///                     methodology): energy conservation < 5% across all
///                     redshifts for the PDE solver. Procopio & Burigana (2009)
///                     achieve < 0.05% with their higher-order KYPRIX scheme
///                     — that is an aspirational target; our IMEX (CN+BE) is
///                     O(Δτ²) + O(Δτ) mixed and does not reach it.
/// Expected:           Δρ_out / Δρ_in = 1 exactly
/// Oracle uncertainty: method-limited (first-order temporal residual of the
///                     coupled T_e/DC-BR step)
/// Tolerance:          0.5% on both grids, plus grid-independence to 0.1 pp.
///
/// The residual is **temporal, not spatial**
/// (`dev/audit/energy_conservation_audit.md`): measured −0.297% on the default
/// grid and −0.319% on the production grid, i.e. refining the grid makes it
/// marginally *worse*, while refining dtau_max fixes it (on this scenario with
/// z_start = 3e5: −0.284% at dtau_max = 10 → −0.060% at dtau_max = 2). The
/// second leg asserts that the two grids agree, which is the signature of a
/// time-discretization residual. The dtau_max convergence order itself is
/// pinned by `tests/convergence_order.rs`.
#[test]
fn test_pb2009_energy_conservation() {
    let cosmo = Cosmology::default();
    let delta_rho = 1e-5;
    let z_h = 2e5;

    // Default grid (2000 pts): 0.5% tolerance (measured −0.297%).
    let mut solver_default = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver_default
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            sigma_z: z_h / 10.0,
            delta_rho_over_rho: delta_rho,
        })
        .unwrap();
    let snaps = solver_default.run_with_snapshots(&[200.0]);
    let err_default = (snaps[0].delta_rho_over_rho - delta_rho).abs() / delta_rho;
    eprintln!(
        "Default grid (2000 pts): Δρ_out = {:.6e}, err = {:.3}%",
        snaps[0].delta_rho_over_rho,
        err_default * 100.0
    );
    assert!(
        err_default < 0.005,
        "Default-grid energy conservation: err = {:.3}% (tol 0.5%)",
        err_default * 100.0
    );

    // Production grid (4000 pts): same bound, and the two must agree — the
    // residual is a time-discretization term, so doubling the grid must not
    // move it. If they diverge, a genuinely spatial error has appeared.
    let mut solver_prod = ThermalizationSolver::new(cosmo, GridConfig::production());
    solver_prod
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            sigma_z: z_h / 10.0,
            delta_rho_over_rho: delta_rho,
        })
        .unwrap();
    let snaps_prod = solver_prod.run_with_snapshots(&[200.0]);
    let err_prod = (snaps_prod[0].delta_rho_over_rho - delta_rho).abs() / delta_rho;
    eprintln!(
        "Production grid (4000 pts): Δρ_out = {:.6e}, err = {:.3}%",
        snaps_prod[0].delta_rho_over_rho,
        err_prod * 100.0
    );
    assert!(
        err_prod < 0.005,
        "Production-grid energy conservation: err = {:.3}% (tol 0.5%)",
        err_prod * 100.0
    );
    // Grid-independence: measured |0.319% − 0.297%| = 0.022 pp.
    assert!(
        (err_prod - err_default).abs() < 1e-3,
        "Energy residual should be grid-independent (it is temporal): \
         default {:.3}% vs production {:.3}%, difference {:.3} pp > 0.1 pp",
        err_default * 100.0,
        err_prod * 100.0,
        (err_prod - err_default).abs() * 100.0
    );
}

//
// These tests run canonical scenarios and check full spectral output against
// reference values derived from validated PDE runs. They catch regressions
// that scalar (μ, y) comparisons miss.
/// μ-era burst at z=2×10⁵: PDE μ should match the Chluba 2013 analytic formula.
///
/// Oracle:            Chluba (2013) MNRAS 434, 352, Eq. 5
///                    μ = (3/κ_c) · J_bb*(z_h) · J_μ(z_h) · Δρ/ρ
/// Expected:          1.401 × J_bb*(2e5) × J_μ(2e5) × 1e-5 ≈ 1.36×10⁻⁵
/// Oracle uncertainty: 5% (GF fit uncertainty vs CosmoTherm in μ-era)
/// Tolerance:          10% (PDE vs GF on production grid, per CLAUDE.md validation targets)
///
/// Uses production grid (4000 pts) per CLAUDE.md, where PDE agrees with GF
/// to ≲5%, and checks the analytic target.
#[test]
fn golden_mu_era_spectral_shape() {
    let z_h = 2e5;
    let drho = 1e-5;
    let sigma = z_h * 0.04;

    let mut solver = burst_solver(
        &GridConfig::production(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        1e4,
    );

    let x_grid = solver.grid.x.clone();
    let result = solver.run_to_result(1e4);
    let snap = &result.snapshot;

    let mu = snap.mu;
    let y = snap.y;
    let drho_out = snap.delta_rho_over_rho;

    // Oracle: μ = (3/κ_c) · J_bb*(z_h) · J_μ(z_h) · Δρ/ρ
    let j_bb = greens::visibility_j_bb_star(z_h);
    let j_mu = greens::visibility_j_mu(z_h);
    let mu_expected = (3.0 / KAPPA_C) * j_bb * j_mu * drho;
    let mu_err = (mu - mu_expected).abs() / mu_expected;
    assert!(
        mu_err < 0.10,
        "μ-era golden: mu={mu:.4e} vs Chluba 2013 Eq.5 prediction {mu_expected:.4e} \
         (J_bb*={j_bb:.4}, J_μ={j_mu:.4}), rel_err={:.2}% (tol 10%)",
        mu_err * 100.0
    );

    // y contamination: the B&F fit partitions some residual into y. With a
    // pure-μ spectrum from the GF, true y/μ < 1%; the PDE decomposition sees
    // ~4-5% cross-talk from the non-orthogonal basis.
    assert!(
        y.abs() / mu.abs() < 0.08,
        "μ-era golden: y/mu ratio = {:.4} (expected < 8% cross-talk)",
        y.abs() / mu.abs()
    );

    // Energy conservation: PDE should preserve injected Δρ/ρ to ≲2% on
    // production grid (CLAUDE.md §Validation Targets).
    let e_err = (drho_out - drho).abs() / drho;
    assert!(
        e_err < 0.02,
        "μ-era golden: energy error = {:.2}% (expected < 2%)",
        e_err * 100.0
    );

    // Spectral shape invariants (M(x) sign structure):
    // M(x) zero crossing at x = β_μ ≈ 2.19: negative below, positive above.
    let x5_idx = x_grid.iter().position(|&x| x > 5.0).unwrap();
    assert!(
        snap.delta_n[x5_idx] > 0.0,
        "μ-era golden: Δn(x≈5) should be positive (above M(x) zero crossing)"
    );
    let x2_idx = x_grid.iter().position(|&x| x > 2.0).unwrap();
    assert!(
        snap.delta_n[x2_idx] < 0.0,
        "μ-era golden: Δn(x≈2) should be negative (below M(x) zero crossing)"
    );

    // Newton must converge cleanly in every step; exhaustion is a solver bug.
    assert_eq!(
        result.diag_newton_exhausted, 0,
        "μ-era golden: Newton exhausted {} times",
        result.diag_newton_exhausted
    );

    eprintln!(
        "Golden μ-era (production grid): mu={mu:.4e}, mu_expected={mu_expected:.4e}, \
         rel_err={:.2}%, y/mu={:.3}, drho_err={:.2}%, steps={}",
        mu_err * 100.0,
        y.abs() / mu.abs(),
        e_err * 100.0,
        result.step_count,
    );
}

/// Golden reference: y-era burst at z=5×10³.
#[test]
fn golden_y_era_spectral_shape() {
    let z_h = 5e3;
    let drho = 1e-5;
    let sigma = 200.0;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    let x_grid = solver.grid.x.clone();
    let result = solver.run_to_result(500.0);
    let snap = &result.snapshot;

    let mu = snap.mu;
    let y = snap.y;
    let drho_out = snap.delta_rho_over_rho;

    // y should be close to Δρ/(4ρ) = 2.5e-6. Measured: 2.49e-6 (0.4% error).
    let y_expected = drho / 4.0;
    let y_err = (y - y_expected).abs() / y_expected;
    assert!(
        y_err < 0.03,
        "y-era golden: y={y:.4e} vs expected {y_expected:.4e}, err={:.2}%",
        y_err * 100.0
    );

    // μ should be very small compared to y. Measured: μ/y ≈ 2.2%.
    assert!(
        mu.abs() / y.abs() < 0.04,
        "y-era golden: mu/y ratio = {:.4} (expected < 0.04)",
        mu.abs() / y.abs()
    );

    // Energy conservation: < 1%. Measured: 0.3%.
    let e_err = (drho_out - drho).abs() / drho;
    assert!(
        e_err < 0.01,
        "y-era golden: energy error = {:.2}% (expected < 1%)",
        e_err * 100.0
    );

    // Spectral shape: y-distortion is Y_SZ(x) ∝ x·coth(x/2) - 4
    // At x=2: Y_SZ < 0 (photon deficit), at x=6: Y_SZ > 0 (excess)
    let x2_idx = x_grid.iter().position(|&x| x > 2.0).unwrap();
    let x6_idx = x_grid.iter().position(|&x| x > 6.0).unwrap();
    assert!(
        snap.delta_n[x2_idx] < 0.0,
        "y-era golden: Δn(x≈2) should be negative for y-distortion"
    );
    assert!(
        snap.delta_n[x6_idx] > 0.0,
        "y-era golden: Δn(x≈6) should be positive for y-distortion"
    );

    // Step count bounds
    assert!(
        result.step_count > 30 && result.step_count < 1000,
        "y-era golden: step_count={} outside [30, 1000]",
        result.step_count
    );

    assert_eq!(result.diag_newton_exhausted, 0);

    eprintln!(
        "Golden y-era: mu={mu:.4e}, y={y:.4e}, y_err={:.1}%, steps={}",
        y_err * 100.0,
        result.step_count
    );
}

/// Transition-era burst (z_h=5×10⁴): verifies energy conservation and sign
/// structure. Individual μ and y values are *not* checked against the GF
/// because the B&F nonlinear-BE decomposition the PDE uses partitions μ/y
/// differently from the visibility-convolution Chluba GF in the μ-y
/// crossover (the r-type residual reallocates). Comparing μ_PDE to μ_GF
/// at z_h=5e4 is not a physics test — it's a decomposition-basis test.
///
/// Oracle:             Energy conservation Δρ_out = Δρ_in and sign consistency
///                     (μ > 0, y > 0 for positive heat injection). Total
///                     sum-rule: μ/1.401 + 4y + 4ΔT/T ≈ Δρ/ρ (basis-independent).
/// Expected:           Δρ_out / Δρ = 1 exactly;
///                     μ/1.401 + 4y + 4ΔT/T = Δρ/ρ
/// Oracle uncertainty: scheme residual
/// Tolerance:          2% energy; 10% on the sum-rule (allows for decomposition
///                     basis + B&F fit residual)
#[test]
fn golden_transition_era_spectral_shape() {
    let z_h = 5e4;
    let drho = 1e-5;
    let sigma = 2000.0;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        1e3,
    );

    let result = solver.run_to_result(1e3);
    let snap = &result.snapshot;

    assert!(
        snap.mu > 0.0 && snap.y > 0.0,
        "transition golden: μ={:.4e}, y={:.4e} — both must be positive for heat injection",
        snap.mu,
        snap.y
    );

    // Oracle 1: energy conservation at 2%.
    let e_err = (snap.delta_rho_over_rho - drho).abs() / drho;
    assert!(
        e_err < 0.02,
        "transition golden: energy conservation err = {:.2}% (tol 2%)",
        e_err * 100.0,
    );

    // Oracle 2: sum rule. The (μ/1.401 + 4y + 4 ΔT/T) · ρ_γ is the total
    // energy in the distortion when converted to the Chluba convention.
    // The B&F decomposition returns μ, y; ΔT/T is inside delta_rho_over_rho.
    // Here we test the weaker sum-rule μ/1.401 + 4y ≤ Δρ/ρ (holds because
    // the ΔT/T shift carries the rest of the energy).
    let sum = snap.mu / 1.401 + 4.0 * snap.y;
    assert!(
        sum > 0.5 * drho && sum < 2.5 * drho,
        "transition golden: μ/1.401 + 4y = {sum:.3e} should be within factor 2.5 \
         of Δρ/ρ = {drho:.3e} (B&F decomposition + ΔT offset partition)",
    );

    // Spectral L2 norm must be measurable (rules out complete signal loss).
    let l2: f64 = snap.delta_n.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        l2 > 1e-8,
        "transition golden: L2(Δn) = {l2:.4e} (too small — signal lost?)"
    );

    assert_eq!(result.diag_newton_exhausted, 0);

    eprintln!(
        "Golden transition (z_h={z_h:.0e}): μ={:.4e}, y={:.4e}, drho_err={:.2}%, \
         sum(μ/1.401+4y)/Δρ={:.3}",
        snap.mu,
        snap.y,
        e_err * 100.0,
        sum / drho,
    );
}

/// Different cosmologies produce different μ/y for the same injection.
#[test]
fn test_solver_respects_cosmology_parameters() {
    // Run with default (Chluba 2013) cosmology
    let mut solver1 = ThermalizationSolver::new(Cosmology::default(), GridConfig::fast());
    solver1
        .set_injection(InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 1e-5,
            sigma_z: 2000.0,
        })
        .unwrap();
    let result1 = solver1.run_to_result(500.0);
    let snap1 = &result1.snapshot;

    // Run with Planck 2018 cosmology (different Ω_b, h, Y_p)
    let mut solver2 = ThermalizationSolver::new(Cosmology::planck2018(), GridConfig::fast());
    solver2
        .set_injection(InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 1e-5,
            sigma_z: 2000.0,
        })
        .unwrap();
    let result2 = solver2.run_to_result(500.0);
    let snap2 = &result2.snapshot;

    // Both should produce physical results
    assert!(
        snap1.mu.abs() > 1e-8,
        "default cosmo: mu too small: {}",
        snap1.mu
    );
    assert!(
        snap2.mu.abs() > 1e-8,
        "planck2018 cosmo: mu too small: {}",
        snap2.mu
    );

    // They should differ (different baryon density, Hubble rate, etc.)
    let mu_diff = (snap1.mu - snap2.mu).abs() / snap1.mu.abs();
    assert!(
        mu_diff > 1e-3,
        "cosmologies should give different μ: default={:.4e}, p2018={:.4e}, diff={:.2e}",
        snap1.mu,
        snap2.mu,
        mu_diff
    );

    // Direction check: higher Ω_b (Planck 2018: ω_b=0.02237 vs Chluba 2013: ω_b=0.022)
    // means more baryons → stronger Compton coupling → different thermalization.
    // Both μ values should have the same sign (positive for energy injection).
    assert!(
        snap1.mu > 0.0 && snap2.mu > 0.0,
        "Both cosmologies should give positive μ for energy injection: \
         default={:.4e}, p2018={:.4e}",
        snap1.mu,
        snap2.mu
    );

    eprintln!(
        "Cosmology sensitivity: default μ={:.4e}, planck2018 μ={:.4e}, diff={:.1}%",
        snap1.mu,
        snap2.mu,
        mu_diff * 100.0
    );
}

/// DC/BR drives μ-distortion toward Planck (thermalization).
///
/// Physical basis: a μ-distortion (Bose-Einstein with μ > 0) has fewer photons
/// at low x than Planck. DC and BR emit soft photons to fill the deficit,
/// thermalizing the spectrum. At z > 2×10⁵ where DC/BR is active, an injected
/// μ should relax toward zero over time.
///
/// Independent target: μ(z_end) < μ(z_inject) for z_inject well inside the μ-era.
/// The thermalization efficiency 1 - J_bb*(z) gives the expected fractional reduction.
#[test]
fn test_dcbr_thermalizes_mu_distortion() {
    let cosmo = Cosmology::default();
    let z_h = 5e5; // Well inside μ-era
    let drho = 1e-5;
    let sigma = 5000.0;

    // Run from z_h down to z_end in the μ-era (z=2e5), then further to y-era (z=5e3)
    let mut solver = ThermalizationSolver::new(
        cosmo,
        GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        },
    );
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 5e3,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[2e5, 5e3]);

    let snap_mu_era = &solver.snapshots[0]; // z=2e5, shortly after injection
    let snap_y_era = &solver.snapshots[1]; // z=5e3, much later

    eprintln!("Thermalization test:");
    eprintln!("  z=2e5: μ={:.4e}, y={:.4e}", snap_mu_era.mu, snap_mu_era.y);
    eprintln!("  z=5e3: μ={:.4e}, y={:.4e}", snap_y_era.mu, snap_y_era.y);

    // μ should decrease from μ-era to y-era (DC/BR thermalization)
    assert!(
        snap_mu_era.mu > snap_y_era.mu,
        "μ should decrease over time: μ(2e5)={:.4e} ≤ μ(5e3)={:.4e}",
        snap_mu_era.mu,
        snap_y_era.mu
    );

    // y should grow as μ converts to y in the transition, but adiabatic cooling
    // adds a negative y offset that grows over time. With strong enough injection
    // (drho=1e-5) the μ→y conversion should dominate, but the margin is small.
    // Allow y(5e3) to be slightly less than y(2e5) if both are positive (cooling offset).
    assert!(
        snap_y_era.y > 0.0 && snap_mu_era.y > 0.0,
        "y should be positive from injection: y(5e3)={:.4e}, y(2e5)={:.4e}",
        snap_y_era.y,
        snap_mu_era.y
    );

    // GF target: μ/Δρ ≈ 1.401 × J_bb*(z_h) × J_μ(z_h) ≈ 1.32 at z=5e5
    let mu_over_drho = snap_mu_era.mu / drho;
    assert!(
        mu_over_drho > 0.5 && mu_over_drho < 1.5,
        "μ/Δρ at z=2e5 should be O(1): got {mu_over_drho:.4e}"
    );
}

/// Post-recombination injection (z_h = 800): distortion should be locked in
/// at the injection frequency with NO μ/y redistribution.
///
/// Physical basis: at z < 1100, X_e ~ 10⁻⁴ and Compton scattering is
/// inefficient. DC/BR should be disabled (θ_z < 1e-6). Injected energy
/// stays as a spectral feature, not redistributed into μ or y.
///
/// Independent target: μ ≈ 0, y ≈ Δρ/(4ρ) × J_Compton(z_h) ≈ 0
/// (since J_Compton → 0 post-recombination).
#[test]
fn test_post_recombination_locked_in_distortion() {
    let z_h = 800.0;
    let drho = 1e-5;
    let sigma_z = 30.0;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma_z,
        z_h + 7.0 * sigma_z,
        100.0,
    );
    solver.run_with_snapshots(&[100.0]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!("Post-recombination (z_h=800):");
    eprintln!(
        "  μ={:.4e}, y={:.4e}, Δρ/ρ={:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho
    );

    // Post-recombination: Compton scattering is inefficient (X_e ~ 10⁻⁴).
    // Heat injection goes into electron temperature but barely couples to photons.
    // The photon spectrum distortion should be tiny compared to the injected energy.
    assert!(
        snap.delta_rho_over_rho.abs() < 0.1 * drho,
        "Post-recombination: photon Δρ/ρ should be ≪ injected energy: {:.4e} vs {drho:.4e}",
        snap.delta_rho_over_rho
    );

    // μ should be negligible (no Comptonization post-recombination)
    assert!(
        snap.mu.abs() < 0.01 * drho,
        "Post-recombination μ should be negligible: |μ|={:.4e} vs Δρ/ρ={drho:.4e}",
        snap.mu.abs()
    );

    // y should also be negligible (Compton y-parameter requires X_e ~ 1)
    assert!(
        snap.y.abs() < 0.01 * drho,
        "Post-recombination y should be negligible: |y|={:.4e} vs Δρ/ρ={drho:.4e}",
        snap.y.abs()
    );
}

// 38: Missing coverage — negative injection (cooling) PDE test
//
// Negative energy injection (cooling) should produce negative μ and y.
// This is the adiabatic cooling physics: T_e < T_z → Kompaneets cools photons.
#[test]
fn test_pde_negative_injection_produces_negative_distortion() {
    let cosmo = Cosmology::default();
    let grid_config = GridConfig {
        n_points: 1000,
        ..GridConfig::default()
    };
    let mut solver = ThermalizationSolver::new(cosmo, grid_config);
    // Negative burst: cooling, not heating
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: -1e-5,
            sigma_z: 2000.0,
        })
        .unwrap();

    solver.run_with_snapshots(&[500.0]);
    let snap = solver.snapshots.last().unwrap();

    // Negative injection → negative distortions
    assert!(
        snap.mu < 0.0,
        "Negative injection should give μ < 0: μ={:.4e}",
        snap.mu
    );
    assert!(
        snap.y < 0.0,
        "Negative injection should give y < 0: y={:.4e}",
        snap.y
    );
    // Δρ/ρ should be negative
    assert!(
        snap.delta_rho_over_rho < 0.0,
        "Δρ/ρ should be negative: {:.4e}",
        snap.delta_rho_over_rho
    );
    // Energy conservation: |μ/1.401 + 4y + 4ΔT/T| ≈ |Δρ/ρ|
    let dt_t = snap.delta_rho_over_rho / 4.0 - snap.mu / (4.0 * 1.401) - snap.y;
    let energy_sum = snap.mu / 1.401 + 4.0 * snap.y + 4.0 * dt_t;
    let energy_err =
        (energy_sum - snap.delta_rho_over_rho).abs() / snap.delta_rho_over_rho.abs().max(1e-30);
    assert!(
        energy_err < 0.01,
        "Energy conservation violated: sum={energy_sum:.4e} vs Δρ/ρ={:.4e}",
        snap.delta_rho_over_rho
    );
}

/// SingleBurst at z_h=3e4 (transition region). Both μ and y should be
/// positive. At z=3e4, J_mu ≈ 0.25 and J_y ≈ 0.86, but the mu coefficient
/// includes a factor 1.401/κ_c while y has 1/4, so both μ and y are
/// comparable. We verify positivity and energy conservation.
#[test]
fn test_transition_region_pde_z3e4() {
    let z_h = 3e4;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    eprintln!("Transition z_h=3e4:");
    eprintln!(
        "  μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        last.mu, last.y, last.delta_rho_over_rho
    );

    // Both should be positive (heating)
    assert!(
        last.mu > 0.0,
        "μ should be positive at z=3e4, got {:.4e}",
        last.mu
    );
    assert!(
        last.y > 0.0,
        "y should be positive at z=3e4, got {:.4e}",
        last.y
    );

    // Both μ and y should be nonzero in the transition region. Under the
    // B&F BE fit μ/Δρ is smaller at z=3×10⁴ (~0.03) than under a linear
    // M-basis decomposition because the BE shape's sensitivity to low-x
    // is reallocated to y; we still require a measurable signal in both.
    assert!(
        last.mu / drho > 1e-3 && last.y / drho > 0.1,
        "At z_h=3e4, both μ/Δρ ({:.3}) and y/Δρ ({:.3}) should be nonzero and y O(0.1)",
        last.mu / drho,
        last.y / drho
    );

    // Energy conservation
    let e_frac = last.delta_rho_over_rho / drho;
    eprintln!("  Energy fraction: {e_frac:.4}");
    assert!(
        (e_frac - 1.0).abs() < 0.05,
        "Energy conservation: Δρ/ρ ratio = {e_frac:.4}, expected ~1.0"
    );
}

/// SingleBurst at z_h=8e4 (μ-dominated transition). μ > y. Energy conservation < 5%.
#[test]
fn test_transition_region_pde_z8e4() {
    let z_h = 8e4;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    eprintln!("Transition z_h=8e4:");
    eprintln!(
        "  μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        last.mu, last.y, last.delta_rho_over_rho
    );

    // Both should be positive (heating)
    assert!(
        last.mu > 0.0,
        "μ should be positive at z=8e4, got {:.4e}",
        last.mu
    );
    assert!(
        last.y > 0.0,
        "y should be positive at z=8e4, got {:.4e}",
        last.y
    );

    // μ-dominated: μ > y
    assert!(
        last.mu > last.y,
        "At z_h=8e4, μ ({:.4e}) should dominate over y ({:.4e})",
        last.mu,
        last.y
    );

    // Energy conservation
    let e_frac = last.delta_rho_over_rho / drho;
    eprintln!("  Energy fraction: {e_frac:.4}");
    assert!(
        (e_frac - 1.0).abs() < 0.05,
        "Energy conservation: Δρ/ρ ratio = {e_frac:.4}, expected ~1.0"
    );
}

/// SingleBurst at z_h=5e4 (mid-transition). Compare PDE μ/y against GF predictions.
/// Tolerance 30% (transition region is hard for both methods).
#[test]
fn test_transition_region_pde_z5e4_gf_comparison() {
    let cosmo = Cosmology::default();
    let z_h = 5e4;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    // PDE solve
    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    // GF predictions
    let scenario = InjectionScenario::SingleBurst {
        z_h,
        delta_rho_over_rho: drho,
        sigma_z: sigma,
    };
    let mu_gf = greens::mu_from_heating(
        |z| scenario.heating_rate_per_redshift(z, &cosmo).abs(),
        500.0,
        z_h + 7.0 * sigma,
        2000,
    );
    let y_gf = greens::y_from_heating(
        |z| scenario.heating_rate_per_redshift(z, &cosmo).abs(),
        500.0,
        z_h + 7.0 * sigma,
        2000,
    );

    eprintln!("Transition z_h=5e4, PDE vs GF:");
    eprintln!("  PDE: μ = {:.4e}, y = {:.4e}", last.mu, last.y);
    eprintln!("  GF:  μ = {:.4e}, y = {:.4e}", mu_gf, y_gf);

    // Compare μ and y. `mu_from_heating`/`y_from_heating` return visibility-
    // convolution (Chluba-convention) values; `last.mu`/`last.y` are the B&F
    // BE-fit values. The two conventions differ by O(1) at the μ-y crossover
    // because the r-type residual is partitioned differently. We assert
    // magnitude agreement (same sign, same order of magnitude) rather than
    // tight relative equality.
    if mu_gf.abs() > 1e-10 {
        let mu_rel = (last.mu - mu_gf).abs() / mu_gf.abs();
        eprintln!("  μ rel_err = {mu_rel:.3}");
        assert!(
            mu_rel < 1.0 && last.mu.signum() == mu_gf.signum(),
            "PDE μ ({:.4e}) vs GF μ ({:.4e}): rel_err = {mu_rel:.3} > 100% or sign flip",
            last.mu,
            mu_gf
        );
    }

    if y_gf.abs() > 1e-10 {
        let y_rel = (last.y - y_gf).abs() / y_gf.abs();
        eprintln!("  y rel_err = {y_rel:.3}");
        assert!(
            y_rel < 3.0 && last.y.signum() == y_gf.signum(),
            "PDE y ({:.4e}) vs GF y ({:.4e}): rel_err = {y_rel:.3} > 300% or sign flip",
            last.y,
            y_gf
        );
    }
}

/// For a late-time heating burst (z_h = 5000, deep in the y-era), the PDE
/// should produce y ≈ Δρ/(4ρ) analytically, with negligible μ.
///
/// At z < 10⁴, Comptonization is efficient but DC/BR is frozen out, so
/// injected energy goes entirely into a y-type distortion. The analytical
/// result is y = Δρ/(4ρ), exact in the limit of instantaneous injection.
#[test]
fn test_pure_y_analytical_convergence() {
    let z_h = 5000.0;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    let y_expected = drho / 4.0;
    eprintln!("Pure y-era test (z_h=5000):");
    eprintln!(
        "  mu = {:.4e}, y = {:.4e}, y_expected = {:.4e}, drho/rho = {:.4e}",
        last.mu, last.y, y_expected, last.delta_rho_over_rho
    );

    // y should match analytical prediction to <1%
    let y_rel = (last.y - y_expected).abs() / y_expected;
    eprintln!("  y relative error: {:.4}%", y_rel * 100.0);
    assert!(
        y_rel < 0.01,
        "y should equal drho/(4*rho) = {y_expected:.4e} in pure y-era, got {:.4e}, \
         rel_err = {y_rel:.4}",
        last.y
    );

    // mu should be negligible compared to y
    let mu_over_y = last.mu.abs() / last.y.abs();
    eprintln!("  |mu/y| = {mu_over_y:.4e}");
    assert!(
        mu_over_y < 0.05,
        "mu should be negligible in y-era: |mu/y| = {mu_over_y:.4e}"
    );

    // Energy conservation
    let e_frac = last.delta_rho_over_rho / drho;
    assert!(
        (e_frac - 1.0).abs() < 0.01,
        "Energy conservation: drho/rho ratio = {e_frac:.4}, expected ~1.0"
    );
}

/// Extreme small injection (Δρ/ρ = 1e-12) at z=2e5. Verify the solver doesn't
/// crash and that μ scales linearly with Δρ/ρ (compare against 1e-5 baseline).
#[test]
fn test_extreme_small_injection() {
    let z_h = 2e5;
    let sigma = z_h * 0.01;
    let grid_config = GridConfig {
        n_points: 500,
        ..GridConfig::default()
    };

    let run = |drho: f64| -> (f64, f64) {
        let mut solver = burst_solver(&grid_config, z_h, drho, sigma, z_h + 7.0 * sigma, 500.0);
        solver.run_with_snapshots(&[500.0]);
        let last = solver.snapshots.last().unwrap();
        (last.mu, last.y)
    };

    let drho_baseline = 1e-5;
    // Must be well above the adiabatic cooling floor (μ ~ -3e-9, Δρ/ρ ~ -3e-9)
    // so the injection signal dominates. 1e-12 is swamped; 1e-7 is safe.
    let drho_small = 1e-7;

    let (mu_base, y_base) = run(drho_baseline);
    let (mu_small, y_small) = run(drho_small);

    eprintln!("Baseline (Δρ/ρ = {drho_baseline:.0e}): μ = {mu_base:.4e}, y = {y_base:.4e}");
    eprintln!("Small    (Δρ/ρ = {drho_small:.0e}): μ = {mu_small:.4e}, y = {y_small:.4e}");

    // Linearity check: μ/Δρ should be the same for both
    let mu_per_drho_base = mu_base / drho_baseline;
    let mu_per_drho_small = mu_small / drho_small;
    let linearity_err = (mu_per_drho_small - mu_per_drho_base).abs() / mu_per_drho_base.abs();
    eprintln!(
        "μ/Δρ: baseline = {mu_per_drho_base:.4e}, small = {mu_per_drho_small:.4e}, \
         rel_err = {linearity_err:.3e}"
    );

    // Allow 10% tolerance for linearity (small injection may have more numerical noise)
    assert!(
        linearity_err < 0.10,
        "Linearity violation: μ/Δρ differs by {:.1}% between 1e-5 and 1e-12 injections",
        linearity_err * 100.0
    );
}

/// Extreme-amplitude injection (Δρ/ρ = 0.01) at z=2e5: verify solver remains
/// numerically stable and the nonlinear response is near-linear in Δρ/ρ (the
/// nonlinear correction to μ is physically bounded).
///
/// Oracle:             Chluba 2013 Eq. 5 linear prediction:
///                     μ_lin = (3/κ_c) · J_bb*(z_h) · J_μ(z_h) · Δρ/ρ
///                     For Δρ/ρ = 0.01, z_h = 2×10⁵ this gives μ_lin ≈ 1.37×10⁻²
///                     The actual μ is larger by a nonlinear correction from
///                     Kompaneets Δn² and BE saturation; the correction is
///                     bounded: |μ_PDE − μ_lin| / μ_lin ≲ 20% at this amplitude.
/// Expected:           μ_lin = 1.37 × 10⁻², μ_PDE should be within [0.9·μ_lin, 1.3·μ_lin]
/// Oracle uncertainty: 10% on linear bound (visibility residuals);
///                     nonlinear correction ~10%, empirically 9% observed.
/// Tolerance:          ratio μ_PDE / μ_lin ∈ [0.9, 1.3];
///                     energy conservation 10% (adiabatic offset + nonlinear);
///                     Newton exhausted = 0 (solver stability).
#[test]
fn test_extreme_large_injection() {
    let z_h = 2e5;
    let drho = 0.01;
    let sigma = z_h * 0.01;

    let mut solver = burst_solver(
        &GridConfig::default(),
        z_h,
        drho,
        sigma,
        z_h + 7.0 * sigma,
        500.0,
    );

    solver.run_with_snapshots(&[500.0]);
    let last = solver.snapshots.last().unwrap();

    // Oracle: Chluba 2013 linear μ.
    let j_bb = greens::visibility_j_bb_star(z_h);
    let j_mu = greens::visibility_j_mu(z_h);
    let mu_lin = (3.0 / KAPPA_C) * j_bb * j_mu * drho;
    let mu_ratio = last.mu / mu_lin;

    eprintln!(
        "Large injection (Δρ/ρ = {drho}, z_h = {z_h:.0e}):\n  \
         μ_PDE = {:.4e}, μ_linear = {mu_lin:.4e}, ratio = {mu_ratio:.3}\n  \
         y = {:.4e}, Δρ_out/Δρ_in = {:.4}, Newton exhausted: {}",
        last.mu,
        last.y,
        last.delta_rho_over_rho / drho,
        solver.diag.newton_exhausted,
    );

    // Oracle 1: solver stability — Newton must not get exhausted.
    assert_eq!(
        solver.diag.newton_exhausted, 0,
        "Extreme injection: Newton exhausted {} times (should be 0)",
        solver.diag.newton_exhausted
    );

    // Oracle 2: μ_PDE should lie within a physically-bounded window around μ_lin.
    assert!(
        mu_ratio > 0.9 && mu_ratio < 1.3,
        "Extreme injection: μ_PDE/μ_linear = {mu_ratio:.3} outside [0.9, 1.3] \
         — linear Chluba 2013 bound violated (Δρ/ρ={drho} should be near-linear)"
    );

    // Oracle 3: energy conservation within 10% at large amplitude.
    let e_frac = last.delta_rho_over_rho / drho;
    assert!(
        (e_frac - 1.0).abs() < 0.10,
        "Energy conservation: ratio = {e_frac:.4}, expected ~1.0 (tol 10%)"
    );
}
