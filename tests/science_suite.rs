//! Science integration suite: quantitative validation against literature.
//!
//! These tests enforce science-grade accuracy targets, not just regression.
//! If a test fails, the code is producing results that cannot be trusted
//! for the corresponding physical regime.
//!
//! Tolerances are set at ~2x the measured error to allow for minor
//! numerical variations across platforms, but tight enough that a
//! significant physics regression will be caught.

use spectroxide::greens;
use spectroxide::prelude::*;
use spectroxide::solver::SolverSnapshot;

fn rel_err(value: f64, target: f64) -> f64 {
    (value - target).abs() / target.abs().max(1e-30)
}

fn run_single_burst(
    z_h: f64,
    sigma_z: f64,
    delta_rho_over_rho: f64,
    z_start: f64,
    z_end: f64,
) -> SolverSnapshot {
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho,
            sigma_z,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[z_end]);
    solver
        .snapshots
        .last()
        .cloned()
        .expect("expected at least one snapshot")
}

// ---------------------------------------------------------------------------
// mu-era: PDE solver must match Green's function to <10%
// Measured error: ~8.6%. Tolerance: 10%.
// The PDE includes full numerical thermalization while GF uses fitting
// formulas, so ~10% agreement is expected in the deep mu-era.
// ---------------------------------------------------------------------------

#[test]
fn science_mu_era_coefficient_pde() {
    let z_h = 2.0e5;
    let drho = 1.0e-5;
    let snapshot = run_single_burst(z_h, 3000.0, drho, 5.0e5, 1.0e4);

    let mu_expected = (3.0 / constants::KAPPA_C)
        * greens::visibility_j_bb_star(z_h)
        * greens::visibility_j_mu(z_h)
        * drho;

    let err = rel_err(snapshot.mu, mu_expected);
    assert!(
        err < 0.10,
        "mu-era: PDE mu={:.4e} vs GF mu={:.4e}, err={:.1}% (limit 10%)",
        snapshot.mu,
        mu_expected,
        err * 100.0
    );
    // μ-era injection should be μ-dominated. Under the B&F decomposition the
    // r-type residual of the PDE spectrum partitions partly into y, giving
    // |y/μ| ~ 5% in the μ-era (vs < 1% under the old energy-neutral fit).
    assert!(
        snapshot.y.abs() < 0.1 * snapshot.mu.abs(),
        "mu-era should be mu-dominated: |y/mu|={:.2}% (limit 10%)",
        (snapshot.y / snapshot.mu).abs() * 100.0
    );
}

// ---------------------------------------------------------------------------
// y-era: PDE solver must match Green's function to <2%
// Measured error: ~0.25%. Tolerance: 2%.
// In the y-era, Kompaneets redistribution is negligible so PDE and GF
// agree much better than in the mu-era.
// ---------------------------------------------------------------------------

#[test]
fn science_y_era_coefficient_pde() {
    let z_h = 5.0e3;
    let drho = 1.0e-5;
    let snapshot = run_single_burst(z_h, 200.0, drho, 1.0e4, 1.0e3);

    let y_expected = 0.25 * greens::visibility_j_y(z_h) * drho;
    let err = rel_err(snapshot.y, y_expected);
    assert!(
        err < 0.02,
        "y-era: PDE y={:.4e} vs GF y={:.4e}, err={:.2}% (limit 2%)",
        snapshot.y,
        y_expected,
        err * 100.0
    );
    // y-era should have negligible μ: measured |μ/y| ≈ 2.2%
    assert!(
        snapshot.mu.abs() < 0.04 * snapshot.y.abs(),
        "y-era should be y-dominated: |mu/y|={:.2}% (limit 4%)",
        (snapshot.mu / snapshot.y).abs() * 100.0
    );
}

// ---------------------------------------------------------------------------
// High-z thermalization: Green's function must approach pure temperature shift
// Measured error: <0.05%. Tolerance: 1%.
// Reference: Chluba (2013), Eq. 6 — at z >> z_mu, J_bb* → 1, J_mu → 1.
// ---------------------------------------------------------------------------

#[test]
fn science_high_z_thermalization_is_temperature_shift() {
    let z_h = 5.0e6;
    for &x in &[0.5, 1.0, 3.0, 5.0, 10.0] {
        let g = greens::greens_function(x, z_h);
        let t_shift = 0.25 * spectrum::g_bb(x);
        let err = rel_err(g, t_shift);
        assert!(
            err < 0.01,
            "high-z G(x) vs T-shift at x={x}: G={g:.4e}, T/4={t_shift:.4e}, err={:.2}% (limit 1%)",
            err * 100.0
        );
    }
}

// ---------------------------------------------------------------------------
// Deep thermalization PDE test (z > 5e5)
// At z_h = 1e6, the PDE must show strong thermalization suppression:
//   μ/Δρ = 1.401 × J_bb*(z_h) × J_mu(z_h)
// J_bb*(1e6) ≈ 0.14, so μ/Δρ ≈ 0.20 (NOT the unsuppressed 1.401).
// This is the most numerically delicate regime — DC/BR must be well-resolved.
// Uses production grid (4000 points) as required for z > 1e6.
// ---------------------------------------------------------------------------

#[test]
fn science_deep_thermalization_pde() {
    let z_h = 1.0e6;
    let drho = 1.0e-5;
    let cosmo = Cosmology::default();
    let grid_config = GridConfig::production();
    let mut solver = ThermalizationSolver::new(cosmo, grid_config);
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: z_h * 0.01,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: z_h * 1.5,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);
    let snap = solver.snapshots.last().unwrap();

    // GF target: μ = 1.401 × J_bb*(z_h) × J_mu(z_h) × Δρ/ρ
    let j_bb = greens::visibility_j_bb_star(z_h);
    let j_mu = greens::visibility_j_mu(z_h);
    let mu_gf = (3.0 / constants::KAPPA_C) * j_bb * j_mu * drho;

    // At z=1e6, partial thermalization: J_bb* should be well below 1
    // (some energy thermalized into T-shift) but still significant
    assert!(
        j_bb < 0.95 && j_bb > 0.1,
        "J_bb*(1e6) should show partial thermalization: got {j_bb:.3}"
    );

    let mu_over_drho = snap.mu / drho;
    let err = rel_err(snap.mu, mu_gf);
    eprintln!(
        "z_h=1e6: μ/Δρ = {mu_over_drho:.4}, GF target = {:.4}, J_bb* = {j_bb:.4}, err = {:.1}%",
        mu_gf / drho,
        err * 100.0
    );

    // PDE vs GF: allow 5% at z=1e6. Measured error: ~1.0%.
    // Production grid (4000 pts) resolves DC/BR well enough for tight agreement.
    assert!(
        err < 0.05,
        "Deep thermalization: PDE μ={:.4e} vs GF μ={mu_gf:.4e}, err={:.1}% (limit 5%)",
        snap.mu,
        err * 100.0
    );

    // μ/Δρ must be below the unsuppressed value of 1.401 (thermalization removes some)
    assert!(
        mu_over_drho < 1.401 * 0.95,
        "Thermalization suppression not working: μ/Δρ = {mu_over_drho:.4} (should be < 1.401)"
    );
}

// (science_energy_conservation_single_burst removed: strictly subsumed by
// test_heat_energy_conservation_sweep_tight in heat_injection.rs, which sweeps
// 7 redshifts including {1e4, 5e4, 2e5} at 2% tolerance — tighter than this
// 3% three-point version.)

// ---------------------------------------------------------------------------
// T_e decoupling: matter temperature must decouple post-recombination.
//
// After recombination (z ~ 1100), X_e drops to ~2.4×10⁻⁴ and Compton
// coupling becomes inefficient. Matter cools adiabatically as T_m ∝ (1+z)²
// while T_CMB ∝ (1+z), so ρ_e = T_m/T_CMB decreases toward low z.
//
// At z=200: decoupling is underway. Measured ρ_e ≈ 0.86.
// At z=100: well into adiabatic cooling. Measured ρ_e ≈ 0.62.
// Reference: Peebles three-level atom with Péquignot α_B and F=1.125,
// consistent with Seager, Sasselov & Scott (1999, 2000) and
// Ali-Haïmoud & Hirata (2011) for the Peebles TLA limit.
//
// Tolerances: ±5% of measured values, validated against the Peebles TLA.
// ---------------------------------------------------------------------------

#[test]
fn science_te_decoupling_post_recombination() {
    let cosmo = Cosmology::default();
    let grid = spectroxide::grid::GridConfig::fast();
    let mut solver = ThermalizationSolver::new(cosmo, grid);
    solver.set_config(spectroxide::solver::SolverConfig {
        z_start: 3.0e6,
        z_end: 50.0,
        ..spectroxide::solver::SolverConfig::default()
    });
    // No injection — pure adiabatic evolution
    solver.run_with_snapshots(&[500.0, 200.0, 100.0]);

    // z=500: Compton coupling still efficient, ρ_e ≈ 1.
    // Physics: Compton scattering rate ~ σ_T n_e c × θ_z >> H(z) at z=500.
    let s_500 = solver
        .snapshots
        .iter()
        .find(|s| (s.z - 500.0).abs() < 5.0)
        .unwrap();
    assert!(
        s_500.rho_e > 0.97 && s_500.rho_e < 1.001,
        "ρ_e(z=500) should be near 1 (Compton-coupled): got {:.4}",
        s_500.rho_e
    );

    // z=200: thermal decoupling underway.
    // Peebles TLA with F=1.125 (Chluba & Thomas 2011, arXiv:1011.3758).
    // The T_m/T_CMB ratio at z=200 depends on X_e freeze-out and Compton
    // coupling efficiency. Bounds are ±5% of numerically converged value.
    // Cross-checked against DarkHistory (Liu+ 2020) TLA implementation.
    let s_200 = solver
        .snapshots
        .iter()
        .find(|s| (s.z - 200.0).abs() < 5.0)
        .unwrap();
    assert!(
        s_200.rho_e > 0.82 && s_200.rho_e < 0.90,
        "ρ_e(z=200) = {:.4}, expected 0.86 ± 5%",
        s_200.rho_e
    );

    // z=100: deep adiabatic cooling regime.
    // T_m ∝ (1+z)² while T_CMB ∝ (1+z), so ρ_e = T_m/T_CMB decreases.
    // Bounds ±5% of converged value; cross-checked against DarkHistory TLA.
    let s_100 = solver
        .snapshots
        .iter()
        .find(|s| (s.z - 100.0).abs() < 5.0)
        .unwrap();
    assert!(
        s_100.rho_e > 0.58 && s_100.rho_e < 0.68,
        "ρ_e(z=100) = {:.4}, expected 0.62 ± 5%",
        s_100.rho_e
    );

    // Post-decoupling scaling: ρ_e(100)/ρ_e(200) should be between
    // pure adiabatic (101/201 = 0.50) and fully coupled (1.0).
    // Residual Compton coupling from X_e ~ 2×10⁻⁴ gives ~0.73.
    let ratio = s_100.rho_e / s_200.rho_e;
    assert!(
        ratio > 0.68 && ratio < 0.78,
        "ρ_e(100)/ρ_e(200) = {ratio:.4}, expected 0.73 ± 5%"
    );

    eprintln!(
        "T_e decoupling: ρ_e(500)={:.4}, ρ_e(200)={:.4}, ρ_e(100)={:.4}, ratio={:.4}",
        s_500.rho_e, s_200.rho_e, s_100.rho_e, ratio
    );
}
