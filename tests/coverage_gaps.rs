//! Tests filling coverage gaps identified during repo health audit.
//!
//! Organization:
//!   Section 1  — Energy conservation per injection scenario
//!   Section 2  — extract_y() validation
//!   Section 4  — InjectionScenario metadata query functions
//!   Section 5  — Grid edge cases (refinement, log_fraction, find_index, builder)
//!   Section 7  — Newton iteration exhaustion
//!   Section 8  — Boundary condition verification
//!   Section 9  — Coupled vs decoupled DC/BR comparison
//!
//! (Sections 3 and 6 removed in 2026-04 test-suite triage: serialization
//! plumbing and tabulated-file error-path tests were D/F-grade coverage
//! padding — no physics content, and `std::fs` / string-match assertions
//! don't catch physics bugs.)

use spectroxide::constants::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::solver::{
    ENERGY_CHECK_MAX_LATE_FRACTION, ENERGY_CHECK_Z_LATE, ENERGY_CLOSURE_ABS_FLOOR,
    ENERGY_CLOSURE_REL_TOL, SolverConfig, ThermalizationSolver,
};

// ============================================================================
// Section 1: Energy conservation per injection scenario
//
// For each scenario, verify that the PDE output's Δρ/ρ matches the injected
// amount to within a reasonable tolerance. The injected amount is computed
// from first principles for each scenario type.
// ============================================================================

// (run_single_burst helper removed along with its three callers; see below.)

// (energy_conservation_single_burst_{y_era,mu_era,transition} removed:
// strictly subsumed by test_heat_energy_conservation_sweep_tight in
// heat_injection.rs, which sweeps 7 redshifts including {5e3, 5e4, 2e5}
// at 2% tolerance — tighter than each of the singletons it replaces.)

#[test]
fn energy_conservation_decaying_particle() {
    // Decaying particle: total injected energy depends on lifetime vs cosmic time.
    // Use a short-lived particle (z_X ~ 1e5) so most energy is deposited.
    let cosmo = Cosmology::default();
    // f_x = 1e5 eV per baryon gives integrated Δρ/ρ ~ 5×10⁻⁷, well above the
    // adiabatic-cooling floor ~3×10⁻⁹ so the PDE signal isn't swamped by noise.
    let f_x = 1e5;
    let t_at_z = cosmo.cosmic_time(1e5);
    let gamma_x = 1.0 / (t_at_z * 0.5); // half-life at z ~ 1e5

    let z_start = 3e6_f64;
    let z_end = 500.0_f64;

    // Independent reference: integrate the analytic heating rate over cosmic time
    // to get total injected Δρ/ρ. Tolerance 10% accounts for numerical integration
    // error accumulating over z=[3e6, 500] plus small adiabatic-cooling offset.
    //   Δρ/ρ = ∫ heating_rate(z) × dt  = ∫_{z_end}^{z_start} heating_rate(z) / (H(z)·(1+z)) dz
    let scenario = InjectionScenario::DecayingParticle { f_x, gamma_x };
    let n_z = 2000;
    let log_z_min = z_end.ln();
    let log_z_max = z_start.ln();
    let mut drho_expected = 0.0;
    for i in 0..n_z {
        let lz_a = log_z_min + (i as f64 / n_z as f64) * (log_z_max - log_z_min);
        let lz_b = log_z_min + ((i + 1) as f64 / n_z as f64) * (log_z_max - log_z_min);
        let z_a = lz_a.exp();
        let z_b = lz_b.exp();
        let z_mid = 0.5 * (z_a + z_b);
        let rate = scenario.heating_rate(z_mid, &cosmo);
        // dz = z_mid × d(ln z); integrand: rate/[H(1+z)] × dz
        drho_expected += rate / (cosmo.hubble(z_mid) * (1.0 + z_mid)) * (z_b - z_a);
    }

    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::DecayingParticle { f_x, gamma_x })
        .z_range(z_start, z_end)
        .build()
        .unwrap();
    let result = solver.run_to_result(z_end);
    let s = &result.snapshot;

    let rel_err = (s.delta_rho_over_rho - drho_expected).abs() / drho_expected;
    eprintln!(
        "DecayingParticle: solver drho = {:.6e}, integrated drho = {:.6e}, rel err = {:.2}%",
        s.delta_rho_over_rho,
        drho_expected,
        rel_err * 100.0
    );
    assert!(
        s.delta_rho_over_rho > 0.0 && s.delta_rho_over_rho.is_finite(),
        "DecayingParticle drho must be positive and finite: {:.6e}",
        s.delta_rho_over_rho
    );
    // 10% tolerance on integrated-injection reference (trapezoid error plus
    // O(1%) adiabatic-cooling offset at the solver's z_end).
    assert!(
        rel_err < 0.10,
        "DecayingParticle energy conservation: solver Δρ/ρ={:.6e} vs integrated={:.6e}, err={:.2}% > 10%",
        s.delta_rho_over_rho,
        drho_expected,
        rel_err * 100.0
    );
    assert!(
        s.mu > 0.0,
        "DecayingParticle mu should be positive (heating in μ-era): got {:.6e}",
        s.mu
    );
}

#[test]
fn energy_conservation_annihilating_dm_swave() {
    let cosmo = Cosmology::default();
    // Use a large enough f_ann to dominate over adiabatic cooling.
    // Adiabatic cooling gives Δρ/ρ ~ -4e-9 from z=3e6. We need f_ann
    // that injects significantly more energy.
    let f_ann = 2e-22; // eV/s — gives ~10⁻⁵ level distortion
    let z_start = 3e6_f64;
    let z_end = 500.0_f64;

    // Independent reference: integrate heating_rate over cosmic time to get
    // total injected Δρ/ρ. (The prior "decomposition identity" assertion was
    // tautological — ΔT/T was computed by the solver to make that sum
    // numerically exact, regardless of physics.)
    let scenario = InjectionScenario::AnnihilatingDM { f_ann };
    let n_z = 2000;
    let log_z_min = z_end.ln();
    let log_z_max = z_start.ln();
    let mut drho_expected = 0.0;
    for i in 0..n_z {
        let lz_a = log_z_min + (i as f64 / n_z as f64) * (log_z_max - log_z_min);
        let lz_b = log_z_min + ((i + 1) as f64 / n_z as f64) * (log_z_max - log_z_min);
        let z_a = lz_a.exp();
        let z_b = lz_b.exp();
        let z_mid = 0.5 * (z_a + z_b);
        let rate = scenario.heating_rate(z_mid, &cosmo);
        drho_expected += rate / (cosmo.hubble(z_mid) * (1.0 + z_mid)) * (z_b - z_a);
    }

    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::AnnihilatingDM { f_ann })
        .z_range(z_start, z_end)
        .build()
        .unwrap();
    let result = solver.run_to_result(z_end);
    let s = &result.snapshot;

    let rel_err = (s.delta_rho_over_rho - drho_expected).abs() / drho_expected.abs();
    eprintln!(
        "AnnihilatingDM: solver drho = {:.6e}, integrated drho = {:.6e}, rel err = {:.2}%; mu={:.6e}, y={:.6e}",
        s.delta_rho_over_rho,
        drho_expected,
        rel_err * 100.0,
        s.mu,
        s.y
    );

    assert!(
        s.delta_rho_over_rho > 0.0 && s.delta_rho_over_rho.is_finite(),
        "AnnihilatingDM drho should be positive and finite: got {:.6e}",
        s.delta_rho_over_rho
    );
    // 15% tolerance: continuous injection running from z=3e6 to 500 accumulates
    // numerical error across the full thermalization + μ + y eras. For a single
    // SingleBurst in one era, test_heat_energy_conservation_sweep_tight uses 2%.
    assert!(
        rel_err < 0.15,
        "AnnihilatingDM energy conservation: solver Δρ/ρ={:.6e} vs integrated={:.6e}, err={:.2}% > 15%",
        s.delta_rho_over_rho,
        drho_expected,
        rel_err * 100.0
    );
    // Continuous injection spans μ-era and y-era → both components present.
    assert!(
        s.mu > 0.0 && s.y > 0.0,
        "AnnihilatingDM: both μ and y should be positive: μ={:.6e}, y={:.6e}",
        s.mu,
        s.y
    );
}

#[test]
fn energy_conservation_annihilating_dm_pwave() {
    let cosmo = Cosmology::default();
    let f_ann = 2e-25;

    let mut solver = ThermalizationSolver::builder(cosmo.clone())
        .grid(GridConfig::default())
        .injection(InjectionScenario::AnnihilatingDMPWave { f_ann })
        .z_range(3e6, 500.0)
        .build()
        .unwrap();
    let result = solver.run_to_result(500.0);
    let s = &result.snapshot;

    assert!(
        s.delta_rho_over_rho > 0.0 && s.delta_rho_over_rho.is_finite(),
        "AnnihilatingDMPWave drho should be positive: got {:.6e}",
        s.delta_rho_over_rho
    );

    // p-wave rate ∝ (1+z)³ vs s-wave ∝ (1+z)², so p-wave should deposit
    // more energy at high z (mu-era) relative to low z (y-era).
    let mut solver_s = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::AnnihilatingDM { f_ann })
        .z_range(3e6, 500.0)
        .build()
        .unwrap();
    let result_s = solver_s.run_to_result(500.0);
    let s_s = &result_s.snapshot;

    // p-wave injects more energy (extra factor of (1+z))
    assert!(
        s.delta_rho_over_rho > s_s.delta_rho_over_rho,
        "p-wave should inject more energy than s-wave: p={:.6e}, s={:.6e}",
        s.delta_rho_over_rho,
        s_s.delta_rho_over_rho
    );
    // p-wave should have higher mu/drho ratio (more high-z weighted)
    let mu_ratio_p = s.mu / s.delta_rho_over_rho;
    let mu_ratio_s = s_s.mu / s_s.delta_rho_over_rho;
    eprintln!("p-wave: mu/drho={mu_ratio_p:.4}, s-wave: mu/drho={mu_ratio_s:.4}");
    assert!(
        mu_ratio_p > mu_ratio_s,
        "p-wave should have higher mu/drho than s-wave: p={mu_ratio_p:.4}, s={mu_ratio_s:.4}"
    );
}

#[test]
fn energy_conservation_photon_injection_x1() {
    // Monochromatic photon injection at x=1 (near spectral peak).
    // The injection adds ΔN/N photons as a Gaussian in x centered at x_inj.
    // The source profile is (ΔN/N) × G₂/x² × gauss(x; x_inj, σ_x).
    // Energy injected: Δρ/ρ = (ΔN/N) × G₂ × x_inj / G₃ (for narrow Gaussian).
    let cosmo = Cosmology::default();
    let x_inj = 1.0;
    let dn_over_n = 1e-5;
    let z_h = 5e4;
    let sigma_z = z_h * 0.01;

    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig {
            n_points: 4000,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::MonochromaticPhotonInjection {
            x_inj,
            delta_n_over_n: dn_over_n,
            z_h,
            sigma_z,
            sigma_x: 0.05,
        })
        .z_range(z_h + 7.0 * sigma_z, 500.0)
        .build()
        .unwrap();
    let result = solver.run_to_result(500.0);
    let s = &result.snapshot;

    // Expected Δρ/ρ: ∫ x³ × (G₂/x²) × gauss(x) dx / G₃ × ΔN/N
    // ≈ G₂ × x_inj / G₃ × ΔN/N for narrow Gaussian
    let expected_drho = dn_over_n * G2_PLANCK * x_inj / G3_PLANCK;
    let err = (s.delta_rho_over_rho / expected_drho - 1.0).abs();
    eprintln!(
        "PhotonInjection x=1: drho={:.6e}, expected={expected_drho:.6e}, err={err:.2e}",
        s.delta_rho_over_rho
    );
    // The scenario's source is G₂·gauss(x)/x², so ∫x³Δn dx / G₃ = α_ρ x_inj ΔN/N
    // is exact for it — no finite-σ_x correction (unlike the x_inj²-normalised
    // helper in tests/heat_injection.rs; see dev/audit/energy_conservation_audit.md).
    // Measured err = 3.4e-4; tolerance 2e-3 leaves 6× for the first-order
    // temporal residual to grow with z_h.
    assert!(
        err < 2e-3,
        "Photon injection energy conservation at x=1: err = {err:.2e} > 0.2%"
    );
}

#[test]
fn energy_conservation_tabulated_heating() {
    // Create a tabulated heating scenario that mimics a Gaussian burst.
    let cosmo = Cosmology::default();
    let z_h = 5e4_f64;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    // Build a table of dq/dz values
    let n_z = 200;
    let z_lo = z_h - 5.0 * sigma;
    let z_hi = z_h + 5.0 * sigma;
    let z_table: Vec<f64> = (0..n_z)
        .map(|i| z_lo + (z_hi - z_lo) * i as f64 / (n_z - 1) as f64)
        .collect();
    let rate_table: Vec<f64> = z_table
        .iter()
        .map(|&z| {
            let g = (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
            drho * g
        })
        .collect();

    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::TabulatedHeating {
            z_table,
            rate_table,
        })
        .z_range(z_hi + sigma, 500.0)
        .build()
        .unwrap();
    let result = solver.run_to_result(500.0);
    let s = &result.snapshot;
    let err = (s.delta_rho_over_rho / drho - 1.0).abs();
    eprintln!(
        "TabulatedHeating: drho={:.6e}, expected={drho:.6e}, err={err:.2e}",
        s.delta_rho_over_rho
    );
    // Measured err = 1.7e-3 (the trapezoidal table interpolation plus the
    // first-order temporal residual); tolerance 5e-3 keeps 3× margin.
    assert!(
        err < 5e-3,
        "TabulatedHeating energy conservation: err = {err:.2e} > 0.5%"
    );
}

// ============================================================================
// Section 2: extract_y() validation
// ============================================================================

#[test]
fn extract_y_matches_joint() {
    let cosmo = Cosmology::default();
    let z_h = 5e3;
    let drho = 1e-5;
    let sigma = z_h * 0.01;
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: z_h + 7.0 * sigma,
        z_end: 500.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[500.0]);

    let (mu_standalone, y_standalone) = solver.extract_mu_y_joint();

    // Cross-check against snapshot values (which use the same decomposition).
    let snap = solver.snapshots.last().unwrap();
    assert!(
        (y_standalone - snap.y).abs() < 1e-15,
        "extract_mu_y_joint() y should match snapshot: standalone={y_standalone:.10e}, snap={:.10e}",
        snap.y
    );
    assert!(
        (mu_standalone - snap.mu).abs() < 1e-15,
        "extract_mu_y_joint() μ should match snapshot: standalone={mu_standalone:.10e}, snap={:.10e}",
        snap.mu
    );
    // In y-era, y should dominate
    assert!(
        y_standalone > 0.0,
        "y should be positive in y-era: got {y_standalone:.6e}"
    );
    eprintln!("extract: mu={mu_standalone:.6e}, y={y_standalone:.6e}");
}

// ============================================================================
// Section 4: heating_rate_per_redshift sign convention
// (characteristic_redshift and suggested_x_min tests removed in 2026-04 triage:
// D-grade API-contract tests that assert `metadata_query() == input_field` —
// they test the struct pattern-match, not physics.)
// ============================================================================

#[test]
fn heating_rate_per_redshift_sign_convention() {
    // For positive energy injection, heating_rate > 0 but dq/dz < 0
    // because energy enters as z decreases.
    let cosmo = Cosmology::default();
    let burst = InjectionScenario::SingleBurst {
        z_h: 1e5,
        delta_rho_over_rho: 1e-5,
        sigma_z: 1e3,
    };
    let rate = burst.heating_rate(1e5, &cosmo);
    let rate_per_z = burst.heating_rate_per_redshift(1e5, &cosmo);
    assert!(rate > 0.0, "heating_rate should be positive at z_h");
    assert!(
        rate_per_z < 0.0,
        "heating_rate_per_redshift should be negative (energy enters as z decreases)"
    );
    // |dq/dz| = rate / (H(1+z))
    let expected_abs = rate / (cosmo.hubble(1e5) * (1.0 + 1e5));
    assert!(
        (rate_per_z.abs() / expected_abs - 1.0).abs() < 1e-10,
        "Magnitude should match: got {:.6e}, expected {expected_abs:.6e}",
        rate_per_z.abs()
    );
}

// ============================================================================
// Section 5: Grid find_index
// (builder/overlap/log_fraction tests removed in 2026-04 triage: D/F-grade
// tests that asserted monotonicity or `n > n_base` — properties that cannot
// detect any real bug in the refinement logic.)
// ============================================================================

#[test]
fn grid_find_index_exact_match() {
    let grid = FrequencyGrid::uniform(0.0, 10.0, 11); // 0, 1, 2, ..., 10
    assert_eq!(grid.find_index(5.0), 5);
    assert_eq!(grid.find_index(0.0), 0);
    assert_eq!(grid.find_index(10.0), 10);
}

#[test]
fn grid_find_index_between_points() {
    let grid = FrequencyGrid::uniform(0.0, 10.0, 11);
    // 4.3 is closer to 4.0 than 5.0
    assert_eq!(grid.find_index(4.3), 4);
    // 4.7 is closer to 5.0 than 4.0
    assert_eq!(grid.find_index(4.7), 5);
}

#[test]
fn grid_find_index_below_min() {
    let grid = FrequencyGrid::uniform(1.0, 10.0, 10);
    assert_eq!(grid.find_index(0.5), 0, "Below x_min should return 0");
    assert_eq!(grid.find_index(-1.0), 0, "Negative should return 0");
}

#[test]
fn grid_find_index_above_max() {
    let grid = FrequencyGrid::uniform(1.0, 10.0, 10);
    assert_eq!(
        grid.find_index(100.0),
        grid.n - 1,
        "Above x_max should return last index"
    );
}

// ============================================================================
// Section 7: Newton iteration exhaustion
// ============================================================================

#[test]
fn newton_exhaustion_recorded_in_diagnostics() {
    // With a large-amplitude burst and the minimum max_newton_iter=2, at least
    // one step in the μ-era should exhaust the Newton budget. Verify that
    // (a) the exhaustion counter actually increments, and (b) the solver
    // completes with finite output despite exhausted steps.
    //
    // (The prior version only checked finiteness and printed the exhaustion
    // count — it would have passed even if `diag_newton_exhausted` were
    // hardcoded to 0, leaving the diagnostic plumbing untested.)
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid_fast()
        .injection(InjectionScenario::SingleBurst {
            z_h: 2e5,
            delta_rho_over_rho: 1e-4,
            sigma_z: 2e3,
        })
        .z_range(2.14e5, 1e4)
        .max_newton_iter(2)
        .build()
        .unwrap();
    let result = solver.run_to_result(1e4);
    eprintln!(
        "Newton exhausted: {} / {} steps",
        result.diag_newton_exhausted, result.step_count
    );
    let s = &result.snapshot;
    assert!(
        s.mu.is_finite() && s.y.is_finite(),
        "mu, y should be finite even with few Newton iters: mu={}, y={}",
        s.mu,
        s.y
    );
    // Physics check: with drho=1e-4 (10× the standard test amplitude) and
    // max_newton_iter=2, the μ-era nonlinear Newton iteration should exhaust
    // at least once — confirming the counter is actually wired up.
    assert!(
        result.diag_newton_exhausted >= 1,
        "Expected diag_newton_exhausted ≥ 1 with max_newton_iter=2 and drho=1e-4: got {}",
        result.diag_newton_exhausted
    );
}

// ============================================================================
// Section 8: Boundary condition verification
// ============================================================================

#[test]
fn boundary_conditions_delta_n_zero_at_edges() {
    let cosmo = Cosmology::default();
    let z_h = 5e4;
    let drho = 1e-5;
    let sigma = z_h * 0.01;
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        })
        .z_range(z_h + 7.0 * sigma, 500.0)
        .build()
        .unwrap();
    let result = solver.run_to_result(500.0);
    let s = &result.snapshot;
    let n = s.delta_n.len();

    // Dirichlet BC: Δn at the HIGH-x boundary should be essentially zero.
    // The Kompaneets operator pins Δn=0 at both edges, but DC/BR emission
    // creates a physical low-x Rayleigh-Jeans tail, so Δn[0] can be nonzero.
    // At x_max ~ 50, n_pl ~ e^{-50} ≈ 0, so distortions cannot propagate there.
    assert!(
        s.delta_n[n - 1].abs() < 1e-20,
        "Δn at x_max should be ~0: got {:.6e}",
        s.delta_n[n - 1]
    );
    // Interior should have nonzero signal
    let max_interior = s.delta_n[1..n - 1]
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(
        max_interior > 1e-20,
        "Interior Δn should be nonzero: max = {max_interior:.6e}"
    );
    // Low-x: Δn can be nonzero from DC/BR, but should be bounded
    eprintln!(
        "Δn at x_min = {:.6e}, Δn at x_max = {:.6e}",
        s.delta_n[0],
        s.delta_n[n - 1]
    );
}

// ============================================================================
// Section 9: Coupled vs decoupled DC/BR comparison
// ============================================================================

#[test]
fn coupled_vs_split_dcbr_consistency() {
    let cosmo = Cosmology::default();
    let z_h = 2e5;
    let drho = 1e-5;
    let sigma = z_h * 0.01;

    let run = |split: bool| -> (f64, f64, f64) {
        let mut builder = ThermalizationSolver::builder(cosmo.clone())
            .grid(GridConfig::default())
            .injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: sigma,
            })
            .z_range(z_h + 7.0 * sigma, 500.0);
        if split {
            builder = builder.split_dcbr();
        }
        let mut solver = builder.build().unwrap();
        let result = solver.run_to_result(500.0);
        let s = &result.snapshot;
        (s.mu, s.y, s.delta_rho_over_rho)
    };

    let (mu_coupled, y_coupled, drho_coupled) = run(false);
    let (mu_split, y_split, drho_split) = run(true);

    eprintln!("Coupled: mu={mu_coupled:.6e}, y={y_coupled:.6e}, drho={drho_coupled:.6e}");
    eprintln!("Split:   mu={mu_split:.6e}, y={y_split:.6e}, drho={drho_split:.6e}");

    let mu_rel = (mu_coupled - mu_split).abs() / mu_coupled.abs().max(1e-30);
    let drho_rel = (drho_coupled - drho_split).abs() / drho_coupled.abs().max(1e-30);

    eprintln!("mu relative diff: {mu_rel:.4e}");
    eprintln!("drho relative diff: {drho_rel:.4e}");

    // Coupled (Newton-coupled Kompaneets+DC/BR) and split (operator-split
    // backward-Euler DC/BR after a Crank-Nicolson Kompaneets) use different
    // numerical schemes. They should agree on the integrated physics but not
    // step-for-step: mu in particular picks up a small splitting-error bias
    // at z ~ 500 where DC/BR is tiny but non-zero, because the Taylor branch
    // threshold |ρ−1|<0.01 activates at slightly different times in each
    // scheme. Energy conservation (Δρ/ρ) is unaffected and is held to 5%
    // below as the physically conserved quantity.
    assert!(
        mu_rel < 0.25,
        "Coupled vs split mu should agree to <25%: rel = {mu_rel:.4e}"
    );
    assert!(
        drho_rel < 0.05,
        "Coupled vs split drho should agree to <5%: rel = {drho_rel:.4e}"
    );
}

// ============================================================================
// Section 10: post-run energy-closure and small-grid warnings (R-1), and
// progress messages kept out of `warnings` (R-5)
//
// Every target below is the injected energy, which the test fixes by
// construction (a burst's Δρ/ρ, a constant table's rate × Δz), not a value
// read from the solver.
// ============================================================================

fn has_warning(warnings: &[String], prefix: &str) -> bool {
    warnings.iter().any(|w| w.starts_with(prefix))
}

fn burst_run(
    z_h: f64,
    drho: f64,
    z_start: f64,
    z_end: f64,
    n_points: usize,
) -> (f64, Vec<String>, usize) {
    let sigma_z = (0.04 * z_h).max(100.0);
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z,
        })
        .z_range(z_start, z_end)
        .build()
        .unwrap();
    let r = solver.run_to_result(z_end);
    (r.snapshot.delta_rho_over_rho, r.warnings, r.step_count)
}

/// `injected_delta_rho_between` against closed forms: a full Gaussian burst
/// integrates to its Δρ/ρ, the half above its peak to Δρ/2, and a constant
/// table rate r to r·(z_hi − z_lo). Photon injection has no known heat.
#[test]
fn injected_delta_rho_between_closed_forms() {
    let burst = InjectionScenario::SingleBurst {
        z_h: 2e5,
        delta_rho_over_rho: 1e-5,
        sigma_z: 8e3,
    };
    let full = burst.injected_delta_rho_between(1e3, 5e6).unwrap();
    assert!((full / 1e-5 - 1.0).abs() < 1e-9, "full burst {full:e}");
    let half = burst.injected_delta_rho_between(2e5, 5e6).unwrap();
    assert!((half / 5e-6 - 1.0).abs() < 1e-9, "half burst {half:e}");
    // ±1σ holds erf(1/√2) = 0.682689492137 of the Gaussian.
    let one_sigma = burst.injected_delta_rho_between(1.92e5, 2.08e5).unwrap();
    assert!((one_sigma / 1e-5 - 0.682_689_492_137).abs() < 1e-9);

    let z_table: Vec<f64> = (0..50)
        .map(|i| 1e3 * 10f64.powf(3.0 * i as f64 / 49.0))
        .collect();
    let table = InjectionScenario::TabulatedHeating {
        rate_table: vec![2e-12; z_table.len()],
        z_table,
    };
    let inside = table.injected_delta_rho_between(2e3, 5e5).unwrap();
    assert!((inside / (2e-12 * (5e5 - 2e3)) - 1.0).abs() < 1e-9);
    // Clipped to the table's own range [1e3, 1e6].
    let clipped = table.injected_delta_rho_between(10.0, 1e7).unwrap();
    assert!((clipped / (2e-12 * (1e6 - 1e3)) - 1.0).abs() < 1e-9);

    // Alternating heating and cooling, r·sin(2π z / P) over 10 whole periods:
    // net ≈ 0, gross = (2/π)·r·Δz.
    let (r, period) = (5e-10, 5e4);
    let z_sin: Vec<f64> = (0..20_001)
        .map(|i| 1e5 + 5e5 * i as f64 / 20_000.0)
        .collect();
    let sin_table = InjectionScenario::TabulatedHeating {
        rate_table: z_sin
            .iter()
            .map(|&z| r * (2.0 * std::f64::consts::PI * z / period).sin())
            .collect(),
        z_table: z_sin,
    };
    let gross = sin_table.injected_abs_delta_rho_between(1e5, 6e5).unwrap();
    let gross_exact = 2.0 / std::f64::consts::PI * r * 5e5;
    assert!((gross / gross_exact - 1.0).abs() < 1e-3, "gross {gross:e}");
    let net = sin_table.injected_delta_rho_between(1e5, 6e5).unwrap();
    assert!(net.abs() < 1e-3 * gross_exact, "net {net:e}");

    let photon = InjectionScenario::MonochromaticPhotonInjection {
        x_inj: 1.0,
        delta_n_over_n: 1e-5,
        z_h: 1e5,
        sigma_z: 4e3,
        sigma_x: 0.05,
    };
    assert!(photon.injected_delta_rho_between(1e3, 5e6).is_none());
}

/// R-1 reproduction: a 100-point grid started at z = 5e6 ends with
/// Δρ/ρ ≈ −1.5e-2 for an injected +1e-5 and used to exit cleanly. The
/// energy-closure and small-grid warnings must both fire. The same failed
/// run with the burst at z_h = 1600, where nearly all of the heat lands
/// below z = 2000, must skip the energy check (late-injection rule). The
/// run takes > 100,000 steps, so it also checks that the progress line
/// stays out of `warnings` (R-5).
#[test]
fn energy_closure_warns_on_coarse_grid() {
    let (drho, warnings, steps) = burst_run(3000.0, 1e-5, 5e6, 500.0, 100);
    assert!(steps > 100_000, "needs a progress line; steps = {steps}");
    assert!(
        (drho / 1e-5 - 1.0).abs() > 0.05,
        "precondition: this grid must fail closure, drho = {drho:e}"
    );
    assert!(has_warning(&warnings, "Energy closure"), "{warnings:?}");
    assert!(
        has_warning(&warnings, "Frequency grid has n_points=100"),
        "{warnings:?}"
    );
    assert!(!has_warning(&warnings, "Progress"), "{warnings:?}");

    let (drho_late, warnings_late, _) = burst_run(1600.0, 1e-5, 5e6, 500.0, 100);
    assert!(
        (drho_late / 1e-5 - 1.0).abs() > 0.05,
        "drho = {drho_late:e}"
    );
    assert!(
        !has_warning(&warnings_late, "Energy closure"),
        "{warnings_late:?}"
    );
}

/// A resolved burst closes within 5%: no energy warning, and the 1000-point
/// `GridConfig::fast()` size does not trigger the small-grid warning.
#[test]
fn energy_closure_silent_when_resolved() {
    let (drho, warnings, _) = burst_run(2e5, 1e-5, 2.6e5, 1e5, 1000);
    assert!((drho / 1e-5 - 1.0).abs() < 0.05, "drho = {drho:e}");
    assert!(!has_warning(&warnings, "Energy closure"), "{warnings:?}");
    assert!(
        !has_warning(&warnings, "Frequency grid has n_points"),
        "{warnings:?}"
    );
}

/// A burst at z_h = 1000 heats the electrons past the ρ_e cap (the
/// "Substantial heating" warning) and delivers about half its energy. All of
/// its heat lies below z = 2000, so the late-injection rule silences the
/// energy check before the cap rule is reached. The unit test
/// `test_energy_closure_skip_rules` in `solver.rs` and
/// `energy_closure_capped_shortfall_silent` isolate the cap rule.
#[test]
fn energy_closure_skipped_for_late_capped_burst() {
    let (drho, warnings, _) = burst_run(1000.0, 1e-5, 1700.0, 200.0, 500);
    assert!(
        has_warning(&warnings, "Substantial heating"),
        "{warnings:?}"
    );
    assert!(drho / 1e-5 - 1.0 < -0.05, "drho = {drho:e}");
    assert!(!has_warning(&warnings, "Energy closure"), "{warnings:?}");
}

/// Runs a tabulated heating table on `n_points` from `z_start` to `z_end`
/// and returns the final Δρ/ρ, the heat the table injects over that range,
/// and the warnings.
fn table_run(
    z_table: Vec<f64>,
    rate_table: Vec<f64>,
    z_start: f64,
    z_end: f64,
    n_points: usize,
) -> (f64, f64, Vec<String>) {
    let scenario = InjectionScenario::TabulatedHeating {
        z_table,
        rate_table,
    };
    let injected = scenario.injected_delta_rho_between(z_end, z_start).unwrap();
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points,
            ..GridConfig::default()
        })
        .injection(scenario)
        .z_range(z_start, z_end)
        .build()
        .unwrap();
    let r = solver.run_to_result(z_end);
    (r.snapshot.delta_rho_over_rho, injected, r.warnings)
}

/// A capped run with an excess must still warn. Constant heating
/// d(Δρ/ρ)/dz = 5.8e-11 on z ∈ [500, 5e6] injects 5.8e-11 × (5e6 − 500)
/// = 2.9e-4, only 0.03% of it below z = 2000, so the late rule does not
/// apply. The heat below recombination drives ρ_e to its cap, and the
/// 500-point grid leaves the spectrum with 24% more energy than was
/// injected (measured; finding R-1 in dev/REVIEW_2026-09-22.md gives +3%
/// for this table at 1000 points). The cap only removes heat, so the excess is a grid error and the
/// energy check must say so.
#[test]
fn energy_closure_capped_excess_warns() {
    let z_table: Vec<f64> = (0..200)
        .map(|i| 500.0 * 1e4_f64.powf(i as f64 / 199.0))
        .collect();
    let rate = vec![5.8e-11; z_table.len()];
    let (drho, injected, warnings) = table_run(z_table, rate, 5e6, 500.0, 500);
    eprintln!(
        "capped excess: drho = {drho:.4e}, injected = {injected:.4e}, ratio = {:.4}",
        drho / injected
    );
    assert!(
        (injected / (5.8e-11 * (5e6 - 500.0)) - 1.0).abs() < 1e-9,
        "injected = {injected:e}"
    );
    assert!(
        has_warning(&warnings, "Substantial heating"),
        "{warnings:?}"
    );
    assert!(
        drho / injected - 1.0 > 0.05,
        "precondition: excess, drho = {drho:e}"
    );
    let closure: Vec<&String> = warnings
        .iter()
        .filter(|w| w.starts_with("Energy closure"))
        .collect();
    assert_eq!(closure.len(), 1, "{warnings:?}");
    assert!(closure[0].contains("cannot cause an excess"), "{closure:?}");
}

/// A capped run with a shortfall stays silent, with the cap as the only skip
/// rule. A burst at z_h = 2e5 (Δρ/ρ = 1e-5) is followed by constant heating
/// of 3e-9 per unit z on z ∈ [500, 800]. That tail carries 9e-7, 8.3% of the
/// total, below the 10% late-injection limit, so the late rule does not
/// apply. The tail drives ρ_e to its cap, and most of the tail heat never
/// reaches the photons: they end 7.7% short (measured), beyond the 5%
/// tolerance. The run already warns "Substantial heating", so the energy
/// check stays silent.
#[test]
fn energy_closure_capped_shortfall_silent() {
    let (z_h, sigma, burst) = (2e5_f64, 8e3_f64, 1e-5_f64);
    let tail_rate = 3.0e-9;
    let z_table: Vec<f64> = (0..4000)
        .map(|i| 500.0 * 600_f64.powf(i as f64 / 3999.0))
        .collect();
    let rate: Vec<f64> = z_table
        .iter()
        .map(|&z| {
            let g = burst / ((2.0 * std::f64::consts::PI).sqrt() * sigma)
                * (-0.5 * ((z - z_h) / sigma).powi(2)).exp();
            g + if z <= 800.0 { tail_rate } else { 0.0 }
        })
        .collect();
    let scenario = InjectionScenario::TabulatedHeating {
        z_table: z_table.clone(),
        rate_table: rate.clone(),
    };
    let gross = scenario.injected_abs_delta_rho_between(500.0, 3e5).unwrap();
    let late = scenario
        .injected_abs_delta_rho_between(500.0, ENERGY_CHECK_Z_LATE)
        .unwrap();
    let tol = ENERGY_CLOSURE_REL_TOL * gross + ENERGY_CLOSURE_ABS_FLOOR;
    let (drho, injected, warnings) = table_run(z_table, rate, 3e5, 500.0, 1000);
    eprintln!(
        "capped shortfall: drho = {drho:.4e}, injected = {injected:.4e}, ratio = {:.4}",
        drho / injected
    );
    assert!(
        late / gross < ENERGY_CHECK_MAX_LATE_FRACTION,
        "precondition: the late rule must not apply, late fraction {}",
        late / gross
    );
    assert!(
        has_warning(&warnings, "Substantial heating"),
        "{warnings:?}"
    );
    assert!(
        drho - injected < -tol,
        "precondition: shortfall beyond the tolerance, drho = {drho:e}"
    );
    assert!(!has_warning(&warnings, "Energy closure"), "{warnings:?}");
}

/// Constant tabulated heating, d(Δρ/ρ)/dz = 1e-11 on z ∈ [5e3, 2e6], injects
/// 1e-11 × (2e6 − 5e3) = 1.995e-5. At 300 points the spectrum holds ~38%
/// more than that and the check fires; at 1000 points it closes within 2%.
#[test]
fn energy_closure_tabulated_heating() {
    let z_table: Vec<f64> = (0..100)
        .map(|i| 5e3 * (2e6_f64 / 5e3).powf(i as f64 / 99.0))
        .collect();
    let injected = 1e-11 * (2e6 - 5e3);
    let run = |n_points: usize| {
        let mut solver = ThermalizationSolver::builder(Cosmology::default())
            .grid(GridConfig {
                n_points,
                ..GridConfig::default()
            })
            .injection(InjectionScenario::TabulatedHeating {
                rate_table: vec![1e-11; z_table.len()],
                z_table: z_table.clone(),
            })
            .z_range(2e6, 1e3)
            .build()
            .unwrap();
        let r = solver.run_to_result(1e3);
        (r.snapshot.delta_rho_over_rho, r.warnings)
    };
    let (drho_coarse, w_coarse) = run(300);
    assert!(
        (drho_coarse / injected - 1.0).abs() > 0.05,
        "drho = {drho_coarse:e}"
    );
    assert!(has_warning(&w_coarse, "Energy closure"), "{w_coarse:?}");
    let (drho_fine, w_fine) = run(1000);
    assert!(
        (drho_fine / injected - 1.0).abs() < 0.05,
        "drho = {drho_fine:e}"
    );
    assert!(!has_warning(&w_fine, "Energy closure"), "{w_fine:?}");
}

/// A 50-point grid started at z = 5e6 drives Δn to NaN within the first
/// steps. The run must stop and return that as an `Err` (so the CLI exits
/// with an error and Python raises), not panic. The panicking wrapper
/// `run_to_result` must carry the same message, and the same solver must run
/// cleanly afterwards on a resolvable range (the failure does not stick).
///
/// No input produces NaN deterministically (`set_initial_delta_n` rejects
/// it), so this test relies on the 50-point instability. If a robustness fix
/// removes it, find another trigger rather than deleting the test.
#[test]
fn nan_run_returns_error_not_panic() {
    let build = || {
        ThermalizationSolver::builder(Cosmology::default())
            .grid(GridConfig {
                n_points: 50,
                ..GridConfig::default()
            })
            .injection(InjectionScenario::SingleBurst {
                z_h: 3000.0,
                delta_rho_over_rho: 1e-5,
                sigma_z: 120.0,
            })
            .z_range(5e6, 500.0)
            .build()
            .unwrap()
    };
    let mut solver = build();
    let err = match solver.try_run_to_result(500.0) {
        Ok(r) => panic!(
            "expected an error, got Δρ/ρ = {:e}",
            r.snapshot.delta_rho_over_rho
        ),
        Err(e) => e,
    };
    assert!(err.contains("NaN/Inf detected in delta_n"), "{err}");
    assert_eq!(solver.diag.failure.as_deref(), Some(err.as_str()));

    // Reuse after a failure: a short μ-era run on the same solver succeeds.
    solver.set_config(SolverConfig {
        z_start: 2.6e5,
        z_end: 2.5e5,
        ..SolverConfig::default()
    });
    let ok = solver.try_run_to_result(2.5e5);
    assert!(ok.is_ok(), "{:?}", ok.err());
    assert!(solver.diag.failure.is_none());

    let mut solver = build();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = solver.run_to_result(500.0);
    }))
    .expect_err("run_to_result must still panic on NaN");
    let msg = panic.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("NaN/Inf detected in delta_n"), "{msg}");
}

/// A coarse 50-point grid runs to completion without panicking and carries
/// the small-grid warning.
///
/// This was `smallest_accepted_grid_runs_and_warns` on the 10-point grid,
/// the smallest that validation accepts. There the answer is garbage with or
/// without ADR 0004: before it, Δρ/ρ = 15.9 with ρ_e at its cap; after it,
/// NaN. At 50 points and above both versions are finite and agree to 0.3%
/// (`dev/audit/fix_a_cn_old_half_ab.md`), so the test moved there
/// (follows ADR 0004).
#[test]
fn coarse_grid_runs_and_warns() {
    let (drho, warnings, _) = burst_run(2e5, 1e-5, 2.6e5, 1e5, 50);
    assert!(drho.is_finite());
    assert!(
        has_warning(&warnings, "Frequency grid has n_points=50"),
        "{warnings:?}"
    );
}
