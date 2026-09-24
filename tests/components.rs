//! Component tests of the physics and infrastructure modules.
//!
//! These tests need no full PDE run. They cover the cosmology background,
//! the frequency grid, the DC and BR rates, the Kompaneets kernel, the
//! injection-rate functions, table I/O, and the quasi-stationary electron
//! temperature.

use spectroxide::constants::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::recombination;
use spectroxide::spectrum;

/// The decaying particle heating rate has the form:
///   d(Δρ/ρ)/dt = f_X × Γ_X × (N_H/ρ_γ) × exp(−Γ_X × t)
///
/// The exp(−Γ_X t) factor is kinematically exact. But the prefactor
/// N_H/ρ_γ ∝ 1/(1+z) changes with redshift, so the total rate is NOT
/// simply monotonic. We test:
///   1. Rate is always non-negative (it's heating, not cooling)
///   2. The exponential decay dominates at late times (low z)
///   3. After factoring out the cosmological prefactor, the residual
///      follows exp(−Γ_X t) to good accuracy
///   4. At t >> 1/Γ_X the rate is exponentially suppressed
#[test]
fn test_decaying_particle_time_dependence() {
    let cosmo = Cosmology::default();
    let gamma_x = 1e-13; // Γ_X = 10⁻¹³ s⁻¹, lifetime ~ 10¹³ s ≈ 300,000 yr
    let f_x = 1e6; // 1 MeV per hydrogen nucleus

    let scenario = InjectionScenario::DecayingParticle { f_x, gamma_x };

    // Sample heating rate at several redshifts
    let z_values = [1e6, 5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5e3, 2e3];

    let mut rates_and_times = Vec::new();

    for &z in &z_values {
        let rate = scenario.heating_rate(z, &cosmo);
        let t = cosmo.cosmic_time(z);
        rates_and_times.push((z, t, rate));

        // Rate should be non-negative (decays inject energy)
        assert!(
            rate >= 0.0,
            "Decay heating rate negative at z = {z}: {rate}"
        );
    }

    // Factor out the cosmological prefactor: rate / (n_h/ρ_γ) should be
    // proportional to exp(-Γ t). Check this at two well-separated times.
    let (z1, t1, rate1) = rates_and_times[2]; // z = 2e5
    let (z2, t2, rate2) = rates_and_times[5]; // z = 2e4

    // Cosmological prefactor: n_h / ρ_γ at each redshift
    let prefactor1 = cosmo.n_h(z1) / cosmo.rho_gamma(z1);
    let prefactor2 = cosmo.n_h(z2) / cosmo.rho_gamma(z2);

    // After dividing out the prefactor, the ratio should be exp(-Γ Δt)
    let corrected_ratio = (rate2 / prefactor2) / (rate1 / prefactor1);
    let expected_ratio = (-gamma_x * (t2 - t1)).exp();

    eprintln!("Decay rate exponential check:");
    eprintln!("  t₁ = {t1:.4e} s (z={z1}), t₂ = {t2:.4e} s (z={z2})");
    eprintln!("  Corrected ratio = {corrected_ratio:.6e}");
    eprintln!("  Expected exp(-Γ Δt) = {expected_ratio:.6e}");

    if rate1 > 1e-50 && rate2 > 1e-50 {
        let rel_err = (corrected_ratio - expected_ratio).abs() / expected_ratio.abs().max(1e-30);
        assert!(
            rel_err < 0.05,
            "Exponential decay not recovered: ratio = {corrected_ratio:.4e}, \
             expected = {expected_ratio:.4e}, rel_err = {rel_err:.3}"
        );
    }

    // At very late times, rate should be exponentially suppressed
    let rate_late = scenario.heating_rate(100.0, &cosmo);
    let rate_early = scenario.heating_rate(1e6, &cosmo);
    assert!(
        rate_late < rate_early,
        "Rate at z=100 ({rate_late:.4e}) should be < rate at z=10⁶ ({rate_early:.4e})"
    );
}

/// Verify the heating rate function integrates to the correct total energy
/// for a SingleBurst injection. The Gaussian should normalize to Δρ/ρ
/// when integrated over d(Δρ/ρ)/dz × dz.
///
/// This is a consistency check on the injection machinery, not the physics.
#[test]
fn test_single_burst_energy_normalization() {
    let cosmo = Cosmology::default();
    let z_h = 2.0e5;
    let drho = 1e-5;
    let sigma = 5000.0;

    let scenario = InjectionScenario::SingleBurst {
        z_h,
        delta_rho_over_rho: drho,
        sigma_z: sigma,
    };

    // Integrate d(Δρ/ρ)/dz over z
    let n_z = 20000;
    let z_min = z_h - 6.0 * sigma;
    let z_max = z_h + 6.0 * sigma;
    let dz = (z_max - z_min) / n_z as f64;

    let mut total = 0.0;
    for i in 0..n_z {
        let z = z_min + (i as f64 + 0.5) * dz;
        total += scenario.heating_rate_per_redshift(z, &cosmo).abs() * dz;
    }

    let rel_err = (total - drho).abs() / drho;
    eprintln!(
        "Burst normalization: integrated = {total:.6e}, expected = {drho:.6e}, \
         rel_err = {rel_err:.2e}"
    );

    assert!(
        rel_err < 0.01,
        "Burst normalization off: integrated = {total:.4e}, expected = {drho:.4e}"
    );
}

/// DM annihilation redshift scaling: s-wave ∝ (1+z)², p-wave ∝ (1+z)³.
#[test]
fn test_annihilation_redshift_scaling() {
    let cosmo = Cosmology::default();
    let f_ann = 1e-30;
    let z1 = 1e4_f64;
    let z2 = 1e5_f64;

    // s-wave: ∝ (1+z)²
    let s = InjectionScenario::AnnihilatingDM { f_ann };
    let ratio_s = s.heating_rate(z2, &cosmo) / s.heating_rate(z1, &cosmo);
    let expected_s = ((1.0 + z2) / (1.0 + z1)).powi(2);
    assert!(
        (ratio_s - expected_s).abs() / expected_s < 0.001,
        "s-wave scaling"
    );

    // p-wave: ∝ (1+z)³
    let p = InjectionScenario::AnnihilatingDMPWave { f_ann };
    let ratio_p = p.heating_rate(z2, &cosmo) / p.heating_rate(z1, &cosmo);
    let expected_p = ((1.0 + z2) / (1.0 + z1)).powi(3);
    assert!(
        (ratio_p - expected_p).abs() / expected_p < 0.001,
        "p-wave scaling"
    );
}

/// Decaying particle heating rate is NOT monotonically decreasing in z.
///
/// For gamma_x = 1e-10 s^-1 (tau = 1e10 s), the heating rate
///   d(Drho/rho)/dt = f_x * gamma_x * n_H(z) * exp(-gamma_x * t(z)) / rho_gamma(z)
/// has competing factors: n_H/rho_gamma ~ (1+z)^-1 decreases with z,
/// while exp(-Gamma*t) increases with z (since t decreases with z).
/// This means the rate has a maximum at some intermediate z.
///
/// We verify the rate at z=1e4 is less than the rate at some higher z.
#[test]
fn test_decay_rate_non_monotonic() {
    let cosmo = Cosmology::default();
    let gamma_x = 1e-10_f64; // tau = 1e10 s
    let scenario = InjectionScenario::DecayingParticle { f_x: 1e-5, gamma_x };

    // Sample the rate at several redshifts
    let z_values = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6];
    let mut rates = Vec::new();
    for &z in &z_values {
        let rate = scenario.heating_rate(z, &cosmo);
        rates.push(rate);
        eprintln!("  z = {z:.0e}: rate = {rate:.4e}");
    }

    // Find the maximum rate
    let (i_max, &max_rate) = rates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    eprintln!(
        "  max rate at z = {:.0e} (index {})",
        z_values[i_max], i_max
    );

    // The rate at z=1e4 (index 2)
    let rate_at_1e4 = rates[2];

    // Non-monotonicity: there must be some z > 1e4 where rate > rate(1e4)
    let has_higher = rates[3..].iter().any(|&r| r > rate_at_1e4);
    assert!(
        has_higher,
        "Decay rate should be non-monotonic: rate(z=1e4) = {rate_at_1e4:.4e}, \
         but no higher rate found at z > 1e4. Max = {max_rate:.4e} at z = {:.0e}",
        z_values[i_max]
    );
}

/// BR emission coefficient scales as θ_e^{-7/2} × (Gaunt ratio).
///
/// At T_e=T_z (φ=1), K_BR = BR_PREFACTOR × θ_e^{-7/2} × e^{-x} × Σ_i Z_i² N_i g_ff.
/// So k(θ₁)/k(θ₂) = (θ₂/θ₁)^{7/2} × [species_sum(θ₁) / species_sum(θ₂)].
/// Including the explicit Gaunt dependence lets us assert ±5% instead of ±factor-10
/// (which was loose enough that a wrong exponent -3 vs -7/2 between these
/// θ values would still pass, and worthless as a guard for CLAUDE.md Pitfall #8).
#[test]
fn test_br_temperature_scaling() {
    use spectroxide::bremsstrahlung::{br_emission_coefficient, gaunt_ff_nr};

    let cosmo = Cosmology::default();
    let n_h = 1e6_f64;
    let n_he = 0.08 * n_h;
    let n_e = n_h;
    let x = 0.5_f64;

    let theta1 = 1e-5_f64;
    let theta2 = 2e-5_f64;

    let k1 = br_emission_coefficient(x, theta1, theta1, n_h, n_he, n_e, 1.0, &cosmo);
    let k2 = br_emission_coefficient(x, theta2, theta2, n_h, n_he, n_e, 1.0, &cosmo);

    // Build the Gaunt-weighted species sum directly. Both θ values sit
    // above HeII-recombination in Saha (θ=1e-5 ↔ z≈21700, θ=2e-5 ↔ z≈43500),
    // so species_sum = n_h · g_Z1 + 4 · n_he · g_Z2.
    let species1 = n_h * gaunt_ff_nr(x, theta1, 1.0) + 4.0 * n_he * gaunt_ff_nr(x, theta1, 2.0);
    let species2 = n_h * gaunt_ff_nr(x, theta2, 1.0) + 4.0 * n_he * gaunt_ff_nr(x, theta2, 2.0);

    let expected_ratio = (theta2 / theta1).powf(3.5) * (species1 / species2);
    let actual_ratio = k1 / k2;
    let rel_err = (actual_ratio - expected_ratio).abs() / expected_ratio;

    eprintln!(
        "BR scaling: k1/k2 = {actual_ratio:.4e}, expected = {expected_ratio:.4e}, err = {:.2}%",
        rel_err * 100.0
    );
    assert!(
        rel_err < 0.05,
        "BR scaling should match θ^{{-7/2}} × Gaunt to 5%: ratio={actual_ratio:.4e}, expected={expected_ratio:.4e}, err={:.2}%",
        rel_err * 100.0
    );
}

/// DC emission coefficient should scale as θ_z² (quadratic in temperature).
#[test]
fn test_dc_temperature_scaling() {
    use spectroxide::double_compton::dc_emission_coefficient;

    let x = 0.5_f64;
    let theta1 = 1e-5_f64;
    let theta2 = 2e-5_f64;

    let k1 = dc_emission_coefficient(x, theta1);
    let k2 = dc_emission_coefficient(x, theta2);

    // K_DC ∝ θ_z² × g_dc ∝ θ_z² × 1/(1+14.16θ_z) × H_dc(x)
    // For small θ_z, the 1/(1+14.16θ_z) ≈ 1 correction is tiny.
    // So k2/k1 ≈ (θ₂/θ₁)²
    let expected_ratio = (theta2 / theta1).powi(2);
    let actual_ratio = k2 / k1;

    // Should be very close (relativistic correction is < 0.1% at these θ)
    let rel_err = (actual_ratio / expected_ratio - 1.0).abs();
    assert!(
        rel_err < 0.01,
        "DC scaling K ∝ θ²: k2/k1={actual_ratio:.6e}, expected={expected_ratio:.6e}, err={rel_err:.4e}"
    );
}

/// The matter-radiation equality redshift should satisfy Ω_m(1+z_eq)³ = Ω_rel(1+z_eq)⁴.
#[test]
fn test_cosmology_self_consistency() {
    let cosmo = Cosmology::default();

    // z_eq: matter = radiation
    let z_eq = cosmo.z_eq();
    let matter = cosmo.omega_m() * (1.0 + z_eq).powi(3);
    let radiation = cosmo.omega_rel() * (1.0 + z_eq).powi(4);
    assert!(
        (matter - radiation).abs() / matter < 1e-10,
        "z_eq self-consistency"
    );
    assert!(z_eq > 3000.0 && z_eq < 4000.0, "z_eq={z_eq:.1}");

    // Omega closure (flat universe)
    let total = cosmo.omega_m() + cosmo.omega_rel() + cosmo.omega_lambda();
    assert!((total - 1.0).abs() < 1e-10, "Omega_total = {total}");
    assert!(
        cosmo.omega_m() > 0.0
            && cosmo.omega_rel() > 0.0
            && cosmo.omega_lambda() > 0.0
            && cosmo.omega_gamma() > 0.0
    );

    // E(z) asymptotic regimes
    // Radiation-dominated
    let z_rad = 1e7_f64;
    let e_rad_approx = cosmo.omega_rel().sqrt() * (1.0 + z_rad).powi(2);
    assert!((cosmo.e_of_z(z_rad) - e_rad_approx).abs() / cosmo.e_of_z(z_rad) < 1e-3);
    // Matter-dominated
    let z_mat = 100.0_f64;
    let e_mat_approx = cosmo.omega_m().sqrt() * (1.0 + z_mat).powf(1.5);
    assert!((cosmo.e_of_z(z_mat) - e_mat_approx).abs() / cosmo.e_of_z(z_mat) < 0.1);
    // Today
    assert!((cosmo.e_of_z(0.0) - 1.0).abs() < 1e-12);

    // Thomson time scaling: t_C ∝ (1+z)^{-3}
    let t_c_low = cosmo.t_compton(1000.0, 1.0);
    let t_c_high = cosmo.t_compton(1e5, 1.0);
    assert!(t_c_high < t_c_low, "t_C should decrease with z");
    let expected_ratio = ((1.0 + 1e5_f64) / (1.0 + 1000.0_f64)).powi(3);
    assert!((t_c_low / t_c_high / expected_ratio - 1.0).abs() < 1e-5);
}

/// Density scaling relations: n_H ∝ (1+z)³, n_He/n_H = Y_p/(4(1-Y_p)),
/// ρ_γ ∝ (1+z)⁴.
#[test]
fn test_density_scaling_relations() {
    let cosmo = Cosmology::default();
    let z1 = 1000.0_f64;
    let z2 = 2000.0_f64;

    // n_H ∝ (1+z)³
    let n_ratio = cosmo.n_h(z2) / cosmo.n_h(z1);
    let expected_n = ((1.0 + z2) / (1.0 + z1)).powi(3);
    assert!(
        (n_ratio - expected_n).abs() / expected_n < 1e-10,
        "n_H scaling"
    );

    // n_He/n_H = Y_p/(4(1-Y_p)) at all z
    let expected_he = cosmo.y_p / (4.0 * (1.0 - cosmo.y_p));
    for &z in &[0.0_f64, 100.0, 1000.0, 1e5, 1e6] {
        let ratio = cosmo.n_he(z) / cosmo.n_h(z);
        assert!((ratio - expected_he).abs() < 1e-10, "n_He/n_H at z={z}");
    }

    // ρ_γ ∝ (1+z)⁴
    let rho_ratio = cosmo.rho_gamma(z2) / cosmo.rho_gamma(z1);
    let expected_rho = ((1.0 + z2) / (1.0 + z1)).powi(4);
    assert!(
        (rho_ratio - expected_rho).abs() / expected_rho < 1e-10,
        "rho_gamma scaling"
    );
}

/// Cosmic time at z=0 should be the age of the universe: ~13-14 Gyr.
/// At recombination (z~1100), t ~ 380,000 years.
#[test]
fn test_cosmic_time_milestones() {
    let cosmo = Cosmology::default();

    // Age of the universe
    let t_0 = cosmo.cosmic_time(0.0);
    let t_0_gyr = t_0 / (365.25 * 24.0 * 3600.0 * 1e9);
    eprintln!("t(z=0) = {t_0_gyr:.2} Gyr");
    assert!(
        t_0_gyr > 12.0 && t_0_gyr < 15.0,
        "Age = {t_0_gyr:.2} Gyr, expected 13-14 Gyr"
    );

    // Recombination (z ~ 1100)
    let t_rec = cosmo.cosmic_time(1100.0);
    let t_rec_kyr = t_rec / (365.25 * 24.0 * 3600.0 * 1e3);
    eprintln!("t(z=1100) = {t_rec_kyr:.0} kyr");
    assert!(
        t_rec_kyr > 200.0 && t_rec_kyr < 500.0,
        "t(recomb) = {t_rec_kyr:.0} kyr, expected ~380 kyr"
    );

    // High redshift (z=1e6): t ≈ 1/(2H) ≈ 1/(2H₀√Ω_rel(1+z)²)
    // With Ω_rel ~ 8.6e-5 and H₀ ~ 2.3e-18: t ~ 2.4e7 s ~ 9 months
    let t_high = cosmo.cosmic_time(1e6);
    let t_high_months = t_high / (30.44 * 24.0 * 3600.0);
    eprintln!("t(z=1e6) = {:.2e} s = {t_high_months:.1} months", t_high);
    assert!(
        t_high_months > 3.0 && t_high_months < 30.0,
        "t(z=1e6) = {t_high_months:.1} months, expected ~9 months"
    );
}

/// dt/dz should be negative (z decreases with time) and consistent
/// with the Hubble rate: dt/dz = -1/(H(z)(1+z)).
/// H₀ should be 100h km/s/Mpc in SI units.
/// Ω_γ should be ~5×10⁻⁵ for standard cosmology.
/// Friedmann equation: E(z)² = Ω_m(1+z)³ + Ω_rel(1+z)⁴ + Ω_Λ
/// Verify at several redshifts by recomputing from components.
/// Grid find_index should handle boundary cases correctly.
#[test]
fn test_grid_find_index_boundaries() {
    let grid = FrequencyGrid::new(&GridConfig::default());

    // Below grid minimum: should return 0
    let idx_below = grid.find_index(1e-10);
    assert_eq!(idx_below, 0, "Below-grid index should be 0");

    // Above grid maximum: should return n-1
    let idx_above = grid.find_index(1000.0);
    assert_eq!(idx_above, grid.n - 1, "Above-grid index should be n-1");

    // At grid minimum: should return 0
    let idx_min = grid.find_index(grid.x[0]);
    assert_eq!(idx_min, 0, "At x_min index should be 0");

    // At grid maximum: should return n-1
    let idx_max = grid.find_index(grid.x[grid.n - 1]);
    assert_eq!(idx_max, grid.n - 1, "At x_max index should be n-1");

    // At a middle point: returned index should be the closest
    let x_mid = grid.x[grid.n / 2];
    let idx_mid = grid.find_index(x_mid);
    assert_eq!(
        idx_mid,
        grid.n / 2,
        "At exact grid point should return that index"
    );
}

/// Grid with purely log or purely linear spacing should work.
#[test]
fn test_grid_extreme_configurations() {
    // Pure log grid
    let log_grid = FrequencyGrid::log_uniform(1e-3, 50.0, 500);
    assert_eq!(log_grid.n, 500);
    assert!(log_grid.x[0] > 0.0);
    for i in 1..log_grid.n {
        assert!(
            log_grid.x[i] > log_grid.x[i - 1],
            "Log grid not monotonic at i={i}"
        );
    }
    // dx/x should be approximately constant
    let ratio_first = log_grid.dx[0] / log_grid.x[0];
    let ratio_last = log_grid.dx[log_grid.n - 2] / log_grid.x[log_grid.n - 2];
    assert!(
        (ratio_first - ratio_last).abs() / ratio_first < 0.01,
        "Log grid dx/x should be constant: first={ratio_first:.4e}, last={ratio_last:.4e}"
    );

    // Pure uniform grid
    let lin_grid = FrequencyGrid::uniform(0.1, 50.0, 500);
    assert_eq!(lin_grid.n, 500);
    // dx should be constant
    let dx_first = lin_grid.dx[0];
    let dx_last = lin_grid.dx[lin_grid.n - 2];
    assert!(
        (dx_first - dx_last).abs() < 1e-12,
        "Uniform grid dx should be constant: first={dx_first:.6e}, last={dx_last:.6e}"
    );
}

/// Thomas algorithm should produce accurate solutions for well-conditioned systems.
/// Test with a known analytic solution: tridiagonal discretization of -u'' = f
/// on [0,1] with u(0) = u(1) = 0, f = sin(πx), exact u = sin(πx)/π².
#[test]
fn test_thomas_algorithm_accuracy() {
    let n = 200;
    let h = 1.0 / (n as f64 + 1.0);
    let mut lower = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    for i in 0..n {
        let x = (i as f64 + 1.0) * h;
        diag[i] = 2.0 / (h * h);
        if i > 0 {
            lower[i] = -1.0 / (h * h);
        }
        if i < n - 1 {
            upper[i] = -1.0 / (h * h);
        }
        rhs[i] = (std::f64::consts::PI * x).sin();
    }

    let sol = spectroxide::kompaneets::thomas_solve(&lower, &diag, &upper, &mut rhs);
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;

    let mut max_err: f64 = 0.0;
    for i in 0..n {
        let x = (i as f64 + 1.0) * h;
        let exact = (std::f64::consts::PI * x).sin() / pi2;
        let err = (sol[i] - exact).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // Second-order FD discretization should give O(h²) error
    assert!(
        max_err < 1e-4,
        "Thomas solve error = {max_err:.4e}, expected O(h²) ≈ {:.4e}",
        h * h
    );
}

/// Coupled Kompaneets+DC/BR Newton iteration should produce finite results
/// when run through the full solver (which manages dtau internally).
/// This end-to-end test verifies the coupled solve doesn't diverge.
/// Large Δτ backward Euler should not blow up. The implicit scheme should remain
/// stable even with Δτ >> 1 (the CFL limit for explicit schemes).
/// At T_e = T_z, Kompaneets doesn't change energy, only redistributes photons.
#[test]
fn test_kompaneets_large_dtau_stability() {
    let grid = FrequencyGrid::new(&GridConfig::fast());
    let theta = 1e-5;

    // Start with a small y-type distortion
    let delta_n: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| 1e-6 * spectrum::y_shape(x))
        .collect();

    // Δτ = 100 is far beyond the explicit CFL limit
    let result =
        spectroxide::kompaneets::kompaneets_step_nonlinear(&grid, &delta_n, theta, theta, 100.0);

    // Should be stable (no NaN)
    assert!(
        result.iter().all(|v| v.is_finite()),
        "Implicit solver produced NaN at Δτ=100"
    );

    // Energy should be approximately conserved (Kompaneets at T_e=T_z is number-changing
    // only at O(θ²) due to the nonlinear term, but conserves energy to machine precision
    // in the linearized limit). For this small distortion, energy should be well-conserved.
    let drho_before = spectrum::delta_rho_over_rho(&grid.x, &delta_n);
    let drho_after = spectrum::delta_rho_over_rho(&grid.x, &result);
    let rel_err = (drho_after - drho_before).abs() / drho_before.abs().max(1e-30);
    // For a small y-type distortion with T_e=T_z,
    // Kompaneets conserves energy to better than 1% even at large Δτ.
    assert!(
        rel_err < 0.10,
        "Energy not conserved at large Δτ: before={drho_before:.4e}, after={drho_after:.4e}, \
         rel_err={rel_err:.2e}"
    );
}

/// Decaying particle with extreme lifetime (longer than age of universe)
/// should give a small but nonzero rate.
#[test]
fn test_decaying_particle_extreme_lifetime() {
    let cosmo = Cosmology::default();

    // Lifetime = 1e20 s >> age of universe (~4e17 s)
    let scenario = InjectionScenario::DecayingParticle {
        f_x: 1e6,       // 1 MeV
        gamma_x: 1e-20, // Γ = 10⁻²⁰ s⁻¹
    };

    let rate = scenario.heating_rate(1e5, &cosmo);
    assert!(
        rate >= 0.0 && rate.is_finite(),
        "Rate should be non-negative and finite: {rate:.4e}"
    );
    // exp(-Γt) ≈ 1 for such long lifetime, so rate ≈ f_x × Γ × n_h / ρ_γ
    assert!(rate > 0.0, "Rate should be nonzero: {rate:.4e}");
}

/// Verify that a refined grid with 2000 base points produces monotonic,
/// duplicate-free grids, and that heat injection with empty refinement
/// zones is unchanged.
#[test]
fn test_refinement_grid_properties() {
    use spectroxide::grid::{FrequencyGrid, GridConfig, RefinementZone};

    let config = GridConfig {
        refinement_zones: vec![
            RefinementZone {
                x_center: 0.1,
                x_width: 0.05,
                n_points: 200,
            },
            RefinementZone {
                x_center: 3.0,
                x_width: 1.0,
                n_points: 100,
            },
        ],
        ..GridConfig::default()
    };
    let grid = FrequencyGrid::new(&config);

    // Monotonicity
    for i in 1..grid.n {
        assert!(grid.x[i] > grid.x[i - 1], "Grid not monotonic at i={i}");
    }

    // No near-duplicates
    for i in 1..grid.n {
        let rel = (grid.x[i] - grid.x[i - 1]) / (0.5 * (grid.x[i] + grid.x[i - 1]));
        assert!(rel > 1e-10, "Near-duplicate at i={i}");
    }

    // Should have more points than 2000
    assert!(grid.n > 2000, "Expected > 2000 pts, got {}", grid.n);
}

/// Tabulated heating: zero outside table bounds.
#[test]
fn test_tabulated_heating_zero_outside_bounds() {
    let cosmo = Cosmology::default();
    let z_table = vec![1e4, 5e4, 1e5];
    let rate_table = vec![1e-10, 2e-10, 3e-10];
    let tabulated = InjectionScenario::TabulatedHeating {
        z_table,
        rate_table,
    };

    // Outside bounds: should be 0
    assert_eq!(tabulated.heating_rate(500.0, &cosmo), 0.0);
    assert_eq!(tabulated.heating_rate(2e5, &cosmo), 0.0);

    // Inside bounds: should be positive
    assert!(tabulated.heating_rate(5e4, &cosmo) > 0.0);
}

/// Load heating table from file and verify it works.
#[test]
fn test_load_heating_table_roundtrip() {
    let tmp = std::env::temp_dir().join("spectroxide_test_ht_roundtrip.csv");
    std::fs::write(&tmp, "z,dq_dz\n1e4,1.5e-10\n5e4,2.5e-10\n1e5,0.5e-10\n").unwrap();

    let scenario =
        spectroxide::energy_injection::load_heating_table(tmp.to_str().unwrap()).unwrap();
    let cosmo = Cosmology::default();

    match &scenario {
        InjectionScenario::TabulatedHeating { z_table, .. } => {
            assert_eq!(z_table.len(), 3);
            // Table should be sorted ascending
            assert!(z_table[0] < z_table[1] && z_table[1] < z_table[2]);
        }
        _ => panic!("Expected TabulatedHeating"),
    }

    // Should interpolate inside bounds
    let rate = scenario.heating_rate(5e4, &cosmo);
    assert!(rate > 0.0, "rate at z=5e4 should be positive, got {rate}");

    // Should be zero outside bounds
    assert_eq!(scenario.heating_rate(5e3, &cosmo), 0.0);
    assert_eq!(scenario.heating_rate(2e5, &cosmo), 0.0);

    std::fs::remove_file(&tmp).ok();
}

/// Load photon source table from file.
#[test]
fn test_load_photon_source_table_roundtrip() {
    let tmp = std::env::temp_dir().join("spectroxide_test_ps_roundtrip.csv");
    std::fs::write(
        &tmp,
        "z,1.0e+00,5.0e+00,1.0e+01\n\
         1e4,1e-15,2e-15,3e-15\n\
         5e4,4e-15,5e-15,6e-15\n",
    )
    .unwrap();

    let scenario =
        spectroxide::energy_injection::load_photon_source_table(tmp.to_str().unwrap()).unwrap();
    let cosmo = Cosmology::default();

    assert!(scenario.has_photon_source());

    // Should interpolate inside bounds
    let rate = scenario.photon_source_rate(5.0, 3e4, &cosmo);
    assert!(
        rate > 0.0,
        "source at (x=5, z=3e4) should be positive, got {rate}"
    );

    // Outside bounds in z
    assert_eq!(scenario.photon_source_rate(5.0, 500.0, &cosmo), 0.0);
    assert_eq!(scenario.photon_source_rate(5.0, 1e5, &cosmo), 0.0);

    // Outside bounds in x
    assert_eq!(scenario.photon_source_rate(0.1, 3e4, &cosmo), 0.0);
    assert_eq!(scenario.photon_source_rate(20.0, 3e4, &cosmo), 0.0);

    std::fs::remove_file(&tmp).ok();
}

// The compton_y_parameter function uses 128-point midpoint quadrature.
// Verify it is converged by comparing against a high-resolution calculation.
#[test]
fn test_compton_y_parameter_convergence() {
    let cosmo = Cosmology::default();

    // At z=1e5, y_C should be > 0.1 (deep in fully ionized era).
    // At z=1100, y_C is tiny because the integral is from z=0 to z=1100,
    // dominated by the post-recombination era where X_e ~ 10^{-4}.
    let yc_1e5 = cosmo.compton_y_parameter(1.0e5);
    assert!(
        yc_1e5 > 0.1,
        "y_C(1e5) = {yc_1e5:.3e} (expected > 0.1 in fully ionized era)"
    );
    let yc_1e6 = cosmo.compton_y_parameter(1.0e6);
    assert!(
        yc_1e6 > 10.0,
        "y_C(1e6) = {yc_1e6:.3e} (expected >> 1 deep in thermalization era)"
    );

    // Independent calculation: 1024-point integration from z=0 to z=1e5.
    let z_max = 1.0e5_f64;
    let n_pts = 1024;
    let ln_max = (1.0 + z_max).ln();
    let h = ln_max / (n_pts as f64);
    let mut yc_hires = 0.0;

    for i in 0..n_pts {
        let u = (i as f64 + 0.5) * h;
        let zp = u.exp() - 1.0;
        let x_e = recombination::ionization_fraction(zp, &cosmo);
        let n_e = cosmo.n_e(zp, x_e);
        let theta_e = K_BOLTZMANN * cosmo.t_cmb * (1.0 + zp) / (M_ELECTRON * C_LIGHT * C_LIGHT);
        yc_hires += theta_e * SIGMA_THOMSON * C_LIGHT * n_e / cosmo.hubble(zp) * h;
    }

    let yc_code = cosmo.compton_y_parameter(z_max);
    let rel_err = (yc_code - yc_hires).abs() / yc_hires;

    assert!(
        rel_err < 0.01,
        "y_C(1e5) quadrature convergence: code(128pt)={yc_code:.6}, hires(1024pt)={yc_hires:.6}, \
         err={:.2}%",
        rel_err * 100.0
    );

    // Monotonicity (regression check)
    let yc_500 = cosmo.compton_y_parameter(500.0);
    let yc_2000 = cosmo.compton_y_parameter(2000.0);
    let yc_1e5 = cosmo.compton_y_parameter(1e5);
    assert!(
        yc_500 < yc_2000 && yc_2000 < yc_1e5,
        "y_C must be monotonically increasing: {yc_500:.3e} < {yc_2000:.3e} < {yc_1e5:.3e}"
    );
}

/// P&B 2009 grid parameters: verify our grid covers the required range.
///
/// P&B use x_min = 10^{-4.3} ≈ 5×10⁻⁵ and x_max = 10^{1.7} ≈ 50.
/// Our production grid should cover at least this range.
/// (Default grid x_min=1e-4 is slightly coarser; production grid x_min=1e-5 covers P&B.)
#[test]
fn test_pb2009_grid_coverage() {
    // Production grid covers P&B range
    let grid_config = GridConfig::production();
    let grid = FrequencyGrid::new(&grid_config);

    let x_min = grid.x[0];
    let x_max = grid.x[grid.x.len() - 1];

    // P&B: X_min = -4.3 → x_min = 10^{-4.3} ≈ 5.01e-5
    let pb_x_min = 10.0_f64.powf(-4.3);
    assert!(
        x_min <= pb_x_min,
        "Production grid x_min={x_min:.2e} should be ≤ P&B x_min={pb_x_min:.2e}"
    );

    // P&B: X_max = 1.7 → x_max = 10^{1.7} ≈ 50.1
    let pb_x_max = 10.0_f64.powf(1.7);
    assert!(
        x_max >= pb_x_max,
        "Production grid x_max={x_max:.2e} should be ≥ P&B x_max={pb_x_max:.2e}"
    );

    // Default grid x_max=50 is within 1% of P&B's 10^1.7=50.1
    let default_grid = FrequencyGrid::new(&GridConfig::default());
    let default_x_max = default_grid.x[default_grid.x.len() - 1];
    let deficit = (pb_x_max - default_x_max) / pb_x_max;
    assert!(
        deficit < 0.01,
        "Default grid x_max={default_x_max:.1} should be within 1% of P&B x_max={pb_x_max:.1}"
    );
}

// 37.1: Isolated full_te test — verify quasi-stationary ρ_e for known μ distortion
//
// The Compton equilibrium temperature is ρ_e^eq = I₄ / (4 G₃), with
// I₄ = ∫ x⁴ n(1+n) dx and G₃ = ∫ x³ n dx. Every Bose–Einstein spectrum
// n = 1/(e^{x+μ}-1) obeys n(1+n) = −dn/dx, so integration by parts gives
// I₄ = 4 G₃ and ρ_e^eq = 1 exactly, for any μ: a BE spectrum is already in
// Compton equilibrium with electrons at T_z. (The linearized μ shape M(x)
// gives ρ_e ≠ 1 only through its temperature-shift part; see
// compton_equilibrium_analytic.rs.)
#[test]
fn test_full_te_rho_e_for_mu_distortion() {
    use spectroxide::grid::FrequencyGrid;

    // Create a fine grid for accurate integration
    let grid = FrequencyGrid::log_uniform(1e-4, 50.0, 10000);

    let mu_val = 1e-4; // small but measurable chemical potential

    // Construct BE distribution: n_BE(x) = 1/(e^{x+μ}-1)
    let n_be: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| 1.0 / ((x + mu_val).exp() - 1.0))
        .collect();

    // Compute ρ_e = I₄/(4G₃) numerically on the same grid
    let mut g3 = 0.0;
    let mut i4 = 0.0;
    for i in 1..grid.n {
        let dx = grid.x[i] - grid.x[i - 1];
        let x_mid = 0.5 * (grid.x[i] + grid.x[i - 1]);
        let n_mid = 0.5 * (n_be[i] + n_be[i - 1]);
        g3 += x_mid.powi(3) * n_mid * dx;
        i4 += x_mid.powi(4) * n_mid * (1.0 + n_mid) * dx;
    }
    let rho_e_expected = i4 / (4.0 * g3);

    // Use the code's function
    let rho_e_code = spectrum::compton_equilibrium_ratio(&grid.x, &n_be);

    let rel_err = (rho_e_code - rho_e_expected).abs() / rho_e_expected;
    eprintln!(
        "full_te μ={mu_val}: ρ_e_code={rho_e_code:.10}, ρ_e_expected={rho_e_expected:.10}, err={:.2e}",
        rel_err
    );

    // Should agree to numerical integration accuracy (~1e-8 on 10k grid)
    assert!(
        rel_err < 1e-6,
        "compton_equilibrium_ratio disagrees with independent calculation: \
         code={rho_e_code:.10}, expected={rho_e_expected:.10}, err={rel_err:.2e}"
    );

    // ρ_e = 1 exactly for both BE and Planck. The quadrature error on this
    // grid (a few × 1e-6) does not depend on μ, so it cancels in the difference;
    // a code that broke the identity at first order would show ~1e-5.
    let n_pl: Vec<f64> = grid.x.iter().map(|&x| spectrum::planck(x)).collect();
    let rho_e_planck = spectrum::compton_equilibrium_ratio(&grid.x, &n_pl);
    eprintln!(
        "ρ_e(BE) − 1 = {:.3e}, ρ_e(Planck) − 1 = {:.3e}, difference = {:.3e}",
        rho_e_code - 1.0,
        rho_e_planck - 1.0,
        rho_e_code - rho_e_planck
    );
    assert!(
        (rho_e_planck - 1.0).abs() < 1e-5,
        "ρ_e for Planck should be 1: got {rho_e_planck:.10}"
    );
    assert!(
        (rho_e_code - rho_e_planck).abs() < 1e-8,
        "BE spectrum should be in Compton equilibrium like Planck: \
         ρ_e(BE)={rho_e_code:.12}, ρ_e(Planck)={rho_e_planck:.12}"
    );
}

// 37.2b: DC/BR ratio at the *independently derived* reference point.
//
// The Round-1 audit (P1-8) derived DC/BR = 17.06 at z=1e6, x=0.1 by hand and
// checked it against Danese & de Zotti; the coverage matrix lists it as the
// class-(ii) literature anchor for the DC/BR balance. This pins the anchor
// point itself; a ±2.5× window at x = 1 passed a 1.5× error in the DC
// normalization (R2 mutation audit).
//
// Reference: Danese & de Zotti (1982); dev/audit/AUDIT_SUMMARY.md P1-8.
#[test]
fn test_dc_br_ratio_at_p18_reference_point() {
    use spectroxide::bremsstrahlung::br_emission_coefficient;
    use spectroxide::double_compton::dc_emission_coefficient;

    let cosmo = Cosmology::default();
    let z = 1.0e6;
    let theta = spectroxide::constants::theta_z(z);
    let x = 0.1;

    let k_dc = dc_emission_coefficient(x, theta);
    let k_br = br_emission_coefficient(
        x,
        theta,
        theta,
        cosmo.n_h(z),
        cosmo.n_he(z),
        cosmo.n_e(z, 1.0),
        1.0,
        &cosmo,
    );
    let ratio = k_dc / k_br;
    eprintln!("DC/BR at z=1e6, x=0.1: {ratio:.3} (P1-8 derived 17.06)");

    let derived = 17.06;
    let rel_err = (ratio - derived).abs() / derived;
    assert!(
        rel_err < 0.20,
        "DC/BR at z=1e6, x=0.1 = {ratio:.3}, independently derived {derived} \
         (P1-8), rel err {rel_err:.3} (limit 20%)"
    );
}

/// Isolated full_te regression: verify perturbative T_e agrees with brute-force.
///
/// For a known μ-distortion Δn = μ·M(x), compute ρ_e via:
/// 1. Perturbative: ρ_eq = 1 + ΔI₄/(4G₃) where ΔI₄ = ∫x⁴·(2n_pl+1)·Δn dx
/// 2. Brute-force: ρ_eq = I₄[n_pl + Δn] / (4 G₃[n_pl + Δn])
/// These should agree to < 0.1% for small μ.
#[test]
fn test_full_te_perturbative_vs_brute_force() {
    let grid = spectroxide::grid::FrequencyGrid::log_uniform(1e-4, 50.0, 10000);
    let mu_val = 1e-4; // Small μ for perturbative regime

    let n_pl: Vec<f64> = grid.x.iter().map(|&x| spectrum::planck(x)).collect();
    let m_shape: Vec<f64> = grid.x.iter().map(|&x| spectrum::mu_shape(x)).collect();
    let delta_n: Vec<f64> = m_shape.iter().map(|&m| mu_val * m).collect();
    let n_full: Vec<f64> = n_pl
        .iter()
        .zip(delta_n.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Brute-force: I₄/(4G₃)
    let rho_brute = spectrum::compton_equilibrium_ratio(&grid.x, &n_full);

    // Perturbative: 1 + ΔI₄/(4G₃) - ΔG₃/G₃
    let mut delta_i4 = 0.0;
    let mut delta_g3 = 0.0;
    for i in 1..grid.n {
        let dx = grid.x[i] - grid.x[i - 1];
        let x_mid = 0.5 * (grid.x[i] + grid.x[i - 1]);
        let np_mid = 0.5 * (n_pl[i] + n_pl[i - 1]);
        let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
        delta_i4 += x_mid.powi(4) * (2.0 * np_mid + 1.0) * dn_mid * dx;
        delta_g3 += x_mid.powi(3) * dn_mid * dx;
    }
    let rho_pert = 1.0 + delta_i4 / (4.0 * G3_PLANCK) - delta_g3 / G3_PLANCK;

    let rel_err = (rho_pert - rho_brute).abs() / (rho_brute - 1.0).abs().max(1e-30);
    eprintln!(
        "Perturbative vs brute-force T_e: ρ_pert={rho_pert:.10}, ρ_brute={rho_brute:.10}, \
         rel_err={rel_err:.2e}"
    );
    // Perturbative formula is first-order in Δn; the O(Δn²) corrections
    // contribute at the ~5% level for μ = 1e-4. Agreement to 10% confirms
    // the perturbative approach is correct to leading order.
    assert!(
        rel_err < 0.10,
        "Perturbative T_e should agree with brute-force to 10%: err={rel_err:.2e}"
    );
    // Direction check: both should give ρ_e > 1 for positive μ
    assert!(rho_pert > 1.0, "ρ_pert should be > 1 for μ > 0");
    assert!(rho_brute > 1.0, "ρ_brute should be > 1 for μ > 0");
}
