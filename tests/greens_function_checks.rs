//! Green's function module checks.
//!
//! Tests the GF spectral shapes, analytic limits, energy conservation,
//! and PDE cross-validation. Renamed from chluba2013_checks.rs because
//! no test in this file compares against actual Chluba (2013) numerical
//! values — they are internal consistency and analytic-limit tests.
//!
//! 1. G_th(x, z_h) spectral shapes at multiple z_h (PDE vs GF)
//! 2. Analytic limits: pure temperature shift, pure mu, pure y
//! 3. Energy conservation: ∫x³ G_th dx / G₃ ≈ 1
//! 4. PDE cross-validation of GF decomposition accuracy
//! 5. (removed: GF decomposition duplicate)
//! 6. Free-free Gaunt factor: Draine (2011) classical-limit anchor and
//!    invariance of K_BR under θ_z at fixed ν and T_e

use spectroxide::constants::*;
use spectroxide::greens;
use spectroxide::prelude::*;

fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-30)
}

// =========================================================================
// 1. G_th spectral shapes: PDE solver reproduces GF at multiple z_h
//    (This is the analog of computing Fig. 1)
// =========================================================================

#[test]
fn chluba2013_pde_spectral_shape_mu_era() {
    // At z_h = 2e5 (deep mu-era), the PDE spectral shape should
    // closely match the Green's function shape point-by-point,
    // not just in the extracted mu/y parameters.
    let z_h = 2.0e5;
    let drho = 1.0e-5;
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
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
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[1.0e4]);
    let snap = solver.snapshots.last().unwrap();

    // Compare PDE Delta_n(x) with GF Delta_n(x) at several x values.
    // Exclude x near the mu zero-crossing (beta_mu ≈ 2.19) where the
    // absolute value is tiny and relative errors are amplified.
    let x_test = [1.0, 3.0, 5.0, 8.0, 12.0];
    for &x in &x_test {
        let idx = solver.grid.x.iter().position(|&xi| xi > x).unwrap_or(0);
        if idx == 0 || idx >= solver.grid.n {
            continue;
        }
        // Linear interpolation
        let x0 = solver.grid.x[idx - 1];
        let x1 = solver.grid.x[idx];
        let t = (x - x0) / (x1 - x0);
        let dn_pde = snap.delta_n[idx - 1] + t * (snap.delta_n[idx] - snap.delta_n[idx - 1]);
        let dn_gf = greens::greens_function(x, z_h) * drho;

        if dn_gf.abs() > 1e-10 {
            let err = rel_err(dn_pde, dn_gf);
            eprintln!(
                "x={x:.1}: PDE={dn_pde:.4e}, GF={dn_gf:.4e}, err={:.1}%",
                err * 100.0
            );
            // Allow 20% for shape comparison (tighter than mu/y which are integrated)
            assert!(
                err < 0.20,
                "Spectral shape mismatch at x={x}: PDE={dn_pde:.4e}, GF={dn_gf:.4e}, err={:.1}%",
                err * 100.0
            );
        }
    }
}

// =========================================================================
// 2. Analytic limits from Chluba (2013)
// =========================================================================

#[test]
fn chluba2013_limit_pure_temperature_shift() {
    // At z_h >> z_mu (≈ 2e6), the distortion is fully thermalized
    // and G_th(x, z_h) → (1/4) G(x) = (1/4) x e^x/(e^x-1)^2
    //
    // This is because all injected energy becomes a temperature shift:
    // Delta_n = (Delta T/T) * G(x) with Delta T/T = (1/4) Delta_rho/rho.
    //
    // Chluba (2013), Section 2.1.
    let z_h = 5.0e6;
    let x_test = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0];

    for &x in &x_test {
        let g_th = greens::greens_function(x, z_h);
        let t_shift = 0.25 * spectrum::g_bb(x);

        let err = rel_err(g_th, t_shift);
        assert!(
            err < 0.005,
            "Pure T-shift limit at z_h=5e6, x={x}: G_th={g_th:.6e}, (1/4)G={t_shift:.6e}, err={:.3}%",
            err * 100.0
        );
    }
}

#[test]
fn chluba2013_limit_pure_mu() {
    // At z_h ~ 3e5 (deep mu-era, below full thermalization):
    // G_th(x, z_h) ≈ (3/κ_c) * J_bb*(z_h) * J_mu(z_h) * M(x)
    //
    // The y-component should be negligible.
    // Chluba (2013), Section 2.2.
    let z_h = 3.0e5;
    let j_mu = greens::visibility_j_mu(z_h);
    let j_bb = greens::visibility_j_bb_star(z_h);
    let j_y = greens::visibility_j_y(z_h);

    // J_mu should be ~1, J_y should be ~0 at z=3e5
    assert!(j_mu > 0.99, "J_mu(3e5) should be ~1: got {j_mu:.4}");
    assert!(j_y < 0.05, "J_y(3e5) should be ~0: got {j_y:.4}");

    // Check that the spectral shape is mu-dominated
    for &x in &[1.0, 3.0, 5.0, 8.0] {
        let mu_part = (3.0 / KAPPA_C) * j_mu * j_bb * spectrum::mu_shape(x);
        let y_part = 0.25 * j_y * spectrum::y_shape(x);

        // mu-component should dominate
        if mu_part.abs() > 1e-10 {
            let y_fraction = y_part.abs() / mu_part.abs();
            assert!(
                y_fraction < 0.05,
                "At z_h=3e5, x={x}: y/mu ratio = {y_fraction:.3} (should be <0.05)"
            );
        }
    }
}

#[test]
fn chluba2013_limit_pure_y() {
    // At z_h ~ 2e3 (y-era):
    // G_th(x, z_h) ≈ (1/4) * J_y(z_h) * Y_SZ(x)
    //
    // The mu-component should be negligible.
    // Chluba (2013), Section 2.3.
    let z_h = 2.0e3;
    let j_mu = greens::visibility_j_mu(z_h);
    let j_y = greens::visibility_j_y(z_h);

    assert!(j_mu < 0.01, "J_mu(2e3) should be ~0: got {j_mu:.4}");
    assert!(j_y > 0.99, "J_y(2e3) should be ~1: got {j_y:.4}");

    // y-parameter should be ~(1/4) * Delta_rho/rho
    let y_expected = 0.25 * j_y;
    assert!(
        rel_err(y_expected, 0.25) < 0.01,
        "y coefficient at z=2e3: {y_expected:.6} (expected ~0.25)"
    );
}

// =========================================================================
// 3. Energy conservation: ∫x³ G_th dx / G₃ = 1 for all z_h
//    (Tests that the GF preserves injected energy at all redshifts)
// =========================================================================

/// Energy conservation of the Green's function ∫x³ G_th dx / G₃ = 1, split
/// by regime to expose the ansatz's transition-region residual instead of
/// hiding it behind a single wide tolerance.
///
/// Oracle:            J_μ·J_bb* + J_y + (1 − J_bb*) = 1 in pure μ- and y-eras
///                    (Chluba 2013 MNRAS 434, 352, §3; Arsenadze et al. 2025 J_y fit)
/// Expected:          E/G₃ = 1 exactly
/// Oracle uncertainty: 2-3% in pure regimes (fit residuals of J_μ, J_y, J_bb*)
/// Tolerance:
///   - Pure μ-era   (z_h ≥ 3e5):   3%
///   - Pure y-era   (z_h ≤ 3e3):   3%
///   - Transition   (3e3 < z_h < 3e5): logged only, bounded ≤ 22%
///
/// Previous version asserted 20% globally, which cannot detect a 3% regression
/// in the μ- or y-era where the ansatz *should* be exact to ~2%. That wide
/// tolerance was chosen to accommodate the known ~17% peak in the transition
/// region — but widening everywhere hides regressions everywhere.
#[test]
fn chluba2013_energy_conservation() {
    let x = spectroxide::grid::FrequencyGrid::log_uniform(1e-3, 50.0, 5000);
    let g3 = std::f64::consts::PI.powi(4) / 15.0;

    let mut transition_max_err: f64 = 0.0;
    let mut transition_max_z: f64 = 0.0;

    for &z_h in &[2e3_f64, 1e4, 3e4, 5e4, 8e4, 1e5, 2e5, 5e5, 2e6] {
        let mut energy = 0.0;
        for i in 1..x.n {
            let dx = x.x[i] - x.x[i - 1];
            let x_mid = 0.5 * (x.x[i] + x.x[i - 1]);
            let g = greens::greens_function(x_mid, z_h);
            energy += x_mid.powi(3) * g * dx;
        }
        let ratio = energy / g3;
        let err = rel_err(ratio, 1.0);
        eprintln!(
            "Energy integral at z_h={z_h:.0e}: E/G3 = {ratio:.6}  (err {:.2}%)",
            err * 100.0
        );

        if z_h >= 3e5 || z_h <= 3e3 {
            // Pure regime: ansatz should conserve to a few percent.
            assert!(
                err < 0.03,
                "GF energy in pure regime at z_h={z_h:.0e}: E/G₃ = {ratio:.6} \
                 (err {:.2}%, tol 3%)",
                err * 100.0
            );
        } else {
            // Transition: log the max but bound it loosely to catch regressions
            // worse than the known ~17% peak.
            if err > transition_max_err {
                transition_max_err = err;
                transition_max_z = z_h;
            }
        }
    }

    eprintln!(
        "Transition region max error: {:.2}% at z_h={:.0e}",
        transition_max_err * 100.0,
        transition_max_z
    );
    assert!(
        transition_max_err < 0.22,
        "Transition-region GF energy error {:.2}% at z_h={:.0e} exceeds 22% — \
         the ansatz residual has grown beyond its historical maximum, \
         investigate J_y fit or ansatz form.",
        transition_max_err * 100.0,
        transition_max_z,
    );
}

// =========================================================================
// 4. Visibility functions: physical properties
//    (Tests physical constraints, NOT the fitting formulas against themselves)
// =========================================================================

// chluba2013_visibility_function_physical_properties removed: duplicated by
// test_visibility_function_physical_constraints in heat_injection.rs (wider
// z-range, finer sampling) and test_visibility_functions_physical_bounds
// in greens.rs unit tests.

#[test]
fn chluba2013_visibility_pde_cross_validation() {
    // The strongest test of the visibility functions: compare GF predictions
    // (which use the fitting formulas) against PDE results (which use no
    // fitting formulas). This tests whether the fitting formulas actually
    // match the physics.
    //
    // GF prediction: μ = 1.401 × J_bb*(z_h) × J_mu(z_h) × Δρ/ρ
    // PDE: extract μ from full numerical solution

    let cosmo = Cosmology::default();
    let drho = 1.0e-5;

    // Test at z_h = 2e5 (deep mu-era where GF should be accurate)
    let z_h = 2.0e5;
    let mu_gf =
        (3.0 / KAPPA_C) * greens::visibility_j_bb_star(z_h) * greens::visibility_j_mu(z_h) * drho;

    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
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

    let err = rel_err(snap.mu, mu_gf);
    eprintln!(
        "PDE μ = {:.4e}, GF μ = {mu_gf:.4e}, err = {:.1}%",
        snap.mu,
        err * 100.0
    );
    // PDE vs GF should agree within 12% in the deep mu-era
    assert!(
        err < 0.12,
        "Visibility PDE cross-validation failed: μ_PDE={:.4e}, μ_GF={mu_gf:.4e}, err={:.1}%",
        snap.mu,
        err * 100.0
    );
}

// =========================================================================
// 5. GF decomposition: energy-conserving t_part sums correctly
// chluba2013_gf_energy_integral_normalized removed: exact duplicate of
// chluba2013_energy_conservation above (same integral, same 20% tolerance).

// =========================================================================
// 6. Free-free Gaunt factor: classical-limit anchor and T_e dependence
//
// The physical thermal Gaunt factor depends on hν/kT_e and T_e only, never
// on the photon temperature T_z. These tests pin the Gaunt fit against a
// textbook formula (not against the code's own expression) and check that
// every K_BR code path evaluates it at x_e = x/ρ_e = hν/kT_e.
// =========================================================================

/// √3/π and C = 2^{5/2} e^{−5γ_E/2}/α, typed as literals. The mpmath
/// check (dps = 40, α = 7.2973525693e-3, CODATA 2018) gives
/// C = 183.107323379400751..., √3/π = 0.551328895421792049...
const SQRT3_OVER_PI: f64 = 0.551_328_895_421_792_1;
const DRAINE_C: f64 = 183.107_323_379_400_75;

/// Classical (low-frequency, kT_e ≪ Z² Ry) thermal Gaunt factor, Draine (2011),
/// *Physics of the Interstellar and Intergalactic Medium*, Eq. 10.9:
///
///   g_ff = (√3/π) [ ln( (2kT_e)^{3/2} / (π Z e² m_e^{1/2} ν) ) − (5/2) γ_E ]   (Gaussian e²)
///
/// In code variables, with e² = α ħ c, hν = x_e kT_e and kT_e = θ_e m_e c²,
/// the log argument is (2θ_e m_e c²)^{3/2} / (π Z α ħ c m_e^{1/2} · x_e θ_e m_e c²/(2πħ))
/// = 2^{5/2} θ_e^{1/2} / (α Z x_e). Folding e^{−5γ_E/2} into the constant:
///
///   g_D(x_e, θ_e, Z) = (√3/π) ln( C θ_e^{1/2} / (Z x_e) ),   C = 2^{5/2} e^{−5γ_E/2}/α.
fn gaunt_draine(x_e: f64, theta_e: f64, z: f64) -> f64 {
    SQRT3_OVER_PI * (DRAINE_C * theta_e.sqrt() / (z * x_e)).ln()
}

/// The code's fit, g = 1 + softplus[(√3/π) ln(2.25 θ_e^{1/2}/(Z x_e)) + 1.425],
/// tends to 1 + (√3/π) ln(2.25 θ_e^{1/2}/(Z x_e)) + 1.425 when the argument is
/// large. That equals g_D when 1.425 = (√3/π) ln(C/2.25) − 1 = 1.425374, so the
/// fit's low-frequency limit is Draine's formula up to 3.7e-4 plus the softplus
/// remainder ln(1 + e^{−arg}). The test points keep arg ≥ 9 (remainder ≤ 1.3e-4),
/// so the absolute tolerance is 1e-3. The point is the *variables*: with x = x_e
/// and θ_e the fit reproduces the classical √θ_e/ν dependence exactly.
#[test]
fn gaunt_ff_classical_limit_matches_draine() {
    use spectroxide::bremsstrahlung::gaunt_ff_nr;

    // θ_e ≤ 3e-6 keeps kT_e ≤ 0.1 Z² Ry (Ry/m_ec² = 2.7e-5), where the classical
    // formula applies.
    for &theta_e in &[1e-8_f64, 1e-6, 3e-6] {
        for &x_e in &[1e-13_f64, 1e-12, 1e-11] {
            for &z in &[1.0_f64, 2.0] {
                let g_code = gaunt_ff_nr(x_e, theta_e, z);
                let g_d = gaunt_draine(x_e, theta_e, z);
                // arg = g_D − 1.000374, so g_D > 10 keeps arg ≥ 9.
                assert!(g_d > 10.0, "test point outside the asymptotic regime");
                let diff = (g_code - g_d).abs();
                assert!(
                    diff < 1e-3,
                    "g_ff(x_e={x_e:e}, θ_e={theta_e:e}, Z={z}) = {g_code:.6} vs \
                     Draine Eq. 10.9 {g_d:.6} (|Δ| = {diff:.2e}, tol 1e-3)"
                );
            }
        }
    }

    // Monotone decreasing in x_e, floor g ≥ 1, and larger Z gives smaller g.
    let mut prev = f64::MAX;
    for &x in &[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0] {
        let g = gaunt_ff_nr(x, 1e-4, 1.0);
        assert!(
            g >= 1.0 && g <= prev,
            "g_ff not monotone/≥1 at x_e={x}: {g}"
        );
        assert!(
            gaunt_ff_nr(x, 1e-4, 2.0) < g,
            "g_ff(Z=2) ≥ g_ff(Z=1) at x_e={x}"
        );
        prev = g;
    }
}

/// K_BR prefactor α λ_e³/(2π√(6π)), λ_e = h/(m_e c), from CODATA 2018 values typed
/// here (Chluba & Sunyaev 2012, Eq. 14). Nothing is imported from constants.rs.
fn br_prefactor_literal() -> f64 {
    let alpha = 7.297_352_569_3e-3_f64;
    let h = 6.626_070_15e-34_f64;
    let m_e = 9.109_383_701_5e-31_f64;
    let c = 299_792_458.0_f64;
    let lam = h / (m_e * c);
    alpha * lam.powi(3) / (2.0 * std::f64::consts::PI * (6.0 * std::f64::consts::PI).sqrt())
}

/// Every K_BR code path evaluated at (x, θ_e, θ_z); returns (name, K_BR).
fn all_kbr_paths(
    x: f64,
    theta_e: f64,
    theta_z: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    y_he_ii: f64,
    y_he_i: f64,
) -> Vec<(&'static str, f64)> {
    use spectroxide::bremsstrahlung::*;
    let pre = br_precompute(theta_e, theta_z, n_h, n_he, n_e, 1.0, y_he_ii, y_he_i).unwrap();
    let expc = gaunt_expc_factor(x.ln());
    vec![
        (
            "with_he",
            br_emission_coefficient_with_he(
                x, theta_e, theta_z, n_h, n_he, n_e, 1.0, y_he_ii, y_he_i,
            ),
        ),
        ("fast", br_emission_coefficient_fast(x, &pre)),
        (
            "fast_preln",
            br_emission_coefficient_fast_preln(x, x.ln(), &pre),
        ),
        (
            "expc (production)",
            br_emission_coefficient_expc(x, expc, &pre),
        ),
        (
            "and_drho_expc (production)",
            br_emission_coefficient_and_drho_expc(x, expc, &pre).0,
        ),
    ]
}

/// Draine anchor through K_BR at ρ_e ≠ 1. For a pure-hydrogen plasma,
/// K_BR = P θ_e^{−7/2} e^{−x_e} φ^{−3} N_HII g_ff(x_e, θ_e, 1), with φ = 1/ρ_e and P the
/// literal prefactor, so g_eff = K_BR φ³ e^{x_e} θ_e^{7/2} / (P N_HII) must equal the
/// Draine g_D(x_e, θ_e, 1) to the 1e-3 of `gaunt_ff_classical_limit_matches_draine`.
/// A Gaunt fit evaluated at x = x_e ρ_e instead of x_e is off by (√3/π) ln(1/ρ_e),
/// which is 0.38 at ρ_e = 0.5 (the P-1 bug in dev/REVIEW_2026-09-22.md).
#[test]
fn br_coefficient_gaunt_uses_electron_frequency_draine_anchor() {
    let cosmo = Cosmology::default();
    let p = br_prefactor_literal();
    let theta_e = 1e-6_f64;
    let n_h = 1e6_f64;
    for &rho_e in &[0.5_f64, 0.62, 1.0, 1.6] {
        let theta_z = theta_e / rho_e;
        let phi = theta_z / theta_e;
        for &x_e in &[1e-11_f64, 1e-10, 1e-9] {
            let x = x_e * rho_e;
            let g_d = gaunt_draine(x_e, theta_e, 1.0);
            let mut paths = all_kbr_paths(x, theta_e, theta_z, n_h, 0.0, n_h, 0.0, 0.0);
            // The Saha entry point too: with N_He = 0 its helium fractions drop out.
            paths.push((
                "br_emission_coefficient",
                spectroxide::bremsstrahlung::br_emission_coefficient(
                    x, theta_e, theta_z, n_h, 0.0, n_h, 1.0, &cosmo,
                ),
            ));
            for (name, k) in paths {
                let g_eff = k * phi.powi(3) * x_e.exp() * theta_e.powf(3.5) / (p * n_h);
                let diff = (g_eff - g_d).abs();
                assert!(
                    diff < 1e-3,
                    "{name}: ρ_e={rho_e}, x_e={x_e:e}: g_eff={g_eff:.6} vs Draine {g_d:.6} \
                     (|Δ| = {diff:.2e}, tol 1e-3)"
                );
            }
        }
    }
}

/// Invariance under θ_z at fixed ν and T_e. The BR absorption rate per unit time
/// depends on ν, T_e and the ion densities only. In the code's variables
/// K_BR = P θ_e^{−7/2} e^{−x_e} φ^{−3} Σ_i Z_i² N_i g_ff(x_e, θ_e, Z_i), so
///
///   I ≡ K_BR φ³ e^{x_e} θ_e^{7/2} = P Σ_i Z_i² N_i g_ff(x_e, θ_e, Z_i)
///
/// is a function of (x_e, θ_e, N_i) alone. Changing θ_z at fixed θ_e and
/// x_e = x θ_z/θ_e (so x = x_e ρ_e moves with it) must leave I unchanged to
/// rounding, for H⁺, He⁺ and He²⁺ alike. The helium fractions are held fixed; the
/// Saha entry point is excluded because it legitimately maps θ_z to a redshift.
#[test]
fn br_coefficient_invariant_under_theta_z_at_fixed_nu_and_te() {
    let (n_h, n_he, n_e) = (1.0e6_f64, 8.0e4, 1.1e6);
    let (y_he_ii, y_he_i) = (0.3_f64, 0.9);
    for &theta_e in &[3e-8_f64, 1e-6, 1e-5] {
        for &x_e in &[1e-4_f64, 1e-2, 0.3, 3.0] {
            let invariant = |rho_e: f64| -> Vec<(&'static str, f64)> {
                let theta_z = theta_e / rho_e;
                let phi = theta_z / theta_e;
                let x = x_e * rho_e;
                all_kbr_paths(x, theta_e, theta_z, n_h, n_he, n_e, y_he_ii, y_he_i)
                    .into_iter()
                    .map(|(n, k)| (n, k * phi.powi(3) * (x * phi).exp() * theta_e.powf(3.5)))
                    .collect()
            };
            let reference = invariant(1.0);
            for &rho_e in &[0.3_f64, 0.62, 1.4] {
                for ((name, i_ref), (_, i_rho)) in reference.iter().zip(invariant(rho_e)) {
                    let rel = (i_rho - i_ref).abs() / i_ref.abs();
                    assert!(
                        rel < 1e-12,
                        "{name}: K_BR φ³ e^(x_e) θ_e^(7/2) changed with θ_z at θ_e={theta_e:e}, \
                         x_e={x_e}, ρ_e={rho_e}: {i_rho:e} vs {i_ref:e} at ρ_e=1 (rel {rel:.2e})"
                    );
                }
            }
        }
    }
}
