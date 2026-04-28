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
//! 5. BRpack Gaunt factor spot-checks

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
///                    (Chluba 2013 MNRAS 436, 2232, §3; Arsenadze et al. 2025 J_y fit)
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
// 7. BRpack Gaunt factor spot-checks
//    Verify the softplus Gaunt factor at specific (x, θ_e) values.
// =========================================================================

/// Non-relativistic free-free Gaunt factor: closed-form regression test
/// against the CRB 2020 softplus interpolation formula.
///
/// Oracle:             Chluba, Ravenni & Bolliet (2020) MNRAS 492, 177
///                     non-relativistic softplus fit (implemented at
///                     src/bremsstrahlung.rs:64):
///                     g_ff(x, θ_e, Z) = 1 + softplus[√3/π · ln(2.25·√θ_e/(x·Z)) + 1.425]
///                     where softplus(a) = ln(1+exp(a)).
/// Expected:           hand-computed from the formula above at each point.
/// Oracle uncertainty: machine ε (exact formula).
/// Tolerance:          1e-10 relative.
///
/// This is a regression test against the CRB 2020 paper formula. The oracle
/// is the paper's published expression, not the code — if the code changes
/// to a different fit form, this test must fail and be re-derived.
/// (Simple Born approximation g_Born = (√3/π)·ln(2.25/(γ_E·x)) does NOT
/// apply here because x >> √θ_e at the test points; CRB use the softplus
/// interpolation to smoothly handle the hard-photon regime.)
#[test]
fn brpack_gaunt_factor_spot_checks() {
    use spectroxide::bremsstrahlung::gaunt_ff_nr;

    let s3_pi = (3.0_f64).sqrt() / std::f64::consts::PI;
    let softplus = |a: f64| (1.0 + a.exp()).ln();
    let expected = |x: f64, theta_e: f64, z: f64| -> f64 {
        1.0 + softplus(s3_pi * ((2.25 / (x * z)).ln() + 0.5 * theta_e.ln()) + 1.425)
    };

    let test_points = [
        (0.01_f64, 1e-4_f64, 1.0_f64),
        (0.1, 1e-4, 1.0),
        (1.0, 1e-4, 1.0),
        (5.0, 1e-4, 1.0),
        (0.01, 1e-3, 1.0),
        (1.0, 1e-3, 1.0),
        (0.1, 1e-4, 2.0),
    ];
    for &(x, theta_e, z) in &test_points {
        let g_code = gaunt_ff_nr(x, theta_e, z);
        let g_expected = expected(x, theta_e, z);
        let rel_err = (g_code - g_expected).abs() / g_expected;
        eprintln!(
            "g_ff(x={x}, θ_e={theta_e:.0e}, Z={z}): code={g_code:.6}, \
             formula={g_expected:.6}, rel_err={:.2e}",
            rel_err,
        );
        assert!(
            rel_err < 1e-10,
            "CRB 2020 formula at (x={x}, θ_e={theta_e}, Z={z}): \
             code {g_code} vs formula {g_expected} (rel_err {rel_err:.2e}, tol 1e-10)"
        );
        assert!(g_code >= 1.0, "g_ff must be ≥ 1 (softplus + 1 floor)");
    }

    // Z-dependence sign: larger Z → smaller g (ln argument decreases).
    let g_z1 = gaunt_ff_nr(0.1, 1e-4, 1.0);
    let g_z2 = gaunt_ff_nr(0.1, 1e-4, 2.0);
    assert!(
        g_z1 > g_z2,
        "g_ff should decrease with Z: Z=1 gave {g_z1}, Z=2 gave {g_z2}"
    );

    // Monotonicity in x at fixed θ_e.
    let mut prev_gff = f64::MAX;
    for &x in &[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0] {
        let g = gaunt_ff_nr(x, 1e-4, 1.0);
        assert!(
            g <= prev_gff + 0.01,
            "Gaunt factor not monotonically decreasing at x={x}: {g} > prev {prev_gff}"
        );
        prev_gff = g;
    }
}
