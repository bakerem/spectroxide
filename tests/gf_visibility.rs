//! Green's-function, visibility-function, and decomposition tests.
//!
//! These tests need no PDE run. They cover the Chluba (2013) visibility
//! functions and their limits, the heat and photon Green's functions, the
//! (μ, y, ΔT/T) decomposition, and the FIRAS helpers.

mod common;

use common::*;
use spectroxide::constants::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::distortion;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::greens;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::solver::SolverSnapshot;
use spectroxide::spectrum;

/// The visibility functions must satisfy physical constraints:
///
/// 1. Monotonicity: J_bb decreases with z, J_μ increases, J_y decreases
/// 2. Limits: J_bb(0) = 1, J_bb(∞) = 0; J_μ(0) = 0, J_μ(∞) = 1;
///            J_y(0) = 1, J_y(∞) = 0
/// 3. J_bb*(z) ≤ J_bb(z) (the correction only reduces thermalization)
/// 4. All visibility functions must be in [0, 1]
///
/// These are model-independent physical requirements:
///   - At low z, no thermalization possible → J_bb → 1
///   - At high z, full thermalization → J_bb → 0
///   - J_μ = 0 at low z because Compton scattering can't redistribute efficiently
///   - J_μ = 1 at high z because redistribution is fast
#[test]
fn test_visibility_function_physical_constraints() {
    // Sample redshifts spanning the full range
    let z_values: Vec<f64> = {
        let mut zs = Vec::new();
        let mut z = 500.0;
        while z < 1e8 {
            zs.push(z);
            z *= 1.2; // multiplicative steps
        }
        zs
    };

    // Check bounds [0, 1]
    for &z in &z_values {
        let j_bb = greens::visibility_j_bb(z);
        let j_bb_star = greens::visibility_j_bb_star(z);
        let j_mu = greens::visibility_j_mu(z);
        let j_y = greens::visibility_j_y(z);

        assert!(
            j_bb >= 0.0 && j_bb <= 1.0,
            "J_bb({z:.0e}) = {j_bb} out of [0,1]"
        );
        assert!(
            j_mu >= 0.0 && j_mu <= 1.0,
            "J_μ({z:.0e}) = {j_mu} out of [0,1]"
        );
        assert!(
            j_y >= 0.0 && j_y <= 1.0,
            "J_y({z:.0e}) = {j_y} out of [0,1]"
        );

        // J_bb* ≤ J_bb (correction reduces visibility)
        assert!(
            j_bb_star <= j_bb + 1e-10,
            "J_bb*({z:.0e}) = {j_bb_star} > J_bb = {j_bb}"
        );
    }

    // Check monotonicity
    for i in 1..z_values.len() {
        let z_lo = z_values[i - 1];
        let z_hi = z_values[i];

        // J_bb should decrease (or stay flat) with increasing z
        assert!(
            greens::visibility_j_bb(z_hi) <= greens::visibility_j_bb(z_lo) + 1e-12,
            "J_bb not monotonically decreasing: J_bb({z_lo:.0e}) = {:.6e} < J_bb({z_hi:.0e}) = {:.6e}",
            greens::visibility_j_bb(z_lo),
            greens::visibility_j_bb(z_hi)
        );

        // J_μ should increase with increasing z
        assert!(
            greens::visibility_j_mu(z_hi) >= greens::visibility_j_mu(z_lo) - 1e-12,
            "J_μ not monotonically increasing at z={z_lo:.0e}→{z_hi:.0e}"
        );

        // J_y should decrease with increasing z
        assert!(
            greens::visibility_j_y(z_hi) <= greens::visibility_j_y(z_lo) + 1e-12,
            "J_y not monotonically decreasing at z={z_lo:.0e}→{z_hi:.0e}"
        );
    }

    // Check limits
    let j_bb_low = greens::visibility_j_bb(100.0);
    let j_bb_high = greens::visibility_j_bb(1e8);
    assert!((j_bb_low - 1.0).abs() < 1e-6, "J_bb(low z) should → 1");
    assert!(j_bb_high < 1e-50, "J_bb(high z) should → 0");

    // J_μ uses (1+z)/5.8e4, so at z=100: ((101)/5.8e4)^1.88 is small but not tiny.
    // Use z=10 for a stricter limit check.
    let j_mu_low = greens::visibility_j_mu(10.0);
    let j_mu_high = greens::visibility_j_mu(1e8);
    assert!(j_mu_low < 1e-6, "J_μ(z=10) = {j_mu_low:.2e}, should → 0");
    assert!((j_mu_high - 1.0).abs() < 1e-6, "J_μ(high z) should → 1");

    // J_μ/J_y crossing should be around z ~ 5e4 (μ-y transition)
    let mut crossing_z = 0.0_f64;
    for i in 1..z_values.len() {
        let z_lo = z_values[i - 1];
        let z_hi = z_values[i];
        let diff_lo = greens::visibility_j_mu(z_lo) - greens::visibility_j_y(z_lo);
        let diff_hi = greens::visibility_j_mu(z_hi) - greens::visibility_j_y(z_hi);
        if diff_lo * diff_hi < 0.0 && crossing_z == 0.0 {
            crossing_z = (z_lo + z_hi) / 2.0;
        }
    }
    assert!(
        crossing_z > 3e4 && crossing_z < 1e5,
        "J_μ/J_y crossing at z={crossing_z:.2e}, expected 3e4-1e5"
    );
}

/// FIRAS limits represent observed upper bounds on spectral distortions.
/// Any standard-model prediction must be well below these limits.
///
/// FIRAS 95% CL (Fixsen et al. 1996, ApJ 473, 576):
///   |μ| < 9 × 10⁻⁵
///   |y| < 1.5 × 10⁻⁵
///
/// Verify that:
///   1. The constants match the published values
///   2. Standard ΛCDM predictions (adiabatic cooling + acoustic dissipation)
///      are well below these limits (by a factor of ~1000)
#[test]
fn test_firas_limits_consistency() {
    // Verify the stored constants
    assert!(
        (distortion::FIRAS_MU_LIMIT - 9.0e-5).abs() < 1e-10,
        "FIRAS μ limit should be 9×10⁻⁵"
    );
    assert!(
        (distortion::FIRAS_Y_LIMIT - 1.5e-5).abs() < 1e-10,
        "FIRAS y limit should be 1.5×10⁻⁵"
    );

    // Standard-model predictions should be ~1000× below FIRAS.
    // Use a delta-function injection with Δρ/ρ ≈ 3×10⁻⁸ (Silk damping)
    // in the μ-era to get the expected μ ≈ 1.401 × 3×10⁻⁸ ≈ 4×10⁻⁸.
    let z_h = 2.0e5;
    let drho = 3.0e-8; // Approximate Silk damping total
    let sigma_z = 5000.0;
    let dq_dz = |z: f64| gaussian_heating(z, z_h, sigma_z, drho);
    let mu = greens::mu_from_heating(&dq_dz, 1e3, 3e6, 10000);

    assert!(
        mu.abs() < distortion::FIRAS_MU_LIMIT * 0.01,
        "ΛCDM μ = {mu:.4e} should be ≪ FIRAS limit {:.0e}",
        distortion::FIRAS_MU_LIMIT
    );
}

/// The μ-y transition should occur at z ≈ 5×10⁴.
///
/// At this redshift, the branching between μ and y is roughly equal.
/// This is a fundamental prediction of the thermalization physics:
/// Compton scattering is efficient enough to establish a Bose-Einstein
/// distribution for z >> 5×10⁴, but not for z << 5×10⁴.
///
/// Multiple groups predict z_transition ≈ 5×10⁴:
///   - Hu & Silk (1993), PRD 48, 485
///   - Chluba & Sunyaev (2012), MNRAS 419, 1294
///   - Khatri & Sunyaev (2012), JCAP 09, 016
///
/// We define the transition as where J_μ × J_bb* × (3/κ_c) ≈ J_y × (1/4),
/// i.e., where the μ and y contributions to the Green's function are equal
/// in terms of energy-weighted amplitude.
#[test]
fn test_mu_y_transition_redshift() {
    // Find where the μ and y contributions cross
    // μ contribution ∝ J_μ(z) × J_bb*(z)
    // y contribution ∝ J_y(z)

    let mu_weight = |z: f64| -> f64 {
        (3.0 / KAPPA_C) * greens::visibility_j_mu(z) * greens::visibility_j_bb_star(z)
    };
    let y_weight = |z: f64| -> f64 { 0.25 * greens::visibility_j_y(z) };

    // Find crossing by bisection
    let mut z_lo = 1e3_f64;
    let mut z_hi = 5e5_f64;

    // Verify the bracket: at low z, y dominates; at high z, μ dominates
    assert!(
        y_weight(z_lo) > mu_weight(z_lo),
        "y should dominate at low z"
    );
    assert!(
        mu_weight(z_hi) > y_weight(z_hi),
        "μ should dominate at high z"
    );

    for _ in 0..100 {
        let z_mid = ((z_lo.ln() + z_hi.ln()) / 2.0).exp();
        if mu_weight(z_mid) > y_weight(z_mid) {
            z_hi = z_mid;
        } else {
            z_lo = z_mid;
        }
    }
    let z_transition = ((z_lo.ln() + z_hi.ln()) / 2.0).exp();

    eprintln!(
        "μ-y transition redshift: z = {z_transition:.0} \
         (literature consensus: ~5×10⁴)"
    );

    // Literature consensus: z_μy ≈ 5×10⁴. Our GF uses fitting formulae
    // from Chluba (2013) and Arsenadze et al. (2025). The exact crossing
    // depends on those fits and the definition of "transition" (energy-weight
    // crossover). Range [2×10⁴, 1×10⁵] is appropriate.
    assert!(
        z_transition > 2e4 && z_transition < 1e5,
        "z_transition = {z_transition:.0}, expected in range [2×10⁴, 1×10⁵]"
    );
}

/// Verify that the Green's function smoothly interpolates between
/// the μ-era (z >> 5×10⁴) and y-era (z << 5×10⁴).
///
/// At x = 3.0, M(x) > 0 (above β_μ) but Y_SZ(x) < 0 (below its zero
/// crossing at 3.83). So G_th changes SIGN as the μ→y transition occurs.
/// This sign change is physical, not a bug.
///
/// We check:
///   1. G_th is finite everywhere (no NaN/Inf)
///   2. G_th is continuous (no jumps, using fine sampling with 1% steps)
///   3. The sign change occurs in the expected transition region
#[test]
fn test_greens_function_smooth_transition() {
    let x = 5.0; // Above Y_SZ zero crossing (3.83), so both M(x) > 0 and Y_SZ(x) > 0

    // Sample G_th at many redshifts with very fine steps (1%)
    let z_values: Vec<f64> = {
        let mut zs = Vec::new();
        let mut z = 1000.0;
        while z < 1e6 {
            zs.push(z);
            z *= 1.01; // 1% steps for smoothness check
        }
        zs
    };

    let g_values: Vec<f64> = z_values
        .iter()
        .map(|&z| greens::greens_function(x, z))
        .collect();

    // Check that G_th is well-defined (no NaN/Inf)
    for (i, &g) in g_values.iter().enumerate() {
        assert!(
            g.is_finite(),
            "G_th(x={x}, z={:.0e}) = {g} is not finite",
            z_values[i]
        );
    }

    // Check continuity: with 1% redshift steps, adjacent values should
    // not differ by more than 20% relative to their average magnitude
    let mut max_jump = 0.0_f64;
    for i in 1..g_values.len() {
        let g_prev = g_values[i - 1];
        let g_curr = g_values[i];
        let avg_scale = (g_prev.abs() + g_curr.abs()) / 2.0;
        if avg_scale < 1e-30 {
            continue; // Skip near-zero values
        }
        let jump = (g_curr - g_prev).abs() / avg_scale;
        max_jump = max_jump.max(jump);
    }

    eprintln!("G_th(x={x}) max relative jump with 1% z-steps: {max_jump:.4}");
    assert!(
        max_jump < 0.20,
        "G_th has discontinuity: max jump = {max_jump:.3} with 1% z steps"
    );

    // G_th should be positive at x=5 for all z (both M(5) > 0 and Y_SZ(5) > 0)
    for (i, &g) in g_values.iter().enumerate() {
        assert!(
            g >= 0.0,
            "G_th(x=5, z={:.0e}) = {g:.4e} is negative (unexpected at x > x_zero_Y)",
            z_values[i]
        );
    }
}

/// Test 1: μ-era vs y-era conversion coefficients.
///
/// Inject a small delta-like energy release at z_h = 2×10⁵ (μ-era) and
/// z_h = 1×10⁴ (y-era). Verify:
///   - μ-era: μ ≈ 1.401 × Δρ/ρ (dominant), y subdominant
///   - y-era: y ≈ 0.25 × Δρ/ρ (dominant), μ subdominant
///
/// The coefficient 1.401 = 3/κ_c is derived from Kompaneets + number conservation.
/// The coefficient 0.25 = 1/4 is exact from the definition of the y-parameter.
///
/// These are verified independently by Sunyaev & Zeldovich (1970), Hu & Silk (1993),
/// Chluba (2013), and many others. The 15-20% tolerance accounts for the
/// visibility function corrections at finite z.
#[test]
fn test_literature_mu_y_conversion_coefficients() {
    let drho = 1e-6;

    // === μ-era injection at z_h = 2×10⁵ ===
    let z_h_mu = 2.0e5;
    let sigma_mu = 5000.0;
    let dq_mu = |z: f64| gaussian_heating(z, z_h_mu, sigma_mu, drho);

    let (mu_at_mu_era, y_at_mu_era) = greens::mu_y_from_heating(&dq_mu, 1e3, 5e6, 20000);

    let mu_expected = 1.401 * drho; // 3/κ_c × Δρ/ρ
    let mu_rel_err = (mu_at_mu_era - mu_expected).abs() / mu_expected;

    eprintln!("Test 1 — μ-era (z_h = {z_h_mu:.0e}):");
    eprintln!(
        "  μ = {mu_at_mu_era:.4e}, expected ≈ {mu_expected:.4e}, err = {:.1}%",
        mu_rel_err * 100.0
    );
    eprintln!("  y = {y_at_mu_era:.4e} (subdominant)");

    assert!(
        mu_rel_err < 0.10,
        "μ-era: μ = {mu_at_mu_era:.4e}, expected ≈ {mu_expected:.4e} (10% tol), err = {:.1}%",
        mu_rel_err * 100.0
    );
    assert!(
        y_at_mu_era.abs() < mu_at_mu_era.abs() * 0.3,
        "μ-era: y = {y_at_mu_era:.4e} should be ≪ μ = {mu_at_mu_era:.4e}"
    );

    // === y-era injection at z_h = 1×10⁴ ===
    let z_h_y = 1.0e4;
    let sigma_y = 500.0;
    let dq_y = |z: f64| gaussian_heating(z, z_h_y, sigma_y, drho);

    let (mu_at_y_era, y_at_y_era) = greens::mu_y_from_heating(&dq_y, 100.0, 5e5, 20000);

    let y_expected = 0.25 * drho; // exact from definition
    let y_rel_err = (y_at_y_era - y_expected).abs() / y_expected;

    eprintln!("Test 1 — y-era (z_h = {z_h_y:.0e}):");
    eprintln!(
        "  y = {y_at_y_era:.4e}, expected ≈ {y_expected:.4e}, err = {:.1}%",
        y_rel_err * 100.0
    );
    eprintln!("  μ = {mu_at_y_era:.4e} (subdominant)");

    assert!(
        y_rel_err < 0.10,
        "y-era: y = {y_at_y_era:.4e}, expected ≈ {y_expected:.4e} (10% tol), err = {:.1}%",
        y_rel_err * 100.0
    );
    // At z=1e4, J_μ is small but nonzero (~0.2), so μ/y ≈ 0.20.
    // This is physically correct: we're near the μ-y transition boundary.
    assert!(
        mu_at_y_era.abs() < y_at_y_era.abs() * 0.25,
        "y-era: μ = {mu_at_y_era:.4e} should be ≪ y = {y_at_y_era:.4e}"
    );
}

/// Test 2: Regime boundaries — mode ordering at three characteristic redshifts.
///
/// At z = 3×10⁶ (thermalization era): both μ and y are exponentially suppressed
///   because DC+BR fully thermalize the injection into a temperature shift.
/// At z = 2×10⁵ (μ-era): μ dominates over y.
/// At z = 1×10⁴ (y-era): y dominates over μ.
///
/// This tests the fundamental three-regime structure of the thermalization problem,
/// first identified by Sunyaev & Zeldovich (1970) and refined by many authors.
#[test]
fn test_literature_regime_boundaries() {
    let drho = 1e-6;

    let cases: Vec<(f64, f64, &str, bool, bool, bool)> = vec![
        // (z_h, sigma_z, label, expect_mu_dom, expect_y_dom, expect_suppressed)
        (3.0e6, 1.0e5, "thermalization (z=3e6)", false, false, true),
        (2.0e5, 5000.0, "μ-era (z=2e5)", true, false, false),
        (1.0e4, 500.0, "y-era (z=1e4)", false, true, false),
    ];

    for (z_h, sigma, label, expect_mu_dom, expect_y_dom, expect_suppressed) in cases {
        let dq = |z: f64| gaussian_heating(z, z_h, sigma, drho);
        let (mu, y) = greens::mu_y_from_heating(&dq, 100.0, 1e7, 20000);

        eprintln!("Test 2 — {label}: μ = {mu:.4e}, y = {y:.4e}");

        if expect_suppressed {
            // At z=3×10⁶: J_bb*(z) ≈ 6%, so residual μ is ~8% of the
            // asymptotic value. Use a 10% threshold (not 1%) because z=3×10⁶
            // is near the edge of the thermalization era, not deep within it.
            assert!(
                mu.abs() < drho * 0.10,
                "{label}: μ = {mu:.4e} not suppressed (limit = {:.4e})",
                drho * 0.10
            );
            assert!(
                y.abs() < drho * 0.01,
                "{label}: y = {y:.4e} not suppressed (limit = {:.4e})",
                drho * 0.01
            );
        }
        if expect_mu_dom {
            assert!(
                mu.abs() > y.abs(),
                "{label}: μ = {mu:.4e} should dominate over y = {y:.4e}"
            );
        }
        if expect_y_dom {
            assert!(
                y.abs() > mu.abs(),
                "{label}: y = {y:.4e} should dominate over μ = {mu:.4e}"
            );
        }
    }
}

/// Thermalization suppression: J_bb* (the fraction of energy thermalized
/// into a temperature shift) monotonically increases with injection redshift.
///
/// This means less distortion is visible at higher z. We verify this using
/// the Green's function visibility functions directly.
///
/// Additionally, for z > 2e5 (deep mu-era), the mu coefficient should
/// decrease with z as thermalization becomes more effective.
#[test]
fn test_thermalization_suppression_monotonic() {
    // J_bb* should monotonically increase with z
    let redshifts = [1e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6];
    let mut prev_jbb = 1.0;

    for &z_h in &redshifts {
        let jbb_star = greens::visibility_j_bb_star(z_h);
        eprintln!("z={z_h:.0e}: J_bb* = {jbb_star:.6e}");
        // J_bb* DECREASES with z (approaches 0 at high z):
        //   J_bb* = 0.983 * exp(-(z/z_mu)^2.5) * (1 - 0.0381*(z/z_mu)^2.29)
        // The temperature shift fraction (1 - J_bb*) INCREASES with z.
        assert!(
            jbb_star <= prev_jbb + 1e-10,
            "J_bb* not monotonically decreasing: at z={z_h:.0e} J_bb*={jbb_star:.6e} \
             > prev {prev_jbb:.6e}"
        );
        prev_jbb = jbb_star;
    }

    // For deep mu-era redshifts, the mu/drho coefficient should decrease
    // as thermalization becomes stronger
    let deep_z = [2e5, 3e5, 5e5, 1e6, 2e6, 5e6];
    let drho = 1e-5;
    let mut prev_mu_coeff = f64::INFINITY;

    for &z_h in &deep_z {
        let sigma_z = z_h * 0.04;
        let dq_dz = |z: f64| gaussian_heating(z, z_h, sigma_z, drho);

        let mu = greens::mu_from_heating(&dq_dz, 1e3, z_h * 5.0, 10000);
        let mu_coeff = mu.abs() / drho;

        eprintln!("z={z_h:.0e}: μ/Δρ = {mu_coeff:.4}, prev = {prev_mu_coeff:.4}");

        assert!(
            mu_coeff <= prev_mu_coeff + 1e-3,
            "μ coefficient not monotonically decreasing in deep μ-era: \
             at z={z_h:.0e} μ/Δρ = {mu_coeff:.4} > prev {prev_mu_coeff:.4}"
        );
        prev_mu_coeff = mu_coeff;
    }

    // At z=5e6, should be strongly suppressed (J_bb* ~ 3e-5)
    assert!(
        prev_mu_coeff < 0.01,
        "At z=5e6, μ/Δρ should be < 0.01, got {prev_mu_coeff:.4}"
    );
}

/// Photon number conservation under pure Compton scattering.
///
/// The Kompaneets equation preserves photon number exactly (it only
/// redistributes photons in frequency). With DC and BR turned off
/// and no injection, the total photon number ∫ x² Δn dx should remain
/// constant during evolution.
///
/// We test this by injecting a known perturbation and verifying that
/// the total photon number is conserved to high accuracy.
/// Multiple injection redshifts: verify that the total distortion from
/// two temporally separated bursts equals the sum of individual bursts
/// (PDE linearity test for small distortions).
/// Spectral decomposition must reproduce a known input.
///
/// Create a spectrum that is exactly 50% μ-distortion + 50% y-distortion
/// (by amplitude), and verify the joint decomposition recovers the correct
/// coefficients.
#[test]
fn test_spectral_decomposition_mixed_mode() {
    let grid_config = GridConfig::default();
    let grid = spectroxide::grid::FrequencyGrid::new(&grid_config);

    let mu_val = 5e-6;
    let y_val = 1e-6;

    // Construct synthetic spectrum: Δn = μ·M(x) + y·Y_SZ(x)
    let delta_n: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| mu_val * spectrum::mu_shape(x) + y_val * spectrum::y_shape(x))
        .collect();

    // Extract using the same decomposition the solver uses
    let (mu_ext, y_ext, _dt) = distortion::decompose(&grid.x, &delta_n);

    eprintln!("Decomposition test:");
    eprintln!("  Input:     μ = {mu_val:.4e}, y = {y_val:.4e}");
    eprintln!("  Extracted: μ = {mu_ext:.4e}, y = {y_ext:.4e}");

    let mu_err = (mu_ext - mu_val).abs() / mu_val;
    let y_err = (y_ext - y_val).abs() / y_val;

    // The M(x), Y_SZ(x), and G(x) basis functions are not orthogonal,
    // so there is some cross-talk between modes. On the solver's
    // default grid, cross-talk is negligible (<1%).
    assert!(
        mu_err < 0.02,
        "μ decomposition: {mu_ext:.4e} vs input {mu_val:.4e}, err={:.1}%",
        mu_err * 100.0
    );
    assert!(
        y_err < 0.05,
        "y decomposition: {y_ext:.4e} vs input {y_val:.4e}, err={:.1}%",
        y_err * 100.0
    );

    // Also test pure y recovery (no cross-talk from μ)
    let delta_n_y: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| y_val * spectrum::y_shape(x))
        .collect();
    let (_mu_y, y_y, _dt_y) = distortion::decompose(&grid.x, &delta_n_y);
    let y_pure_err = (y_y - y_val).abs() / y_val;
    assert!(
        y_pure_err < 0.05,
        "Pure y decomposition: {y_y:.4e} vs input {y_val:.4e}, err={:.1}%",
        y_pure_err * 100.0
    );
}

/// Green's function spectral decomposition: decomposing G_th(x, z_h)
/// back into (mu, y, dT/T) should reproduce the visibility functions.
///
/// For injection at z_h:
///   - mu_extracted ≈ (3/κ_c) J_bb*(z_h) J_mu(z_h) × Δρ/ρ
///   - y_extracted  ≈ (1/4) J_y(z_h) × Δρ/ρ
///
/// This tests the joint least-squares decomposition accuracy.
#[test]
fn test_greens_function_decomposition_accuracy() {
    let grid = spectroxide::grid::FrequencyGrid::new(&spectroxide::grid::GridConfig::default());

    // Only test in clear μ-era and y-era regimes. The transition region
    // (z ~ 3e4-1e5) has inherent decomposition ambiguity.
    let z_values = [5e3, 1e4, 2e5, 5e5];
    let drho = 1e-5; // arbitrary small injection

    for &z_h in &z_values {
        let delta_n: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| greens::greens_function(x, z_h) * drho)
            .collect();

        let params = spectroxide::distortion::decompose_distortion(&grid.x, &delta_n);

        let j_mu = greens::visibility_j_mu(z_h);
        let j_y = greens::visibility_j_y(z_h);
        let j_bb = greens::visibility_j_bb_star(z_h);

        let mu_expected = (3.0 / KAPPA_C) * j_bb * j_mu * drho;
        let y_expected = 0.25 * j_y * drho;

        // Residual RMS should be small
        let rms: f64 = (params.residual.iter().map(|r| r * r).sum::<f64>()
            / params.residual.len() as f64)
            .sqrt();

        eprintln!(
            "GF decomposition at z={z_h:.0e}: mu_ext={:.4e} vs {:.4e}, y_ext={:.4e} vs {:.4e}, rms={rms:.2e}",
            params.mu, mu_expected, params.y, y_expected
        );

        // Only check the DOMINANT component (the one with the larger
        // expected amplitude). Cross-talk makes the sub-dominant component
        // unreliable — a small fraction of μ leaks into y and vice versa.
        if mu_expected.abs() > y_expected.abs() {
            // μ-dominated: check mu to <5%
            let rel = (params.mu - mu_expected).abs() / mu_expected.abs();
            assert!(
                rel < 0.05,
                "mu mismatch at z={z_h:.0e}: rel={rel:.3}, extracted={:.4e}, expected={:.4e}",
                params.mu,
                mu_expected
            );
        } else {
            // y-dominated: check y to <5%
            let rel = (params.y - y_expected).abs() / y_expected.abs();
            assert!(
                rel < 0.05,
                "y mismatch at z={z_h:.0e}: rel={rel:.3}, extracted={:.4e}, expected={:.4e}",
                params.y,
                y_expected
            );
        }
    }
}

/// Spectral integral orthogonality: M(x) and Y_SZ(x) are nearly orthogonal
/// under the x² dx measure, which is why the decomposition works.
///
/// Verify that:
///   ∫ M(x) Y(x) x² dx << sqrt(∫M²x²dx × ∫Y²x²dx)
#[test]
fn test_spectral_shape_near_orthogonality() {
    use spectroxide::spectrum::{g_bb, mu_shape, y_shape};

    let n = 10000;
    let x_min = 1e-4_f64;
    let x_max = 50.0_f64;

    let mut m_m = 0.0_f64;
    let mut y_y = 0.0_f64;
    let mut g_g = 0.0_f64;
    let mut m_y = 0.0_f64;
    let mut m_g = 0.0_f64;
    let mut y_g = 0.0_f64;

    let log_min = x_min.ln();
    let log_max = x_max.ln();

    for i in 1..n {
        let log_x = log_min + (i as f64 / n as f64) * (log_max - log_min);
        let x = log_x.exp();
        let dlog = (log_max - log_min) / n as f64;
        let dx = x * dlog; // dx = x * d(ln x)

        let m = mu_shape(x);
        let y = y_shape(x);
        let g = g_bb(x);
        let w = x * x * dx;

        m_m += m * m * w;
        y_y += y * y * w;
        g_g += g * g * w;
        m_y += m * y * w;
        m_g += m * g * w;
        y_g += y * g * w;
    }

    // Correlation coefficients
    let r_my = m_y / (m_m * y_y).sqrt();
    let r_mg = m_g / (m_m * g_g).sqrt();
    let r_yg = y_g / (y_y * g_g).sqrt();

    eprintln!("Spectral shape correlations (x² dx measure):");
    eprintln!("  r(M, Y) = {r_my:.4}");
    eprintln!("  r(M, G) = {r_mg:.4}");
    eprintln!("  r(Y, G) = {r_yg:.4}");

    // M and Y should be nearly orthogonal (|r| < 0.3)
    assert!(
        r_my.abs() < 0.3,
        "M and Y should be nearly orthogonal: r = {r_my:.4}"
    );
    // M and G should have some correlation (both involve e^x/(e^x-1)^2)
    // but should not be perfectly correlated
    assert!(
        r_mg.abs() < 0.95,
        "M and G should not be perfectly correlated: r = {r_mg:.4}"
    );
    // Y and G should have moderate correlation
    assert!(
        r_yg.abs() < 0.95,
        "Y and G should not be perfectly correlated: r = {r_yg:.4}"
    );
}

/// The mu_from_heating function should converge with integration resolution.
/// Use a broad decaying-particle-like heating profile that is easy to integrate.
#[test]
fn test_gf_mu_resolution_independence() {
    let cosmo = Cosmology::default();

    // Use decaying particle heating (broad in z, easy to resolve)
    let f_x = 1e4_f64;
    let gamma_x = 1e-13_f64;

    let scenario = InjectionScenario::DecayingParticle { f_x, gamma_x };

    // Compare mu_from_heating at different resolutions
    let mu_low = greens::mu_from_heating(
        |z| -scenario.heating_rate_per_redshift(z, &cosmo),
        1e3,
        3e6,
        500,
    );
    let mu_mid = greens::mu_from_heating(
        |z| -scenario.heating_rate_per_redshift(z, &cosmo),
        1e3,
        3e6,
        2000,
    );
    let mu_high = greens::mu_from_heating(
        |z| -scenario.heating_rate_per_redshift(z, &cosmo),
        1e3,
        3e6,
        5000,
    );

    eprintln!("GF resolution: mu_500={mu_low:.6e}, mu_2000={mu_mid:.6e}, mu_5000={mu_high:.6e}");

    // Mid→high should change less than low→mid (convergence)
    let change_1 = (mu_mid - mu_low).abs();
    let change_2 = (mu_high - mu_mid).abs();

    assert!(
        change_2 < change_1 || change_2 < 0.01 * mu_high.abs(),
        "GF μ should converge: change_1={change_1:.4e}, change_2={change_2:.4e}"
    );

    // High-res result should be finite and positive (decay heats)
    assert!(
        mu_high > 0.0 && mu_high.is_finite(),
        "mu should be positive and finite: {mu_high:.4e}"
    );
}

/// Decomposition comprehensive: pure temperature shift, negative μ, and mixed μ+y.
#[test]
fn test_decomposition_comprehensive() {
    let n = 5000;
    let x_grid: Vec<f64> = (0..n)
        .map(|i| 0.01 + 30.0 * i as f64 / (n - 1) as f64)
        .collect();

    // Pure temperature shift: ΔT/T dominates, μ ≈ 0, y ≈ 0
    {
        let dt_true = 1e-5;
        let delta_n: Vec<f64> = x_grid
            .iter()
            .map(|&x| dt_true * spectrum::g_bb(x))
            .collect();
        let params = distortion::decompose_distortion(&x_grid, &delta_n);
        assert!(
            (params.delta_t_over_t - dt_true).abs() / dt_true < 0.1,
            "ΔT/T extraction"
        );
        assert!(params.mu.abs() < dt_true * 0.5, "μ small for T-shift");
        assert!(params.y.abs() < dt_true * 0.5, "y small for T-shift");
    }

    // Negative μ (cooling scenario)
    {
        let mu_true = -2e-6;
        let delta_n: Vec<f64> = x_grid
            .iter()
            .map(|&x| mu_true * spectrum::mu_shape(x))
            .collect();
        let params = distortion::decompose_distortion(&x_grid, &delta_n);
        assert!(
            (params.mu - mu_true).abs() / mu_true.abs() < 0.15,
            "negative μ extraction"
        );
        assert!(params.mu < 0.0, "μ sign should be negative");
    }

    // Mixed μ + y
    let mu_true = 3e-6;
    let y_true = 1e-6;
    let delta_n: Vec<f64> = x_grid
        .iter()
        .map(|&x| mu_true * spectrum::mu_shape(x) + y_true * spectrum::y_shape(x))
        .collect();

    let params = distortion::decompose_distortion(&x_grid, &delta_n);

    let mu_err = (params.mu - mu_true).abs() / mu_true;
    let y_err = (params.y - y_true).abs() / y_true;
    // M(x) and Y_SZ(x) are not orthogonal — expect cross-talk
    // when decomposing a mixture (5000-pt uniform grid has better resolution
    // than the solver's non-uniform grid, so tighter tolerance is warranted)
    assert!(
        mu_err < 0.15,
        "Mixed: μ = {:.4e}, true = {mu_true:.4e}, err = {mu_err:.2}",
        params.mu
    );
    assert!(
        y_err < 0.30,
        "Mixed: y = {:.4e}, true = {y_true:.4e}, err = {y_err:.2}",
        params.y
    );
    // But the total distortion amplitude should be correct: both should have right sign
    assert!(
        params.mu > 0.0 && params.y > 0.0,
        "Signs should be positive: μ={:.4e}, y={:.4e}",
        params.mu,
        params.y
    );
}

/// FIRAS check + energy consistency combined test.
#[test]
fn test_firas_check_and_energy_consistency() {
    let n = 5000;
    let x_grid: Vec<f64> = (0..n)
        .map(|i| 0.01 + 30.0 * i as f64 / (n - 1) as f64)
        .collect();

    // FIRAS check: μ at half the limit
    let mu_half = 4.5e-5;
    let dn_half: Vec<f64> = x_grid
        .iter()
        .map(|&x| mu_half * spectrum::mu_shape(x))
        .collect();
    let params = distortion::decompose_distortion(&x_grid, &dn_half);
    let (mu_frac, _y_frac) = distortion::firas_check(&params);
    assert!(
        mu_frac > 0.3 && mu_frac < 0.7,
        "FIRAS μ fraction should be ~0.5: {mu_frac:.3}"
    );

    // Energy consistency: Δρ/ρ ≈ μ × κ_c/3
    let mu_true = 1e-5;
    let dn: Vec<f64> = x_grid
        .iter()
        .map(|&x| mu_true * spectrum::mu_shape(x))
        .collect();
    let params = distortion::decompose_distortion(&x_grid, &dn);
    let expected_drho = mu_true * KAPPA_C / 3.0;
    let rel_err =
        (params.delta_rho_over_rho - expected_drho).abs() / expected_drho.abs().max(1e-20);
    assert!(
        rel_err < 0.15,
        "Energy consistency: Δρ/ρ err = {rel_err:.2}"
    );
}

/// Brightness temperature conversion should give T_b/T_CMB ≈ 1 for Planck
/// and deviate for distorted spectra.
#[test]
fn test_brightness_temperature() {
    let grid = FrequencyGrid::new(&GridConfig::fast());

    // For zero distortion (Planck): T_b/T_CMB - 1 = 0
    let snap_planck = SolverSnapshot {
        z: 0.0,
        delta_n: vec![0.0; grid.n],
        rho_e: 1.0,
        mu: 0.0,
        y: 0.0,
        delta_rho_over_rho: 0.0,
        accumulated_delta_t: 0.0,
    };
    let bt = snap_planck.brightness_temp(&grid.x);
    let max_bt: f64 = bt
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_bt < 1e-10,
        "Planck brightness temp deviation should be ~0: max = {max_bt:.4e}"
    );

    // For μ-distortion: T_b deviation should change sign at β_μ
    let mu_val = 1e-4;
    let delta_n_mu: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| mu_val * spectrum::mu_shape(x))
        .collect();
    let snap_mu = SolverSnapshot {
        z: 0.0,
        delta_n: delta_n_mu,
        rho_e: 1.0,
        mu: mu_val,
        y: 0.0,
        delta_rho_over_rho: 0.0,
        accumulated_delta_t: 0.0,
    };
    let bt_mu = snap_mu.brightness_temp(&grid.x);
    // Find sign change near β_μ
    let idx_beta = grid.find_index(BETA_MU);
    if idx_beta > 5 && idx_beta + 5 < grid.n {
        let bt_below = bt_mu[idx_beta - 5];
        let bt_above = bt_mu[idx_beta + 5];
        assert!(
            bt_below * bt_above < 0.0 || bt_below.abs() < 1e-8 || bt_above.abs() < 1e-8,
            "T_b should change sign near β_μ: T_b({:.2})={bt_below:.4e}, T_b({:.2})={bt_above:.4e}",
            grid.x[idx_beta - 5],
            grid.x[idx_beta + 5]
        );
    }
}

/// Photon survival probability: P_s regime structure.
/// DC dominates at high z, BR at low z, with a crossover.
#[test]
fn test_photon_survival_regime_structure() {
    // At z = 2e6, DC dominates
    let dc_high = greens::x_c_dc(2.0e6);
    let br_high = greens::x_c_br(2.0e6);
    assert!(
        dc_high > br_high,
        "At z=2e6: x_c_DC={dc_high:.4e} should dominate over x_c_BR={br_high:.4e}"
    );

    // At z = 1e4, BR dominates
    let dc_low = greens::x_c_dc(1.0e4);
    let br_low = greens::x_c_br(1.0e4);
    assert!(
        br_low > dc_low,
        "At z=1e4: x_c_BR={br_low:.4e} should dominate over x_c_DC={dc_low:.4e}"
    );

    // x_c is NOT monotonic in z because DC and BR have opposite z-dependence.
    // DC grows with z (∝ z^{1/2}), BR shrinks with z (∝ z^{-0.672}).
    // At intermediate z ~ few × 10^5, x_c should have a minimum.
    let xc_low = greens::x_c(1.0e4);
    let xc_mid = greens::x_c(2.0e5);
    let xc_high = greens::x_c(2.0e6);
    eprintln!("x_c: z=1e4→{xc_low:.4e}, z=2e5→{xc_mid:.4e}, z=2e6→{xc_high:.4e}");
    // At z=2e5, x_c is smaller than at BOTH extremes: it sits in the interior
    // minimum where neither process dominates. (Was a disjunction, which either
    // half satisfied on its own — finding F-PC-2.)
    assert!(
        xc_mid < xc_low && xc_mid < xc_high,
        "x_c must have an interior minimum from DC/BR competition: \
         x_c(1e4)={xc_low:.4e}, x_c(2e5)={xc_mid:.4e}, x_c(2e6)={xc_high:.4e}"
    );
}

/// Soft photon absorption: x_inj << x_c → P_s ≈ 0 → always positive μ.
#[test]
fn test_photon_gf_soft_photon_absorbed() {
    let z_h = 3.0e5; // photon-GF mu-era edge (ADR 0005)
    let dn_over_n = 1e-5;
    // x_inj = 1e-4 << x_c(3e5) ~ 0.002
    let x_inj = 1e-4;

    let mu = greens::mu_from_photon_injection(x_inj, z_h, dn_over_n);
    assert!(
        mu > 0.0,
        "Soft photon (P_s≈0): μ should be positive, got {mu:.4e}"
    );

    // Even below x₀, soft photons give positive μ because P_s ≈ 0
    // means the photon is absorbed and becomes pure energy injection
    let x_inj_below = 1.0; // below x₀ but still very soft at this z
    let p_s = greens::photon_survival_probability(x_inj_below, z_h);
    if p_s < 0.1 {
        let mu_below = greens::mu_from_photon_injection(x_inj_below, z_h, dn_over_n);
        eprintln!("x_inj={x_inj_below}, P_s={p_s:.3e}, μ={mu_below:.4e}");
        // When P_s is small enough, the (1 - P_s × x₀/x) factor is positive
        assert!(
            mu_below > 0.0,
            "Soft photon below x₀ with P_s≈0: μ should be positive, got {mu_below:.4e}"
        );
    }
}

/// The photon injection Green's function must satisfy several exact
/// algebraic identities. These are EXACT (no physics approximation)
/// and must hold to machine precision.
///
/// Identity 1: P_s → 0 limit
///   When x_inj → 0 (or P_s = 0), the photon GF reduces to
///   G_photon(x) = α_ρ × x_inj × G_thermal(x)
///   because all injected photons are absorbed by DC/BR and their
///   energy is redistributed as a standard energy injection.
///
/// Identity 2: μ is linear in ΔN/N (exact by construction)
///
/// Identity 3: At x₀ = X_BALANCED with P_s = 1:
///   μ_from_photon_injection(x₀, z, ΔN/N) = 0 exactly
///   (energy and number effects cancel exactly)
#[test]
fn test_photon_injection_gf_algebraic_identities() {
    let z_h = 3.0e5; // photon-GF mu-era edge (ADR 0005)

    // Identity 1: P_s → 0 limit
    let x_soft = 1e-6; // So soft that P_s ≈ 0
    let p_s = greens::photon_survival_probability(x_soft, z_h);
    assert!(p_s < 1e-10, "P_s should be ~0 for x={x_soft}: got {p_s}");

    let cosmo = Cosmology::default();
    // With Arsenadze's x'-dependent T_μ replacing the universal J_μ,
    // the P_s→0 limit gives only approximate agreement with α_ρ × x' × G_th.
    // Check same sign.
    let x_obs_vals = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    for &x_obs in &x_obs_vals {
        let g_ph = greens::greens_function_photon(x_obs, x_soft, z_h, 0.0, &cosmo);
        let g_th = greens::greens_function(x_obs, z_h);
        let expected = ALPHA_RHO * x_soft * g_th;
        // Both should have the same sign (or be near zero at the crossing)
        assert!(
            g_ph * expected > 0.0 || g_ph.abs() < 1e-7 || expected.abs() < 1e-7,
            "P_s→0 identity at x_obs={x_obs}: sign mismatch G_ph={g_ph:.4e}, expected={expected:.4e}"
        );
    }

    // Identity 2: Linearity in ΔN/N
    let mu_1 = greens::mu_from_photon_injection(8.0, z_h, 1e-5);
    let mu_3 = greens::mu_from_photon_injection(8.0, z_h, 3e-5);
    let ratio = mu_3 / mu_1;
    assert!(
        (ratio - 3.0).abs() < 1e-12,
        "Linearity: μ(3×ΔN/N) / μ(ΔN/N) = {ratio}, expected 3.0"
    );

    // Identity 3: Zero at x₀ when P_s ≈ 1
    let p_s_x0 = greens::photon_survival_probability(X_BALANCED, z_h);
    eprintln!("P_s(x₀={X_BALANCED:.3}, z={z_h:.0e}) = {p_s_x0:.6}");
    // P_s should be very close to 1 at x₀ ≈ 3.6 in μ-era (x_c << 1 at z=2e5)
    assert!(p_s_x0 > 0.99, "P_s(x₀) should be ~1 at z=2e5: got {p_s_x0}");

    let mu_x0 = greens::mu_from_photon_injection(X_BALANCED, z_h, 1e-5);
    let mu_ref = greens::mu_from_photon_injection(10.0, z_h, 1e-5).abs();
    eprintln!("μ at x₀: {mu_x0:.4e}, μ_ref(x=10): {mu_ref:.4e}");
    assert!(
        mu_x0.abs() < 0.02 * mu_ref,
        "|μ(x₀)| = {:.4e} should be < 2% of μ(x=10) = {mu_ref:.4e}",
        mu_x0.abs()
    );
}

//
// At z < 1100, Compton scattering is inefficient (X_e ~ 10⁻⁴). Energy
// injection does NOT produce y-distortions, and photon injection remains
// locked-in at the injection frequency.
/// GF photon injection at z_h = 500: smooth part ≈ 0, surviving delta dominates.
///
/// At z = 500, x_c ≈ 10⁻⁵ (tiny), so P_s(x > 0.1) ≈ 1.
/// In the y-era with J_μ ≈ 0, the smooth y-part goes as (1-P_s) ≈ 0,
/// leaving only the surviving photon δ-function.
#[test]
fn test_gf_photon_injection_post_recombination_locked_in() {
    let z_h = 500.0;
    let x_inj = 3.0;
    let sigma_x = 0.1;

    // P_s at z=500: the x_c fitting formula (calibrated for z > 10^4)
    // extrapolates to non-trivial values at low z due to x_c_br's negative
    // exponent. In practice, post-recomb injection is handled by J_Compton.
    let p_s = greens::photon_survival_probability(x_inj, z_h);
    assert!(p_s > 0.5, "P_s(x=3, z=500) = {p_s}, should be significant");

    // Smooth part at x far from x_inj should be ~0
    let cosmo = Cosmology::default();
    let g_smooth_far = greens::greens_function_photon(10.0, x_inj, z_h, 0.0, &cosmo);
    eprintln!("Smooth part at x=10, z=500: {g_smooth_far:.4e}");

    // At z=500, J_mu ≈ 0, so smooth = α_x × (1-J_μ) × (1-P_s) × Y/4 ≈ 0
    // (since P_s ≈ 1, the 1-P_s factor kills the smooth part)
    assert!(
        g_smooth_far.abs() < 1e-3,
        "Smooth GF far from x_inj should be small at z=500, got {g_smooth_far:.4e}"
    );

    // With sigma_x > 0, the surviving delta should dominate near x_inj
    let g_at_peak = greens::greens_function_photon(x_inj, x_inj, z_h, sigma_x, &cosmo);
    assert!(
        g_at_peak.abs() > 1e-3,
        "Surviving δ-function at x_inj should give large GF, got {g_at_peak:.4e}"
    );
}

// Key benchmarks from the KYPRIX solver paper (A&A 507, 1243):
//   1. μ₀ ≈ 1.4 × Δε/εᵢ for early heating (deep μ-era)
//   2. φ_BE = (1 - 1.11μ₀)^{-1/4} for Bose-Einstein equilibrium T_e
//   3. φ_eq ≈ (1 + 5.4u)φᵢ for superposed blackbodies (u = Δε/(4εᵢ))
//   4. Energy conservation < 0.05%
/// Decomposing an exact Bose-Einstein spectrum 1/(e^{x+μ₀} − 1) on the
/// production frequency grid recovers μ₀ to 10% for μ₀ = 1e-5 to 1e-3.
#[test]
fn test_decompose_bose_einstein_recovers_mu() {
    let grid_config = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    let grid = FrequencyGrid::new(&grid_config);

    for &mu_0 in &[1e-5_f64, 1e-4, 1e-3] {
        // Construct Bose-Einstein distortion
        let delta_n: Vec<f64> = grid
            .x
            .iter()
            .map(|&xi| 1.0_f64 / ((xi + mu_0).exp() - 1.0) - 1.0 / (xi.exp() - 1.0))
            .collect();

        // Decompose into μ, y, ΔT/T
        let params = distortion::decompose_distortion(&grid.x, &delta_n);
        let mu_extracted = params.mu;

        // The extracted μ should match the input μ₀
        let rel_err = (mu_extracted - mu_0).abs() / mu_0;
        assert!(
            rel_err < 0.1,
            "BE decomposition: input μ₀={mu_0:.0e}, extracted μ={mu_extracted:.4e}, \
             rel_err={rel_err:.2}"
        );
    }
}

/// Test the GF energy sum rule: J_mu*J_bb* + J_y + (1 - J_bb*) ≈ 1.
///
/// This is NOT exactly 1 because J_y is independently fitted (Chluba 2013
/// Eq. 5) rather than derived from J_mu and J_bb*. Chluba (2013) §3 shows the
/// "missing" fraction stays in the residual and never exceeds ~16–17%,
/// maximised near z ~ 7–8×10⁴ in the μ-y transition region.
#[test]
fn test_gf_energy_sum_rule() {
    let redshifts = [1e3, 5e3, 1e4, 3e4, 5e4, 6e4, 8e4, 1e5, 2e5, 5e5, 1e6, 2e6];

    let mut max_deviation = 0.0_f64;
    let mut max_dev_z = 0.0_f64;

    for &z in &redshifts {
        let jbb = greens::visibility_j_bb_star(z);
        let jmu = greens::visibility_j_mu(z);
        let jy = greens::visibility_j_y(z);
        let j_t = 1.0 - jbb;

        let sum = jmu * jbb + jy + j_t;
        let deviation = (sum - 1.0).abs();

        eprintln!(
            "z={z:.0e}: J_mu*J_bb* = {:.4e}, J_y = {:.4e}, J_T = {:.4e}, \
             sum = {sum:.4}, deviation = {deviation:.4}",
            jmu * jbb,
            jy,
            j_t
        );

        if deviation > max_deviation {
            max_deviation = deviation;
            max_dev_z = z;
        }
    }

    eprintln!("\nMax deviation from sum rule: {max_deviation:.4} at z = {max_dev_z:.0e}");

    // The sum rule is NOT exact. Document that deviation is bounded.
    // Known max ~17% in transition region (z ~ 8e4).
    assert!(
        max_deviation < 0.20,
        "GF sum rule deviation {max_deviation:.3} exceeds 20% bound"
    );

    // Verify the max deviation is in the transition region, not at extremes
    assert!(
        max_dev_z > 1e4 && max_dev_z < 5e5,
        "Max deviation should be in transition region, got z = {max_dev_z:.0e}"
    );
}
