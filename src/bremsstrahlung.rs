//! Bremsstrahlung (free-free) emission and absorption.
//!
//! Photon-number-changing process: e + ion → e + ion + γ.
//! Dominates over double Compton at lower redshifts (z < few × 10⁵).
//!
//! The BR emission coefficient:
//!   K_BR(x, θ_e) = (α λ_e³ / (2π√(6π))) θ_e^{-7/2} e^{-xφ}/φ³ Σ_i Z_i² N_i g_ff(Z_i, x, θ_e)
//!
//! In the code, we express this as a rate per Thomson scattering time τ:
//!   K_BR(x) = prefactor × Σ_i Z_i² (N_i/N_e) g_ff(Z_i, x, θ_e)
//!
//! References:
//! - Karzas & Latter (1961) — original Gaunt factors
//! - Itoh et al. (2000) — relativistic thermal BR fits
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 14]
//! - Chluba, Ravenni & Bolliet (2020), MNRAS 492, 177 (BRpack)

use crate::constants::*;
use crate::spectrum::planck;

/// Precomputed constant: α λ_e³ / (2π √(6π))
/// This is the BR coefficient prefactor that is independent of x, θ_e, and species.
const BR_PREFACTOR: f64 = ALPHA_FS * LAMBDA_ELECTRON * LAMBDA_ELECTRON * LAMBDA_ELECTRON
    / (2.0 * std::f64::consts::PI * 4.341_607_527_349_605_5);
// 4.341607527... = sqrt(6π). Hardcoded since sqrt is not const fn.

/// Precomputed √3/π for Gaunt factor formula.
const S3_PI: f64 = 0.5513_2889_5421_7921;

/// Precomputed ln(2.25) for Z=1 Gaunt factor: ln(2.25/x) = LN_2_25 - ln(x)
const LN_2_25: f64 = 0.8109_3021_6216_3288;

/// Precomputed ln(2.25/2) = ln(1.125) for Z=2 Gaunt factor
const LN_1_125: f64 = 0.1177_8303_5656_3834;

/// Non-relativistic thermally-averaged free-free Gaunt factor.
///
/// Uses a softplus interpolation that smoothly approaches the classical Born
/// limit at low frequencies while remaining ≥ 1 at high frequencies:
///   g_ff = 1 + softplus((√3/π)(ln(2.25/(x Z)) + 0.5 ln(θ_e)) + 1.425)
///
/// The argument uses Z (nuclear charge, linear) not Z², following the
/// Coulomb parameter convention where η_Z = Z e²/(ℏv) enters linearly
/// inside the logarithm. The 0.5×ln(θ_e) is an empirical interpolation
/// between the Born and classical regimes.
///
/// This approximation is from CosmoTherm (private communication, J. Chluba),
/// calibrated against the exact Karzas & Latter (1961) Gaunt factor tabulations
/// and the BRpack library (Chluba, Ravenni & Bolliet 2020, MNRAS 492, 177).
/// The offset constant 1.425 improves agreement in the transition region.
pub fn gaunt_ff_nr(x: f64, theta_e: f64, z_charge: f64) -> f64 {
    if theta_e < 1e-30 {
        return 1.0;
    }
    gaunt_ff_nr_fast(x, z_charge, 0.5 * theta_e.ln())
}

/// Fast Gaunt factor with precomputed 0.5*ln(θ_e) hoisted out of grid loop.
#[inline]
fn gaunt_ff_nr_fast(x: f64, z_charge: f64, half_ln_theta_e: f64) -> f64 {
    if x < 1e-30 {
        return 1.0;
    }
    let arg = S3_PI * ((2.25 / (x * z_charge)).ln() + half_ln_theta_e) + 1.425;
    1.0 + softplus(arg)
}

/// Fast Gaunt factor with precomputed ln(x) and 0.5*ln(θ_e).
///
/// Avoids the ln() call inside the grid loop by using precomputed ln_x.
/// `ln(2.25/(x*Z))` = `ln(2.25/Z) - ln(x)`.
#[inline]
fn gaunt_ff_nr_fast_preln(ln_x: f64, z_charge: f64, half_ln_theta_e: f64) -> f64 {
    // Guard against x < 1e-30 (ln_x < -69.0) to match gaunt_ff_nr_fast and prevent Inf
    if ln_x < -69.0 {
        return 1.0;
    }
    let ln_2_25_over_z = if z_charge < 1.5 { LN_2_25 } else { LN_1_125 };
    let arg = S3_PI * (ln_2_25_over_z - ln_x + half_ln_theta_e) + 1.425;
    1.0 + softplus(arg)
}

/// Softplus function with asymptotic shortcuts.
///
/// softplus(x) = ln(1 + exp(x))
/// - If x > 20: softplus ≈ x (error < 2e-9)
/// - If x < -20: softplus ≈ exp(x) (error < 2e-9)
/// Saves exp+ln calls for most grid points in asymptotic regimes.
#[inline]
fn softplus(arg: f64) -> f64 {
    if arg > 20.0 {
        arg
    } else if arg < -20.0 {
        arg.exp()
    } else {
        (1.0 + arg.exp()).ln()
    }
}

/// BR emission coefficient K_BR at frequency x.
///
/// K_BR(x) = (α λ_e³/(2π√(6π))) θ_e^{-7/2} × (e^{-xφ}/φ³) × Σ_i Z_i² N_i g_ff
///
/// This is the coefficient such that:
///   dn/dτ|_BR = (K_BR / x³) [n_eq − n]
///
/// where n_eq is the Planck distribution at T_e.
/// The rate per unit volume is ∝ N_e × N_i (two-body process). Converting
/// to per Thomson time (÷ N_e σ_T c) cancels one N_e, leaving N_i × λ_e³
/// (dimensionless) in the coefficient.
///
/// # Arguments
/// * `x` - dimensionless frequency hν/(kT_z)
/// * `theta_e` - electron temperature kT_e/(m_e c²)
/// * `theta_z` - reference temperature kT_z/(m_e c²)
/// * `n_h` - hydrogen number density [1/m³]
/// * `n_he` - helium number density [1/m³]
/// * `n_e` - electron number density [1/m³]
/// * `x_e_frac` - ionization fraction X_e = N_e/N_H
pub fn br_emission_coefficient(
    x: f64,
    theta_e: f64,
    theta_z: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    cosmo: &crate::cosmology::Cosmology,
) -> f64 {
    if theta_e < 1e-30 || n_e < 1e-30 {
        return 0.0;
    }

    let phi = theta_z / theta_e;

    // Temperature factor: θ_e^{-7/2} × e^{-xφ} / φ³
    let temp_factor = theta_e.powf(-3.5) * (-x * phi).exp() / phi.powi(3);

    // Species sum: Σ_i Z_i² (N_i) g_ff(Z_i, x, θ_e)
    // H⁺: Z=1, He²⁺: Z=2, He⁺: Z=1

    // H⁺ and He⁺ both have Z=1, so their Gaunt factors are identical
    let g_z1 = gaunt_ff_nr(x, theta_e, 1.0);
    let g_z2 = gaunt_ff_nr(x, theta_e, 2.0);

    // He ionization from Saha equations (via the temperature → redshift mapping).
    //
    // NOTE: Saha overestimates the speed of helium recombination at z ≲ 2000,
    // where non-equilibrium delays (Peebles-like) become important. Because
    // Saha drives n_HeII → 0 rapidly below ~2000 while the true residual HeII
    // fraction decays more slowly, this under-counts HeII's contribution to
    // BR at post-HeII-recombination redshifts. The effect is bounded at
    // ~4% of the BR species_sum at z ~ 10³ (where the Saha approximation is
    // worst) and is negligible for thermal-distortion accuracy. If sub-percent
    // BR in the post-recombination tail is ever required, wire in the full
    // recombination::x_e treatment for helium.
    // Derive z from T_rad using the caller's cosmology T_CMB, not the hardcoded
    // default. Otherwise a non-default cosmo.t_cmb shifts the Saha ionization
    // curve in temperature by the ratio (cosmo.t_cmb / T_CMB_0).
    let t_rad = theta_z * M_E_C2 / K_BOLTZMANN;
    let z_approx = (t_rad / cosmo.t_cmb - 1.0).max(0.0);

    let y_he_ii = crate::recombination::saha_he_ii(z_approx, cosmo);
    let y_he_i = crate::recombination::saha_he_i(z_approx, cosmo);

    let n_hii = x_e_frac.min(1.0) * n_h;
    let n_heiii = y_he_ii * n_he;
    let n_heii = (y_he_i - y_he_ii).max(0.0) * n_he;

    let species_sum = n_hii * g_z1 + 4.0 * n_heiii * g_z2 + n_heii * g_z1;

    BR_PREFACTOR * temp_factor * species_sum
}

/// BR emission coefficient with pre-computed He ionization fractions.
///
/// Same physics as `br_emission_coefficient` but avoids redundant Saha
/// evaluations when called in a grid loop (z_approx is identical for all x).
pub fn br_emission_coefficient_with_he(
    x: f64,
    theta_e: f64,
    theta_z: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    y_he_ii: f64,
    y_he_i: f64,
) -> f64 {
    if theta_e < 1e-30 || n_e < 1e-30 {
        return 0.0;
    }

    let phi = theta_z / theta_e;
    let temp_factor = theta_e.powf(-3.5) * (-x * phi).exp() / phi.powi(3);

    // H⁺ and He⁺ both have Z=1, so their Gaunt factors are identical
    let g_z1 = gaunt_ff_nr(x, theta_e, 1.0);
    let g_he2 = gaunt_ff_nr(x, theta_e, 2.0);

    let n_hii = x_e_frac.min(1.0) * n_h;
    let n_heiii = y_he_ii * n_he;
    let n_heii = (y_he_i - y_he_ii).max(0.0) * n_he;

    let species_sum = n_hii * g_z1 + 4.0 * n_heiii * g_he2 + n_heii * g_z1;

    BR_PREFACTOR * temp_factor * species_sum
}

/// Precomputed x-independent BR factors for use in the grid loop.
///
/// Call `br_precompute` once per timestep, then `br_emission_coefficient_fast`
/// for each grid point. This hoists θ_e^{-7/2}/φ³ and species densities
/// out of the inner loop.
pub struct BrPrecomputed {
    /// BR_PREFACTOR × θ_e^{-7/2} / φ³
    pub base_factor: f64,
    /// φ = θ_z / θ_e (for the exp(-xφ) x-dependent part)
    pub phi: f64,
    pub n_hii: f64,
    pub n_heiii: f64,
    pub n_heii: f64,
    /// 0.5 * ln(θ_e), precomputed for fast Gaunt factor evaluation
    pub half_ln_theta_e: f64,
}

/// Precompute x-independent BR factors.
pub fn br_precompute(
    theta_e: f64,
    theta_z: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    y_he_ii: f64,
    y_he_i: f64,
) -> Option<BrPrecomputed> {
    if theta_e < 1e-30 || n_e < 1e-30 {
        return None;
    }
    let phi = theta_z / theta_e;
    let base_factor = BR_PREFACTOR * theta_e.powf(-3.5) / (phi * phi * phi);
    Some(BrPrecomputed {
        base_factor,
        phi,
        n_hii: x_e_frac.min(1.0) * n_h,
        n_heiii: y_he_ii * n_he,
        n_heii: (y_he_i - y_he_ii).max(0.0) * n_he,
        half_ln_theta_e: 0.5 * theta_e.ln(),
    })
}

/// Fast BR emission coefficient using precomputed x-independent factors.
///
/// Only computes the x-dependent parts: exp(-xφ) and Gaunt factors.
#[inline]
pub fn br_emission_coefficient_fast(x: f64, pre: &BrPrecomputed) -> f64 {
    let exp_xphi = (-x * pre.phi).exp();
    // H⁺ and He⁺ both have Z=1, so their Gaunt factors are identical
    let g_z1 = gaunt_ff_nr_fast(x, 1.0, pre.half_ln_theta_e);
    let g_he2 = gaunt_ff_nr_fast(x, 2.0, pre.half_ln_theta_e);

    let species_sum = pre.n_hii * g_z1 + 4.0 * pre.n_heiii * g_he2 + pre.n_heii * g_z1;

    pre.base_factor * exp_xphi * species_sum
}

/// Fast BR emission coefficient with precomputed ln(x).
///
/// Same as `br_emission_coefficient_fast` but avoids ln() calls in the
/// Gaunt factor by using a precomputed ln(x) value.
#[inline]
pub fn br_emission_coefficient_fast_preln(x: f64, ln_x: f64, pre: &BrPrecomputed) -> f64 {
    let exp_xphi = (-x * pre.phi).exp();
    let g_z1 = gaunt_ff_nr_fast_preln(ln_x, 1.0, pre.half_ln_theta_e);
    let g_he2 = gaunt_ff_nr_fast_preln(ln_x, 2.0, pre.half_ln_theta_e);

    let species_sum = pre.n_hii * g_z1 + 4.0 * pre.n_heiii * g_he2 + pre.n_heii * g_z1;

    pre.base_factor * exp_xphi * species_sum
}

/// BR emission rate integrated over frequency (for electron temperature equation).
///
/// H_BR = (1/(4 G₃ θ_z)) ∫ [1 − n(e^{x_e} − 1)] K_BR(x)/x³ · x³ dx
///      = (1/(4 G₃ θ_z)) ∫ [1 − n(e^{x_e} − 1)] K_BR(x) dx
///
/// This enters the electron temperature evolution as a cooling term,
/// analogous to dc_heating_integral() in double_compton.rs.
pub fn br_heating_integral(
    x_grid: &[f64],
    delta_n: &[f64],
    theta_z: f64,
    theta_e: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    y_he_ii: f64,
    y_he_i: f64,
) -> f64 {
    if theta_e < 1e-30 || n_e < 1e-30 {
        return 0.0;
    }
    let phi = theta_z / theta_e;
    let mut integral = 0.0;

    for i in 1..x_grid.len() {
        let dx = x_grid[i] - x_grid[i - 1];
        let x_mid = 0.5 * (x_grid[i] + x_grid[i - 1]);
        let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
        let n_mid = planck(x_mid) + dn_mid;
        let x_e = x_mid * phi;

        let k_br = br_emission_coefficient_with_he(
            x_mid, theta_e, theta_z, n_h, n_he, n_e, x_e_frac, y_he_ii, y_he_i,
        );
        // Integrand: [1 - n (e^{x_e} - 1)] K_BR dx
        let factor = 1.0 - n_mid * x_e.exp_m1();
        integral += factor * k_br * dx;
    }

    integral / (4.0 * crate::constants::G3_PLANCK * theta_z)
}

/// Compute the BR contribution to the photon equation RHS (test-only).
///
/// Production code uses the coupled inplace solver with precomputed rates.
///
/// **Do not promote this to production.** Same near-cancellation hazard as
/// `double_compton::dc_rhs`: the naive `(e^{x_e} − 1)` form loses precision
/// near `ρ_e = 1`. The production path (`solver.rs::compute_emission_rates`)
/// uses the analytical Taylor expansion when `|ρ_e − 1| < 0.01`.
#[cfg(test)]
pub fn br_rhs(
    x_grid: &[f64],
    delta_n: &[f64],
    theta_z: f64,
    theta_e: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    cosmo: &crate::cosmology::Cosmology,
) -> Vec<f64> {
    let n = x_grid.len();
    let phi = theta_z / theta_e;
    let mut rhs = vec![0.0; n];

    for i in 0..n {
        let xi = x_grid[i];
        let k_br = br_emission_coefficient(xi, theta_e, theta_z, n_h, n_he, n_e, x_e_frac, cosmo);
        let x_e = xi * phi; // frequency at electron temperature

        // Chluba & Sunyaev (2012) Eq. (8) form (matches dc_rhs):
        //   dn/dτ|_BR = (K_BR/x³) [1 - n(e^{x_e} - 1)]
        let n_current = planck(xi) + delta_n[i];
        rhs[i] = k_br / xi.powi(3) * (1.0 - n_current * x_e.exp_m1());
    }

    rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaunt_ff_positive_and_physical() {
        assert!(gaunt_ff_nr(0.1, 1e-5, 1.0) > 0.0);
        assert!(gaunt_ff_nr(1.0, 1e-4, 1.0) > 0.0);
        assert!(gaunt_ff_nr(0.001, 1e-6, 1.0) > 0.0);
        // Z=2 (He²⁺) should also be positive
        assert!(gaunt_ff_nr(0.1, 1e-5, 2.0) > 0.0);
        // Z=2 Gaunt factor differs from Z=1 due to ln(2.25/(x*Z))
        let g1 = gaunt_ff_nr(1.0, 1e-5, 1.0);
        let g2 = gaunt_ff_nr(1.0, 1e-5, 2.0);
        assert!(g1 != g2, "Z=1 and Z=2 Gaunt factors should differ");

        // Physical check: Gaunt factor should be >= 1.0 for all inputs
        // (the softplus function adds >= 0 to 1.0 base)
        for &x in &[0.001, 0.01, 0.1, 1.0, 10.0] {
            for &theta in &[1e-6, 1e-5, 1e-4, 1e-3] {
                let g = gaunt_ff_nr(x, theta, 1.0);
                assert!(g >= 1.0, "g_ff(x={x}, θ={theta}) = {g} < 1.0");
            }
        }

        // At very low x (RJ regime), Gaunt factor should be large (~5-15)
        // Born approximation: g ≈ (√3/π) ln(2.25 θ_e / x) for small x
        let g_lowx = gaunt_ff_nr(1e-4, 1e-4, 1.0);
        assert!(
            g_lowx > 3.0,
            "Gaunt factor at very low x should be > 3 (classical limit): got {g_lowx:.2}"
        );

        // Gaunt factor should decrease with increasing x (less classical)
        let g_highx = gaunt_ff_nr(1.0, 1e-4, 1.0);
        assert!(
            g_lowx > g_highx,
            "Gaunt factor should decrease with x: g(1e-4)={g_lowx:.2}, g(1)={g_highx:.2}"
        );
    }

    #[test]
    fn test_br_coefficient_positive() {
        let theta_e = 1e-5;
        let theta_z = theta_e;
        let n_h = 1e6; // dummy density
        let n_he = 0.08 * n_h;
        let n_e = n_h;
        let x_e = 1.0;
        let cosmo = crate::cosmology::Cosmology::default();

        for &x in &[0.001, 0.01, 0.1, 1.0] {
            let k = br_emission_coefficient(x, theta_e, theta_z, n_h, n_he, n_e, x_e, &cosmo);
            assert!(k >= 0.0, "K_BR should be non-negative at x={x}: K={k}");
        }
    }

    #[test]
    fn test_br_rhs_zero_for_planck() {
        let x: Vec<f64> = (1..100).map(|i| 0.01 * i as f64).collect();
        let delta_n = vec![0.0; x.len()];
        let theta = 1e-5;
        let n_h = 1e6;
        let n_he = 0.08 * n_h;
        let n_e = n_h;
        let cosmo = crate::cosmology::Cosmology::default();

        let rhs = br_rhs(&x, &delta_n, theta, theta, n_h, n_he, n_e, 1.0, &cosmo);
        let max_rhs: f64 = rhs
            .iter()
            .map(|v| v.abs())
            .fold(0.0, |a, b| if b.is_nan() { f64::NAN } else { a.max(b) });
        assert!(
            max_rhs < 1e-20,
            "BR RHS should be zero for Planck with T_e=T_z: max = {max_rhs}"
        );
    }

    #[test]
    fn test_br_heating_integral_zero_for_planck() {
        // For Planck spectrum with T_e = T_z, BR heating integral should vanish
        let x: Vec<f64> = (1..500).map(|i| 0.001 + 0.06 * i as f64).collect();
        let delta_n = vec![0.0; x.len()];
        let theta = 4.6e-4; // z ~ 1e6
        let n_h = 1e12;
        let n_he = 0.08 * n_h;
        let n_e = n_h;

        let h_br = br_heating_integral(&x, &delta_n, theta, theta, n_h, n_he, n_e, 1.0, 1.0, 1.0);
        assert!(
            h_br.abs() < 1e-10,
            "BR heating integral should be ~0 for Planck: H_BR = {h_br}"
        );
    }

    #[test]
    fn test_br_rhs_nonzero_for_te_ne_tz() {
        // When T_e > T_z, BR should drive photon creation at low x (positive RHS)
        // and absorption at high x. The detailed balance factor [1 - n(e^{x_e}-1)]
        // must be present to get the correct sign pattern.
        let x: Vec<f64> = (1..100).map(|i| 0.01 * i as f64).collect();
        let delta_n = vec![0.0; x.len()];
        let theta_e = 1.001e-5; // T_e slightly above T_z
        let theta_z = 1e-5;
        let n_h = 1e6;
        let n_he = 0.08 * n_h;
        let n_e = n_h;
        let cosmo = crate::cosmology::Cosmology::default();

        let rhs = br_rhs(&x, &delta_n, theta_z, theta_e, n_h, n_he, n_e, 1.0, &cosmo);

        // At low x, T_e > T_z means equilibrium has more photons → emission (positive)
        assert!(
            rhs[1] > 0.0,
            "BR RHS should be positive at low x when T_e > T_z: rhs[1] = {}",
            rhs[1]
        );

        // Verify consistency with dc_rhs form: both use [1 - n*(e^{x_e}-1)]
        // so for Planck spectrum (delta_n=0), the sign at low x should match
        let dc_rhs_vals = crate::double_compton::dc_rhs(&x, &delta_n, theta_z, theta_e);
        assert!(
            dc_rhs_vals[1] > 0.0,
            "DC RHS should also be positive at low x when T_e > T_z"
        );
    }

    #[test]
    fn test_softplus_all_branches() {
        // Normal branch: -20 < x < 20
        let sp_0 = softplus(0.0);
        assert!((sp_0 - (2.0_f64).ln()).abs() < 1e-14, "softplus(0) = ln(2)");

        // High branch: x > 20
        let sp_high = softplus(25.0);
        assert!(
            (sp_high - 25.0).abs() < 1e-8,
            "softplus(25) ≈ 25: got {sp_high}"
        );

        // Low branch: x < -20
        let sp_low = softplus(-25.0);
        let expected = (-25.0_f64).exp();
        assert!(
            (sp_low - expected).abs() < 1e-20,
            "softplus(-25) ≈ exp(-25)"
        );
    }

    /// Verify all BR fast variants (fast, fast_preln, with_he) match the reference
    /// br_emission_coefficient to machine precision across multiple parameter combos.
    /// Consolidates: test_br_precompute_and_fast_consistency, test_br_with_he_matches_standard,
    /// test_br_emission_fast_matches_full, test_br_emission_fast_preln_matches.
    #[test]
    fn test_br_fast_variants_match_reference() {
        let cases: Vec<(f64, f64, f64, f64, f64, f64, f64)> = vec![
            // (x, theta_e, theta_z, n_h, n_he, n_e, x_e_frac)
            (0.001, 1e-5, 1e-5, 1e6, 8e4, 1e6, 1.0),
            (0.01, 1e-5, 1e-5, 1e6, 8e4, 1e6, 1.0),
            (0.1, 1e-4, 9e-5, 1e8, 8e6, 1.1e8, 1.1),
            (1.0, 5e-6, 5e-6, 1e10, 8e8, 1e10, 1.0),
            (10.0, 1e-3, 1e-3, 1e12, 8e10, 1e12, 1.0),
        ];

        let cosmo = crate::cosmology::Cosmology::default();

        for &(x, theta_e, theta_z, n_h, n_he, n_e, x_e_frac) in &cases {
            let y_he_ii = 0.9;
            let y_he_i = 1.0;

            // Reference: br_emission_coefficient_with_he
            let k_ref = br_emission_coefficient_with_he(
                x, theta_e, theta_z, n_h, n_he, n_e, x_e_frac, y_he_ii, y_he_i,
            );

            // Fast variant
            let pre =
                br_precompute(theta_e, theta_z, n_h, n_he, n_e, x_e_frac, y_he_ii, y_he_i).unwrap();
            let k_fast = br_emission_coefficient_fast(x, &pre);
            let rel_fast = (k_fast - k_ref).abs() / k_ref.max(1e-30);
            assert!(
                rel_fast < 1e-10,
                "fast mismatch at x={x}: ref={k_ref}, fast={k_fast}, rel={rel_fast}"
            );

            // Fast with pre-computed ln(x)
            let k_preln = br_emission_coefficient_fast_preln(x, x.ln(), &pre);
            let rel_preln = (k_preln - k_ref).abs() / k_ref.max(1e-30);
            assert!(
                rel_preln < 1e-10,
                "fast_preln mismatch at x={x}: ref={k_ref}, preln={k_preln}, rel={rel_preln}"
            );
        }

        // Also verify br_emission_coefficient (with Saha) matches br_emission_coefficient_with_he
        // at a specific point where we can get the same He ionization fractions
        let theta_e = 1e-5;
        let theta_z = theta_e;
        let n_h = 1e6;
        let n_he = 0.08 * n_h;
        let n_e = n_h;
        let x_e = 1.0;
        let z_approx = (theta_z / crate::constants::theta_z(0.0) - 1.0).max(100.0);
        let y_he_i = crate::recombination::saha_he_i(z_approx, &cosmo);
        let y_he_ii = crate::recombination::saha_he_ii(z_approx, &cosmo);

        let k_saha = br_emission_coefficient(0.1, theta_e, theta_z, n_h, n_he, n_e, x_e, &cosmo);
        let k_he = br_emission_coefficient_with_he(
            0.1, theta_e, theta_z, n_h, n_he, n_e, x_e, y_he_ii, y_he_i,
        );
        let rel = (k_he - k_saha).abs() / k_saha.max(1e-30);
        assert!(
            rel < 1e-10,
            "with_he vs saha mismatch: saha={k_saha}, he={k_he}, rel={rel}"
        );
    }

    #[test]
    fn test_br_precompute_none_for_zero_theta() {
        let pre = br_precompute(0.0, 1e-5, 1e6, 1e4, 1e6, 1.0, 0.0, 1.0);
        assert!(pre.is_none(), "Should return None for theta_e = 0");
    }

    #[test]
    fn test_br_precompute_returns_none_for_tiny_inputs() {
        // theta_e < 1e-30
        let pre = br_precompute(1e-31, 1e-5, 1e6, 8e4, 1e6, 1.0, 0.0, 1.0);
        assert!(pre.is_none(), "Should return None for theta_e < 1e-30");

        // n_e < 1e-30
        let pre = br_precompute(1e-5, 1e-5, 1e6, 8e4, 1e-31, 1.0, 0.0, 1.0);
        assert!(pre.is_none(), "Should return None for n_e < 1e-30");

        // Both tiny
        let pre = br_precompute(1e-31, 1e-5, 1e6, 8e4, 1e-31, 1.0, 0.0, 1.0);
        assert!(pre.is_none(), "Should return None when both are tiny");
    }

    #[test]
    fn test_gaunt_ff_nr_guards() {
        // theta_e < 1e-30 → returns 1.0
        assert_eq!(gaunt_ff_nr(0.1, 1e-31, 1.0), 1.0);
        assert_eq!(gaunt_ff_nr(0.1, 0.0, 1.0), 1.0);

        // x < 1e-30 → returns 1.0 (via gaunt_ff_nr_fast guard)
        assert_eq!(gaunt_ff_nr(1e-31, 1e-5, 1.0), 1.0);
        assert_eq!(gaunt_ff_nr(0.0, 1e-5, 1.0), 1.0);

        // Normal inputs → returns > 1.0 (softplus adds >= 0 to 1.0)
        let g = gaunt_ff_nr(0.1, 1e-5, 1.0);
        assert!(g >= 1.0, "Gaunt factor should be >= 1.0, got {g}");
    }

    #[test]
    fn test_br_hii_capped_at_nh() {
        // When x_e_frac > 1 (includes helium electrons), n_hii should not exceed n_h
        let theta_e = 1e-5;
        let theta_z = theta_e;
        let n_h = 1e6;
        let n_he = 0.08 * n_h;
        let n_e = 1.16 * n_h; // x_e > 1 due to helium
        let x_e = 1.16; // total electron fraction including He
        let cosmo = crate::cosmology::Cosmology::default();

        let k = br_emission_coefficient(0.1, theta_e, theta_z, n_h, n_he, n_e, x_e, &cosmo);
        // Should be finite and not blow up from overcounting
        assert!(k.is_finite() && k >= 0.0, "K_BR should be finite: {k}");

        // Compare with x_e = 1.0 — should be similar (H contribution capped)
        let k_ref = br_emission_coefficient(0.1, theta_e, theta_z, n_h, n_he, n_e, 1.0, &cosmo);
        // With x_e capped at 1.0 for H+ contribution, H+ part should be identical.
        // Only He ionization fractions differ (x_e=1.16 vs 1.0 changes y_he_i/y_he_ii).
        // Total should agree within 20% since H+ dominates.
        assert!(
            (k / k_ref.max(1e-30) - 1.0).abs() < 0.2,
            "BR with x_e=1.16 should be within 20% of x_e=1.0: {k:.4e} vs {k_ref:.4e}"
        );
    }

    #[test]
    fn test_br_hardcoded_constants() {
        // Verify hardcoded transcriptions against computed values.
        // sqrt(6π)
        let sqrt_6pi = (6.0 * std::f64::consts::PI).sqrt();
        assert!(
            (sqrt_6pi - 4.341_607_527_349_605_5).abs() < 1e-14,
            "sqrt(6π) mismatch: {sqrt_6pi}"
        );

        // √3/π
        let s3_pi_exact = 3.0_f64.sqrt() / std::f64::consts::PI;
        assert!(
            (S3_PI - s3_pi_exact).abs() < 1e-15,
            "S3_PI mismatch: hardcoded={S3_PI}, computed={s3_pi_exact}"
        );

        // ln(2.25)
        let ln_225_exact = 2.25_f64.ln();
        assert!(
            (LN_2_25 - ln_225_exact).abs() < 1e-15,
            "LN_2_25 mismatch: hardcoded={LN_2_25}, computed={ln_225_exact}"
        );

        // ln(2.25/2) = ln(1.125)
        let ln_1125_exact = 1.125_f64.ln();
        assert!(
            (LN_1_125 - ln_1125_exact).abs() < 1e-15,
            "LN_1_125 mismatch: hardcoded={LN_1_125}, computed={ln_1125_exact}"
        );

        // BR_PREFACTOR = α λ_e³ / (2π √(6π))
        let prefactor_exact =
            ALPHA_FS * LAMBDA_ELECTRON.powi(3) / (2.0 * std::f64::consts::PI * sqrt_6pi);
        assert!(
            (BR_PREFACTOR - prefactor_exact).abs() / prefactor_exact < 1e-14,
            "BR_PREFACTOR mismatch: hardcoded={BR_PREFACTOR}, computed={prefactor_exact}"
        );
    }

    /// Verify K_BR has physically reasonable absolute magnitude at z=10^5.
    ///
    /// The historical /n_e bug made K_BR ~ 10^11 instead of O(1)–O(10^3).
    /// This test catches that class of error by computing K_BR from first
    /// principles and asserting the order of magnitude.
    ///
    /// Hand calculation at z=10^5, x=0.1, rho_e=1 (T_e=T_z):
    ///   BR_PREFACTOR ≈ 6.1e-40 m³
    ///   θ_z^{-7/2} ≈ (4.60e-5)^{-3.5} ≈ 2.1e16
    ///   exp(-0.1) ≈ 0.90
    ///   n_HII ≈ 1.9e14 m^{-3} (fully ionized H at z=10^5)
    ///   g_ff ≈ 3
    ///   K_BR ≈ 6.1e-40 × 2.1e16 × 0.90 × 1.9e14 × 3 ≈ 6.6e-9
    #[test]
    fn test_br_emission_coefficient_magnitude() {
        let cosmo = crate::cosmology::Cosmology::default();
        let z = 1.0e5;
        let tz = crate::constants::theta_z(z);
        let n_h = cosmo.n_h(z);
        let n_he = cosmo.n_he(z);
        let x_e = 1.0 + 2.0 * cosmo.f_he(); // fully ionized
        let n_e = cosmo.n_e(z, x_e);

        // At equilibrium: T_e = T_z, so theta_e = theta_z
        let k_br = br_emission_coefficient(0.1, tz, tz, n_h, n_he, n_e, x_e, &cosmo);

        eprintln!("K_BR(x=0.1, z=1e5) = {k_br:.4e}");
        eprintln!("  BR_PREFACTOR = {BR_PREFACTOR:.4e}");
        eprintln!("  theta_z = {tz:.4e}");
        eprintln!("  n_H = {n_h:.4e}, n_He = {n_he:.4e}, n_e = {n_e:.4e}");

        // Must be positive
        assert!(k_br > 0.0, "K_BR must be positive, got {k_br}");

        // Order-of-magnitude check: should be O(10^{-9}) to O(10^{-7}),
        // NOT O(10^{11}) as the historical bug produced.
        assert!(
            k_br > 1e-12 && k_br < 1e-4,
            "K_BR(x=0.1, z=1e5) = {k_br:.4e} is outside physically reasonable \
             range [1e-12, 1e-4]. Historical /n_e bug gave ~10^11."
        );

        // Verify hand calculation to within 2 OOM
        let hand_estimate = 6.6e-9;
        let ratio = k_br / hand_estimate;
        assert!(
            ratio > 0.01 && ratio < 100.0,
            "K_BR = {k_br:.4e} differs from hand estimate {hand_estimate:.4e} by {ratio:.1e}x"
        );

        // Also check at x=1 (less Boltzmann suppression of g_ff)
        let k_br_x1 = br_emission_coefficient(1.0, tz, tz, n_h, n_he, n_e, x_e, &cosmo);
        eprintln!("K_BR(x=1.0, z=1e5) = {k_br_x1:.4e}");
        assert!(
            k_br_x1 > 1e-12 && k_br_x1 < 1e-4,
            "K_BR(x=1.0, z=1e5) = {k_br_x1:.4e} outside [1e-12, 1e-4]"
        );

        // x=0.1 should have more emission than x=1 (exponential suppression)
        assert!(
            k_br > k_br_x1,
            "K_BR should decrease with x: K(0.1)={k_br:.4e} < K(1.0)={k_br_x1:.4e}"
        );
    }
}
