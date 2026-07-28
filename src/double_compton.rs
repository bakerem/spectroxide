//! Double Compton (DC) scattering: γ + e → γ + γ + e
//!
//! Photon-number-changing process that dominates at z > few × 10⁵.
//! In the soft photon limit, DC emission creates low-frequency photons
//! that drive the spectrum toward a Planck distribution.
//!
//! The DC emission coefficient:
//!   K_DC(x, θ_z, θ_e) = (4α/3π) θ_z² g_dc(x, θ_z, θ_e)
//!
//! where g_dc is the DC Gaunt factor including relativistic corrections.
//!
//! References:
//! - Lightman (1981) — original DC coefficient
//! - Chluba, Sazonov & Sunyaev (2007), A&A 468, 785 [arXiv:0705.3033]
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 10-13]

use crate::constants::*;
// planck is only used by the cfg(test) dc_rhs helper and the unit tests since
// the production dc_heating_integral was removed (R2 mutation audit).
#[cfg(test)]
use crate::spectrum::planck;

/// DC Gaunt factor in the soft photon limit with relativistic corrections.
///
/// g_dc(x, θ_z, θ_e) ≈ (I₄^pl / (1 + 14.16 θ_z)) · H_dc(x)
///
/// where I₄^pl = 4π⁴/15 and H_dc is the high-frequency suppression.
///
/// The factor (1 + 14.16 θ_z)⁻¹ is the leading-order relativistic correction
/// from thermal averaging (Chluba+ 2007), evaluated at the **photon
/// temperature** θ_z. Strictly the thermal average runs over the electron
/// distribution, so θ_e would be the cleaner choice when T_e ≠ T_z
/// (injection-driven heating); however CS2012 parametrises DC through θ_z
/// and the difference is <1% for θ_z ≲ 10⁻³ even when |ρ_e−1| ~ 0.1.
pub fn dc_gaunt_factor(x: f64, theta_z: f64) -> f64 {
    I4_PLANCK * dc_relativistic_correction(theta_z) * dc_high_freq_suppression(x)
}

/// Leading-order relativistic correction to the DC Gaunt factor,
/// (1 + 14.16 θ_z)⁻¹ (Chluba, Sazonov & Sunyaev 2007).
///
/// Single source of truth: [`dc_gaunt_factor`] and [`dc_prefactor`] both call
/// this, so the correction cannot drift between the diagnostic path and the
/// solver hot loop (R2 mutation audit, fix A1).
#[inline]
pub fn dc_relativistic_correction(theta_z: f64) -> f64 {
    1.0 / (1.0 + 14.16 * theta_z)
}

/// (4α/3π), the numerical prefactor of the DC emission coefficient.
const DC_ALPHA_COEFF: f64 = 4.0 * ALPHA_FS / (3.0 * std::f64::consts::PI);

/// High-frequency suppression factor for DC emission.
///
/// H_dc^pl(x) ≈ exp(-2x) [1 + 3x/2 + 29x²/24 + 11x³/16 + 5x⁴/12]
///
/// This accounts for the suppression of DC emission at high photon energies
/// where the soft-photon approximation breaks down.
///
/// Reference: Chluba & Sunyaev (2012), Eq. 13.
pub fn dc_high_freq_suppression(x: f64) -> f64 {
    if x > 100.0 {
        return 0.0;
    }
    (-2.0 * x).exp() * (1.0 + x * (1.5 + x * (29.0 / 24.0 + x * (11.0 / 16.0 + x * (5.0 / 12.0)))))
}

/// DC emission coefficient K_DC(x, θ_z, θ_e).
///
/// K_DC = (4α/3π) θ_z² g_dc(x, θ_z, θ_e)
///
/// The emission/absorption term in the photon equation is:
///   dn/dτ|_DC = (K_DC / x³) [n_eq - n]
///
/// where n_eq is the equilibrium distribution (Planck at the electron temperature).
/// Delegates to [`dc_prefactor`] × [`dc_high_freq_suppression`] so that this
/// function and the solver hot loop share one implementation of K_DC. Before
/// the R2 mutation audit these were two hand-maintained copies of the same
/// arithmetic and only *this* one was covered by the DC/BR literature anchor,
/// leaving the production prefactor unpinned (fix A1).
pub fn dc_emission_coefficient(x: f64, theta_z: f64) -> f64 {
    dc_emission_coefficient_fast(x, dc_prefactor(theta_z))
}

/// Precompute the x-independent DC prefactor: (4α/3π) θ_z² × I₄^pl / (1 + 14.16 θ_z)
///
/// The only x-dependent part remaining is H_dc(x) = exp(-2x) × polynomial.
pub fn dc_prefactor(theta_z: f64) -> f64 {
    DC_ALPHA_COEFF * theta_z * theta_z * I4_PLANCK * dc_relativistic_correction(theta_z)
}

/// Fast DC emission coefficient using precomputed x-independent prefactor.
#[inline]
pub fn dc_emission_coefficient_fast(x: f64, dc_pre: f64) -> f64 {
    dc_pre * dc_high_freq_suppression(x)
}

/// Compute the DC contribution to the photon equation RHS (test-only).
///
/// Production code uses the coupled inplace solver with precomputed rates.
///
/// **Do not promote this to production.** The naive `(e^{x_e} − 1)` form used
/// here loses precision as `|ρ_e − 1| → 0` because the source term subtracts
/// two nearly equal numbers. `solver.rs::compute_emission_rates` switches to
/// the analytical Taylor expansion `x(ρ_e−1)/ρ_e · n_pl(1+n_pl)` when
/// `|ρ_e − 1| < 0.01` to avoid this. Any caller that evaluates the DC source
/// at the sub-percent T_e deviations typical of post-recombination physics
/// needs the same Taylor branch.
#[cfg(test)]
pub fn dc_rhs(x: &[f64], delta_n: &[f64], theta_z: f64, theta_e: f64) -> Vec<f64> {
    let n = x.len();
    let phi = theta_z / theta_e;
    let mut rhs = vec![0.0; n];

    for i in 0..n {
        let xi = x[i];
        let k_dc = dc_emission_coefficient(xi, theta_z);
        let x_e = xi * phi; // frequency at electron temperature

        // Chluba & Sunyaev (2012) Eq. (8):
        //   dn/dτ|_DC = (K_DC/x³) [1 - n(e^{x_e} - 1)]
        let n_current = planck(xi) + delta_n[i];
        rhs[i] = k_dc / xi.powi(3) * (1.0 - n_current * x_e.exp_m1());
    }

    rhs
}

// NOTE: the standalone `dc_heating_integral` was removed (2026-07-06, R2
// mutation audit). It duplicated the DC-heating logic that the production solver
// actually uses — `dcbr_heating_with_derivative` in solver.rs — but was called
// by no production path (only a single test exercising the trivial Planck→0
// case, which could not catch scaling/normalization mutations). Mutation testing
// flagged 41 surviving mutants on this dead duplicate; removing it eliminates the
// duplicate-divergence hazard. The live DC heating term lives in solver.rs.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_suppression_low_x() {
        // H_dc(0) = 1
        assert!((dc_high_freq_suppression(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_dc_suppression_high_x() {
        // H_dc should vanish for large x
        assert!(dc_high_freq_suppression(50.0) < 1e-20);
    }

    #[test]
    fn test_dc_coefficient_positive() {
        let theta_z = 4.6e-4; // z ≈ 10^6
        for &x in &[0.001, 0.01, 0.1, 1.0, 5.0] {
            assert!(
                dc_emission_coefficient(x, theta_z) >= 0.0,
                "K_DC should be non-negative at x={x}"
            );
        }
    }

    #[test]
    fn test_dc_rhs_zero_for_planck() {
        // For Planck spectrum with T_e = T_z, DC should produce no change
        let x: Vec<f64> = (0..100).map(|i| 0.01 + 0.3 * i as f64).collect();
        let delta_n = vec![0.0; x.len()];
        let theta = 4.6e-4;

        let rhs = dc_rhs(&x, &delta_n, theta, theta);
        let max_rhs: f64 = rhs
            .iter()
            .map(|v| v.abs())
            .fold(0.0, |a, b| if b.is_nan() { f64::NAN } else { a.max(b) });
        // The [1 - n(e^x - 1)] form has O(ε) cancellation, so tolerance is machine epsilon
        assert!(
            max_rhs < 1e-12,
            "DC RHS should be zero for Planck spectrum: max = {max_rhs}"
        );
    }

    /// Detailed balance (Kirchhoff) at T_e ≠ T_z (R2 mutation audit, fix P4).
    ///
    /// DC emission and absorption must cancel identically when the photon field
    /// is a Planck spectrum at the **electron** temperature, for any ρ_e. In the
    /// code's normalisation x = hν/kT_z that spectrum is
    ///   n_eq(x) = 1/(exp(x·φ) − 1),  φ ≡ θ_z/θ_e = 1/ρ_e.
    ///
    /// The Planck test above only covers ρ_e = 1, where φ = 1 and any error in
    /// the φ convention — the exact inversion CLAUDE.md pitfall #1 warns about —
    /// is invisible. This also brackets the |ρ_e − 1| = 0.01 switch where the
    /// solver changes to the Taylor branch (pitfall #5).
    ///
    /// Scale-free assertion: the equilibrium residual is compared against the
    /// residual of a 1%-off-equilibrium spectrum, so no absolute tolerance has
    /// to be guessed.
    #[test]
    fn test_dc_detailed_balance_at_shifted_electron_temperature() {
        let x: Vec<f64> = (1..200).map(|i| 0.05 * i as f64).collect();
        let theta_z_val = 4.6e-4;

        for &rho_e in &[0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1] {
            let theta_e = theta_z_val * rho_e;
            let phi = theta_z_val / theta_e;

            let n_eq: Vec<f64> = x.iter().map(|&xi| 1.0 / (xi * phi).exp_m1()).collect();
            let dn_eq: Vec<f64> = x
                .iter()
                .zip(&n_eq)
                .map(|(&xi, &neq)| neq - planck(xi))
                .collect();
            // 1% above equilibrium: the source term becomes −0.01 K_DC/x³.
            let dn_off: Vec<f64> = x
                .iter()
                .zip(&n_eq)
                .map(|(&xi, &neq)| 1.01 * neq - planck(xi))
                .collect();

            let rhs_eq = dc_rhs(&x, &dn_eq, theta_z_val, theta_e);
            let rhs_off = dc_rhs(&x, &dn_off, theta_z_val, theta_e);

            for (i, &xi) in x.iter().enumerate() {
                assert!(
                    rhs_eq[i].abs() < 1e-10 * rhs_off[i].abs(),
                    "DC detailed balance broken at ρ_e={rho_e}, x={xi}: \
                     equilibrium residual {:.3e} vs 1%-departure {:.3e}",
                    rhs_eq[i],
                    rhs_off[i]
                );
                // Sign: an over-populated spectrum must be driven down.
                assert!(
                    rhs_off[i] < 0.0,
                    "DC must absorb an over-populated spectrum at ρ_e={rho_e}, \
                     x={xi}: rhs = {:.3e}",
                    rhs_off[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_gaunt_factor_literature_values() {
        // At theta_z = 0 (non-relativistic limit):
        // g_dc(x, 0) = I4_pl * 1.0 * H_dc(x)
        // At x=0: H_dc(0) = 1.0, so g_dc = I4_pl = 4 * G3 = 4 * pi^4/15
        let i4_pl = I4_PLANCK;
        let g_at_zero = dc_gaunt_factor(0.0, 0.0);
        assert!(
            (g_at_zero - i4_pl).abs() < 1e-10,
            "g_dc(0,0) should be I4_pl = {i4_pl}, got {g_at_zero}"
        );

        // At x=10: H_dc(10) = e^{-20} × (1 + 15 + 29×100/24 + 11×1000/16 + 5×10000/12)
        // Manually computed from the polynomial definition (Chluba & Sunyaev 2012, Eq. 13):
        let h_dc_10_manual = (-20.0_f64).exp()
            * (1.0 + 15.0 + 29.0 * 100.0 / 24.0 + 11.0 * 1000.0 / 16.0 + 5.0 * 10000.0 / 12.0);
        // h_dc_10 ≈ 2.061e-6 × 4991.0 ≈ 0.01029
        let g_at_10 = dc_gaunt_factor(10.0, 0.0);
        let expected = i4_pl * h_dc_10_manual;
        assert!(
            (g_at_10 - expected).abs() / expected < 1e-10,
            "g_dc(10,0) = {g_at_10:.6e}, expected {expected:.6e} (manually computed)"
        );
        // Verify strong suppression: g(10)/g(0) ≈ H_dc(10) ≈ 0.01
        assert!(
            g_at_10 < 0.02 * g_at_zero,
            "g_dc at x=10 should be strongly suppressed: {g_at_10} vs {g_at_zero}"
        );
    }

    #[test]
    fn test_dc_high_freq_suppression_extremes() {
        // x=200 (very high) → exactly 0.0 due to the x>100 guard
        assert_eq!(dc_high_freq_suppression(200.0), 0.0);

        // x=0.01 (very low) → approximately 1.0
        let h = dc_high_freq_suppression(0.01);
        assert!(
            (h - 1.0).abs() < 0.05,
            "H_dc(0.01) should be close to 1.0, got {h}"
        );

        // More precisely: exp(-0.02) * (1 + 0.015 + ...) ≈ 0.9802 * 1.015 ≈ 0.995
        assert!(
            h > 0.99 && h < 1.0,
            "H_dc(0.01) should be in (0.99, 1.0), got {h}"
        );
    }

    /// DC Gaunt factor at moderate x: verify H_dc against manually computed polynomial.
    ///   H_dc(x) = e^{-2x} (1 + 3x/2 + 29x²/24 + 11x³/16 + 5x⁴/12)
    #[test]
    fn test_dc_gaunt_factor_at_specific_x_values() {
        let i4_pl = I4_PLANCK;

        // x=1: H_dc(1) = e^{-2} × (1 + 1.5 + 29/24 + 11/16 + 5/12)
        //              = e^{-2} × (1 + 1.5 + 1.20833 + 0.6875 + 0.41667)
        //              = e^{-2} × 4.8125 ≈ 0.1353 × 4.8125 ≈ 0.6512
        let h_dc_1 = (-2.0_f64).exp() * (1.0 + 1.5 + 29.0 / 24.0 + 11.0 / 16.0 + 5.0 / 12.0);
        let g1 = dc_gaunt_factor(1.0, 0.0);
        let expected_g1 = i4_pl * h_dc_1;
        assert!(
            (g1 - expected_g1).abs() / expected_g1 < 1e-12,
            "g_dc(1,0) = {g1:.6e}, expected {expected_g1:.6e}"
        );

        // x=5: suppressed by e^{-10} × polynomial
        // H_dc(5) = e^{-10} × (1 + 7.5 + 29×25/24 + 11×125/16 + 5×625/12)
        //         = e^{-10} × (1 + 7.5 + 30.21 + 85.94 + 260.42)
        //         = 4.54e-5 × 385.07 ≈ 0.0175
        let g5 = dc_gaunt_factor(5.0, 0.0);
        assert!(
            g5 < 0.05 * i4_pl,
            "g_dc(5,0) = {g5:.6e} should be < 5% of g_dc(0,0) = {i4_pl:.6e}"
        );
    }

    #[test]
    fn test_dc_drives_to_equilibrium() {
        // With a positive distortion, DC should drive it toward zero (absorption)
        let x: Vec<f64> = (0..100).map(|i| 0.01 + 0.1 * i as f64).collect();
        // Positive distortion at low x
        let delta_n: Vec<f64> = x
            .iter()
            .map(|&xi| if xi < 1.0 { 1e-4 } else { 0.0 })
            .collect();
        let theta = 4.6e-4;

        let rhs = dc_rhs(&x, &delta_n, theta, theta);
        // At low x where Δn > 0, RHS should be negative (absorption)
        for (i, &xi) in x.iter().enumerate() {
            if xi < 0.5 && delta_n[i] > 0.0 {
                assert!(
                    rhs[i] < 0.0,
                    "DC should absorb excess photons at x={xi}: RHS = {}",
                    rhs[i]
                );
            }
        }
    }

    /// Verify K_DC has physically reasonable absolute magnitude.
    ///
    /// At θ_z = 4.60e-5 (z ~ 10^5), x = 1:
    ///   K_DC = (4α/3π) θ_z² I₄^pl H_dc(1) / (1 + 14.16 θ_z)
    ///        ≈ 3.10e-3 × 2.12e-9 × 25.98 × 0.558
    ///        ≈ 9.5e-11
    ///
    /// This catches formula errors (missing factors, wrong powers of θ_z).
    #[test]
    fn test_dc_emission_coefficient_magnitude() {
        let theta = 4.60e-5; // θ_z at z ~ 10^5
        let x = 1.0;

        let k_dc = dc_emission_coefficient(x, theta);
        eprintln!("K_DC(x=1, θ_z=4.60e-5) = {k_dc:.4e}");

        // Hand calculation:
        let four_alpha_3pi = 4.0 * ALPHA_FS / (3.0 * std::f64::consts::PI);
        let theta_sq = theta * theta;
        let i4 = I4_PLANCK;
        let h_dc_1 = (-2.0_f64).exp() * (1.0 + 1.5 + 29.0 / 24.0 + 11.0 / 16.0 + 5.0 / 12.0);
        let rel_corr = 1.0 / (1.0 + 14.16 * theta);
        let hand = four_alpha_3pi * theta_sq * i4 * h_dc_1 * rel_corr;
        eprintln!("  Hand estimate = {hand:.4e}");
        eprintln!(
            "  Components: 4α/3π={four_alpha_3pi:.4e}, θ²={theta_sq:.4e}, \
                   I4={i4:.4}, H_dc(1)={h_dc_1:.4}, rel_corr={rel_corr:.6}"
        );

        // Must be positive
        assert!(k_dc > 0.0, "K_DC must be positive");

        // Order of magnitude: should be ~10^{-11}, not wildly different
        assert!(
            k_dc > 1e-14 && k_dc < 1e-7,
            "K_DC = {k_dc:.4e} outside physically reasonable range [1e-14, 1e-7]"
        );

        // The hand expression above is an independent transcription of CS2012
        // Eq. 13, so this is an identity, not an approximation: match to
        // round-off. (Was 10% before the R2 mutation audit — loose enough that
        // a 1.5× error in the DC normalisation passed.)
        let rel_err = (k_dc - hand).abs() / hand;
        assert!(
            rel_err < 1e-12,
            "K_DC = {k_dc:.6e} differs from hand estimate {hand:.6e} by {rel_err:.2e}"
        );
    }

    /// K_DC must be one implementation, not two (R2 mutation audit, fix A1).
    ///
    /// `dc_emission_coefficient` (used by `greens.rs` and `kompaneets.rs`) and
    /// `dc_prefactor` × `dc_high_freq_suppression` (the solver hot loop, which
    /// hoists the prefactor out of the grid loop) must agree exactly. They were
    /// separate copies of the same arithmetic until this audit; only the former
    /// was covered by the DC/BR literature anchor, so mutations to the latter —
    /// including one that scaled the DC rate by 1.53× — passed the whole suite.
    ///
    /// Mirrors the equivalent BR check (`test_br_fast_matches_reference`).
    #[test]
    fn test_dc_prefactor_matches_emission_coefficient() {
        for &z in &[1e4, 1e5, 1e6, 3e6, 1e7] {
            let theta = crate::constants::theta_z(z);
            let pre = dc_prefactor(theta);
            for &x in &[1e-3, 0.01, 0.1, 1.0, 5.0, 20.0] {
                let slow = dc_emission_coefficient(x, theta);
                let fast = dc_emission_coefficient_fast(x, pre);
                assert_eq!(
                    slow, fast,
                    "K_DC paths diverge at z={z:.0e}, x={x}: slow={slow:.17e}, fast={fast:.17e}"
                );
            }
        }
    }

    /// The DC relativistic correction (1 + 14.16 θ_z)⁻¹ must actually be applied.
    ///
    /// Every pre-audit Gaunt test passed `theta_z = 0.0`, where the correction is
    /// identically 1, so `/`→`*` and `+`→`-` inside it were unobservable. Probe
    /// it where it matters: at z = 10⁷, 14.16 θ_z ≈ 6.5%.
    ///
    /// Reference: Chluba, Sazonov & Sunyaev (2007), A&A 468, 785.
    #[test]
    fn test_dc_relativistic_correction_applied() {
        for &z in &[1e6, 3e6, 1e7] {
            let theta = crate::constants::theta_z(z);
            let expected = 1.0 / (1.0 + 14.16 * theta);
            assert!(expected < 1.0, "correction must suppress at z={z:.0e}");

            let corr = dc_relativistic_correction(theta);
            assert!(
                (corr - expected).abs() / expected < 1e-14,
                "dc_relativistic_correction({theta:.4e}) = {corr:.12}, expected {expected:.12}"
            );

            // It must reach both consumers: the Gaunt factor and the prefactor.
            // Ratios against the θ_z→0 limit isolate the correction exactly.
            let gaunt_ratio = dc_gaunt_factor(1.0, theta) / dc_gaunt_factor(1.0, 0.0);
            assert!(
                (gaunt_ratio - expected).abs() / expected < 1e-12,
                "dc_gaunt_factor at z={z:.0e} carries correction {gaunt_ratio:.12}, \
                 expected {expected:.12}"
            );

            // dc_prefactor ∝ θ_z² × correction, so divide the θ_z² scaling out.
            let pre_ratio = dc_prefactor(theta) / (DC_ALPHA_COEFF * theta * theta * I4_PLANCK);
            assert!(
                (pre_ratio - expected).abs() / expected < 1e-12,
                "dc_prefactor at z={z:.0e} carries correction {pre_ratio:.12}, \
                 expected {expected:.12}"
            );
        }

        // At z = 10⁷ the correction is a 6% effect — large enough that flipping
        // the sign or the division is a >12% error in the DC rate.
        let theta_hi = crate::constants::theta_z(1e7);
        assert!(
            dc_relativistic_correction(theta_hi) < 0.94,
            "correction at z=1e7 should be ≲0.94, got {}",
            dc_relativistic_correction(theta_hi)
        );
    }

    /// Absolute value of K_DC, derived by hand from CS2012 Eq. 13 rather than
    /// read off code output (CLAUDE.md pitfall #9).
    ///
    /// At z = 10⁶ (θ_z = 4.5971×10⁻⁴), x = 1:
    ///   4α/3π      = 3.09709×10⁻³
    ///   θ_z²       = 2.11332×10⁻⁷
    ///   I₄ = 4π⁴/15 = 25.975758
    ///   H_dc(1) = e⁻²(1 + 3/2 + 29/24 + 11/16 + 5/12) = 0.1353353 × 4.8125 = 0.651302
    ///   (1+14.16 θ_z)⁻¹ = 0.9935326
    ///   ⟹ K_DC = 1.1002×10⁻⁸
    ///
    /// This is the anchor on the DC *normalisation* that the suite lacked: the
    /// pre-audit tests constrained it only to within a factor ~1.5.
    #[test]
    fn test_dc_emission_coefficient_absolute_value_z1e6() {
        let theta = crate::constants::theta_z(1e6);
        let k_dc = dc_emission_coefficient(1.0, theta);
        let derived = 1.1002e-8;
        let rel_err = (k_dc - derived).abs() / derived;
        assert!(
            rel_err < 5e-4,
            "K_DC(x=1, z=1e6) = {k_dc:.6e}, hand-derived {derived:.6e}, \
             rel err {rel_err:.2e} (limit 5e-4)"
        );
    }

    /// Verify H_dc polynomial coefficients match Chluba & Sunyaev (2012) Eq. 13.
    ///
    /// H_dc^pl(x) = exp(-2x) [1 + 3x/2 + 29x²/24 + 11x³/16 + 5x⁴/12]
    ///
    /// The Horner-form code computes this as:
    ///   exp(-2x) * (1 + x*(1.5 + x*(29/24 + x*(11/16 + x*(5/12)))))
    ///
    /// We verify both forms agree and check specific values.
    #[test]
    fn test_dc_polynomial_coefficients_cs2012() {
        // Coefficients from CS2012 Eq. 13
        let c0 = 1.0;
        let c1 = 3.0 / 2.0;
        let c2 = 29.0 / 24.0;
        let c3 = 11.0 / 16.0;
        let c4 = 5.0 / 12.0;

        for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
            // Expanded form (independent computation)
            let poly = c0 + c1 * x + c2 * x * x + c3 * x * x * x + c4 * x * x * x * x;
            let h_expanded = (-2.0 * x).exp() * poly;

            // Code's Horner form
            let h_code = dc_high_freq_suppression(x);

            let rel_err = (h_code - h_expanded).abs() / h_expanded.max(1e-30);
            assert!(
                rel_err < 1e-14,
                "H_dc({x}): code={h_code:.6e}, expanded={h_expanded:.6e}, err={rel_err:.2e}"
            );
        }

        // Verify H_dc(0) = 1 exactly (all terms vanish except c0)
        let h0 = dc_high_freq_suppression(0.0);
        assert!((h0 - 1.0).abs() < 1e-15, "H_dc(0) = {h0}, expected 1.0");

        // Verify specific numerical values for regression
        // H_dc(0.5) = exp(-1) * (1 + 0.75 + 0.3021 + 0.0859 + 0.0260) = 0.3679 * 2.1640 ≈ 0.796
        let h_half = dc_high_freq_suppression(0.5);
        let expected_half = (-1.0_f64).exp()
            * (1.0 + 0.75 + 29.0 / 24.0 * 0.25 + 11.0 / 16.0 * 0.125 + 5.0 / 12.0 * 0.0625);
        assert!(
            (h_half - expected_half).abs() < 1e-14,
            "H_dc(0.5) = {h_half:.6e}, expected {expected_half:.6e}"
        );
    }
}
