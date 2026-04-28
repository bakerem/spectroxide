//! Spectral shapes and distributions for CMB spectral distortions.
//!
//! All functions here take the dimensionless frequency `x = hν/(kT_z)` and
//! return an occupation-number perturbation `Δn`. They are normalized so
//! that the corresponding distortion amplitude (μ, y, ΔT/T) is the
//! coefficient that multiplies the shape — e.g. `Δn_μ(x) = μ · M(x)`,
//! `Δn_y(x) = y · Y_SZ(x)`, `Δn_T(x) = (ΔT/T) · G_bb(x)`.
//!
//! Provided shapes:
//! - [`planck`] — equilibrium blackbody `n_pl(x) = 1/(e^x − 1)`
//!   (small-`x` and large-`x` branches preserve precision).
//! - [`bose_einstein`] — `1/(e^{x+μ} − 1)`, the chemical-potential family.
//! - `mu_shape` — μ-distortion shape `M(x)` (Sunyaev-Zel'dovich 1970).
//! - `y_shape` — y-distortion shape `Y_SZ(x)` (Zeldovich-Sunyaev 1969).
//! - `temperature_shift_shape` — `G_bb(x) = x e^x / (e^x − 1)²`,
//!   the temperature-shift mode `δT/T`.
//!
//! Conventions match those used by [`crate::greens`] and the Python
//! `spectroxide.greens` module so PDE and Green's-function results can be
//! decomposed against the same basis.

use crate::constants::*;

/// Planck (blackbody) occupation number: n_pl(x) = 1/(e^x - 1)
#[inline]
pub fn planck(x: f64) -> f64 {
    if x < 1e-6 {
        // Taylor expansion for small x: 1/x - 1/2 + x/12 - ...
        1.0 / x - 0.5 + x / 12.0
    } else if x > 500.0 {
        (-x).exp()
    } else {
        // expm1 preserves precision near x = 0 where (e^x - 1) loses digits
        // (cf. audit L4 in infra).
        1.0 / x.exp_m1()
    }
}

/// Bose-Einstein distribution: n_BE(x, μ) = 1/(e^(x+μ) - 1)
#[inline]
pub fn bose_einstein(x: f64, mu: f64) -> f64 {
    let y = x + mu;
    if y < 1e-6 {
        1.0 / y - 0.5 + y / 12.0
    } else if y > 500.0 {
        (-y).exp()
    } else {
        1.0 / y.exp_m1()
    }
}

/// Blackbody derivative: G_bb(x) = x e^x / (e^x - 1)²
/// This is -x ∂n_pl/∂x = x²/(4T) ∂B_ν/∂T normalized
#[inline]
pub fn g_bb(x: f64) -> f64 {
    if x < 1e-6 {
        1.0 / x - 1.0 / 12.0 * x
    } else if x > 500.0 {
        x * (-x).exp()
    } else {
        // G_bb(x) = x e^x / (e^x - 1)² = x / (e^x - 1) × e^x / (e^x - 1)
        //        = x · n_pl · (1 + n_pl), but here we take the direct form.
        let em1 = x.exp_m1();
        x * (1.0 + em1) / (em1 * em1)
    }
}

/// μ-distortion spectral shape: M(x) = (x/β_μ - 1) · e^x / (e^x - 1)²
///
/// The chemical potential distortion; crosses zero at x = β_μ ≈ 2.19.
/// Normalized such that μ = 1.401 × (Δρ/ρ) for energy injection in μ-era.
/// Reference: Chluba (2013), Eq. (5b).
#[inline]
pub fn mu_shape(x: f64) -> f64 {
    (x / BETA_MU - 1.0) * g_bb(x) / x
}

/// Y-distortion (Sunyaev-Zeldovich) spectral shape:
/// Y_SZ(x) = G_bb(x) · [x·coth(x/2) - 4]
///
/// This is the classic SZ spectral function; crosses zero at x ≈ 3.83.
/// Reference: Zeldovich & Sunyaev (1969).
#[inline]
pub fn y_shape(x: f64) -> f64 {
    if x < 1e-6 {
        // Small-x expansion: G_bb ≈ 1/x - x/12, x·coth(x/2) - 4 ≈ -2 + x²/6.
        // Product: (1/x)(-2) + (1/x)(x²/6) + (-x/12)(-2) = -2/x + x/6 + x/6 = -2/x + x/3.
        -2.0 / x + x / 3.0
    } else {
        let coth_half = (x / 2.0).cosh() / (x / 2.0).sinh();
        g_bb(x) * (x * coth_half - 4.0)
    }
}

/// Numerical integral of x^n * n_pl(x) over [0, x_max] using the trapezoidal rule
/// on a logarithmic grid. Used for validation against analytic G_n values.
pub fn spectral_integral(n: i32, x_min: f64, x_max: f64, num_points: usize) -> f64 {
    // Use log-spaced grid for better accuracy at low x
    let log_min = x_min.ln();
    let log_max = x_max.ln();
    let dlog = (log_max - log_min) / (num_points as f64 - 1.0);

    let mut result = 0.0;
    let mut prev_f = 0.0;
    let mut prev_x = 0.0;

    for i in 0..num_points {
        let log_x = log_min + i as f64 * dlog;
        let x = log_x.exp();
        let f = x.powi(n) * planck(x);

        if i > 0 {
            result += 0.5 * (prev_f + f) * (x - prev_x);
        }
        prev_f = f;
        prev_x = x;
    }
    result
}

/// Compute the Compton equilibrium temperature ratio T_e^eq / T_z
/// from the photon spectrum.
///
/// T_e^eq = (I₄ / (4 G₃)) T_z where I₄ = ∫x⁴ n(1+n) dx, G₃ = ∫x³ n dx.
/// For a Planck spectrum, T_e^eq = T_z exactly.
pub fn compton_equilibrium_ratio(x_grid: &[f64], n: &[f64]) -> f64 {
    let mut g3 = 0.0;
    let mut i4 = 0.0;

    for i in 1..x_grid.len() {
        let dx = x_grid[i] - x_grid[i - 1];
        let x_mid = 0.5 * (x_grid[i] + x_grid[i - 1]);
        let n_mid = 0.5 * (n[i] + n[i - 1]);

        let n_l = n[i - 1];
        let n_r = n[i];
        let nn1_mid = 0.5 * (n_l * (1.0 + n_l) + n_r * (1.0 + n_r));
        g3 += x_mid.powi(3) * n_mid * dx;
        i4 += x_mid.powi(4) * nn1_mid * dx;
    }

    if g3.abs() < 1e-300 {
        return 1.0;
    }
    i4 / (4.0 * g3)
}

/// Trapezoidal integral of x^power × Δn over the grid, divided by norm.
fn weighted_integral(x_grid: &[f64], delta_n: &[f64], power: i32, norm: f64) -> f64 {
    let mut integral = 0.0;
    for i in 1..x_grid.len() {
        let dx = x_grid[i] - x_grid[i - 1];
        let x_mid = 0.5 * (x_grid[i] + x_grid[i - 1]);
        let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
        integral += x_mid.powi(power) * dn_mid * dx;
    }
    integral / norm
}

/// Compute fractional energy in distortion: Δρ/ρ = ∫x³ Δn dx / G₃
pub fn delta_rho_over_rho(x_grid: &[f64], delta_n: &[f64]) -> f64 {
    weighted_integral(x_grid, delta_n, 3, G3_PLANCK)
}

/// Compute fractional photon number change: ΔN/N = ∫x² Δn dx / G₂
pub fn delta_n_over_n(x_grid: &[f64], delta_n: &[f64]) -> f64 {
    weighted_integral(x_grid, delta_n, 2, G2_PLANCK)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Planck identity: dn_pl/dx + n_pl(1 + n_pl) = 0 exactly.
    ///
    /// This cancellation underpins the Kompaneets flux-split (CLAUDE.md
    /// pitfall #1). Using finite differences would introduce O(dx²) ≈ 3e-3
    /// error — ~1000× the physical signal. The identity must hold to machine
    /// precision at every x.
    ///
    /// Derivation: n_pl = 1/(eˣ-1) ⇒ dn_pl/dx = -eˣ/(eˣ-1)².
    /// n_pl(1+n_pl) = [1/(eˣ-1)]·[eˣ/(eˣ-1)] = eˣ/(eˣ-1)². Sum = 0.
    #[test]
    fn test_planck_identity_analytical() {
        for &x in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let n = planck(x);
            let ex = x.exp();
            // Analytical dn_pl/dx
            let dn_dx = -ex / (ex - 1.0).powi(2);
            let rhs = n * (1.0 + n);
            // The identity: dn_dx + rhs = 0
            let residual = dn_dx + rhs;
            // Relative error against the magnitude of either term.
            // Threshold 1e-13: comfortably below the O(dx²)~3e-3 error that a
            // finite-difference implementation would produce, while allowing
            // for ~few ulps from the two exp() evaluations.
            let rel_err = residual.abs() / rhs.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel_err < 1e-13,
                "Planck identity violated at x={x}: dn_dx={dn_dx:.6e}, \
                 n(1+n)={rhs:.6e}, residual={residual:.6e}, rel_err={rel_err:.3e}"
            );
        }
    }

    #[test]
    fn test_planck_low_x() {
        // n_pl(x) ≈ 1/x for x → 0
        let x = 1e-8;
        let n = planck(x);
        assert!((n - 1.0 / x).abs() / (1.0 / x) < 1e-6);
    }

    #[test]
    fn test_planck_moderate_x() {
        let x = 1.0;
        let expected = 1.0 / (1.0_f64.exp() - 1.0);
        assert!((planck(x) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_bose_einstein_reduces_to_planck() {
        let x = 3.0;
        assert!((bose_einstein(x, 0.0) - planck(x)).abs() < 1e-14);
    }

    #[test]
    fn test_mu_shape_sign_change_at_beta_mu() {
        // M(β_μ) = 0 is algebraic (factor (x/β_μ − 1) in the definition), so
        // asserting |M(β_μ)| < tol is tautological. Test the *physical* claim
        // instead: M changes sign across β_μ (negative on the low-x side,
        // positive on the high-x side). Reference: Chluba & Sunyaev 2012,
        // MNRAS 419, 1294.
        let eps = 1e-3;
        let m_below = mu_shape(BETA_MU - eps);
        let m_above = mu_shape(BETA_MU + eps);
        assert!(
            m_below < 0.0 && m_above > 0.0,
            "M should flip sign at β_μ: M({:.4})={m_below:.3e}, M({:.4})={m_above:.3e}",
            BETA_MU - eps,
            BETA_MU + eps
        );
    }

    #[test]
    fn test_y_shape_zero_crossing() {
        // Y_SZ crosses zero at x ≈ 3.83
        // Find it by bisection
        let mut x_lo = 3.5_f64;
        let mut x_hi = 4.2_f64;
        for _ in 0..100 {
            let x_mid = (x_lo + x_hi) / 2.0;
            if y_shape(x_mid) > 0.0 {
                x_hi = x_mid;
            } else {
                x_lo = x_mid;
            }
        }
        let x_zero = (x_lo + x_hi) / 2.0;
        assert!(
            (x_zero - 3.83).abs() < 0.01,
            "Y_SZ zero at x = {x_zero}, expected ~3.83"
        );
    }

    #[test]
    fn test_spectral_integral_g3() {
        // ∫x³ n_pl dx = π⁴/15
        let g3 = spectral_integral(3, 1e-6, 80.0, 100_000);
        let expected = std::f64::consts::PI.powi(4) / 15.0;
        let rel_err = (g3 - expected).abs() / expected;
        assert!(
            rel_err < 1e-6,
            "G₃ = {g3}, expected {expected}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn test_spectral_integral_g2() {
        // ∫x² n_pl dx = 2ζ(3)
        let g2 = spectral_integral(2, 1e-6, 80.0, 100_000);
        let expected = G2_PLANCK;
        let rel_err = (g2 - expected).abs() / expected;
        assert!(
            rel_err < 1e-5,
            "G₂ = {g2}, expected {expected}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn test_g_bb_all_branches() {
        // Low-x branch: x < 1e-6
        let x_low = 1e-8;
        let g_low = g_bb(x_low);
        assert!(
            (g_low - 1.0 / x_low).abs() / (1.0 / x_low) < 1e-2,
            "g_bb({x_low}) = {g_low}, expected ~1/x"
        );

        // High-x branch: x > 500
        let x_high = 600.0;
        let g_high = g_bb(x_high);
        let expected = x_high * (-x_high).exp();
        assert!(
            (g_high - expected).abs() < 1e-250,
            "g_bb({x_high}) = {g_high}, expected {expected}"
        );

        // Normal branch: 1e-6 < x < 500
        let x_mid = 3.0;
        let g_mid = g_bb(x_mid);
        let ex = x_mid.exp();
        let expected_mid = x_mid * ex / (ex - 1.0).powi(2);
        assert!(
            (g_mid - expected_mid).abs() < 1e-14,
            "g_bb({x_mid}) = {g_mid}, expected {expected_mid}"
        );
    }

    #[test]
    fn test_planck_high_x_branch() {
        let x = 600.0;
        let n = planck(x);
        assert!((n - (-x).exp()).abs() < 1e-250, "planck({x}) = {n}");
    }

    #[test]
    fn test_bose_einstein_branches() {
        // Low-x branch
        let x_low = 1e-8;
        let mu = 0.0;
        let n = bose_einstein(x_low, mu);
        assert!((n - planck(x_low)).abs() < 1e-6, "BE ≈ Planck at mu=0");

        // High-x branch
        let n_high = bose_einstein(500.0, 1.0);
        assert!((n_high - (-501.0_f64).exp()).abs() < 1e-200);
    }

    #[test]
    fn test_delta_rho_over_rho_and_delta_n_over_n() {
        // For a known μ-distortion, Δρ/ρ = μ/1.401
        let n_pts = 5000;
        let x_min: f64 = 1e-4;
        let x_max: f64 = 50.0;
        let log_min = x_min.ln();
        let log_max = x_max.ln();
        let x_grid: Vec<f64> = (0..n_pts)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n_pts - 1) as f64).exp())
            .collect();

        // Pure temperature shift: Δn = ΔT/T × G_bb
        let dt_over_t = 1e-5;
        let dn: Vec<f64> = x_grid.iter().map(|&x| dt_over_t * g_bb(x)).collect();

        let drho = delta_rho_over_rho(&x_grid, &dn);
        // For G_bb: ∫x³ G_bb dx = 4G₃, so Δρ/ρ = 4 ΔT/T
        assert!(
            (drho - 4.0 * dt_over_t).abs() / (4.0 * dt_over_t) < 0.01,
            "Δρ/ρ = {drho:.4e}, expected {:.4e}",
            4.0 * dt_over_t
        );

        let dn_n = delta_n_over_n(&x_grid, &dn);
        // For G_bb: ∫x² G_bb dx = 3G₂, so ΔN/N = 3 ΔT/T
        assert!(
            (dn_n - 3.0 * dt_over_t).abs() / (3.0 * dt_over_t) < 0.01,
            "ΔN/N = {dn_n:.4e}, expected {:.4e}",
            3.0 * dt_over_t
        );
    }

    #[test]
    fn test_compton_equilibrium_planck() {
        // For Planck spectrum, T_e^eq / T_z = 1
        let n_points = 5000;
        let x_min = 1e-4_f64;
        let x_max = 50.0_f64;
        let log_min = x_min.ln();
        let log_max = x_max.ln();
        let x_grid: Vec<f64> = (0..n_points)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n_points - 1) as f64).exp())
            .collect();
        let n_vals: Vec<f64> = x_grid.iter().map(|&x| planck(x)).collect();

        let ratio = compton_equilibrium_ratio(&x_grid, &n_vals);
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "T_e^eq/T_z = {ratio}, expected 1.0 for Planck"
        );
    }
}
