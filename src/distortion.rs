//! Distortion extraction and characterization.
//!
//! Given the photon distortion Δn(x), extract the standard distortion
//! parameters (μ, y, temperature shift) and compute residuals.

use crate::constants::*;
use crate::spectrum::{
    bose_einstein, delta_n_over_n, delta_rho_over_rho, g_bb, mu_shape, planck, y_shape,
};

/// Default frequency band for Gram-Schmidt and B&F decompositions.
///
/// PIXIE-like window in dimensionless frequency at T₀ = 2.725 K:
/// x ∈ [0.5, 18] corresponds to ν ∈ [28, 1020] GHz. Matches the
/// experimental band used in CJ2014 Appendix A.
pub const DEFAULT_DECOMP_X_MIN: f64 = 0.5;
pub const DEFAULT_DECOMP_X_MAX: f64 = 18.0;

/// Complete distortion decomposition result.
#[derive(Debug, Clone)]
pub struct DistortionParams {
    /// Chemical potential μ
    pub mu: f64,
    /// Compton y-parameter
    pub y: f64,
    /// Temperature shift ΔT/T
    pub delta_t_over_t: f64,
    /// Fractional energy: Δρ/ρ
    pub delta_rho_over_rho: f64,
    /// Fractional photon number change: ΔN/N
    pub delta_n_over_n: f64,
    /// Residual distortion (not captured by μ, y, T)
    pub residual: Vec<f64>,
}

/// Collect trapezoidal weights and indices for grid points within [x_min, x_max].
fn band_weights(x_grid: &[f64], x_min: f64, x_max: f64) -> (Vec<usize>, Vec<f64>) {
    let n = x_grid.len();
    let mut idx = Vec::new();
    let mut w = Vec::new();
    for i in 0..n {
        if x_grid[i] < x_min || x_grid[i] > x_max {
            continue;
        }
        let dx = if n == 1 {
            1.0
        } else if i == 0 {
            x_grid[1] - x_grid[0]
        } else if i == n - 1 {
            x_grid[n - 1] - x_grid[n - 2]
        } else {
            0.5 * (x_grid[i + 1] - x_grid[i - 1])
        };
        idx.push(i);
        w.push(dx);
    }
    (idx, w)
}

/// CJ2014 Appendix A: Gram-Schmidt decomposition over a frequency band.
///
/// Reference: Chluba & Jeong (2014), arXiv:1306.5751, Appendix A.
///
/// Constructs an orthonormal basis (e_y, e_μ, e_T) for the three-dimensional
/// subspace spanned by (Y_SZ, M, G) via Gram-Schmidt in the order
///   1. e_y  = Y_SZ / |Y_SZ|
///   2. e_μ  = M⊥  / |M⊥|,   with M⊥  = M  − (M·e_y) e_y
///   3. e_T  = G⊥  / |G⊥|,   with G⊥  = G  − (G·e_y) e_y − (G·e_μ) e_μ
/// under the inner product ⟨a, b⟩ = ∫_{x_min}^{x_max} a(x) b(x) dx
/// (trapezoidal rule on the supplied grid). This generalises CJ2014's
/// uniform-channel flat sum to our non-uniform x-grid and reduces to it in
/// the continuum limit.
///
/// After projection, the coefficients (a_y, a_μ, a_T) = (⟨Δn, e_y⟩, ⟨Δn, e_μ⟩,
/// ⟨Δn, e_T⟩) are mapped back to (μ, y, ΔT/T) via exact back-substitution of
///   Δn ≈ μ M + y Y_SZ + (ΔT/T) G,
/// giving
///   ΔT/T = a_T / |G⊥|
///   μ    = (a_μ − ΔT/T · G_μ)           / |M⊥|
///   y    = (a_y − ΔT/T · G_y − μ · M_y) / |Y_SZ|
/// with M_y = M·e_y·|Y_SZ|, G_y = G·e_y·|Y_SZ|, G_μ = G·e_μ·|M⊥|.
pub fn decompose_gram_schmidt(
    x_grid: &[f64],
    delta_n: &[f64],
    x_min: f64,
    x_max: f64,
) -> DistortionParams {
    assert_eq!(
        x_grid.len(),
        delta_n.len(),
        "x_grid and delta_n length mismatch"
    );
    let n = x_grid.len();

    let drho_over_rho_val = delta_rho_over_rho(x_grid, delta_n);
    let dn_over_n_val = delta_n_over_n(x_grid, delta_n);

    let (idx, w) = band_weights(x_grid, x_min, x_max);
    let k = idx.len();

    let degenerate_return = || DistortionParams {
        mu: 0.0,
        y: 0.0,
        delta_t_over_t: 0.0,
        delta_rho_over_rho: drho_over_rho_val,
        delta_n_over_n: dn_over_n_val,
        residual: delta_n.to_vec(),
    };
    if k < 3 {
        return degenerate_return();
    }

    // Pre-evaluate shape vectors and distortion on the band.
    let m_vec: Vec<f64> = idx.iter().map(|&i| mu_shape(x_grid[i])).collect();
    let y_vec: Vec<f64> = idx.iter().map(|&i| y_shape(x_grid[i])).collect();
    let g_vec: Vec<f64> = idx.iter().map(|&i| g_bb(x_grid[i])).collect();
    let dn_vec: Vec<f64> = idx.iter().map(|&i| delta_n[i]).collect();

    let inner = |a: &[f64], b: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..k {
            s += a[i] * b[i] * w[i];
        }
        s
    };

    let y_norm2 = inner(&y_vec, &y_vec);
    if y_norm2 < 1e-100 {
        return degenerate_return();
    }
    let y_norm = y_norm2.sqrt();
    let e_y: Vec<f64> = y_vec.iter().map(|v| v / y_norm).collect();

    let m_y = inner(&m_vec, &e_y);
    let m_perp: Vec<f64> = m_vec
        .iter()
        .zip(e_y.iter())
        .map(|(mi, ei)| mi - m_y * ei)
        .collect();
    let m_perp_norm2 = inner(&m_perp, &m_perp);
    if m_perp_norm2 < 1e-100 {
        return degenerate_return();
    }
    let m_perp_norm = m_perp_norm2.sqrt();
    let e_mu: Vec<f64> = m_perp.iter().map(|v| v / m_perp_norm).collect();

    let g_y = inner(&g_vec, &e_y);
    let g_mu = inner(&g_vec, &e_mu);
    let g_perp: Vec<f64> = g_vec
        .iter()
        .zip(e_y.iter())
        .zip(e_mu.iter())
        .map(|((gi, ey), em)| gi - g_y * ey - g_mu * em)
        .collect();
    let g_perp_norm2 = inner(&g_perp, &g_perp);
    if g_perp_norm2 < 1e-100 {
        return degenerate_return();
    }
    let g_perp_norm = g_perp_norm2.sqrt();

    // Projections of Δn onto the orthonormal basis.
    let a_y = inner(&dn_vec, &e_y);
    let a_mu = inner(&dn_vec, &e_mu);
    let a_t = {
        let e_t: Vec<f64> = g_perp.iter().map(|v| v / g_perp_norm).collect();
        inner(&dn_vec, &e_t)
    };

    // Back-substitute. Using M = m_y·e_y + |M⊥|·e_μ and
    // G = g_y·e_y + g_μ·e_μ + |G⊥|·e_T in Δn = μ M + y Y_SZ + (ΔT/T) G:
    //   a_y = (ΔT/T)·g_y + μ·m_y + y·|Y_SZ|
    //   a_μ = (ΔT/T)·g_μ + μ·|M⊥|
    //   a_T = (ΔT/T)·|G⊥|
    let delta_t = a_t / g_perp_norm;
    let mu = (a_mu - delta_t * g_mu) / m_perp_norm;
    let y = (a_y - delta_t * g_y - mu * m_y) / y_norm;

    let mut residual = vec![0.0; n];
    for i in 0..n {
        let xx = x_grid[i];
        residual[i] = delta_n[i] - mu * mu_shape(xx) - y * y_shape(xx) - delta_t * g_bb(xx);
    }

    DistortionParams {
        mu,
        y,
        delta_t_over_t: delta_t,
        delta_rho_over_rho: drho_over_rho_val,
        delta_n_over_n: dn_over_n_val,
        residual,
    }
}

/// Bianchini & Fabbian (2022) nonlinear fit: μ inside the BE exponential.
///
/// Reference: Bianchini & Fabbian (2022), arXiv:2206.02762, Eqs. (1)–(4).
///
/// Model:
///   Δn_model(x; μ, δ, y) = [n_pl(x/(1+δ)) − n_pl(x)]
///                        + [n_BE(x+μ)    − n_pl(x)]
///                        + y · Y_SZ(x)
/// with δ ≡ ΔT/T₀. Fits (μ, δ, y) by Levenberg-Marquardt on the band
/// [x_min, x_max] with a trapezoidal inner product matching
/// `decompose_gram_schmidt`.
///
/// Initial guess: bootstrap from `decompose_gram_schmidt` (converted via
/// δ_BF = δ_GS + μ/β_μ). This gives the linearised optimum for free; the
/// LM iterations only refine the O(μ²) nonlinear correction.
///
/// In the small-(μ, δ, y) limit the model reduces to a linear fit of
///   Δn ≈ δ·G(x) + μ·(−G(x)/x) + y·Y_SZ(x),
/// which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since
/// M(x) = G(x)/β_μ − G(x)/x. The two methods therefore give the SAME μ and y
/// to O(μ²), but a DIFFERENT ΔT/T: a pure B&F BE distortion with chemical
/// potential μ_BF has ΔT/T = 0 in the B&F parameterisation and ΔT/T = −μ_BF/β_μ
/// in CJ2014. Concretely: δ_BF = δ_CJ + μ/β_μ.
pub fn decompose_nonlinear_be(
    x_grid: &[f64],
    delta_n: &[f64],
    x_min: f64,
    x_max: f64,
) -> DistortionParams {
    assert_eq!(
        x_grid.len(),
        delta_n.len(),
        "x_grid and delta_n length mismatch"
    );
    let n = x_grid.len();

    let drho_over_rho_val = delta_rho_over_rho(x_grid, delta_n);
    let dn_over_n_val = delta_n_over_n(x_grid, delta_n);

    let (idx, w) = band_weights(x_grid, x_min, x_max);
    let k = idx.len();
    if k < 3 {
        return DistortionParams {
            mu: 0.0,
            y: 0.0,
            delta_t_over_t: 0.0,
            delta_rho_over_rho: drho_over_rho_val,
            delta_n_over_n: dn_over_n_val,
            residual: delta_n.to_vec(),
        };
    }

    // B&F 2022 model: nonlinear in μ (BE chemical potential inside the
    // exponential) but linear in δ ≡ ΔT/T_0 (Taylor expansion of the
    // blackbody to first order in ΔT, as in their Eq. 1).
    let model_at = |xi: f64, mu: f64, delta: f64, y_par: f64| -> f64 {
        (bose_einstein(xi, mu) - planck(xi)) + delta * g_bb(xi) + y_par * y_shape(xi)
    };
    let chi2_at = |mu: f64, delta: f64, y_par: f64| -> f64 {
        let mut s = 0.0;
        for q in 0..k {
            let xi = x_grid[idx[q]];
            let r = delta_n[idx[q]] - model_at(xi, mu, delta, y_par);
            s += r * r * w[q];
        }
        s
    };

    // Bootstrap from GS (linearised answer, translated to B&F parameterisation).
    let gs = decompose_gram_schmidt(x_grid, delta_n, x_min, x_max);
    let mut mu = gs.mu;
    let mut delta = gs.delta_t_over_t + gs.mu / BETA_MU;
    let mut y_par = gs.y;

    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-12;
    let mut lambda = 1e-6_f64;
    let mut prev_chi2 = chi2_at(mu, delta, y_par);

    for _ in 0..MAX_ITER {
        let mut ata = [[0.0_f64; 3]; 3];
        let mut atr = [0.0_f64; 3];
        for q in 0..k {
            let xi = x_grid[idx[q]];
            let wi = w[q];
            let r = delta_n[idx[q]] - model_at(xi, mu, delta, y_par);

            // ∂ model / ∂μ  = d n_BE(x+μ)/dμ = − e^{x+μ}/(e^{x+μ}−1)²
            let xpm = xi + mu;
            let d_mu_j = if xpm.abs() < 1e-6 {
                -1.0 / (xpm * xpm)
            } else {
                let em = xpm.exp_m1();
                -(1.0 + em) / (em * em)
            };
            // ∂ model / ∂δ = G_bb(x)  (linear temperature-shift term)
            let d_delta_j = g_bb(xi);
            let d_y_j = y_shape(xi);

            let jac = [d_mu_j, d_delta_j, d_y_j];
            for a in 0..3 {
                atr[a] += jac[a] * r * wi;
                for b in 0..3 {
                    ata[a][b] += jac[a] * jac[b] * wi;
                }
            }
        }

        // LM step with backtracking: grow λ until step reduces χ², shrink on
        // acceptance. Caps after 20 tries to avoid infinite loops.
        let mut accepted = false;
        let mut step_mu = 0.0_f64;
        let mut step_d = 0.0_f64;
        let mut step_y = 0.0_f64;
        for _ls in 0..20 {
            let mut a = ata;
            for i in 0..3 {
                a[i][i] = ata[i][i] * (1.0 + lambda) + 1e-40;
            }
            let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
            let c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
            let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
            let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
            if det.abs() < 1e-50 {
                lambda *= 10.0;
                continue;
            }
            let c10 = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]);
            let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
            let c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]);
            let c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
            let c21 = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]);
            let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];

            step_mu = (c00 * atr[0] + c10 * atr[1] + c20 * atr[2]) / det;
            step_d = (c01 * atr[0] + c11 * atr[1] + c21 * atr[2]) / det;
            step_y = (c02 * atr[0] + c12 * atr[1] + c22 * atr[2]) / det;

            let mu_new = mu + step_mu;
            let delta_new = delta + step_d;
            let y_new = y_par + step_y;
            let chi2_new = chi2_at(mu_new, delta_new, y_new);
            if chi2_new < prev_chi2 {
                mu = mu_new;
                delta = delta_new;
                y_par = y_new;
                prev_chi2 = chi2_new;
                lambda = (lambda * 0.5).max(1e-10);
                accepted = true;
                break;
            }
            lambda *= 2.0;
        }
        if !accepted {
            break;
        }
        let step = step_mu.abs().max(step_d.abs()).max(step_y.abs());
        let scale = mu.abs().max(delta.abs()).max(y_par.abs()).max(1.0);
        if step < TOL * scale {
            break;
        }
    }

    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = delta_n[i] - model_at(x_grid[i], mu, delta, y_par);
    }

    DistortionParams {
        mu,
        y: y_par,
        delta_t_over_t: delta,
        delta_rho_over_rho: drho_over_rho_val,
        delta_n_over_n: dn_over_n_val,
        residual,
    }
}

/// Decompose a spectral distortion into μ, y, and temperature shift components.
///
/// Default method: Bianchini & Fabbian (2022) nonlinear fit on the band
/// [`DEFAULT_DECOMP_X_MIN`, `DEFAULT_DECOMP_X_MAX`] = [0.5, 18].
///
/// For the linear alternative (CJ2014 Appendix A Gram-Schmidt), call
/// [`decompose_gram_schmidt`] directly. The two methods agree on μ and y to
/// O(μ²) at realistic injection amplitudes (μ ≲ 10⁻³); they differ by a
/// parameterisation-only offset δ_BF = δ_GS + μ/β_μ in the extracted ΔT/T.
///
/// Note: B&F absorbs μ inside the Bose-Einstein exponential, so the returned
/// μ is the physical chemical potential (matching FIRAS-convention fits) —
/// NOT Chluba's orthogonalised "M-shape" μ. The relation is μ_BF = μ_M to
/// leading order; at μ ≳ 0.1 (rare in practice) the nonlinear BE shape
/// diverges from linear M(x) and the methods materially differ.
pub fn decompose_distortion(x_grid: &[f64], delta_n: &[f64]) -> DistortionParams {
    decompose_nonlinear_be(x_grid, delta_n, DEFAULT_DECOMP_X_MIN, DEFAULT_DECOMP_X_MAX)
}

/// Convenience wrapper: returns (mu, y, delta_t_over_t) tuple.
pub fn decompose(x_grid: &[f64], delta_n: &[f64]) -> (f64, f64, f64) {
    let params = decompose_distortion(x_grid, delta_n);
    (params.mu, params.y, params.delta_t_over_t)
}

/// Number of grid points falling inside the default μ/y decomposition
/// band [`DEFAULT_DECOMP_X_MIN`, `DEFAULT_DECOMP_X_MAX`].
///
/// `decompose_distortion` silently returns mu=y=0 when fewer than three
/// grid points fall in the band, which is the right behaviour for the
/// solver hot loop but a footgun for callers who set a custom (too-narrow)
/// `x_min`/`x_max`. Solvers should sample this once at startup and surface
/// a warning before running.
pub fn decomposition_band_count(x_grid: &[f64]) -> usize {
    let (idx, _w) = band_weights(x_grid, DEFAULT_DECOMP_X_MIN, DEFAULT_DECOMP_X_MAX);
    idx.len()
}

/// FIRAS 95% CL upper limits on spectral distortion parameters.
///
/// Reference: Fixsen et al. (1996), ApJ 473, 576
pub const FIRAS_MU_LIMIT: f64 = 9.0e-5;
pub const FIRAS_Y_LIMIT: f64 = 1.5e-5;

/// Check distortion parameters against FIRAS limits.
/// Returns (mu_fraction, y_fraction) as fraction of the FIRAS limit.
pub fn firas_check(params: &DistortionParams) -> (f64, f64) {
    (
        params.mu.abs() / FIRAS_MU_LIMIT,
        params.y.abs() / FIRAS_Y_LIMIT,
    )
}

/// Convert distortion Δn(x) to specific intensity ΔI_ν in MJy/sr.
///
/// ΔI_ν = (2hν³/c²) Δn(x) = (2hν³/c²) Δn(x)
///
/// where ν = x k_B T_0 / h.
pub fn delta_n_to_intensity_mjy(x: f64, delta_n: f64, t_cmb: f64) -> f64 {
    let nu = x * crate::constants::K_BOLTZMANN * t_cmb / crate::constants::HPLANCK;
    let prefactor = 2.0 * crate::constants::HPLANCK * nu.powi(3)
        / (crate::constants::C_LIGHT * crate::constants::C_LIGHT);
    // Convert W/m²/Hz/sr → MJy/sr (1 MJy = 10^{-20} W/m²/Hz)
    prefactor * delta_n * 1e20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_pure_mu() {
        // Create a pure μ-distortion and verify extraction.
        // The energy-neutral basis fit over [1, 15] should recover μ well.
        let n = 5000;
        let x_min = 0.01_f64;
        let x_max = 30.0_f64;
        let x_grid: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();
        let mu_true = 1e-5;
        let delta_n: Vec<f64> = x_grid.iter().map(|&x| mu_true * mu_shape(x)).collect();

        let params = decompose_distortion(&x_grid, &delta_n);
        let rel_err = (params.mu - mu_true).abs() / mu_true;
        assert!(
            rel_err < 0.01,
            "Extracted μ = {:.3e}, true = {:.3e}, rel_err = {rel_err}",
            params.mu,
            mu_true
        );
        // y contamination should be negligible for a pure μ input
        assert!(
            params.y.abs() < 0.01 * mu_true,
            "Pure μ input produced spurious y = {:.3e} (μ = {:.3e})",
            params.y,
            mu_true
        );
    }

    #[test]
    fn test_decompose_pure_y() {
        let n = 5000;
        let x_min = 0.01_f64;
        let x_max = 30.0_f64;
        let x_grid: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();
        let y_true = 1e-6;
        let delta_n: Vec<f64> = x_grid.iter().map(|&x| y_true * y_shape(x)).collect();

        let params = decompose_distortion(&x_grid, &delta_n);
        let rel_err = (params.y - y_true).abs() / y_true;
        assert!(
            rel_err < 0.01,
            "Extracted y = {:.3e}, true = {:.3e}, rel_err = {rel_err}",
            params.y,
            y_true
        );
        // μ contamination should be negligible for a pure y input
        assert!(
            params.mu.abs() < 0.01 * y_true,
            "Pure y input produced spurious μ = {:.3e} (y = {:.3e})",
            params.mu,
            y_true
        );
    }

    #[test]
    fn test_decompose_mixed_mu_y() {
        let n = 5000;
        let x_min = 0.01_f64;
        let x_max = 30.0_f64;
        let x_grid: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();

        let mu_true = 5e-6;
        let y_true = 2e-6;
        let delta_n: Vec<f64> = x_grid
            .iter()
            .map(|&x| mu_true * mu_shape(x) + y_true * y_shape(x))
            .collect();

        let params = decompose_distortion(&x_grid, &delta_n);
        let mu_err = (params.mu - mu_true).abs() / mu_true;
        let y_err = (params.y - y_true).abs() / y_true;
        assert!(mu_err < 0.01, "Mixed μ err: {mu_err:.4}");
        assert!(y_err < 0.01, "Mixed y err: {y_err:.4}");
    }

    #[test]
    fn test_decompose_pure_delta_t() {
        // A pure temperature-shift distortion Δn = (ΔT/T) × G(x)
        // should be recovered with μ ≈ 0, y ≈ 0, ΔT/T ≈ dt_true.
        //
        // From the decomposition step 3:
        //   ΔT/T = Δρ/ρ / 4 − μ/(4×1.401) − y
        // For a pure G(x) input: Δρ/ρ = 4×(ΔT/T), so ΔT/T is recovered exactly
        // from energy conservation, independent of the least-squares step.
        let n = 5000;
        let x_min = 0.01_f64;
        let x_max = 30.0_f64;
        let x_grid: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();
        let dt_true = 1e-6;
        let delta_n: Vec<f64> = x_grid.iter().map(|&x| dt_true * g_bb(x)).collect();

        let params = decompose_distortion(&x_grid, &delta_n);

        // Temperature shift should be recovered to < 1%
        let dt_err = (params.delta_t_over_t - dt_true).abs() / dt_true;
        assert!(
            dt_err < 0.01,
            "Extracted ΔT/T = {:.3e}, true = {:.3e}, rel_err = {dt_err:.4}",
            params.delta_t_over_t,
            dt_true
        );
        // μ and y contamination should be negligible
        assert!(
            params.mu.abs() < 0.01 * dt_true,
            "Pure ΔT/T produced spurious μ = {:.3e} (dt = {:.3e})",
            params.mu,
            dt_true
        );
        assert!(
            params.y.abs() < 0.01 * dt_true,
            "Pure ΔT/T produced spurious y = {:.3e} (dt = {:.3e})",
            params.y,
            dt_true
        );
    }

    // (test_decompose_convenience removed in 2026-04 triage: identical 5000-pt
    // pure-μ setup as test_decompose_pure_mu; the only difference is calling
    // `decompose()` (a 3-tuple convenience wrapper) instead of
    // `decompose_distortion()`. Trivial wrapper test.)

    #[test]
    fn test_firas_check_values() {
        let params = DistortionParams {
            mu: 4.5e-5, // half the FIRAS limit
            y: 7.5e-6,  // half the FIRAS limit
            delta_t_over_t: 0.0,
            delta_rho_over_rho: 0.0,
            delta_n_over_n: 0.0,
            residual: vec![],
        };
        let (mu_frac, y_frac) = firas_check(&params);
        assert!((mu_frac - 0.5).abs() < 1e-10, "mu_frac={mu_frac}");
        assert!((y_frac - 0.5).abs() < 1e-10, "y_frac={y_frac}");
    }

    #[test]
    fn test_delta_n_to_intensity_mjy() {
        // At x=1 with T_cmb=2.726K, verify intensity has correct sign and magnitude
        let t_cmb = 2.726;
        let delta_n = 1e-5;
        let intensity = delta_n_to_intensity_mjy(1.0, delta_n, t_cmb);
        assert!(
            intensity > 0.0,
            "Positive Δn should give positive intensity"
        );
        assert!(intensity.is_finite());

        // Linearity: 2× Δn → 2× intensity
        let intensity2 = delta_n_to_intensity_mjy(1.0, 2.0 * delta_n, t_cmb);
        assert!((intensity2 / intensity - 2.0).abs() < 1e-10);

        // Negative Δn → negative intensity
        let neg = delta_n_to_intensity_mjy(1.0, -delta_n, t_cmb);
        assert!(neg < 0.0);
    }

    // ============================================================================
    // CJ2014 Gram-Schmidt and Bianchini-Fabbian nonlinear BE decomposition
    // ============================================================================

    fn log_grid(n: usize, x_min: f64, x_max: f64) -> Vec<f64> {
        let lmin = x_min.ln();
        let lmax = x_max.ln();
        (0..n)
            .map(|i| (lmin + (lmax - lmin) * i as f64 / (n - 1) as f64).exp())
            .collect()
    }

    const X_LO: f64 = DEFAULT_DECOMP_X_MIN;
    const X_HI: f64 = DEFAULT_DECOMP_X_MAX;

    #[test]
    fn test_gram_schmidt_pure_mu() {
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let mu_true = 1e-5;
        let dn: Vec<f64> = x_grid.iter().map(|&x| mu_true * mu_shape(x)).collect();
        let p = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        assert!(
            (p.mu - mu_true).abs() / mu_true < 1e-4,
            "μ: got {:.6e}, expected {:.6e}",
            p.mu,
            mu_true
        );
        assert!(p.y.abs() < 1e-10 * mu_true.abs());
        // Pure Chluba-M has zero ΔT/T in the CJ2014 basis (by construction).
        assert!(
            p.delta_t_over_t.abs() < 1e-8 * mu_true.abs() + 1e-15,
            "ΔT/T = {:.3e}",
            p.delta_t_over_t
        );
    }

    #[test]
    fn test_gram_schmidt_pure_y() {
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let y_true = 1e-6;
        let dn: Vec<f64> = x_grid.iter().map(|&x| y_true * y_shape(x)).collect();
        let p = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        assert!((p.y - y_true).abs() / y_true < 1e-4, "y: got {:.6e}", p.y);
        assert!(p.mu.abs() < 1e-10 * y_true.abs());
        assert!(p.delta_t_over_t.abs() < 1e-10 * y_true.abs());
    }

    #[test]
    fn test_gram_schmidt_pure_delta_t() {
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let dt_true = 1e-6;
        let dn: Vec<f64> = x_grid.iter().map(|&x| dt_true * g_bb(x)).collect();
        let p = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        assert!(
            (p.delta_t_over_t - dt_true).abs() / dt_true < 1e-4,
            "ΔT/T: got {:.6e}",
            p.delta_t_over_t
        );
        assert!(p.mu.abs() < 1e-10 * dt_true.abs());
        assert!(p.y.abs() < 1e-10 * dt_true.abs());
    }

    #[test]
    fn test_bf_vs_gs_pure_mu() {
        // For pure Chluba-M (photon-number conserving), both methods should
        // give μ_BF = μ_GS = μ_true and ΔT/T_BF = μ/β_μ vs ΔT/T_GS = 0.
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let mu_true = 1e-5;
        let dn: Vec<f64> = x_grid.iter().map(|&x| mu_true * mu_shape(x)).collect();
        let gs = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        let bf = decompose_nonlinear_be(&x_grid, &dn, X_LO, X_HI);
        let rel_mu = (bf.mu - gs.mu).abs() / gs.mu.abs();
        assert!(
            rel_mu < 1e-4,
            "μ mismatch: BF={:.6e}, GS={:.6e}, rel={:.2e}",
            bf.mu,
            gs.mu,
            rel_mu
        );
        // y should be zero for pure Chluba-M; the nonlinear BE fit acquires
        // a spurious y at O(μ²) from the Taylor remainder of n_BE(x+μ).
        let y_tol = 10.0 * mu_true * mu_true;
        assert!(
            bf.y.abs() < y_tol && gs.y.abs() < y_tol,
            "y should be ≲ O(μ²)={:.1e}: BF={:.3e}, GS={:.3e}",
            y_tol,
            bf.y,
            gs.y
        );
        let predicted_offset = gs.mu / BETA_MU;
        let actual_offset = bf.delta_t_over_t - gs.delta_t_over_t;
        assert!(
            (actual_offset - predicted_offset).abs() / predicted_offset.abs() < 1e-3,
            "ΔT offset: predicted μ/β_μ = {:.6e}, observed = {:.6e}",
            predicted_offset,
            actual_offset
        );
    }

    #[test]
    fn test_bf_pure_bose_einstein() {
        // Inject a true Bose-Einstein distortion Δn = n_BE(x+μ_true) − n_pl(x).
        // B&F should recover μ_BF = μ_true and ΔT/T_BF ≈ 0.
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let mu_true = 2e-5;
        let dn: Vec<f64> = x_grid
            .iter()
            .map(|&x| bose_einstein(x, mu_true) - planck(x))
            .collect();
        let bf = decompose_nonlinear_be(&x_grid, &dn, X_LO, X_HI);
        assert!(
            (bf.mu - mu_true).abs() / mu_true < 1e-3,
            "B&F μ: got {:.6e}, expected {:.6e}",
            bf.mu,
            mu_true
        );
        assert!(
            bf.delta_t_over_t.abs() < 1e-4 * mu_true.abs(),
            "B&F ΔT/T = {:.3e} should be ≈ 0 for pure BE",
            bf.delta_t_over_t
        );
        // Gram-Schmidt on the SAME input should give the same μ but absorb
        // the photon-number-carrying part into ΔT/T = −μ/β_μ.
        let gs = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        let rel = (gs.mu - bf.mu).abs() / bf.mu.abs();
        assert!(
            rel < 1e-3,
            "GS vs BF μ: {} vs {}, rel={}",
            gs.mu,
            bf.mu,
            rel
        );
        let predicted = -bf.mu / BETA_MU;
        let rel_dt = (gs.delta_t_over_t - predicted).abs() / predicted.abs();
        assert!(
            rel_dt < 1e-3,
            "GS ΔT/T: got {:.6e}, predicted {:.6e}",
            gs.delta_t_over_t,
            predicted
        );
    }

    #[test]
    fn test_bf_vs_gs_greens_function_spectrum() {
        // Realistic mixed distortion from the analytic Green's function at
        // the μ-y crossover, where all three shapes are non-negligible.
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let z_h = 5e4;
        let drho = 1e-5;
        let dn: Vec<f64> = x_grid
            .iter()
            .map(|&x| drho * crate::greens::greens_function(x, z_h))
            .collect();
        let gs = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        let bf = decompose_nonlinear_be(&x_grid, &dn, X_LO, X_HI);

        eprintln!(
            "GF z_h={:.0e}: GS μ={:.3e} y={:.3e} dT={:.3e}  |  BF μ={:.3e} y={:.3e} dT={:.3e}",
            z_h, gs.mu, gs.y, gs.delta_t_over_t, bf.mu, bf.y, bf.delta_t_over_t
        );

        let rel_mu = (bf.mu - gs.mu).abs() / gs.mu.abs();
        let rel_y = (bf.y - gs.y).abs() / gs.y.abs();
        assert!(rel_mu < 1e-4, "μ match on GF spectrum: rel={:.2e}", rel_mu);
        assert!(rel_y < 1e-4, "y match on GF spectrum: rel={:.2e}", rel_y);
        // Predicted ΔT offset: δ_BF = δ_GS + μ/β_μ
        let predicted = gs.mu / BETA_MU;
        let observed = bf.delta_t_over_t - gs.delta_t_over_t;
        assert!(
            (observed - predicted).abs() / predicted.abs() < 1e-3,
            "ΔT offset: predicted {:.3e}, observed {:.3e}",
            predicted,
            observed
        );
    }

    #[test]
    fn test_bf_vs_gs_mixed() {
        let x_grid = log_grid(4000, 1e-3, 40.0);
        let mu_t = 3e-6;
        let y_t = 1e-6;
        let dt_t = 5e-7;
        let dn: Vec<f64> = x_grid
            .iter()
            .map(|&x| mu_t * mu_shape(x) + y_t * y_shape(x) + dt_t * g_bb(x))
            .collect();
        let gs = decompose_gram_schmidt(&x_grid, &dn, X_LO, X_HI);
        let bf = decompose_nonlinear_be(&x_grid, &dn, X_LO, X_HI);

        // μ and y agree to within O(μ²) nonlinearity.
        assert!((gs.mu - mu_t).abs() / mu_t < 1e-4, "GS μ");
        assert!((gs.y - y_t).abs() / y_t < 1e-4, "GS y");
        assert!((bf.mu - mu_t).abs() / mu_t < 1e-3, "BF μ");
        assert!((bf.y - y_t).abs() / y_t < 1e-3, "BF y");
        let rel_mu = (bf.mu - gs.mu).abs() / gs.mu.abs();
        let rel_y = (bf.y - gs.y).abs() / gs.y.abs();
        assert!(rel_mu < 1e-4, "μ match: rel={:.2e}", rel_mu);
        assert!(rel_y < 1e-4, "y match: rel={:.2e}", rel_y);

        // ΔT/T offset: δ_BF = δ_GS + μ/β_μ
        let predicted_offset = gs.mu / BETA_MU;
        let observed_offset = bf.delta_t_over_t - gs.delta_t_over_t;
        assert!(
            (observed_offset - predicted_offset).abs() / predicted_offset.abs() < 1e-3,
            "ΔT offset: predicted {:.3e}, observed {:.3e}",
            predicted_offset,
            observed_offset
        );
    }
}
