//! Distortion extraction and characterization.
//!
//! Given the photon distortion Δn(x), extract the standard distortion
//! parameters (μ, y, temperature shift) and compute residuals. Includes
//! helpers for the Primordial Inflation Explorer (PIXIE) decomposition band
//! and the Far Infrared Absolute Spectrophotometer (FIRAS) limits on μ and y.

use crate::constants::*;
use crate::spectrum::{
    bose_einstein, delta_n_over_n, delta_rho_over_rho, g_bb, mu_shape, planck, y_shape,
};

/// Default frequency band for Gram-Schmidt and B&F decompositions.
///
/// Lower edge of the PIXIE-like decomposition window in dimensionless
/// frequency at T₀ = 2.725 K: x ∈ [0.5, 18] corresponds to
/// ν ∈ [28, 1020] GHz, the experimental band used in CJ2014 Appendix A.
pub const DEFAULT_DECOMP_X_MIN: f64 = 0.5;
/// Upper edge of the PIXIE-like decomposition window; see
/// [`DEFAULT_DECOMP_X_MIN`].
pub const DEFAULT_DECOMP_X_MAX: f64 = 18.0;

/// Complete distortion decomposition result.
#[derive(Debug, Clone)]
pub struct DistortionParams {
    /// Chemical potential μ.
    pub mu: f64,
    /// Compton y-parameter.
    pub y: f64,
    /// Temperature shift ΔT/T.
    pub delta_t_over_t: f64,
    /// Fractional energy: Δρ/ρ.
    pub delta_rho_over_rho: f64,
    /// Fractional photon number change: ΔN/N.
    pub delta_n_over_n: f64,
    /// Residual distortion (not captured by μ, y, T).
    pub residual: Vec<f64>,
}

/// Collects intensity weights and indices for grid points within [x_min, x_max].
///
/// Each weight is x⁶ times the trapezoid width dx, so a weighted sum of squared
/// Δn residuals is the squared intensity residual ∫ [x³(Δn − model)]² dx.
/// Chluba & Jeong (2014) Appendix A project intensity in channels uniform in ν,
/// and FIRAS measures intensity; ADR 0006 records the choice.
///
/// Precondition: the supplied grid should extend beyond [x_min, x_max] on both
/// sides. The half-weight rule at the ends keys off the *parent array's* edges,
/// so a grid pre-trimmed to exactly the band would get full (double) weight at
/// the band edges. All current call sites pass grids spanning past [0.5, 18].
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
        w.push(dx * x_grid[i].powi(6));
    }
    (idx, w)
}

/// Performs the CJ2014 Appendix A Gram-Schmidt decomposition over a frequency band.
///
/// Reference: Chluba & Jeong (2014), arXiv:1306.5751, Appendix A.
///
/// Constructs an orthonormal basis (e_y, e_μ, e_T) for the three-dimensional
/// subspace spanned by (Y_SZ, M, G) using Gram-Schmidt in the order
///   1. e_y  = Y_SZ / |Y_SZ|
///   2. e_μ  = M⊥  / |M⊥|,   with M⊥  = M  − (M·e_y) e_y
///   3. e_T  = G⊥  / |G⊥|,   with G⊥  = G  − (G·e_y) e_y − (G·e_μ) e_μ
/// under the intensity inner product ⟨a, b⟩ = ∫_{x_min}^{x_max} x⁶ a(x) b(x) dx
/// (trapezoidal rule on the supplied grid). CJ2014 build their vectors from
/// intensities ΔI ∝ x³Δn in channels uniform in ν and sum them without weight;
/// this integral is the continuum limit of that sum on the solver's non-uniform
/// grid. With these shapes it reproduces their basis norms {|Y_SZ|, |M⊥|, |G⊥|}
/// to 0.3% (ADR 0006).
///
/// After projection, the coefficients (a_y, a_μ, a_T) = (⟨Δn, e_y⟩, ⟨Δn, e_μ⟩,
/// ⟨Δn, e_T⟩) are mapped back to (μ, y, ΔT/T) through exact back-substitution of
///   Δn ≈ μ M + y Y_SZ + (ΔT/T) G,
/// giving
///   ΔT/T = a_T / |G⊥|
///   μ    = (a_μ − ΔT/T · g_μ)           / |M⊥|
///   y    = (a_y − ΔT/T · g_y − μ · m_y) / |Y_SZ|
/// with the plain projections m_y = ⟨M, e_y⟩, g_y = ⟨G, e_y⟩, g_μ = ⟨G, e_μ⟩.
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

/// Fits the Bianchini & Fabbian (2022) nonlinear model, with μ inside the BE exponential.
///
/// Reference: Bianchini & Fabbian (2022), arXiv:2206.02762, Eqs. (1)–(4).
///
/// Model:
///   Δn_model(x; μ, δ, y) = δ · G_bb(x)
///                        + [n_BE(x+μ)    − n_pl(x)]
///                        + y · Y_SZ(x)
/// with δ ≡ ΔT/T₀, nonlinear in μ (inside the Bose-Einstein exponential) but
/// linear in δ (first-order Taylor expansion of the blackbody, as in their
/// Eq. 1). Fits (μ, δ, y) by Levenberg-Marquardt on the band
/// [x_min, x_max] with the intensity inner product of
/// `decompose_gram_schmidt`, i.e. it minimizes ∫ [x³(Δn − model)]² dx (ADR 0006).
///
/// Initial guess: bootstrap from `decompose_gram_schmidt` (converted using
/// δ_BF = δ_GS + μ/β_μ). This gives the linearized optimum for free; the
/// LM iterations only refine the O(μ²) nonlinear correction.
///
/// In the small-(μ, δ, y) limit the model reduces to a linear fit of
///   Δn ≈ δ·G(x) + μ·(−G(x)/x) + y·Y_SZ(x),
/// which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since
/// M(x) = G(x)/β_μ − G(x)/x. The two methods therefore give the same μ and y
/// to O(μ²), but a different ΔT/T: a pure B&F BE distortion with chemical
/// potential μ_BF has ΔT/T = 0 in the B&F parameterization and ΔT/T = −μ_BF/β_μ
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

/// Decomposes a spectral distortion into μ, y, and temperature shift components.
///
/// Default method: Bianchini & Fabbian (2022) nonlinear fit on the band
/// [`DEFAULT_DECOMP_X_MIN`, `DEFAULT_DECOMP_X_MAX`] = [0.5, 18], weighted by
/// intensity: it minimizes ∫ [x³(Δn − model)]² dx (ADR 0006).
///
/// For the linear alternative (CJ2014 Appendix A Gram-Schmidt), call
/// [`decompose_gram_schmidt`] directly. The two methods agree on μ and y to
/// O(μ²) at realistic injection amplitudes (μ ≲ 10⁻³); they differ by a
/// parameterization-only offset δ_BF = δ_GS + μ/β_μ in the extracted ΔT/T.
///
/// Note: B&F absorbs μ inside the Bose-Einstein exponential, so the returned
/// μ is the physical chemical potential (matching FIRAS-convention fits) —
/// not Chluba's orthogonalized "M-shape" μ. The relation is μ_BF = μ_M to
/// leading order; at μ ≳ 0.1 (rare in practice) the nonlinear BE shape
/// diverges from linear M(x) and the methods materially differ.
///
/// Domain of validity: for spectra with support outside span{M, Y_SZ, G_bb} —
/// for example, frozen or locked-in photon-injection bumps from z < 1100 that never
/// Comptonized — the returned (μ, y, ΔT/T) is the in-band intensity-weighted best fit, not a
/// physical decomposition. Inspect `residual` before interpreting the triple
/// in that regime.
pub fn decompose_distortion(x_grid: &[f64], delta_n: &[f64]) -> DistortionParams {
    decompose_nonlinear_be(x_grid, delta_n, DEFAULT_DECOMP_X_MIN, DEFAULT_DECOMP_X_MAX)
}

/// Decomposes Δn into μ and y after removing the temperature shift by photon-number
/// conservation.
///
/// The temperature shift is fixed, not fitted: ΔT/T = ∫x²Δn dx / ∫x²G dx over the
/// whole grid, the same number-conserving split the solver and CosmoTherm use
/// (Chluba & Sunyaev 2012). That G component is removed from Δn and from the μ
/// shape −G/x and Y_SZ; μ and y then come from a linear least-squares fit to the
/// stripped spectrum with the intensity weight of [`decompose_distortion`],
/// ∫ [x³(Δn − model)]² dx on [`DEFAULT_DECOMP_X_MIN`, `DEFAULT_DECOMP_X_MAX`].
///
/// This tracks the visibility functions of Chluba (2013), whose M and Y_SZ carry
/// no photon number, so their temperature term holds all of it: on the 118
/// Table 1 bursts μ/[(3/κ_c)Δρ/ρ] agrees with J_bb*·J_μ to 0.031 (0.012 for
/// z_h ≥ 3×10⁵) and 4y/(Δρ/ρ) with J_y to 0.006. [`decompose_distortion`], which
/// fits ΔT/T freely, gives J_μ up to 1.25 there, as in Chluba & Jeong (2014)
/// Fig. 1. Use this function when comparing against visibility-function targets
/// (ADR 0006).
///
/// μ is the amplitude of the linearized Bose–Einstein shape −G/x, which equals the
/// nonlinear μ of [`decompose_nonlinear_be`] to O(μ²). The returned
/// `delta_t_over_t` is the number-conserving shift.
pub fn decompose_number_conserving(x_grid: &[f64], delta_n: &[f64]) -> DistortionParams {
    assert_eq!(
        x_grid.len(),
        delta_n.len(),
        "x_grid and delta_n length mismatch"
    );
    let n = x_grid.len();
    let drho_over_rho_val = delta_rho_over_rho(x_grid, delta_n);
    let dn_over_n_val = delta_n_over_n(x_grid, delta_n);

    let g: Vec<f64> = x_grid.iter().map(|&x| g_bb(x)).collect();
    let m: Vec<f64> = x_grid.iter().zip(&g).map(|(&x, &gi)| -gi / x).collect();
    let yv: Vec<f64> = x_grid.iter().map(|&x| y_shape(x)).collect();
    // Photon-number coefficient of each vector on G_bb, with one quadrature for all.
    let g_number = delta_n_over_n(x_grid, &g);
    let number_coeff = |v: &[f64]| delta_n_over_n(x_grid, v) / g_number;
    let (a_dn, a_m, a_y) = (number_coeff(delta_n), number_coeff(&m), number_coeff(&yv));

    let (idx, w) = band_weights(x_grid, DEFAULT_DECOMP_X_MIN, DEFAULT_DECOMP_X_MAX);
    if idx.len() < 3 {
        return DistortionParams {
            mu: 0.0,
            y: 0.0,
            delta_t_over_t: a_dn,
            delta_rho_over_rho: drho_over_rho_val,
            delta_n_over_n: dn_over_n_val,
            residual: delta_n.to_vec(),
        };
    }
    // Normal equations for the stripped spectrum on the stripped shapes.
    let (mut mm, mut my, mut yy, mut md, mut yd) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (q, &i) in idx.iter().enumerate() {
        let (ms, ys, ds) = (
            m[i] - a_m * g[i],
            yv[i] - a_y * g[i],
            delta_n[i] - a_dn * g[i],
        );
        mm += w[q] * ms * ms;
        my += w[q] * ms * ys;
        yy += w[q] * ys * ys;
        md += w[q] * ms * ds;
        yd += w[q] * ys * ds;
    }
    let det = mm * yy - my * my;
    let mu = (md * yy - yd * my) / det;
    let y = (yd * mm - md * my) / det;

    let residual = (0..n)
        .map(|i| delta_n[i] - a_dn * g[i] - mu * (m[i] - a_m * g[i]) - y * (yv[i] - a_y * g[i]))
        .collect();
    DistortionParams {
        mu,
        y,
        delta_t_over_t: a_dn,
        delta_rho_over_rho: drho_over_rho_val,
        delta_n_over_n: dn_over_n_val,
        residual,
    }
}

/// Returns the (mu, y, delta_t_over_t) tuple as a convenience wrapper.
pub fn decompose(x_grid: &[f64], delta_n: &[f64]) -> (f64, f64, f64) {
    let params = decompose_distortion(x_grid, delta_n);
    (params.mu, params.y, params.delta_t_over_t)
}

/// Returns the number of grid points falling inside the default μ/y decomposition
/// band [`DEFAULT_DECOMP_X_MIN`, `DEFAULT_DECOMP_X_MAX`].
///
/// `decompose_distortion` silently returns mu=y=0 when fewer than three
/// grid points fall in the band, which is the right behavior for the
/// solver hot loop but a footgun for callers who set a custom (too-narrow)
/// `x_min`/`x_max`. Solvers should sample this once at startup and surface
/// a warning before running.
pub fn decomposition_band_count(x_grid: &[f64]) -> usize {
    let (idx, _w) = band_weights(x_grid, DEFAULT_DECOMP_X_MIN, DEFAULT_DECOMP_X_MAX);
    idx.len()
}

/// FIRAS 95% confidence level (CL) upper limit on |μ|.
///
/// Reference: Fixsen et al. (1996), ApJ 473, 576
pub const FIRAS_MU_LIMIT: f64 = 9.0e-5;
/// FIRAS 95% CL upper limit on |y| (same reference).
pub const FIRAS_Y_LIMIT: f64 = 1.5e-5;

/// Checks distortion parameters against FIRAS limits.
/// Returns (mu_fraction, y_fraction) as fraction of the FIRAS limit.
pub fn firas_check(params: &DistortionParams) -> (f64, f64) {
    (
        params.mu.abs() / FIRAS_MU_LIMIT,
        params.y.abs() / FIRAS_Y_LIMIT,
    )
}

/// Converts distortion Δn(x) to specific intensity ΔI_ν in MJy/sr.
///
/// ΔI_ν = (2hν³/c²) Δn(x), where ν = x k_B T_0 / h.
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

    /// PIXIE-like channels of Chluba & Jeong (2014): 30–1000 GHz in steps of `step_ghz`,
    /// as dimensionless x at T₀ = 2.725 K. SI constants typed literally (exact since 2019).
    fn cj2014_channels(step_ghz: f64) -> Vec<f64> {
        let (h, k_b, t0): (f64, f64, f64) = (6.626_070_15e-34, 1.380_649e-23, 2.725);
        let n = ((1000.0 - 30.0) / step_ghz).round() as usize + 1;
        (0..n)
            .map(|i| h * (30.0 + step_ghz * i as f64) * 1e9 / (k_b * t0))
            .collect()
    }

    /// Solves the least-squares fit of `d` on the columns of `a` (3 columns) by normal
    /// equations and Cramer's rule. Independent of the code under test.
    fn lstsq3(a: &[[f64; 3]], d: &[f64]) -> [f64; 3] {
        let mut n = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for (row, &di) in a.iter().zip(d) {
            for p in 0..3 {
                b[p] += row[p] * di;
                for q in 0..3 {
                    n[p][q] += row[p] * row[q];
                }
            }
        }
        let det = |m: &[[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d0 = det(&n);
        let mut out = [0.0; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let mut m = n;
            for r in 0..3 {
                m[r][c] = b[r];
            }
            *o = det(&m) / d0;
        }
        out
    }

    /// Anchors the intensity metric to Chluba & Jeong (2014) Appendix A, which gives
    /// the Gram–Schmidt norms {|Y_SZ|, |M⊥|, |G⊥|} ≃ {73.3, 7.99, 21.4} × 10⁻¹⁸
    /// W m⁻² Hz⁻¹ sr⁻¹ for 15 GHz channels with G_T, Y_SZ, M in intensity units
    /// (ΔI = 2(kT₀)³/(hc)² x³ Δn). The Δn metric gives |M⊥|/|Y_SZ| = 0.230 instead
    /// of 0.109, so this pins the x³ weight of ADR 0006.
    #[test]
    fn test_cj2014_basis_norms_are_intensity_norms() {
        let (h, k_b, c, t0): (f64, f64, f64, f64) =
            (6.626_070_15e-34, 1.380_649e-23, 299_792_458.0, 2.725);
        let i0 = 2.0 * (k_b * t0).powi(3) / (h * c).powi(2);
        let xs = cj2014_channels(15.0);
        let vec = |f: fn(f64) -> f64| -> Vec<f64> {
            xs.iter().map(|&x| i0 * x.powi(3) * f(x) * 1e18).collect()
        };
        let (yv, mv, gv) = (vec(y_shape), vec(mu_shape), vec(g_bb));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
        let ny = dot(&yv, &yv).sqrt();
        let ey: Vec<f64> = yv.iter().map(|v| v / ny).collect();
        let my = dot(&mv, &ey);
        let mp: Vec<f64> = mv.iter().zip(&ey).map(|(m, e)| m - my * e).collect();
        let nm = dot(&mp, &mp).sqrt();
        let em: Vec<f64> = mp.iter().map(|v| v / nm).collect();
        let (gy, gm) = (dot(&gv, &ey), dot(&gv, &em));
        let gp: Vec<f64> = (0..xs.len())
            .map(|i| gv[i] - gy * ey[i] - gm * em[i])
            .collect();
        let ng = dot(&gp, &gp).sqrt();
        for (got, want, name) in [(ny, 73.3, "|Y_SZ|"), (nm, 7.99, "|M⊥|"), (ng, 21.4, "|G⊥|")]
        {
            let err = (got - want).abs() / want;
            assert!(
                err < 0.01,
                "{name} = {got:.3} vs CJ2014 {want} (err {:.2}%)",
                err * 100.0
            );
        }
    }

    /// `decompose_gram_schmidt` and the default `decompose_nonlinear_be` must be the
    /// CJ2014 intensity estimator: on a spectrum with a
    /// component outside span{M, Y_SZ, G}, where the metric decides the answer, it
    /// must match an independent least-squares fit to intensities in 1 GHz channels
    /// over the same band. The out-of-span term x·G_bb moves the unweighted fit's y
    /// by several times the tolerance.
    #[test]
    fn test_gram_schmidt_matches_cj2014_channel_fit() {
        let xs = cj2014_channels(1.0);
        let (lo, hi) = (xs[0], *xs.last().unwrap());
        let (mu0, y0, t0) = (1e-5, 3e-6, 2e-6);
        let spec = |x: f64| mu0 * mu_shape(x) + y0 * y_shape(x) + t0 * g_bb(x) + 2e-6 * x * g_bb(x);
        let a: Vec<[f64; 3]> = xs
            .iter()
            .map(|&x| {
                [
                    x.powi(3) * mu_shape(x),
                    x.powi(3) * y_shape(x),
                    x.powi(3) * g_bb(x),
                ]
            })
            .collect();
        let d: Vec<f64> = xs.iter().map(|&x| x.powi(3) * spec(x)).collect();
        let [mu_cj, y_cj, _] = lstsq3(&a, &d);

        let x_grid = log_grid(8000, 1e-3, 40.0);
        let dn: Vec<f64> = x_grid.iter().map(|&x| spec(x)).collect();
        let bf = decompose_nonlinear_be(&x_grid, &dn, lo, hi);
        assert!(
            (bf.mu - mu_cj).abs() < 0.01 * mu0 && (bf.y - y_cj).abs() < 0.01 * y0,
            "B&F μ = {:.5e}, y = {:.5e} vs CJ2014 channel fit {mu_cj:.5e}, {y_cj:.5e}",
            bf.mu,
            bf.y
        );
        let p = decompose_gram_schmidt(&x_grid, &dn, lo, hi);
        assert!(
            (p.mu - mu_cj).abs() < 0.01 * mu0,
            "GS μ = {:.5e} vs CJ2014 channel fit {mu_cj:.5e}",
            p.mu
        );
        assert!(
            (p.y - y_cj).abs() < 0.01 * y0,
            "GS y = {:.5e} vs CJ2014 channel fit {y_cj:.5e}",
            p.y
        );
    }

    /// Anchors `decompose_number_conserving` with closed-form photon-number integrals.
    ///
    /// (a) Δn = ε·n_pl: ∫x²n_pl dx = 2ζ(3) and ∫x²G dx = 6ζ(3), so ΔT/T = ε/3.
    /// (b) Out of span: Δn = μM + yY_SZ + a·xG. M and Y_SZ carry no photon number, and
    ///     ∫x³G dx = 4π⁴/15, so the stripped spectrum is μM + yY_SZ + a(x − c)G with
    ///     c = (4π⁴/15)/(6ζ(3)). The reference μ, y come from a 2×2 least-squares fit
    ///     of x³·a(x − c)G on x³M, x³Y_SZ over [0.5, 18] on a separate uniform grid.
    #[test]
    fn test_decompose_number_conserving_analytic() {
        const ZETA3: f64 = 1.202_056_903_159_594_3;
        let x_grid = log_grid(6000, 1e-4, 60.0);
        let eps = 1e-5;
        let dn: Vec<f64> = x_grid.iter().map(|&x| eps * planck(x)).collect();
        let p = decompose_number_conserving(&x_grid, &dn);
        let err = (p.delta_t_over_t - eps / 3.0).abs() / (eps / 3.0);
        assert!(
            err < 1e-4,
            "ε·n_pl: ΔT/T = {:.6e}, expected ε/3 (err {err:.1e})",
            p.delta_t_over_t
        );

        let (mu0, y0, a) = (1e-5, 2e-6, 1e-6);
        let c = (4.0 * std::f64::consts::PI.powi(4) / 15.0) / (6.0 * ZETA3);
        let n = 200_001;
        let (mut mm, mut my, mut yy, mut md, mut yd) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let x = 0.5 + 17.5 * i as f64 / (n - 1) as f64;
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 } * x.powi(6);
            let (m, yv, r) = (mu_shape(x), y_shape(x), a * (x - c) * g_bb(x));
            mm += w * m * m;
            my += w * m * yv;
            yy += w * yv * yv;
            md += w * m * r;
            yd += w * yv * r;
        }
        let det = mm * yy - my * my;
        let (dmu, dy) = ((md * yy - yd * my) / det, (yd * mm - md * my) / det);
        let dn: Vec<f64> = x_grid
            .iter()
            .map(|&x| mu0 * mu_shape(x) + y0 * y_shape(x) + a * x * g_bb(x))
            .collect();
        let p = decompose_number_conserving(&x_grid, &dn);
        assert!(
            (p.mu - (mu0 + dmu)).abs() < 1e-3 * dmu.abs().max(1e-3 * mu0),
            "out of span: μ = {:.6e}, reference {:.6e}",
            p.mu,
            mu0 + dmu
        );
        assert!(
            (p.y - (y0 + dy)).abs() < 1e-3 * dy.abs().max(1e-3 * y0),
            "out of span: y = {:.6e}, reference {:.6e}",
            p.y,
            y0 + dy
        );
    }

    /// The number-conserving decomposition recovers pure shapes, puts a pure
    /// temperature shift entirely into ΔT/T, and ignores any added G_bb.
    #[test]
    fn test_decompose_number_conserving_pure_shapes() {
        let x_grid = log_grid(4000, 1e-4, 50.0);
        let make = |f: &dyn Fn(f64) -> f64| x_grid.iter().map(|&x| f(x)).collect::<Vec<f64>>();
        let (mu0, y0, t0) = (1e-5, 2e-6, 3e-6);
        let p = decompose_number_conserving(&x_grid, &make(&|x| mu0 * mu_shape(x)));
        assert!((p.mu - mu0).abs() < 1e-3 * mu0, "pure μ: μ = {:.5e}", p.mu);
        assert!(p.y.abs() < 1e-3 * mu0, "pure μ: y = {:.3e}", p.y);
        let p = decompose_number_conserving(&x_grid, &make(&|x| y0 * y_shape(x)));
        assert!((p.y - y0).abs() < 1e-3 * y0, "pure y: y = {:.5e}", p.y);
        assert!(p.mu.abs() < 1e-3 * y0, "pure y: μ = {:.3e}", p.mu);
        let p = decompose_number_conserving(&x_grid, &make(&|x| t0 * g_bb(x)));
        assert!(
            (p.delta_t_over_t - t0).abs() < 1e-6 * t0,
            "pure G: ΔT/T = {:.5e}",
            p.delta_t_over_t
        );
        assert!(
            p.mu.abs() < 1e-9 * t0 && p.y.abs() < 1e-9 * t0,
            "pure G: μ = {:.3e}, y = {:.3e}",
            p.mu,
            p.y
        );
        let mix = make(&|x| mu0 * mu_shape(x) + y0 * y_shape(x));
        let shifted = make(&|x| mu0 * mu_shape(x) + y0 * y_shape(x) + t0 * g_bb(x));
        let (a, b) = (
            decompose_number_conserving(&x_grid, &mix),
            decompose_number_conserving(&x_grid, &shifted),
        );
        assert!(
            (a.mu - b.mu).abs() < 1e-9 * mu0 && (a.y - b.y).abs() < 1e-9 * y0,
            "added G_bb moved μ or y"
        );
    }
}
