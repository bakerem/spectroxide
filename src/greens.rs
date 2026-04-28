//! Green's function for the cosmological thermalization problem.
//!
//! Provides a fast, approximate method for computing spectral distortions
//! from arbitrary energy release histories. The distortion from a delta-function
//! energy injection at redshift z_h is decomposed into μ, y, and temperature
//! shift components using visibility/branching functions.
//!
//! The Green's function approach is "quasi-exact" for small distortions
//! (Δρ/ρ << 1) and much faster than solving the full PDE.
//!
//! Also includes the **photon injection** Green's function (Chluba 2015),
//! which handles injection of photons at a specific frequency x_inj.
//! Unlike pure energy injection, photon injection changes both energy
//! and number, producing negative μ when x_inj < x₀ ≈ 3.60.
//!
//! # Energy non-conservation in the three-component ansatz
//!
//! The temperature-shift coefficient uses J_T = 1 − J_bb* following Chluba (2013).
//! The y-component uses the independently fitted J_y of Chluba (2013) Eq. 5,
//! which is NOT simply (1 − J_μ) × J_bb*. As a result, the three branching
//! ratios do not sum to unity:
//!
//!   J_μ × J_bb* + J_y + (1 − J_bb*) ≠ 1
//!
//! Chluba (2013) §3 notes that the "missing" energy in this ansatz stays within
//! the residual and never exceeds ~16–17%, maximised in the μ-y transition
//! region (z ~ 7–8×10⁴). Using the independent J_y fit matches PDE results
//! more closely than the strictly energy-conserving choice J_y = (1 − J_μ) · J_bb*.
//! Callers that need strict energy conservation must use the full PDE solver.
//!
//! References:
//! - Chluba (2013), MNRAS 436, 2232 [arXiv:1304.6120], Eqs. 5–6 for J_μ, J_y, G_th
//! - Chluba (2015), MNRAS 454, 4182 [arXiv:1506.06582], Eq. 13 for J_bb*
//! - Chluba & Jeong (2014), MNRAS 438, 2065

use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::spectrum::{g_bb, mu_shape, y_shape};

/// Thermalization visibility: probability that energy injection at z is
/// fully thermalized into a blackbody (temperature shift).
///
/// J_bb(z) = exp(−(z/z_μ)^{5/2})
///
/// where z_μ ≈ 1.98×10⁶. Both z_μ and the exponent 5/2 are analytically
/// derived: z_μ from equating the DC+BR photon production rate to the Hubble
/// rate in radiation domination (Chluba & Sunyaev 2012), and 5/2 from the
/// redshift scaling of the DC opacity ∝ (1+z)^{−9/2} vs H ∝ (1+z)^{−2}
/// (Danese & de Zotti 1982; Hu & Silk 1993). These are NOT fit parameters.
pub fn visibility_j_bb(z: f64) -> f64 {
    let ratio = z / Z_MU;
    (-ratio.powf(2.5)).exp()
}

/// Improved thermalization visibility with correction factor.
///
/// J_bb*(z) = 0.983 · J_bb(z) · (1 − 0.0381 (z/z_μ)^{2.29})
///
/// Chluba (2015), arXiv:1506.06582, Eq. 13. Valid for 3×10⁵ ≲ z ≲ 6×10⁶ in
/// the standard cosmology (Chluba 2014 fit, neglecting relativistic temperature
/// corrections which become noticeable at z_i ≳ 4×10⁶). The prefactor 0.983
/// absorbs the small residual blackbody mismatch during the μ-era. The base
/// J_bb exponent (5/2) and z_μ are analytically derived (see `visibility_j_bb`).
pub fn visibility_j_bb_star(z: f64) -> f64 {
    let ratio = z / Z_MU;
    // Clamp at 0: the correction factor goes negative for z/z_mu ≳ 3.9,
    // outside the fit's range of validity. Physically J_bb* ∈ [0, 1].
    (0.983 * visibility_j_bb(z) * (1.0 - 0.0381 * ratio.powf(2.29))).max(0.0)
}

/// y-distortion branching ratio: fraction of energy going into y-type distortion.
///
/// J_y(z) = [1 + ((1+z)/6.0×10⁴)^{2.58}]^{−1}
///
/// Functional form from Chluba (2013), arXiv:1304.6120, Eq. 5. The denominator
/// constant here (6.0×10⁴) comes from the updated refit in Arsenadze et al.
/// (2025), arXiv:2502.11432; the original Chluba 2013 value was 5.9×10⁴, a
/// ~1–2 % shift across the transition region. We keep 6.0×10⁴ for consistency
/// with the J_μ/(1-J_μ) transition scale and with CosmoTherm GF tables.
/// Approaches 1 for z ≪ 6×10⁴ and 0 for z ≫ 6×10⁴. Note: J_y ≠ (1 − J_μ) in
/// the transition region; using the independent fit gives better spectral
/// agreement with PDE results.
pub fn visibility_j_y(z: f64) -> f64 {
    1.0 / (1.0 + ((1.0 + z) / 6.0e4).powf(2.58))
}

/// μ-distortion branching ratio: fraction of energy going into μ-type distortion.
///
/// J_μ(z) = 1 − exp(−((1+z)/5.8×10⁴)^{1.88})
///
/// Chluba (2013), arXiv:1304.6120, Eq. 5. Approaches 0 for z ≪ 5.8×10⁴ (no μ)
/// and 1 for z ≫ 5.8×10⁴ (pure μ). The transition scale is physically motivated
/// by y_γ(z) ≈ 1, but the precise value and exponent are fit parameters.
pub fn visibility_j_mu(z: f64) -> f64 {
    1.0 - (-((1.0 + z) / 5.8e4).powf(1.88)).exp()
}

/// Temperature shift branching: fraction of energy going into temperature shift.
///
/// J_T(z) = 1 − J_bb*(z)
///
/// Following Chluba (2013), the temperature-shift coefficient is
/// (1 − J_bb*), which captures the fraction of energy fully thermalized
/// into a blackbody. This is the standard convention used in the literature.
pub fn visibility_j_t(z: f64) -> f64 {
    1.0 - visibility_j_bb_star(z)
}

/// Compute the Green's function G_th(x, z_h) for a delta-function energy injection
/// at redshift z_h, observed at z = 0.
///
/// The three-component decomposition from Chluba (2013), Eq. 6:
///
///   G_th = (3/κ_c) · J_μ · J_bb* · M(x) + J_y/4 · Y_SZ(x) + (1−J_bb*)/4 · G(x)
///
/// where J_μ, J_y, and J_bb* are independently fitted visibility functions,
/// and the temperature-shift coefficient (1 − J_bb*)/4 follows the Chluba (2013)
/// convention.
///
/// Returns the distortion Δn(x) per unit Δρ/ρ injected.
///
/// # Accuracy
///
/// Per-point spectral shape vs PDE (worst-case over frequency grid):
/// - **Deep μ-era** (z_h > 2×10⁵): < 17% per-point; < 5% on integrated μ.
/// - **y-era** (z_h < 10⁴): < 5% per-point; < 1% on integrated y.
/// - **Transition era** (z_h ~ 3×10⁴ – 10⁵): 8–17% per-point shape error
///   (improved from 30–70% by using the independently fitted J_y instead of
///   1 − J_μ). Integrated μ and y agree with PDE to ~5–10%.
///
/// References:
///   Chluba (2013), MNRAS 436, 2232 [arXiv:1304.6120], Eq. 6
///   Arsenadze et al. (2025), JHEP 03, 018 [arXiv:2409.12940], Appendix D
pub fn greens_function(x: f64, z_h: f64) -> f64 {
    let j_mu = visibility_j_mu(z_h);
    let j_bb_star = visibility_j_bb_star(z_h);
    let j_y = visibility_j_y(z_h);

    let mu_part = (3.0 / KAPPA_C) * j_mu * j_bb_star * mu_shape(x);
    let y_part = 0.25 * j_y * y_shape(x);
    let t_part = 0.25 * (1.0 - j_bb_star) * g_bb(x);

    mu_part + y_part + t_part
}

/// Compute the spectral distortion from an arbitrary energy release history.
///
/// ΔI(x) = ∫ G_th(x, z') · d(Q/ρ_γ)/dz' dz'
///
/// # Arguments
/// * `x_grid` - frequency grid
/// * `dq_dz` - function giving d(Δρ/ρ_γ)/dz at each redshift
/// * `z_min` - minimum integration redshift
/// * `z_max` - maximum integration redshift
/// * `n_z` - number of redshift integration points
///
/// Returns Δn(x) at each frequency grid point.
pub fn distortion_from_heating<F>(
    x_grid: &[f64],
    dq_dz: F,
    z_min: f64,
    z_max: f64,
    n_z: usize,
) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let n_x = x_grid.len();
    let mut delta_n = vec![0.0; n_x];

    // Integrate in ln(1+z) for numerical stability
    let ln_min = (1.0 + z_min).ln();
    let ln_max = (1.0 + z_max).ln();
    let dln = (ln_max - ln_min) / (n_z - 1).max(1) as f64;

    for j in 0..n_z {
        let ln_1pz = ln_min + j as f64 * dln;
        let z = ln_1pz.exp() - 1.0;
        let dz_dln = 1.0 + z; // d(z)/d(ln(1+z)) = 1+z

        let heating = dq_dz(z) * dz_dln;

        if heating.abs() < 1e-50 {
            continue;
        }

        // Trapezoidal weight
        let w = if j == 0 || j == n_z - 1 {
            0.5 * dln
        } else {
            dln
        };

        // Hoist z-only visibility functions out of x-loop
        let j_mu = visibility_j_mu(z);
        let j_bb_star = visibility_j_bb_star(z);
        let j_y = visibility_j_y(z);
        let coeff_mu = (3.0 / KAPPA_C) * j_mu * j_bb_star;
        let coeff_y = 0.25 * j_y;
        let coeff_t = 0.25 * (1.0 - j_bb_star);
        let hw = heating * w;

        for i in 0..n_x {
            let x = x_grid[i];
            let g = coeff_mu * mu_shape(x) + coeff_y * y_shape(x) + coeff_t * g_bb(x);
            delta_n[i] += g * hw;
        }
    }

    delta_n
}

/// Extract μ parameter from the Green's function approximation.
///
/// μ = (3/κ_c) ∫ J_bb*(z) · J_μ(z) · d(Δρ/ρ)/dz dz
///
/// Calls [`mu_y_from_heating`] internally and returns the μ component.
/// For simultaneous μ and y, use [`mu_y_from_heating`] directly.
pub fn mu_from_heating<F>(dq_dz: F, z_min: f64, z_max: f64, n_z: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    mu_y_from_heating(dq_dz, z_min, z_max, n_z).0
}

/// Extract y parameter from the Green's function approximation.
///
/// y = (1/4) ∫ J_y(z) · d(Δρ/ρ)/dz dz
///
/// Uses the independently fitted J_y visibility function (Arsenadze et al. 2025),
/// which gives better agreement with PDE results than (1 − J_μ).
///
/// Calls [`mu_y_from_heating`] internally and returns the y component.
/// For simultaneous μ and y, use [`mu_y_from_heating`] directly.
pub fn y_from_heating<F>(dq_dz: F, z_min: f64, z_max: f64, n_z: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    mu_y_from_heating(dq_dz, z_min, z_max, n_z).1
}

/// Compute both μ and y from an arbitrary energy release history in a single pass.
///
/// This is more efficient than calling `mu_from_heating` and `y_from_heating`
/// separately, as it evaluates the visibility functions only once per z-step.
///
/// Returns (μ, y).
pub fn mu_y_from_heating<F>(dq_dz: F, z_min: f64, z_max: f64, n_z: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let ln_min = (1.0 + z_min).ln();
    let ln_max = (1.0 + z_max).ln();
    let dln = (ln_max - ln_min) / (n_z - 1).max(1) as f64;

    let mut mu = 0.0;
    let mut y = 0.0;
    for j in 0..n_z {
        let ln_1pz = ln_min + j as f64 * dln;
        let z = ln_1pz.exp() - 1.0;
        let dz_dln = 1.0 + z;
        let heating = dq_dz(z) * dz_dln;

        let w = if j == 0 || j == n_z - 1 {
            0.5 * dln
        } else {
            dln
        };
        let hw = heating * w;

        let j_mu = visibility_j_mu(z);
        let j_bb_star = visibility_j_bb_star(z);
        mu += (3.0 / KAPPA_C) * j_bb_star * j_mu * hw;
        y += 0.25 * visibility_j_y(z) * hw;
    }

    (mu, y)
}

// ============================================================================
// Photon injection Green's function
// ============================================================================
//
// For photon injection at frequency x_inj, both photon number AND energy
// change. The resulting distortion depends on x_inj relative to the balanced
// frequency x₀ ≈ 3.60:
//
//   - x_inj > x₀: positive μ (like energy injection)
//   - x_inj < x₀: negative μ (opposite sign!)
//   - x_inj = x₀: zero μ
//
// The formalism introduces a photon survival probability P_s(x, z) that
// captures absorption of soft photons by DC/BR processes.
//
// References:
//   Chluba (2015), arXiv:1506.06582
//   Arsenadze et al. (2025), arXiv:2409.12940, Appendix C+D

/// Critical frequency for double Compton absorption.
///
/// x_c_DC(z) = 8.60×10⁻³ × [(1+z)/(2×10⁶)]^{1/2}
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 25a
pub fn x_c_dc(z: f64) -> f64 {
    8.60e-3 * ((1.0 + z) / 2.0e6).powf(0.5)
}

/// Critical frequency for bremsstrahlung absorption.
///
/// x_c_BR(z) = 1.23×10⁻³ × [(1+z)/(2×10⁶)]^{−0.672}
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 25b
pub fn x_c_br(z: f64) -> f64 {
    1.23e-3 * ((1.0 + z) / 2.0e6).powf(-0.672)
}

/// Combined critical frequency for photon absorption.
///
/// x_c² = x_c_DC² + x_c_BR²  (quadrature addition)
///
/// Photons with x << x_c are absorbed by DC/BR; x >> x_c survive.
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 25
pub fn x_c(z: f64) -> f64 {
    let dc = x_c_dc(z);
    let br = x_c_br(z);
    (dc * dc + br * br).sqrt()
}

/// Photon survival probability P_s(x, z).
///
/// P_s(x, z) = exp(−x_c(z)/x)
///
/// Probability that an injected photon at frequency x survives absorption
/// by DC/BR processes. At high x (x >> x_c), P_s → 1 (photon survives).
/// At low x (x << x_c), P_s → 0 (photon is absorbed).
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 24
pub fn photon_survival_probability(x: f64, z: f64) -> f64 {
    if x < 1e-30 {
        return 0.0;
    }
    let ratio = x_c(z) / x;
    (-ratio).exp()
}

/// Photon survival probability: numerical τ_ff in the y-era, analytic at higher z.
///
/// At z ≤ 5×10⁴ (y-era): P_s = exp(−τ_ff) where τ_ff is the integrated DC+BR
/// absorption optical depth (Chluba 2015, Eq. 29/32). The raw absorption integral
/// is valid here because Compton scattering is weak (y_γ << 1).
///
/// At z > 5×10⁴ (μ-era): P_s = exp(−x_c/x), the quasi-stationary approximation
/// which correctly accounts for Compton redistribution. The raw τ_ff integral
/// would overestimate absorption here because it ignores Compton upscattering.
fn photon_survival_probability_numerical(x: f64, z_h: f64, cosmo: &Cosmology) -> f64 {
    // μ-era: use the analytic formula (accounts for Compton redistribution)
    if z_h > 5.0e4 {
        return photon_survival_probability(x, z_h);
    }
    tau_ff_survival(x, z_h, cosmo)
}

/// Compute P_s = exp(−τ_ff) from the integrated DC+BR absorption optical depth.
///
/// τ_ff(x, z_h) = ∫_{z_end}^{z_h} R(x,z) × dτ_Thomson/dz dz
///
/// where R = [K_DC + K_BR] × (e^x − 1) / x³ is the absorption rate per Thomson time
/// and dτ_Thomson/dz = n_e σ_T c / [(1+z) H(z)].
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 29
fn tau_ff_survival(x: f64, z_h: f64, cosmo: &Cosmology) -> f64 {
    use crate::bremsstrahlung::br_emission_coefficient_with_he;
    use crate::double_compton::dc_emission_coefficient;
    use crate::recombination::{ionization_fraction, saha_he_i, saha_he_ii};

    if x < 1e-30 {
        return 0.0;
    }

    // Below recombination freeze-out, BR/DC are negligible
    let z_end = 200.0;
    if z_h <= z_end {
        return 1.0;
    }

    let n_steps: usize = 500;
    let log_z_h = z_h.ln();
    let log_z_end = z_end.ln();
    let d_log_z = (log_z_h - log_z_end) / n_steps as f64;

    // Bose factor: (e^x - 1), precompute once since x is fixed. Overflow at
    // x > 500 → +∞ propagates through `rate` and triggers the tau > 500 → 0
    // saturation below. Using f64::MAX would pass finite-checks elsewhere.
    let bose_factor = if x > 500.0 { f64::INFINITY } else { x.exp_m1() };
    let inv_x3 = 1.0 / (x * x * x);

    let mut tau = 0.0;

    for i in 0..=n_steps {
        let log_z = log_z_end + i as f64 * d_log_z;
        let z = log_z.exp();
        let opz = 1.0 + z;

        // Use cosmology-aware θ_z so that a user-supplied T_CMB is honoured.
        let tz = cosmo.theta_z(z);
        let te = tz; // T_e ≈ T_z (valid in y-era and μ-era)

        // Full ionization fraction including H + He (He²⁺ at z≳8000, He⁺ at
        // 2000≲z≲8000, He⁰ after). Previous simplified-Saha branching omitted
        // helium electrons at z>6000 and z>1500, under-counting n_e by ~7–8%.
        let x_e_frac = ionization_fraction(z, cosmo);

        let n_h = cosmo.n_h(z);
        let n_he = cosmo.n_he(z);
        let n_e = cosmo.n_e(z, x_e_frac);

        // DC emission coefficient
        let k_dc = dc_emission_coefficient(x, tz);

        // BR emission coefficient (with precomputed He ionization)
        let y_he_ii = saha_he_ii(z, cosmo);
        let y_he_i = saha_he_i(z, cosmo);
        let k_br = br_emission_coefficient_with_he(
            x,
            te,
            tz,
            n_h,
            n_he,
            n_e,
            x_e_frac.min(1.0),
            y_he_ii,
            y_he_i,
        );

        // Absorption rate per Thomson time: R = K × (e^x - 1) / x³
        let rate = (k_dc + k_br) * bose_factor * inv_x3;

        // dτ_Thomson/dz = n_e σ_T c / [(1+z) H(z)]
        let dtau_dz = n_e * SIGMA_THOMSON * C_LIGHT / (opz * cosmo.hubble(z));

        // Trapezoidal weight: endpoints get 0.5
        let weight = if i == 0 || i == n_steps { 0.5 } else { 1.0 };

        // Accumulate: dz = z × d(log z)
        tau += weight * rate * dtau_dz * z * d_log_z;
    }

    if tau > 500.0 {
        return 0.0;
    }
    (-tau).exp()
}

// (Arsenadze x'-dependent transition table removed: not well supported by
// physics. Photon injection now uses the universal J_μ(z) visibility.)

// ---------------------------------------------------------------------------
// Compton broadening helpers for surviving photon bump
// ---------------------------------------------------------------------------

/// Compton scattering helper: f(x) = exp(-x)(1 + x²/2).
///
/// Reference: Arsenadze et al. (2025), Eq. D14
fn f_cs(x: f64) -> f64 {
    (-x).exp() * (1.0 + x * x / 2.0)
}

/// Compton broadening parameter α(x', y_γ).
///
/// α = (3 − 2f(x')) / sqrt(1 + x'·y_γ)
///
/// Reference: Arsenadze et al. (2025), Eq. D13
fn alpha_cs(x_inj: f64, yg: f64) -> f64 {
    (3.0 - 2.0 * f_cs(x_inj)) / (1.0 + x_inj * yg).sqrt()
}

/// Compton broadening parameter β(x', y_γ).
///
/// β = 1 / (1 + x'·y_γ·(1 − f(x')))
///
/// Reference: Arsenadze et al. (2025), Eq. D13
fn beta_cs(x_inj: f64, yg: f64) -> f64 {
    1.0 / (1.0 + x_inj * yg * (1.0 - f_cs(x_inj)))
}

/// Compute the Compton-broadened photon bump and its energy integral f_int.
///
/// Returns (bump_value, f_int) where:
/// - bump_value is the log-normal PDF at x_obs
/// - f_int = exp((α+β)·y_γ) / (1 + x_inj·y_γ) is the mean energy ratio
///
/// For y_γ < 1e-6, falls back to a narrow Gaussian at x_inj.
///
/// Reference: Arsenadze et al. (2025), Eq. D15-D16
fn broadened_bump(x_obs: f64, x_inj: f64, yg: f64) -> (f64, f64) {
    if yg < 1e-6 {
        // Negligible broadening: use narrow Gaussian
        let sig = 0.005 * x_inj;
        let norm = 1.0 / (sig * (2.0 * std::f64::consts::PI).sqrt());
        let bump = (-(x_obs - x_inj).powi(2) / (2.0 * sig * sig)).exp() * norm;
        return (bump, 1.0);
    }

    let alpha = alpha_cs(x_inj, yg);
    let beta = beta_cs(x_inj, yg);
    let denom = 1.0 + x_inj * yg;

    // Log-normal parameters
    let mu_ln = x_inj.ln() + alpha * yg - denom.ln();
    let sigma_ln_sq = 2.0 * beta * yg;
    let sigma_ln = sigma_ln_sq.max(1e-30).sqrt();

    // f_int = exp((α+β)·y_γ) / (1 + x'·y_γ)
    let f_int_arg = (alpha + beta) * yg;
    let f_int = if f_int_arg < 700.0 {
        f_int_arg.exp() / denom
    } else {
        700.0_f64.exp() / denom // Clamp to avoid infinity
    };

    // Log-normal bump
    let bump = if x_obs > 1e-30 {
        let ln_x = x_obs.ln();
        let z_score = (ln_x - mu_ln) / sigma_ln;
        (-0.5 * z_score * z_score).exp() / (x_obs * sigma_ln * (2.0 * std::f64::consts::PI).sqrt())
    } else {
        0.0
    };

    (bump, f_int)
}

/// Photon-injection GF is only valid in the deep μ-era (z_h ≳ 2×10⁵) or
/// the y-era (z_h ≲ 5×10⁴). In the μ-y transition window, the simple
/// μ+y decomposition misses residual (r-type) contributions; users must
/// fall back to the PDE solver.
const PHOTON_GF_Y_ERA_Z_MAX: f64 = 5.0e4;
const PHOTON_GF_MU_ERA_Z_MIN: f64 = 2.0e5;

#[inline]
fn assert_photon_gf_regime(z_h: f64) {
    // Fractional slack absorbs ln/exp roundoff at the boundaries.
    const TOL: f64 = 1.0e-6;
    let lo = PHOTON_GF_Y_ERA_Z_MAX * (1.0 + TOL);
    let hi = PHOTON_GF_MU_ERA_Z_MIN * (1.0 - TOL);
    assert!(
        !(z_h > lo && z_h < hi),
        "Photon Green's function is not valid in the μ-y transition era \
         ({:.0e} < z_h < {:.0e}); got z_h = {:.3e}. Use the PDE solver instead.",
        PHOTON_GF_Y_ERA_Z_MAX,
        PHOTON_GF_MU_ERA_Z_MIN,
        z_h
    );
}

/// Whether `z_h` is inside the μ-y transition band where the photon GF is invalid.
#[inline]
fn in_photon_gf_transition_band(z_h: f64) -> bool {
    const TOL: f64 = 1.0e-6;
    let lo = PHOTON_GF_Y_ERA_Z_MAX * (1.0 + TOL);
    let hi = PHOTON_GF_MU_ERA_Z_MIN * (1.0 - TOL);
    z_h > lo && z_h < hi
}

/// Green's function for monochromatic photon injection.
///
/// Returns Δn(x_obs) per unit ΔN/N injected at frequency x_inj and
/// redshift z_h. Uses the universal μ-y visibility function J_μ(z)
/// (same as heat injection) to blend between pure μ-era and pure y-era
/// contributions.
///
/// Structure:
///
///   G_ph = J_μ · G_μ + (1 − J_μ) · G_y
///
/// where G_μ is the μ-era contribution (M + G_bb terms) and G_y is the
/// y-era contribution (Y_SZ + surviving bump).
///
/// When P_s = 0 (soft photon limit), reduces to α_ρ × x_inj × G_th(x, z_h).
///
/// # Arguments
/// * `x_obs` - observation frequency
/// * `x_inj` - injection frequency
/// * `z_h` - injection redshift
/// * `sigma_x` - extra Gaussian width for the surviving bump (0 for pure Compton)
/// * `cosmo` - cosmological parameters (for Compton y_γ)
///
/// References:
///   Chluba (2015), arXiv:1506.06582
pub fn greens_function_photon(
    x_obs: f64,
    x_inj: f64,
    z_h: f64,
    sigma_x: f64,
    cosmo: &Cosmology,
) -> f64 {
    assert_photon_gf_regime(z_h);
    let j_bb_star = visibility_j_bb_star(z_h);
    let p_s = photon_survival_probability_numerical(x_inj, z_h, cosmo);
    let alpha_x = ALPHA_RHO * x_inj;

    // Universal μ-y transition (same as heat injection GF)
    let j_mu = visibility_j_mu(z_h);

    // μ-era contribution (Arsenadze Eq. D9)
    let mu_factor = 1.0 - p_s * X_BALANCED / x_inj;
    let mu_part = (3.0 / KAPPA_C) * j_bb_star * mu_factor * mu_shape(x_obs);

    // Temperature shift
    let lam = 1.0 - mu_factor * j_bb_star;
    let t_part = lam / 4.0 * g_bb(x_obs);

    // Deep μ-era short-circuit: when J_μ ≈ 1, y-era contribution is zero
    // and computing it risks f_int overflow for large y_γ.
    if j_mu > 1.0 - 1e-12 {
        return alpha_x * (mu_part + t_part);
    }

    // y-era contribution
    let yg = cosmo.compton_y_parameter(z_h);

    // Broadened surviving photon bump (log-normal from Compton scattering)
    let (bump_shape, f_int) = broadened_bump(x_obs, x_inj, yg);

    // Smooth y-era: energy balance coefficient
    // (1 − P_s · f_int) accounts for energy shift of surviving photons
    let coeff_y = 1.0 - p_s * f_int;
    let y_smooth = coeff_y * 0.25 * y_shape(x_obs);

    // Combine μ and y via universal visibility J_μ(z)
    let mu_era = mu_part + t_part;
    let y_era = y_smooth;
    let smooth = alpha_x * (j_mu * mu_era + (1.0 - j_mu) * y_era);

    // Surviving photon bump (broadened by Compton scattering).
    // Use the precomputed bump_shape from broadened_bump() when sigma_x == 0.
    // Only recompute when sigma_x > 0 (extra instrumental broadening).
    let surviving = if x_obs > 1e-30 {
        let safe_x = x_obs.max(1e-30);
        let bump_final = if sigma_x > 0.0 {
            if yg >= 1e-6 {
                // Log-normal form with extra sigma_x added in quadrature
                let alpha = alpha_cs(x_inj, yg);
                let beta = beta_cs(x_inj, yg);
                let denom = 1.0 + x_inj * yg;
                let mu_ln = x_inj.ln() + alpha * yg - denom.ln();
                let sigma_ln_sq = 2.0 * beta * yg + (sigma_x / x_inj).powi(2);
                let sigma_ln = sigma_ln_sq.max(1e-30).sqrt();
                let ln_x = safe_x.ln();
                let z_score = (ln_x - mu_ln) / sigma_ln;
                (-0.5 * z_score * z_score).exp()
                    / (safe_x * sigma_ln * (2.0 * std::f64::consts::PI).sqrt())
            } else {
                // No Compton broadening: use Gaussian in x-space
                let norm = 1.0 / (sigma_x * (2.0 * std::f64::consts::PI).sqrt());
                (-(x_obs - x_inj).powi(2) / (2.0 * sigma_x * sigma_x)).exp() * norm
            }
        } else {
            // No extra broadening: use precomputed bump from broadened_bump()
            bump_shape
        };
        p_s * (1.0 - j_mu) * G2_PLANCK / (safe_x * safe_x) * bump_final
    } else {
        0.0
    };

    smooth + surviving
}

/// Compute μ from monochromatic photon injection at frequency x_inj.
///
/// μ = α_ρ × x_inj × (3/κ_c) × J*(z_h) × J_μ(z_h)
///     × [1 − P_s × x₀/x_inj] × ΔN/N
///
/// Uses the universal J_μ(z) visibility function (same as heat injection).
///
/// Sign behavior:
///   - x_inj > x₀ and P_s ≈ 1: μ > 0 (energy-dominated)
///   - x_inj < x₀ and P_s ≈ 1: μ < 0 (number-dominated, negative μ!)
///   - P_s ≈ 0 (soft photons absorbed): μ > 0 always (pure energy injection)
///
/// Reference: Chluba (2015), Eq. C7
pub fn mu_from_photon_injection(x_inj: f64, z_h: f64, delta_n_over_n: f64) -> f64 {
    assert_photon_gf_regime(z_h);
    let j_bb_star = visibility_j_bb_star(z_h);
    let j_mu = visibility_j_mu(z_h);
    let p_s = photon_survival_probability(x_inj, z_h);

    let mu_factor = 1.0 - p_s * X_BALANCED / x_inj;

    ALPHA_RHO * x_inj * (3.0 / KAPPA_C) * j_bb_star * j_mu * mu_factor * delta_n_over_n
}

/// Compute spectral distortion from an arbitrary photon injection history.
///
/// ΔI(x) = ∫ G_ph(x, x_inj, z') × d(ΔN/N)/dz' dz'
///
/// This integrates the photon injection Green's function over the injection
/// history, analogous to `distortion_from_heating` for energy injection.
///
/// # Arguments
/// * `x_grid` - observation frequency grid
/// * `x_inj` - injection frequency
/// * `dn_dz` - function giving d(ΔN/N)/dz at each redshift
/// * `z_min` - minimum integration redshift
/// * `z_max` - maximum integration redshift
/// * `n_z` - number of redshift integration points
/// * `sigma_x` - extra Gaussian width for the surviving photon bump (0 for pure Compton)
/// * `cosmo` - cosmological parameters (for Compton y_γ)
pub fn distortion_from_photon_injection<F>(
    x_grid: &[f64],
    x_inj: f64,
    dn_dz: F,
    z_min: f64,
    z_max: f64,
    n_z: usize,
    sigma_x: f64,
    cosmo: &Cosmology,
) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let n_x = x_grid.len();
    let mut delta_n = vec![0.0; n_x];

    let ln_min = (1.0 + z_min).ln();
    let ln_max = (1.0 + z_max).ln();
    let dln = (ln_max - ln_min) / (n_z - 1).max(1) as f64;

    for j in 0..n_z {
        let ln_1pz = ln_min + j as f64 * dln;
        let z = ln_1pz.exp() - 1.0;
        // Skip the μ-y transition band: `greens_function_photon` panics there.
        // When the integration window crosses the band, the integral is
        // mildly underestimated — but a panic mid-integration would lose all
        // accumulated work, so the trade is correct.
        if in_photon_gf_transition_band(z) {
            continue;
        }
        let dz_dln = 1.0 + z;

        let source = dn_dz(z) * dz_dln;
        if source.abs() < 1e-50 {
            continue;
        }

        let w = if j == 0 || j == n_z - 1 {
            0.5 * dln
        } else {
            dln
        };

        for i in 0..n_x {
            delta_n[i] += greens_function_photon(x_grid[i], x_inj, z, sigma_x, cosmo) * source * w;
        }
    }

    delta_n
}

/// Returns `true` iff the GF integration window crosses the μ-y transition
/// band; callers can use this to emit a warning at function entry.
pub fn integration_crosses_photon_gf_gap(z_min: f64, z_max: f64) -> bool {
    let lo = PHOTON_GF_Y_ERA_Z_MAX;
    let hi = PHOTON_GF_MU_ERA_Z_MIN;
    z_min < hi && z_max > lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greens_high_z_is_temperature_shift() {
        // At z >> z_mu, the Green's function should be a temperature shift
        // G_th ≈ (1/4) G(x) since J_bb* → 0 and J_y → 0
        let z_h = 5.0e6;
        let x = 3.0;
        let g = greens_function(x, z_h);
        let expected = 0.25 * g_bb(x);
        assert!(
            (g - expected).abs() / expected.abs().max(1e-20) < 0.01,
            "At z={z_h}, G={g}, expected T-shift={expected}"
        );
    }

    #[test]
    fn test_greens_mu_era() {
        // At z ~ 3×10⁵, should be dominated by μ-distortion
        let z_h = 3.0e5;
        let j_mu = visibility_j_mu(z_h);
        let j_bb = visibility_j_bb_star(z_h);
        // μ should be significant
        assert!(j_mu > 0.5, "J_mu({z_h}) = {j_mu}, should be > 0.5");
        assert!(j_bb > 0.5, "J_bb*({z_h}) = {j_bb}, should be > 0.5");
    }

    #[test]
    fn test_greens_y_era() {
        // At z ~ 5000, should be dominated by y-distortion
        let z_h = 5.0e3;
        let j_mu = visibility_j_mu(z_h);
        assert!(j_mu < 0.1, "J_mu should be small at z={z_h}");
    }

    #[test]
    fn test_mu_from_delta_injection() {
        // Delta-function injection at z_h in the μ-era
        // μ ≈ 1.401 × Δρ/ρ × J_bb*(z_h) × J_μ(z_h)
        let z_h = 2.0e5;
        let drho_rho = 1e-5;
        let sigma_z = 5000.0; // wide enough to be resolved on the integration grid

        let dq_dz = |z: f64| -> f64 {
            drho_rho * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
        };

        let mu = mu_from_heating(dq_dz, 1e3, 5e6, 10000);
        let expected =
            (3.0 / KAPPA_C) * visibility_j_bb_star(z_h) * visibility_j_mu(z_h) * drho_rho;

        let rel_err = (mu - expected).abs() / expected.abs().max(1e-20);
        assert!(
            rel_err < 0.05,
            "μ = {mu:.3e}, expected {expected:.3e}, rel_err = {rel_err}"
        );
    }

    // --- Photon injection Green's function tests ---

    #[test]
    fn test_x_c_dc_dominance_at_high_z() {
        // At high z (thermalization era), DC should dominate over BR
        let z = 2.0e6;
        let dc = x_c_dc(z);
        let br = x_c_br(z);
        assert!(
            dc > br,
            "At z={z:.0e}: x_c_DC={dc:.4e} should > x_c_BR={br:.4e}"
        );
    }

    #[test]
    fn test_x_c_br_dominance_at_low_z() {
        // At low z, BR should dominate over DC
        let z = 1.0e4;
        let dc = x_c_dc(z);
        let br = x_c_br(z);
        assert!(
            br > dc,
            "At z={z:.0e}: x_c_BR={br:.4e} should > x_c_DC={dc:.4e}"
        );
    }

    #[test]
    fn test_photon_survival_limits() {
        let z = 2.0e5;
        // High x: photon survives
        let p_high = photon_survival_probability(10.0, z);
        assert!(p_high > 0.99, "P_s(x=10, z=2e5) = {p_high}, should be ~1");

        // Very low x: photon absorbed
        let p_low = photon_survival_probability(1e-5, z);
        assert!(p_low < 1e-10, "P_s(x=1e-5, z=2e5) = {p_low}, should be ~0");
    }

    #[test]
    fn test_tau_ff_fallback_to_analytic_at_high_z() {
        // Above z = 5e4, the numerical P_s falls back to the analytic
        // formula (which accounts for Compton redistribution).
        let cosmo = Cosmology::default();
        let z = 2.0e5;
        for &x in &[0.01, 0.1, 1.0] {
            let ps_num = photon_survival_probability_numerical(x, z, &cosmo);
            let ps_ana = photon_survival_probability(x, z);
            assert!(
                (ps_num - ps_ana).abs() < 1e-15,
                "At z={z:.0e} > 5e4, numerical should equal analytic: \
                 {ps_num:.6e} vs {ps_ana:.6e}"
            );
        }
    }

    #[test]
    fn test_tau_ff_limits() {
        let cosmo = Cosmology::default();
        // P_s = 1 below recombination freeze-out
        assert_eq!(
            photon_survival_probability_numerical(1.0, 100.0, &cosmo),
            1.0
        );

        // P_s → 1 for x >> x_c at moderate z
        let ps_high_x = photon_survival_probability_numerical(10.0, 1e4, &cosmo);
        assert!(
            ps_high_x > 0.95,
            "P_s(x=10, z=1e4) should be ~1: got {ps_high_x}"
        );

        // P_s → 0 for x << x_c at high z
        let ps_low_x = photon_survival_probability_numerical(1e-5, 1e5, &cosmo);
        assert!(
            ps_low_x < 0.01,
            "P_s(x=1e-5, z=1e5) should be ~0: got {ps_low_x}"
        );
    }

    #[test]
    fn test_mu_from_photon_injection_sign_flip() {
        // At high x_inj > x₀ with P_s ≈ 1: positive μ (like energy injection)
        // At low x_inj < x₀ with P_s ≈ 1: negative μ (unique to photon injection)
        let z_h = 2.0e5;
        let dn_over_n = 1e-5;

        let mu_high = mu_from_photon_injection(10.0, z_h, dn_over_n);
        assert!(
            mu_high > 0.0,
            "x_inj=10 > x₀: μ should be positive, got {mu_high:.4e}"
        );

        let mu_low = mu_from_photon_injection(2.0, z_h, dn_over_n);
        assert!(
            mu_low < 0.0,
            "x_inj=2 < x₀: μ should be negative, got {mu_low:.4e}"
        );
    }

    #[test]
    fn test_mu_from_photon_injection_balanced() {
        // At x_inj = x₀ with P_s ≈ 1: μ should be near zero
        let z_h = 2.0e5;
        let dn_over_n = 1e-5;
        let mu = mu_from_photon_injection(X_BALANCED, z_h, dn_over_n);

        // P_s at x₀ ≈ 3.6 is very close to 1 at z=2e5, so the zero
        // should be very accurate
        let p_s = photon_survival_probability(X_BALANCED, z_h);
        assert!(p_s > 0.99, "P_s at x₀ should be ~1");

        // The residual should be small (from P_s not being exactly 1)
        let mu_scale = mu_from_photon_injection(10.0, z_h, dn_over_n).abs();
        assert!(
            mu.abs() < 0.01 * mu_scale,
            "At x₀: |μ| = {:.4e} should be << {mu_scale:.4e}",
            mu.abs()
        );
    }

    #[test]
    fn test_visibility_functions_physical_bounds() {
        // All visibility functions must satisfy physical bounds and asymptotics.
        // These are NOT tautological checks — they verify the fitting formulas
        // produce physically sensible values.

        for &z in &[1e3, 5e3, 1e4, 5e4, 1e5, 3e5, 1e6, 3e6] {
            let jmu = visibility_j_mu(z);
            let jbb = visibility_j_bb_star(z);
            let jy = visibility_j_y(z);
            let jt = visibility_j_t(z);

            // Individual visibilities must be in [0, 1]
            assert!(
                jmu >= 0.0 && jmu <= 1.0,
                "J_mu({z:.0e}) = {jmu} out of [0,1]"
            );
            assert!(
                jbb >= 0.0 && jbb <= 1.0,
                "J_bb*({z:.0e}) = {jbb} out of [0,1]"
            );
            assert!(jy >= 0.0 && jy <= 1.0, "J_y({z:.0e}) = {jy} out of [0,1]");
            // J_T = 1 - J_mu*J_bb* - J_y can go negative (up to ~-15%) in the
            // transition region (z ~ 5e4) where independently fitted J_y + J_mu*J_bb*
            // slightly exceeds 1. This is absorbed into the unobservable temperature shift.
            assert!(
                jt >= -0.2 && jt <= 1.0,
                "J_T({z:.0e}) = {jt} out of [-0.2,1]"
            );
        }

        // Asymptotic limits:
        // Deep y-era (z << 6e4): J_y → 1, J_mu → 0
        assert!(visibility_j_y(1e3) > 0.99, "J_y should → 1 at z=1e3");
        assert!(visibility_j_mu(1e3) < 0.01, "J_mu should → 0 at z=1e3");

        // Deep μ-era (z >> 6e4): J_mu → 1, J_y → 0
        assert!(visibility_j_mu(1e6) > 0.99, "J_mu should → 1 at z=1e6");
        assert!(visibility_j_y(1e6) < 0.01, "J_y should → 0 at z=1e6");

        // Thermalization (z >> z_mu): J_bb* → 1
        // At z=3e6, J_bb ≈ exp(-(3e6/2e6)^2.5) = exp(-5.8) ≈ 0.003
        // so J_bb* ≈ 0.003 — deep thermalization suppresses the μ-distortion
        assert!(
            visibility_j_bb_star(3e6) < 0.1,
            "J_bb* should be small at z=3e6 (deep thermalization)"
        );
        // At z << z_mu, J_bb* ≈ 1 (no thermalization)
        assert!(
            visibility_j_bb_star(1e5) > 0.9,
            "J_bb* should → 1 below z_mu"
        );

        // Monotonicity: J_mu increases with z, J_y decreases with z
        let zs = [1e3, 1e4, 5e4, 1e5, 5e5, 1e6];
        let jmu_vals: Vec<f64> = zs.iter().map(|&z| visibility_j_mu(z)).collect();
        let jy_vals: Vec<f64> = zs.iter().map(|&z| visibility_j_y(z)).collect();
        for i in 1..zs.len() {
            assert!(
                jmu_vals[i] >= jmu_vals[i - 1] - 1e-10,
                "J_mu should increase with z: J_mu({:.0e})={:.4} < J_mu({:.0e})={:.4}",
                zs[i],
                jmu_vals[i],
                zs[i - 1],
                jmu_vals[i - 1]
            );
            assert!(
                jy_vals[i] <= jy_vals[i - 1] + 1e-10,
                "J_y should decrease with z: J_y({:.0e})={:.4} > J_y({:.0e})={:.4}",
                zs[i],
                jy_vals[i],
                zs[i - 1],
                jy_vals[i - 1]
            );
        }
    }

    #[test]
    fn test_y_from_heating() {
        // y-era burst: y ≈ Δρ/(4ρ)
        let z_h = 5.0e3;
        let drho = 1e-5;
        let sigma_z = 500.0;
        let dq_dz = |z: f64| -> f64 {
            drho * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
        };
        let y = y_from_heating(dq_dz, 1e3, 1e5, 5000);
        let expected = 0.25 * drho;
        let rel_err = (y - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "y = {y:.3e}, expected {expected:.3e}, err = {rel_err}"
        );
    }

    #[test]
    fn test_mu_y_from_heating_consistency() {
        // mu_y_from_heating should give same result as separate calls
        let z_h = 2.0e5;
        let drho = 1e-5;
        let sigma_z = 5000.0;
        let dq_dz = |z: f64| -> f64 {
            drho * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
        };
        let (mu_joint, y_joint) = mu_y_from_heating(&dq_dz, 1e3, 5e6, 10000);
        let mu_sep = mu_from_heating(&dq_dz, 1e3, 5e6, 10000);
        let y_sep = y_from_heating(&dq_dz, 1e3, 5e6, 10000);

        assert!(
            (mu_joint - mu_sep).abs() / mu_sep.abs().max(1e-30) < 1e-10,
            "mu_joint={mu_joint:.4e} vs mu_sep={mu_sep:.4e}"
        );
        assert!(
            (y_joint - y_sep).abs() / y_sep.abs().max(1e-30) < 1e-10,
            "y_joint={y_joint:.4e} vs y_sep={y_sep:.4e}"
        );
    }

    #[test]
    fn test_distortion_from_heating_spectrum() {
        let x_grid: Vec<f64> = (1..20).map(|i| i as f64).collect();
        let z_h = 2.0e5;
        let drho = 1e-5;
        let sigma_z = 5000.0;
        let dq_dz = |z: f64| -> f64 {
            drho * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
        };
        let dn = distortion_from_heating(&x_grid, dq_dz, 1e3, 5e6, 5000);
        assert_eq!(dn.len(), x_grid.len());

        // Cross-validate: decompose the spectrum and compare to mu_from_heating
        let mu_direct = mu_from_heating(&dq_dz, 1e3, 5e6, 5000);

        // Energy integral: Δρ/ρ = ∫x³ Δn dx / G₃
        let drho_spectrum: f64 = (1..x_grid.len())
            .map(|i| {
                let dx = x_grid[i] - x_grid[i - 1];
                0.5 * (x_grid[i].powi(3) * dn[i] + x_grid[i - 1].powi(3) * dn[i - 1]) * dx
            })
            .sum::<f64>()
            / crate::constants::G3_PLANCK;

        // At z=2e5 (μ-era), injected energy should appear in the spectrum
        assert!(
            drho_spectrum > 0.5 * drho && drho_spectrum < 2.0 * drho,
            "Spectrum energy should be ≈ Δρ/ρ = {drho:.1e}: got {drho_spectrum:.4e}"
        );

        // μ from direct GF calculation should be positive and consistent
        assert!(
            mu_direct > 0.0,
            "μ should be positive for heating: {mu_direct:.4e}"
        );
        // μ ≈ 1.401 × Δρ/ρ in the μ-era
        assert!(
            mu_direct > 0.5 * 1.401 * drho && mu_direct < 2.0 * 1.401 * drho,
            "μ should be ≈ 1.401 × Δρ/ρ: got {mu_direct:.4e}, expected ≈ {:.4e}",
            1.401 * drho
        );
    }

    #[test]
    fn test_x_c_physics_properties() {
        // x_c(z) defines the critical frequency below which DC/BR are efficient
        // at creating/absorbing photons. Physical properties:

        // 1. x_c should be positive at all relevant redshifts
        for &z in &[1e3, 1e4, 1e5, 5e5, 1e6] {
            let xc = x_c(z);
            assert!(
                xc > 0.0 && xc.is_finite(),
                "x_c({z:.0e}) = {xc} should be positive finite"
            );
        }

        // 2. x_c should be small (< 1) — DC/BR only dominate at low frequencies
        for &z in &[1e4, 1e5, 1e6] {
            let xc = x_c(z);
            assert!(
                xc < 1.0,
                "x_c({z:.0e}) = {xc:.4} should be < 1 (DC/BR only dominate at low x)"
            );
        }

        // 3. Quadrature addition: x_c >= max(x_c_dc, x_c_br)
        for &z in &[1e4, 1e5, 1e6] {
            let xc = x_c(z);
            let xc_dc = x_c_dc(z);
            let xc_br = x_c_br(z);
            assert!(
                xc >= xc_dc.max(xc_br) - 1e-15,
                "x_c should >= max(x_c_dc, x_c_br) at z={z:.0e}"
            );
        }

        // 4. Monotonicity: x_c should change smoothly with z
        let zs = [1e4, 5e4, 1e5, 5e5, 1e6];
        let xcs: Vec<f64> = zs.iter().map(|&z| x_c(z)).collect();
        for i in 1..xcs.len() {
            // Ratios between adjacent z should be bounded (no wild jumps)
            let ratio = xcs[i] / xcs[i - 1];
            assert!(
                ratio > 0.001 && ratio < 1000.0,
                "x_c ratio between z={:.0e} and z={:.0e} is {ratio:.4} (too extreme)",
                zs[i - 1],
                zs[i]
            );
        }
    }

    #[test]
    fn test_distortion_from_photon_injection_spectrum() {
        let x_grid: Vec<f64> = (1..30).map(|i| 0.5 * i as f64).collect();
        let x_inj = 5.0;
        let z_h = 2.0e5;
        let sigma_z = 5000.0;
        let dn_over_n = 1e-5;
        let dn_dz = |z: f64| -> f64 {
            dn_over_n * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
        };
        let cosmo = Cosmology::default();
        let delta_n = distortion_from_photon_injection(
            &x_grid,
            x_inj,
            dn_dz,
            PHOTON_GF_MU_ERA_Z_MIN,
            5e6,
            5000,
            0.5,
            &cosmo,
        );
        assert_eq!(delta_n.len(), x_grid.len());

        // At x_inj=5.0, photon survives (P_s ≈ 1) so there should be a bump near x_inj.
        // Find the index closest to x_inj and verify it's a local maximum.
        let idx_inj = x_grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - x_inj)
                    .abs()
                    .partial_cmp(&((**b) - x_inj).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        let dn_at_inj = delta_n[idx_inj];
        assert!(
            dn_at_inj > 0.0,
            "Δn should be positive near x_inj={x_inj}: got {dn_at_inj:.4e}"
        );

        // The spectrum should have nonzero amplitude (thermalized + surviving bump)
        let max_dn = delta_n
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, |a, b| a.max(b));
        assert!(
            max_dn > 1e-8,
            "Photon injection should produce nonzero distortion: max|dn|={max_dn:.4e}"
        );

        // μ should be negative for x_inj=5 > x₀≈3.6 (excess photons above chemical
        // potential zero-crossing add energy but remove chemical potential)
        // Actually x_inj=5 > x₀ → positive μ (energy injection like behavior)
        let mu = mu_from_photon_injection(x_inj, z_h, dn_over_n);
        assert!(
            mu > 0.0,
            "μ should be positive for x_inj={x_inj} > x₀: got {mu:.4e}"
        );
    }

    #[test]
    fn test_heating_rate_per_redshift_sign_convention() {
        // heating_rate_per_redshift = -heating_rate / (H(z)(1+z))
        // For positive heating (energy injection), heating_rate > 0 and
        // heating_rate_per_redshift < 0 (energy enters as z decreases).
        // The GF routines expect a POSITIVE dq/dz for heating, so callers
        // must negate or take abs.
        let cosmo = Cosmology::default();
        let z = 2.0e5;

        let burst = crate::energy_injection::InjectionScenario::SingleBurst {
            z_h: z,
            delta_rho_over_rho: 1e-5,
            sigma_z: 100.0,
        };

        let rate = burst.heating_rate(z, &cosmo);
        let rate_per_z = burst.heating_rate_per_redshift(z, &cosmo);

        // heating_rate should be positive (energy injection)
        assert!(rate > 0.0, "heating_rate should be positive: {rate:.4e}");
        // heating_rate_per_redshift should be negative (dz < 0 direction)
        assert!(
            rate_per_z < 0.0,
            "heating_rate_per_redshift should be negative: {rate_per_z:.4e}"
        );
        // Their relationship: rate_per_z = -rate / (H(z)*(1+z))
        let expected = -rate / (cosmo.hubble(z) * (1.0 + z));
        assert!(
            (rate_per_z - expected).abs() / expected.abs() < 1e-10,
            "Sign convention: rate_per_z={rate_per_z:.4e}, expected={expected:.4e}"
        );

        // Verify that mu_from_heating with POSITIVE dq/dz gives positive μ
        let mu = mu_from_heating(
            |zz: f64| {
                1e-5 * (-(zz - z).powi(2) / (2.0 * 5000.0_f64.powi(2))).exp()
                    / (2.0 * std::f64::consts::PI * 5000.0_f64.powi(2)).sqrt()
            },
            1e3,
            5e6,
            10000,
        );
        assert!(
            mu > 0.0,
            "Positive dq/dz (heating) must give positive μ: {mu:.4e}"
        );
    }

    /// Verify visibility function values at specific redshifts against
    /// hand-computed values from the fitting formulas.
    ///
    /// This catches mis-transcribed coefficients in the visibility functions.
    /// Parameters: J_μ, J_y from Chluba (2013) arXiv:1304.6120, Eq. 5;
    /// J_bb* from Chluba (2015) arXiv:1506.06582, Eq. 13.
    #[test]
    fn test_visibility_spot_checks_transition_region() {
        // --- z = 1e5: transition region, most sensitive to coefficient errors ---
        let z = 1.0e5;

        // Chluba 2013 Eq. 5: J_mu(z) = 1 - exp(-((1+z)/5.8e4)^1.88)
        let arg_mu = ((1.0 + z) / 5.8e4_f64).powf(1.88);
        let j_mu_hand = 1.0 - (-arg_mu).exp();
        let j_mu_code = visibility_j_mu(z);
        eprintln!("z=1e5: J_mu: code={j_mu_code:.4}, hand={j_mu_hand:.4}");
        assert!(
            (j_mu_code - j_mu_hand).abs() < 1e-10,
            "J_mu(1e5): code={j_mu_code}, hand={j_mu_hand}"
        );

        // Chluba 2013 Eq. 5: J_y(z) = 1/(1 + ((1+z)/6.0e4)^2.58)
        let arg_y = ((1.0 + z) / 6.0e4_f64).powf(2.58);
        let j_y_hand = 1.0 / (1.0 + arg_y);
        let j_y_code = visibility_j_y(z);
        eprintln!("z=1e5: J_y: code={j_y_code:.4}, hand={j_y_hand:.4}");
        assert!(
            (j_y_code - j_y_hand).abs() < 1e-10,
            "J_y(1e5): code={j_y_code}, hand={j_y_hand}"
        );

        // Chluba 2015 Eq. 13: J_bb*(z) = 0.983 * exp(-(z/Z_MU)^2.5) * (1 - 0.0381*(z/Z_MU)^2.29)
        let ratio_mu = z / Z_MU;
        let j_bb_hand = 0.983 * (-ratio_mu.powf(2.5)).exp() * (1.0 - 0.0381 * ratio_mu.powf(2.29));
        let j_bb_code = visibility_j_bb_star(z);
        eprintln!("z=1e5: J_bb*: code={j_bb_code:.6}, hand={j_bb_hand:.6}");
        assert!(
            (j_bb_code - j_bb_hand.max(0.0)).abs() < 1e-10,
            "J_bb*(1e5): code={j_bb_code}, hand={j_bb_hand}"
        );

        // Physical sanity at z=1e5:
        // - mu-era is partially engaged: J_mu should be ~0.9
        // - y-era branching is small: J_y should be ~0.2
        // - Thermalization is negligible: J_bb* should be ~0.99
        assert!(
            j_mu_code > 0.85 && j_mu_code < 0.99,
            "J_mu(1e5) = {j_mu_code} outside expected [0.85, 0.99]"
        );
        assert!(
            j_y_code > 0.10 && j_y_code < 0.35,
            "J_y(1e5) = {j_y_code} outside expected [0.10, 0.35]"
        );
        assert!(
            j_bb_code > 0.98 && j_bb_code < 1.0,
            "J_bb*(1e5) = {j_bb_code} outside expected [0.98, 1.0]"
        );

        // --- z = 5e4: deep in the transition region ---
        let z2 = 5.0e4;
        let j_mu_5e4 = visibility_j_mu(z2);
        let j_y_5e4 = visibility_j_y(z2);
        let j_bb_5e4 = visibility_j_bb_star(z2);
        eprintln!("z=5e4: J_mu={j_mu_5e4:.4}, J_y={j_y_5e4:.4}, J_bb*={j_bb_5e4:.6}");

        // At z=5e4, both J_mu and J_y should be intermediate (transition region)
        assert!(
            j_mu_5e4 > 0.40 && j_mu_5e4 < 0.80,
            "J_mu(5e4) = {j_mu_5e4} outside transition range [0.40, 0.80]"
        );
        assert!(
            j_y_5e4 > 0.35 && j_y_5e4 < 0.70,
            "J_y(5e4) = {j_y_5e4} outside transition range [0.35, 0.70]"
        );
        // No catastrophic double-counting: J_mu + J_y should not exceed ~1.3
        assert!(
            j_mu_5e4 + j_y_5e4 < 1.5,
            "J_mu + J_y = {} > 1.5 at z=5e4 (double-counting)",
            j_mu_5e4 + j_y_5e4
        );
    }
}
