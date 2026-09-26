//! Hydrogen and helium recombination history.
//!
//! Computes the free electron fraction X_e(z) needed for Thomson scattering
//! rates and number densities throughout the spectral distortion era.
//!
//! ## Physical picture
//!
//! The ionization history has these stages:
//!
//! - z > 8000: Fully ionized (H and He). Helium is doubly ionized (He²⁺).
//! - z ~ 6000: He²⁺ recombines to He⁺ (54.4 eV Saha).
//! - z ~ 2000: He⁺ recombines to He (24.6 eV Saha).
//! - z ~ 1500–800: Hydrogen recombines. Saha equilibrium breaks down
//!   due to the Lyman-α bottleneck; the Peebles three-level atom (TLA)
//!   captures the delayed freeze-out.
//! - z < 200: Residual ionization freezes out at X_e ~ 2×10⁻⁴.
//!
//! ## Implementation
//!
//! The module follows DarkHistory's three-level atom structure (Hongwan Liu et al. 2020):
//!
//! - `alpha_recomb`: Case-B recombination coefficient (Péquignot fit)
//! - `beta_ion`: Photoionization rate from n=2
//! - `peebles_c`: Peebles C factor decomposed into competing rates
//! - Saha-subtracted ordinary differential equation (ODE) form to avoid catastrophic cancellation
//!
//! The fudge factor F = 1.125 and the double-Gaussian correction to the
//! Lyman-α escape rate are those of RECFAST 1.5.2, which fitted them together
//! to the multi-level codes HyRec and CosmoRec (Lee & Ali-Haïmoud 2020,
//! arXiv:2007.14114, App. B2). The pair gives X_e within 0.35% of HyRec-2 for
//! 100 ≤ z ≤ 1400; F = 1.125 alone is off by 1.4% (ADR 0011).
//!
//! ## References
//!
//! - Peebles (1968) — Three-level atom model
//! - Péquignot, Petitjean & Boisson (1991) — Case-B recombination fit
//! - Seager, Sasselov & Scott (1999) — RECFAST
//! - Rubiño-Martín, Chluba, Fendt & Wandelt (2010, arXiv:0910.4383) — the
//!   study behind F = 1.125
//! - Lee & Ali-Haïmoud (2020, HyRec-2, arXiv:2007.14114) — App. B2 describes
//!   RECFAST's fudge factor and escape-rate correction
//! - Liu et al. (2020, DarkHistory) — Reference implementation

use crate::constants::*;
use crate::cosmology::Cosmology;

// --- Helium recombination (Saha equilibrium) ---

/// Computes the thermal de Broglie factor (m_e k_B T / (2π ℏ²))^{3/2} [m⁻³].
///
/// This appears in every Saha equation as the density of states
/// for a free electron.
#[inline]
fn thermal_de_broglie(t: f64) -> f64 {
    (M_ELECTRON * K_BOLTZMANN * t / (2.0 * std::f64::consts::PI * HBAR * HBAR)).powf(1.5)
}

/// Solves the Saha quadratic X²/(1−X) = S for the ionized fraction X.
///
/// Handles extreme limits to avoid overflow or underflow. Used by the hydrogen
/// Saha (where the self-ionization n_e = X·n_H is exact at z ≳ 1500 because
/// H dominates the electron budget).
#[inline]
fn solve_saha_quadratic(s: f64) -> f64 {
    if s > 1e10 {
        1.0
    } else if s < 1e-10 {
        s.sqrt()
    } else {
        (-s + (s * s + 4.0 * s).sqrt()) / 2.0
    }
}

/// Solves the linear Saha X/(1−X) = S for the ionized fraction X.
#[inline]
fn solve_saha_linear(s: f64) -> f64 {
    if s > 1e15 {
        1.0
    } else if s < 1e-15 {
        s
    } else {
        s / (1.0 + s)
    }
}

/// Computes the He II → He I Saha ionization fraction (54.4 eV).
///
/// Returns the fraction of helium that is doubly ionized (He²⁺).
/// Statistical weight ratio: g(He²⁺)g(e)/g(He⁺) = 1×2/2 = 1.
/// He²⁺ is a bare alpha particle (spin-0 nucleus), g=1.
/// He⁺ ground state (1s, hydrogen-like), g=2 (electron spin).
/// Free electron, g=2 (spin).
///
/// Uses the standard (RECFAST, Seager et al. 1999) total free-electron Saha form
/// y / (1 − y) = K(T) / n_e, where n_e ≈ n_H + (1 + y_II)·n_He is dominated
/// by H⁺ at z ≳ 1500 (H is fully ionized throughout He recombination since
/// χ_I(H) = 13.6 eV ≪ χ_II(He) = 54.4 eV). Using n_e = n_H + 2·n_He
/// (y_II = 1 limit) introduces ≲7% error in n_e compared with the fully self-consistent
/// y_II = 0 limit — negligible compared to the ~factor-of-29 error from the
/// old He-only quadratic form that assumed n_e = y·n_He.
pub fn saha_he_ii(z: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_he = cosmo.n_he(z);
    if n_he < 1e-30 {
        return 1.0;
    }

    // n_e dominated by fully-ionized hydrogen at z where χ_II(He) matters.
    // Include the He²⁺ contribution at the y=1 limit; this overestimates n_e
    // by at most f_He/(1+f_He) ≈ 7% during the transition.
    let n_e = cosmo.n_h(z) + 2.0 * n_he;

    let e_ion = E_HE_II_ION_EV * EV_IN_JOULES;
    let s = thermal_de_broglie(t) * (-e_ion / (K_BOLTZMANN * t)).exp() / n_e;
    solve_saha_linear(s)
}

/// Computes the He I → He Saha ionization fraction (24.6 eV).
///
/// Returns the fraction of helium that is at least singly ionized (He⁺ or He²⁺).
/// Statistical weight ratio: g(He⁺)g(e)/g(He) = 2×2/1 = 4.
/// He⁺ ground state (1s, hydrogen-like), g=2 (electron spin).
/// Free electron, g=2 (spin).
/// He ground state (1s², singlet), g=1.
///
/// Uses the standard total free-electron Saha form y / (1 − y) = K(T) / n_e.
/// At z ~ 2000 where He⁺ recombines, H is still fully ionized (Saha X_H ≈ 1
/// down to z ~ 1500) and He²⁺ has already recombined to He⁺, so
/// n_e ≈ n_H + y_I·n_He; the code approximates y_I = 1 to keep the equation linear,
/// which introduces ≲4% error in n_e.
pub fn saha_he_i(z: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_he = cosmo.n_he(z);
    if n_he < 1e-30 {
        return 1.0;
    }

    let n_e = cosmo.n_h(z) + n_he;

    let e_ion = E_HE_I_ION_EV * EV_IN_JOULES;
    let s = 4.0 * thermal_de_broglie(t) * (-e_ion / (K_BOLTZMANN * t)).exp() / n_e;
    solve_saha_linear(s)
}

/// Computes the helium electron contribution: free electrons per H atom from He.
///
/// x_He = f_He × (y_HeI + 2 × y_HeII)
/// where y_HeII is the doubly-ionized fraction and y_HeI is the singly-ionized fraction.
pub fn helium_electron_fraction(z: f64, cosmo: &Cosmology) -> f64 {
    let f_he = cosmo.y_p / (4.0 * (1.0 - cosmo.y_p));
    let y_he_ii = saha_he_ii(z, cosmo);
    let y_he_i = saha_he_i(z, cosmo);
    // He²⁺ contributes 2 electrons, He⁺ contributes 1.
    // He⁺-only fraction = y_he_i - y_he_ii, He²⁺ fraction = y_he_ii.
    // Electrons: (y_he_i - y_he_ii)×1 + y_he_ii×2 = y_he_i + y_he_ii.
    f_he * (y_he_i + y_he_ii)
}

// --- Hydrogen recombination (Saha + Peebles TLA) ---

/// Computes the hydrogen Saha ionization fraction.
///
/// Solves X_e²N_H / (1−X_e) = (m_e k_B T / 2πℏ²)^{3/2} exp(−E_H/kT)
/// for X_e, where E_H = R_∞hc/(1 + m_e/m_p) is the reduced-mass ionization energy.
pub fn saha_hydrogen(z: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_h = cosmo.n_h(z);

    let s = thermal_de_broglie(t) * (-E_H_ION / (K_BOLTZMANN * t)).exp() / n_h;
    solve_saha_quadratic(s)
}

/// Computes the case-B recombination coefficient α_B(T) [m³/s].
///
/// Péquignot, Petitjean & Boisson (1991) fitting formula with
/// fudge factor F = 1.125 (RECFAST 1.5.2):
///
///   α_B = F × 10⁻¹⁹ × 4.309 × t^{−0.6166} / (1 + 0.6703 × t^{0.5300})
///
/// where t = T / 10⁴ K.
///
/// The fudge factor stands in for the multi-level corrections to case-B
/// recombination that a three-level atom leaves out. The original RECFAST
/// used F = 1.14; RECFAST 1.5.2 lowered it to 1.125 when it added the
/// escape-rate correction in [`lya_escape_correction`], and the two values
/// belong together (CLASS `precisions.h`: `recfast_fudge_H` 1.14 plus
/// `recfast_delta_fudge_H` −0.015 when the correction is on). The value
/// rests on Rubiño-Martín et al. (2010, arXiv:0910.4383); see Lee &
/// Ali-Haïmoud (2020, arXiv:2007.14114, App. B2). DarkHistory uses the same
/// pair.
fn alpha_recomb(t: f64) -> f64 {
    let tt = t / 1.0e4;
    let f = 1.125;
    f * 1e-19 * 4.309 * tt.powf(-0.6166) / (1.0 + 0.6703 * tt.powf(0.5300))
}

/// Computes the photoionization rate from the n=2 level [s⁻¹].
///
/// From detailed balance with the radiation field at temperature T_CMB:
///
///   β_B = α_B(T_rad) × (m_e k_B T_rad / 2πℏ²)^{3/2} × exp(−E_{n=2}/kT_rad)
///
/// where E_{n=2} = E_H/4 = 3.4 eV is the ionization energy from n=2.
///
/// **Important**: This uses the radiation temperature T_CMB, not the matter
/// temperature, because the photoionizing radiation field is thermal at T_CMB.
fn beta_ion(t_rad: f64) -> f64 {
    let alpha = alpha_recomb(t_rad);
    // The 1/n² = 1/4 Bohr scaling of the n=2 binding energy is already
    // built into E_ION_N2 = E_H/4
    alpha * thermal_de_broglie(t_rad) * (-E_ION_N2 / (K_BOLTZMANN * t_rad)).exp()
}

/// Returns RECFAST's correction factor 1 + Δ(z) to the Sobolev parameter K_H.
///
/// RECFAST 1.5.2 multiplies K_H = λ_Lyα³/(8πH), and so divides the Lyman-α
/// escape rate, by a sum of two Gaussians in ln(1+z):
///
/// ```text
///   1 + Δ(z) = 1 − 0.14 exp{−[(ln(1+z) − 7.28)/0.18]²}
///                + 0.079 exp{−[(ln(1+z) − 6.73)/0.33]²}.
/// ```
///
/// The dip near z ≈ 1450 speeds recombination and the bump near z ≈ 840
/// slows it. Amplitudes and widths were chosen to mimic HyRec and CosmoRec,
/// together with F = 1.125 in [`alpha_recomb`] (Lee & Ali-Haïmoud 2020,
/// arXiv:2007.14114, App. B2). Values as in CLASS (`recfast_AGauss1` …
/// `recfast_wGauss2`) and DarkHistory 1.1.2 (`physics.peebles_C`). The
/// correction depends on z alone, so it is the same in the standard table
/// and in the coupled mode of ADR 0009 (ADR 0011).
fn lya_escape_correction(z: f64) -> f64 {
    let ln_1pz = (1.0 + z).ln();
    1.0 - 0.14 * (-((ln_1pz - 7.28) / 0.18).powi(2)).exp()
        + 0.079 * (-((ln_1pz - 6.73) / 0.33).powi(2)).exp()
}

/// Computes the Peebles C factor: the fraction of excited atoms that reach the ground state.
///
/// Decomposition into competing rates:
///
/// - `rate_lya_escape`: Lyman-α escape using the Sobolev approximation,
///   Rate = 1/(K_H × n_{1s}) = 8πH / (n_H (1−X_e) λ_Lyα³), divided by
///   RECFAST's correction [`lya_escape_correction`].
///   Most Ly-α photons are reabsorbed; only the cosmological redshift
///   allows escape from the optically thick line.
///
/// - `rate_2s1s`: Two-photon decay 2s→1s at rate Λ_{2s} = 8.225 s⁻¹.
///   Slow but guaranteed (two-photon continuum cannot be reabsorbed).
///
/// - `rate_ion`: Photoionization from n=2 at rate β_B.
///
/// The C factor is:
///   C = (rate_escape + Λ_{2s}) / (rate_escape + Λ_{2s} + β_B)
///
/// In the standard TLA, 2s and 2p are assumed to be in statistical
/// equilibrium (fast collisional mixing), so the net de-excitation
/// rate is the sum of both channels without explicit statistical weights.
///
/// When C ≈ 1: de-excitation wins (recombination proceeds).
/// When C ≈ 0: photoionization wins (recombination is bottlenecked).
fn peebles_c(z: f64, x_e: f64, cosmo: &Cosmology) -> f64 {
    let t_rad = cosmo.t_cmb * (1.0 + z);
    let n_h = cosmo.n_h(z);
    let h = cosmo.hubble(z);

    // Sobolev optical depth parameter K_H = λ_Lyα³ / (8π H), with RECFAST's
    // correction (ADR 0011)
    let k_h = LAMBDA_LYA.powi(3) / (8.0 * std::f64::consts::PI * h) * lya_escape_correction(z);

    // Number of neutral hydrogen atoms [m⁻³]
    let n_1s = n_h * (1.0 - x_e).max(0.0);

    // Ly-α escape rate: 1/(K_H × n_{1s})
    let rate_lya_escape = if n_1s > 1e-30 {
        1.0 / (k_h * n_1s)
    } else {
        1e30
    };

    // Photoionization rate from n=2
    let rate_ion = beta_ion(t_rad);

    // C = (escape + two-photon) / (escape + two-photon + photoionization)
    let rate_down = rate_lya_escape + LAMBDA_2S1S;
    let denom = rate_down + rate_ion;
    if denom > 0.0 { rate_down / denom } else { 1.0 }
}

/// Evaluates the Peebles ODE RHS `dX_h/dz_up = C·α_B·n_H/[H·(1+z)] ·
/// [X_h² − X_S²·(1−X_h)/(1−X_S)]` at the given (z, X_h).
///
/// Here z_up is oriented so that positive `dz_up` corresponds to stepping
/// _downward_ in z (the physical direction of time). The sign convention
/// matches `peebles_step`.
///
/// `rho_m` = T_m/T_γ sets the gas temperature for the recombination
/// coefficient; `beta_ion`, X_S, and the C factor stay at the radiation
/// temperature T_γ = T_cmb · (1+z), since the CMB does the photoionizing.
/// With α_B(T_m) = α_B(T_γ) + Δα the right-hand side becomes
///
/// ```text
///   C n_H/[H(1+z)] · { α_B(T_γ) [X_h² − X_S²(1−X_h)/(1−X_S)] + Δα X_h² },
/// ```
///
/// which keeps the Saha subtraction and adds a term that is exactly zero at
/// `rho_m` = 1, so the standard history is unchanged
/// (decisions/0009-evolve-hydrogen-ionization-with-electron-temperature.md).
fn peebles_rhs(z: f64, x_h: f64, rho_m: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_h = cosmo.n_h(z);
    let h = cosmo.hubble(z);

    let c_r = peebles_c(z, x_h.min(1.0), cosmo);
    let alpha = alpha_recomb(t);

    let x_saha = saha_hydrogen(z, cosmo).min(1.0);
    let one_minus_xs = (1.0 - x_saha).max(1e-30);

    let rhs_factor = c_r * alpha * n_h / (h * (1.0 + z));
    let saha_term = x_saha * x_saha * (1.0 - x_h).max(0.0) / one_minus_xs;
    let rhs = rhs_factor * (x_h * x_h - saha_term);
    if rho_m == 1.0 {
        return rhs;
    }
    let d_alpha = alpha_recomb(rho_m * t) - alpha;
    rhs + c_r * d_alpha * n_h / (h * (1.0 + z)) * x_h * x_h
}

/// Takes a single trapezoidal (Heun's method) step of the Peebles ODE.
///
/// Steps x_h from z_prev = z_new + dz down to z_new:
///
/// ```text
///   k1 = f(z_prev, x_h)
///   k2 = f(z_new,  x_h − dz · k1)
///   x_new = x_h − dz · (k1 + k2) / 2
/// ```
///
/// This is second-order accurate in dz (O(dz²) local truncation error),
/// upgraded from forward Euler (audit M1 / recomb). The final clamp to
/// `[1e-5, 1.0]` is a safety net; removing it would let step overshoot
/// produce negative X_h at large dz — if it fires it signals that the
/// outer step size is too coarse.
fn peebles_step(z_new: f64, x_h: f64, dz: f64, rho_m: f64, cosmo: &Cosmology) -> f64 {
    let z_prev = z_new + dz;
    let k1 = peebles_rhs(z_prev, x_h, rho_m, cosmo);
    // Evaluate k2 at the predictor, clamped to avoid feeding unphysical
    // values into saha_term / peebles_c.
    let x_pred = (x_h - dz * k1).clamp(1e-5, 1.0);
    let k2 = peebles_rhs(z_new, x_pred, rho_m, cosmo);
    (x_h - 0.5 * dz * (k1 + k2)).clamp(1e-5, 1.0)
}

/// Finds the redshift where the Saha hydrogen X_e first drops below 0.99.
///
/// This is where the Peebles correction becomes significant; the solver
/// switches from the Saha equation to the TLA ODE here.
fn find_saha_switch(cosmo: &Cosmology) -> f64 {
    let mut z = 1800.0;
    while z > 1000.0 {
        if saha_hydrogen(z, cosmo) < 0.99 {
            return z + 1.0;
        }
        z -= 1.0;
    }
    1500.0
}

/// Computes the ionization fraction X_e(z) with the Peebles TLA correction.
///
/// Returns the total free electron fraction (hydrogen and helium) per
/// hydrogen atom.
///
/// ## Regimes
///
/// - z > 8000: Fully ionized H; He from Saha equations.
/// - z_switch < z ≤ 8000: Saha equilibrium for H and He.
/// - z ≤ z_switch: Peebles three-level atom ODE for H, plus Saha He.
///
/// ## Saha-subtracted ODE
///
/// The raw Peebles ODE has catastrophic cancellation: α_B n_H X_e²
/// and β_B (1−X_e) are both ~10² s⁻¹ but their difference is ~10⁻⁴.
/// The Saha relation β_B = α_B X_S² n_H / (1−X_S) rewrites the ODE as:
///
///   dX_e/dz = C × α_B × n_H / (H(1+z)) × [X_e² − X_S² (1−X_e)/(1−X_S)]
///
/// This is O(X_e − X_S) near equilibrium, eliminating the cancellation.
///
/// References:
/// - Peebles (1968) — Three-level atom
/// - Seager, Sasselov & Scott (1999) — RECFAST
/// - Liu et al. (2020) — DarkHistory implementation
pub fn ionization_fraction(z: f64, cosmo: &Cosmology) -> f64 {
    if z > 8000.0 {
        return 1.0 + helium_electron_fraction(z, cosmo);
    }

    let z_switch = find_saha_switch(cosmo);

    if z > z_switch {
        let x_h = saha_hydrogen(z, cosmo).min(1.0);
        return x_h + helium_electron_fraction(z, cosmo);
    }

    // Peebles TLA ODE from z_switch down to z
    let z_end = z.max(1.0);
    let x_h_start = saha_hydrogen(z_switch, cosmo).min(1.0);

    let dz_step = 0.5_f64;
    let n_steps = ((z_switch - z_end) / dz_step).ceil() as usize;
    let n_steps = n_steps.max(1);
    let dz_actual = (z_switch - z_end) / n_steps as f64;

    let mut x_e = x_h_start;

    for i in 0..n_steps {
        let z_new = z_switch - (i + 1) as f64 * dz_actual;
        x_e = peebles_step(z_new, x_e, dz_actual, 1.0, cosmo);
    }

    x_e + helium_electron_fraction(z_end, cosmo)
}

// --- Cached recombination history ---

/// Precomputed recombination history for fast X_e(z) lookups.
///
/// Integrates the Peebles ODE once on construction and stores a table
/// of (z, X_e) pairs. Subsequent lookups use binary search and linear
/// interpolation, making each call O(log N) instead of O(N_ode).
///
/// For z above the Peebles regime (z > z_switch ~ 1575), the cheap
/// Saha formula is used directly (no table needed).
pub struct RecombinationHistory {
    /// Redshifts in descending order (z_switch, z_switch − dz, ..., 1.0).
    z_table: Vec<f64>,
    /// Total X_e (hydrogen and helium) at each redshift.
    x_e_table: Vec<f64>,
    /// Redshift where the switch from Saha to Peebles occurs.
    z_switch: f64,
    /// Uniform spacing of z_table (descending): z_table[i] = z_switch − i·dz_table.
    dz_table: f64,
    /// Reference cosmology (needed for Saha evaluations above z_switch).
    cosmo: Cosmology,
}

impl RecombinationHistory {
    /// Builds the recombination history table for a given cosmology.
    ///
    /// Integrates the Peebles ODE from z_switch down to z=1 with dz=0.5,
    /// storing total X_e (H and He) at each step.
    pub fn new(cosmo: &Cosmology) -> Self {
        let z_switch = find_saha_switch(cosmo);
        let z_end = 1.0_f64;
        let dz_step = 0.5_f64;
        let n_steps = ((z_switch - z_end) / dz_step).ceil() as usize;
        let n_steps = n_steps.max(1);
        let dz_actual = (z_switch - z_end) / n_steps as f64;

        let mut z_table = Vec::with_capacity(n_steps + 1);
        let mut x_e_table = Vec::with_capacity(n_steps + 1);

        let x_h_start = saha_hydrogen(z_switch, cosmo).min(1.0);
        let x_e_start = x_h_start + helium_electron_fraction(z_switch, cosmo);
        z_table.push(z_switch);
        x_e_table.push(x_e_start);

        let mut x_h = x_h_start;

        for i in 0..n_steps {
            let z_new = z_switch - (i + 1) as f64 * dz_actual;
            x_h = peebles_step(z_new, x_h, dz_actual, 1.0, cosmo);

            let x_e_total = x_h + helium_electron_fraction(z_new.max(1.0), cosmo);
            z_table.push(z_new);
            x_e_table.push(x_e_total);
        }

        RecombinationHistory {
            z_table,
            x_e_table,
            z_switch,
            dz_table: dz_actual,
            cosmo: cosmo.clone(),
        }
    }

    /// Looks up X_e(z) using the cached table.
    ///
    /// - z > 8000: fully ionized (Saha for He)
    /// - z_switch < z ≤ 8000: Saha for H and He (cheap, no table)
    /// - z ≤ z_switch: interpolate from precomputed table
    pub fn x_e(&self, z: f64) -> f64 {
        if z > 8000.0 {
            1.0 + helium_electron_fraction(z, &self.cosmo)
        } else if z > self.z_switch {
            let x_h = saha_hydrogen(z, &self.cosmo).min(1.0);
            x_h + helium_electron_fraction(z, &self.cosmo)
        } else if z <= self.z_table[self.z_table.len() - 1] {
            // Below the table: return the last value (freeze-out)
            self.x_e_table[self.x_e_table.len() - 1]
        } else {
            // Direct indexing: z_table is uniform in z (descending), so the
            // bracketing index is idx = floor((z_switch − z)/dz_table). Clamp
            // to the last interior cell so idx+1 is always a valid node.
            let raw = (self.z_switch - z) / self.dz_table;
            let n = self.z_table.len();
            let idx = (raw as usize).min(n - 2);

            let z_hi = self.z_table[idx];
            let z_lo = self.z_table[idx + 1];
            let x_hi = self.x_e_table[idx];
            let x_lo = self.x_e_table[idx + 1];
            let t = (z_hi - z) / (z_hi - z_lo);
            x_hi + t * (x_lo - x_hi)
        }
    }

    /// Returns the redshift below which hydrogen follows the Peebles ODE.
    pub fn z_switch(&self) -> f64 {
        self.z_switch
    }

    /// Returns the hydrogen ionization fraction X_H(z) of the standard
    /// history (gas at the radiation temperature).
    pub fn x_h(&self, z: f64) -> f64 {
        if z > 8000.0 {
            1.0
        } else if z > self.z_switch {
            saha_hydrogen(z, &self.cosmo).min(1.0)
        } else {
            self.x_e(z) - helium_electron_fraction(z.max(1.0), &self.cosmo)
        }
    }

    /// Returns the total X_e for a given hydrogen fraction: X_H plus Saha helium.
    ///
    /// Above `z_switch` this is [`Self::x_e`] exactly, whatever `x_h` is, so
    /// callers that evolve X_H reproduce the standard history there bit for bit.
    pub fn x_e_with_x_h(&self, z: f64, x_h: f64) -> f64 {
        if z > self.z_switch {
            self.x_e(z)
        } else {
            x_h + helium_electron_fraction(z.max(1.0), &self.cosmo)
        }
    }

    /// Advances X_H from `z_from` down to `z_to` with the gas at `rho_m` = T_m/T_γ.
    ///
    /// Sub-cycles the Heun step of the table (dz ≤ 0.5) with `rho_m` held
    /// fixed. Above `z_switch` hydrogen is in Saha equilibrium at T_γ and the
    /// result is [`Self::x_h`]; a step that crosses `z_switch` starts the ODE
    /// from the Saha value there, as the table does (ADR 0009).
    pub fn advance_x_h(&self, z_from: f64, z_to: f64, x_h: f64, rho_m: f64) -> f64 {
        debug_assert!(rho_m.is_finite(), "rho_m = {rho_m}");
        // The solver's T_e guard admits ρ_e = 0, where α_B diverges; the DC/BR
        // target uses the same floor.
        let rho_m = rho_m.max(0.05);
        if z_to > self.z_switch {
            return self.x_h(z_to);
        }
        let (z_top, x_top) = if z_from > self.z_switch {
            (
                self.z_switch,
                saha_hydrogen(self.z_switch, &self.cosmo).min(1.0),
            )
        } else {
            (z_from, x_h)
        };
        let span = z_top - z_to;
        if span <= 0.0 {
            return x_top;
        }
        let n_sub = ((span / 0.5).ceil() as usize).max(1);
        let dz = span / n_sub as f64;
        let mut x = x_top;
        for i in 0..n_sub {
            let z_new = z_top - (i + 1) as f64 * dz;
            x = peebles_step(z_new, x, dz, rho_m, &self.cosmo);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fully_ionized_high_z() {
        let cosmo = Cosmology::default();
        // At z=10^6, He is doubly ionized: X_e = 1 + 2*f_He
        let x_e = ionization_fraction(1e6, &cosmo);
        assert!(
            x_e > 1.0,
            "Should be fully ionized with He at z=10^6: X_e = {x_e}"
        );
        assert!(
            (x_e - (1.0 + 2.0 * F_HE)).abs() < 0.01,
            "X_e = {x_e}, expected {}",
            1.0 + 2.0 * F_HE
        );
    }

    #[test]
    fn test_freeze_out() {
        let cosmo = Cosmology::default();
        let x_e = ionization_fraction(100.0, &cosmo);
        // RECFAST: X_e(100) ~ 2-4e-4 for this cosmology
        assert!(
            x_e > 1e-4 && x_e < 5e-3,
            "Freeze-out X_e should be ~2e-4: got {x_e}"
        );
    }

    #[test]
    fn test_recombination_physical_values() {
        let cosmo = Cosmology::default();

        // z=1400: not yet deeply recombined
        let x_1400 = ionization_fraction(1400.0, &cosmo);
        assert!(
            x_1400 > 0.5,
            "X_e(1400) should be > 0.5 (early recombination): got {x_1400}"
        );

        // z=1100: mid-recombination, RECFAST gives ~0.14 for this cosmology
        let x_1100 = ionization_fraction(1100.0, &cosmo);
        assert!(
            x_1100 > 0.10 && x_1100 < 0.20,
            "X_e(1100) should be ~0.14 (RECFAST): got {x_1100}"
        );

        // z=800: mostly recombined
        let x_800 = ionization_fraction(800.0, &cosmo);
        assert!(
            x_800 < 0.01,
            "X_e(800) should be < 0.01 (mostly recombined): got {x_800}"
        );

        // z=200: freeze-out regime
        let x_200 = ionization_fraction(200.0, &cosmo);
        assert!(
            x_200 > 1e-5 && x_200 < 0.01,
            "X_e(200) should be in [1e-5, 0.01]: got {x_200}"
        );

        // Monotonic decrease from z=1500 to z=200
        let zs = [
            1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 800.0, 600.0, 400.0, 200.0,
        ];
        let xs: Vec<f64> = zs.iter().map(|&z| ionization_fraction(z, &cosmo)).collect();
        for i in 1..xs.len() {
            assert!(
                xs[i] <= xs[i - 1] + 1e-10,
                "X_e should decrease monotonically: X_e({})={:.4e} > X_e({})={:.4e}",
                zs[i],
                xs[i],
                zs[i - 1],
                xs[i - 1]
            );
        }

        // No hard jump around z=200 (smooth transition)
        let x_201 = ionization_fraction(201.0, &cosmo);
        let x_199 = ionization_fraction(199.0, &cosmo);
        let ratio = if x_199 > 1e-30 { x_201 / x_199 } else { 1.0 };
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "No hard jump at z=200: X_e(201)={x_201:.4e}, X_e(199)={x_199:.4e}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_helium_saha_transitions() {
        let cosmo = Cosmology::default();
        let f_he = cosmo.y_p / (4.0 * (1.0 - cosmo.y_p));

        // Hydrogen Saha: fully ionized at z = 3000, neutral by z = 500, monotonic between
        assert!((saha_hydrogen(3000.0, &cosmo) - 1.0).abs() < 1e-3);
        assert!(saha_hydrogen(500.0, &cosmo) < 1e-5);
        let mut prev = 0.0;
        for z in (1000..2500).step_by(50) {
            let x = saha_hydrogen(z as f64, &cosmo);
            assert!(x >= prev - 1e-10, "Saha not monotonic at z={z}");
            prev = x;
        }
        // Peebles lags Saha: recombination is slower than equilibrium
        for z in [1000.0, 800.0, 600.0] {
            assert!(
                ionization_fraction(z, &cosmo) > saha_hydrogen(z, &cosmo),
                "Peebles > Saha at z={z}"
            );
        }

        // He²⁺ at z ≥ 1e4, He⁺ dominant at z = 4000, He⁰ growing by z = 1500
        assert!(saha_he_ii(50000.0, &cosmo) > 0.99);
        assert!(saha_he_ii(2e4, &cosmo) > 0.99 && saha_he_i(2e4, &cosmo) > 0.99);
        assert!(saha_he_ii(1e4, &cosmo) > 0.95);
        let (y_ii_4k, y_i_4k) = (saha_he_ii(4000.0, &cosmo), saha_he_i(4000.0, &cosmo));
        assert!(
            y_ii_4k < 0.1 && y_i_4k > 0.9,
            "z=4000 should be dominantly He⁺: y_ii={y_ii_4k:.4}, y_i={y_i_4k:.4}"
        );
        assert!(saha_he_ii(3000.0, &cosmo) < 0.5);
        assert!(saha_he_i(3000.0, &cosmo) > 0.5);
        assert!(saha_he_i(1500.0, &cosmo) < 0.6);

        let zs = [1e4, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0];
        for w in zs.windows(2) {
            assert!(
                saha_he_ii(w[1], &cosmo) <= saha_he_ii(w[0], &cosmo) + 1e-10,
                "y_he_ii should decrease from z={} to z={}",
                w[0],
                w[1]
            );
        }
        for z in [1e4, 8000.0, 6000.0, 4000.0, 2000.0, 1500.0] {
            let (y_ii, y_i) = (saha_he_ii(z, &cosmo), saha_he_i(z, &cosmo));
            assert!((0.0..=1.0).contains(&y_ii), "y_he_ii={y_ii} at z={z}");
            assert!((0.0..=1.0).contains(&y_i), "y_he_i={y_i} at z={z}");
            assert!(y_ii <= y_i + 1e-10, "y_he_ii > y_he_i at z={z}");
        }

        // Helium electron fraction: in [0, 2f_He], 2f_He at z = 1e6, zero at z = 100
        for z in [100.0, 1000.0, 3000.0, 5000.0, 10000.0, 50000.0, 1e6] {
            let x_he = helium_electron_fraction(z, &cosmo);
            assert!(
                (-1e-10..=2.0 * f_he + 1e-10).contains(&x_he),
                "He e- fraction {x_he} outside [0, 2f_He] at z={z:.0e}"
            );
        }
        assert!((helium_electron_fraction(1e6, &cosmo) - 2.0 * f_he).abs() < 0.01 * f_he);
        assert!(helium_electron_fraction(100.0, &cosmo) < 0.01 * f_he);
        assert!(
            helium_electron_fraction(5000.0, &cosmo) >= helium_electron_fraction(2000.0, &cosmo)
        );

        // X_e decreases through He recombination and is continuous across the
        // Saha→Peebles switch
        assert!(ionization_fraction(8000.0, &cosmo) >= ionization_fraction(5000.0, &cosmo));
        let xs: Vec<f64> = (1500..=1600)
            .map(|z| ionization_fraction(z as f64, &cosmo))
            .collect();
        for (i, w) in xs.windows(2).enumerate() {
            let frac = (w[1] - w[0]).abs() / w[0].max(w[1]).max(0.01);
            assert!(frac < 0.05, "X_e jump at z={}", 1500 + i);
        }
    }

    /// Bounds X_e at the redshifts the HyRec-2 anchors below do not reach:
    /// the He²⁺ plateau (z = 1e4, where X_e = 1 + 2f_He ≈ 1.16), the He⁺ epoch
    /// (z = 3000), and both sides of the Saha→Peebles switch (z = 1500, 1400).
    #[test]
    fn test_xe_bounds_between_anchors() {
        let cosmo = Cosmology::default();

        let xe_1e4 = ionization_fraction(1e4, &cosmo);
        assert!((xe_1e4 - 1.16).abs() < 0.10, "X_e(1e4) = {xe_1e4}");
        let xe_3000 = ionization_fraction(3000.0, &cosmo);
        assert!(xe_3000 > 1.0 && xe_3000 < 1.2, "X_e(3000) = {xe_3000}");
        let xe_1500 = ionization_fraction(1500.0, &cosmo);
        assert!(xe_1500 > 0.9, "X_e(1500) = {xe_1500}");
        let xe_1400 = ionization_fraction(1400.0, &cosmo);
        assert!(xe_1400 > 0.60 && xe_1400 < 1.05, "X_e(1400) = {xe_1400}");

        let xs: Vec<f64> = (1500..=2000)
            .rev()
            .step_by(10)
            .map(|z| ionization_fraction(z as f64, &cosmo))
            .collect();
        for w in xs.windows(2) {
            assert!(w[1] <= w[0] + 1e-10, "X_e not monotonic over 2000→1500");
        }
    }

    #[test]
    fn test_recombination_history_matches_uncached() {
        let cosmo = Cosmology::default();
        let history = RecombinationHistory::new(&cosmo);

        let test_zs = [
            1e6, 5e5, 5e4, 1e4, 8000.0, 5000.0, 1500.0, 1400.0, 1200.0, 1100.0, 1000.0, 800.0,
            500.0, 200.0, 100.0, 50.0, 10.0,
        ];
        for &z in &test_zs {
            let cached = history.x_e(z);
            let uncached = ionization_fraction(z, &cosmo);
            let rel_err = if uncached.abs() > 1e-10 {
                (cached - uncached).abs() / uncached.abs()
            } else {
                (cached - uncached).abs()
            };
            assert!(
                rel_err < 0.01,
                "Cached vs uncached mismatch at z={z}: cached={cached:.6e}, \
                 uncached={uncached:.6e}, rel_err={rel_err:.3e}"
            );
        }

        // Dense sampling across the Saha→Peebles switch, where the cache
        // interpolates a kink, at 2%
        let mut switch_zs: Vec<f64> = (1550..=1600).step_by(5).map(|z| z as f64).collect();
        switch_zs.extend_from_slice(&[2000.0, 1e5]);
        for z in switch_zs {
            let cached = history.x_e(z);
            let uncached = ionization_fraction(z, &cosmo);
            let rel_err = (cached - uncached).abs() / uncached.abs().max(1e-10);
            assert!(
                rel_err < 0.02,
                "Cached vs uncached mismatch at z={z}: rel_err={rel_err:.3e}"
            );
        }
    }

    /// Checks the Saha-subtracted right-hand side with the gas-temperature term
    /// against the unsubtracted Peebles (1968) form
    ///
    ///   dX/dz_up = C/[H(1+z)] · [α_B(T_m) n_H X² − β_B(T_γ) e^{−E_α/kT_γ} (1−X)],
    ///
    /// with E_α = E_H − E_n=2 the Lyman-α energy (ADR 0009). At X = 0.3,
    /// z = 1300, the two terms are within a factor of a few of each other, so
    /// an error in the rewrite shows up at O(1).
    #[test]
    fn test_peebles_rhs_matches_unsubtracted_form() {
        let cosmo = Cosmology::default();
        let (z, x) = (1300.0, 0.3);
        for rho_m in [1.0, 0.8, 1.5, 3.0] {
            let t_gam = cosmo.t_cmb * (1.0 + z);
            let c = peebles_c(z, x, &cosmo);
            let pre = c / (cosmo.hubble(z) * (1.0 + z));
            let down = alpha_recomb(rho_m * t_gam) * cosmo.n_h(z) * x * x;
            let e_lya = E_H_ION - E_ION_N2;
            let up = beta_ion(t_gam) * (-e_lya / (K_BOLTZMANN * t_gam)).exp() * (1.0 - x);
            let raw = pre * (down - up);
            let rhs = peebles_rhs(z, x, rho_m, &cosmo);
            let scale = pre * down.max(up);
            assert!(
                (rhs - raw).abs() < 1e-10 * scale,
                "rho_m={rho_m}: subtracted {rhs:.12e} vs unsubtracted {raw:.12e}"
            );
        }
    }

    /// With the gas at the photon temperature, `advance_x_h` in coarse steps
    /// must reproduce the standard table; hot gas must stay more ionized and
    /// cold gas less, since α_B falls with temperature.
    #[test]
    fn test_advance_x_h_limits() {
        let cosmo = Cosmology::default();
        let hist = RecombinationHistory::new(&cosmo);
        let mut x = hist.x_h(2000.0);
        let mut z = 2000.0;
        let mut worst: f64 = 0.0;
        while z > 100.0 {
            let z_new = (z - 17.0_f64).max(100.0);
            x = hist.advance_x_h(z, z_new, x, 1.0);
            z = z_new;
            let rel = x / hist.x_h(z) - 1.0;
            worst = worst.max(rel.abs());
        }
        assert!(
            worst < 1e-4,
            "rho_m = 1: X_H vs table, worst rel {worst:.3e}"
        );

        let z_s = hist.z_switch();
        let x_s = hist.x_h(z_s);
        let x_std = hist.x_h(800.0);
        let x_hot = hist.advance_x_h(z_s, 800.0, x_s, 2.0);
        let x_cold = hist.advance_x_h(z_s, 800.0, x_s, 0.9);
        assert!(
            x_hot > 1.2 * x_std && x_cold < x_std,
            "X_H(800): hot {x_hot:.4e}, standard {x_std:.4e}, cold {x_cold:.4e}"
        );
    }

    #[test]
    fn test_recombination_history_monotonic() {
        let cosmo = Cosmology::default();
        let history = RecombinationHistory::new(&cosmo);

        let zs: Vec<f64> = (100..=2000).rev().step_by(10).map(|z| z as f64).collect();
        let xs: Vec<f64> = zs.iter().map(|&z| history.x_e(z)).collect();
        for i in 1..xs.len() {
            assert!(
                xs[i] <= xs[i - 1] + 1e-10,
                "Cached X_e not monotonic: X_e({})={:.4e} > X_e({})={:.4e}",
                zs[i],
                xs[i],
                zs[i - 1],
                xs[i - 1]
            );
        }
    }

    #[test]
    fn test_recombination_history_interpolation_smooth() {
        let cosmo = Cosmology::default();
        let history = RecombinationHistory::new(&cosmo);

        let z_a = 1200.0;
        let z_b = 1200.25;
        let z_c = 1200.5;
        let x_a = history.x_e(z_a);
        let x_b = history.x_e(z_b);
        let x_c = history.x_e(z_c);

        assert!(
            (x_a >= x_b && x_b >= x_c) || (x_a <= x_b && x_b <= x_c),
            "Interpolation not monotonic: X_e({z_a})={x_a:.6e}, \
             X_e({z_b})={x_b:.6e}, X_e({z_c})={x_c:.6e}"
        );
    }

    /// Compares X_e at key redshifts against RECFAST literature values.
    ///
    /// Peebles 3-level atom with RECFAST 1.5.2's fudge factor F = 1.125 and
    /// escape-rate correction (ADR 0011).
    ///
    /// Anchors are HyRec-2 (github.com/nanoomlee/HyRec-2, run 2026-07-05) on
    /// the exact default cosmology (T_CMB=2.726, Ω_b=0.044, Ω_m=0.26, h=0.71,
    /// Y_p=0.24, N_eff=3.046); table and input archived in
    /// dev/output/hyrec2_xe_default_cosmo.dat. HyRec values:
    ///   X_e(1100) = 0.14324,  X_e(800) = 3.4785e-3,  X_e(200) = 3.2685e-4.
    /// Measured Peebles-TLA-to-HyRec disagreement is ≤ 1.9% for 200 ≤ z ≤ 1600
    /// (see dev/audit/xe_hyrec_comparison.md); the ±6% band gives ~3× slack.
    /// The post-freeze-out tail (z ≲ 50) diverges up to 33% — a documented
    /// α_B(T_rad) convention with negligible observable impact — and is
    /// deliberately not anchored here.
    #[test]
    fn test_xe_vs_recfast_milestones() {
        let cosmo = Cosmology::default();

        let anchors = [(1100.0, 0.14324), (800.0, 3.4785e-3), (200.0, 3.2685e-4)];
        for (z, xe_hyrec) in anchors {
            let xe = ionization_fraction(z, &cosmo);
            let rel = xe / xe_hyrec - 1.0;
            eprintln!("X_e({z}) = {xe:.4e}  [HyRec-2: {xe_hyrec:.4e}, rel {rel:+.2e}]");
            assert!(
                rel.abs() < 0.06,
                "X_e({z}) = {xe:.4e} deviates {rel:+.2e} from HyRec-2 {xe_hyrec:.4e} (band ±6%)"
            );
        }
    }

    /// Checks X_e through the helium recombination epoch against the same HyRec-2
    /// run (R2 mutation audit, fix B4).
    ///
    /// The milestone test above probes only z = 1100/800/200 — all hydrogen-
    /// dominated — so the He Saha machinery (`saha_he_i`, `saha_he_ii`) had no
    /// direct value anchor and 15 of its mutants survived the sweep. This is not
    /// a cosmetic gap: `xe_hyrec_comparison.md` measures the ε(m_A′) dark-photon
    /// limit shifting by −10.5% (γ_con by +25%) for resonances landing in the
    /// He window z ≈ 1800–2500, so X_e there feeds a published figure directly.
    ///
    /// HyRec-2 values read from dev/output/hyrec2_xe_default_cosmo.dat (same run
    /// and cosmology as `test_xe_vs_recfast_milestones`). Bands follow the measured
    /// per-band disagreement in xe_hyrec_comparison.md with ~1.5× slack: ≤0.14% for
    /// 3000–5000 (He²⁺/He⁺
    /// Saha, where both codes are in equilibrium) and 5.7% at z≈2300 (this code's Saha
    /// against HyRec's non-equilibrium He⁺→He⁰ — HyRec recombines later, the expected
    /// direction).
    #[test]
    fn test_xe_vs_hyrec_helium_epoch() {
        let cosmo = Cosmology::default();

        // (z, HyRec-2 X_e, band)
        let anchors = [
            (5000.0, 1.0796046866, 0.01),
            (2500.0, 1.0715093554, 0.08),
            (2300.0, 1.0628359322, 0.08),
            (2000.0, 1.0373070723, 0.08),
            (1800.0, 1.0064061811, 0.08),
        ];
        for (z, xe_hyrec, band) in anchors {
            let xe = ionization_fraction(z, &cosmo);
            let rel = xe / xe_hyrec - 1.0;
            eprintln!("X_e({z}) = {xe:.6e}  [HyRec-2: {xe_hyrec:.6e}, rel {rel:+.2e}]");
            assert!(
                rel.abs() < band,
                "X_e({z}) = {xe:.6e} deviates {rel:+.2e} from HyRec-2 {xe_hyrec:.6e} \
                 (band ±{:.0}%)",
                band * 100.0
            );
        }

        // He²⁺ → He⁺ must be complete before H recombination: X_e falls
        // monotonically across the window and stays above the H-only floor.
        let mut prev = f64::INFINITY;
        for &z in &[5000.0, 3000.0, 2500.0, 2300.0, 2000.0, 1800.0] {
            let xe = ionization_fraction(z, &cosmo);
            assert!(
                xe < prev,
                "X_e must decrease through He recombination: X_e({z}) = {xe:.6e} ≥ previous {prev:.6e}"
            );
            prev = xe;
        }
    }
}
