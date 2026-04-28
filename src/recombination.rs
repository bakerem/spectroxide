//! Hydrogen and helium recombination history.
//!
//! Computes the free electron fraction X_e(z) needed for Thomson scattering
//! rates and number densities throughout the spectral distortion era.
//!
//! ## Physical picture
//!
//! - z > 8000: Fully ionized (H + He). Helium is doubly ionized (He²⁺).
//! - z ~ 6000: He²⁺ recombines to He⁺ (54.4 eV Saha).
//! - z ~ 2000: He⁺ recombines to He (24.6 eV Saha).
//! - z ~ 1500–800: Hydrogen recombines. Saha equilibrium breaks down
//!   due to the Lyman-α bottleneck; the Peebles three-level atom (TLA)
//!   captures the delayed freeze-out.
//! - z < 200: Residual ionization freezes out at X_e ~ 2×10⁻⁴.
//!
//! ## Implementation
//!
//! Follows DarkHistory's three-level atom structure (Hongwan Liu et al. 2020):
//! - `alpha_recomb`: Case-B recombination coefficient (Péquignot fit)
//! - `beta_ion`: Photoionization rate from n=2
//! - `peebles_c`: Peebles C factor decomposed into competing rates
//! - Saha-subtracted ODE form to avoid catastrophic cancellation
//!
//! The fudge factor F=1.125 follows Chluba & Thomas (2011, arXiv:1011.3758),
//! matching DarkHistory. This gives ~1% accuracy in X_e, sufficient for
//! spectral distortion calculations.
//!
//! ## References
//!
//! - Peebles (1968) — Three-level atom model
//! - Péquignot, Petitjean & Boisson (1991) — Case-B recombination fit
//! - Seager, Sasselov & Scott (1999) — RECFAST
//! - Chluba & Thomas (2011, arXiv:1011.3758) — Updated fudge factor
//! - Liu et al. (2020, DarkHistory) — Reference implementation

use crate::constants::*;
use crate::cosmology::Cosmology;

// --- Helium recombination (Saha equilibrium) ---

/// Thermal de Broglie factor: (m_e k_B T / (2π ℏ²))^{3/2} [m⁻³].
///
/// This appears in every Saha equation as the density of states
/// for a free electron.
#[inline]
fn thermal_de_broglie(t: f64) -> f64 {
    (M_ELECTRON * K_BOLTZMANN * t / (2.0 * std::f64::consts::PI * HBAR * HBAR)).powf(1.5)
}

/// Solve the Saha quadratic X²/(1−X) = S for the ionized fraction X.
///
/// Handles extreme limits to avoid overflow/underflow. Used by the hydrogen
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

/// Solve the linear Saha X/(1−X) = S for the ionized fraction X.
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

/// He II → He I Saha ionization fraction (54.4 eV).
///
/// Returns the fraction of helium that is doubly ionized (He²⁺).
/// Statistical weight ratio: g(He²⁺)g(e)/g(He⁺) = 1×2/2 = 1.
/// He²⁺ is a bare alpha particle (spin-0 nucleus), g=1.
/// He⁺ ground state (1s, hydrogen-like), g=2 (electron spin).
/// Free electron, g=2 (spin).
///
/// Uses the standard (RECFAST/Seager+1999) total free-electron Saha form
/// y / (1 − y) = K(T) / n_e, where n_e ≈ n_H + (1 + y_II)·n_He is dominated
/// by H⁺ at z ≳ 1500 (H is fully ionized throughout He recombination since
/// χ_I(H) = 13.6 eV ≪ χ_II(He) = 54.4 eV). Using n_e = n_H + 2·n_He
/// (y_II = 1 limit) introduces ≲7% error in n_e vs the fully self-consistent
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

/// He I → He Saha ionization fraction (24.6 eV).
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
/// n_e ≈ n_H + y_I·n_He; we approximate y_I = 1 to keep the equation linear,
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

/// Helium electron contribution: free electrons per H atom from He.
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

/// Hydrogen Saha ionization fraction.
///
/// Solves X_e²N_H / (1−X_e) = (m_e k_B T / 2πℏ²)^{3/2} exp(−E_Rydberg/kT)
/// for X_e.
pub fn saha_hydrogen(z: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_h = cosmo.n_h(z);

    let s = thermal_de_broglie(t) * (-E_RYDBERG / (K_BOLTZMANN * t)).exp() / n_h;
    solve_saha_quadratic(s)
}

/// Case-B recombination coefficient α_B(T) [m³/s].
///
/// Péquignot, Petitjean & Boisson (1991) fitting formula with
/// fudge factor F = 1.125 (Chluba & Thomas 2011):
///
///   α_B = F × 10⁻¹⁹ × 4.309 × t^{−0.6166} / (1 + 0.6703 × t^{0.5300})
///
/// where t = T / 10⁴ K.
///
/// The fudge factor accounts for higher-order corrections to Case-B
/// recombination (stimulated recombination, two-photon processes).
/// F = 1.14 was used in the original RECFAST; F = 1.125 is the updated
/// value from Chluba & Thomas (2011), also used by DarkHistory.
fn alpha_recomb(t: f64) -> f64 {
    let tt = t / 1.0e4;
    let f = 1.125; // Chluba & Thomas (2011)
    f * 1e-19 * 4.309 * tt.powf(-0.6166) / (1.0 + 0.6703 * tt.powf(0.5300))
}

/// Photoionization rate from n=2 level [s⁻¹].
///
/// From detailed balance with the radiation field at temperature T_CMB:
///
///   β_B = α_B(T_rad) × (m_e k_B T_rad / 2πℏ²)^{3/2} × exp(−E_{n=2}/kT_rad)
///
/// where E_{n=2} = E_Rydberg/4 = 3.4 eV is the ionization energy from n=2.
///
/// **Important**: This uses the radiation temperature T_CMB, NOT the matter
/// temperature, because the photoionizing radiation field is thermal at T_CMB.
fn beta_ion(t_rad: f64) -> f64 {
    let alpha = alpha_recomb(t_rad);
    // The factor of 1/4 from the n=2 statistical weight is already
    // built into E_ION_N2 = E_Rydberg/4
    alpha * thermal_de_broglie(t_rad) * (-E_ION_N2 / (K_BOLTZMANN * t_rad)).exp()
}

/// Peebles C factor: fraction of excited atoms that reach the ground state.
///
/// Decomposition into competing rates:
///
/// - `rate_lya_escape`: Lyman-α escape via Sobolev approximation.
///   Rate = 1/(K_H × n_{1s}) = 8πH / (n_H (1−X_e) λ_Lyα³).
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

    // Sobolev optical depth parameter: K_H = λ_Lyα³ / (8π H)
    let k_h = LAMBDA_LYA.powi(3) / (8.0 * std::f64::consts::PI * h);

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

/// Evaluate the Peebles ODE RHS `dX_h/dz_up = C·α_B·n_H/[H·(1+z)] ·
/// [X_h² − X_S²·(1−X_h)/(1−X_S)]` at the given (z, X_h).
///
/// Here z_up is oriented so that positive `dz_up` corresponds to stepping
/// _downward_ in z (the physical direction of time). The sign convention
/// matches `peebles_step`.
///
/// NOTE: `alpha_recomb` and `beta_ion` are evaluated at the **radiation
/// temperature** T_γ = T_cmb · (1+z), not the matter temperature T_m. During
/// H recombination Compton coupling still enforces T_m ≈ T_γ, so the error is
/// ≲1%. Full accuracy (≲0.1%, as in RECFAST) would require the coupled
/// matter-temperature ODE; this solver's purpose is spectral-distortion
/// templates, where 1–5% disagreement with RECFAST is accepted.
fn peebles_rhs(z: f64, x_h: f64, cosmo: &Cosmology) -> f64 {
    let t = cosmo.t_cmb * (1.0 + z);
    let n_h = cosmo.n_h(z);
    let h = cosmo.hubble(z);

    let c_r = peebles_c(z, x_h.min(1.0), cosmo);
    let alpha = alpha_recomb(t);

    let x_saha = saha_hydrogen(z, cosmo).min(1.0);
    let one_minus_xs = (1.0 - x_saha).max(1e-30);

    let rhs_factor = c_r * alpha * n_h / (h * (1.0 + z));
    let saha_term = x_saha * x_saha * (1.0 - x_h).max(0.0) / one_minus_xs;
    rhs_factor * (x_h * x_h - saha_term)
}

/// Single trapezoidal (Heun's method) step of the Peebles ODE.
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
fn peebles_step(z_new: f64, x_h: f64, dz: f64, cosmo: &Cosmology) -> f64 {
    let z_prev = z_new + dz;
    let k1 = peebles_rhs(z_prev, x_h, cosmo);
    // Evaluate k2 at the predictor, clamped to avoid feeding unphysical
    // values into saha_term / peebles_c.
    let x_pred = (x_h - dz * k1).clamp(1e-5, 1.0);
    let k2 = peebles_rhs(z_new, x_pred, cosmo);
    (x_h - 0.5 * dz * (k1 + k2)).clamp(1e-5, 1.0)
}

/// Find the redshift where the Saha hydrogen X_e first drops below 0.99.
///
/// This is where the Peebles correction becomes significant and we
/// switch from the Saha equation to the TLA ODE.
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

/// Ionization fraction X_e(z) with Peebles TLA correction.
///
/// Returns the total free electron fraction (hydrogen + helium) per
/// hydrogen atom.
///
/// ## Regimes
///
/// - z > 8000: Fully ionized H; He from Saha equations.
/// - z_switch < z ≤ 8000: Saha equilibrium for H + He.
/// - z ≤ z_switch: Peebles three-level atom ODE for H, plus Saha He.
///
/// ## Saha-subtracted ODE
///
/// The raw Peebles ODE has catastrophic cancellation: α_B n_H X_e²
/// and β_B (1−X_e) are both ~10² s⁻¹ but their difference is ~10⁻⁴.
/// Using the Saha relation β_B = α_B X_S² n_H / (1−X_S), we rewrite:
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
        x_e = peebles_step(z_new, x_e, dz_actual, cosmo);
    }

    x_e + helium_electron_fraction(z_end, cosmo)
}

// --- Cached recombination history ---

/// Precomputed recombination history for fast X_e(z) lookups.
///
/// Integrates the Peebles ODE once on construction and stores a table
/// of (z, X_e) pairs. Subsequent lookups use binary search + linear
/// interpolation, making each call O(log N) instead of O(N_ode).
///
/// For z above the Peebles regime (z > z_switch ~ 1575), the cheap
/// Saha formula is used directly (no table needed).
pub struct RecombinationHistory {
    /// Redshifts in descending order (z_switch, z_switch − dz, ..., 1.0)
    z_table: Vec<f64>,
    /// Total X_e (hydrogen + helium) at each redshift
    x_e_table: Vec<f64>,
    /// Redshift where Saha → Peebles switch occurs
    z_switch: f64,
    /// Uniform spacing of z_table (descending): z_table[i] = z_switch − i·dz_table
    dz_table: f64,
    /// Reference cosmology (needed for Saha evaluations above z_switch)
    cosmo: Cosmology,
}

impl RecombinationHistory {
    /// Build the recombination history table for a given cosmology.
    ///
    /// Integrates the Peebles ODE from z_switch down to z=1 with dz=0.5,
    /// storing total X_e (H + He) at each step.
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
            x_h = peebles_step(z_new, x_h, dz_actual, cosmo);

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

    /// Look up X_e(z) using the cached table.
    ///
    /// - z > 8000: fully ionized (Saha for He)
    /// - z_switch < z ≤ 8000: Saha for H + He (cheap, no table)
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

        // At z=50000 (T ~ 136,000 K): He should be fully doubly ionized
        let y_ii = saha_he_ii(50000.0, &cosmo);
        assert!(y_ii > 0.99, "He²⁺ fraction at z=50000 should be ~1: {y_ii}");

        // At z=10000 (T ~ 27,000 K): He II recombining
        let y_ii_mid = saha_he_ii(10000.0, &cosmo);
        eprintln!("He²⁺ at z=10000: {y_ii_mid:.3}");

        // At z=3000 (T ~ 8,200 K): He I should be mostly ionized
        let y_i = saha_he_i(3000.0, &cosmo);
        assert!(y_i > 0.5, "He⁺ fraction at z=3000 should be >0.5: {y_i}");

        // (High-z X_e = 1 + 2f_He tested in test_fully_ionized_high_z)

        // X_e should decrease smoothly through He recombination
        let x_e_8k = ionization_fraction(8000.0, &cosmo);
        let x_e_5k = ionization_fraction(5000.0, &cosmo);
        eprintln!("X_e(z=8000)={x_e_8k:.4}, X_e(z=5000)={x_e_5k:.4}");
        assert!(
            x_e_8k >= x_e_5k,
            "X_e should decrease from z=8000 to z=5000"
        );
    }

    #[test]
    fn test_recombination_history_matches_uncached() {
        let cosmo = Cosmology::default();
        let history = RecombinationHistory::new(&cosmo);

        let test_zs = [
            1e6, 5e4, 8000.0, 5000.0, 1500.0, 1400.0, 1200.0, 1100.0, 1000.0, 800.0, 500.0, 200.0,
            100.0, 10.0,
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

    /// Compare X_e at key redshifts against RECFAST literature values.
    ///
    /// Peebles 3-level atom with fudge factor F=1.125 (Chluba & Thomas 2011)
    /// agrees with RECFAST (Seager, Sasselov & Scott 1999) to ~1-5%.
    ///
    /// For the default cosmology (T_CMB=2.726, Ω_b=0.044, h=0.71, Y_p=0.24):
    ///   X_e(1100) ≈ 0.14 (RECFAST: 0.142 for similar params)
    ///   X_e(800)  ≈ 3e-3 (RECFAST: 0.0034)
    ///   X_e(200)  ≈ 3e-4 (freeze-out)
    #[test]
    fn test_xe_vs_recfast_milestones() {
        let cosmo = Cosmology::default();

        let xe_1100 = ionization_fraction(1100.0, &cosmo);
        let xe_800 = ionization_fraction(800.0, &cosmo);
        let xe_200 = ionization_fraction(200.0, &cosmo);
        eprintln!("X_e(1100) = {xe_1100:.4}  [RECFAST: ~0.14]");
        eprintln!("X_e(800)  = {xe_800:.4e}  [RECFAST: ~3e-3]");
        eprintln!("X_e(200)  = {xe_200:.4e}  [freeze-out: ~3e-4]");

        // z=1100: mid-recombination, RECFAST gives ~0.14
        assert!(
            xe_1100 > 0.10 && xe_1100 < 0.20,
            "X_e(1100) = {xe_1100:.4}, RECFAST expects ~0.14 ± 0.03"
        );

        // z=800: mostly recombined, RECFAST gives ~3e-3
        assert!(
            xe_800 > 5e-4 && xe_800 < 0.01,
            "X_e(800) = {xe_800:.4e}, RECFAST expects ~3e-3"
        );

        // z=200: freeze-out, should be ~2-4 × 10^-4
        assert!(
            xe_200 > 1e-4 && xe_200 < 2e-3,
            "X_e(200) = {xe_200:.4e}, expected freeze-out ~3e-4"
        );
    }
}
