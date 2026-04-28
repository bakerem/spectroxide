//! Background cosmology: Hubble rate, cosmic time, number densities.
//!
//! Implements a flat ΛCDM background with radiation (photons + neutrinos).

use crate::constants::*;

/// Cosmological parameters with cached derived quantities.
#[derive(Debug, Clone)]
pub struct Cosmology {
    /// CMB temperature today, in K.
    pub t_cmb: f64,
    /// Physical baryon density ω_b = Ω_b h²
    pub omega_b: f64,
    /// Physical CDM density ω_cdm = Ω_cdm h²
    pub omega_cdm: f64,
    /// Dimensionless Hubble parameter h = H₀/(100 km/s/Mpc)
    pub h: f64,
    /// Effective number of neutrino species
    pub n_eff: f64,
    /// Primordial helium mass fraction Y_p
    pub y_p: f64,

    // --- Cached derived quantities (computed once at construction) ---
    cached_h0: f64,
    cached_omega_b_frac: f64,
    cached_omega_cdm_frac: f64,
    cached_omega_m: f64,
    cached_omega_gamma: f64,
    cached_omega_rel: f64,
    cached_omega_lambda: f64,
    /// (1-Y_p) * rho_b0 / M_PROTON — multiply by (1+z)³ to get n_H(z)
    cached_n_h_prefactor: f64,
    /// Y_p / (4(1-Y_p))
    cached_f_he: f64,
    /// 3 Ω_b / (4 Ω_γ) — multiply by 1/(1+z) to get R(z)
    cached_baryon_photon_prefactor: f64,
    /// ρ_γ(z=0) = (π²/15)(kT₀)⁴/(ℏc)³ — multiply by (1+z)⁴ to get ρ_γ(z)
    cached_rho_gamma_0: f64,
}

impl Cosmology {
    /// Construct a Cosmology with all cached derived quantities.
    /// Validate cosmological parameters.
    ///
    /// Returns `Err` with a descriptive message if any parameter is
    /// non-physical or would cause numerical failure.
    pub fn validate(&self) -> Result<(), String> {
        if !self.t_cmb.is_finite() || self.t_cmb <= 0.0 {
            return Err(format!(
                "t_cmb must be positive and finite, got {}",
                self.t_cmb
            ));
        }
        if !self.omega_b.is_finite() || self.omega_b <= 0.0 {
            return Err(format!(
                "omega_b must be positive and finite, got {}",
                self.omega_b
            ));
        }
        if !self.omega_cdm.is_finite() || self.omega_cdm < 0.0 {
            return Err(format!(
                "omega_cdm must be non-negative and finite, got {}",
                self.omega_cdm
            ));
        }
        if !self.h.is_finite() || self.h <= 0.0 {
            return Err(format!("h must be positive and finite, got {}", self.h));
        }
        if self.h > 10.0 {
            return Err(format!(
                "h={} implies H0={} km/s/Mpc, which is nonsensical",
                self.h,
                self.h * 100.0
            ));
        }
        if !self.n_eff.is_finite() || self.n_eff < 0.0 || self.n_eff > 20.0 {
            return Err(format!("n_eff must be in [0, 20], got {}", self.n_eff));
        }
        if !self.y_p.is_finite() || self.y_p < 0.0 || self.y_p >= 1.0 {
            return Err(format!("y_p must be in [0, 1), got {}", self.y_p));
        }
        Ok(())
    }

    /// Construct a Cosmology from dimensionless parameters, validating the inputs.
    ///
    /// Returns `Err` if any parameter is non-finite, negative where physical
    /// positivity is required, or outside the supported range.
    pub fn new(
        t_cmb: f64,
        omega_b: f64,
        omega_cdm: f64,
        h: f64,
        n_eff: f64,
        y_p: f64,
    ) -> Result<Self, String> {
        // Validate before computing derived quantities to avoid div-by-zero / Inf
        let tmp = Cosmology {
            t_cmb,
            omega_b,
            omega_cdm,
            h,
            n_eff,
            y_p,
            cached_h0: 0.0,
            cached_omega_b_frac: 0.0,
            cached_omega_cdm_frac: 0.0,
            cached_omega_m: 0.0,
            cached_omega_gamma: 0.0,
            cached_omega_rel: 0.0,
            cached_omega_lambda: 0.0,
            cached_n_h_prefactor: 0.0,
            cached_f_he: 0.0,
            cached_baryon_photon_prefactor: 0.0,
            cached_rho_gamma_0: 0.0,
        };
        tmp.validate()?;
        Ok(Self::new_unchecked(
            t_cmb, omega_b, omega_cdm, h, n_eff, y_p,
        ))
    }

    /// Construct a Cosmology without validation.
    ///
    /// Escape hatch for hardcoded presets and test fixtures where the inputs
    /// are known a priori to be valid. Using non-finite or zero `h` / `y_p`
    /// here will produce `NaN`/`Inf` in derived quantities — prefer [`Self::new`]
    /// for any input the caller does not fully control.
    pub fn new_unchecked(
        t_cmb: f64,
        omega_b: f64,
        omega_cdm: f64,
        h: f64,
        n_eff: f64,
        y_p: f64,
    ) -> Self {
        let h0 = 100.0 * h * KM_PER_MPC;
        let h2 = h * h;
        let omega_b_frac = omega_b / h2;
        let omega_cdm_frac = omega_cdm / h2;
        let omega_m = omega_b_frac + omega_cdm_frac;

        let rho_gamma = std::f64::consts::PI.powi(2) / 15.0 * K_BOLTZMANN.powi(4) * t_cmb.powi(4)
            / (HBAR.powi(3) * C_LIGHT.powi(3));
        let rho_crit = 3.0 * h0 * h0 * C_LIGHT * C_LIGHT / (8.0 * std::f64::consts::PI * G_NEWTON);
        let omega_gamma = rho_gamma / rho_crit;
        let omega_rel =
            omega_gamma * (1.0 + n_eff * (7.0 / 8.0) * (4.0_f64 / 11.0).powf(4.0 / 3.0));
        let omega_lambda = 1.0 - omega_m - omega_rel;

        // n_H prefactor: (1-Y_p) * rho_b0 / M_PROTON
        // rho_b0 = Omega_b * rho_crit (rho_crit without c² since we use SI densities)
        let rho_crit_mass = 3.0 * h0 * h0 / (8.0 * std::f64::consts::PI * G_NEWTON);
        let rho_b0 = omega_b_frac * rho_crit_mass;
        let n_h_prefactor = (1.0 - y_p) * rho_b0 / M_PROTON;

        let f_he = y_p / (4.0 * (1.0 - y_p));
        let baryon_photon_prefactor = 3.0 * omega_b_frac / (4.0 * omega_gamma);

        Cosmology {
            t_cmb,
            omega_b,
            omega_cdm,
            h,
            n_eff,
            y_p,
            cached_h0: h0,
            cached_omega_b_frac: omega_b_frac,
            cached_omega_cdm_frac: omega_cdm_frac,
            cached_omega_m: omega_m,
            cached_omega_gamma: omega_gamma,
            cached_omega_rel: omega_rel,
            cached_omega_lambda: omega_lambda,
            cached_n_h_prefactor: n_h_prefactor,
            cached_f_he: f_he,
            cached_baryon_photon_prefactor: baryon_photon_prefactor,
            cached_rho_gamma_0: rho_gamma,
        }
    }

    /// Planck 2015 cosmological parameters (matching CosmoTherm DI files).
    pub fn planck2015() -> Self {
        Self::new_unchecked(2.726, 0.02225, 0.1198, 0.6727, 3.046, 0.2467)
    }

    /// Planck 2018 cosmological parameters (Planck Collaboration VI, 2020).
    pub fn planck2018() -> Self {
        // omega_b = Omega_b * h^2 = 0.04930 * 0.6736^2 = 0.02237
        // omega_cdm = (Omega_m - Omega_b) * h^2 = (0.3153 - 0.04930) * 0.6736^2 = 0.1200
        Self::new_unchecked(2.7255, 0.02237, 0.1200, 0.6736, 3.044, 0.2454)
    }

    /// H₀ in 1/s
    #[inline]
    pub fn h0(&self) -> f64 {
        self.cached_h0
    }

    /// Ω_b = ω_b / h²
    #[inline]
    pub fn omega_b_frac(&self) -> f64 {
        self.cached_omega_b_frac
    }

    /// Ω_cdm = ω_cdm / h²
    #[inline]
    pub fn omega_cdm_frac(&self) -> f64 {
        self.cached_omega_cdm_frac
    }

    /// Ω_m = Ω_b + Ω_cdm
    #[inline]
    pub fn omega_m(&self) -> f64 {
        self.cached_omega_m
    }

    /// Ω_γ (photon density parameter)
    #[inline]
    pub fn omega_gamma(&self) -> f64 {
        self.cached_omega_gamma
    }

    /// Ω_rel (all relativistic species: photons + neutrinos)
    #[inline]
    pub fn omega_rel(&self) -> f64 {
        self.cached_omega_rel
    }

    /// Ω_Λ = 1 - Ω_m - Ω_rel (flat universe)
    #[inline]
    pub fn omega_lambda(&self) -> f64 {
        self.cached_omega_lambda
    }

    /// E(z) = H(z)/H₀ = sqrt(Ω_m(1+z)³ + Ω_rel(1+z)⁴ + Ω_Λ)
    #[inline]
    pub fn e_of_z(&self, z: f64) -> f64 {
        let opz = 1.0 + z;
        (self.cached_omega_m * opz.powi(3)
            + self.cached_omega_rel * opz.powi(4)
            + self.cached_omega_lambda)
            .sqrt()
    }

    /// Dimensionless photon temperature: θ_z(z) = k_B T_z / (m_e c²)
    /// Uses this cosmology's T_CMB rather than the hardcoded default.
    #[inline]
    pub fn theta_z(&self, z: f64) -> f64 {
        crate::constants::K_BOLTZMANN * self.t_cmb * (1.0 + z) / crate::constants::M_E_C2
    }

    /// Hubble rate H(z) in 1/s
    #[inline]
    pub fn hubble(&self, z: f64) -> f64 {
        self.cached_h0 * self.e_of_z(z)
    }

    /// dt/dz in seconds
    pub fn dt_dz(&self, z: f64) -> f64 {
        -1.0 / (self.hubble(z) * (1.0 + z))
    }

    /// Matter-radiation equality redshift
    pub fn z_eq(&self) -> f64 {
        self.cached_omega_m / self.cached_omega_rel - 1.0
    }

    /// Hydrogen number density at z [1/m³]
    /// N_H = (1 - Y_p) ρ_b / m_p
    #[inline]
    pub fn n_h(&self, z: f64) -> f64 {
        self.cached_n_h_prefactor * (1.0 + z).powi(3)
    }

    /// Helium number density at z [1/m³]
    /// N_He = Y_p / (4(1-Y_p)) × N_H
    #[inline]
    pub fn n_he(&self, z: f64) -> f64 {
        self.cached_f_he * self.n_h(z)
    }

    /// Helium-to-hydrogen ratio f_He = Y_p / (4(1-Y_p))
    #[inline]
    pub fn f_he(&self) -> f64 {
        self.cached_f_he
    }

    /// Free electron density at z (1/m³), given ionization fraction X_e.
    #[inline]
    pub fn n_e(&self, z: f64, x_e: f64) -> f64 {
        x_e * self.n_h(z)
    }

    /// Thomson scattering time t_C = 1/(σ_T N_e c), in s.
    #[inline]
    pub fn t_compton(&self, z: f64, x_e: f64) -> f64 {
        1.0 / (SIGMA_THOMSON * self.n_e(z, x_e) * C_LIGHT)
    }

    /// Photon energy density ρ_γ at z [J/m³]
    #[inline]
    pub fn rho_gamma(&self, z: f64) -> f64 {
        self.cached_rho_gamma_0 * (1.0 + z).powi(4)
    }

    /// Photon number density n_γ at z [1/m³]
    pub fn n_gamma(&self, z: f64) -> f64 {
        // n_γ = (2ζ(3)/π²) (kT/ℏc)³
        let kt_over_hbar_c = K_BOLTZMANN * self.t_cmb * (1.0 + z) / (HBAR * C_LIGHT);
        2.0 * ZETA_3 / std::f64::consts::PI.powi(2) * kt_over_hbar_c.powi(3)
    }

    /// Baryon-to-photon energy density ratio R(z) = 3ρ_b/(4ρ_γ).
    ///
    /// In the tight-coupling limit this sets the sound speed:
    ///   c_s² = 1 / (3(1+R)).
    #[inline]
    pub fn baryon_photon_ratio(&self, z: f64) -> f64 {
        self.cached_baryon_photon_prefactor / (1.0 + z)
    }

    /// Compton y-parameter integrated from z' = 0 to z.
    ///
    /// y_C(z) = ∫₀ᶻ (kT_e / m_e c²) × σ_T n_e c / ((1+z') H(z')) dz'
    ///
    /// Quantifies the total Compton optical depth for energy exchange.
    /// At z >> 1100: y_C >> 1 (efficient Comptonization).
    /// At z << 1100: y_C << 1 (Compton scattering inefficient).
    ///
    /// Uses 128-point midpoint quadrature in ln(1+z).
    pub fn compton_y_parameter(&self, z: f64) -> f64 {
        if z < 1e-6 {
            return 0.0;
        }
        let recomb = crate::recombination::RecombinationHistory::new(self);
        self.compton_y_parameter_with_recomb(z, &recomb)
    }

    /// Compton y-parameter with a pre-built RecombinationHistory.
    ///
    /// Use this variant in loops to avoid rebuilding the recombination
    /// table (~3000 ODE steps) on every call.
    pub fn compton_y_parameter_with_recomb(
        &self,
        z: f64,
        recomb: &crate::recombination::RecombinationHistory,
    ) -> f64 {
        if z < 1e-6 {
            return 0.0;
        }

        let ln_min = 0.0_f64; // ln(1+0)
        let ln_max = (1.0 + z).ln();

        let n = 128;
        let h = (ln_max - ln_min) / (n as f64);
        let mut result = 0.0;

        for i in 0..n {
            let u = ln_min + (i as f64 + 0.5) * h;
            let zp = u.exp() - 1.0;
            let opz = 1.0 + zp;

            let x_e = recomb.x_e(zp);
            let n_e = self.n_e(zp, x_e);
            // Matter temperature tracks T_γ while Compton coupling is strong
            // (z ≳ z_dec ≈ 200) and drops as (1+z)² after decoupling. Using
            // T_γ below z_dec overestimates the integrand and, multiplied by
            // the already-tiny post-recombination n_e, produces a spurious
            // contribution to J_Compton (audit M1 / infra). The n_e factor
            // makes this a few-% effect at worst, but the bias is avoidable.
            const Z_DEC: f64 = 200.0;
            let theta_e = if zp > Z_DEC {
                K_BOLTZMANN * self.t_cmb * opz / M_E_C2
            } else {
                // T_m = T_γ(z_dec) · ((1+z)/(1+z_dec))² = T_cmb · (1+z)² / (1+z_dec)
                let ratio = opz / (1.0 + Z_DEC);
                K_BOLTZMANN * self.t_cmb * (1.0 + Z_DEC) * ratio * ratio / M_E_C2
            };

            // Integrand in ln(1+z): θ_e × σ_T × c × n_e / H(z')
            // The (1+z) from dz = (1+z) d(ln(1+z)) cancels with the 1/(1+z) in the formula
            result += theta_e * SIGMA_THOMSON * C_LIGHT * n_e / self.hubble(zp) * h;
        }

        result
    }

    /// Cosmic time t(z) by numerical integration, in s.
    ///
    /// Integrates from z_upper (≈ 10^9) down to z.
    pub fn cosmic_time(&self, z: f64) -> f64 {
        let u_low = (1.0 + z).ln();
        let u_high = (1.0 + 1.0e9_f64).ln();

        let n = 64;
        let h = (u_high - u_low) / (n as f64);
        let mut result = 0.0;
        for i in 0..n {
            let u = u_low + (i as f64 + 0.5) * h;
            let zp = u.exp() - 1.0;
            let dzp_du = 1.0 + zp;
            result += (1.0 / (self.hubble(zp) * (1.0 + zp))) * dzp_du * h;
        }
        result
    }
}

impl Default for Cosmology {
    /// Default parameters matching Chluba (2013) Green's function paper.
    ///
    /// These are intentionally **not** the latest Planck values. The defaults
    /// match CosmoTherm v1.0.3 (Y_p=0.24, T₀=2.726 K, Ω_m=0.26, Ω_b=0.044,
    /// h=0.71, N_eff=3.046) so that PDE output can be validated against
    /// published CosmoTherm results without cosmology mismatch.
    ///
    /// For Planck-era parameters use [`Cosmology::planck2015`] or
    /// [`Cosmology::planck2018`].
    fn default() -> Self {
        Self::new_unchecked(
            T_CMB_0,
            0.044 * 0.71 * 0.71, // Ω_b=0.044, h=0.71
            (0.26 - 0.044) * 0.71 * 0.71,
            0.71,
            N_EFF,
            Y_P,
        )
    }
}

/// Midpoint quadrature nodes and weights on [a, b].
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let cosmo = Cosmology::default();
        assert!(
            (cosmo.omega_m() - 0.26).abs() < 0.001,
            "Omega_m = {}, expected ~0.26",
            cosmo.omega_m()
        );
        assert!(
            (cosmo.omega_b_frac() - 0.044).abs() < 0.001,
            "Omega_b = {}, expected ~0.044",
            cosmo.omega_b_frac()
        );
    }

    #[test]
    fn test_hubble_today() {
        let cosmo = Cosmology::default();
        let h0 = cosmo.hubble(0.0);
        let h0_expected = 100.0 * 0.71 * KM_PER_MPC;
        assert!(
            (h0 - h0_expected).abs() / h0_expected < 1e-10,
            "H(0) = {h0:.3e}, expected {h0_expected:.3e}"
        );
    }

    #[test]
    fn test_z_eq() {
        let cosmo = Cosmology::default();
        let z_eq = cosmo.z_eq();
        // For default params (Ω_m=0.26, h=0.71, T_cmb=2.726), z_eq ≈ 3300
        assert!(
            z_eq > 3000.0 && z_eq < 3600.0,
            "z_eq = {z_eq}, expected ~3300 for default cosmology"
        );
    }

    #[test]
    fn test_e_of_z_today() {
        let cosmo = Cosmology::default();
        assert!((cosmo.e_of_z(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_radiation_dominated() {
        let cosmo = Cosmology::default();
        let z = 1.0e6;
        let e = cosmo.e_of_z(z);
        let e_rad = cosmo.omega_rel().sqrt() * (1.0 + z).powi(2);
        assert!(
            (e - e_rad).abs() / e_rad < 0.01,
            "E({z}) = {e:.3e}, radiation approx = {e_rad:.3e}"
        );
    }

    #[test]
    fn test_n_h_positive_and_scaling() {
        let cosmo = Cosmology::default();
        assert!(cosmo.n_h(0.0) > 0.0);
        assert!(cosmo.n_h(1000.0) > cosmo.n_h(0.0));
        // n_H scales as (1+z)³
        let ratio = cosmo.n_h(1000.0) / cosmo.n_h(0.0);
        let expected = 1001.0_f64.powi(3);
        assert!(
            (ratio / expected - 1.0).abs() < 1e-10,
            "n_H should scale as (1+z)³: ratio = {ratio:.6e}, expected = {expected:.6e}"
        );
    }

    #[test]
    fn test_cosmic_time_order_and_magnitude() {
        let cosmo = Cosmology::default();
        let t1 = cosmo.cosmic_time(1000.0);
        let t2 = cosmo.cosmic_time(100.0);
        assert!(t2 > t1, "t(z=100) should be > t(z=1000)");
        // t(z=0) ~ age of universe ~ 13.7 Gyr ≈ 4.3e17 s
        let t0 = cosmo.cosmic_time(0.0);
        assert!(
            t0 > 3e17 && t0 < 5e17,
            "t(z=0) = {t0:.3e} s, expected ~4.3e17 s (13.7 Gyr)"
        );
    }

    #[test]
    fn test_compton_y_parameter_high_z() {
        let cosmo = Cosmology::default();
        // At z=1e5, y_C should be moderate (limited by recombination at z~1100)
        let yc = cosmo.compton_y_parameter(1e5);
        assert!(yc > 0.1, "y_C(1e5) = {yc:.3e}, should be > 0.1");
        // At z=1e6 (deep in fully ionized era), y_C should be large
        let yc_deep = cosmo.compton_y_parameter(1e6);
        assert!(
            yc_deep > 10.0,
            "y_C(1e6) = {yc_deep:.3e}, should be >> 1 (deep thermalization era)"
        );
    }

    #[test]
    fn test_compton_y_parameter_low_z() {
        let cosmo = Cosmology::default();
        // At z << 1100, y_C should be small (inefficient Comptonization)
        let yc = cosmo.compton_y_parameter(500.0);
        assert!(
            yc < 0.1,
            "y_C(500) = {yc:.3e}, should be << 1 (post-recombination)"
        );
    }

    #[test]
    fn test_compton_y_parameter_monotonic() {
        let cosmo = Cosmology::default();
        let y1 = cosmo.compton_y_parameter(500.0);
        let y2 = cosmo.compton_y_parameter(2000.0);
        let y3 = cosmo.compton_y_parameter(1e5);
        assert!(
            y3 > y2 && y2 > y1,
            "y_C should be monotonically increasing: {y1:.3e} < {y2:.3e} < {y3:.3e}"
        );
    }

    #[test]
    fn test_baryon_photon_ratio() {
        let cosmo = Cosmology::default();
        // R = ρ_b / (4/3 ρ_γ), scales as 1/(1+z)
        let r_0 = cosmo.baryon_photon_ratio(0.0);
        assert!(
            r_0 > 100.0,
            "R(0) = {r_0:.1}, expected >> 1 (matter dominated today)"
        );

        let r_1100 = cosmo.baryon_photon_ratio(1100.0);
        // At recombination, R ~ 0.6
        assert!(
            r_1100 > 0.1 && r_1100 < 2.0,
            "R(1100) = {r_1100:.3}, expected ~0.6"
        );

        // Should scale as 1/(1+z)
        let ratio = r_0 / r_1100;
        assert!(
            (ratio - 1101.0).abs() / 1101.0 < 0.01,
            "R should scale as 1/(1+z)"
        );
    }

    #[test]
    fn test_density_accessors() {
        let cosmo = Cosmology::default();
        // Test various density accessors for physical sanity
        assert!(cosmo.omega_gamma() > 0.0);
        assert!(cosmo.omega_rel() > cosmo.omega_gamma()); // includes neutrinos
        assert!(cosmo.omega_lambda() > 0.0);
        assert!(cosmo.omega_cdm_frac() > 0.0);
        assert!(cosmo.rho_gamma(0.0) > 0.0);
        assert!(cosmo.n_he(0.0) > 0.0);
        assert!(cosmo.n_e(0.0, 1.0) > 0.0);
        assert!(cosmo.t_compton(1e4, 1.0) > 0.0);
        assert!(cosmo.dt_dz(1e4) < 0.0); // dt/dz < 0 (time increases as z decreases)

        // Quantitative checks against known values
        // n_gamma(0) = (2ζ(3)/π²)(kT/ℏc)³ ≈ 4.1e8 /m³ for T=2.726 K
        let n_gam = cosmo.n_gamma(0.0);
        assert!(
            (n_gam - 4.1e8).abs() / 4.1e8 < 0.02,
            "n_gamma(0) = {n_gam:.3e}, expected ~4.1e8 /m³"
        );
        // n_H(0) = (1-Y_p) × 3H₀²Ω_b/(8πG m_p) ≈ 0.19 /m³ for default params
        let n_h = cosmo.n_h(0.0);
        assert!(
            (n_h - 0.19).abs() / 0.19 < 0.05,
            "n_H(0) = {n_h:.4}, expected ~0.19 /m³"
        );
    }

    #[test]
    fn test_planck2015_preset() {
        let cosmo = Cosmology::planck2015();
        assert!((cosmo.h - 0.6727).abs() < 0.001);
        assert!((cosmo.omega_b - 0.02225).abs() < 1e-5);
        assert!((cosmo.y_p - 0.2467).abs() < 1e-4);
    }

    #[test]
    fn test_planck2018_preset() {
        let cosmo = Cosmology::planck2018();
        assert!(
            (cosmo.h - 0.6736).abs() < 0.001,
            "h={}, expected 0.6736",
            cosmo.h
        );
        assert!(
            (cosmo.omega_b - 0.02237).abs() < 1e-5,
            "omega_b={}, expected 0.02237",
            cosmo.omega_b
        );
        assert!(
            (cosmo.omega_cdm - 0.1200).abs() < 1e-4,
            "omega_cdm={}, expected 0.1200",
            cosmo.omega_cdm
        );
        assert!(
            (cosmo.t_cmb - 2.7255).abs() < 0.001,
            "t_cmb={}, expected 2.7255",
            cosmo.t_cmb
        );
        assert!(
            (cosmo.y_p - 0.2454).abs() < 1e-4,
            "y_p={}, expected 0.2454",
            cosmo.y_p
        );
        assert!(
            (cosmo.n_eff - 3.044).abs() < 0.01,
            "n_eff={}, expected 3.044",
            cosmo.n_eff
        );
        // Derived quantities should be valid
        assert!(cosmo.hubble(0.0) > 0.0);
        assert!(cosmo.n_h(0.0) > 0.0);
    }

    #[test]
    fn test_cached_f_he() {
        let cosmo = Cosmology::default();
        let expected = cosmo.y_p / (4.0 * (1.0 - cosmo.y_p));
        assert!((cosmo.f_he() - expected).abs() < 1e-15);
    }

    #[test]
    fn test_cosmology_new_custom_params() {
        // Verify that Cosmology::new() correctly computes all derived quantities
        // from custom parameters (not just the defaults).
        let cosmo = Cosmology::new(2.725, 0.022, 0.12, 0.67, 3.046, 0.245).unwrap();

        // Check that stored params are correct
        assert!((cosmo.t_cmb - 2.725).abs() < 1e-10);
        assert!((cosmo.omega_b - 0.022).abs() < 1e-10);
        assert!((cosmo.h - 0.67).abs() < 1e-10);
        assert!((cosmo.y_p - 0.245).abs() < 1e-10);

        // Verify derived f_he is recomputed (not stale from defaults)
        let expected_f_he = 0.245 / (4.0 * (1.0 - 0.245));
        assert!(
            (cosmo.f_he() - expected_f_he).abs() < 1e-12,
            "f_he = {}, expected {expected_f_he}",
            cosmo.f_he()
        );

        // n_H(0) should reflect the custom omega_b and Y_p
        let n_h_0 = cosmo.n_h(0.0);
        assert!(n_h_0 > 0.0 && n_h_0.is_finite());

        // Different omega_b should give different n_H
        let cosmo2 = Cosmology::new(2.725, 0.044, 0.12, 0.67, 3.046, 0.245).unwrap();
        let n_h_0_2 = cosmo2.n_h(0.0);
        // omega_b doubled → n_H doubled
        let ratio = n_h_0_2 / n_h_0;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Doubling omega_b should double n_H: ratio={ratio:.4}"
        );

        // Hubble rate should depend on h
        let h_z0 = cosmo.hubble(0.0);
        assert!(h_z0 > 0.0 && h_z0.is_finite());
    }
}
