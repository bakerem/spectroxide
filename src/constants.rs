//! Physical and cosmological constants in SI units.
//!
//! All values from CODATA 2018 / Particle Data Group unless noted.
//!
//! Grouped as:
//! - **Fundamental**: `C_LIGHT`, `HBAR`, `HPLANCK`, `K_BOLTZMANN`, `M_ELECTRON`,
//!   `M_PROTON`, `SIGMA_THOMSON`, `ALPHA_FS` — feed Compton scattering rates,
//!   Planck normalisations, and DC/BR emission prefactors.
//! - **Unit conversions**: `EV_IN_JOULES`, `MPC_IN_METERS`.
//! - **Atomic physics**: `E_RYDBERG_*`, `E_HE_*`, `LAMBDA_LYA`, `LAMBDA_2S1S` —
//!   used by [`crate::recombination`] for the Peebles 3-level atom.
//! - **Derived**: `LAMBDA_ELECTRON`, `M_E_C2`, `M_E_C2_EV`.
//! - **Cosmology**: `T_CMB_0`, `Y_P`, `N_EFF`, and Planck spectral integrals
//!   (`G1_PLANCK = π²/6`, `G2_PLANCK = 2ζ(3)`, `G3_PLANCK = π⁴/15`,
//!   `I4_PLANCK = 4 G₃`), plus the derived μ-channel coefficients
//!   `BETA_MU`, `KAPPA_C`, `ALPHA_RHO`, and the era-boundary redshifts
//!   `Z_MU` (μ-distortion freeze-out) and `Z_MU_Y` (μ→y transition).
//!   These are used by both the PDE solver and the Green's-function
//!   visibility functions.

// Fundamental constants
pub const C_LIGHT: f64 = 2.997_924_58e8; // m/s (exact)
pub const HBAR: f64 = 1.054_571_817e-34; // J·s
pub const HPLANCK: f64 = 6.626_070_15e-34; // J·s (exact)
pub const K_BOLTZMANN: f64 = 1.380_649e-23; // J/K (exact)
pub const G_NEWTON: f64 = 6.674_30e-11; // m³/(kg·s²)
pub const M_ELECTRON: f64 = 9.109_383_7015e-31; // kg
pub const M_PROTON: f64 = 1.672_621_923_69e-27; // kg
pub const SIGMA_THOMSON: f64 = 6.652_458_7321e-29; // m²
pub const ALPHA_FS: f64 = 7.297_352_5693e-3; // fine structure constant

// Unit conversions
/// 1 eV in Joules (exact by 2019 SI redefinition)
pub const EV_IN_JOULES: f64 = 1.602_176_634e-19;
/// 1 Mpc in meters
pub const MPC_IN_METERS: f64 = 3.085_677_581e22;

// Hydrogen atomic physics
/// Hydrogen ionization energy (1s ground state), in eV.
pub const E_RYDBERG_EV: f64 = 13.605_693_122_994;
/// Hydrogen ionization energy, in J.
pub const E_RYDBERG: f64 = E_RYDBERG_EV * EV_IN_JOULES;
/// Ionization energy from n=2 level (E_Rydberg / 4), in J.
pub const E_ION_N2: f64 = E_RYDBERG_EV / 4.0 * EV_IN_JOULES;
/// Lyman-alpha wavelength, in m.
pub const LAMBDA_LYA: f64 = 1.215_670e-7;
/// 2s→1s two-photon decay rate [s⁻¹]
pub const LAMBDA_2S1S: f64 = 8.2245809;

// Mathematical constants
/// Riemann zeta function ζ(3) = 1.202...
pub const ZETA_3: f64 = 1.202_056_903_159_594_3;

// Helium atomic physics
/// He²⁺ ionization energy (He II → He I), in eV.
pub const E_HE_II_ION_EV: f64 = 54.4178;
/// He⁺ ionization energy (He I → He), in eV.
pub const E_HE_I_ION_EV: f64 = 24.5874;

// Derived constants
/// Electron Compton wavelength: h / (m_e * c)
pub const LAMBDA_ELECTRON: f64 = HPLANCK / (M_ELECTRON * C_LIGHT);

/// m_e c^2 in Joules
pub const M_E_C2: f64 = M_ELECTRON * C_LIGHT * C_LIGHT;

/// m_e c^2 in eV
pub const M_E_C2_EV: f64 = 0.510_998_950e6;

// Cosmological constants
/// Default CMB temperature today, in K.
///
/// Value 2.726 K (Mather et al. 1999) chosen for compatibility with
/// CosmoTherm v1.0.3 reference data used in validation. Differs by 0.02%
/// from the Planck-era Fixsen (2009) value 2.7255 K; the
/// [`crate::cosmology::Cosmology::planck2018`] preset overrides this.
pub const T_CMB_0: f64 = 2.726;

/// Helium mass fraction
pub const Y_P: f64 = 0.24;

/// Helium number fraction relative to hydrogen: f_He = Y_p / (4*(1-Y_p))
pub const F_HE: f64 = Y_P / (4.0 * (1.0 - Y_P));

/// Effective number of neutrino species
pub const N_EFF: f64 = 3.046;

/// km/s/Mpc → 1/s
pub const KM_PER_MPC: f64 = 3.240_779_29e-20;

// Spectral integral constants (for Planck distribution)
/// G_1 = ∫₀^∞ x n_pl(x) dx = π²/6 = ζ(2)
pub const G1_PLANCK: f64 = 1.644_934_066_848_226_4; // π²/6

/// G_2 = ∫₀^∞ x² n_pl(x) dx = 2ζ(3)
pub const G2_PLANCK: f64 = 2.404_113_806_319_188_6; // 2*ζ(3)

/// G_3 = ∫₀^∞ x³ n_pl(x) dx = π⁴/15
pub const G3_PLANCK: f64 = 6.493_939_402_266_829; // π⁴/15

/// I_4 = ∫₀^∞ x⁴ n_pl (1+n_pl) dx = 4π⁴/15
pub const I4_PLANCK: f64 = 4.0 * G3_PLANCK;

/// β_μ = 3ζ(3)/ζ(2) ≈ 2.1923 — frequency of μ-distortion zero crossing
pub const BETA_MU: f64 = 3.0 * ZETA_3 / G1_PLANCK;

/// κ_c = 3 ∫x³ M(x) dx / G₃ = 12/β_μ − 9G₂/G₃ ≈ 2.1419
///
/// Derived analytically using ∫₀^∞ xⁿ eˣ/(eˣ−1)² dx = n·G_{n−1}:
///   ∫x³ M(x) dx = ∫x³ (x/β_μ − 1) g_bb/x dx = (4G₃/β_μ) − 3G₂
///   κ_c = 3[(4G₃/β_μ) − 3G₂]/G₃ = 12/β_μ − 9G₂/G₃
pub const KAPPA_C: f64 = 12.0 / BETA_MU - 9.0 * G2_PLANCK / G3_PLANCK;

/// α_μ = 1/β_μ = π²/(18ζ(3)) — relates μ to energy
pub const ALPHA_MU: f64 = 1.0 / BETA_MU;

/// κ_γ = 8π / λ_e³ — photon phase space density prefactor.
///
/// The photon energy density per electron is:
///   ρ̃_γ = κ_γ × θ_z⁴ × G₃ / n_e
///
/// where λ_e is the electron Compton wavelength.
/// This enters the expansion correction Λ in the quasi-stationary T_e equation.
pub const KAPPA_GAMMA: f64 =
    8.0 * std::f64::consts::PI / (LAMBDA_ELECTRON * LAMBDA_ELECTRON * LAMBDA_ELECTRON);

/// α_ρ = G₂/G₃ — ratio of photon number to energy spectral integrals.
///
/// This constant determines the critical frequency x₀ = 4/(3α_ρ) at which
/// photon injection produces zero μ-distortion. Below x₀, injection produces
/// negative μ; above x₀, positive μ.
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 30
pub const ALPHA_RHO: f64 = G2_PLANCK / G3_PLANCK; // ≈ 0.3702

/// x₀ = 4/(3α_ρ) ≈ 3.60 — balanced injection frequency.
///
/// Photon injection at x = x₀ produces zero net μ-distortion because the
/// energy-per-photon exactly matches the mean energy of the background.
///
/// Reference: Chluba (2015), arXiv:1506.06582, Eq. 31
pub const X_BALANCED: f64 = 4.0 / (3.0 * ALPHA_RHO); // ≈ 3.60

// Thermalization redshifts (approximate)
/// μ-era thermalization redshift (Chluba 2013, MNRAS 436, 2232)
pub const Z_MU: f64 = 1.98e6;
/// μ-y transition redshift
pub const Z_MU_Y: f64 = 5.0e4;

/// Dimensionless temperature at the *default* CMB temperature T_CMB_0 = 2.726 K.
///
/// Convenience helper for tests and quick calculations. Production code must
/// use [`crate::cosmology::Cosmology::theta_z`] so that a user-supplied T_CMB
/// (e.g. Planck 2018's 2.7255 K) is honoured.
#[doc(hidden)]
#[inline]
pub fn theta_z(z: f64) -> f64 {
    K_BOLTZMANN * T_CMB_0 * (1.0 + z) / M_E_C2
}

/// Photon temperature at redshift `z` assuming the default T_CMB_0.
///
/// Convenience helper; see the note on [`theta_z`] for production usage.
#[doc(hidden)]
#[inline]
pub fn t_z(z: f64) -> f64 {
    T_CMB_0 * (1.0 + z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theta_z_value() {
        // θ_z ≈ 4.60e-10 * (1+z) at z=0
        let theta_0 = theta_z(0.0);
        assert!(
            (theta_0 - 4.60e-10).abs() < 0.05e-10,
            "theta_z(0) = {theta_0:.3e}, expected ~4.60e-10"
        );
    }

    #[test]
    fn test_beta_mu() {
        // Literature value
        assert!(
            (BETA_MU - 2.1923).abs() < 0.001,
            "beta_mu = {BETA_MU}, expected ~2.1923"
        );
        // Identity: β_μ ≡ 3ζ(3)/ζ(2) = 3·ZETA_3 / (π²/6) to machine precision
        let expected = 3.0 * ZETA_3 / (std::f64::consts::PI.powi(2) / 6.0);
        assert!(
            (BETA_MU - expected).abs() / expected < 1e-14,
            "BETA_MU = {BETA_MU}, 3ζ(3)/ζ(2) = {expected}"
        );
    }

    #[test]
    fn test_g1_g2_identities() {
        // G₁ = π²/6 = ζ(2) exact
        let g1_exact = std::f64::consts::PI.powi(2) / 6.0;
        assert!(
            (G1_PLANCK - g1_exact).abs() / g1_exact < 1e-14,
            "G1 = {G1_PLANCK}, π²/6 = {g1_exact}"
        );
        // G₂ = 2ζ(3) exact
        let g2_exact = 2.0 * ZETA_3;
        assert!(
            (G2_PLANCK - g2_exact).abs() / g2_exact < 1e-14,
            "G2 = {G2_PLANCK}, 2ζ(3) = {g2_exact}"
        );
        // I₄ = 4·G₃ exact
        assert!(
            (I4_PLANCK - 4.0 * G3_PLANCK).abs() / I4_PLANCK < 1e-14,
            "I4 = {I4_PLANCK}, 4·G₃ = {}",
            4.0 * G3_PLANCK
        );
    }

    #[test]
    fn test_g3_planck() {
        let pi = std::f64::consts::PI;
        let expected = pi.powi(4) / 15.0;
        assert!(
            (G3_PLANCK - expected).abs() / expected < 1e-14,
            "G3 = {G3_PLANCK}, expected {expected}"
        );
    }

    #[test]
    fn test_alpha_rho() {
        // Numerical transcription guard: α_ρ = G₂/G₃ = 30ζ(3)/π⁴ ≈ 0.37020884...
        // (Evaluating 30·ZETA_3/π⁴ against ALPHA_RHO in code would be tautological —
        // ALPHA_RHO is defined as G2_PLANCK/G3_PLANCK which reduces to that ratio.)
        assert!(
            (ALPHA_RHO - 0.370_208_84).abs() < 1e-6,
            "alpha_rho = {ALPHA_RHO}, expected ~0.37020884 (from 30ζ(3)/π⁴)"
        );
    }

    #[test]
    fn test_x_balanced() {
        // x₀ = 4/(3α_ρ) ≈ 3.60
        assert!(
            (X_BALANCED - 3.602).abs() < 0.01,
            "x_balanced = {X_BALANCED}, expected ~3.602"
        );
        // Cross-check: x₀ = 4G₃/(3G₂)
        let x0_alt = 4.0 * G3_PLANCK / (3.0 * G2_PLANCK);
        assert!(
            (X_BALANCED - x0_alt).abs() < 1e-14,
            "x_balanced definitions disagree"
        );
    }

    #[test]
    fn test_f_he() {
        // Y_p=0.24: f_He = 0.24/(4*0.76) ≈ 0.0789
        assert!((F_HE - 0.07895).abs() < 0.001);
    }

    /// Verify κ_c against the analytical formula and a numerical quadrature.
    ///
    /// κ_c = 12/β_μ − 9G₂/G₃ ≈ 2.1419
    ///
    /// Also verified via numerical integration: κ_c = 3∫x³M(x)dx / G₃
    /// where M(x) = (x/β_μ − 1) G_bb(x) / x.
    #[test]
    fn test_kappa_c_analytical_and_numerical() {
        // Analytical form (how KAPPA_C is defined in constants.rs)
        let kappa_from_formula = 12.0 / BETA_MU - 9.0 * G2_PLANCK / G3_PLANCK;
        assert!(
            (KAPPA_C - kappa_from_formula).abs() < 1e-14,
            "KAPPA_C mismatch: stored={KAPPA_C}, computed={kappa_from_formula}"
        );

        // Literature value: κ_c ≈ 2.1419
        assert!(
            (KAPPA_C - 2.1419).abs() < 0.001,
            "KAPPA_C = {KAPPA_C}, expected ~2.1419"
        );

        // Numerical quadrature: κ_c = 3 ∫x³ M(x) dx / G₃
        // where M(x) = (x/β_μ − 1) G_bb(x)/x and G_bb(x) = x eˣ/(eˣ−1)².
        // So M(x) = (x/β_μ − 1) eˣ/(eˣ−1)².
        //
        // Using ∫xⁿ eˣ/(eˣ−1)² dx = n Gₙ₋₁ (integration by parts):
        //   ∫x³ M(x) dx = (1/β_μ)×4G₃ − 3G₂
        //   κ_c = 3(4G₃/β_μ − 3G₂)/G₃ = 12/β_μ − 9G₂/G₃
        let n = 10000;
        let x_min = 0.001_f64;
        let x_max = 30.0_f64;
        let dx = (x_max - x_min) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = x_min + (i as f64 + 0.5) * dx;
            let ex = x.exp();
            // M(x) = (x/β_μ − 1) eˣ/(eˣ−1)²
            let m_x = (x / BETA_MU - 1.0) * ex / ((ex - 1.0) * (ex - 1.0));
            integral += x * x * x * m_x * dx;
        }
        let kappa_numerical = 3.0 * integral / G3_PLANCK;
        let rel_err = (KAPPA_C - kappa_numerical).abs() / KAPPA_C;
        assert!(
            rel_err < 0.001,
            "κ_c numerical = {kappa_numerical:.6}, analytical = {KAPPA_C:.6}, err = {rel_err:.2e}"
        );
    }
}
