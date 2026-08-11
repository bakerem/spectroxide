//! Helpers for resonant axion–photon (γ ↔ a) conversion in the narrow-width
//! (Landau–Zener) approximation.
//!
//! The formalism is that of Cyr, Chluba & Manoj (2024), which mirrors the
//! photon → dark-photon treatment of Chluba, Cyr & Johnson (2024): a resonant
//! conversion occurs where the photon plasma mass matches the axion mass,
//! `m_a ≃ ω_pl`, and the CMB monopole is depleted by the conversion
//! probability `P(x) = 1 − exp(−γ_con x)`.
//!
//! Two differences from the dark-photon case:
//!
//! 1. **Frequency dependence flips.** The axion probability carries `x` in the
//!    numerator (`1 − exp(−γ_con x)`), so *high*-frequency (Wien-tail) photons
//!    convert preferentially. The dark photon carries `1/x`, depleting the
//!    Rayleigh–Jeans tail instead. (Cyr, Chluba & Manoj 2024, Eq. 2.)
//! 2. **The `γ_con` prefactor** replaces `ε² m²` with `κ² (1+z)⁴ T_CMB(z)`,
//!    where `κ = g_aγγ B_rms` (Eq. 3b). Writing the paper's `d ln m_γ²/dz` in
//!    terms of `|d ln ω_pl²/d ln a|` cancels two powers of `(1+z)`, giving
//!
//!    γ_con = π κ² (1+z_res)⁴ T_CMB(z_res) / [ m_a² H(z_res) |d ln ω_pl²/d ln a| ].
//!
//! The resonance condition and its redshift derivative are identical to the
//! dark-photon problem, so this module reuses
//! [`crate::dark_photon::resonance_redshift`] and
//! [`crate::dark_photon::dln_omega_pl_sq_dlna`] unchanged.
//!
//! **Scope.** This is the *monopole* / plasma-frequency treatment,
//! `m_γ² ≈ ω_pl²`, which gives a single frequency-independent resonance
//! redshift. It is valid for `m_a ≳ few×10⁻¹⁰ eV` in the fully-ionized era.
//! The frequency-dependent HI/HeI/HeII corrections to `m_γ²` (Cyr, Chluba &
//! Manoj 2024, Sec. II B, Eqs. 7–12) — which shift `z_con(ω)` and produce
//! multiple crossings around recombination — are not modeled here; they are the
//! natural v2 extension.
//!
//! References:
//! - Cyr, Chluba & Manoj (2024), arXiv:2411.13701
//! - Chluba, Cyr & Johnson (2024), MNRAS 535, 1874

use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::dark_photon::{dln_omega_pl_sq_dlna, resonance_redshift};

/// Normalization of the axion mixing coupling `κ = g_aγγ B_rms` in eV,
/// per unit of the dimensionless combination `ε = (g_aγγ/10⁻¹⁰GeV⁻¹)(B_rms/nG)`.
///
/// Cyr, Chluba & Manoj (2024), Eq. 3b: `κ = g_aγγ B_rms ≈ 1.95×10⁻³⁰ eV × ε`.
pub const KAPPA_PER_EPSILON_EV: f64 = 1.95e-30;

/// Axion mixing coupling `κ = g_aγγ B_rms⁰` (today), in eV.
///
/// `g_agamma` is the axion–photon coupling in GeV⁻¹ and `b_rms` the comoving
/// RMS transverse magnetic field today in nG. Uses the paper's normalization
/// (Eq. 3b/3c): `κ = 1.95×10⁻³⁰ eV × (g_aγγ/10⁻¹⁰GeV⁻¹)(B_rms/nG)`.
pub fn kappa_ev(g_agamma: f64, b_rms: f64) -> f64 {
    KAPPA_PER_EPSILON_EV * (g_agamma / 1.0e-10) * b_rms
}

/// NWA axion–photon conversion parameter `γ_con` (dimensionless).
///
/// γ_con = π κ² (1+z_res)⁴ T_CMB(z_res) / [ m_a² H(z_res) |d ln ω_pl²/d ln a| ],
/// following Cyr, Chluba & Manoj (2024), Eq. 3a in the monopole limit
/// `m_γ² ≈ ω_pl²`. Returns `None` if no plasma-frequency resonance exists in the
/// supported redshift range.
///
/// - `g_agamma`: axion–photon coupling g_aγγ in GeV⁻¹.
/// - `b_rms`: comoving RMS transverse B-field today in nG.
/// - `m_ev`: axion mass in eV.
///
/// Returned tuple: `(gamma_con, z_res)`.
pub fn gamma_con_axion(
    g_agamma: f64,
    b_rms: f64,
    m_ev: f64,
    cosmo: &Cosmology,
) -> Option<(f64, f64)> {
    let z_res = resonance_redshift(m_ev, cosmo)?;
    let kappa = kappa_ev(g_agamma, b_rms);
    let t_cmb_ev = K_BOLTZMANN * cosmo.t_cmb * (1.0 + z_res) / EV_IN_JOULES;
    let hbar_ev_s = HBAR / EV_IN_JOULES;
    let h_ev = hbar_ev_s * cosmo.hubble(z_res);
    let d = dln_omega_pl_sq_dlna(z_res, cosmo);
    let one_plus_z4 = (1.0 + z_res).powi(4);
    let gc =
        std::f64::consts::PI * kappa * kappa * one_plus_z4 * t_cmb_ev / (m_ev * m_ev * h_ev * d);
    Some((gc, z_res))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dark_photon::plasma_frequency_ev;
    use approx::assert_relative_eq;

    #[test]
    fn kappa_normalization_matches_paper() {
        // Eq. 3b: g_aγγ = 10⁻¹⁰ GeV⁻¹, B_rms = 1 nG ⇒ κ = 1.95×10⁻³⁰ eV.
        let kappa = kappa_ev(1.0e-10, 1.0);
        assert_relative_eq!(kappa, 1.95e-30, max_relative = 1e-12);
    }

    #[test]
    fn gamma_con_scales_as_g_squared_and_b_squared() {
        // γ_con ∝ κ² = (g_aγγ B_rms)², so doubling either coupling quadruples it.
        let cosmo = Cosmology::default();
        let (g0, _) = gamma_con_axion(1.0e-11, 1.0, 1e-7, &cosmo).unwrap();
        let (g_g, _) = gamma_con_axion(2.0e-11, 1.0, 1e-7, &cosmo).unwrap();
        let (g_b, _) = gamma_con_axion(1.0e-11, 2.0, 1e-7, &cosmo).unwrap();
        assert_relative_eq!(g_g / g0, 4.0, max_relative = 1e-10);
        assert_relative_eq!(g_b / g0, 4.0, max_relative = 1e-10);
    }

    #[test]
    fn resonance_redshift_matches_dark_photon_condition() {
        // The resonance condition m = ω_pl is identical to the dark photon's,
        // so z_res depends only on the mass, not on the coupling channel.
        let cosmo = Cosmology::default();
        for m_ev in [3e-8, 1e-7, 1e-6, 1e-5] {
            let (_gc, z_res) = gamma_con_axion(1e-11, 1.0, m_ev, &cosmo).unwrap();
            let omega = plasma_frequency_ev(z_res, &cosmo);
            assert_relative_eq!(omega, m_ev, max_relative = 1e-5);
        }
    }

    #[test]
    fn gamma_con_is_dimensionless_and_positive() {
        // Spot-check that the assembled prefactor is finite, positive, and of a
        // sensible magnitude for a benchmark point (not NaN/Inf from unit slips).
        let cosmo = Cosmology::default();
        let (gc, z_res) = gamma_con_axion(1e-10, 1.0, 1e-7, &cosmo).unwrap();
        assert!(gc.is_finite() && gc > 0.0, "γ_con = {gc}");
        assert!(
            (z_res - 3.21e4).abs() / 3.21e4 < 0.05,
            "z_res = {z_res:.3e}"
        );
    }
}
