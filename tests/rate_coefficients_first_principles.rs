//! First-principles DC/BR rate-coefficient magnitudes (Part II §II.3 of
//! dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md).
//!
//! Motivation (CLAUDE.md Pitfall #8): the 10¹¹× BR bug (a spurious `/n_e`)
//! survived 375 tests because every test used the code's own coefficient or a
//! DC-dominated regime. This file recomputes the coefficient magnitudes from
//! literature formulas with **CODATA constants typed literally here — importing
//! nothing from `constants.rs`** — and asserts:
//!   (a) the code/first-principles ratio ≈ 1 (magnitude + dimensionlessness), and
//!   (b) the ratio is **z-independent** across 3 redshifts. A per-volume-vs-per-
//!       Thomson-time / wrong-density-factor bug shows up as an O(10^n) offset
//!       (a) or a (1+z)-dependent drift (b); a wrong θ power also drifts (b).
//!
//! The Gaunt factors are re-used from the code (`gaunt_ff_nr`) — they carry
//! their own exact-identity coverage (`brpack_gaunt_factor_spot_checks`,
//! greens_function_checks.rs:336, rel 1e-10 vs CRB-2020) — so this file
//! isolates the PREFACTOR × density × temperature assembly, which is exactly
//! where the historical dimensional bug lived. Two-body BR keeps ΣZ²N_i after
//! Thomson normalization; one-body DC keeps no density factor.

use spectroxide::bremsstrahlung::{br_emission_coefficient, gaunt_ff_nr};
use spectroxide::cosmology::Cosmology;
use spectroxide::double_compton::dc_emission_coefficient;

// --- CODATA 2018, typed literally (import nothing from constants.rs) ---------
const ALPHA_FS: f64 = 7.297_352_5693e-3;
const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const M_ELECTRON: f64 = 9.109_383_7015e-31; // kg
const C_LIGHT: f64 = 2.997_924_58e8; // m/s
const PI: f64 = std::f64::consts::PI;

/// λ_e = h/(m_e c), the electron Compton wavelength [m].
fn lambda_e() -> f64 {
    H_PLANCK / (M_ELECTRON * C_LIGHT)
}

/// I_pl = ∫₀^∞ x⁴ n_pl(1+n_pl) dx = 4π⁴/15, computed here by quadrature so the
/// analytic value is itself pinned, not assumed.
fn i_pl_quadrature() -> f64 {
    let (x_min, x_max, n) = (1e-4_f64, 60.0_f64, 200_000usize);
    let (lo, hi) = (x_min.ln(), x_max.ln());
    let mut acc = 0.0;
    let mut x_prev = 0.0;
    let mut f_prev = 0.0;
    for i in 0..n {
        let x = (lo + (hi - lo) * i as f64 / (n - 1) as f64).exp();
        let em1 = x.exp_m1();
        let nn1 = (1.0 + em1) / (em1 * em1); // n_pl(1+n_pl)
        let f = x.powi(4) * nn1;
        if i > 0 {
            acc += 0.5 * (f + f_prev) * (x - x_prev);
        }
        x_prev = x;
        f_prev = f;
    }
    acc
}

#[test]
fn i_pl_matches_analytic() {
    let num = i_pl_quadrature();
    let analytic = 4.0 * PI.powi(4) / 15.0;
    let rel = (num - analytic).abs() / analytic;
    eprintln!("II.3|I_pl quadrature={num:.6} vs 4π⁴/15={analytic:.6} rel={rel:.3e}");
    assert!(rel < 1e-4, "I_pl quadrature off from 4π⁴/15: {rel:.3e}");
}

// ===========================================================================
// DC — one-body, coefficient depends only on θ_z (no density factor)
// ===========================================================================

#[test]
fn dc_coefficient_first_principles() {
    // K_DC(x→0, θ_z) = (4α/3π) θ_z² I_pl / (1 + 14.16 θ_z).
    // The (1+14.16θ_z) relativistic correction is the code's cited Chluba+2007
    // thermal-averaging factor; include it literally so the ratio isolates the
    // (4α/3π)θ_z² I_pl magnitude. Use x≪1 so the H_dc(x) suppression → 1.
    let i_pl = 4.0 * PI.powi(4) / 15.0;
    let cosmo = Cosmology::default();
    let x = 1e-4; // H_dc(1e-4) = 1 − O(1e-4)
    let mut ratios = Vec::new();
    for &z in &[3e5, 1e6, 2e6] {
        let theta_z = cosmo.theta_z(z);
        let code = dc_emission_coefficient(x, theta_z);
        let anchor =
            4.0 * ALPHA_FS / (3.0 * PI) * theta_z * theta_z * i_pl / (1.0 + 14.16 * theta_z);
        let ratio = code / anchor;
        eprintln!(
            "II.3|DC z={z:.1e} θ_z={theta_z:.3e} code={code:.4e} anchor={anchor:.4e} ratio={ratio:.6}"
        );
        // Magnitude ≈ 1 (allowing the tiny H_dc(x) deficit); a dimensional /
        // density-factor bug would be orders of magnitude off.
        assert!(
            (ratio - 1.0).abs() < 2e-3,
            "DC magnitude off at z={z:.1e}: ratio={ratio:.6}"
        );
        ratios.push(ratio);
    }
    // z-independence: ratio = H_dc(x), x fixed ⇒ identical across z. A wrong
    // θ_z power or a spurious (1+z) factor would break this.
    let spread = ratios.iter().cloned().fold(f64::MIN, f64::max)
        - ratios.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("II.3|DC ratio z-spread = {spread:.3e}");
    assert!(
        spread < 1e-4,
        "DC code/anchor ratio is z-dependent: spread={spread:.3e}"
    );
}

// ===========================================================================
// BR — two-body, keeps ΣZ²N_i (one density factor after Thomson normalization)
// ===========================================================================

#[test]
fn br_coefficient_first_principles() {
    // K_BR = (α λ_e³/(2π√(6π))) θ_e^{-7/2} (e^{-xφ}/φ³) Σ_i Z_i² N_i g_ff.
    // Anchor prefactor from literal CODATA; densities from the cosmology
    // (scaling ∝(1+z)³) are passed to BOTH the code and the anchor, so an
    // internal spurious /n_e would surface as ratio ≈ 1/n_e — an O(10^n) offset
    // AND a (1+z)³ drift. Gaunt factors reused from the code (separately
    // class-1 verified). Redshifts chosen where H and He are fully ionized so
    // the code's internal Saha (y_HeII, y_HeI → 1) matches the anchor's
    // fully-ionized species sum n_H·g1 + 4 n_He·g2.
    let br_pre = ALPHA_FS * lambda_e().powi(3) / (2.0 * PI * (6.0 * PI).sqrt());
    let cosmo = Cosmology::default();
    let x = 0.5;
    let x_e = 1.0; // fully ionized hydrogen

    let mut ratios = Vec::new();
    for &z in &[5e5, 1e6, 2e6] {
        let theta_z = cosmo.theta_z(z);
        let theta_e = theta_z; // φ = 1 (equilibrium)
        let phi = theta_z / theta_e;

        let n_h = cosmo.n_h(z);
        let n_he = cosmo.n_he(z);
        let n_e = cosmo.n_e(z, x_e);

        let code = br_emission_coefficient(x, theta_e, theta_z, n_h, n_he, n_e, x_e, &cosmo);

        // Fully-ionized species sum: H⁺ (Z=1) + He²⁺ (Z=2), He⁺ absent.
        let g1 = gaunt_ff_nr(x, theta_e, 1.0);
        let g2 = gaunt_ff_nr(x, theta_e, 2.0);
        let species = n_h * g1 + 4.0 * n_he * g2;
        let temp_factor = theta_e.powf(-3.5) * (-x * phi).exp() / phi.powi(3);
        let anchor = br_pre * temp_factor * species;

        let ratio = code / anchor;
        eprintln!(
            "II.3|BR z={z:.1e} n_H={n_h:.3e} code={code:.4e} anchor={anchor:.4e} ratio={ratio:.6}"
        );
        // Magnitude ≈ 1 within Gaunt/He-Saha slop (few %). A per-volume vs
        // per-Thomson-time bug is O(10^n) here.
        assert!(
            (ratio - 1.0).abs() < 0.03,
            "BR magnitude off at z={z:.1e}: ratio={ratio:.6} (dimensional/density bug?)"
        );
        ratios.push(ratio);
    }
    // z-independence isolates a density-factor / (1+z) error from Gaunt slop.
    let spread = ratios.iter().cloned().fold(f64::MIN, f64::max)
        - ratios.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("II.3|BR ratio z-spread = {spread:.3e}");
    assert!(
        spread < 1e-2,
        "BR code/anchor ratio is z-dependent (spread={spread:.3e}) — density-factor error"
    );
}
