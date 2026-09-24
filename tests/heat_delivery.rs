//! Heat delivery near and after recombination (ADR 0004).
//!
//! After recombination the gas couples to the photons only weakly, so the
//! fraction of injected heat that reaches the photons, rather than being lost
//! to adiabatic cooling of the gas, is a real physical number close to but
//! below 1. The targets here come from an independent integration of the
//! gas-temperature excess with Compton exchange and adiabatic cooling
//! (`dev/scripts/heatloss/heat_delivery_expectation.py` for the burst,
//! `decay_delivery_expectation.py` for the decay, `baseline_expectation.py` for
//! the no-injection cooling). The scripts type their CODATA constants and take
//! only X_e(z) from spectroxide, by default from the Python
//! `ionization_fraction`. They are not read off from solver output (CLAUDE.md
//! pitfall #9). Provenance: `dev/audit/fix_a_cn_old_half_ab.md`.
//!
//! Before ADR 0004 the old Crank-Nicolson half of the coupled step used the
//! step-start ρ_e while the gas row used the backward-Euler ρ_e, which lost
//! about ½ Δln X_e of each step's heat. Both tests fail on that code: the
//! burst delivered 0.938 and the decay 0.849.
//!
//! Delivered heat is the photon Δρ/ρ of a run with injection minus that of a
//! run with zero amplitude and otherwise identical settings. The baseline
//! removes the solver's own adiabatic-cooling distortion, which is present
//! without any injection.

use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::GridConfig;
use spectroxide::solver::ThermalizationSolver;

/// Runs `scenario` from `z_start` to `z_end` on the 4000-point production
/// grid with default step control, and returns the photon Δρ/ρ.
fn photon_drho(scenario: InjectionScenario, z_start: f64, z_end: f64) -> f64 {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig::production())
        .injection(scenario)
        .z_range(z_start, z_end)
        .build()
        .unwrap();
    let r = solver.run_to_result(z_end);
    let drho = r.snapshot.delta_rho_over_rho;
    assert!(drho.is_finite(), "non-finite Δρ/ρ: {drho}");
    drho
}

/// A Gaussian burst of Δρ/ρ = 1e-8 at z_h = 1000 (σ_z = 100, the CLI's
/// default max(0.04 z_h, 100)) delivers 0.999862 of its heat to the photons by z = 200.
/// The rest goes to adiabatic cooling of the gas excess. Target:
/// `python dev/scripts/heatloss/heat_delivery_expectation.py 1000`.
///
/// Before ADR 0004 this run delivered 0.938 at Δτ_max = 10, 3, and 1 alike.
#[test]
fn burst_at_recombination_delivers_independent_fraction() {
    const DRHO: f64 = 1e-8;
    const EXPECTED: f64 = 0.999862; // independent integration
    // The whole physical loss is 1 − EXPECTED = 1.38e-4, so the tolerance must
    // sit well below it or a run that delivered every joule (1.0) would pass.
    // 3e-5 is a fifth of the loss and four times the measured offset (7e-6).
    const TOL: f64 = 3e-5;
    let burst = |amp: f64| InjectionScenario::SingleBurst {
        z_h: 1000.0,
        delta_rho_over_rho: amp,
        sigma_z: 100.0,
    };
    let with = photon_drho(burst(DRHO), 1700.0, 200.0);
    let base = photon_drho(burst(0.0), 1700.0, 200.0);
    let delivered = (with - base) / DRHO;
    eprintln!("burst z_h=1000: with={with:e} base={base:e} delivered={delivered:.6}");
    assert!(
        (delivered - EXPECTED).abs() < TOL,
        "delivered fraction {delivered:.6}, expected {EXPECTED} ± {TOL:e}"
    );
}

/// Injected Δρ/ρ_γ of a decaying particle between `z_lo` and `z_hi`, computed
/// from constants typed here and nothing imported from spectroxide.
///
/// The solver's convention (`energy_injection.rs`) is a heating rate
/// f_X Γ N_H e^{−Γt} per unit volume, divided by ρ_γ with T_0 = 2.726 K, so
///
///   Δρ/ρ_γ = ∫ f_X Γ (N_H0/ρ_γ0) a e^{−Γ t(a)} dt,   dt = da / (a H).
///
/// For t(a) we use the exact matter-plus-radiation age,
///
///   H_0 t(a) = 2/(3 Ω_m²) [(Ω_m a − 2Ω_r) √(Ω_m a + Ω_r) + 2 Ω_r^{3/2}],
///
/// which neglects Λ. At z ≥ 200 that changes H by Ω_Λ a³/Ω_m < 3e-7, far
/// below the tolerances used here. The default cosmology is Ω_b = 0.044,
/// Ω_m = 0.26, h = 0.71, Y_p = 0.24, N_eff = 3.046. For the N-4 case below
/// (Γ = 7.1838e-14 s⁻¹, f_X = 10 eV, z = 200 to 5e4) this gives 6.537e-9,
/// the value quoted in the A/B record.
fn injected_decay_drho(f_x_ev: f64, gamma: f64, z_lo: f64, z_hi: f64) -> f64 {
    use std::f64::consts::PI;
    // CODATA 2018 and IAU.
    const C: f64 = 299_792_458.0; // m/s
    const G_N: f64 = 6.674_30e-11; // m³ kg⁻¹ s⁻²
    const K_B: f64 = 1.380_649e-23; // J/K
    const HBAR: f64 = 1.054_571_817e-34; // J s
    const M_P: f64 = 1.672_621_923_69e-27; // kg
    const EV: f64 = 1.602_176_634e-19; // J
    const MPC: f64 = 3.085_677_581_491_367e22; // m
    const T0: f64 = 2.726; // K
    const H: f64 = 0.71;
    const OMEGA_B: f64 = 0.044;
    const OMEGA_M: f64 = 0.26;
    const Y_P: f64 = 0.24;
    const N_EFF: f64 = 3.046;

    let h0 = 100.0e3 * H / MPC; // 1/s
    let rho_crit_mass = 3.0 * h0 * h0 / (8.0 * PI * G_N); // kg/m³
    let rho_gamma0 = PI * PI / 15.0 * (K_B * T0).powi(4) / (HBAR * C).powi(3); // J/m³
    let omega_g = rho_gamma0 / (rho_crit_mass * C * C);
    let omega_r = omega_g * (1.0 + N_EFF * 7.0 / 8.0 * (4.0_f64 / 11.0).powf(4.0 / 3.0));
    let n_h0 = (1.0 - Y_P) * OMEGA_B * rho_crit_mass / M_P; // 1/m³

    let age = |a: f64| {
        2.0 / (3.0 * h0 * OMEGA_M * OMEGA_M)
            * ((OMEGA_M * a - 2.0 * omega_r) * (OMEGA_M * a + omega_r).sqrt()
                + 2.0 * omega_r.powf(1.5))
    };
    // Integrand in u = ln a: d(Δρ/ρ)/du = f_X Γ (N_H0/ρ_γ0) a e^{−Γt} / H.
    let integrand = |u: f64| {
        let a = u.exp();
        let hub = h0 * (omega_r / a.powi(4) + OMEGA_M / a.powi(3)).sqrt();
        f_x_ev * EV * gamma * n_h0 / rho_gamma0 * a * (-gamma * age(a)).exp() / hub
    };
    // Composite Simpson in ln a. The integrand is smooth, so 4000 intervals
    // put the quadrature error far below 1e-8 relative.
    let (u_lo, u_hi) = ((1.0 / (1.0 + z_hi)).ln(), (1.0 / (1.0 + z_lo)).ln());
    let n = 4000;
    let step = (u_hi - u_lo) / n as f64;
    let mut acc = integrand(u_lo) + integrand(u_hi);
    for i in 1..n {
        acc += if i % 2 == 1 { 4.0 } else { 2.0 } * integrand(u_lo + i as f64 * step);
    }
    acc * step / 3.0
}

/// A decaying particle with lifetime at z = 1000 (Γ = 1/t(z = 1000) =
/// 7.1838e-14 s⁻¹, f_X = 10 eV), run from z = 5e4 to 200, delivers 0.99417 of
/// its injected energy to the photons. The independent integration puts 0.53%
/// into adiabatic cooling of the gas excess and leaves 0.05% in the gas at
/// z = 200. This is finding N-4 of `dev/REVIEW_2026-09-22.md`. Target:
/// `python dev/scripts/heatloss/decay_delivery_expectation.py 10 7.1838e-14 5e4 200`.
///
/// Before ADR 0004 this run delivered 0.849 at default steps and 0.908 at a
/// hundredfold smaller Δτ_max.
#[test]
fn decay_at_recombination_delivers_independent_fraction() {
    const GAMMA: f64 = 7.1838e-14; // 1/s
    const F_X: f64 = 10.0; // eV
    const Z_START: f64 = 5e4;
    const Z_END: f64 = 200.0;
    const EXPECTED: f64 = 0.99417; // independent integration

    let injected = injected_decay_drho(F_X, GAMMA, Z_END, Z_START);
    // Guard on the helper itself: the A/B record quotes 6.5373e-9.
    assert!(
        (injected / 6.5373e-9 - 1.0).abs() < 1e-3,
        "injected Δρ/ρ {injected:e}, record quotes 6.5373e-9"
    );

    let decay = |f_x: f64| InjectionScenario::DecayingParticle {
        f_x,
        gamma_x: GAMMA,
    };
    let with = photon_drho(decay(F_X), Z_START, Z_END);
    // Validation needs f_X > 0; 1e-20 eV injects about 1e-29 in Δρ/ρ.
    let base = photon_drho(decay(1e-20), Z_START, Z_END);
    let delivered = (with - base) / injected;
    eprintln!(
        "decay tau(z=1000): injected={injected:e} with={with:e} base={base:e} \
         delivered={delivered:.6}"
    );
    assert!(
        (delivered - EXPECTED).abs() < 1e-3,
        "delivered fraction {delivered:.6}, expected {EXPECTED} ± 1e-3"
    );
}
