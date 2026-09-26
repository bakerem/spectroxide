//! Hydrogen ionization evolved with the electron temperature (ADR 0009).
//!
//! The solver advances X_H with the Peebles three-level atom, with the
//! recombination coefficient at the gas temperature T_e = ρ_e T_z. These tests
//! pin the coupled history against targets from outside the code:
//!
//! - HyRec-2 (github.com/nanoomlee/HyRec-2, run 2026-07-05 on the default
//!   cosmology; `dev/output/hyrec2_xe_default_cosmo.dat`, columns z, X_e,
//!   T_m in K). HyRec evolves T_m, so below z ≈ 500, where the gas cools
//!   faster than the photons, it separates the coupled history from the old
//!   fixed one, which put the gas at T_z.
//! - The fixed-history mode, which must reproduce the standard table exactly
//!   and must leave every run that ends above hydrogen recombination
//!   unchanged.

mod common;

use common::fast_grid;
use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::recombination::RecombinationHistory;
use spectroxide::solver::{SolverSnapshot, ThermalizationSolver};

const T_CMB: f64 = 2.726; // K, the default cosmology and the HyRec-2 run

/// Runs the no-injection solver from z = 3000 to 50 and returns snapshots at `zs`.
fn null_run(zs: &[f64], fixed: bool) -> Vec<SolverSnapshot> {
    let mut b = ThermalizationSolver::builder(Cosmology::default())
        .grid(fast_grid())
        .z_range(3000.0, 50.0);
    if fixed {
        b = b.fixed_ionization();
    }
    let mut solver = b.build().unwrap();
    solver.run_with_snapshots(zs).to_vec()
}

/// X_e and T_e of a run with no injection against HyRec-2 after freeze-out.
///
/// HyRec-2 values (X_e, T_m/K) typed from the archived table. The bands are
/// several times the measured offsets of the coupled history (X_e: −0.10%,
/// −0.10%, +0.18% at z = 200, 100, 50; T_e: 0.07%, 0.12%, 0.23%; with the
/// escape-rate correction of ADR 0011). Before the
/// coupled-mode step cap (ADR 0009 addendum) T_e was off by 0.7%, 1.4%, 2.4%. The fixed history is
/// off by 2.1%, 5.3%, and 10.3% in X_e and fails every X_e band here; that is
/// checked below, so the test tells the two apart.
#[test]
fn null_run_matches_hyrec_after_freeze_out() {
    // (z, HyRec X_e, HyRec T_m [K], X_e band, T_e band)
    let anchors = [
        (200.0, 3.2684558721e-4, 4.6695799420e2, 0.015, 0.005),
        (100.0, 2.6391260679e-4, 1.6809854560e2, 0.015, 0.005),
        (50.0, 2.3023894096e-4, 5.0843003834e1, 0.02, 0.005),
    ];
    let zs: Vec<f64> = anchors.iter().map(|a| a.0).collect();
    let coupled = null_run(&zs, false);
    let fixed = null_run(&zs, true);
    for (z, xe_hy, tm_hy, xe_band, te_band) in anchors {
        let s = coupled.iter().find(|s| s.z == z).unwrap();
        let f = fixed.iter().find(|s| s.z == z).unwrap();
        let rel_xe = s.x_e / xe_hy - 1.0;
        let rel_te = s.rho_e * T_CMB * (1.0 + z) / tm_hy - 1.0;
        let rel_fixed = f.x_e / xe_hy - 1.0;
        eprintln!(
            "z={z}: X_e {:.5e} (HyRec {xe_hy:.5e}, rel {rel_xe:+.2e}; fixed rel {rel_fixed:+.2e}), \
             T_e rel {rel_te:+.2e}",
            s.x_e
        );
        assert!(
            rel_xe.abs() < xe_band,
            "z={z}: X_e rel {rel_xe:+.3e} > {xe_band}"
        );
        assert!(
            rel_te.abs() < te_band,
            "z={z}: T_e rel {rel_te:+.3e} > {te_band}"
        );
        assert!(
            rel_fixed.abs() > xe_band,
            "z={z}: the fixed history (rel {rel_fixed:+.3e}) passes the band, so the \
             anchor no longer tests the coupling"
        );
    }
}

/// The fixed-history mode returns the standard table's X_e exactly.
#[test]
fn fixed_ionization_reproduces_standard_table() {
    let zs = [1500.0, 1100.0, 800.0, 300.0, 100.0];
    let snaps = null_run(&zs, true);
    let table = RecombinationHistory::new(&Cosmology::default());
    for s in &snaps {
        assert_eq!(s.x_e, table.x_e(s.z), "z={}", s.z);
    }
}

/// Above hydrogen recombination the coupled and fixed modes are identical bit
/// for bit, however hot the gas: hydrogen is in Saha equilibrium at T_z there.
#[test]
fn runs_above_recombination_are_unchanged() {
    let run = |fixed: bool| {
        let mut b = ThermalizationSolver::builder(Cosmology::default())
            .grid(fast_grid())
            .injection(InjectionScenario::SingleBurst {
                z_h: 5000.0,
                delta_rho_over_rho: 3e-3,
                sigma_z: 200.0,
            })
            .z_range(7000.0, 1700.0);
        if fixed {
            b = b.fixed_ionization();
        }
        let mut solver = b.build().unwrap();
        solver
            .run_with_snapshots(&[1700.0])
            .last()
            .cloned()
            .unwrap()
    };
    let (a, b) = (run(false), run(true));
    assert_eq!(a.x_e, b.x_e);
    assert_eq!(a.rho_e, b.rho_e);
    assert_eq!(a.delta_n, b.delta_n);
    assert_eq!(a.mu, b.mu);
    assert_eq!(a.y, b.y);
}

/// Runs a burst from z = 1700 to 200 with snapshots every `dz_snap` in z,
/// which caps the step size, and returns (Δρ/ρ, X_e at z = 200).
fn heated_run(z_h: f64, drho: f64, dz_snap: Option<f64>, fixed: bool) -> (f64, f64) {
    let mut b = ThermalizationSolver::builder(Cosmology::default())
        .grid(fast_grid())
        .injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: f64::max(0.04 * z_h, 100.0),
        })
        .z_range(1700.0, 200.0);
    if fixed {
        b = b.fixed_ionization();
    }
    let mut solver = b.build().unwrap();
    let mut zs = vec![200.0];
    if let Some(dz) = dz_snap {
        let mut z = 1700.0 - dz;
        while z > 200.0 {
            zs.push(z);
            z -= dz;
        }
    }
    let s = solver.run_with_snapshots(&zs).last().cloned().unwrap();
    (s.delta_rho_over_rho, s.x_e)
}

/// Smooth heating through recombination converges at the default steps.
///
/// A decaying particle with lifetime at z ≈ 1000 (f_X = 1e4 eV) holds ρ_e
/// between 1.5 and 7 from z = 1000 to 800. The backward-Euler T_e step is
/// first order, and X_H inherits its error. At the old 0.05 z step cap X_e
/// came out 3%, 7%, and 7% low at z = 1000, 900, and 800; with the coupled-mode
/// cap `XE_COUPLED_DZ_FRAC` = 0.005 it is 0.37%, 0.76%, and 0.81% low against
/// Δz = 0.25. The band, 1.5%, fails at the old cap.
#[test]
fn smooth_heating_xe_converges_with_step() {
    let zs = [1000.0, 900.0, 800.0];
    let coarse = decay_xe(1e4, &zs, None);
    let fine = decay_xe(1e4, &zs, Some(0.25));
    for ((z, c), f) in zs.iter().zip(&coarse).zip(&fine) {
        let rel = c / f - 1.0;
        eprintln!("z={z}: X_e default {c:.6e}, dz=0.25 {f:.6e}, rel {rel:+.3e}");
        assert!(rel.abs() < 0.015, "z={z}: X_e step error {rel:+.3e}");
    }
}

/// A burst hot enough to slow recombination, against an independent
/// integration of the coupled X_H and T_m equations
/// (`dev/scripts/heatloss/coupled_xe_expectation.py`, written from Peebles
/// 1968 and RECFAST without reading the solver).
///
/// Burst at z_h = 600, σ_z = 100, Δρ/ρ = 1e-6, run from z = 1700 to 200; the
/// gas peaks at T_m ≈ 22 T_z. The oracle gives X_e(200) = 8.902e-4, against
/// 3.26e-4 with no heating, and a delivered fraction of 0.9970575
/// (`600 --drho 1e-6`). With both runs on the standard X_H history, as the
/// solver's fixed mode has them, it gives 0.9909854 (`--table-xe`), so the
/// coupling adds 6.0721e-3. Oracle values include RECFAST's escape-rate
/// correction (ADR 0011).
///
/// The test checks X_e(200) and that change, the coupled minus the fixed-mode
/// delivered fraction; the difference cancels the 1000-point grid's delivery
/// offset (about 4e-4), common to both modes. The fixed control runs at
/// Δz = 1 because the coupled-mode step cap does not apply to it. Measured:
/// X_e +0.07%, change −0.37%. The fixed history misses X_e by −63% and the
/// change by −100%.
///
/// Both codes omit collisional ionization. Real gas at 3.5e4 K would ionize by
/// collisions far faster than this; the test pins the three-level atom model
/// of ADR 0009, not the real gas.
#[test]
fn hot_burst_matches_independent_coupled_integration() {
    const XE_ORACLE: f64 = 8.902e-4;
    const GAIN_ORACLE: f64 = 0.9970575 - 0.9909854;
    let delivered = |fixed: bool| {
        let dz = if fixed { Some(1.0) } else { None };
        let (with, xe) = heated_run(600.0, 1e-6, dz, fixed);
        let (base, _) = heated_run(600.0, 1e-30, dz, fixed);
        ((with - base) / 1e-6, xe)
    };
    let (f_coupled, xe) = delivered(false);
    let (f_fixed, _) = delivered(true);
    let gain = f_coupled - f_fixed;
    let rel_xe = xe / XE_ORACLE - 1.0;
    let rel_gain = gain / GAIN_ORACLE - 1.0;
    eprintln!(
        "coupled {f_coupled:.6}, fixed {f_fixed:.6}, gain {gain:.4e} (oracle {GAIN_ORACLE:.4e}, \
         rel {rel_gain:+.2e}); X_e(200) {xe:.4e} (oracle {XE_ORACLE:.4e}, rel {rel_xe:+.2e})"
    );
    assert!(rel_xe.abs() < 0.01, "X_e(200) rel {rel_xe:+.3e}");
    assert!(rel_gain.abs() < 0.015, "delivery gain rel {rel_gain:+.3e}");
}

/// Runs a decaying particle (Γ = 7.1838e-14 s⁻¹, lifetime at z ≈ 1000) from
/// z = 3000 to 200 and returns X_e at `zs_out`. `dz_snap` caps the step.
fn decay_xe(f_x: f64, zs_out: &[f64], dz_snap: Option<f64>) -> Vec<f64> {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(fast_grid())
        .injection(InjectionScenario::DecayingParticle {
            f_x,
            gamma_x: 7.1838e-14,
        })
        .z_range(3000.0, 200.0)
        .build()
        .unwrap();
    let mut zs: Vec<f64> = zs_out.to_vec();
    if let Some(dz) = dz_snap {
        let mut z = 3000.0 - dz;
        while z > 200.0 {
            zs.push(z);
            z -= dz;
        }
    }
    let snaps = solver.run_with_snapshots(&zs).to_vec();
    zs_out
        .iter()
        .map(|&z| snaps.iter().find(|s| s.z == z).unwrap().x_e)
        .collect()
}
