//! Adversarial input tests: verify that bad inputs produce clear errors,
//! not garbage results or unhelpful panics.
//!
//! Tests are organized as parameterized loops to cover many cases concisely.
//! Each group documents the attack vectors and expected defenses.

use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::RefinementZone;
use spectroxide::prelude::*;

// ==========================================================================
// Section A: Bad Cosmology parameters
// ==========================================================================

/// Cosmology::new() should reject non-positive, non-finite, or out-of-range params.
#[test]
fn test_cosmo_rejects_bad_params() {
    let cases: Vec<(f64, f64, f64, f64, f64, f64, &str)> = vec![
        (2.726, 0.022, 0.12, 0.0, 3.046, 0.24, "h=0"),
        (0.0, 0.022, 0.12, 0.71, 3.046, 0.24, "T_cmb=0"),
        (f64::NAN, 0.022, 0.12, 0.71, 3.046, 0.24, "T_cmb=NaN"),
        (
            f64::NEG_INFINITY,
            0.022,
            0.12,
            0.71,
            3.046,
            0.24,
            "T_cmb=-Inf",
        ),
        (2.726, 0.022, 0.12, 0.71, 3.046, 1.0, "Y_p=1"),
        (2.726, 0.022, 0.12, 0.71, 3.046, -0.1, "Y_p<0"),
        (2.726, 0.022, 0.12, -0.7, 3.046, 0.24, "negative h"),
        (2.726, 0.022, 0.12, f64::NAN, 3.046, 0.24, "NaN h"),
        (2.726, 0.022, 0.12, f64::INFINITY, 3.046, 0.24, "Inf h"),
        (2.726, 0.022, 0.12, 100.0, 3.046, 0.24, "huge h"),
        (2.726, -0.022, 0.12, 0.71, 3.046, 0.24, "negative omega_b"),
        (2.726, 0.022, -0.12, 0.71, 3.046, 0.24, "negative omega_cdm"),
        (2.726, 0.022, 0.12, 0.71, -1.0, 0.24, "negative N_eff"),
        (2.726, 0.022, 0.12, 0.71, 100.0, 0.24, "huge N_eff"),
    ];
    for (t_cmb, omega_b_h2, omega_cdm_h2, h, n_eff, y_p, label) in cases {
        let result = Cosmology::new(t_cmb, omega_b_h2, omega_cdm_h2, h, n_eff, y_p);
        assert!(result.is_err(), "Cosmology::new() should reject {label}");
    }
}

// ==========================================================================
// Section B: Bad GridConfig
// ==========================================================================

/// validate() should reject bad grid parameters.
#[test]
fn test_grid_rejects_bad_params() {
    let bad_configs = vec![
        (
            GridConfig {
                x_min: 0.0,
                ..GridConfig::default()
            },
            "x_min=0",
        ),
        (
            GridConfig {
                x_min: -1.0,
                ..GridConfig::default()
            },
            "x_min<0",
        ),
        (
            GridConfig {
                x_min: 10.0,
                x_max: 1.0,
                x_transition: 5.0,
                ..GridConfig::default()
            },
            "x_min>x_max",
        ),
        (
            GridConfig {
                n_points: 0,
                ..GridConfig::default()
            },
            "n_points=0",
        ),
        (
            GridConfig {
                n_points: 1,
                ..GridConfig::default()
            },
            "n_points=1",
        ),
        (
            GridConfig {
                x_max: 5.0,
                x_transition: 0.1,
                ..GridConfig::default()
            },
            "x_max too small",
        ),
        (
            GridConfig {
                log_fraction: -0.5,
                ..GridConfig::default()
            },
            "log_fraction<0",
        ),
        (
            GridConfig {
                log_fraction: 1.5,
                ..GridConfig::default()
            },
            "log_fraction>1",
        ),
        (
            GridConfig {
                log_fraction: 1.0,
                ..GridConfig::default()
            },
            "log_fraction=1",
        ),
        (
            GridConfig {
                x_transition: 100.0,
                ..GridConfig::default()
            },
            "x_transition outside",
        ),
        (
            GridConfig {
                x_min: f64::NAN,
                ..GridConfig::default()
            },
            "NaN x_min",
        ),
        (
            GridConfig {
                x_max: f64::NAN,
                ..GridConfig::default()
            },
            "NaN x_max",
        ),
    ];
    for (gc, label) in bad_configs {
        assert!(
            gc.validate().is_err(),
            "validate() should reject grid with {label}"
        );
    }
}

/// Grid constructors should panic on degenerate inputs.
#[test]
fn test_grid_constructor_panics() {
    assert!(
        std::panic::catch_unwind(|| FrequencyGrid::from_points(vec![])).is_err(),
        "empty points"
    );
    assert!(
        std::panic::catch_unwind(|| FrequencyGrid::from_points(vec![1.0])).is_err(),
        "single point"
    );
    assert!(
        std::panic::catch_unwind(|| FrequencyGrid::log_uniform(0.01, 50.0, 1)).is_err(),
        "n=1"
    );
}

/// Refinement zone validation.
#[test]
fn test_grid_refinement_zone_rejects_bad() {
    let bad_zones = vec![
        (
            RefinementZone {
                x_center: 5.0,
                x_width: 0.0,
                n_points: 100,
            },
            "zero width",
        ),
        (
            RefinementZone {
                x_center: 5.0,
                x_width: 1.0,
                n_points: 0,
            },
            "zero points",
        ),
        (
            RefinementZone {
                x_center: 200.0,
                x_width: 1.0,
                n_points: 100,
            },
            "outside range",
        ),
    ];
    for (zone, label) in bad_zones {
        let gc = GridConfig {
            refinement_zones: vec![zone],
            ..GridConfig::default()
        };
        assert!(
            gc.validate().is_err(),
            "Should reject refinement zone: {label}"
        );
    }
}

// ==========================================================================
// Section C: Bad SolverConfig
// ==========================================================================

#[test]
fn test_solver_config_rejects_bad_params() {
    let bad_configs = vec![
        (
            SolverConfig {
                z_start: 100.0,
                z_end: 1e6,
                ..SolverConfig::default()
            },
            "backwards z",
        ),
        (
            SolverConfig {
                z_start: 1e5,
                z_end: 1e5,
                ..SolverConfig::default()
            },
            "equal z",
        ),
        (
            SolverConfig {
                dy_max: 0.0,
                ..SolverConfig::default()
            },
            "zero dy_max",
        ),
        (
            SolverConfig {
                dtau_max: -1.0,
                ..SolverConfig::default()
            },
            "negative dtau_max",
        ),
        (
            SolverConfig {
                dy_max: f64::NAN,
                ..SolverConfig::default()
            },
            "NaN dy_max",
        ),
        (
            SolverConfig {
                dz_min: -1.0,
                ..SolverConfig::default()
            },
            "negative dz_min",
        ),
        (
            SolverConfig {
                z_start: -1.0,
                ..SolverConfig::default()
            },
            "negative z_start",
        ),
        (
            SolverConfig {
                z_end: -1.0,
                ..SolverConfig::default()
            },
            "negative z_end",
        ),
        (
            SolverConfig {
                z_start: 1e8,
                ..SolverConfig::default()
            },
            "z_start too high",
        ),
        (
            SolverConfig {
                max_newton_iter: 1,
                ..SolverConfig::default()
            },
            "max_newton_iter too low",
        ),
    ];
    for (sc, label) in bad_configs {
        assert!(sc.validate().is_err(), "validate() should reject: {label}");
    }
}

// ==========================================================================
// Section D: Bad InjectionScenario
// ==========================================================================

/// Single burst bad parameters.
#[test]
fn test_injection_burst_rejects_bad() {
    let bad = [
        InjectionScenario::SingleBurst {
            z_h: 1e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 0.0,
        },
        InjectionScenario::SingleBurst {
            z_h: 1e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: -100.0,
        },
        InjectionScenario::SingleBurst {
            z_h: 1e5,
            delta_rho_over_rho: f64::NAN,
            sigma_z: 100.0,
        },
        InjectionScenario::SingleBurst {
            z_h: 1e5,
            delta_rho_over_rho: f64::INFINITY,
            sigma_z: 100.0,
        },
        InjectionScenario::SingleBurst {
            z_h: f64::NAN,
            delta_rho_over_rho: 1e-5,
            sigma_z: 100.0,
        },
    ];
    for (i, inj) in bad.iter().enumerate() {
        assert!(inj.validate().is_err(), "Burst case {i} should fail");
    }
}

/// DM and decay scenarios bad parameters.
#[test]
fn test_injection_dm_decay_rejects_bad() {
    let bad = [
        InjectionScenario::DecayingParticle {
            f_x: 0.0,
            gamma_x: 1e-20,
        },
        InjectionScenario::DecayingParticle {
            f_x: 1e3,
            gamma_x: -1e-20,
        },
        InjectionScenario::AnnihilatingDM { f_ann: f64::NAN },
    ];
    for (i, inj) in bad.iter().enumerate() {
        assert!(inj.validate().is_err(), "DM/decay case {i} should fail");
    }
}

/// Photon injection and tabulated source bad parameters.
#[test]
fn test_injection_photon_tabulated_rejects_bad() {
    #[allow(clippy::useless_vec)] // Inner vec![] needed for TabulatedHeating fields
    let bad = vec![
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: f64::NAN,
            delta_n_over_n: 1e-4,
            z_h: 5e4,
            sigma_z: 2000.0,
            sigma_x: 0.005,
        },
        InjectionScenario::TabulatedHeating {
            z_table: vec![],
            rate_table: vec![],
        },
        InjectionScenario::TabulatedHeating {
            z_table: vec![1e3, 1e4, 1e5],
            rate_table: vec![1.0, 2.0],
        },
        InjectionScenario::DecayingParticlePhoton {
            x_inj_0: 1.0,
            f_inj: 1e-10,
            gamma_x: f64::NAN,
        },
    ];
    for (i, inj) in bad.iter().enumerate() {
        assert!(
            inj.validate().is_err(),
            "Photon/tabulated case {i} should fail"
        );
    }
}

// ==========================================================================
// Section E: Builder and compound tests
// ==========================================================================

#[test]
fn test_builder_rejects_bad_inputs() {
    let cosmo = Cosmology::default();

    // Bad cosmology — use the unchecked constructor to obtain an invalid
    // Cosmology instance and verify the builder rejects it.
    let bad_cosmo = Cosmology::new_unchecked(2.726, 0.022, 0.12, -0.7, 3.046, 0.24);
    assert!(
        ThermalizationSolver::builder(bad_cosmo)
            .grid_fast()
            .build()
            .is_err()
    );

    // Bad grid
    assert!(
        ThermalizationSolver::builder(cosmo.clone())
            .grid(GridConfig {
                n_points: 0,
                ..GridConfig::default()
            })
            .build()
            .is_err()
    );

    // Bad injection
    assert!(
        ThermalizationSolver::builder(cosmo.clone())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 1e5,
                delta_rho_over_rho: f64::NAN,
                sigma_z: 100.0,
            })
            .build()
            .is_err()
    );

    // set_injection() rejects bad input
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::fast());
    assert!(
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 1e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: -100.0,
            })
            .is_err()
    );

    // z_start below injection window
    assert!(
        ThermalizationSolver::builder(cosmo)
            .injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: 3000.0,
            })
            .z_range(1e5, 1e4)
            .build()
            .is_err()
    );
}

#[test]
fn test_warning_thresholds() {
    // Strong distortion warning
    let strong = InjectionScenario::SingleBurst {
        z_h: 1e5,
        delta_rho_over_rho: 0.1,
        sigma_z: 3000.0,
    };
    assert!(!strong.warn_strong_distortion().is_empty());

    // Small distortion: no warning
    let weak = InjectionScenario::SingleBurst {
        z_h: 1e5,
        delta_rho_over_rho: 1e-5,
        sigma_z: 3000.0,
    };
    assert!(weak.warn_strong_distortion().is_empty());
}

// ==========================================================================
// Section G: Injection edge cases and characteristic redshifts
// ==========================================================================

#[test]
fn test_injection_characteristic_redshifts() {
    // Burst: has characteristic redshift
    let burst = InjectionScenario::SingleBurst {
        z_h: 2e5,
        delta_rho_over_rho: 1e-5,
        sigma_z: 3000.0,
    };
    let (z_center, z_upper) = burst.characteristic_redshift().unwrap();
    assert!((z_center - 2e5).abs() < 1.0);
    assert!(z_upper > z_center);
    assert!((z_upper - (2e5 + 7.0 * 3000.0)).abs() < 1.0);

    // Continuous: no characteristic redshift
    assert!(
        InjectionScenario::DecayingParticle {
            f_x: 1e3,
            gamma_x: 1e-13
        }
        .characteristic_redshift()
        .is_none()
    );
}

// ==========================================================================
// Section H: Missing validation coverage
// ==========================================================================

#[test]
fn test_injection_monochromatic_rejects_bad() {
    let bad_cases = [
        // x_inj = 0
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 0.0,
            delta_n_over_n: 1e-5,
            z_h: 2e5,
            sigma_z: 3000.0,
            sigma_x: 0.01,
        },
        // x_inj < 0
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: -1.0,
            delta_n_over_n: 1e-5,
            z_h: 2e5,
            sigma_z: 3000.0,
            sigma_x: 0.01,
        },
        // sigma_x = 0
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 1.0,
            delta_n_over_n: 1e-5,
            z_h: 2e5,
            sigma_z: 3000.0,
            sigma_x: 0.0,
        },
        // sigma_z = 0
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 1.0,
            delta_n_over_n: 1e-5,
            z_h: 2e5,
            sigma_z: 0.0,
            sigma_x: 0.01,
        },
        // NaN delta_n_over_n
        InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 1.0,
            delta_n_over_n: f64::NAN,
            z_h: 2e5,
            sigma_z: 3000.0,
            sigma_x: 0.01,
        },
    ];
    for (i, case) in bad_cases.iter().enumerate() {
        assert!(
            case.validate().is_err(),
            "MonochromaticPhotonInjection case {i} should be rejected"
        );
    }
}

#[test]
fn test_injection_decaying_particle_photon_rejects_bad() {
    let bad_cases = [
        // x_inj_0 = 0
        InjectionScenario::DecayingParticlePhoton {
            f_inj: 1e-5,
            gamma_x: 1e-13,
            x_inj_0: 0.0,
        },
        // x_inj_0 < 0
        InjectionScenario::DecayingParticlePhoton {
            f_inj: 1e-5,
            gamma_x: 1e-13,
            x_inj_0: -1.0,
        },
        // gamma_x = 0
        InjectionScenario::DecayingParticlePhoton {
            f_inj: 1e-5,
            gamma_x: 0.0,
            x_inj_0: 5.0,
        },
        // gamma_x < 0
        InjectionScenario::DecayingParticlePhoton {
            f_inj: 1e-5,
            gamma_x: -1e-13,
            x_inj_0: 5.0,
        },
        // NaN f_inj
        InjectionScenario::DecayingParticlePhoton {
            f_inj: f64::NAN,
            gamma_x: 1e-13,
            x_inj_0: 5.0,
        },
    ];
    for (i, case) in bad_cases.iter().enumerate() {
        assert!(
            case.validate().is_err(),
            "DecayingParticlePhoton case {i} should be rejected"
        );
    }
}

#[test]
fn test_injection_annihilating_dm_pwave_rejects_bad() {
    let bad_cases = [
        InjectionScenario::AnnihilatingDMPWave { f_ann: f64::NAN },
        InjectionScenario::AnnihilatingDMPWave {
            f_ann: f64::INFINITY,
        },
    ];
    for (i, case) in bad_cases.iter().enumerate() {
        assert!(
            case.validate().is_err(),
            "AnnihilatingDMPWave case {i} should be rejected"
        );
    }
}

// ==========================================================================
// Section G: Numerically adversarial but valid inputs
// ==========================================================================

/// Injection at very narrow sigma_z (σ_z = 10) should not crash.
/// The solver's adaptive stepping must handle this near-delta function.
#[test]
fn test_very_narrow_sigma_z() {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid_fast()
        .injection(InjectionScenario::SingleBurst {
            delta_rho_over_rho: 1e-5,
            z_h: 1e5,
            sigma_z: 10.0,
        })
        .z_range(1.1e5, 9e4)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[9e4]);
    let snap = solver.snapshots.last().unwrap();

    // Should produce a finite, nonzero distortion
    assert!(
        snap.delta_rho_over_rho.is_finite(),
        "Very narrow σ_z: Δρ/ρ should be finite, got {}",
        snap.delta_rho_over_rho
    );
    // Energy should be deposited (not entirely missed)
    assert!(
        snap.delta_rho_over_rho.abs() > 1e-7,
        "Very narrow σ_z: Δρ/ρ should be nonzero, got {:.4e}",
        snap.delta_rho_over_rho
    );
}

/// Two near-simultaneous bursts at the same z should produce ~2x the signal.
#[test]
fn test_two_near_simultaneous_bursts() {
    let cosmo = Cosmology::default();
    let drho = 1e-5;
    let z_h = 2e5;
    let sigma_z = 3000.0;

    // Single burst
    let mut solver1 = ThermalizationSolver::builder(cosmo.clone())
        .grid_fast()
        .injection(InjectionScenario::SingleBurst {
            delta_rho_over_rho: drho,
            z_h,
            sigma_z,
        })
        .z_range(3e5, 200.0)
        .build()
        .unwrap();
    solver1.run_with_snapshots(&[200.0]);
    let mu1 = solver1.snapshots.last().unwrap().mu;

    // Double amplitude
    let mut solver2 = ThermalizationSolver::builder(cosmo.clone())
        .grid_fast()
        .injection(InjectionScenario::SingleBurst {
            delta_rho_over_rho: 2.0 * drho,
            z_h,
            sigma_z,
        })
        .z_range(3e5, 200.0)
        .build()
        .unwrap();
    solver2.run_with_snapshots(&[200.0]);
    let mu2 = solver2.snapshots.last().unwrap().mu;

    // Linearity: μ(2×Δρ) ≈ 2×μ(Δρ) to < 5% for small distortions
    let ratio = mu2 / mu1;
    eprintln!("Linearity: μ(2Δρ)/μ(Δρ) = {ratio:.4} (expected ~2.0)");
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "Linearity violated: ratio = {ratio:.4} (expected ~2.0)"
    );
}

/// Injection near z_start boundary should not lose energy.
///
/// Uses direct solver construction (not builder) to bypass the z_start
/// validation, testing what happens when the solver starts during injection.
#[test]
fn test_injection_near_z_start() {
    let z_h = 2.5e5;
    let sigma_z = 3000.0;

    // Tight z_start: only 3σ above z_h (bypasses builder validation)
    let z_start_tight = z_h + 3.0 * sigma_z; // 2.59e5

    let mut solver = ThermalizationSolver::new(Cosmology::default(), GridConfig::fast());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            delta_rho_over_rho: 1e-5,
            z_h,
            sigma_z,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start: z_start_tight,
        z_end: 200.0,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[200.0]);
    let snap_tight = solver.snapshots.last().unwrap();

    // Wide z_start: 7σ above z_h (standard)
    let z_start_wide = z_h + 7.0 * sigma_z; // 2.71e5

    let mut solver2 = ThermalizationSolver::builder(Cosmology::default())
        .grid_fast()
        .injection(InjectionScenario::SingleBurst {
            delta_rho_over_rho: 1e-5,
            z_h,
            sigma_z,
        })
        .z_range(z_start_wide, 200.0)
        .build()
        .unwrap();
    solver2.run_with_snapshots(&[200.0]);
    let snap_wide = solver2.snapshots.last().unwrap();

    // Tight start loses energy from the upper Gaussian tail (>3σ ≈ 0.1%)
    // but should not lose > 10%
    let energy_ratio =
        snap_tight.delta_rho_over_rho.abs() / snap_wide.delta_rho_over_rho.abs().max(1e-30);
    eprintln!(
        "Energy near z_start: tight/wide = {energy_ratio:.4} (tight Δρ={:.4e}, wide Δρ={:.4e})",
        snap_tight.delta_rho_over_rho, snap_wide.delta_rho_over_rho
    );
    assert!(
        energy_ratio > 0.90,
        "Tight z_start lost > 10% energy: ratio = {energy_ratio:.4}"
    );
}
