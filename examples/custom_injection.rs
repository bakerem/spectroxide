//! Example: custom injection scenarios with the PDE solver.
//!
//! Demonstrates how to define arbitrary energy injection histories using
//! `InjectionScenario::Custom` and evolve them through the full PDE.
//!
//! Four examples:
//!   1. Power-law heating: d(Δρ/ρ)/dt ∝ (1+z)^α
//!   2. Two-burst injection: sum of Gaussians at different redshifts
//!   3. Wrapping a built-in scenario with modifications (redshift cutoff)
//!   4. Tabulated heating loaded from a CSV file (Python interface mechanism)
//!
//! Run with:
//!   cargo run --release --example custom_injection

use spectroxide::prelude::*;

fn main() {
    let cosmo = Cosmology::default();

    // =========================================================================
    // Example 1: Power-law heating rate
    //
    //   d(Δρ/ρ)/dt = A × (1+z)^α     [units: 1/s]
    //
    // Steep positive powers concentrate energy at high z → μ-dominated.
    // Negative/flat powers spread energy to low z → y-dominated.
    // =========================================================================
    eprintln!("=== Example 1: Power-law heating ===\n");

    for alpha in [-2.0_f64, 0.0, 3.0, 5.0] {
        // Small amplitude to stay in the linear regime
        let amplitude = 1e-30;

        let scenario = InjectionScenario::Custom(Box::new(move |z: f64, _cosmo: &Cosmology| {
            amplitude * (1.0 + z).powf(alpha)
        }));

        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver.config.z_start = 3e6;
        solver.config.z_end = 500.0;
        solver.set_injection(scenario).unwrap();

        let snaps = solver.run_with_snapshots(&[500.0]);
        if let Some(snap) = snaps.last() {
            eprintln!(
                "  alpha = {:+.0}: mu = {:+.4e}, y = {:+.4e}, Δρ/ρ = {:+.4e}",
                alpha, snap.mu, snap.y, snap.delta_rho_over_rho
            );
        }
    }

    // =========================================================================
    // Example 2: Two-burst injection
    //
    // Sum of two Gaussian bursts at z₁ = 3×10⁵ (μ-era) and z₂ = 5×10³ (y-era).
    // The PDE handles the full nonlinear thermalization at each epoch.
    // =========================================================================
    eprintln!("\n=== Example 2: Two-burst injection ===\n");

    let z1 = 3e5_f64;
    let sigma1 = 1.2e4_f64;
    let drho1 = 5e-6_f64;

    let z2 = 5e3_f64;
    let sigma2 = 200.0_f64;
    let drho2 = 5e-6_f64;

    // Custom closure: sum of two Gaussian bursts
    // heating_rate = d(Δρ/ρ)/dt = d(Δρ/ρ)/dz × H(z) × (1+z)
    let two_burst = InjectionScenario::Custom(Box::new(move |z: f64, cosmo: &Cosmology| {
        let gauss = |z_h: f64, sigma: f64, drho: f64| -> f64 {
            let g = (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
            drho * g * cosmo.hubble(z) * (1.0 + z)
        };
        gauss(z1, sigma1, drho1) + gauss(z2, sigma2, drho2)
    }));

    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver.config.z_start = 3e6;
    solver.config.z_end = 500.0;
    solver.set_injection(two_burst).unwrap();

    let snaps = solver.run_with_snapshots(&[500.0]);
    if let Some(snap) = snaps.last() {
        eprintln!(
            "  Two-burst: mu = {:+.4e}, y = {:+.4e}, Δρ/ρ = {:+.4e}",
            snap.mu, snap.y, snap.delta_rho_over_rho
        );
    }

    // Compare: run each burst individually with the built-in SingleBurst
    for (label, z_h, sigma, drho) in [
        ("Burst 1 (z=3e5, μ-era)", z1, sigma1, drho1),
        ("Burst 2 (z=5e3, y-era)", z2, sigma2, drho2),
    ] {
        let scenario = InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        };
        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver.config.z_start = 3e6;
        solver.config.z_end = 500.0;
        solver.set_injection(scenario).unwrap();
        let snaps = solver.run_with_snapshots(&[500.0]);
        if let Some(snap) = snaps.last() {
            eprintln!(
                "  {label}: mu = {:+.4e}, y = {:+.4e}, Δρ/ρ = {:+.4e}",
                snap.mu, snap.y, snap.delta_rho_over_rho
            );
        }
    }

    // =========================================================================
    // Example 3: Modified built-in scenario
    //
    // Take a decaying particle and add a redshift cutoff: the decay only
    // contributes above z > z_cut. This models a particle that is created
    // at z_cut and decays at higher redshifts.
    //
    // Custom wraps the built-in heating_rate() and zeroes it outside the window.
    // =========================================================================
    eprintln!("\n=== Example 3: Decaying particle with z-cutoff ===\n");

    let f_x = 5e5_f64;
    let gamma_x = 1e5_f64; // z_X = 1e5

    // Standard decaying particle (full z range)
    let standard = InjectionScenario::DecayingParticle { f_x, gamma_x };
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver.config.z_start = 3e6;
    solver.config.z_end = 500.0;
    solver.set_injection(standard).unwrap();
    let snaps = solver.run_with_snapshots(&[500.0]);
    if let Some(snap) = snaps.last() {
        eprintln!(
            "  Standard (all z):   mu = {:+.4e}, y = {:+.4e}",
            snap.mu, snap.y
        );
    }

    // Early-only: only inject for z > 5×10⁴ → should be μ-dominated
    let z_cut = 5e4_f64;
    let inner = InjectionScenario::DecayingParticle { f_x, gamma_x };
    let early_only = InjectionScenario::Custom(Box::new(move |z: f64, cosmo: &Cosmology| {
        if z > z_cut {
            inner.heating_rate(z, cosmo)
        } else {
            0.0
        }
    }));
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver.config.z_start = 3e6;
    solver.config.z_end = 500.0;
    solver.set_injection(early_only).unwrap();
    let snaps = solver.run_with_snapshots(&[500.0]);
    if let Some(snap) = snaps.last() {
        eprintln!(
            "  z > {z_cut:.0e} only:   mu = {:+.4e}, y = {:+.4e}",
            snap.mu, snap.y
        );
    }

    // Late-only: only inject for z < 5×10⁴ → should be y-dominated
    let inner = InjectionScenario::DecayingParticle { f_x, gamma_x };
    let late_only = InjectionScenario::Custom(Box::new(move |z: f64, cosmo: &Cosmology| {
        if z < z_cut {
            inner.heating_rate(z, cosmo)
        } else {
            0.0
        }
    }));
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
    solver.config.z_start = 3e6;
    solver.config.z_end = 500.0;
    solver.set_injection(late_only).unwrap();
    let snaps = solver.run_with_snapshots(&[500.0]);
    if let Some(snap) = snaps.last() {
        eprintln!(
            "  z < {z_cut:.0e} only:   mu = {:+.4e}, y = {:+.4e}",
            snap.mu, snap.y
        );
    }

    // =========================================================================
    // Example 4: Tabulated heating from a file
    //
    // Write a CSV file with a tabulated heating rate, then load it with
    // `load_heating_table()`. This is the mechanism used by the Python
    // interface to pass arbitrary user-defined functions to the PDE solver.
    // =========================================================================
    eprintln!("\n=== Example 4: Tabulated heating from file ===\n");

    {
        // Write a temporary CSV mimicking a SingleBurst at z=2e5
        let z_h = 2e5_f64;
        let sigma = z_h * 0.04;
        let drho = 1e-5_f64;
        let tmp_path = std::env::temp_dir().join("spectroxide_example_heating.csv");

        let mut f = std::fs::File::create(&tmp_path).expect("Failed to create temp file");
        use std::io::Write;
        writeln!(f, "z,dq_dz").unwrap();
        let n = 1000;
        let z_lo = (z_h - 6.0 * sigma).max(100.0);
        let z_hi = z_h + 6.0 * sigma;
        for i in 0..n {
            let z = z_lo + (z_hi - z_lo) * i as f64 / (n - 1) as f64;
            let gauss = (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
            writeln!(f, "{z:.6e},{:.6e}", drho * gauss).unwrap();
        }
        drop(f);

        // Load and run
        let scenario =
            spectroxide::energy_injection::load_heating_table(tmp_path.to_str().unwrap()).unwrap();
        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver.config.z_start = z_h + 7.0 * sigma;
        solver.config.z_end = 500.0;
        solver.set_injection(scenario).unwrap();
        let snaps = solver.run_with_snapshots(&[500.0]);
        if let Some(snap) = snaps.last() {
            eprintln!(
                "  Tabulated burst (z_h=2e5): mu = {:+.4e}, y = {:+.4e}, Δρ/ρ = {:+.4e}",
                snap.mu, snap.y, snap.delta_rho_over_rho
            );
        }

        // Compare with built-in SingleBurst
        let scenario = InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        };
        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver.config.z_start = z_h + 7.0 * sigma;
        solver.config.z_end = 500.0;
        solver.set_injection(scenario).unwrap();
        let snaps = solver.run_with_snapshots(&[500.0]);
        if let Some(snap) = snaps.last() {
            eprintln!(
                "  Built-in burst (z_h=2e5):  mu = {:+.4e}, y = {:+.4e}, Δρ/ρ = {:+.4e}",
                snap.mu, snap.y, snap.delta_rho_over_rho
            );
        }

        std::fs::remove_file(&tmp_path).ok();
    }

    eprintln!("\nDone.");
}
