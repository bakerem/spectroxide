//! One-off: measure the temporal discretization error of μ at the DEFAULT
//! dy_max = 0.02 directly (no extrapolation), against a dy_max = 0.001
//! reference. Scenario matches tests/convergence_order.rs::run_full_physics.

use spectroxide::prelude::*;

fn run(dy_max: f64) -> (f64, f64, usize) {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points: 4000,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h: 2.0e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 3000.0,
        })
        .z_range(5.0e5, 1.0e4)
        .dy_max(dy_max)
        .dtau_max(200.0)
        .build()
        .unwrap();
    solver.run_with_snapshots(&[1.0e4]);
    let s = solver.snapshots.last().unwrap();
    (s.mu, s.y, solver.step_count)
}

fn run_dtau(dtau_max: f64) -> (f64, f64, usize) {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points: 4000,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h: 2.0e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 3000.0,
        })
        .z_range(5.0e5, 1.0e4)
        .dtau_max(dtau_max)
        .build()
        .unwrap();
    solver.run_with_snapshots(&[1.0e4]);
    let s = solver.snapshots.last().unwrap();
    (s.mu, s.y, solver.step_count)
}

fn main() {
    for dy in [0.02, 0.01, 0.001] {
        let (mu, y, steps) = run(dy);
        println!("dy_max={dy}: mu={mu:.8e} y={y:.8e} steps={steps}");
    }
    // Production temporal control: dtau_max refinement at default dy_max=0.02.
    for dtau in [20.0, 10.0, 5.0, 2.5] {
        let (mu, y, steps) = run_dtau(dtau);
        println!("dtau_max={dtau}: mu={mu:.8e} y={y:.8e} steps={steps}");
    }
    // Also with the default dtau_max (10) rather than the test value 200,
    // i.e. the actual production configuration.
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points: 4000,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h: 2.0e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 3000.0,
        })
        .z_range(5.0e5, 1.0e4)
        .build()
        .unwrap();
    solver.run_with_snapshots(&[1.0e4]);
    let s = solver.snapshots.last().unwrap();
    println!(
        "defaults (dy=0.02, dtau_max=10): mu={:.8e} y={:.8e} steps={}",
        s.mu, s.y, solver.step_count
    );
}
