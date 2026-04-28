//! Criterion benchmarks for the PDE solver.
//!
//! Run with: cargo bench
//!
//! Three canonical scenarios covering different solver regimes:
//!   - μ-era burst: tests high-z DC/BR thermalization
//!   - y-era burst: tests low-z Kompaneets redistribution
//!   - Photon injection: tests grid refinement + full T_e mode

use criterion::{Criterion, criterion_group, criterion_main};
use spectroxide::prelude::*;

fn bench_mu_era_burst(c: &mut Criterion) {
    c.bench_function("mu_era_burst_z5e5", |b| {
        b.iter(|| {
            let mut solver = ThermalizationSolver::builder(Cosmology::default())
                .grid(GridConfig {
                    n_points: 1000,
                    ..GridConfig::default()
                })
                .injection(InjectionScenario::SingleBurst {
                    z_h: 5e5,
                    delta_rho_over_rho: 1e-5,
                    sigma_z: 2e4,
                })
                .z_range(5e5 + 7.0 * 2e4, 1e4)
                .build()
                .unwrap();
            solver.run_to_result(1e4)
        })
    });
}

fn bench_y_era_burst(c: &mut Criterion) {
    c.bench_function("y_era_burst_z5e3", |b| {
        b.iter(|| {
            let mut solver = ThermalizationSolver::builder(Cosmology::default())
                .grid(GridConfig {
                    n_points: 1000,
                    ..GridConfig::default()
                })
                .injection(InjectionScenario::SingleBurst {
                    z_h: 5e3,
                    delta_rho_over_rho: 1e-5,
                    sigma_z: 200.0,
                })
                .z_range(5e3 + 7.0 * 200.0, 500.0)
                .build()
                .unwrap();
            solver.run_to_result(500.0)
        })
    });
}

fn bench_photon_injection(c: &mut Criterion) {
    c.bench_function("photon_injection_x1_z1e5", |b| {
        b.iter(|| {
            let mut solver = ThermalizationSolver::builder(Cosmology::default())
                .grid(GridConfig {
                    n_points: 2000,
                    ..GridConfig::default()
                })
                .injection(InjectionScenario::MonochromaticPhotonInjection {
                    x_inj: 1.0,
                    z_h: 1e5,
                    sigma_z: 4000.0,
                    delta_n_over_n: 1e-5,
                    sigma_x: 0.05,
                })
                .z_range(1e5 + 7.0 * 4000.0, 1e4)
                .build()
                .unwrap();
            solver.run_to_result(1e4)
        })
    });
}

criterion_group!(
    benches,
    bench_mu_era_burst,
    bench_y_era_burst,
    bench_photon_injection
);
criterion_main!(benches);
