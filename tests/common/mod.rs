//! Shared setup for the integration tests.
//!
//! Only setup and bookkeeping live here. Hand-written physics oracles (shape
//! functions, CODATA literals) stay in each test file, so that a test's
//! target never shares code with the solver or with another test's target.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use spectroxide::constants::*;
use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::GridConfig;
use spectroxide::solver::{SolverConfig, SolverSnapshot, ThermalizationSolver};

/// NaN-safe maximum: asserts all elements are finite before folding.
/// Panics with a descriptive message if any NaN/Inf is found.
pub fn assert_finite_max(iter: impl Iterator<Item = f64>) -> f64 {
    iter.inspect(|&x| assert!(x.is_finite(), "NaN/Inf detected in assert_finite_max"))
        .fold(0.0, f64::max)
}

pub fn fast_grid() -> GridConfig {
    GridConfig {
        n_points: 1000,
        ..GridConfig::default()
    }
}

/// Helper: Gaussian heating profile dQ/dz for a delta-like injection.
pub fn gaussian_heating(z: f64, z_h: f64, sigma_z: f64, drho: f64) -> f64 {
    drho * (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
        / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt()
}

/// Injected Δρ/ρ between `z_lo` and `z_hi`, integrated directly from the
/// scenario's heating rate (the input, not solver output) on a log-z grid.
pub fn injected_drho(scenario: &InjectionScenario, cosmo: &Cosmology, z_lo: f64, z_hi: f64) -> f64 {
    let n = 20_000;
    let (l0, l1) = ((1.0 + z_lo).ln(), (1.0 + z_hi).ln());
    let h = (l1 - l0) / n as f64;
    let f = |l: f64| {
        let z = l.exp() - 1.0;
        -scenario.heating_rate_per_redshift(z, cosmo) * (1.0 + z)
    };
    (0..n)
        .map(|k| 0.5 * h * (f(l0 + k as f64 * h) + f(l0 + (k + 1) as f64 * h)))
        .sum()
}

/// Helper: the Gaussian photon-injection initial condition on a given grid.
///
/// Amplitude is normalised with `x_inj²`, so the array's exact moments are
/// ΔN/N = dn_over_n·(1 + σ²/x²) and Δρ/ρ = α_ρ x (dn_over_n)(1 + 3σ²/x²) —
/// not the σ → 0 values. Tests that need the injected energy must use those
/// (see `test_photon_injection_energy_conservation_tight`).
pub fn photon_injection_ic(x_grid: &[f64], x_inj: f64, dn_over_n: f64, sigma_x: f64) -> Vec<f64> {
    let amplitude =
        dn_over_n * G2_PLANCK / (x_inj * x_inj * sigma_x * (2.0 * std::f64::consts::PI).sqrt());
    x_grid
        .iter()
        .map(|&x| amplitude * (-(x - x_inj).powi(2) / (2.0 * sigma_x * sigma_x)).exp())
        .collect()
}

/// Helper: set up a Gaussian photon injection initial condition on a solver.
/// Returns the solver with delta_n set to a Gaussian at x_inj with given sigma
/// and total ΔN/N = dn_over_n.
///
/// Note `set_initial_delta_n` stashes the array and only installs it on the
/// next `run_with_snapshots`, so `solver.delta_n` is still zero on return;
/// call [`photon_injection_ic`] directly if the IC array itself is needed.
pub fn setup_photon_injection(
    cosmo: &Cosmology,
    grid_config: &GridConfig,
    x_inj: f64,
    dn_over_n: f64,
    sigma_x: f64,
    z_start: f64,
    z_end: f64,
) -> ThermalizationSolver {
    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config.clone());
    let initial_dn = photon_injection_ic(&solver.grid.x, x_inj, dn_over_n, sigma_x);
    solver.set_initial_delta_n(initial_dn);
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    solver
}

/// Helper: compute Δρ/ρ from a delta_n array on a frequency grid.
pub fn delta_rho_over_rho(x_grid: &[f64], delta_n: &[f64]) -> f64 {
    // Trapezoidal integration of x³ Δn dx / G₃
    let n = x_grid.len();
    let mut integral = 0.0;
    for i in 0..n - 1 {
        let dx = x_grid[i + 1] - x_grid[i];
        let f0 = x_grid[i].powi(3) * delta_n[i];
        let f1 = x_grid[i + 1].powi(3) * delta_n[i + 1];
        integral += 0.5 * (f0 + f1) * dx;
    }
    integral / G3_PLANCK
}

/// Helper: compute ΔN/N from a delta_n array on a frequency grid.
pub fn delta_n_over_n(x_grid: &[f64], delta_n: &[f64]) -> f64 {
    // Trapezoidal integration of x² Δn dx / G₂
    let n = x_grid.len();
    let mut integral = 0.0;
    for i in 0..n - 1 {
        let dx = x_grid[i + 1] - x_grid[i];
        let f0 = x_grid[i].powi(2) * delta_n[i];
        let f1 = x_grid[i + 1].powi(2) * delta_n[i + 1];
        integral += 0.5 * (f0 + f1) * dx;
    }
    integral / G2_PLANCK
}

/// One finished PDE run: the frequency grid and the snapshot at `z_end`.
pub struct Run {
    pub x: Vec<f64>,
    pub snap: SolverSnapshot,
}

/// Runs `solver` to `z_end` and keeps the grid and the final snapshot.
pub fn finish(mut solver: ThermalizationSolver, z_end: f64) -> Run {
    solver.run_with_snapshots(&[z_end]);
    Run {
        x: solver.grid.x.clone(),
        snap: solver.snapshots.last().cloned().expect("no snapshot"),
    }
}

/// Builds a `SingleBurst` solver the way most tests do: `new`, then
/// `set_injection`, then `set_config` with only `z_start` and `z_end` set.
pub fn burst_solver(
    grid: &GridConfig,
    z_h: f64,
    drho: f64,
    sigma_z: f64,
    z_start: f64,
    z_end: f64,
) -> ThermalizationSolver {
    let mut solver = ThermalizationSolver::new(Cosmology::default(), grid.clone());
    solver
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z,
        })
        .unwrap();
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    solver
}

/// Computes `run` once per test binary for each `key`; later callers with the
/// same key get the stored result. The solver is deterministic, so the stored
/// run equals a fresh one bit for bit. Keys must spell out every input.
pub fn memo(key: String, run: impl FnOnce() -> Run) -> &'static Run {
    type Cells = Mutex<HashMap<String, &'static OnceLock<Run>>>;
    static CELLS: OnceLock<Cells> = OnceLock::new();
    let cell = *CELLS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .entry(key)
        .or_insert_with(|| Box::leak(Box::new(OnceLock::new())));
    cell.get_or_init(run)
}

/// The standard burst: default grid, σ_z = 0.01 z_h, from 1.5 z_h to z = 500.
pub fn standard_burst(z_h: f64, drho: f64) -> &'static Run {
    memo(format!("burst {z_h:?} {drho:?}"), || {
        let solver = burst_solver(
            &GridConfig::default(),
            z_h,
            drho,
            0.01 * z_h,
            1.5 * z_h,
            500.0,
        );
        finish(solver, 500.0)
    })
}

/// A Gaussian photon-injection run built by [`setup_photon_injection`] with
/// the default cosmology.
pub fn photon_run(
    grid: &GridConfig,
    x_inj: f64,
    dn_over_n: f64,
    sigma_x: f64,
    z_start: f64,
    z_end: f64,
) -> &'static Run {
    let key = format!("photon {grid:?} {x_inj:?} {dn_over_n:?} {sigma_x:?} {z_start:?} {z_end:?}");
    memo(key, || {
        let cosmo = Cosmology::default();
        let solver =
            setup_photon_injection(&cosmo, grid, x_inj, dn_over_n, sigma_x, z_start, z_end);
        finish(solver, z_end)
    })
}

/// A run with no injection and no initial Δn, used as the baseline that
/// photon-injection tests subtract.
pub fn baseline_run(grid: &GridConfig, z_start: f64, z_end: f64) -> &'static Run {
    memo(format!("baseline {grid:?} {z_start:?} {z_end:?}"), || {
        let mut solver = ThermalizationSolver::new(Cosmology::default(), grid.clone());
        solver.set_config(SolverConfig {
            z_start,
            z_end,
            ..SolverConfig::default()
        });
        finish(solver, z_end)
    })
}
