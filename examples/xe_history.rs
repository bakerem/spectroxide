//! X_e and T_e histories of one tabulated-heating run, for comparisons against
//! other recombination codes (`dev/notebooks/xe_darkhistory_comparison.ipynb`).
//!
//! The CLI reports only the final state, so a history from it takes one run per
//! redshift. This runs once with snapshots every `dz_snap` in z, which also caps
//! the step size, and prints `z,x_e,rho_e` rows for the snapshots in `z_out`.
//!
//! Usage:
//!   `cargo run --release --example xe_history -- <heating.csv> <z_start> <z_out>
//!    <dz_snap> [--fixed-ionization]`
//!
//! `z_out` is a comma-separated list of output redshifts. `dz_snap` = 0 adds no
//! snapshots beyond `z_out`, so the solver takes its default steps. The run uses
//! the `planck2018` cosmology and the production grid.

use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::load_heating_table;
use spectroxide::solver::ThermalizationSolver;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!(
            "usage: xe_history <heating.csv> <z_start> <z_out> <dz_snap> [--fixed-ionization]"
        );
        std::process::exit(2);
    }
    let injection = load_heating_table(&args[0]).unwrap_or_else(|e| panic!("{e}"));
    let z_start: f64 = args[1].parse().expect("z_start");
    let z_out: Vec<f64> = args[2]
        .split(',')
        .map(|s| s.trim().parse().expect("z_out"))
        .collect();
    let dz_snap: f64 = args[3].parse().expect("dz_snap");
    let fixed = args.iter().any(|a| a == "--fixed-ionization");
    let z_end = z_out.iter().cloned().fold(f64::INFINITY, f64::min);

    let mut b = ThermalizationSolver::builder(Cosmology::planck2018())
        .injection(injection)
        .z_range(z_start, z_end);
    if fixed {
        b = b.fixed_ionization();
    }
    let mut solver = b.build().unwrap_or_else(|e| panic!("{e}"));

    let mut zs = z_out.clone();
    if dz_snap > 0.0 {
        let mut z = z_start - dz_snap;
        while z > z_end {
            zs.push(z);
            z -= dz_snap;
        }
    }
    let snaps = solver.run_with_snapshots(&zs).to_vec();

    println!("z,x_e,rho_e");
    for z in &z_out {
        let s = snaps
            .iter()
            .min_by(|a, b| (a.z - z).abs().total_cmp(&(b.z - z).abs()))
            .expect("no snapshots");
        assert!(
            (s.z - z).abs() < 1e-6 * z,
            "no snapshot at z = {z} (nearest {})",
            s.z
        );
        println!("{},{:.15e},{:.15e}", z, s.x_e, s.rho_e);
    }
}
