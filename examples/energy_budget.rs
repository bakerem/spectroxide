//! Energy-conservation budget for the PDE solver.
//!
//! Decomposes the measured Δρ/ρ deviation into its actual sources instead of
//! quoting a single number. Findings it was built to establish
//! (`dev/audit/energy_conservation_audit.md`):
//!
//!   * the x-quadrature and the analytic G₃ normalisation contribute ≲10⁻⁴ —
//!     the deviation is not a bookkeeping error;
//!   * the heat-injection deficit is the first-order-in-Δτ temporal error of
//!     the coupled T_e / DC-BR step (`dtau_max` controls it; the grid does
//!     not), and it is generated inside the injection window;
//!   * the photon-injection "1%" is mostly the finite width of the Gaussian
//!     initial condition, whose exact energy is
//!     α_ρ x₀ (ΔN/N)(1 + 3σ²/x₀²), not α_ρ x₀ (ΔN/N).
//!
//! Usage: `cargo run --release --example energy_budget [mode]`
//! with mode ∈ {all, quad, heat, photon, pb2009, figure, steps}. Default `all`;
//! `steps` (per-step localisation) and `figure` (deep-μ end at the paper
//! figure's settings) are excluded from `all` because they are slow.

use spectroxide::constants::{ALPHA_RHO, G2_PLANCK, G3_PLANCK, KAPPA_C};
use spectroxide::prelude::*;

/// Trapezoid of the product x³Δn — the quadrature used by the test suite.
fn drho_trapz(x: &[f64], dn: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..x.len() {
        let f0 = x[i - 1].powi(3) * dn[i - 1];
        let f1 = x[i].powi(3) * dn[i];
        s += 0.5 * (f0 + f1) * (x[i] - x[i - 1]);
    }
    s / G3_PLANCK
}

/// Midpoint-x × trapezoid-Δn — the quadrature inside `spectrum::weighted_integral`.
fn drho_midx(x: &[f64], dn: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..x.len() {
        let xm = 0.5 * (x[i] + x[i - 1]);
        let dm = 0.5 * (dn[i] + dn[i - 1]);
        s += xm.powi(3) * dm * (x[i] - x[i - 1]);
    }
    s / G3_PLANCK
}

fn dnn_trapz(x: &[f64], dn: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..x.len() {
        let f0 = x[i - 1].powi(2) * dn[i - 1];
        let f1 = x[i].powi(2) * dn[i];
        s += 0.5 * (f0 + f1) * (x[i] - x[i - 1]);
    }
    s / G2_PLANCK
}

struct HeatCfg {
    label: &'static str,
    n_points: usize,
    dtau_max: f64,
    dy_max: f64,
    no_dcbr: bool,
}

struct HeatOut {
    drho: f64,
    mu: f64,
    y: f64,
    steps: usize,
}

fn run_heat(cfg: &HeatCfg, z_h: f64, drho_inj: f64) -> HeatOut {
    let grid = GridConfig {
        n_points: cfg.n_points,
        ..GridConfig::default()
    };
    let mut s = ThermalizationSolver::new(Cosmology::default(), grid);
    s.disable_dcbr = cfg.no_dcbr;
    if drho_inj != 0.0 {
        s.set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho_inj,
            sigma_z: z_h * 0.01,
        })
        .unwrap();
    }
    s.set_config(SolverConfig {
        z_start: z_h * 1.5,
        z_end: 500.0,
        dtau_max: cfg.dtau_max,
        dy_max: cfg.dy_max,
        ..SolverConfig::default()
    });
    s.run_with_snapshots(&[500.0]);
    let last = s.snapshots.last().unwrap();
    HeatOut {
        drho: drho_trapz(&s.grid.x, &last.delta_n),
        mu: last.mu,
        y: last.y,
        steps: s.step_count,
    }
}

/// Exact shape moments on the solver grid: ∫x³G_bb dx = 4G₃, ∫x³M dx = (κ_c/3)G₃.
/// Bounds the quadrature + truncation contribution to every Δρ/ρ in the suite.
fn mode_quad() {
    println!("== quadrature of the exact shape moments (relative error)");
    for (label, gc) in [
        ("default  (N=2000, x∈[1e-4,50])", GridConfig::default()),
        ("production(N=4000, x∈[1e-5,60])", GridConfig::production()),
    ] {
        let g = FrequencyGrid::new(&gc);
        let gbb: Vec<f64> = g.x.iter().map(|&x| spectroxide::spectrum::g_bb(x)).collect();
        let m: Vec<f64> = g
            .x
            .iter()
            .map(|&x| spectroxide::spectrum::mu_shape(x))
            .collect();
        let k3 = KAPPA_C / 3.0;
        println!(
            "  {label}:  x³G_bb trapz={:+.3e} midx={:+.3e} | x³M trapz={:+.3e} midx={:+.3e}",
            drho_trapz(&g.x, &gbb) / 4.0 - 1.0,
            drho_midx(&g.x, &gbb) / 4.0 - 1.0,
            drho_trapz(&g.x, &m) / k3 - 1.0,
            drho_midx(&g.x, &m) / k3 - 1.0,
        );
    }
    println!();
}

/// Heat injection: which knob moves the deficit, and what it costs in μ/y.
fn mode_heat() {
    let drho_inj = 1e-5;
    let cfgs = [
        HeatCfg { label: "dtau=10 (default)", n_points: 2000, dtau_max: 10.0, dy_max: 0.02, no_dcbr: false },
        HeatCfg { label: "dtau=10, N=4000  ", n_points: 4000, dtau_max: 10.0, dy_max: 0.02, no_dcbr: false },
        HeatCfg { label: "dtau=10, dy=0.005", n_points: 2000, dtau_max: 10.0, dy_max: 0.005, no_dcbr: false },
        HeatCfg { label: "dtau=10, no DC/BR", n_points: 2000, dtau_max: 10.0, dy_max: 0.02, no_dcbr: true },
        HeatCfg { label: "dtau=2           ", n_points: 2000, dtau_max: 2.0, dy_max: 0.02, no_dcbr: false },
        HeatCfg { label: "dtau=1           ", n_points: 2000, dtau_max: 1.0, dy_max: 0.02, no_dcbr: false },
    ];
    println!("== heat injection (SingleBurst, σ_z = 0.01 z_h, Δρ/ρ = 1e-5, z_end = 500)");
    println!("   err_net subtracts an identical zero-injection run (adiabatic cooling).");
    for &z_h in &[1e4_f64, 1e5, 5e5] {
        println!("  z_h = {z_h:.0e}");
        let mut rows = Vec::new();
        for cfg in &cfgs {
            let o = run_heat(cfg, z_h, drho_inj);
            let b = run_heat(cfg, z_h, 0.0);
            println!(
                "    {}  err_raw={:+7.3}%  err_net={:+7.3}%  cool={:+.2e}  mu={:.5e} y={:.5e}  steps={:6}",
                cfg.label,
                (o.drho / drho_inj - 1.0) * 100.0,
                ((o.drho - b.drho) / drho_inj - 1.0) * 100.0,
                b.drho / drho_inj,
                o.mu,
                o.y,
                o.steps
            );
            rows.push((cfg.label, o.mu, o.y));
        }
        let (_, ref_mu, ref_y) = *rows.last().unwrap();
        for (label, mu, y) in &rows {
            println!(
                "      vs dtau=1: {}  dmu={:+7.3}%  dy={:+7.3}%",
                label,
                (mu / ref_mu - 1.0) * 100.0,
                (y / ref_y - 1.0) * 100.0
            );
        }
    }
    println!();
}

/// Photon injection: split the IC's own energy content off the conservation error.
fn mode_photon() {
    let z_h = 3.0e5;
    let dn_over_n = 1e-5;
    println!("== photon injection (Gaussian IC at z_start = 3e5, ΔN/N = 1e-5, z_end = 500)");
    println!("   target = α_ρ x₀ ΔN/N ; analytic = target × (1 + 3σ²/x₀²)");
    println!("   x₀    σ      IC/target-1  analytic/target-1   out/IC-1   out/target-1   [dtau, N]");
    for &(x_inj, dtau_max, n_points) in &[
        (1.5_f64, 10.0_f64, 2000_usize),
        (3.6, 10.0, 2000),
        (5.0, 10.0, 2000),
        (8.0, 10.0, 2000),
        (12.0, 10.0, 2000),
        (12.0, 10.0, 4000),
        (12.0, 2.0, 2000),
        (12.0, 1.0, 2000),
    ] {
        let grid = GridConfig {
            n_points,
            ..GridConfig::default()
        };
        let sigma_x = (0.05_f64 * x_inj).max(0.05);
        let amp = dn_over_n * G2_PLANCK
            / (x_inj * x_inj * sigma_x * (2.0 * std::f64::consts::PI).sqrt());
        let mut s = ThermalizationSolver::new(Cosmology::default(), grid);
        let ic: Vec<f64> = s
            .grid
            .x
            .iter()
            .map(|&x| amp * (-(x - x_inj).powi(2) / (2.0 * sigma_x * sigma_x)).exp())
            .collect();
        let drho_ic = drho_trapz(&s.grid.x, &ic);
        let dnn_ic = dnn_trapz(&s.grid.x, &ic);
        s.set_initial_delta_n(ic);
        s.set_config(SolverConfig {
            z_start: z_h,
            z_end: 500.0,
            dtau_max,
            ..SolverConfig::default()
        });
        s.run_with_snapshots(&[500.0]);
        let last = s.snapshots.last().unwrap();
        let drho_out = drho_trapz(&s.grid.x, &last.delta_n);
        let target = ALPHA_RHO * x_inj * dn_over_n;
        let r = sigma_x / x_inj;
        println!(
            "  {x_inj:5.1} {sigma_x:5.3}   {:+8.3}%      {:+8.3}%        {:+8.3}%   {:+8.3}%    [{dtau_max}, {n_points}]  ΔN/N: IC {:+.3}% (exact {:+.3}%)",
            (drho_ic / target - 1.0) * 100.0,
            3.0 * r * r * 100.0,
            (drho_out / drho_ic - 1.0) * 100.0,
            (drho_out / target - 1.0) * 100.0,
            (dnn_ic / dn_over_n - 1.0) * 100.0,
            r * r * 100.0,
        );
    }
    println!();
}

/// The scenario behind `test_pb2009_energy_conservation`: z_h = 2e5, wide
/// burst, snapshot at z = 200. Grid vs dtau_max, to show which axis matters.
fn mode_pb2009() {
    let drho = 1e-5;
    let z_h = 2e5;
    println!("== P&B-2009 benchmark scenario (z_h = 2e5, σ_z = z_h/10, z_end = 200)");
    for (label, gc) in [
        ("default   ", GridConfig::default()),
        ("production", GridConfig::production()),
    ] {
        for dtau_max in [10.0_f64, 2.0] {
            let mut s = ThermalizationSolver::new(Cosmology::default(), gc.clone());
            s.set_injection(InjectionScenario::SingleBurst {
                z_h,
                sigma_z: z_h / 10.0,
                delta_rho_over_rho: drho,
            })
            .unwrap();
            s.set_config(SolverConfig {
                z_start: 3.0e5,
                z_end: 200.0,
                dtau_max,
                ..SolverConfig::default()
            });
            let snaps = s.run_with_snapshots(&[200.0]);
            let e = snaps[0].delta_rho_over_rho;
            println!(
                "  grid={label} dtau_max={dtau_max:4}:  Δρ_out = {e:.6e}  err = {:+7.3}%  steps = {}",
                (e / drho - 1.0) * 100.0,
                s.step_count
            );
        }
    }
    println!();
}

/// The deep-μ end of the paper figure. `notebooks/paper_figures/
/// energy_conservation.ipynb` sweeps z_h up to 3e6 through the CLI `sweep`
/// path, i.e. σ_z = max(0.04 z_h, 100), z_start = z_h + 7σ_z, number-conserving,
/// dtau_max = 3, N = 8000. Here at N = 4000 to keep it affordable, with
/// dtau_max = 3 vs 1.5 to show whether the figure's high-z points are
/// converged. Uses the snapshot's own `delta_rho_over_rho` (spectral + T-shift),
/// exactly what the figure plots.
fn mode_figure() {
    let drho = 1e-5;
    println!("== paper-figure settings at the deep-μ end (N=4000, NC on)");
    for &z_h in &[5e5_f64, 1e6, 3e6] {
        for dtau_max in [3.0_f64, 1.5] {
            let sigma = (z_h * 0.04_f64).max(100.0);
            let mut s = ThermalizationSolver::new(
                Cosmology::default(),
                GridConfig {
                    n_points: 4000,
                    ..GridConfig::default()
                },
            );
            s.set_injection(InjectionScenario::SingleBurst {
                z_h,
                sigma_z: sigma,
                delta_rho_over_rho: drho,
            })
            .unwrap();
            s.set_config(SolverConfig {
                z_start: z_h + 7.0 * sigma,
                z_end: 500.0,
                dtau_max,
                ..SolverConfig::default()
            });
            let (e, mu) = {
                let snaps = s.run_with_snapshots(&[500.0]);
                (snaps[0].delta_rho_over_rho, snaps[0].mu)
            };
            println!(
                "  z_h={z_h:8.1e} dtau_max={dtau_max:4}:  Δρ_out={e:.6e}  dev={:+7.3}%  mu={mu:.5e}  steps={}",
                (e / drho - 1.0) * 100.0,
                s.step_count
            );
        }
    }
    println!();
}

/// Abramowitz & Stegun 7.1.26 (|ε| < 1.5e-7); no external crates by design.
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Localise the deficit in redshift: running photon energy vs the analytically
/// integrated burst source. Shows the deficit is generated in the injection
/// window and then partly recovered.
fn mode_steps() {
    let z_h = 1.0e5_f64;
    let sigma_z = 0.01 * z_h;
    let drho_inj = 1e-5;
    let (z_start, z_end) = (1.5 * z_h, 500.0);
    for dtau_max in [10.0_f64, 2.0] {
        let cfg = SolverConfig {
            z_start,
            z_end,
            dtau_max,
            ..SolverConfig::default()
        };
        let mut base = ThermalizationSolver::new(Cosmology::default(), GridConfig::default());
        base.number_conserving = false;
        base.set_config(cfg.clone());
        let mut s = ThermalizationSolver::new(Cosmology::default(), GridConfig::default());
        s.number_conserving = false;
        s.set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho_inj,
            sigma_z,
        })
        .unwrap();
        s.set_config(cfg);
        println!("== steps, dtau_max = {dtau_max}");
        println!("       z         E/inj       injected      cooling      residual");
        let mut n = 0usize;
        while s.z > z_end * 1.000_001 {
            s.step();
            base.step();
            n += 1;
            let e = drho_trapz(&s.grid.x, &s.delta_n) / drho_inj;
            let cool = drho_trapz(&base.grid.x, &base.delta_n) / drho_inj;
            let inj = 0.5 * (1.0 + erf((z_h - s.z) / (sigma_z * std::f64::consts::SQRT_2)));
            let in_burst = (s.z - z_h).abs() < 4.0 * sigma_z;
            if n % 400 == 0 || (in_burst && n % 40 == 0) || s.z <= z_end * 1.001 {
                println!(
                    "  {:9.3e}  {:+.6e}  {:+.6e}  {:+.2e}  {:+8.4}%",
                    s.z,
                    e,
                    inj,
                    cool,
                    (e - cool - inj) * 100.0
                );
            }
        }
        println!("  steps = {n}\n");
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match mode.as_str() {
        "quad" => mode_quad(),
        "heat" => mode_heat(),
        "photon" => mode_photon(),
        "pb2009" => mode_pb2009(),
        "figure" => mode_figure(),
        "steps" => mode_steps(),
        "all" => {
            mode_quad();
            mode_pb2009();
            mode_photon();
            mode_heat();
        }
        other => {
            eprintln!("unknown mode {other:?}; expected all|quad|heat|photon|pb2009|figure|steps");
            std::process::exit(2);
        }
    }
}
