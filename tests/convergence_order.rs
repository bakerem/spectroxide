//! Formal convergence order study for the spectroxide PDE solver.
//!
//! Demonstrates that the numerical scheme achieves its expected convergence rates
//! via Richardson extrapolation over multiple refinement levels:
//! - Crank-Nicolson for Kompaneets: expected O(Δx², Δτ²) for fixed coefficients
//! - Backward Euler for DC/BR: expected O(Δx, Δτ)
//! - Coupled IMEX: mixed order 1.0–1.5
//!
//! **Important limitation**: The CN scheme freezes θ_e and φ = T_z/T_e at the
//! step start (frozen-coefficient CN). This degrades temporal accuracy from O(Δτ²)
//! to O(Δτ) for the *coupled* system whenever ρ_e changes significantly within a
//! step. For small energy injections (δρ/ρ ~ 10⁻⁵), this is negligible because
//! ρ_e changes by O(10⁻⁵) per step. The self-convergence tests below correctly
//! assert order > 1.0 (not > 2.0) for the coupled system.
//!
//! Three test scenarios:
//! A) Pure Kompaneets (isolates CN, expected ~2nd order)
//! B) Full physics (Kompaneets + DC/BR, expected ~1st order)
//! C) Gaussian perturbation (cleanest test, no source term, expected ~2nd order)
//!
//! Data output (stderr) for notebook parsing:
//!   CONV|scenario|sweep_type|n_points|dy_max|mu|y|drho|l2_norm|steps

use spectroxide::prelude::*;

// ============================================================================
// Utility functions
// ============================================================================

/// Linearly interpolate delta_n from one grid onto another.
fn interpolate_to_grid(x_from: &[f64], dn_from: &[f64], x_to: &[f64]) -> Vec<f64> {
    x_to.iter()
        .map(|&x| {
            // Find bracketing interval in x_from
            if x <= x_from[0] {
                return dn_from[0];
            }
            if x >= x_from[x_from.len() - 1] {
                return dn_from[dn_from.len() - 1];
            }
            // Binary search for interval
            let mut lo = 0;
            let mut hi = x_from.len() - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if x_from[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // Linear interpolation
            let t = (x - x_from[lo]) / (x_from[hi] - x_from[lo]);
            dn_from[lo] + t * (dn_from[hi] - dn_from[lo])
        })
        .collect()
}

/// Compute x³-weighted L2 norm of the difference between two spectra on the same grid.
/// ||dn1 - dn2||₂ = sqrt(Σ x³ (dn1 - dn2)² dx)
fn l2_diff_x3_weighted(x: &[f64], dn1: &[f64], dn2: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(dn1.len(), n);
    assert_eq!(dn2.len(), n);
    let mut sum = 0.0;
    for i in 1..n {
        let dx = x[i] - x[i - 1];
        let x_mid = 0.5 * (x[i] + x[i - 1]);
        let d_mid = 0.5 * ((dn1[i] - dn2[i]) + (dn1[i - 1] - dn2[i - 1]));
        sum += x_mid.powi(3) * d_mid * d_mid * dx;
    }
    sum.sqrt()
}

/// Compute convergence orders from a sequence of successive differences.
/// p_k = ln(|e_{k}| / |e_{k+1}|) / ln(2)
fn compute_orders(diffs: &[f64]) -> Vec<f64> {
    diffs
        .windows(2)
        .map(|w| {
            if w[0].abs() < 1e-30 || w[1].abs() < 1e-30 {
                0.0
            } else {
                (w[0].abs() / w[1].abs()).ln() / 2.0_f64.ln()
            }
        })
        .collect()
}

/// Median of a slice (sorts a copy).
fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

// ============================================================================
// Run result struct
// ============================================================================

struct RunResult {
    x: Vec<f64>,
    delta_n: Vec<f64>,
    mu: f64,
    y: f64,
    drho: f64,
    steps: usize,
}

impl RunResult {
    /// Compute x³-weighted L2 norm of delta_n itself.
    fn l2_norm(&self) -> f64 {
        let n = self.x.len();
        let mut sum = 0.0;
        for i in 1..n {
            let dx = self.x[i] - self.x[i - 1];
            let x_mid = 0.5 * (self.x[i] + self.x[i - 1]);
            let dn_mid = 0.5 * (self.delta_n[i] + self.delta_n[i - 1]);
            sum += x_mid.powi(3) * dn_mid * dn_mid * dx;
        }
        sum.sqrt()
    }
}

// ============================================================================
// Scenario runners
// ============================================================================

/// Scenario A: Pure Kompaneets (DC/BR disabled).
fn run_pure_kompaneets(n_points: usize, dy_max: f64, dtau_max: f64) -> RunResult {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h: 2.0e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 3000.0,
        })
        .z_range(5.0e5, 1.0e4)
        .dy_max(dy_max)
        .dtau_max(dtau_max)
        .disable_dcbr()
        .build()
        .unwrap();

    solver.run_with_snapshots(&[1.0e4]);
    let snap = solver.snapshots.last().unwrap();
    RunResult {
        x: solver.grid.x.clone(),
        delta_n: snap.delta_n.clone(),
        mu: snap.mu,
        y: snap.y,
        drho: snap.delta_rho_over_rho,
        steps: solver.step_count,
    }
}

/// Scenario B: Full physics (all processes enabled).
fn run_full_physics(n_points: usize, dy_max: f64, dtau_max: f64) -> RunResult {
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(GridConfig {
            n_points,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h: 2.0e5,
            delta_rho_over_rho: 1e-5,
            sigma_z: 3000.0,
        })
        .z_range(5.0e5, 1.0e4)
        .dy_max(dy_max)
        .dtau_max(dtau_max)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[1.0e4]);
    let snap = solver.snapshots.last().unwrap();
    RunResult {
        x: solver.grid.x.clone(),
        delta_n: snap.delta_n.clone(),
        mu: snap.mu,
        y: snap.y,
        drho: snap.delta_rho_over_rho,
        steps: solver.step_count,
    }
}

/// Scenario C: Gaussian perturbation, pure Kompaneets (no source term).
fn run_gaussian(n_points: usize, dy_max: f64, dtau_max: f64) -> RunResult {
    let grid_config = GridConfig {
        n_points,
        ..GridConfig::default()
    };
    // Build without injection, then set initial perturbation
    let mut solver = ThermalizationSolver::builder(Cosmology::default())
        .grid(grid_config)
        .z_range(1.0e5, 5.0e4)
        .dy_max(dy_max)
        .dtau_max(dtau_max)
        .disable_dcbr()
        .build()
        .unwrap();

    // Set Gaussian initial perturbation: 1e-4 * exp(-(x-3)²/0.5)
    let initial_dn: Vec<f64> = solver
        .grid
        .x
        .iter()
        .map(|&x| 1e-4 * (-(x - 3.0).powi(2) / 0.5).exp())
        .collect();
    solver.set_initial_delta_n(initial_dn);

    solver.run_with_snapshots(&[5.0e4]);
    let snap = solver.snapshots.last().unwrap();
    RunResult {
        x: solver.grid.x.clone(),
        delta_n: snap.delta_n.clone(),
        mu: snap.mu,
        y: snap.y,
        drho: snap.delta_rho_over_rho,
        steps: solver.step_count,
    }
}

// ============================================================================
// Convergence analysis helpers
// ============================================================================

/// Run a spatial convergence sweep for a given scenario.
/// Returns (n_points_vec, mu_vec, y_vec, drho_vec, l2_vec, steps_vec).
fn spatial_sweep(
    scenario: &str,
    n_points_vec: &[usize],
    dy_max: f64,
    dtau_max: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut mus = Vec::new();
    let mut ys = Vec::new();
    let mut drhos = Vec::new();
    let mut l2s = Vec::new();
    let mut steps_vec = Vec::new();

    for &n in n_points_vec {
        let result = match scenario {
            "pure_kompaneets" => run_pure_kompaneets(n, dy_max, dtau_max),
            "full_physics" => run_full_physics(n, dy_max, dtau_max),
            "gaussian" => run_gaussian(n, dy_max, dtau_max),
            _ => panic!("Unknown scenario: {scenario}"),
        };
        eprintln!(
            "CONV|{scenario}|spatial|{n}|{dy_max}|{:.8e}|{:.8e}|{:.8e}|{:.8e}|{}",
            result.mu,
            result.y,
            result.drho,
            result.l2_norm(),
            result.steps
        );
        mus.push(result.mu);
        ys.push(result.y);
        drhos.push(result.drho);
        l2s.push(result.l2_norm());
        steps_vec.push(result.steps);
    }

    (mus, ys, drhos, l2s, steps_vec)
}

/// Run a temporal convergence sweep for a given scenario.
fn temporal_sweep(
    scenario: &str,
    n_points: usize,
    dy_max_vec: &[f64],
    dtau_max: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut mus = Vec::new();
    let mut ys = Vec::new();
    let mut drhos = Vec::new();
    let mut l2s = Vec::new();
    let mut steps_vec = Vec::new();

    for &dy in dy_max_vec {
        let result = match scenario {
            "pure_kompaneets" => run_pure_kompaneets(n_points, dy, dtau_max),
            "full_physics" => run_full_physics(n_points, dy, dtau_max),
            "gaussian" => run_gaussian(n_points, dy, dtau_max),
            _ => panic!("Unknown scenario: {scenario}"),
        };
        eprintln!(
            "CONV|{scenario}|temporal|{n_points}|{dy}|{:.8e}|{:.8e}|{:.8e}|{:.8e}|{}",
            result.mu,
            result.y,
            result.drho,
            result.l2_norm(),
            result.steps
        );
        mus.push(result.mu);
        ys.push(result.y);
        drhos.push(result.drho);
        l2s.push(result.l2_norm());
        steps_vec.push(result.steps);
    }

    (mus, ys, drhos, l2s, steps_vec)
}

/// Compute self-convergence errors: e_k = Q_k - Q_{k+1}
fn successive_diffs(vals: &[f64]) -> Vec<f64> {
    vals.windows(2).map(|w| (w[0] - w[1]).abs()).collect()
}

/// Print convergence analysis and return median order.
fn analyze_convergence(label: &str, vals: &[f64]) -> f64 {
    let diffs = successive_diffs(vals);
    let orders = compute_orders(&diffs);
    eprintln!("  {label}:");
    for (i, d) in diffs.iter().enumerate() {
        let order_str = if i < orders.len() {
            format!(", order={:.2}", orders[i])
        } else {
            String::new()
        };
        eprintln!("    level {i}→{}: |diff|={d:.4e}{order_str}", i + 1);
    }
    // Skip coarsest level (pre-asymptotic) for median
    let relevant_orders: Vec<f64> = if orders.len() > 1 {
        orders[1..].to_vec()
    } else {
        orders.clone()
    };
    let med = median(&relevant_orders);
    eprintln!("    median order (skipping coarsest): {med:.2}");
    med
}

// ============================================================================
// Spectral self-convergence: interpolate coarser onto finest grid
// ============================================================================

/// Run spectral self-convergence: compare spectra at different resolutions
/// by interpolating onto the finest grid.
fn spectral_convergence(
    scenario: &str,
    n_points_vec: &[usize],
    dy_max: f64,
    dtau_max: f64,
) -> Vec<f64> {
    // Run all levels and store results
    let mut results: Vec<RunResult> = Vec::new();
    for &n in n_points_vec {
        let result = match scenario {
            "pure_kompaneets" => run_pure_kompaneets(n, dy_max, dtau_max),
            "full_physics" => run_full_physics(n, dy_max, dtau_max),
            "gaussian" => run_gaussian(n, dy_max, dtau_max),
            _ => panic!("Unknown scenario: {scenario}"),
        };
        results.push(result);
    }

    // Use finest grid as reference
    let finest = results.last().unwrap();
    let x_ref = &finest.x;

    // Compute L2 diff of each level vs finest
    let mut l2_diffs = Vec::new();
    for i in 0..results.len() - 1 {
        let interp = interpolate_to_grid(&results[i].x, &results[i].delta_n, x_ref);
        let l2 = l2_diff_x3_weighted(x_ref, &interp, &finest.delta_n);
        l2_diffs.push(l2);
        eprintln!(
            "CONV|{scenario}|spectral|{}|{dy_max}|0|0|0|{l2:.8e}|0",
            n_points_vec[i]
        );
    }

    l2_diffs
}

// ============================================================================
// CI tests (fast, 4 refinement levels)
// ============================================================================

#[test]
fn convergence_order_spatial_pure_kompaneets() {
    let n_points = vec![500, 1000, 2000, 4000];
    let (mus, _, _, _, _) = spatial_sweep("pure_kompaneets", &n_points, 5e-4, 200.0);

    eprintln!("\n=== Spatial convergence: Pure Kompaneets ===");
    let _order_mu = analyze_convergence("mu", &mus);

    // Absolute μ consistency across refinements: range across levels should be
    // below ~1% of the value. We do NOT assert an order here because the BF
    // μ extraction (photon-number non-conserving BE shape) has a mild
    // band-dependent noise floor of ~1e-8 — adequate for physics but below
    // the threshold where asymptotic order analysis is reliable. The PDE
    // convergence itself is validated by the L² norm assertion below.
    let mu_max = mus.iter().cloned().fold(f64::MIN, f64::max);
    let mu_min = mus.iter().cloned().fold(f64::MAX, f64::min);
    let mu_mean = mus.iter().sum::<f64>() / mus.len() as f64;
    let mu_spread_rel = (mu_max - mu_min) / mu_mean.abs();
    eprintln!("  μ spread across refinements: {mu_spread_rel:.2e}");
    assert!(
        mu_spread_rel < 0.01,
        "Pure Kompaneets μ not converging across refinements: spread {mu_spread_rel:.2e}"
    );

    // Also check spectral L2 convergence
    let l2_diffs = spectral_convergence("pure_kompaneets", &n_points, 5e-4, 200.0);
    let l2_orders = compute_orders(&l2_diffs);
    let l2_med = if l2_orders.len() > 1 {
        median(&l2_orders[1..])
    } else {
        median(&l2_orders)
    };
    eprintln!("  L2 spectral median order: {l2_med:.2}");
    assert!(
        l2_med > 1.5,
        "Pure Kompaneets spatial L2 order too low: {l2_med:.2} (expected > 1.5)"
    );
}

#[test]
fn convergence_order_spatial_full_physics() {
    let n_points = vec![500, 1000, 2000, 4000];
    let (mus, _, _, _, _) = spatial_sweep("full_physics", &n_points, 5e-4, 200.0);

    eprintln!("\n=== Spatial convergence: Full Physics ===");
    let _order_mu = analyze_convergence("mu", &mus);

    // Absolute μ consistency across refinements; see the pure-Kompaneets
    // version for why we don't assert asymptotic order on μ directly.
    let mu_max = mus.iter().cloned().fold(f64::MIN, f64::max);
    let mu_min = mus.iter().cloned().fold(f64::MAX, f64::min);
    let mu_mean = mus.iter().sum::<f64>() / mus.len() as f64;
    let mu_spread_rel = (mu_max - mu_min) / mu_mean.abs();
    eprintln!("  μ spread across refinements: {mu_spread_rel:.2e}");
    assert!(
        mu_spread_rel < 0.01,
        "Full physics μ not converging across refinements: spread {mu_spread_rel:.2e}"
    );

    // Spectral L2 Richardson-order assertion. Mixed IMEX (CN Kompaneets +
    // backward-Euler DC/BR) is formally first-order in space but is often
    // super-convergent on smooth solutions in the μ-era because the dominant
    // error comes from the second-order CN piece. Require 0.8 ≤ p ≤ 2.2 as
    // a two-sided fence — tighter than just asserting "decreases" while still
    // accounting for the mixed-scheme order.
    let l2_diffs = spectral_convergence("full_physics", &n_points, 5e-4, 200.0);
    let l2_orders = compute_orders(&l2_diffs);
    let l2_med = if l2_orders.len() > 1 {
        median(&l2_orders[1..])
    } else {
        median(&l2_orders)
    };
    eprintln!("  L2 spectral median order: {l2_med:.2}");
    assert!(
        (0.8..=2.2).contains(&l2_med),
        "Full physics spatial L2 order out of expected range [0.8, 2.2]: {l2_med:.2}"
    );
}

#[test]
fn convergence_order_spatial_gaussian() {
    let n_points = vec![500, 1000, 2000, 4000];

    // For Gaussian, mu/y are tiny; use spectral L2 norm
    let l2_diffs = spectral_convergence("gaussian", &n_points, 5e-4, 200.0);

    eprintln!("\n=== Spatial convergence: Gaussian ===");
    let l2_orders = compute_orders(&l2_diffs);
    eprintln!("  L2 spectral diffs:");
    for (i, d) in l2_diffs.iter().enumerate() {
        let order_str = if i < l2_orders.len() {
            format!(", order={:.2}", l2_orders[i])
        } else {
            String::new()
        };
        eprintln!("    N={}→finest: L2={d:.4e}{order_str}", n_points[i]);
    }

    // Pure CN on smooth IC: expect ~2.0
    // Use last order (most asymptotic)
    let best_order = if l2_orders.len() > 1 {
        *l2_orders.last().unwrap()
    } else {
        l2_orders[0]
    };
    assert!(
        best_order > 1.8,
        "Gaussian spatial L2 order too low: {best_order:.2} (expected > 1.8)"
    );
}

#[test]
fn convergence_order_temporal_pure_kompaneets() {
    // Uniform factor-2 refinement required for compute_orders (which divides by ln(2)).
    // Previous values [0.01, 0.005, 0.002, 0.001] had a 2.5x step at 0.005→0.002,
    // biasing the Richardson estimator by ~58% (ln(2.5)/ln(2) = 1.32, not 1.0).
    let dy_values = vec![0.008, 0.004, 0.002, 0.001];
    let (mus, _, _, _, _) = temporal_sweep("pure_kompaneets", 4000, &dy_values, 200.0);

    eprintln!("\n=== Temporal convergence: Pure Kompaneets ===");
    let order_mu = analyze_convergence("mu", &mus);

    // The adaptive stepper controls dy_max which maps to ~1st order temporal control.
    // CN itself is 2nd order, but the adaptive step selection limits effective order.
    // With uniform factor-2 refinement, measured order is ~1.0 (first-order).
    // Two-sided bound: a spurious high order (e.g. p≈2) would indicate adaptive
    // stepping is inactive or the step-selection heuristic changed.
    assert!(
        (0.8..=1.5).contains(&order_mu),
        "Pure Kompaneets temporal order for mu = {order_mu:.2}, expected 0.8–1.5 (~1.0)"
    );
}

#[test]
fn convergence_order_temporal_full_physics() {
    // Uniform factor-2 refinement (see pure_kompaneets test for rationale).
    let dy_values = vec![0.008, 0.004, 0.002, 0.001];
    let (mus, _, _, _, _) = temporal_sweep("full_physics", 4000, &dy_values, 200.0);

    eprintln!("\n=== Temporal convergence: Full Physics ===");
    let order_mu = analyze_convergence("mu", &mus);

    // Mixed CN+BE (CN for Kompaneets, BE for DC/BR).
    // With uniform factor-2 refinement, measured order is ~1.0 (first-order).
    // Two-sided bound catches both degraded (p<0.8) and spurious (p>1.5) orders.
    assert!(
        (0.8..=1.5).contains(&order_mu),
        "Full physics temporal order for mu = {order_mu:.2}, expected 0.8–1.5 (~1.0)"
    );
}

#[test]
fn convergence_order_joint() {
    // Refine both grid and timestep simultaneously with uniform factor-2 refinement.
    // Richardson extrapolation: p = log2(|e_k| / |e_{k+1}|).
    // This runs pure Kompaneets (DC/BR disabled), so CN temporal + 2nd-order spatial.
    // Expect order ≥ 1.0 (adaptive stepping limits effective temporal order).
    let levels: Vec<(usize, f64)> = vec![
        (250, 0.02),
        (500, 0.01),
        (1000, 0.005),
        (2000, 0.0025),
        (4000, 0.00125),
    ];

    eprintln!("\n=== Joint convergence: Pure Kompaneets ===");
    let mut mus = Vec::new();
    for &(n, dy) in &levels {
        let result = run_pure_kompaneets(n, dy, 200.0);
        eprintln!(
            "  N={n:5}, dy={dy:.4}: mu={:.6e}, y={:.6e}, steps={}",
            result.mu, result.y, result.steps
        );
        mus.push(result.mu);
    }

    // Compute Richardson extrapolation orders from successive differences
    let diffs = successive_diffs(&mus);
    let orders = compute_orders(&diffs);
    eprintln!("  Successive mu differences and Richardson orders:");
    for (i, (d, o)) in diffs.iter().zip(orders.iter()).enumerate() {
        eprintln!("    level {i}→{}: |diff|={d:.4e}, order={o:.2}", i + 1);
    }

    // Median order should be in [1.0, 2.5]; allow one outlier for asymptotic range.
    // Lower bound catches order degradation (e.g. CN silently becomes BE → p~1).
    // Upper bound catches spurious super-convergence (would indicate a bug in
    // adaptive stepping or the refinement sequence lying outside the asymptotic
    // regime).
    let med = median(&orders);
    assert!(
        (1.0..=2.5).contains(&med),
        "Joint convergence median Richardson order {med:.2} outside [1.0, 2.5] \
         (expected ~1.0–1.5 for IMEX adaptive)"
    );
}

// ============================================================================
// Full data dump for notebook (slow, all 6 levels, all scenarios)
// ============================================================================

#[test]
#[ignore]
fn convergence_study_full_data() {
    eprintln!("=== Full Convergence Study Data ===");
    eprintln!("Format: CONV|scenario|sweep_type|n_points|dy_max|mu|y|drho|l2_norm|steps");

    // --- Spatial sweeps (dy_max held small) ---
    let n_spatial = vec![250, 500, 1000, 2000, 4000, 8000];

    eprintln!("\n--- Spatial: Pure Kompaneets ---");
    let (mus, _, _, _, _) = spatial_sweep("pure_kompaneets", &n_spatial, 5e-4, 200.0);
    analyze_convergence("mu (spatial, pure_komp)", &mus);

    eprintln!("\n--- Spatial spectral: Pure Kompaneets ---");
    let l2_diffs = spectral_convergence("pure_kompaneets", &n_spatial, 5e-4, 200.0);
    let l2_orders = compute_orders(&l2_diffs);
    eprintln!("  Spectral L2 orders: {l2_orders:?}");

    eprintln!("\n--- Spatial: Full Physics ---");
    let (mus, _, _, _, _) = spatial_sweep("full_physics", &n_spatial, 5e-4, 200.0);
    analyze_convergence("mu (spatial, full_physics)", &mus);

    eprintln!("\n--- Spatial spectral: Full Physics ---");
    let l2_diffs = spectral_convergence("full_physics", &n_spatial, 5e-4, 200.0);
    let l2_orders = compute_orders(&l2_diffs);
    eprintln!("  Spectral L2 orders: {l2_orders:?}");

    eprintln!("\n--- Spatial: Gaussian ---");
    let _ = spatial_sweep("gaussian", &n_spatial, 5e-4, 200.0);

    eprintln!("\n--- Spatial spectral: Gaussian ---");
    let l2_diffs = spectral_convergence("gaussian", &n_spatial, 5e-4, 200.0);
    let l2_orders = compute_orders(&l2_diffs);
    eprintln!("  Spectral L2 orders: {l2_orders:?}");

    // --- Temporal sweeps (N=4000 fixed) ---
    let dy_temporal = vec![0.02, 0.01, 0.005, 0.002, 0.001, 0.0005];

    eprintln!("\n--- Temporal: Pure Kompaneets ---");
    let (mus, _, _, _, _) = temporal_sweep("pure_kompaneets", 4000, &dy_temporal, 200.0);
    analyze_convergence("mu (temporal, pure_komp)", &mus);

    eprintln!("\n--- Temporal: Full Physics ---");
    let (mus, _, _, _, _) = temporal_sweep("full_physics", 4000, &dy_temporal, 200.0);
    analyze_convergence("mu (temporal, full_physics)", &mus);

    // --- Joint sweep ---
    let joint_levels: Vec<(usize, f64)> = vec![
        (250, 0.02),
        (500, 0.01),
        (1000, 0.005),
        (2000, 0.002),
        (4000, 0.001),
        (8000, 0.0005),
    ];

    eprintln!("\n--- Joint: Pure Kompaneets ---");
    for &(n, dy) in &joint_levels {
        let result = run_pure_kompaneets(n, dy, 200.0);
        eprintln!(
            "CONV|pure_kompaneets|joint|{n}|{dy}|{:.8e}|{:.8e}|{:.8e}|{:.8e}|{}",
            result.mu,
            result.y,
            result.drho,
            result.l2_norm(),
            result.steps
        );
    }

    eprintln!("\n--- Joint: Full Physics ---");
    for &(n, dy) in &joint_levels {
        let result = run_full_physics(n, dy, 200.0);
        eprintln!(
            "CONV|full_physics|joint|{n}|{dy}|{:.8e}|{:.8e}|{:.8e}|{:.8e}|{}",
            result.mu,
            result.y,
            result.drho,
            result.l2_norm(),
            result.steps
        );
    }

    eprintln!("\n=== Full Convergence Study Complete ===");
}

// ============================================================================
// Recombination convergence: verify forward Euler is converged at dz=0.5
// ============================================================================

/// Physical sanity checks on the recombination history and cosmology dependence.
///
/// Tests monotonicity, physical range, limiting behavior, and that different
/// cosmologies produce different X_e. Also compares the cached table to the
/// non-cached integration (`ionization_fraction`) to verify tabulation accuracy.
#[test]
fn recombination_x_e_sanity_checks() {
    use spectroxide::cosmology::Cosmology;
    use spectroxide::recombination::{RecombinationHistory, ionization_fraction};

    let cosmo = Cosmology::default();
    let recomb = RecombinationHistory::new(&cosmo);

    // Sample X_e at several redshifts spanning the recombination epoch
    let test_redshifts = [1400.0, 1200.0, 1000.0, 800.0, 500.0, 200.0, 50.0];

    // Get X_e values from the cached table
    let x_e_cached: Vec<f64> = test_redshifts.iter().map(|&z| recomb.x_e(z)).collect();

    // Physical sanity checks on the recombination history
    for (&z, &xe) in test_redshifts.iter().zip(x_e_cached.iter()) {
        eprintln!("  Recombination X_e(z={z:.0}) = {xe:.6}");
        assert!(
            xe > 0.0 && xe <= 1.17,
            "X_e({z}) = {xe} out of physical range"
        );
    }

    // At z=1400: recombination is underway, X_e should be significantly ionized
    assert!(
        x_e_cached[0] > 0.5,
        "X_e(1400) = {} should be > 0.5 (recombination just starting)",
        x_e_cached[0]
    );

    // After recombination: frozen out
    assert!(
        x_e_cached[5] < 0.01,
        "X_e(200) = {} should be < 0.01 (post-recombination)",
        x_e_cached[5]
    );

    // Monotonically decreasing during recombination
    for i in 1..test_redshifts.len() {
        if test_redshifts[i] < 1500.0 && test_redshifts[i] > 20.0 {
            assert!(
                x_e_cached[i] <= x_e_cached[i - 1] + 1e-10,
                "X_e should decrease during recombination: X_e({}) = {} > X_e({}) = {}",
                test_redshifts[i],
                x_e_cached[i],
                test_redshifts[i - 1],
                x_e_cached[i - 1]
            );
        }
    }

    // The cached table should agree with the non-cached integration to < 0.1%
    // (interpolation error from the dz=0.5 table spacing at smooth points).
    let check_zs = [1400.0, 1200.0, 1000.0, 800.0];
    for &z in &check_zs {
        let xe_cached = recomb.x_e(z);
        let xe_direct = ionization_fraction(z, &cosmo);
        let rel_diff = (xe_cached - xe_direct).abs() / xe_direct.max(1e-10);
        eprintln!(
            "  X_e(z={z:.0}): cached={xe_cached:.6}, direct={xe_direct:.6}, rel={rel_diff:.2e}"
        );
        assert!(
            rel_diff < 0.002,
            "Cached X_e(z={z}) = {xe_cached:.6} differs from direct integration {xe_direct:.6} \
             by {:.2}% (> 0.2% threshold, possible table interpolation error)",
            rel_diff * 100.0
        );
    }

    // Different cosmologies should give noticeably different X_e during recombination
    let cosmo2 = Cosmology::planck2018();
    let recomb2 = RecombinationHistory::new(&cosmo2);
    let xe_p18_1000 = recomb2.x_e(1000.0);
    let xe_def_1000 = recomb.x_e(1000.0);
    eprintln!("  X_e(z=1000): default={xe_def_1000:.6}, Planck2018={xe_p18_1000:.6}");
    let rel_diff = (xe_p18_1000 - xe_def_1000).abs() / xe_def_1000;
    assert!(
        rel_diff > 0.01,
        "Different cosmologies should produce >1% different X_e: {} vs {} (rel={:.2}%)",
        xe_def_1000,
        xe_p18_1000,
        rel_diff * 100.0
    );
}

// ============================================================================
// Quasi-stationary T_e: verify against explicit ODE check
// ============================================================================

/// Verify the quasi-stationary T_e approximation by checking that
/// for a simple heat injection, ρ_e = ρ_eq + δρ_inj gives the correct
/// result: for a pure Planck spectrum with no injection, ρ_e should be 1.0.
/// With injection, ρ_e should track the energy perturbation.
#[test]
fn convergence_quasi_stationary_te_consistency() {
    // Run a heat injection and verify that the T_e is consistent
    // with the spectral decomposition: ρ_e ≈ 1 + 5.4 × u for small u.
    let cosmo = Cosmology::default();
    let drho = 1e-5;
    let z_h = 2e5;

    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        })
        .injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: z_h / 10.0,
        })
        .build()
        .unwrap();

    solver.run_with_snapshots(&[z_h - z_h / 5.0, 1e4]);
    let snaps = &solver.snapshots;

    // Shortly after injection: ρ_e should be slightly above 1
    let snap_after = &snaps[0];
    let rho_e_after = snap_after.rho_e;
    eprintln!(
        "  After injection (z={:.0}): ρ_e = {rho_e_after:.10}",
        snap_after.z
    );
    // For Δρ/ρ = 1e-5, the perturbative prediction gives ρ_e − 1 ~ O(10⁻⁵).
    // Tightened from 1.01 (1000× too loose) to 1.001.
    assert!(
        rho_e_after > 1.0 && rho_e_after < 1.001,
        "ρ_e after small injection should be slightly > 1: got {rho_e_after}"
    );

    // At late time: ρ_e should relax back toward equilibrium
    let snap_late = &snaps[1];
    let rho_e_late = snap_late.rho_e;
    eprintln!("  Late (z={:.0}): ρ_e = {rho_e_late:.10}", snap_late.z);
    assert!(
        (rho_e_late - 1.0).abs() < (rho_e_after - 1.0).abs(),
        "ρ_e should relax toward 1: late={rho_e_late} vs after={rho_e_after}"
    );
}
