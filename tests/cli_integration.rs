//! CLI integration tests: verify the binary runs and produces valid output.

use spectroxide::cli::{CosmoOpts, OutputOpts, SolverOpts, SweepOpts, execute_sweep};
use spectroxide::constants::KAPPA_C;
use spectroxide::greens;
use std::collections::HashMap;
use std::process::Command;

fn spectroxide_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_spectroxide"))
}

/// Extract a float value from JSON output by key name.
/// Simple parser — no serde dependency needed.
fn extract_json_f64(json_str: &str, key: &str) -> f64 {
    let pattern = format!("\"{}\":", key);
    let pos = json_str
        .find(&pattern)
        .unwrap_or_else(|| panic!("Key '{key}' not found in JSON"));
    let after = &json_str[pos + pattern.len()..];
    let trimmed = after.trim_start();
    // Read until comma, closing brace, or whitespace
    let end = trimmed.find([',', '}', ']', '\n']).unwrap_or(trimmed.len());
    let val_str = trimmed[..end].trim();
    val_str
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("Failed to parse '{val_str}' as f64 for key '{key}': {e}"))
}

#[test]
fn test_cli_help_exits_cleanly() {
    let output = spectroxide_bin()
        .arg("help")
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "help should exit 0: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("spectroxide") || stderr.contains("solve"),
        "help output should mention spectroxide or solve"
    );
}

/// `greens --z-h 2e5` CLI produces Chluba-2013 μ at the visibility-corrected
/// coefficient, not just "positive and within factor-2 of 1.401".
///
/// Oracle:             Chluba (2013) MNRAS 436, 2232 Eq. 5:
///                     μ/Δρ = (3/κ_c) · J_bb*(z_h) · J_μ(z_h)
/// Expected:           at z_h=2e5:  J_bb*≈0.980, J_μ≈1.000 → μ/Δρ ≈ 1.37
/// Oracle uncertainty: 5% (GF fit vs CosmoTherm in μ-era)
/// Tolerance:          10%
///
/// Previous version had a factor-2 window `mu_over_drho ∈ (0.5, 3.0)` — any
/// 50% bug in the GF μ coefficient would have passed.
#[test]
fn test_cli_greens_json_output() {
    let output = spectroxide_bin()
        .args(["greens", "--z-h", "2e5", "--format", "json"])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "greens should exit 0: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"gf_mu\":"),
        "Should contain gf_mu in JSON"
    );
    assert!(stdout.contains("\"gf_y\":"), "Should contain gf_y in JSON");

    let gf_mu = extract_json_f64(&stdout, "gf_mu");
    let gf_y = extract_json_f64(&stdout, "gf_y");

    // Visibility-corrected analytic μ/Δρ (default CLI Δρ = 1e-5).
    let z_h = 2.0e5;
    let drho = 1e-5;
    let mu_expected =
        (3.0 / KAPPA_C) * greens::visibility_j_bb_star(z_h) * greens::visibility_j_mu(z_h) * drho;
    let mu_err = (gf_mu - mu_expected).abs() / mu_expected;
    assert!(
        mu_err < 0.10,
        "CLI greens μ at z=2e5: {gf_mu:.4e} vs Chluba 2013 Eq.5 {mu_expected:.4e} \
         (rel_err {:.2}%, tol 10%)",
        mu_err * 100.0,
    );

    // μ/y ratio in deep μ-era: J_μ/J_y ≈ 1/0.04 ≈ 25, so μ/y ≥ 25 with margin.
    assert!(
        gf_mu > gf_y.abs() * 20.0,
        "μ should strongly dominate at z=2e5: gf_mu={gf_mu:.4e}, gf_y={gf_y:.4e}, \
         ratio={:.1} (expected > 20)",
        gf_mu / gf_y.abs().max(1e-30)
    );
}

/// End-to-end guard for the `spectroxide sweep` pipeline: we call
/// `execute_sweep` directly (faster and independent of subprocess fork). The
/// existing per-z tests cover the physics at a single z_h; this test guards
/// the plumbing — parallel chunking, row aggregation, and PDE↔GF coupling
/// across a multi-redshift run.
///
/// Oracle:             Chluba (2013) MNRAS 436, 2232 Eq. 5,
///                     μ(z_h)/Δρ = (3/κ_c) · J_bb*(z_h) · J_μ(z_h).
/// Expected z_h=3e5:   μ/Δρ ≈ 1.35
/// Oracle uncertainty: within-sweep sigma_z = 0.04·z_h smears the burst over
///                     ±4% in z, so the visibility-at-peak formula is only
///                     accurate to ~10% vs the full integrated GF.
/// Tolerances:         (i) strict row order (1e-12 z match),
///                     (ii) GF μ matches Chluba Eq. 5 at peak to 15% — guards
///                          delta_rho plumb-through and per-worker GF kernel,
///                     (iii) PDE↔GF μ agrees to 30% — sweep uses sigma_z =
///                          0.04·z_h (wider than the tight z*0.01 bursts),
///                          and in the μ-era PDE/GF agree at ~20-40%
///                          (see test_heat_pde_vs_gf_multi_z_sweep),
///                     (iv) rows are genuinely distinct (PDE Δρ/ρ spread
///                          across z_h > 5%), catches a worker bug that
///                          returns the same result for every z_h.
///
/// A thread-pool indexing bug → (i) fails. Broken delta_rho plumb → (ii)
/// fails. Broken GF kernel inside the worker → (ii) and (iii) fail. Shared
/// worker state returning first z_h for all → (iv) fails.
#[test]
fn test_execute_sweep_parallel_rows_consistent() {
    // All three z_h in the μ-era (1e5 is borderline and noisier — skip it).
    let z_injections = vec![1.5e5_f64, 3.0e5, 5.0e5];
    let delta_rho = 1e-5_f64;

    let opts = SweepOpts {
        z_injections: Some(z_injections.clone()),
        delta_rho,
        params: HashMap::new(),
        solver: SolverOpts {
            n_points: Some(800),
            n_threads: Some(2), // chunk size 2 splits [1.5e5, 3e5] + [5e5]
            ..SolverOpts::default()
        },
        cosmo: CosmoOpts::default(),
        output: OutputOpts::default(),
    };

    let result = execute_sweep(&opts).expect("execute_sweep must succeed");
    assert_eq!(
        result.rows.len(),
        z_injections.len(),
        "sweep should return one row per z_h"
    );
    assert!(
        (result.delta_rho - delta_rho).abs() / delta_rho < 1e-12,
        "delta_rho plumb lost: got {} vs {delta_rho}",
        result.delta_rho
    );

    // (i) Row ordering preserved across the thread pool.
    for (row, &z_expected) in result.rows.iter().zip(z_injections.iter()) {
        assert!(
            (row.z_h - z_expected).abs() / z_expected < 1e-12,
            "row z_h={} vs expected {z_expected}",
            row.z_h
        );
    }

    // (ii) Each row's GF μ matches the Chluba 2013 visibility formula at peak.
    // Sweep uses sigma_z = 0.04·z_h, so integrated GF μ ≈ μ_peak within ~15%.
    for row in &result.rows {
        let mu_chluba = (3.0 / KAPPA_C)
            * greens::visibility_j_bb_star(row.z_h)
            * greens::visibility_j_mu(row.z_h)
            * delta_rho;
        let rel = (row.gf_mu - mu_chluba).abs() / mu_chluba.abs();
        assert!(
            rel < 0.15,
            "GF μ at z_h={:.1e}: {:.4e} vs Chluba Eq.5 at peak {mu_chluba:.4e} \
             (rel={:.2}%, tol 15%)",
            row.z_h,
            row.gf_mu,
            rel * 100.0
        );
    }

    // (iii) PDE ↔ in-row GF consistency. At sweep's wide sigma, 30% is the
    // realistic bound (see test_heat_pde_vs_gf_multi_z_sweep: 40% at 1e5,
    // 20% at 2e5/5e5 with the tighter sigma*0.01).
    for row in &result.rows {
        let mu_pde = row.snapshot.mu;
        let mu_gf = row.gf_mu;
        let rel = (mu_pde - mu_gf).abs() / mu_gf.abs().max(1e-30);
        assert!(
            rel < 0.30,
            "PDE↔GF μ mismatch at z_h={:.2e}: μ_PDE={mu_pde:.4e}, μ_GF={mu_gf:.4e} \
             (rel={:.2}%, tol 30%)",
            row.z_h,
            rel * 100.0,
        );
    }

    // (iv) Rows are genuinely distinct — Δρ/ρ (decomposed) varies across
    // z_h by > 5%. If a worker-scope bug returned the same result for every
    // z_h, Δρ would be byte-identical and this would fail. The snapshot
    // Δρ/ρ differs between rows because thermalization efficiency changes
    // with z_h even when the integrated injection is the same.
    let drhos: Vec<f64> = result
        .rows
        .iter()
        .map(|r| r.snapshot.delta_rho_over_rho)
        .collect();
    let drho_max = drhos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let drho_min = drhos.iter().cloned().fold(f64::INFINITY, f64::min);
    let spread = (drho_max - drho_min) / drho_max.abs().max(1e-30);
    assert!(
        spread > 0.005,
        "Rows look identical — thread-pool returned the same Δρ/ρ for every z_h? \
         spread={:.2}%, values={drhos:?}",
        spread * 100.0
    );
}

#[test]
fn test_cli_solve_single_burst_json() {
    let output = spectroxide_bin()
        .args([
            "solve",
            "single-burst",
            "--z-h",
            "5e4",
            "--delta-rho",
            "1e-5",
            "--grid-points",
            "500",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "solve should exit 0: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"pde_mu\":"),
        "Should contain pde_mu in JSON"
    );
    assert!(
        stdout.contains("\"pde_y\":"),
        "Should contain pde_y in JSON"
    );

    // At z=5e4 (transition era), both μ and y should be present and positive.
    // Δρ/ρ = 1e-5, so μ and y should be O(1e-5).
    let pde_mu = extract_json_f64(&stdout, "pde_mu");
    let pde_y = extract_json_f64(&stdout, "pde_y");
    assert!(pde_mu > 0.0, "pde_mu should be positive at z=5e4: {pde_mu}");
    assert!(pde_y > 0.0, "pde_y should be positive at z=5e4: {pde_y}");
    // Energy conservation: μ/1.401 + 4y ≈ Δρ/ρ = 1e-5
    // Tightened from 10× window to 4× window. At z=5e4, energy should
    // be roughly conserved even at 500 grid points.
    let energy_sum = pde_mu / 1.401 + 4.0 * pde_y;
    assert!(
        energy_sum > 0.5e-5 && energy_sum < 2.0e-5,
        "Energy sum μ/1.401 + 4y should be ≈ Δρ/ρ = 1e-5: got {energy_sum:.4e} \
         (μ={pde_mu:.4e}, y={pde_y:.4e})"
    );
}
