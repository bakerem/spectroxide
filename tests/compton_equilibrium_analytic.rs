//! Analytic amplitude anchor for the perturbative Compton-equilibrium
//! temperature response (Part II §II.2 of
//! dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md).
//!
//! The solver's electron temperature at Compton equilibrium is
//!
//!   rho_e = I4 / (4 G3),   I4 = int x^4 n(1+n) dx,   G3 = int x^3 n dx
//!
//! (`spectrum::compton_equilibrium_ratio`, called by
//! `electron_temp::ElectronTemperature::update_equilibrium`). Linearized about
//! Planck, for dn = y·Y_SZ(x) and dn = mu·M(x) the response is
//! `d rho_eq = COEFF · amplitude`. `test_compton_equilibrium_mu_distortion`
//! only checks the sign/order of magnitude; this file pins the **amplitude**
//! against coefficients computed independently by mpmath quadrature from the
//! analytic shapes (dev/scripts/compton_equilibrium_coefficients.py).
//!
//! The anchor imports nothing from spectroxide: the Y_SZ / M(x) / n_pl shapes
//! and beta_mu are hardcoded here to match the derivation script, then
//! separately cross-checked against `spectrum::{y_shape, mu_shape}` so a
//! convention drift cannot poison the coefficients silently.
//!
//! Scope vs the solver's fused perturbative path: the full-T_e mode computes
//! `Delta rho_eq = ΔI4/(4G3) − ΔG3/G3` in a fused private routine
//! (`solver.rs::update_temperatures`, sets `self.rho_eq`) with no public
//! accessor — per the plan we do not add plumbing to read it. That fused path
//! is cross-validated against the ratio path tested here by the existing
//! `test_full_te_perturbative_vs_brute_force` (heat_injection.rs:11182, 10%).

use spectroxide::spectrum::{compton_equilibrium_ratio, mu_shape, y_shape};

// --- Coefficients from dev/scripts/compton_equilibrium_coefficients.py -------
// (mpmath dps=40; sanity identities int x^2 Y_SZ = 0, int x^3 Y_SZ = 4 G3 hold
// to <1e-40). d rho_eq = COEFF_Y·y for dn=y·Y_SZ; = COEFF_MU·mu for dn=mu·M.
// Pasted verbatim from the script's 17-digit output (do not hand-truncate —
// re-pastes must stay diff-clean); the last digit exceeds f64 resolution.
#[allow(clippy::excessive_precision)]
const COEFF_Y: f64 = 5.3996232391327225;
#[allow(clippy::excessive_precision)]
const COEFF_MU: f64 = 0.45614425920673529;

// beta_mu = 3 zeta(3) / zeta(2) = 18 zeta(3)/pi^2 (cf. constants::BETA_MU).
const BETA_MU: f64 = 2.1922889082043155;

// --- Analytic shapes hardcoded (import nothing from spectroxide) -------------

fn n_pl(x: f64) -> f64 {
    1.0 / x.exp_m1()
}

fn g_bb(x: f64) -> f64 {
    // G_bb(x) = x e^x/(e^x-1)^2 = x n_pl(1+n_pl)
    let em1 = x.exp_m1();
    x * (1.0 + em1) / (em1 * em1)
}

fn ysz(x: f64) -> f64 {
    g_bb(x) * (x * (x / 2.0).cosh() / (x / 2.0).sinh() - 4.0)
}

fn mshape(x: f64) -> f64 {
    (x / BETA_MU - 1.0) * g_bb(x) / x
}

// --- Fine log grid for the ratio quadrature ----------------------------------
// x_max >= 50 for the x^4(1+2 n_pl) M integrand tail (plan §II.2 stumbling
// point); x_min small — the low-x parts of the x^3 / x^4 moments vanish there.
fn fine_grid() -> Vec<f64> {
    let (x_min, x_max, n) = (1e-3_f64, 60.0_f64, 8000usize);
    let (lo, hi) = (x_min.ln(), x_max.ln());
    (0..n)
        .map(|i| (lo + (hi - lo) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

/// d rho_eq per unit amplitude, extracted by the difference method:
/// (ratio(n_pl + amp·shape) − ratio(n_pl)) / amp. The difference cancels the
/// O(grid) error of the baseline ratio (which is ≈1, not exactly 1), isolating
/// the coefficient (CLAUDE.md Pitfall #4 — do NOT read the absolute ratio).
fn measured_coeff(grid: &[f64], shape: &dyn Fn(f64) -> f64, amp: f64) -> f64 {
    let base: Vec<f64> = grid.iter().map(|&x| n_pl(x)).collect();
    let pert: Vec<f64> = grid.iter().map(|&x| n_pl(x) + amp * shape(x)).collect();
    (compton_equilibrium_ratio(grid, &pert) - compton_equilibrium_ratio(grid, &base)) / amp
}

#[test]
fn shape_conventions_match_library() {
    // Guard: the hardcoded anchor shapes must match spectrum.rs pointwise, else
    // a convention mismatch (beta_mu, G_bb form) would silently bias COEFF_*.
    for &x in &[0.3, 0.8, 1.5, 2.19, 3.0, 5.0, 8.0, 12.0] {
        let ry = (ysz(x) - y_shape(x)).abs() / ysz(x).abs().max(1e-6);
        let rm = (mshape(x) - mu_shape(x)).abs() / mshape(x).abs().max(1e-6);
        assert!(ry < 1e-12, "Y_SZ convention mismatch at x={x}: {ry:.3e}");
        assert!(rm < 1e-12, "M(x) convention mismatch at x={x}: {rm:.3e}");
    }
}

#[test]
fn compton_equilibrium_coeff_y() {
    let grid = fine_grid();
    for &amp in &[1e-6, 1e-5, 1e-4] {
        let c = measured_coeff(&grid, &ysz, amp);
        let rel = (c - COEFF_Y).abs() / COEFF_Y;
        eprintln!("II.2|Y: amp={amp:.0e} measured={c:.8} anchor={COEFF_Y:.8} rel={rel:.3e}");
        // Tolerance = grid quadrature + O(amp) higher-order; far below the
        // signature of a missing/extra term in ΔI4−4ΔG3.
        assert!(
            rel < 3e-3,
            "COEFF_Y mismatch at amp={amp:.0e}: measured {c:.8}, rel {rel:.3e}"
        );
    }
}

#[test]
fn compton_equilibrium_coeff_mu() {
    let grid = fine_grid();
    for &amp in &[1e-6, 1e-5, 1e-4] {
        let c = measured_coeff(&grid, &mshape, amp);
        let rel = (c - COEFF_MU).abs() / COEFF_MU;
        eprintln!("II.2|mu: amp={amp:.0e} measured={c:.8} anchor={COEFF_MU:.8} rel={rel:.3e}");
        assert!(
            rel < 3e-3,
            "COEFF_MU mismatch at amp={amp:.0e}: measured {c:.8}, rel {rel:.3e}"
        );
    }
}

#[test]
fn compton_equilibrium_linearity() {
    // The extracted coefficient must be amplitude-independent to O(amp) (the
    // response is linear; the Δn² piece is O(amp²)).
    let grid = fine_grid();
    let cy: Vec<f64> = [1e-6, 1e-5, 1e-4]
        .iter()
        .map(|&a| measured_coeff(&grid, &ysz, a))
        .collect();
    let spread = (cy.iter().cloned().fold(f64::MIN, f64::max)
        - cy.iter().cloned().fold(f64::MAX, f64::min))
        / COEFF_Y;
    eprintln!("II.2|Y linearity spread over amp∈[1e-6,1e-4] = {spread:.3e}");
    assert!(
        spread < 1e-3,
        "COEFF_Y not amplitude-independent: spread {spread:.3e}"
    );
}
