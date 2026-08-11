//! Method of manufactured solutions (MMS) for the Kompaneets + DC/BR operator.
//!
//! Validation-audit Phase 2 (dev/PLAN_VALIDATION_AUDIT_2026-07-02.md, B3).
//! Unlike the self-convergence tests in `convergence_order.rs`, these tests
//! measure the **true error** against a known exact solution: we pick a smooth
//! Δn_m(x, τ) and inject the analytic residual source
//!
//!   S(x, τ) = ∂Δn_m/∂τ − L[Δn_m]
//!
//! through the *production* source path (`DcbrCoupling::photon_source` at the
//! kernel level; `InjectionScenario::TabulatedPhotonSource` at the solver
//! level), so the numerical solution must reproduce Δn_m exactly up to
//! discretization error. Convergence of that error at the scheme's design
//! order is the strongest available verification of the discretization
//! (anchor is analytic, not code-derived).
//!
//! Manufactured solution: Δn_m(x, τ) = a(τ)·g(x) with g a Gaussian bump well
//! inside the grid interior (so the kernel's zero-flux boundary treatment
//! contributes only an exponentially small error floor) and a(τ) oscillatory
//! (so temporal truncation error is visible).
//!
//! Operator pieces verified, with expected orders:
//! - Crank-Nicolson Kompaneets (φ = 1, includes the nonlinear Δn² term via
//!   the Newton iteration): O(Δx²) spatial, O(Δτ²) temporal.
//! - Backward-Euler DC/BR relaxation coupled into the same Newton solve:
//!   O(Δx²) spatial (pointwise term), O(Δτ) temporal.
//! - Full `ThermalizationSolver` (adaptive stepping, T_e coupling, tabulated
//!   photon source with operator splitting under `disable_dcbr`): O(Δτ)
//!   with a quantitative reproduction bound.
//!
//! Also includes the discrete photon-number ledger tests on the coupled
//! Newton path (audit finding P1-2): pure Compton conserves ∫x²Δn dx to
//! machine precision (the flux form telescopes), and with DC/BR + source
//! active the discrete number balance
//!
//!   ΔN = Σ_i w_i [dτ·em_i·(neq_i − Δn_new,i) + S_i]
//!
//! holds to Newton tolerance.

use spectroxide::cosmology::Cosmology;
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::{FrequencyGrid, GridConfig};
use spectroxide::kompaneets::{DcbrCoupling, KompaneetsWorkspace, kompaneets_step_coupled_inplace};
use spectroxide::recombination::RecombinationHistory;
use spectroxide::solver::ThermalizationSolver;

// ============================================================================
// Manufactured solution: Δn_m(x, τ) = a(τ)·g(x)
// ============================================================================

/// Gaussian profile and its first two derivatives.
fn gauss(x: f64, x0: f64, s: f64) -> (f64, f64, f64) {
    let u = (x - x0) / s;
    let g = (-0.5 * u * u).exp();
    let gp = -u / s * g;
    let gpp = (u * u - 1.0) / (s * s) * g;
    (g, gp, gpp)
}

fn planck_n(x: f64) -> f64 {
    1.0 / (x.exp() - 1.0)
}

/// Parameters of the kernel-level manufactured solution.
struct MmsCase {
    x0: f64,
    s: f64,
    a0: f64,
    omega: f64,
    theta: f64, // θ_e = θ_z (φ = 1)
}

impl MmsCase {
    fn default_case() -> Self {
        MmsCase {
            x0: 3.0,
            s: 0.6,
            a0: 1e-3,
            omega: 2.0 * std::f64::consts::PI / 50.0,
            theta: 9.2e-5, // θ_z at z ≈ 2×10⁵
        }
    }

    fn amp(&self, tau: f64) -> f64 {
        self.a0 * (1.0 + 0.5 * (self.omega * tau).sin())
    }

    fn amp_dot(&self, tau: f64) -> f64 {
        0.5 * self.a0 * self.omega * (self.omega * tau).cos()
    }

    fn exact(&self, x: f64, tau: f64) -> f64 {
        let (g, _, _) = gauss(x, self.x0, self.s);
        self.amp(tau) * g
    }

    /// Kompaneets operator applied to Δn_m at (x, τ), per unit Compton τ,
    /// with φ = 1 (the (φ−1) source term vanishes identically):
    ///
    ///   L[Δn] = (θ/x²) ∂/∂x { x⁴ [ ∂Δn/∂x + (2n_pl+1)Δn + Δn² ] }
    ///         = θ (4x·h + x²·h')
    ///
    /// with h = a g' + (2n_pl+1) a g + a² g² and
    /// h' = a g'' + (2n_pl+1) a g' − 2 n_pl(1+n_pl) a g + 2 a² g g'.
    fn kompaneets_op(&self, x: f64, tau: f64) -> f64 {
        let a = self.amp(tau);
        let (g, gp, gpp) = gauss(x, self.x0, self.s);
        let np = planck_n(x);
        let tw = 2.0 * np + 1.0;
        let h = a * gp + tw * a * g + a * a * g * g;
        let hp = a * gpp + tw * a * gp - 2.0 * np * (np + 1.0) * a * g + 2.0 * a * a * g * gp;
        self.theta * (4.0 * x * h + x * x * hp)
    }

    /// Manufactured source per unit τ: S = ∂Δn_m/∂τ − L_K[Δn_m] + em(x)·Δn_m.
    /// `em` is the DC/BR-style relaxation rate toward neq = 0 (pass 0 for
    /// pure Kompaneets).
    fn source(&self, x: f64, tau: f64, em: f64) -> f64 {
        let (g, _, _) = gauss(x, self.x0, self.s);
        self.amp_dot(tau) * g - self.kompaneets_op(x, tau) + em * self.exact(x, tau)
    }
}

// ============================================================================
// Kernel-level MMS driver
// ============================================================================

/// DC/BR-style relaxation rate for the coupled MMS case: em(x) = Λ(e^x−1)/x³,
/// the same 1/x³-divergent shape as the production DC/BR absorption rate.
fn em_rate(x: f64, lambda: f64) -> f64 {
    lambda * (x.exp() - 1.0) / (x * x * x)
}

/// March the production coupled Newton kernel with the manufactured source
/// for `n_steps` of size `dtau`, starting from Δn_m(x, 0).
/// Returns the numerical Δn at τ = n_steps·dtau.
fn run_kernel_mms(
    case: &MmsCase,
    grid: &FrequencyGrid,
    n_steps: usize,
    dtau: f64,
    lambda: f64,
) -> Vec<f64> {
    let n = grid.n;
    let mut delta_n: Vec<f64> = grid.x.iter().map(|&x| case.exact(x, 0.0)).collect();
    let mut ws = KompaneetsWorkspace::new(grid);

    let em: Vec<f64> = grid.x.iter().map(|&x| em_rate(x, lambda)).collect();
    let neq = vec![0.0; n];
    let zeros = vec![0.0; n];
    let mut src = vec![0.0; n];

    for k in 0..n_steps {
        let tau_mid = (k as f64 + 0.5) * dtau;
        for (i, &x) in grid.x.iter().enumerate() {
            src[i] = case.source(x, tau_mid, em[i]) * dtau;
        }
        let dcbr = DcbrCoupling {
            emission_rates: &em,
            n_eq_minus_n_pl: &neq,
            dem_drho_eq: &zeros,
            dneq_drho_eq: &zeros,
            photon_source: Some(&src),
            cn_dcbr: false,
        };
        let (converged, _rho, _corr) = kompaneets_step_coupled_inplace(
            grid,
            &mut delta_n,
            case.theta,
            case.theta,
            dtau,
            Some(&dcbr),
            None,
            &mut ws,
            0.0, // tightest Newton tolerance
            30,
        );
        assert!(converged, "Newton failed to converge at step {k}");
    }
    delta_n
}

/// Relative x³-weighted L2 error of `num` against the exact Δn_m(·, τ_end).
fn rel_l2_error(case: &MmsCase, grid: &FrequencyGrid, num: &[f64], tau_end: f64) -> f64 {
    let mut err2 = 0.0;
    let mut ref2 = 0.0;
    for i in 1..grid.n {
        let dx = grid.dx[i - 1];
        let xm = grid.x_half[i - 1];
        let em = 0.5
            * ((num[i] - case.exact(grid.x[i], tau_end))
                + (num[i - 1] - case.exact(grid.x[i - 1], tau_end)));
        let rm = 0.5 * (case.exact(grid.x[i], tau_end) + case.exact(grid.x[i - 1], tau_end));
        err2 += xm.powi(3) * em * em * dx;
        ref2 += xm.powi(3) * rm * rm * dx;
    }
    (err2 / ref2).sqrt()
}

fn order_between(e_coarse: f64, e_fine: f64) -> f64 {
    (e_coarse / e_fine).ln() / 2.0_f64.ln()
}

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

// ============================================================================
// MMS: pure Kompaneets (CN + Newton), expected 2nd order in space and time
// ============================================================================

const TAU_TOTAL: f64 = 50.0;

#[test]
fn mms_kernel_spatial_order_pure_kompaneets() {
    let case = MmsCase::default_case();
    // dτ fixed small so temporal error is far below the coarsest spatial error.
    let n_steps = 2048;
    let dtau = TAU_TOTAL / n_steps as f64;

    let mut errs = Vec::new();
    for &n in &[400usize, 800, 1600, 3200] {
        let grid = FrequencyGrid::log_uniform(0.2, 30.0, n);
        let num = run_kernel_mms(&case, &grid, n_steps, dtau, 0.0);
        let e = rel_l2_error(&case, &grid, &num, TAU_TOTAL);
        eprintln!("MMS|pure_kompaneets|spatial|N={n}|rel_l2={e:.6e}");
        errs.push(e);
    }
    let mut orders: Vec<f64> = errs.windows(2).map(|w| order_between(w[0], w[1])).collect();
    eprintln!("  spatial orders: {orders:?}");
    let med = median(&mut orders);
    assert!(
        (1.7..=2.4).contains(&med),
        "Pure Kompaneets MMS spatial order {med:.2} outside [1.7, 2.4]; errors {errs:?}"
    );
}

#[test]
fn mms_kernel_temporal_order_pure_kompaneets() {
    let case = MmsCase::default_case();
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, 4000);

    // True errors contain a constant spatial floor; measure the temporal
    // order from successive solution differences on the same grid (the
    // spatial component cancels exactly), and separately bound the finest
    // true error.
    let m_levels = [16usize, 32, 64, 128];
    let sols: Vec<Vec<f64>> = m_levels
        .iter()
        .map(|&m| run_kernel_mms(&case, &grid, m, TAU_TOTAL / m as f64, 0.0))
        .collect();

    let mut diffs = Vec::new();
    for w in sols.windows(2) {
        let mut d2 = 0.0;
        let mut r2 = 0.0;
        for i in 1..grid.n {
            let dx = grid.dx[i - 1];
            let xm = grid.x_half[i - 1];
            let dm = 0.5 * ((w[0][i] - w[1][i]) + (w[0][i - 1] - w[1][i - 1]));
            let rm =
                0.5 * (case.exact(grid.x[i], TAU_TOTAL) + case.exact(grid.x[i - 1], TAU_TOTAL));
            d2 += xm.powi(3) * dm * dm * dx;
            r2 += xm.powi(3) * rm * rm * dx;
        }
        diffs.push((d2 / r2).sqrt());
    }
    let mut orders: Vec<f64> = diffs
        .windows(2)
        .map(|w| order_between(w[0], w[1]))
        .collect();
    eprintln!("MMS|pure_kompaneets|temporal diffs: {diffs:?}, orders: {orders:?}");
    let med = median(&mut orders);
    assert!(
        (1.7..=2.4).contains(&med),
        "CN temporal order {med:.2} outside [1.7, 2.4]; diffs {diffs:?}"
    );

    // Finest-level true error: temporal + spatial floor together.
    let e_fine = rel_l2_error(&case, &grid, &sols[sols.len() - 1], TAU_TOTAL);
    eprintln!("  finest true rel error: {e_fine:.3e}");
    assert!(
        e_fine < 5e-3,
        "Finest pure-Kompaneets MMS true error too large: {e_fine:.3e}"
    );
}

// ============================================================================
// MMS: coupled Kompaneets + DC/BR-style relaxation (backward Euler)
// ============================================================================

/// Relaxation amplitude: em(0.2) ≈ 0.28 Λ/x³-shaped, giving dτ·em up to ~O(10)
/// at the coarsest steps near x_min — the mildly stiff regime backward Euler
/// exists for.
const LAMBDA_DCBR: f64 = 1e-2;

#[test]
fn mms_kernel_spatial_order_coupled_dcbr() {
    let case = MmsCase::default_case();
    let n_steps = 2048;
    let dtau = TAU_TOTAL / n_steps as f64;

    let mut errs = Vec::new();
    for &n in &[400usize, 800, 1600, 3200] {
        let grid = FrequencyGrid::log_uniform(0.2, 30.0, n);
        let num = run_kernel_mms(&case, &grid, n_steps, dtau, LAMBDA_DCBR);
        let e = rel_l2_error(&case, &grid, &num, TAU_TOTAL);
        eprintln!("MMS|coupled_dcbr|spatial|N={n}|rel_l2={e:.6e}");
        errs.push(e);
    }
    let mut orders: Vec<f64> = errs.windows(2).map(|w| order_between(w[0], w[1])).collect();
    eprintln!("  spatial orders: {orders:?}");
    let med = median(&mut orders);
    assert!(
        (1.7..=2.4).contains(&med),
        "Coupled DC/BR MMS spatial order {med:.2} outside [1.7, 2.4]; errors {errs:?}"
    );
}

#[test]
fn mms_kernel_temporal_order_coupled_dcbr() {
    let case = MmsCase::default_case();
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, 4000);

    // Stronger relaxation than the spatial test so the O(dτ) backward-Euler
    // error dominates the O(dτ²) CN error inside the measured range (with
    // LAMBDA_DCBR = 1e-2 the mixture e = C₁dτ + C₂dτ² is still CN-dominated
    // at these step sizes and the measured order sits between 1 and 2).
    let lambda = 0.1;
    let m_levels = [16usize, 32, 64, 128, 256];
    let sols: Vec<Vec<f64>> = m_levels
        .iter()
        .map(|&m| run_kernel_mms(&case, &grid, m, TAU_TOTAL / m as f64, lambda))
        .collect();

    let mut diffs = Vec::new();
    for w in sols.windows(2) {
        let mut d2 = 0.0;
        let mut r2 = 0.0;
        for i in 1..grid.n {
            let dx = grid.dx[i - 1];
            let xm = grid.x_half[i - 1];
            let dm = 0.5 * ((w[0][i] - w[1][i]) + (w[0][i - 1] - w[1][i - 1]));
            let rm =
                0.5 * (case.exact(grid.x[i], TAU_TOTAL) + case.exact(grid.x[i - 1], TAU_TOTAL));
            d2 += xm.powi(3) * dm * dm * dx;
            r2 += xm.powi(3) * rm * rm * dx;
        }
        diffs.push((d2 / r2).sqrt());
    }
    let mut orders: Vec<f64> = diffs
        .windows(2)
        .map(|w| order_between(w[0], w[1]))
        .collect();
    eprintln!("MMS|coupled_dcbr|temporal diffs: {diffs:?}, orders: {orders:?}");
    let med = median(&mut orders);
    // Backward Euler on the relaxation term: first order.
    assert!(
        (0.8..=1.4).contains(&med),
        "BE coupled temporal order {med:.2} outside [0.8, 1.4]; diffs {diffs:?}"
    );
}

/// Spatial order on the production mixed log/linear grid geometry (the
/// log→linear transition and non-uniform dx must not degrade the order).
#[test]
fn mms_kernel_spatial_order_production_grid() {
    let case = MmsCase::default_case();
    let n_steps = 2048;
    let dtau = TAU_TOTAL / n_steps as f64;

    let mut errs = Vec::new();
    for &n in &[500usize, 1000, 2000, 4000] {
        let cfg = GridConfig {
            n_points: n,
            ..GridConfig::default()
        };
        let grid = FrequencyGrid::new(&cfg);
        let num = run_kernel_mms(&case, &grid, n_steps, dtau, 0.0);
        let e = rel_l2_error(&case, &grid, &num, TAU_TOTAL);
        eprintln!("MMS|production_grid|spatial|N={n}|rel_l2={e:.6e}");
        errs.push(e);
    }
    let mut orders: Vec<f64> = errs.windows(2).map(|w| order_between(w[0], w[1])).collect();
    eprintln!("  spatial orders: {orders:?}");
    let med = median(&mut orders);
    assert!(
        (1.7..=2.4).contains(&med),
        "Production-grid MMS spatial order {med:.2} outside [1.7, 2.4]; errors {errs:?}"
    );
}

// ============================================================================
// Discrete photon-number ledger on the coupled Newton path (finding P1-2)
// ============================================================================

/// Number-integral weights consistent with the kernel's flux-divergence form:
/// w_i = x_i²·Δx_cell,i (half cells at the boundaries). With these weights the
/// Kompaneets contribution telescopes to the boundary fluxes exactly.
fn number_weights(grid: &FrequencyGrid) -> Vec<f64> {
    let n = grid.n;
    let mut w = vec![0.0; n];
    w[0] = grid.x[0] * grid.x[0] * 0.5 * grid.dx[0];
    for i in 1..n - 1 {
        w[i] = grid.x[i] * grid.x[i] * 0.5 * (grid.dx[i - 1] + grid.dx[i]);
    }
    w[n - 1] = grid.x[n - 1] * grid.x[n - 1] * 0.5 * grid.dx[n - 2];
    w
}

#[test]
fn photon_number_conserved_coupled_path_pure_compton() {
    // Pure Compton (em = 0, no source) through the production coupled Newton
    // kernel: ∫x²Δn dx must be conserved to machine precision because the
    // conservative flux form telescopes (interior) and the boundary fluxes
    // are exponentially small for an interior bump.
    let case = MmsCase::default_case();
    let cfg = GridConfig {
        n_points: 2000,
        ..GridConfig::default()
    };
    let grid = FrequencyGrid::new(&cfg);
    let w = number_weights(&grid);

    let mut delta_n: Vec<f64> = grid.x.iter().map(|&x| case.exact(x, 0.0)).collect();
    let mut ws = KompaneetsWorkspace::new(&grid);
    let zeros = vec![0.0; grid.n];

    let n_of = |dn: &[f64]| -> f64 { dn.iter().zip(&w).map(|(d, wi)| d * wi).sum() };
    let n0 = n_of(&delta_n);
    let scale: f64 = delta_n.iter().zip(&w).map(|(d, wi)| d.abs() * wi).sum();

    let dtau = 0.5;
    for k in 0..200 {
        let dcbr = DcbrCoupling {
            emission_rates: &zeros,
            n_eq_minus_n_pl: &zeros,
            dem_drho_eq: &zeros,
            dneq_drho_eq: &zeros,
            photon_source: None,
            cn_dcbr: false,
        };
        let (converged, _, _) = kompaneets_step_coupled_inplace(
            &grid,
            &mut delta_n,
            case.theta,
            case.theta,
            dtau,
            Some(&dcbr),
            None,
            &mut ws,
            0.0,
            30,
        );
        assert!(converged, "Newton failed at step {k}");
    }
    let drift = (n_of(&delta_n) - n0).abs() / scale;
    eprintln!("LEDGER|pure_compton|rel_drift={drift:.3e} over 200 steps");
    assert!(
        drift < 1e-11,
        "Photon number not conserved on coupled Newton path: rel drift {drift:.3e}"
    );
}

#[test]
fn photon_number_ledger_identity_with_dcbr_and_source() {
    // With DC/BR relaxation and a photon source active, the per-step discrete
    // number balance follows exactly from the Newton residual:
    //   ΔN_step = Σ_i w_i [ dτ·em_i·(neq_i − Δn_new,i) + S_i ]
    // (Kompaneets telescopes away). Verified to Newton tolerance each step.
    let case = MmsCase::default_case();
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, 1500);
    let w = number_weights(&grid);
    let n = grid.n;

    let em: Vec<f64> = grid.x.iter().map(|&x| em_rate(x, LAMBDA_DCBR)).collect();
    let neq = vec![0.0; n];
    let zeros = vec![0.0; n];
    let mut src = vec![0.0; n];

    let mut delta_n: Vec<f64> = grid.x.iter().map(|&x| case.exact(x, 0.0)).collect();
    let mut ws = KompaneetsWorkspace::new(&grid);
    let scale: f64 = delta_n.iter().zip(&w).map(|(d, wi)| d.abs() * wi).sum();

    let dtau = 0.5;
    let n_of = |dn: &[f64]| -> f64 { dn.iter().zip(&w).map(|(d, wi)| d * wi).sum() };

    for k in 0..100 {
        let tau_mid = (k as f64 + 0.5) * dtau;
        for (i, &x) in grid.x.iter().enumerate() {
            src[i] = case.source(x, tau_mid, em[i]) * dtau;
        }
        let n_before = n_of(&delta_n);
        let dcbr = DcbrCoupling {
            emission_rates: &em,
            n_eq_minus_n_pl: &neq,
            dem_drho_eq: &zeros,
            dneq_drho_eq: &zeros,
            photon_source: Some(&src),
            cn_dcbr: false,
        };
        let (converged, _, _) = kompaneets_step_coupled_inplace(
            &grid,
            &mut delta_n,
            case.theta,
            case.theta,
            dtau,
            Some(&dcbr),
            None,
            &mut ws,
            0.0,
            30,
        );
        assert!(converged, "Newton failed at step {k}");

        let dn_actual = n_of(&delta_n) - n_before;
        let dn_predicted: f64 = (0..n)
            .map(|i| w[i] * (dtau * em[i] * (neq[i] - delta_n[i]) + src[i]))
            .sum();
        let residual = (dn_actual - dn_predicted).abs() / scale;
        assert!(
            residual < 1e-9,
            "Discrete number ledger violated at step {k}: |ΔN_actual − ΔN_pred|/scale = {residual:.3e}"
        );
    }
    eprintln!("LEDGER|dcbr_source|identity held to <1e-9 (relative) for 100 steps");
}

// ============================================================================
// Solver-level MMS: full ThermalizationSolver + TabulatedPhotonSource
// ============================================================================

/// Constrained manufactured profile for the solver-level test:
/// g(x) = g₁ − α g₂ − β g₃ (three Gaussians) with
///   ∫x³ g dx = 0  and  ∫x⁴(2n_pl+1) g dx = 0,
/// which makes the solver's perturbative Compton-equilibrium response
/// Δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃ vanish identically for Δn ∝ g (the quadratic
/// Δn² piece is dropped by the perturbative branch), so the manufactured
/// operator can use φ = 1 without modelling the T_e feedback.
struct SolverProfile {
    /// (x0, s, coefficient) triples
    parts: [(f64, f64, f64); 3],
}

impl SolverProfile {
    fn build() -> Self {
        let bases = [(2.5, 0.5), (4.0, 0.6), (6.0, 0.7)];
        // Moments by Simpson quadrature on a fine grid.
        let moments = |x0: f64, s: f64| -> (f64, f64) {
            let (lo, hi, m) = (0.01, 12.0, 24000usize);
            let h = (hi - lo) / m as f64;
            let integrand = |x: f64| {
                let (g, _, _) = gauss(x, x0, s);
                let np = planck_n(x);
                (x.powi(3) * g, x.powi(4) * (2.0 * np + 1.0) * g)
            };
            let (mut m3, mut m4) = (0.0, 0.0);
            for j in 0..=m {
                let x = lo + j as f64 * h;
                let wt = if j == 0 || j == m {
                    1.0
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                let (f3, f4) = integrand(x);
                m3 += wt * f3;
                m4 += wt * f4;
            }
            (m3 * h / 3.0, m4 * h / 3.0)
        };
        let (a3, a4) = moments(bases[0].0, bases[0].1);
        let (b3, b4) = moments(bases[1].0, bases[1].1);
        let (c3, c4) = moments(bases[2].0, bases[2].1);
        // Solve [b3 c3; b4 c4] [α; β] = [a3; a4]
        let det = b3 * c4 - c3 * b4;
        let alpha = (a3 * c4 - c3 * a4) / det;
        let beta = (b3 * a4 - a3 * b4) / det;
        SolverProfile {
            parts: [
                (bases[0].0, bases[0].1, 1.0),
                (bases[1].0, bases[1].1, -alpha),
                (bases[2].0, bases[2].1, -beta),
            ],
        }
    }

    fn eval(&self, x: f64) -> (f64, f64, f64) {
        let (mut g, mut gp, mut gpp) = (0.0, 0.0, 0.0);
        for &(x0, s, c) in &self.parts {
            let (a, b, d) = gauss(x, x0, s);
            g += c * a;
            gp += c * b;
            gpp += c * d;
        }
        (g, gp, gpp)
    }
}

/// Solver-level manufactured solution parameters.
struct SolverMms {
    profile: SolverProfile,
    a0: f64,
    z_start: f64,
    z_end: f64,
    cosmo: Cosmology,
    recomb: RecombinationHistory,
}

impl SolverMms {
    fn new() -> Self {
        let cosmo = Cosmology::default();
        let recomb = RecombinationHistory::new(&cosmo);
        SolverMms {
            profile: SolverProfile::build(),
            a0: 1e-3,
            z_start: 1.5e4,
            z_end: 1.0e4,
            cosmo,
            recomb,
        }
    }

    fn u(&self, z: f64) -> f64 {
        (self.z_start - z) / (self.z_start - self.z_end)
    }

    fn amp(&self, z: f64) -> f64 {
        self.a0 * (1.0 + 0.5 * (2.0 * std::f64::consts::PI * self.u(z)).sin())
    }

    /// da/dz
    fn amp_dz(&self, z: f64) -> f64 {
        -self.a0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * self.u(z)).cos()
            / (self.z_start - self.z_end)
    }

    fn exact(&self, x: f64, z: f64) -> f64 {
        let (g, _, _) = self.profile.eval(x);
        self.amp(z) * g
    }

    /// Manufactured source in d(Δn)/dt [1/s]:
    ///   S = ∂Δn_m/∂t − (1/t_C)·L_K[Δn_m],  φ = 1.
    fn source_dt(&self, x: f64, z: f64) -> f64 {
        let a = self.amp(z);
        let (g, gp, gpp) = self.profile.eval(x);
        let np = planck_n(x);
        let tw = 2.0 * np + 1.0;
        let h = a * gp + tw * a * g + a * a * g * g;
        let hp = a * gpp + tw * a * gp - 2.0 * np * (np + 1.0) * a * g + 2.0 * a * a * g * gp;
        let theta_z = self.cosmo.theta_z(z);
        let komp_per_tau = theta_z * (4.0 * x * h + x * x * hp);

        let hubble = self.cosmo.hubble(z);
        let dndt_time = self.amp_dz(z) * g * (-hubble * (1.0 + z));
        let t_c = self.cosmo.t_compton(z, self.recomb.x_e(z));
        dndt_time - komp_per_tau / t_c
    }
}

/// Build the dense TabulatedPhotonSource carrying the manufactured source.
/// The x-columns coincide with the solver grid nodes, so bilinear
/// interpolation is exact in x on the solver grid; the z-table is dense
/// enough that linear-in-z error is far below the assertion tolerances.
fn build_source_table(mms: &SolverMms, grid_x: &[f64]) -> InjectionScenario {
    let nz = 2000;
    let z_lo = mms.z_end * 0.95;
    let z_hi = mms.z_start * 1.05;
    let z_table: Vec<f64> = (0..nz)
        .map(|i| z_lo + (z_hi - z_lo) * i as f64 / (nz - 1) as f64)
        .collect();
    let source_2d: Vec<Vec<f64>> = z_table
        .iter()
        .map(|&z| {
            let conv = 1.0 / (mms.cosmo.hubble(z) * (1.0 + z));
            grid_x.iter().map(|&x| mms.source_dt(x, z) * conv).collect()
        })
        .collect();
    InjectionScenario::TabulatedPhotonSource {
        z_table,
        x_grid: grid_x.to_vec(),
        source_2d,
    }
}

fn run_solver_mms(mms: &SolverMms, n_points: usize, dtau_max: f64) -> (Vec<f64>, Vec<f64>) {
    let cfg = GridConfig {
        n_points,
        ..GridConfig::default()
    };
    let grid = FrequencyGrid::new(&cfg);
    let table = build_source_table(mms, &grid.x);

    let mut solver = ThermalizationSolver::builder(mms.cosmo.clone())
        .grid(cfg)
        .injection(table)
        .z_range(mms.z_start, mms.z_end)
        .dtau_max(dtau_max)
        .disable_dcbr()
        .no_number_conserving()
        .build()
        .unwrap();

    let initial: Vec<f64> = solver
        .grid
        .x
        .iter()
        .map(|&x| mms.exact(x, mms.z_start))
        .collect();
    solver.set_initial_delta_n(initial);
    solver.run_with_snapshots(&[mms.z_end]);
    let snap = solver.snapshots.last().unwrap();
    (solver.grid.x.clone(), snap.delta_n.clone())
}

#[test]
fn mms_solver_level_reproduction() {
    // End-to-end MMS through the full production integrator: adaptive
    // stepping, T_e coupling (neutralized by the constrained profile),
    // recombination-history t_C, and the TabulatedPhotonSource path.
    // Under disable_dcbr the source enters by first-order operator splitting,
    // so the expected temporal order is 1; we assert (a) monotone error
    // decrease with the expected ~2× ratio under dtau_max halving, and
    // (b) a quantitative reproduction bound at the finest level.
    let mms = SolverMms::new();

    // True error = C_split·dτ + spatial floor (the floor is dτ-independent at
    // fixed N), so measure the splitting order from successive solution
    // differences on the same grid (floor cancels) and bound the finest true
    // error separately.
    let mut errs = Vec::new();
    let mut sols: Vec<Vec<f64>> = Vec::new();
    let mut xg: Vec<f64> = Vec::new();
    for &dtau_max in &[8.0, 4.0, 2.0] {
        let (x, dn) = run_solver_mms(&mms, 2000, dtau_max);
        let mut e2 = 0.0;
        let mut r2 = 0.0;
        for i in 1..x.len() {
            let dx = x[i] - x[i - 1];
            let xm = 0.5 * (x[i] + x[i - 1]);
            let em = 0.5
                * ((dn[i] - mms.exact(x[i], mms.z_end))
                    + (dn[i - 1] - mms.exact(x[i - 1], mms.z_end)));
            let rm = 0.5 * (mms.exact(x[i], mms.z_end) + mms.exact(x[i - 1], mms.z_end));
            e2 += xm.powi(3) * em * em * dx;
            r2 += xm.powi(3) * rm * rm * dx;
        }
        let e = (e2 / r2).sqrt();
        eprintln!("MMS|solver_level|dtau_max={dtau_max}|rel_l2={e:.6e}");
        errs.push(e);
        sols.push(dn);
        xg = x;
    }

    // True errors must decrease monotonically.
    assert!(
        errs[0] > errs[1] && errs[1] > errs[2],
        "Solver-level MMS true errors not monotone: {errs:?}"
    );

    // First-order splitting: solution differences halve under dtau_max halving.
    let sol_diff = |a: &[f64], b: &[f64]| -> f64 {
        let mut d2 = 0.0;
        let mut r2 = 0.0;
        for i in 1..xg.len() {
            let dx = xg[i] - xg[i - 1];
            let xm = 0.5 * (xg[i] + xg[i - 1]);
            let dm = 0.5 * ((a[i] - b[i]) + (a[i - 1] - b[i - 1]));
            let rm = 0.5 * (mms.exact(xg[i], mms.z_end) + mms.exact(xg[i - 1], mms.z_end));
            d2 += xm.powi(3) * dm * dm * dx;
            r2 += xm.powi(3) * rm * rm * dx;
        }
        (d2 / r2).sqrt()
    };
    let d84 = sol_diff(&sols[0], &sols[1]);
    let d42 = sol_diff(&sols[1], &sols[2]);
    let order = (d84 / d42).ln() / 2.0_f64.ln();
    eprintln!("  splitting diffs: {d84:.3e}, {d42:.3e}, order {order:.2}");
    assert!(
        (0.7..=1.5).contains(&order),
        "Solver-level source-splitting temporal order {order:.2} outside [0.7, 1.5]"
    );

    // Quantitative anchor: the full production integrator reproduces the
    // analytic manufactured solution to 0.1% at dtau_max = 2, N = 2000
    // (measured ≈ 2×10⁻⁴, dominated by the spatial floor at this N).
    assert!(
        errs[errs.len() - 1] < 1e-3,
        "Solver-level MMS reproduction error too large: {:.3e}",
        errs[errs.len() - 1]
    );
}
