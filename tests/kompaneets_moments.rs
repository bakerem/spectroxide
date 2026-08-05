//! Kompaneets moment-hierarchy verification against exact analytic moment
//! identities (Part I of dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md).
//!
//! # What this is (and is NOT) — read before deleting as "redundant"
//!
//! This verifies the **bare Kompaneets operator in isolation** against exact
//! analytic moment identities whose coefficients come from the literature
//! Kompaneets equation, *not* from the code. DC/BR off, expansion off,
//! fixed θ_e, φ = 1 (except T5, which pins the (φ−1) branch).
//!
//! **This is NOT the Chluba photon Green's function** (`greens_function_photon`).
//! That object is the end-to-end cosmological response built on *fitted*
//! visibility functions validated against CosmoTherm at ~2–5%; its tolerance
//! would absorb a wrong drift/recoil coefficient.
//!
//! **Relation to `tests/mms_convergence.rs`:** MMS already verifies the
//! Kompaneets *discretization* — it injects a source built from the code's own
//! flux form and shows the kernel converges to a manufactured solution at
//! design order. MMS cannot catch a wrong coefficient *in that flux form*
//! (recoil `2Δn` vs `Δn`, `x³` vs `x⁴`): it would verify the code against the
//! same wrong equation. This file pins the **formulation** against coefficients
//! derived independently from the physics — `(k−2)(k+1)` and `(k−2)` in the
//! moment hierarchy (★), the Zeldovich–Sunyaev energy law `4 − x₀`, and the
//! analytic Y_SZ source **amplitude** (T5), which no existing test pins.
//! `photon_number_conserved_coupled_path_pure_compton` (mms_convergence.rs)
//! already covers T1; T1 here is a cheap in-harness re-anchor.
//!
//! # Physics (re-derived, not trusted blind)
//!
//! Non-relativistic Kompaneets in the Comptonization variable
//! `y = ∫ θ_e σ_T n_e c dt`, test-particle limit:
//!
//!   ∂n/∂y = (1/x²) ∂/∂x [ x⁴ ( ∂n/∂x + n ) ]   (the +n term is recoil)
//!
//! Moments `M_k(y) = ∫ x^k n dx`. Multiply by `x^{k-2}`, integrate by parts
//! twice (zero-flux boundaries kill surface terms):
//!
//!   dM_k/dy = (k-2)(k+1) M_k − (k-2) M_{k+1}     (★)  exact, recoil included
//!
//! For the code's φ=1 flux `x⁴[dΔn/dx + (2n_pl+1)Δn + Δn²]` the same two
//! integrations by parts give, with NO regime restriction,
//!
//!   dM_k/dy = (k-2)(k+1) M_k − (k-2) [ M_{k+1} + C_k ]   (★′)
//!   C_k = ∫ x^{k+1} ( 2 n_pl Δn + Δn² ) dx
//!
//! (★) is (★′) with C_k → 0 in the test-particle regime. Tier (a) tests use
//! (★) (analytic coefficients only — the independent-physics content). Tier
//! (b) uses (★′) with the measured C_k (exact for the code's flux at continuum
//! level — the regime-robust diagnostic).

use spectroxide::grid::FrequencyGrid;
use spectroxide::kompaneets::{KompaneetsWorkspace, kompaneets_step_coupled_inplace};
use spectroxide::spectrum;

// ---------------------------------------------------------------------------
// Quadrature and profiles
// ---------------------------------------------------------------------------

fn planck_n(x: f64) -> f64 {
    1.0 / x.exp_m1()
}

/// Number-integral weights consistent with the kernel's flux-divergence form:
/// `w_i = x_i²·Δx_cell,i` (half cells at the boundaries). Copied verbatim from
/// `tests/mms_convergence.rs::number_weights` — with these weights the
/// Kompaneets number contribution telescopes to the boundary fluxes exactly
/// (this is the kernel's exact discrete number invariant; any other quadrature
/// breaks T1, plan stumbling point 8).
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

fn gaussian_line(grid: &FrequencyGrid, x0: f64, sigma: f64, amp: f64) -> Vec<f64> {
    grid.x
        .iter()
        .map(|&x| {
            let u = (x - x0) / sigma;
            amp * (-0.5 * u * u).exp()
        })
        .collect()
}

/// Moments `M_k = Σ_i w_i x_i^{k-2} Δn_i` for k = 2..6 (returns `[M2,M3,M4,M5,M6]`).
fn moments_2_6(grid: &FrequencyGrid, w: &[f64], dn: &[f64]) -> [f64; 5] {
    let mut m = [0.0; 5];
    for i in 0..grid.n {
        let wd = w[i] * dn[i];
        let x = grid.x[i];
        let mut xp = 1.0; // x^{k-2}, k=2 → 0
        for entry in m.iter_mut() {
            *entry += wd * xp;
            xp *= x;
        }
    }
    m
}

/// Flux-correction moments `C_k = Σ_i w_i x_i^{k-1} (2 n_pl,i Δn_i + Δn_i²)`
/// for k = 2..5 (returns `[C2,C3,C4,C5]`). Same weights/cell widths as the M_k.
fn cmoms_2_5(grid: &FrequencyGrid, w: &[f64], npl: &[f64], dn: &[f64]) -> [f64; 4] {
    let mut c = [0.0; 4];
    for i in 0..grid.n {
        let f = 2.0 * npl[i] * dn[i] + dn[i] * dn[i];
        let wf = w[i] * f;
        let x = grid.x[i];
        let mut xp = x; // x^{k-1}, k=2 → 1
        for entry in c.iter_mut() {
            *entry += wf * xp;
            xp *= x;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Moment-run harness
// ---------------------------------------------------------------------------

struct MomentRun {
    y: Vec<f64>,      // y[j] = θ_e·dτ·j
    m: Vec<[f64; 5]>, // m[j] = [M2..M6] at y[j]
    c: Vec<[f64; 4]>, // c[j] = [C2..C5] at y[j]
    scale: f64,       // Σ w_i |Δn_i(0)|  (number-metric scale for T1)
    dn_final: Vec<f64>,
    bdry_ratio: f64, // max(|Δn| at first/last node) / peak|Δn|, over the whole run
}

/// March the production coupled Newton kernel with a narrow Gaussian line at
/// x0, DC/BR off (`dcbr = None`), T_e fixed (`rho_coupling = None`), φ = 1
/// (`theta_e = theta_z = theta`). Records M_k, C_k at every step.
fn run_moments(
    x0: f64,
    sigma: f64,
    amp: f64,
    theta: f64,
    dtau: f64,
    n_steps: usize,
    n_grid: usize,
) -> MomentRun {
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, n_grid);
    let w = number_weights(&grid);
    let npl: Vec<f64> = grid.x.iter().map(|&x| planck_n(x)).collect();
    let mut dn = gaussian_line(&grid, x0, sigma, amp);
    let mut ws = KompaneetsWorkspace::new(&grid);

    let scale: f64 = dn.iter().zip(&w).map(|(d, wi)| d.abs() * wi).sum();
    let peak: f64 = dn.iter().fold(0.0_f64, |a, &d| a.max(d.abs()));

    let mut y = Vec::with_capacity(n_steps + 1);
    let mut m = Vec::with_capacity(n_steps + 1);
    let mut c = Vec::with_capacity(n_steps + 1);
    let mut bdry_ratio: f64 = 0.0;

    for step in 0..=n_steps {
        if step > 0 {
            let (converged, _, _) = kompaneets_step_coupled_inplace(
                &grid, &mut dn, theta, theta, dtau, None, None, &mut ws, 0.0, 30,
            );
            assert!(converged, "Newton diverged at step {step}");
        }
        y.push(theta * dtau * step as f64);
        m.push(moments_2_6(&grid, &w, &dn));
        c.push(cmoms_2_5(&grid, &w, &npl, &dn));
        let b = dn[0].abs().max(dn[grid.n - 1].abs()) / peak;
        bdry_ratio = bdry_ratio.max(b);
    }

    MomentRun {
        y,
        m,
        c,
        scale,
        dn_final: dn,
        bdry_ratio,
    }
}

// Default regime (plan §3): x0=7, σ0=0.5, A=1e-3; θ_e·dτ = 2.5e-4;
// y_total = 8e-3 → σ_f ≈ 1.0 (broadening-limited, boundaries clean).
const X0: f64 = 7.0;
const SIGMA0: f64 = 0.5;
const AMP: f64 = 1e-3;
const THETA: f64 = 1e-2;
const DTAU: f64 = 2.5e-2;
const N_STEPS: usize = 32;
const N_GRID: usize = 2000;

fn default_run() -> MomentRun {
    run_moments(X0, SIGMA0, AMP, THETA, DTAU, N_STEPS, N_GRID)
}

// Index helpers: M_k lives at m[k-2], C_k at c[k-2], M_{k+1} at m[k-1].
#[inline]
fn m_of(row: &[f64; 5], k: usize) -> f64 {
    row[k - 2]
}
#[inline]
fn c_of(row: &[f64; 4], k: usize) -> f64 {
    row[k - 2]
}

/// Analytic RHS of (★): (k-2)(k+1) M_k − (k-2) M_{k+1}.
fn rhs_tier_a(row: &[f64; 5], k: usize) -> f64 {
    let kf = k as f64;
    (kf - 2.0) * (kf + 1.0) * m_of(row, k) - (kf - 2.0) * m_of(row, k + 1)
}

/// Analytic RHS of (★′): tier-a − (k-2) C_k.
fn rhs_tier_b(mrow: &[f64; 5], crow: &[f64; 4], k: usize) -> f64 {
    rhs_tier_a(mrow, k) - (k as f64 - 2.0) * c_of(crow, k)
}

// ===========================================================================
// T1 — photon number conserved (anchor; cites the mms_convergence.rs test)
// ===========================================================================

#[test]
fn t1_photon_number_conserved() {
    let r = default_run();
    let m2_0 = m_of(&r.m[0], 2);
    let mut worst = 0.0_f64;
    for row in &r.m {
        let drift = (m_of(row, 2) - m2_0).abs() / r.scale;
        worst = worst.max(drift);
    }
    eprintln!("T1|photon_number|max_rel_drift={worst:.3e} over {N_STEPS} steps");
    // Machine-precision flux telescoping; cf.
    // photon_number_conserved_coupled_path_pure_compton (mms_convergence.rs)
    // which achieves <1e-11 over 200 steps. If T1 fails here but that passes,
    // the weights were rolled wrong (plan §4/§7.8).
    assert!(
        worst < 1e-9,
        "M_2 (photon number) not conserved: max rel drift {worst:.3e}"
    );
}

// ===========================================================================
// T2 — Zeldovich–Sunyaev energy law
// ===========================================================================

#[test]
fn t2_energy_law_zeldovich_sunyaev() {
    let r = default_run();
    let dy = r.y[1] - r.y[0];

    // Measured d ln M_3/dy over the first interval (midpoint y ≈ dy/2 → y→0).
    let slope = (m_of(&r.m[1], 3).ln() - m_of(&r.m[0], 3).ln()) / dy;

    // Tier-a ZS form: 4 − M_4/M_3 at y=0 (omits C_3). Human-readable anchor.
    let ratio0 = m_of(&r.m[0], 4) / m_of(&r.m[0], 3);
    let zs_a = 4.0 - ratio0;

    // Tier-b exact form at the interval midpoint: 4 − (M_4 + C_3)/M_3.
    let mid_m4 = 0.5 * (m_of(&r.m[0], 4) + m_of(&r.m[1], 4));
    let mid_m3 = 0.5 * (m_of(&r.m[0], 3) + m_of(&r.m[1], 3));
    let mid_c3 = 0.5 * (c_of(&r.c[0], 3) + c_of(&r.c[1], 3));
    let zs_b = 4.0 - (mid_m4 + mid_c3) / mid_m3;

    eprintln!(
        "T2|ZS: measured d ln M_3/dy = {slope:.5} | 4−x0 = {:.5} | 4−M4/M3 (tier a) = {zs_a:.5} | tier-b = {zs_b:.5}",
        4.0 - X0
    );

    // (a) M_4/M_3 ≈ x0 for a narrow line: M4/M3 = x0(1 + 3σ²/x0² + …). Measure
    //     the deviation rather than trusting the series coefficient.
    let ratio_dev = (ratio0 - X0).abs();
    let ratio_bound = 3.0 * SIGMA0 * SIGMA0 / X0 + 0.02; // width term + quadrature slop
    assert!(
        ratio_dev < ratio_bound,
        "M_4/M_3 = {ratio0:.4} deviates from x0 = {X0} by {ratio_dev:.4} > {ratio_bound:.4}"
    );

    // (b) Exact tier-b: slope must equal 4 − (M4+C3)/M3 to the FD truncation
    //     floor. This is (★′) for k=3 in log form — regime-robust.
    let res_b = (slope - zs_b).abs();
    eprintln!("T2|tier-b residual = {res_b:.3e}");
    assert!(
        res_b < 5e-4,
        "ZS energy law (tier b) residual {res_b:.3e} exceeds truncation floor"
    );

    // (c) Tier-a ZS form within contamination bound (measured C_3 term).
    let contam = (mid_c3 / mid_m3).abs();
    let res_a = (slope - zs_a).abs();
    eprintln!("T2|tier-a residual = {res_a:.3e} | C_3/M_3 contamination = {contam:.3e}");
    assert!(
        res_a < contam + 5e-4,
        "ZS tier-a residual {res_a:.3e} exceeds contamination {contam:.3e} + floor"
    );
}

// ===========================================================================
// T3 — moment hierarchy k = 3,4,5, two tiers (core test)
// ===========================================================================

/// For each recorded interval, central FD_y(M_k) at the midpoint vs the tier-a
/// (★) and tier-b (★′) RHS. Returns (max relative tier-a residual, max relative
/// tier-b residual, max relative contamination bound) over all intervals.
fn hierarchy_residuals(r: &MomentRun, k: usize) -> (f64, f64, f64) {
    let mut worst_a = 0.0_f64;
    let mut worst_b = 0.0_f64;
    let mut worst_contam = 0.0_f64;
    for j in 0..r.y.len() - 1 {
        let dy = r.y[j + 1] - r.y[j];
        let fd = (m_of(&r.m[j + 1], k) - m_of(&r.m[j], k)) / dy;

        let rhs_a = 0.5 * (rhs_tier_a(&r.m[j], k) + rhs_tier_a(&r.m[j + 1], k));
        let rhs_b = 0.5
            * (rhs_tier_b(&r.m[j], &r.c[j], k) + rhs_tier_b(&r.m[j + 1], &r.c[j + 1], k));

        let denom = rhs_b.abs().max(1e-300);
        worst_a = worst_a.max((fd - rhs_a).abs() / denom);
        worst_b = worst_b.max((fd - rhs_b).abs() / denom);

        // Measured contamination = |(k-2) C_k / RHS_b|.
        let ck = 0.5 * (c_of(&r.c[j], k) + c_of(&r.c[j + 1], k));
        worst_contam = worst_contam.max(((k as f64 - 2.0) * ck).abs() / denom);
    }
    (worst_a, worst_b, worst_contam)
}

/// Tier-b relative residual floor. Established as truncation-dominated by
/// `t4_light_truncation_floor` (refinement-responsive). Far tighter than the
/// O(1) relative error a coefficient bug produces (recoil off by 1 shifts the
/// k=3 RHS by M_4/|RHS| ≈ 7/3 ≈ 230%), far looser than the observed finest-grid
/// residual — the plan §6 anti-tuning window.
const TIER_B_FLOOR: f64 = 3e-3;

fn assert_hierarchy(k: usize) {
    let r = default_run();
    let (res_a, res_b, contam) = hierarchy_residuals(&r, k);
    eprintln!(
        "T3|k={k}: max rel residual tier-a={res_a:.3e} tier-b={res_b:.3e} | contamination bound={contam:.3e}"
    );
    // Tier (b): exact for the code's flux ⇒ truncation floor only.
    assert!(
        res_b < TIER_B_FLOOR,
        "k={k} tier-b residual {res_b:.3e} exceeds truncation floor {TIER_B_FLOOR:.3e} \
         ⇒ formulation/stencil bug (or y-vs-τ factor)"
    );
    // Tier (a): physics identity ⇒ floor + measured contamination.
    let tol_a = TIER_B_FLOOR + 1.5 * contam;
    assert!(
        res_a < tol_a,
        "k={k} tier-a residual {res_a:.3e} exceeds floor+contamination {tol_a:.3e}. \
         If tier-b passed this is regime contamination (shrink y / raise x0), NOT a tolerance to loosen"
    );
}

#[test]
fn t3_hierarchy_k3() {
    assert_hierarchy(3);
}

#[test]
fn t3_hierarchy_k4() {
    assert_hierarchy(4);
}

#[test]
fn t3_hierarchy_k5() {
    // M_6 at x0=7, A=1e-3 is well above the f64 roundoff floor (checked in
    // t3_moment_roundoff_floor); enable k=5.
    assert_hierarchy(5);
}

#[test]
fn t3_moment_roundoff_floor() {
    // Plan §3 / stumbling point 6: verify M_6 is not roundoff-dominated before
    // trusting the k=5 hierarchy. Ratio = |M_6| / (ε · Σ w_i x_i⁴ |Δn_i|)
    // must stay ≫ 1 (target ≥ 1e8).
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, N_GRID);
    let w = number_weights(&grid);
    let dn = gaussian_line(&grid, X0, SIGMA0, AMP);
    let m6 = moments_2_6(&grid, &w, &dn)[4].abs();
    let round: f64 = (0..grid.n)
        .map(|i| w[i] * grid.x[i].powi(4) * dn[i].abs())
        .sum();
    let ratio = m6 / (f64::EPSILON * round);
    eprintln!("T3|M_6/(ε·Σw x⁴|Δn|) = {ratio:.3e}");
    assert!(ratio > 1e8, "M_6 is roundoff-dominated (ratio {ratio:.3e})");
}

// ===========================================================================
// T4-light — the tier-b residual is truncation-dominated (anti-tuning proof)
// ===========================================================================

#[test]
fn t4_light_truncation_floor() {
    // MMS already established O(Δx²)/O(Δτ²) for this operator at φ=1. Here we
    // only show the tier-b residual responds to refinement (⇒ truncation-, not
    // bug-dominated), justifying TIER_B_FLOOR as derived rather than tuned.
    let base = default_run();
    let (_, res_base, _) = hierarchy_residuals(&base, 3);

    // Halve dx (double grid points), hold y and dτ.
    let fine_x = run_moments(X0, SIGMA0, AMP, THETA, DTAU, N_STEPS, 2 * N_GRID);
    let (_, res_fine_x, _) = hierarchy_residuals(&fine_x, 3);

    // Halve dτ (double steps), hold y_total and dx.
    let fine_t = run_moments(X0, SIGMA0, AMP, THETA, DTAU / 2.0, 2 * N_STEPS, N_GRID);
    let (_, res_fine_t, _) = hierarchy_residuals(&fine_t, 3);

    eprintln!(
        "T4-light|k=3 tier-b residual: base={res_base:.3e} halve-dx={res_fine_x:.3e} halve-dtau={res_fine_t:.3e}"
    );

    // The residual must respond to at least one refinement (spatial or
    // temporal) by ≳3×, OR already sit at a small floor. Either way the
    // TIER_B_FLOOR is not masking a constant (bug-like) error.
    let responds = res_fine_x < res_base / 3.0 || res_fine_t < res_base / 3.0;
    let at_floor = res_base < TIER_B_FLOOR / 3.0;
    assert!(
        responds || at_floor,
        "Tier-b residual neither refinement-responsive nor at floor: \
         base={res_base:.3e} dx={res_fine_x:.3e} dtau={res_fine_t:.3e} — TIER_B_FLOOR may be masking a bug"
    );
    // Refinement must never inflate the residual (would signal instability).
    assert!(
        res_fine_x <= 2.0 * res_base && res_fine_t <= 2.0 * res_base,
        "Refinement inflated the tier-b residual — instability?"
    );
}

// ===========================================================================
// T5 — (φ−1) source term: pointwise Y_SZ shape AND amplitude (coverage hole)
// ===========================================================================

/// Y_SZ hardcoded from the analytic form, importing nothing from spectrum.rs:
/// Y_SZ(x) = [x e^x/(e^x−1)²]·[x coth(x/2) − 4].
fn ysz_hardcoded(x: f64) -> f64 {
    let ex = x.exp();
    let gbb = x * ex / ((ex - 1.0) * (ex - 1.0));
    let coth_half = (x / 2.0).cosh() / (x / 2.0).sinh();
    gbb * (x * coth_half - 4.0)
}

/// Run the (φ−1) branch: Δn=0 initially, φ = 1−ε (T_e > T_z), pure Kompaneets.
/// Returns (grid, Δn_final, Δy) with Δy = θ_e·Σdτ.
fn run_phi_source(eps: f64, n_steps: usize, sigma_dtau: f64) -> (FrequencyGrid, Vec<f64>, f64) {
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, N_GRID);
    let theta_z = 1e-2;
    let theta_e = theta_z / (1.0 - eps); // φ = θ_z/θ_e = 1−ε
    let dtau = sigma_dtau / n_steps as f64;
    let mut dn = vec![0.0; grid.n];
    let mut ws = KompaneetsWorkspace::new(&grid);
    for step in 0..n_steps {
        let (converged, _, _) = kompaneets_step_coupled_inplace(
            &grid, &mut dn, theta_e, theta_z, dtau, None, None, &mut ws, 0.0, 30,
        );
        assert!(converged, "Newton diverged at φ-source step {step}");
    }
    let dy = theta_e * sigma_dtau;
    (grid, dn, dy)
}

#[test]
fn t5_phi_source_shape_and_amplitude() {
    let eps = 1e-3; // 1 − φ, T_e > T_z
    let sigma_dtau = 0.1; // Σdτ ⇒ Δy = θ_e·0.1 ≈ 1e-3
    let (grid, dn, dy) = run_phi_source(eps, 4, sigma_dtau);

    // Richardson companion at half dτ (same Δy) to bound temporal truncation.
    let (_, dn_fine, _) = run_phi_source(eps, 8, sigma_dtau);

    // Predicted: Δn(x) = Δy·(1−φ)·Y_SZ(x) = Δy·ε·Y_SZ(x).
    let pred = |x: f64| dy * eps * ysz_hardcoded(x);
    let peak: f64 = grid
        .x
        .iter()
        .map(|&x| pred(x).abs())
        .fold(0.0_f64, f64::max);

    // Pointwise over x ∈ [0.5, 15].
    let mut worst_rel = 0.0_f64;
    let mut worst_rich = 0.0_f64;
    for (i, &x) in grid.x.iter().enumerate() {
        if !(0.5..=15.0).contains(&x) {
            continue;
        }
        worst_rel = worst_rel.max((dn[i] - pred(x)).abs() / peak);
        worst_rich = worst_rich.max((dn[i] - dn_fine[i]).abs() / peak);
    }
    eprintln!(
        "T5|(φ−1) source: max rel shape err = {worst_rel:.3e} | Richardson(dτ) spread = {worst_rich:.3e} | Δy = {dy:.3e}"
    );
    // Tolerance: Richardson-measured truncation + O(ε) + spatial discretization.
    // Well below the θ_z↔θ_e swap signature (which would rescale by φ ≈ 1−ε,
    // i.e. shift the whole amplitude by ε).
    let tol = 5.0 * worst_rich + 3.0 * eps + 5e-3;
    assert!(
        worst_rel < tol,
        "T5 pointwise (φ−1) shape/amplitude error {worst_rel:.3e} exceeds {tol:.3e}"
    );

    // Sign anchors (plan §7.10): T_e > T_z ⇒ Δn > 0 at high x, < 0 at low x,
    // zero crossing near x = 3.830.
    let at = |xt: f64| {
        let i = grid
            .x
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - xt).abs().partial_cmp(&(b.1 - xt).abs()).unwrap())
            .unwrap()
            .0;
        dn[i]
    };
    assert!(at(10.0) > 0.0, "Δn should be > 0 at x=10 for T_e>T_z");
    assert!(at(1.0) < 0.0, "Δn should be < 0 at x=1 for T_e>T_z");
    // Crossing: |Δn| near 3.830 must be small relative to peak.
    let cross = at(3.830).abs() / peak;
    assert!(cross < 0.05, "Y_SZ zero crossing not near 3.830 (|Δn|/peak={cross:.3e})");

    // Library-convention check: hardcoded Y_SZ must match spectrum::y_shape
    // (catches convention drift and validates the library shape normalization).
    let mut worst_lib = 0.0_f64;
    for &x in &grid.x {
        if !(0.5..=15.0).contains(&x) {
            continue;
        }
        let denom = ysz_hardcoded(x).abs().max(1e-6);
        worst_lib = worst_lib.max((ysz_hardcoded(x) - spectrum::y_shape(x)).abs() / denom);
    }
    eprintln!("T5|hardcoded Y_SZ vs spectrum::y_shape: max rel diff = {worst_lib:.3e}");
    assert!(worst_lib < 1e-10, "spectrum::y_shape disagrees with analytic Y_SZ");

    // Number conservation on the (φ−1) branch. The branch is a pure flux
    // divergence, so the discrete operator injects EXACTLY the photon number
    // carried by the source shape — no spurious interior source/sink. On the
    // truncated grid ∫x²Y_SZ dx ≠ 0: the −2/x low-x tail is cut at x_min, and
    // the identity ∫x²Y_SZ = 0 holds only over (0,∞). The correct statement is
    // therefore M_2 = (number of the PREDICTED shape on [x_min,x_max]), not
    // M_2 = 0. A residual beyond that would flag a spurious number source on
    // this flux branch.
    let w = number_weights(&grid);
    let m2_meas: f64 = dn.iter().zip(&w).map(|(d, wi)| d * wi).sum();
    let m2_pred: f64 = grid.x.iter().zip(&w).map(|(&x, wi)| pred(x) * wi).sum();
    let m2_scale: f64 = dn.iter().zip(&w).map(|(d, wi)| d.abs() * wi).sum();
    let m2_rel = (m2_meas - m2_pred).abs() / m2_scale.max(1e-300);
    eprintln!(
        "T5|(φ−1) number: M_2 meas={m2_meas:.4e} pred(truncated Y_SZ)={m2_pred:.4e} → |Δ|/scale = {m2_rel:.3e}"
    );
    assert!(
        m2_rel < 1e-3,
        "(φ−1) branch injects spurious number beyond the source shape: {m2_rel:.3e}"
    );
}

// ===========================================================================
// T6 — linearity diagnostic
// ===========================================================================

#[test]
fn t6_linearity() {
    // Rerun the T3 harness at A, A/2, −A. The tier-b relative residual must be
    // ~amplitude-independent (it is a relative quantity at the truncation
    // floor); a residual scaling ∝ A isolates the Δn² term, an A→−A asymmetry
    // flags something worse.
    let full = run_moments(X0, SIGMA0, AMP, THETA, DTAU, N_STEPS, N_GRID);
    let half = run_moments(X0, SIGMA0, AMP / 2.0, THETA, DTAU, N_STEPS, N_GRID);
    let neg = run_moments(X0, SIGMA0, -AMP, THETA, DTAU, N_STEPS, N_GRID);

    let (_, b_full, cont_full) = hierarchy_residuals(&full, 3);
    let (_, b_half, cont_half) = hierarchy_residuals(&half, 3);
    let (_, b_neg, cont_neg) = hierarchy_residuals(&neg, 3);
    eprintln!(
        "T6|k=3 tier-b rel residual: A={b_full:.3e} A/2={b_half:.3e} −A={b_neg:.3e}"
    );
    eprintln!(
        "T6|k=3 contamination bound: A={cont_full:.3e} A/2={cont_half:.3e} −A={cont_neg:.3e}"
    );

    // All three at the same (truncation) floor.
    for (label, res) in [("A/2", b_half), ("−A", b_neg)] {
        assert!(
            res < TIER_B_FLOOR,
            "T6 tier-b residual at {label} = {res:.3e} exceeds floor"
        );
    }
    // Sign symmetry under A → −A: relative residual comparable (the tier-b RHS
    // is exact for either sign; a large asymmetry signals a sign-dependent bug).
    let asym = (b_full - b_neg).abs() / (b_full + b_neg).max(1e-300);
    eprintln!("T6|A→−A tier-b relative asymmetry = {asym:.3e}");
    assert!(asym < 0.5, "Large A→−A asymmetry in tier-b residual: {asym:.3e}");

    // The Δn²-driven contamination scales ~linearly with A (relative): halving
    // A should roughly halve the contamination bound. Loose sanity check.
    assert!(
        cont_half < cont_full,
        "Contamination bound did not shrink with amplitude (A/2)"
    );
    let _ = cont_neg;
}

// ===========================================================================
// Boundary / regime guards (plan §7.3, §7.4)
// ===========================================================================

#[test]
fn regime_boundary_clean() {
    let r = default_run();
    eprintln!("regime|max boundary |Δn|/peak over run = {:.3e}", r.bdry_ratio);
    // The broadened line must not reach the grid boundaries (else the zero-flux
    // treatment contaminates the moments).
    assert!(
        r.bdry_ratio < 1e-6,
        "Line reached grid boundary (|Δn|/peak = {:.3e}) — shrink y or widen grid",
        r.bdry_ratio
    );
    // Final drift ⟨x⟩ = M_3/M_2 must stay near x0 (broadening-limited, not
    // drifted into a boundary): plan §7.3.
    let last = r.m.last().unwrap();
    let mean_x = m_of(last, 3) / m_of(last, 2);
    eprintln!("regime|⟨x⟩ final = {mean_x:.4} (x0 = {X0})");
    assert!(
        (mean_x - X0).abs() < 1.0,
        "⟨x⟩ drifted too far from x0: {mean_x:.4}"
    );
    let _ = r.dn_final;
}

// ===========================================================================
// II.6 — H-theorem monotonicity (nonlinear-regime coverage, plan §II.6)
// ===========================================================================
//
// Everything above is linearized. For fixed T_e the continuum Kompaneets flow
// monotonically decreases the free-energy functional (code units, φ = T_z/T_e,
// n = n_pl + Δn the FULL occupation):
//
//   F[n] = ∫ x²[ φ·x·n + n ln n − (1+n) ln(1+n) ] dx
//   dF/dy = −∫ x⁴ n(1+n) [ ∂_x( ln(n/(1+n)) + φ x ) ]² dx  ≤ 0
//
// (flux J = x⁴ n(1+n) ∂_x[ln(n/(1+n)) + φx]; one integration by parts,
// zero-flux boundaries.) A structural inequality valid at ANY amplitude — the
// only check here of the solver's nonlinear regime. CN preserves it only to
// O(Δτ²), so positive increments are bounded by the truncation measured by
// halving dτ, not hand-tuned.

/// F[n] = Σ_i w_i [ φ x_i n_i + n_i ln n_i − (1+n_i) ln(1+n_i) ], n_i = n_pl+Δn.
fn free_energy(grid: &FrequencyGrid, w: &[f64], npl: &[f64], dn: &[f64], phi: f64) -> f64 {
    let mut f = 0.0;
    for i in 0..grid.n {
        let n = npl[i] + dn[i];
        debug_assert!(n > 0.0, "n must stay positive for the H-theorem logs");
        f += w[i] * (phi * grid.x[i] * n + n * n.ln() - (1.0 + n) * (1.0 + n).ln());
    }
    f
}

/// Run pure Kompaneets (φ=1) from a positive interior bump; return
/// (max positive per-step ΔF, total F decrease F[0]−F[end]).
fn h_theorem_run(amp_frac: f64, dtau: f64, n_steps: usize) -> (f64, f64) {
    let grid = FrequencyGrid::log_uniform(0.2, 30.0, 2000);
    let w = number_weights(&grid);
    let npl: Vec<f64> = grid.x.iter().map(|&x| planck_n(x)).collect();
    // Interior bump at x0=1 (n_pl≈0.58); positive so n = n_pl+Δn > 0 always.
    let x0 = 1.0;
    let amp = amp_frac * planck_n(x0);
    let mut dn = gaussian_line(&grid, x0, 0.5, amp);
    let mut ws = KompaneetsWorkspace::new(&grid);

    let theta = 1e-2; // φ = 1
    let f0 = free_energy(&grid, &w, &npl, &dn, 1.0);
    let mut f_prev = f0;
    let mut max_pos = 0.0_f64;
    for step in 0..n_steps {
        let (converged, _, _) = kompaneets_step_coupled_inplace(
            &grid, &mut dn, theta, theta, dtau, None, None, &mut ws, 0.0, 30,
        );
        assert!(converged, "Newton diverged at H-theorem step {step}");
        let f = free_energy(&grid, &w, &npl, &dn, 1.0);
        max_pos = max_pos.max(f - f_prev); // positive ⇒ inequality violated
        f_prev = f;
    }
    (max_pos, f0 - f_prev)
}

#[test]
fn t_h_theorem_monotonicity() {
    // Moderate amplitude (max|Δn| ~ 0.1 n_pl) and a large-amplitude run
    // (Δn ~ n_pl) — the genuinely nonlinear regime.
    for &amp_frac in &[0.1, 1.0] {
        let (pos_coarse, drop_coarse) = h_theorem_run(amp_frac, 0.5, 60);
        let (pos_fine, _) = h_theorem_run(amp_frac, 0.25, 120);
        eprintln!(
            "II.6|H-theorem amp_frac={amp_frac}: max +ΔF coarse={pos_coarse:.3e} fine={pos_fine:.3e} | total F drop={drop_coarse:.3e}"
        );
        // F must decrease overall (the flow is dissipative).
        assert!(drop_coarse > 0.0, "F did not decrease (amp_frac={amp_frac})");
        // Any positive per-step increment is O(Δτ²) truncation: it must shrink
        // under dτ halving (or already be negligible), and be tiny vs the total
        // F decrease (not a systematic increase).
        let truncation_ok = pos_fine < pos_coarse || pos_coarse < 1e-12 * drop_coarse.abs();
        assert!(
            truncation_ok,
            "Positive ΔF not truncation-dominated (amp_frac={amp_frac}): coarse={pos_coarse:.3e} fine={pos_fine:.3e}"
        );
        assert!(
            pos_coarse < 1e-3 * drop_coarse.abs().max(1e-300),
            "H-theorem violated beyond truncation (amp_frac={amp_frac}): max +ΔF={pos_coarse:.3e} vs total drop={drop_coarse:.3e}"
        );
    }
}
