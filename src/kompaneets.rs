//! Kompaneets equation: Compton scattering of photons off thermal electrons.
//!
//! The Kompaneets equation describes the Fokker-Planck (diffusion) approximation
//! to Compton scattering in the non-relativistic limit:
//!
//!   dn/dτ|_C = (θ_e / x²) ∂/∂x [x⁴ (∂n/∂x + φ n(n+1))]
//!
//! where φ = T_z/T_e, θ_e = kT_e/(m_e c²), τ = t/t_C.
//!
//! For small distortions Δn = n - n_pl, the linearized form is:
//!
//!   dΔn/dτ|_C = (θ_e / x²) ∂/∂x [x⁴ (∂Δn/∂x + φ(2n_pl+1)Δn)] + source
//!
//! Discretized with second-order conservative finite differences and solved
//! with Crank-Nicolson time stepping → tridiagonal system.
//!
//! References:
//! - Kompaneets (1957), JETP
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 4]

use crate::grid::FrequencyGrid;
use crate::spectrum::planck;

/// Compute the Kompaneets operator applied to Δn on the grid (test-only).
///
/// Returns dΔn/dτ|_C at each grid point. Production code uses the
/// coupled inplace solver instead.
#[cfg(test)]
pub fn kompaneets_rhs(
    grid: &FrequencyGrid,
    delta_n: &[f64],
    theta_e: f64,
    theta_z: f64,
) -> Vec<f64> {
    let n = grid.n;
    let phi = theta_z / theta_e; // T_z / T_e = 1/ρ_e
    let mut rhs = vec![0.0; n];

    // Compute fluxes at cell interfaces using the split form.
    //
    // Full flux: F = x⁴ [dn/dx + φ n(1+n)]
    // With n = n_pl + Δn and the identity dn_pl/dx = -n_pl(1+n_pl):
    //   dn/dx + φ n(1+n) = -n_pl(1+n_pl) + dΔn/dx + φ[n_pl(1+n_pl) + (2n_pl+1)Δn + Δn²]
    //                     = (φ-1)n_pl(1+n_pl) + dΔn/dx + φ(2n_pl+1)Δn + φΔn²
    //
    // This form is numerically stable because each term is computed directly.
    let mut flux = vec![0.0; n - 1];

    for i in 0..n - 1 {
        let x_half = grid.x_half[i];
        let dx = grid.dx[i];

        let n_pl_half = planck(x_half);
        let dn_half = 0.5 * (delta_n[i] + delta_n[i + 1]);

        // Derivative of the distortion at the interface
        let ddn_dx = (delta_n[i + 1] - delta_n[i]) / dx;

        // Source from T_e ≠ T_z (analytical, no cancellation)
        let source = (phi - 1.0) * n_pl_half * (n_pl_half + 1.0);

        // Linearized drift on Δn
        let drift_linear = phi * (2.0 * n_pl_half + 1.0) * dn_half;

        // Nonlinear stimulated term
        let drift_nonlinear = phi * dn_half * dn_half;

        flux[i] = x_half.powi(4) * (ddn_dx + source + drift_linear + drift_nonlinear);
    }

    // Convert flux divergence to dΔn/dτ:
    // dΔn/dτ = (θ_e / x²) × (F_{i+1/2} - F_{i-1/2}) / Δx_cell
    for i in 1..n - 1 {
        let x = grid.x[i];
        let dx_cell = 0.5 * (grid.dx[i - 1] + grid.dx[i]);
        rhs[i] = theta_e / (x * x) * (flux[i] - flux[i - 1]) / dx_cell;
    }

    // Boundary conditions: Δn → 0 at both ends
    rhs[0] = 0.0;
    rhs[n - 1] = 0.0;

    rhs
}

/// Build tridiagonal matrix coefficients for the linearized Kompaneets equation (test-only).
///
/// Production code uses the coupled inplace solver instead.
#[cfg(test)]
pub fn kompaneets_tridiagonal(
    grid: &FrequencyGrid,
    theta_e: f64,
    theta_z: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = grid.n;
    let phi = theta_z / theta_e;

    let mut lower = vec![0.0; n]; // A_i (coefficient of Δn_{i-1})
    let mut diag = vec![0.0; n]; // B_i (coefficient of Δn_i)
    let mut upper = vec![0.0; n]; // C_i (coefficient of Δn_{i+1})
    let mut source = vec![0.0; n]; // Source term from T_e ≠ T_z

    for i in 1..n - 1 {
        let x = grid.x[i];
        let x2 = x * x;

        // Cell widths
        let dx_left = grid.dx[i - 1];
        let dx_right = grid.dx[i];
        let dx_cell = 0.5 * (dx_left + dx_right);

        // Half-point values
        let x_left = grid.x_half[i - 1];
        let x_right = grid.x_half[i];

        // Diffusion coefficients at half-points: D = θ_e x⁴
        let d_left = theta_e * x_left.powi(4);
        let d_right = theta_e * x_right.powi(4);

        // Drift coefficients at half-points: V = θ_e x⁴ φ (2n_pl + 1)
        // For linearization: n(n+1) ≈ n_pl(n_pl+1) + (2n_pl+1)Δn
        let n_pl_left = planck(x_left);
        let n_pl_right = planck(x_right);
        let v_left = theta_e * x_left.powi(4) * phi * (2.0 * n_pl_left + 1.0);
        let v_right = theta_e * x_right.powi(4) * phi * (2.0 * n_pl_right + 1.0);

        // Finite difference: (1/x²) * [1/Δx_cell] * [F_{i+1/2} - F_{i-1/2}]
        // F_{i+1/2} = D_{i+1/2} (Δn_{i+1} - Δn_i)/dx_right + V_{i+1/2} Δn_{i+1/2}
        // F_{i-1/2} = D_{i-1/2} (Δn_i - Δn_{i-1})/dx_left + V_{i-1/2} Δn_{i-1/2}

        // Using upwind-like averaging for drift: Δn_{i+1/2} ≈ (Δn_i + Δn_{i+1})/2
        let coeff = 1.0 / (x2 * dx_cell);

        // Contribution from right flux F_{i+1/2}
        let a_right = d_right / dx_right; // diffusion
        let b_right = v_right * 0.5; // drift (averaged)

        // Contribution from left flux F_{i-1/2}
        let a_left = d_left / dx_left;
        let b_left = v_left * 0.5;

        // F_{i+1/2} = D_R/dx_R * (Δn_{i+1} - Δn_i) + V_R/2 * (Δn_i + Δn_{i+1})
        // F_{i-1/2} = D_L/dx_L * (Δn_i - Δn_{i-1}) + V_L/2 * (Δn_{i-1} + Δn_i)
        //
        // dΔn_i/dτ = coeff * (F_R - F_L) where coeff = 1/(x² Δx_cell)

        // Coefficient of Δn_{i-1}: coeff * (D_L/dx_L - V_L/2)
        lower[i] = coeff * (a_left - b_left);

        // Coefficient of Δn_{i+1}: coeff * (D_R/dx_R + V_R/2)
        upper[i] = coeff * (a_right + b_right);

        // Coefficient of Δn_i: coeff * (-D_R/dx_R + V_R/2 - D_L/dx_L - V_L/2)
        diag[i] = coeff * (-a_right + b_right - a_left - b_left);

        // Source term from T_e ≠ T_z:
        // F^{pl} = x⁴ [dn_pl/dx + φ n_pl(1+n_pl)] = x⁴ (φ-1) n_pl(1+n_pl)
        // HOWEVER, the energy from this source is:
        //   ∫ x³ source dx = -θ_e(φ-1)I₄ = θ_e(1-φ)I₄ > 0 for T_e > T_z
        // So the source should be the DIVERGENCE of F^{pl} = x⁴(φ-1)n(1+n):
        let source_right =
            theta_e * x_right.powi(4) * (phi - 1.0) * n_pl_right * (n_pl_right + 1.0);
        let source_left = theta_e * x_left.powi(4) * (phi - 1.0) * n_pl_left * (n_pl_left + 1.0);
        source[i] = coeff * (source_right - source_left);
    }

    // Boundary conditions: Δn fixed at boundaries
    diag[0] = 1.0;
    diag[n - 1] = 1.0;

    (lower, diag, upper, source)
}

/// Solve a tridiagonal system Ax = d using the Thomas algorithm.
///
/// A is given by (lower, diag, upper) vectors.
/// Modifies `rhs` in place to contain the solution.
/// Uses pre-allocated `work` buffer for the forward sweep (must be same length as `diag`).
pub fn thomas_solve_inplace(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &mut [f64],
    work: &mut [f64],
) {
    let n = diag.len();
    assert!(n >= 2);
    assert!(lower.len() >= n);
    assert!(upper.len() >= n);
    assert!(rhs.len() >= n);
    assert!(work.len() >= n);
    assert!(
        diag[0].abs() > 0.0,
        "Thomas algorithm: zero pivot at row 0 (singular tridiagonal)"
    );

    // SAFETY: All slice accesses below are within [0, n). The asserts above
    // guarantee sufficient length. Using get_unchecked eliminates per-element
    // bounds checks in the forward/back sweeps — this function is called
    // ~10^6 times per solve (Newton iters × timesteps), so the overhead matters.
    unsafe {
        // Forward sweep
        let d0 = *diag.get_unchecked(0);
        *work.get_unchecked_mut(0) = *upper.get_unchecked(0) / d0;
        *rhs.get_unchecked_mut(0) /= d0;

        for i in 1..n - 1 {
            let denom =
                *diag.get_unchecked(i) - *lower.get_unchecked(i) * *work.get_unchecked(i - 1);
            *work.get_unchecked_mut(i) = *upper.get_unchecked(i) / denom;
            *rhs.get_unchecked_mut(i) = (*rhs.get_unchecked(i)
                - *lower.get_unchecked(i) * *rhs.get_unchecked(i - 1))
                / denom;
        }
        // Last row (no upper entry needed)
        let i = n - 1;
        let denom = *diag.get_unchecked(i) - *lower.get_unchecked(i) * *work.get_unchecked(i - 1);
        *work.get_unchecked_mut(i) = 0.0;
        *rhs.get_unchecked_mut(i) =
            (*rhs.get_unchecked(i) - *lower.get_unchecked(i) * *rhs.get_unchecked(i - 1)) / denom;

        // Back substitution
        for i in (0..n - 1).rev() {
            *rhs.get_unchecked_mut(i) -= *work.get_unchecked(i) * *rhs.get_unchecked(i + 1);
        }
    }
}

/// Solve a tridiagonal system (allocating version for tests/convenience).
pub fn thomas_solve(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &mut [f64]) -> Vec<f64> {
    let n = diag.len();
    let mut work = vec![0.0; n];
    let mut result = rhs.to_vec();
    thomas_solve_inplace(lower, diag, upper, &mut result, &mut work);
    result
}

/// Perform one Crank-Nicolson step of the linearized Kompaneets equation (test-only).
///
/// Production code uses the nonlinear coupled inplace solver instead.
#[cfg(test)]
pub fn kompaneets_step(
    grid: &FrequencyGrid,
    delta_n: &[f64],
    theta_e: f64,
    theta_z: f64,
    dtau: f64,
) -> Vec<f64> {
    let n = grid.n;
    let (lower, diag, upper, source) = kompaneets_tridiagonal(grid, theta_e, theta_z);

    // Build RHS: (I + 0.5 Δτ L) Δn + Δτ S
    let mut rhs = vec![0.0; n];
    for i in 1..n - 1 {
        rhs[i] = delta_n[i]
            + 0.5
                * dtau
                * (lower[i] * delta_n[i.saturating_sub(1)]
                    + diag[i] * delta_n[i]
                    + upper[i] * delta_n[(i + 1).min(n - 1)])
            + dtau * source[i];
    }
    // BCs
    rhs[0] = 0.0;
    rhs[n - 1] = 0.0;

    // Build LHS: (I - 0.5 Δτ L)
    let lhs_lower: Vec<f64> = lower.iter().map(|&a| -0.5 * dtau * a).collect();
    let lhs_diag: Vec<f64> = diag
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if i == 0 || i == n - 1 {
                1.0
            } else {
                1.0 - 0.5 * dtau * b
            }
        })
        .collect();
    let lhs_upper: Vec<f64> = upper.iter().map(|&c| -0.5 * dtau * c).collect();

    thomas_solve(&lhs_lower, &lhs_diag, &lhs_upper, &mut rhs)
}

/// Perform one implicit step of the NONLINEAR Kompaneets equation on Δn.
///
/// Allocating convenience wrapper around the inplace solver. Used by tests;
/// production code calls `kompaneets_step_coupled_inplace` directly.
pub fn kompaneets_step_nonlinear(
    grid: &FrequencyGrid,
    delta_n_old: &[f64],
    theta_e: f64,
    theta_z: f64,
    dtau: f64,
) -> Vec<f64> {
    let mut result = delta_n_old.to_vec();
    let mut ws = KompaneetsWorkspace::new(grid);
    let (_converged, _rho_e, _residual) = kompaneets_step_coupled_inplace(
        grid,
        &mut result,
        theta_e,
        theta_z,
        dtau,
        None::<&DcbrCoupling>,
        None,
        &mut ws,
        0.0,
        10,
    );
    result
}

/// Pre-allocated workspace for Kompaneets solver.
///
/// Grid-constant arrays (set once in `new()`) depend only on the frequency grid.
/// Per-step arrays are reused across timesteps to avoid heap allocation
/// (~15 vectors × 1000 elements, called 100K+ times per run).
pub struct KompaneetsWorkspace {
    // Grid-constant (length n_half = ng - 1):
    n_pl_half: Vec<f64>,
    np1_half: Vec<f64>,
    twonp1_half: Vec<f64>,
    x4_half: Vec<f64>,
    inv_dx: Vec<f64>,
    x4_over_dx: Vec<f64>,
    // Grid-constant geometry (length ng, only interior points used):
    inv_x2_dx_cell_geom: Vec<f64>,
    // Grid-constant quadrature weights for ∫x³ f(x) dx (length ng):
    // w_j = share of trapezoidal weight from cells on either side of grid point j.
    quad_weights_x3: Vec<f64>,
    // Per-step reusable buffers (length ng):
    inv_x2_dx_cell: Vec<f64>,
    half_dtau_coeff: Vec<f64>,
    k_old: Vec<f64>,
    dn_old: Vec<f64>,
    j_lower: Vec<f64>,
    j_diag: Vec<f64>,
    j_upper: Vec<f64>,
    rhs_buf: Vec<f64>,
    thomas_work: Vec<f64>,
    // Bordered system workspace (for coupled ρ_e solve):
    c_vec: Vec<f64>,
    v_buf: Vec<f64>,
    thomas_work2: Vec<f64>,
    // Precomputed w_j × em_j for the bordered Newton. Fixed across the
    // Newton iteration (em_rates and weights don't change within a step),
    // so we compute it once per step and reuse it in both the h_dcbr pass
    // and the bp_dot_u/bp_dot_v pass.
    wem: Vec<f64>,
    // DC/BR old-step buffers for Crank-Nicolson DC/BR option:
    pub(crate) dcbr_em_old: Vec<f64>,
    pub(crate) dcbr_neq_old: Vec<f64>,
}

/// Parameters for coupling ρ_e into the bordered Newton system.
///
/// When passed to `kompaneets_step_coupled_inplace`, ρ_e becomes
/// the (N+1)-th unknown solved simultaneously with Δn. The system
/// becomes bordered tridiagonal, solved in O(N) via two Thomas solves.
pub struct RhoECoupling {
    /// ρ_e at the start of this timestep (before update_temperatures).
    pub rho_e_old: f64,
    /// Compton coupling coefficient R = (8/3)(ρ̃_γ/α_h).
    /// This is the rate at which Compton scattering drives ρ_e → ρ_eq,
    /// per unit Thomson optical depth.
    pub r_compton: f64,
    /// Source term in the ρ_e ODE: R·ρ_eq + δρ_inj.
    pub rho_source: f64,
    /// Adiabatic cooling rate: H·t_C.
    pub lambda_exp: f64,
    /// dH_dcbr/dρ_e (finite-difference derivative from update_temperatures).
    pub dh_drho: f64,
    /// Normalization for H_dcbr integral: 1/(4·G₃·θ_z).
    pub h_norm: f64,
}

impl KompaneetsWorkspace {
    /// Create workspace for a given frequency grid. Call once in solver construction.
    pub fn new(grid: &FrequencyGrid) -> Self {
        let ng = grid.n;
        let n_half = ng - 1;

        let mut n_pl_half = vec![0.0; n_half];
        let mut np1_half = vec![0.0; n_half];
        let mut twonp1_half = vec![0.0; n_half];
        let mut x4_half = vec![0.0; n_half];
        let mut inv_dx = vec![0.0; n_half];
        let mut x4_over_dx = vec![0.0; n_half];

        for i in 0..n_half {
            let xh = grid.x_half[i];
            let np = planck(xh);
            n_pl_half[i] = np;
            np1_half[i] = np * (np + 1.0);
            twonp1_half[i] = 2.0 * np + 1.0;
            let x4 = xh * xh * xh * xh;
            x4_half[i] = x4;
            let dx = grid.dx[i];
            inv_dx[i] = 1.0 / dx;
            x4_over_dx[i] = x4 / dx;
        }

        let mut inv_x2_dx_cell_geom = vec![0.0; ng];
        for i in 1..ng - 1 {
            let xi = grid.x[i];
            let dx_cell = 0.5 * (grid.dx[i - 1] + grid.dx[i]);
            inv_x2_dx_cell_geom[i] = 1.0 / (xi * xi * dx_cell);
        }

        // Quadrature weights for ∫x³ f(x) dx: trapezoidal on cell midpoints.
        // w_j = sum of half-weights from cells adjacent to grid point j.
        let mut quad_weights_x3 = vec![0.0; ng];
        for j in 1..ng {
            let w = 0.5 * grid.x_half_cubed[j - 1] * grid.dx[j - 1];
            quad_weights_x3[j - 1] += w;
            quad_weights_x3[j] += w;
        }

        KompaneetsWorkspace {
            n_pl_half,
            np1_half,
            twonp1_half,
            x4_half,
            inv_dx,
            x4_over_dx,
            inv_x2_dx_cell_geom,
            quad_weights_x3,
            inv_x2_dx_cell: vec![0.0; ng],
            half_dtau_coeff: vec![0.0; ng],
            k_old: vec![0.0; ng],
            dn_old: vec![0.0; ng],
            j_lower: vec![0.0; ng],
            j_diag: vec![0.0; ng],
            j_upper: vec![0.0; ng],
            rhs_buf: vec![0.0; ng],
            thomas_work: vec![0.0; ng],
            c_vec: vec![0.0; ng],
            v_buf: vec![0.0; ng],
            thomas_work2: vec![0.0; ng],
            wem: vec![0.0; ng],
            dcbr_em_old: vec![0.0; ng],
            dcbr_neq_old: vec![0.0; ng],
        }
    }
}

/// DC/BR coupling data for implicit backward Euler within the Kompaneets step.
///
/// Instead of a precomputed frozen source `dcbr_source[i] = (neq-Δn_old)(1-e^{-dτ·em})`,
/// we pass the emission rates and equilibrium targets so DC/BR is solved implicitly
/// (backward Euler) inside the Newton iteration. This ensures DC/BR and Kompaneets
/// see the same evolving Δn, matching CosmoTherm's approach.
pub struct DcbrCoupling<'a> {
    /// DC/BR absorption rates: `R[i] = (K/x³)(e^{x_e}-1)`, capped at 1e8.
    pub emission_rates: &'a [f64],
    /// Equilibrium target: `neq[i] = n_pl(x/ρ_eq) - n_pl(x)`.
    pub n_eq_minus_n_pl: &'a [f64],
    /// Analytical derivative d(emission_rates)/d(ρ_eq) at the precomputed
    /// ρ_eq. Used in the bordered Newton c-vector so the Δn-row Jacobian
    /// reflects how the DC/BR absorption rate shifts when the Newton
    /// updates the ρ_e iterate. Dominant term:
    ///   d(em)/d(ρ_eq) = -(K/x³) · exp(x/ρ_eq) · x/ρ_eq².
    /// Pass `None` (empty slice) to retain the legacy Picard-in-ρ_e
    /// behaviour, which is correct to O(Δρ_e per step) but only linearly
    /// convergent at z ≳ 10⁶.
    pub dem_drho_eq: &'a [f64],
    /// Analytical derivative d(n_eq_minus_n_pl)/d(ρ_eq). Formula:
    ///   d(neq)/d(ρ_eq) = n_pl(x/ρ_eq)(1 + n_pl(x/ρ_eq)) · x/ρ_eq².
    pub dneq_drho_eq: &'a [f64],
    /// Integrated photon source over the step: S_i = source_rate(x_i, z_mid)·dt.
    ///
    /// When `Some`, the Newton residual picks up the `−S_i` term directly
    /// rather than relying on the caller to pre-add `S_i` to Δn_old. This
    /// avoids poisoning the Kompaneets CN "old" flux with the source (the
    /// pre-add approach effectively treats the source as injected at
    /// `t_old`, which over-Comptonises by roughly `O(dt · ∂K/∂t)` per step
    /// during a narrow injection window). `None` preserves the legacy
    /// pre-add caller code path.
    pub photon_source: Option<&'a [f64]>,
    /// Use Crank-Nicolson (instead of backward Euler) for DC/BR.
    /// Requires old-step DC/BR buffers in the workspace.
    pub cn_dcbr: bool,
}

/// In-place Kompaneets + DC/BR step using pre-allocated workspace.
///
/// Modifies `delta_n` from old values to new values.
/// Identical physics to `kompaneets_step_nonlinear_coupled` but avoids
/// per-step heap allocations.
///
/// DC/BR is handled via backward Euler within the Newton iteration:
/// the DC/BR residual `dτ × em × (neq - Δn_new)` and Jacobian `dτ × em`
/// are added to the Kompaneets system. Backward Euler is unconditionally
/// stable (amplification → 0 for stiff rates), avoiding the oscillation
/// that Crank-Nicolson would produce for the stiff DC/BR absorption at low x.
///
/// `max_dn_abs` is the current max|Δn| used for adaptive Newton tolerance.
/// Pass 0.0 for the tightest tolerance (equivalent to old fixed 1e-14).
///
/// Returns `(converged, rho_e, last_correction)`. **Note:** the third field
/// is the size of the last Newton *correction* `|δx|`, not the residual
/// `|F(x)|`. At convergence `|δx| < tol` by construction; if the Newton
/// loop exits via `max_newton_iter`, `last_correction` is the final step
/// the solver attempted, which only upper-bounds the residual for
/// contractive iterations. Treat it as a diagnostic, not a proof of
/// small residual.
pub fn kompaneets_step_coupled_inplace(
    grid: &FrequencyGrid,
    delta_n: &mut [f64],
    theta_e: f64,
    theta_z: f64,
    dtau: f64,
    dcbr: Option<&DcbrCoupling>,
    rho_coupling: Option<&RhoECoupling>,
    ws: &mut KompaneetsWorkspace,
    max_dn_abs: f64,
    max_newton_iter: usize,
) -> (bool, f64, f64) {
    let ng = grid.n;
    // θ_e for the CN "old" flux uses the step-start ρ_e when provided by
    // the caller (coupled mode). Without a `RhoECoupling` we assume T_e is
    // constant over the step and use the passed θ_e for both CN half-steps;
    // callers that evolve T_e via `update_temperatures` but do not enable the
    // bordered Newton should pass `rho_coupling = Some(...)` to preserve
    // time-centering (the `dh_drho`/`lambda_exp`/etc. fields can be zeroed).
    let theta_e_old = if let Some(rc) = rho_coupling {
        theta_z * rc.rho_e_old
    } else {
        theta_e
    };

    // Assert workspace sizes so the compiler can elide per-element bounds checks
    // in the hot inner loops. All workspace arrays are allocated to ng or ng-1
    // in KompaneetsWorkspace::new(), but LLVM can't prove this across function
    // boundaries without these hints.
    let n_half = ng - 1;
    assert!(delta_n.len() >= ng);
    assert!(ws.dn_old.len() >= ng);
    assert!(ws.k_old.len() >= ng);
    assert!(ws.rhs_buf.len() >= ng);
    assert!(ws.j_lower.len() >= ng);
    assert!(ws.j_diag.len() >= ng);
    assert!(ws.j_upper.len() >= ng);
    assert!(ws.c_vec.len() >= ng);
    assert!(ws.wem.len() >= ng);
    assert!(ws.inv_x2_dx_cell.len() >= ng);
    assert!(ws.inv_x2_dx_cell_geom.len() >= ng);
    assert!(ws.half_dtau_coeff.len() >= ng);
    assert!(ws.quad_weights_x3.len() >= ng);
    assert!(ws.v_buf.len() >= ng);
    assert!(ws.n_pl_half.len() >= n_half);
    assert!(ws.np1_half.len() >= n_half);
    assert!(ws.twonp1_half.len() >= n_half);
    assert!(ws.x4_half.len() >= n_half);
    assert!(ws.inv_dx.len() >= n_half);
    assert!(ws.x4_over_dx.len() >= n_half);

    // Debug-mode input validation: catch NaN/Inf and unphysical parameters
    // before they propagate through unsafe blocks. Zero cost in release.
    debug_assert!(theta_e.is_finite() && theta_e > 0.0, "theta_e={theta_e}");
    debug_assert!(theta_z.is_finite() && theta_z > 0.0, "theta_z={theta_z}");
    debug_assert!(dtau.is_finite() && dtau >= 0.0, "dtau={dtau}");
    debug_assert!(max_dn_abs.is_finite(), "max_dn_abs={max_dn_abs}");
    debug_assert!(ng >= 3, "grid too small: ng={ng}");

    // For the "old" part of CN, use the step-start ρ_e.
    // When coupled, rho_e_old is the pre-update value; otherwise use theta_e/theta_z.
    let phi_old = if let Some(rc) = rho_coupling {
        1.0 / rc.rho_e_old
    } else {
        theta_z / theta_e
    };

    let mut rho_e = theta_e / theta_z; // initial guess (from update_temperatures)
    let mut phi = 1.0 / rho_e;

    // Save old delta_n for CN formula
    ws.dn_old[..ng].copy_from_slice(&delta_n[..ng]);

    // Fill per-step coefficients with θ_e_old for the K_old precompute
    // below. Inside the Newton loop these are overwritten with θ_e
    // evaluated at the current ρ_e iterate, so K_new and its Jacobian stay
    // time-centred with the evolving electron temperature (genuine CN in θ_e,
    // not frozen-θ_e). In non-coupled mode rho_e is constant within a step,
    // so the refresh reproduces theta_e_old and is effectively a no-op.
    for i in 1..ng - 1 {
        ws.inv_x2_dx_cell[i] = theta_e_old * ws.inv_x2_dx_cell_geom[i];
        ws.half_dtau_coeff[i] = 0.5 * dtau * ws.inv_x2_dx_cell[i];
    }

    // Precompute K(delta_n_old, φ_old) for Crank-Nicolson
    // SAFETY: i ranges over 1..ng-1; all workspace arrays have length >= ng,
    // half-point arrays have length >= ng-1. Asserted above.
    for i in 1..ng - 1 {
        unsafe {
            let dn_old_im1 = *ws.dn_old.get_unchecked(i - 1);
            let dn_old_i = *ws.dn_old.get_unchecked(i);
            let dn_old_ip1 = *ws.dn_old.get_unchecked(i + 1);
            let dn_half_l = 0.5 * (dn_old_im1 + dn_old_i);
            let dn_half_r = 0.5 * (dn_old_i + dn_old_ip1);
            let ddn_dx_l = (dn_old_i - dn_old_im1) * *ws.inv_dx.get_unchecked(i - 1);
            let ddn_dx_r = (dn_old_ip1 - dn_old_i) * *ws.inv_dx.get_unchecked(i);

            let f_l = *ws.x4_half.get_unchecked(i - 1)
                * ((phi_old - 1.0) * *ws.np1_half.get_unchecked(i - 1)
                    + ddn_dx_l
                    + phi_old * *ws.twonp1_half.get_unchecked(i - 1) * dn_half_l
                    + phi_old * dn_half_l * dn_half_l);
            let f_r = *ws.x4_half.get_unchecked(i)
                * ((phi_old - 1.0) * *ws.np1_half.get_unchecked(i)
                    + ddn_dx_r
                    + phi_old * *ws.twonp1_half.get_unchecked(i) * dn_half_r
                    + phi_old * dn_half_r * dn_half_r);
            *ws.k_old.get_unchecked_mut(i) = *ws.inv_x2_dx_cell.get_unchecked(i) * (f_r - f_l);
        }
    }

    // Assert DC/BR slice lengths so bounds checks are elided in the inner loop.
    if let Some(dc) = dcbr {
        assert!(dc.emission_rates.len() >= ng);
        assert!(dc.n_eq_minus_n_pl.len() >= ng);
        assert!(dc.dem_drho_eq.len() >= ng);
        assert!(dc.dneq_drho_eq.len() >= ng);
        if let Some(ps) = dc.photon_source {
            assert!(ps.len() >= ng);
        }
        if dc.cn_dcbr {
            assert!(ws.dcbr_em_old.len() >= ng);
            assert!(ws.dcbr_neq_old.len() >= ng);
        }
    }

    // Hoist Option dispatch out of the inner loop. Extract DC/BR slice references
    // and mode flags once so the hot loop body has no per-point Option checks.
    // The empty slices are never indexed (guarded by has_dcbr / has_phot_src).
    static EMPTY: [f64; 0] = [];
    let (has_dcbr, use_cn, em_rates, neq_vals, dem_drho, dneq_drho, phot_src_vals): (
        bool,
        bool,
        &[f64],
        &[f64],
        &[f64],
        &[f64],
        &[f64],
    ) = if let Some(dc) = dcbr {
        (
            true,
            dc.cn_dcbr,
            dc.emission_rates,
            dc.n_eq_minus_n_pl,
            dc.dem_drho_eq,
            dc.dneq_drho_eq,
            dc.photon_source.unwrap_or(&EMPTY),
        )
    } else {
        (false, false, &EMPTY, &EMPTY, &EMPTY, &EMPTY, &EMPTY)
    };
    let has_phot_src = !phot_src_vals.is_empty();
    let has_rho_coupling = rho_coupling.is_some();

    // Precompute w_j × em_j once per step (only for the bordered solve).
    // em_rates and quad_weights_x3 are constant across Newton iterations;
    // caching this product eliminates ~2× max_newton_iter redundant
    // multiplies per grid point (h_dcbr pass + b'·u / b'·v pass).
    if has_dcbr && has_rho_coupling {
        for j in 0..ng {
            ws.wem[j] = ws.quad_weights_x3[j] * em_rates[j];
        }
    }

    // Newton iteration for the coupled system:
    //   Δn_new - Δn_old = dτ/2 (K_new(φ) + K_old(φ_old))  [CN for Kompaneets]
    //                    + dτ · em · (neq - Δn_new)          [BE for DC/BR]
    let mut converged = false;
    let mut last_max_delta: f64 = f64::NAN;
    for _newton in 0..max_newton_iter {
        // Note on θ_e time-centering (audit H5): in principle a fully time-
        // centred CN-in-θ_e would refresh `inv_x2_dx_cell[i]` and
        // `half_dtau_coeff[i]` with θ_e evaluated at the current ρ_e iterate
        // each Newton pass. We do not do this. Refreshing changes the
        // residual without updating `c_vec` to include the matching
        // ∂(inv_x2_dx_cell)/∂ρ_e contribution, which breaks the bordered
        // Newton Jacobian and can produce catastrophic non-convergence
        // (observed: |Δρ/ρ| ~ 1 for post-recombination scenarios). Instead,
        // the prefactor is held at θ_e_old for the whole step; φ = 1/ρ_e
        // inside the flux is still iterated and the Jacobian is consistent.
        // The time-centring error on θ_e is O(Δρ_e × θ_e) ~ 10⁻⁷ at z ~ 2×10⁶
        // with |Δρ_e| < 10⁻³ — well below the CN truncation floor. Restoring
        // a genuine CN in θ_e would require extending `c_vec` with the
        // ∂(inv_x2_dx_cell)/∂ρ_e contribution.

        // Boundary conditions: zero Kompaneets flux, but allow DC/BR relaxation.
        for &bi in &[0_usize, ng - 1] {
            let (dcbr_res, dcbr_jac) = if has_dcbr {
                let em = em_rates[bi];
                let neq = neq_vals[bi];
                if use_cn {
                    let em_old = ws.dcbr_em_old[bi];
                    let neq_old = ws.dcbr_neq_old[bi];
                    let old_part = 0.5 * dtau * em_old * (neq_old - ws.dn_old[bi]);
                    (
                        old_part + 0.5 * dtau * em * (neq - delta_n[bi]),
                        0.5 * dtau * em,
                    )
                } else {
                    (dtau * em * (neq - delta_n[bi]), dtau * em)
                }
            } else {
                (0.0, 0.0)
            };
            let phot_src = if has_phot_src { phot_src_vals[bi] } else { 0.0 };
            ws.rhs_buf[bi] = -(delta_n[bi] - ws.dn_old[bi] - dcbr_res - phot_src);
            ws.j_diag[bi] = 1.0 + dcbr_jac;
            ws.j_lower[bi] = 0.0;
            ws.j_upper[bi] = 0.0;
        }

        // SAFETY: i ranges over 1..ng-1; all workspace/delta_n arrays have length >= ng,
        // half-point arrays have length >= ng-1. DC/BR arrays (when has_dcbr) >= ng.
        // All asserted at function entry.
        for i in 1..ng - 1 {
            unsafe {
                let dn_im1 = *delta_n.get_unchecked(i - 1);
                let dn_i = *delta_n.get_unchecked(i);
                let dn_ip1 = *delta_n.get_unchecked(i + 1);
                let dn_half_l = 0.5 * (dn_im1 + dn_i);
                let dn_half_r = 0.5 * (dn_i + dn_ip1);

                // Compute K(delta_n, φ) at interior point i (CN "new" part)
                let ddn_dx_l = (dn_i - dn_im1) * *ws.inv_dx.get_unchecked(i - 1);
                let ddn_dx_r = (dn_ip1 - dn_i) * *ws.inv_dx.get_unchecked(i);
                let x4h_l = *ws.x4_half.get_unchecked(i - 1);
                let x4h_r = *ws.x4_half.get_unchecked(i);
                let f_l = x4h_l
                    * ((phi - 1.0) * *ws.np1_half.get_unchecked(i - 1)
                        + ddn_dx_l
                        + phi * *ws.twonp1_half.get_unchecked(i - 1) * dn_half_l
                        + phi * dn_half_l * dn_half_l);
                let f_r = x4h_r
                    * ((phi - 1.0) * *ws.np1_half.get_unchecked(i)
                        + ddn_dx_r
                        + phi * *ws.twonp1_half.get_unchecked(i) * dn_half_r
                        + phi * dn_half_r * dn_half_r);
                let inv_x2dc = *ws.inv_x2_dx_cell.get_unchecked(i);
                let k_i = inv_x2dc * (f_r - f_l);

                // DC/BR: backward Euler or Crank-Nicolson within the Newton iteration.
                //
                // `em_rates` and `neq_vals` are precomputed by the solver
                // at the step-start ρ_eq and held fixed across Newton
                // iterations. `dem_drho`/`dneq_drho` carry their analytical
                // derivatives w.r.t. ρ_eq; we use them below to close the
                // Δn-row Jacobian on ρ_e (c_vec). For CN mode the "old"
                // half is frozen — its ρ_eq derivative does not contribute.
                let (dcbr_residual, dcbr_jac, dcbr_crho) = if has_dcbr {
                    let em = *em_rates.get_unchecked(i);
                    let neq = *neq_vals.get_unchecked(i);
                    let dem = *dem_drho.get_unchecked(i);
                    let dneq = *dneq_drho.get_unchecked(i);
                    // d(dcbr_residual)/d(ρ_eq) from the "new" half, with BE
                    // weight 1 and CN weight 0.5.
                    let new_crho = -dtau * (dem * (neq - dn_i) + em * dneq);
                    if use_cn {
                        let em_old = *ws.dcbr_em_old.get_unchecked(i);
                        let neq_old = *ws.dcbr_neq_old.get_unchecked(i);
                        let old_part =
                            0.5 * dtau * em_old * (neq_old - *ws.dn_old.get_unchecked(i));
                        let new_res = 0.5 * dtau * em * (neq - dn_i);
                        (old_part + new_res, 0.5 * dtau * em, 0.5 * new_crho)
                    } else {
                        (dtau * em * (neq - dn_i), dtau * em, new_crho)
                    }
                } else {
                    (0.0, 0.0, 0.0)
                };

                // Residual: CN for Kompaneets + BE for DC/BR + integrated
                // source over the step. When `photon_source` is provided
                // through `DcbrCoupling`, S_i enters as a known constant
                // (no Jacobian contribution), and the Kompaneets CN "old"
                // flux is computed from the un-injected Δn_old — so the
                // source interacts with DC/BR absorption and Comptonisation
                // over the full step rather than being treated as an
                // impulse at t_old.
                let phot_src = if has_phot_src {
                    *phot_src_vals.get_unchecked(i)
                } else {
                    0.0
                };
                let residual_i = dn_i
                    - *ws.dn_old.get_unchecked(i)
                    - 0.5 * dtau * (k_i + *ws.k_old.get_unchecked(i))
                    - dcbr_residual
                    - phot_src;
                *ws.rhs_buf.get_unchecked_mut(i) = -residual_i;

                // Jacobian: CN (halved) for Kompaneets + BE diagonal for DC/BR
                let hdc = *ws.half_dtau_coeff.get_unchecked(i);
                let n_l = *ws.n_pl_half.get_unchecked(i - 1) + dn_half_l;
                let n_r = *ws.n_pl_half.get_unchecked(i) + dn_half_r;
                let a_l = *ws.x4_over_dx.get_unchecked(i - 1);
                let b_l = x4h_l * phi * (2.0 * n_l + 1.0) * 0.5;
                let a_r = *ws.x4_over_dx.get_unchecked(i);
                let b_r = x4h_r * phi * (2.0 * n_r + 1.0) * 0.5;

                *ws.j_lower.get_unchecked_mut(i) = -hdc * (a_l - b_l);
                *ws.j_diag.get_unchecked_mut(i) = 1.0 - hdc * (-a_r + b_r - a_l - b_l) + dcbr_jac;
                *ws.j_upper.get_unchecked_mut(i) = -hdc * (a_r + b_r);

                // Column vector c_i = dR_i/dρ_e for bordered system.
                // Two contributions:
                //   (a) Kompaneets flux: ∂F/∂ρ_e through the n_pl(1+n_pl)/ρ²
                //       piece (analytical; already in the Planck-subtracted
                //       split form used here).
                //   (b) DC/BR residual: ∂(dcbr_residual)/∂ρ_eq, carried in
                //       `dcbr_crho` above and plumbed via `dem_drho` /
                //       `dneq_drho`. Closing this piece restores formal
                //       quadratic Newton convergence in the ρ_e direction
                //       when DC/BR is strong (high z, photon-injection burst).
                if has_rho_coupling {
                    let inv_rho2 = 1.0 / (rho_e * rho_e);
                    let dfdr_l = -x4h_l * inv_rho2 * n_l * (1.0 + n_l);
                    let dfdr_r = -x4h_r * inv_rho2 * n_r * (1.0 + n_r);
                    let c_kompaneets = -0.5 * dtau * inv_x2dc * (dfdr_r - dfdr_l);
                    *ws.c_vec.get_unchecked_mut(i) = c_kompaneets + dcbr_crho;
                }
            }
        }

        if let Some(rc) = rho_coupling {
            // Bordered solve: (T, c; b', d) × (δΔn; δρ_e) = (r; r_ρ)
            ws.c_vec[0] = 0.0;
            ws.c_vec[ng - 1] = 0.0;

            // Compute H_dcbr = h_norm × Σ wem_j × (neq_j − Δn_j),
            // where wem_j = w_j × em_j was precomputed before the Newton loop.
            let mut h_dcbr = 0.0;
            if has_dcbr {
                for j in 0..ng {
                    h_dcbr += ws.wem[j] * (neq_vals[j] - delta_n[j]);
                }
                h_dcbr *= rc.h_norm;
            }

            // ρ_e residual: dρ_e/dτ = R[(source − H_dcbr) − ρ_e] − H·t_C·ρ_e
            // r_ρ = -(ρ_e − ρ_e_old) + dτ·{R·(source − h_dcbr − ρ_e) − H·t_C·ρ_e}
            let rhs_rho = -(rho_e - rc.rho_e_old)
                + dtau * (rc.r_compton * (rc.rho_source - h_dcbr - rho_e) - rc.lambda_exp * rho_e);

            // d = dR_ρ/dρ_e = 1 + dτ·(R·(1 + dH/dρ_e) + H·t_C)
            let d_rho = 1.0 + dtau * (rc.r_compton * (1.0 + rc.dh_drho) + rc.lambda_exp);

            // Step 1: T·u = r (standard Thomas, result in rhs_buf)
            thomas_solve_inplace(
                &ws.j_lower,
                &ws.j_diag,
                &ws.j_upper,
                &mut ws.rhs_buf,
                &mut ws.thomas_work,
            );

            // Step 2: T·v = c (Thomas on c_vec, result in v_buf)
            ws.v_buf[..ng].copy_from_slice(&ws.c_vec[..ng]);
            thomas_solve_inplace(
                &ws.j_lower,
                &ws.j_diag,
                &ws.j_upper,
                &mut ws.v_buf,
                &mut ws.thomas_work2,
            );

            // Step 3: b'·u and b'·v dot products.
            // b'_j = ∂R_ρ/∂Δn_j = −dτ · R · h_norm · wem_j.
            // Sign: h_dcbr = h_norm·Σ wem·(neq − Δn) ⇒ ∂h_dcbr/∂Δn_j = −h_norm·wem_j.
            // R_ρ = (ρ − ρ_old) − dτ·[R·(source − h_dcbr − ρ) − Λ·ρ], so
            // ∂R_ρ/∂Δn_j = −dτ·R·(−∂h_dcbr/∂Δn_j) = −dτ·R·h_norm·wem_j.
            // Pull the scalar (−dτ·R·h_norm) out of the loop and apply once
            // after the reductions — saves ng muls and keeps the inner loop
            // a clean fused-multiply-add pair that LLVM can vectorise.
            let mut bp_dot_u = 0.0;
            let mut bp_dot_v = 0.0;
            if has_dcbr {
                for j in 0..ng {
                    let wem_j = ws.wem[j];
                    bp_dot_u += wem_j * ws.rhs_buf[j];
                    bp_dot_v += wem_j * ws.v_buf[j];
                }
                let scale = -dtau * rc.r_compton * rc.h_norm;
                bp_dot_u *= scale;
                bp_dot_v *= scale;
            }

            // Step 4: δρ_e = (r_ρ − b'·u) / (d − b'·v)
            //
            // If the bordered-system determinant collapses the problem is
            // degenerate (no unique ρ_e correction). We signal this by
            // producing a NaN so the outer Newton loop breaks immediately
            // and the (false, _, NaN) return propagates non-convergence —
            // masking the failure as `delta_rho = 0` would silently report
            // convergence at max_delta below tol.
            let denom = d_rho - bp_dot_v;
            let delta_rho = if denom.abs() > 1e-30 {
                (rhs_rho - bp_dot_u) / denom
            } else {
                f64::NAN
            };

            // Step 5: δΔn = u − v·δρ_e
            let mut max_delta: f64 = 0.0;
            for j in 0..ng {
                let correction = ws.rhs_buf[j] - ws.v_buf[j] * delta_rho;
                delta_n[j] += correction;
                let d = correction.abs();
                if d > max_delta {
                    max_delta = d;
                }
            }
            rho_e += delta_rho;
            phi = 1.0 / rho_e;

            let rho_delta = delta_rho.abs();
            if rho_delta > max_delta {
                max_delta = rho_delta;
            }
            last_max_delta = max_delta;

            if !max_delta.is_finite() {
                break;
            }
            let tol = 1e-8 * max_dn_abs + 1e-14;
            if max_delta < tol {
                converged = true;
                break;
            }
        } else {
            // Standard tridiagonal solve (no ρ_e coupling)
            thomas_solve_inplace(
                &ws.j_lower,
                &ws.j_diag,
                &ws.j_upper,
                &mut ws.rhs_buf,
                &mut ws.thomas_work,
            );

            let mut max_delta: f64 = 0.0;
            for i in 0..ng {
                delta_n[i] += ws.rhs_buf[i];
                let d = ws.rhs_buf[i].abs();
                if d > max_delta {
                    max_delta = d;
                }
            }
            last_max_delta = max_delta;

            if !max_delta.is_finite() {
                break;
            }
            let tol = 1e-8 * max_dn_abs + 1e-14;
            if max_delta < tol {
                converged = true;
                break;
            }
        }
    }
    (converged, rho_e, last_max_delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::FrequencyGrid;

    #[test]
    fn test_kompaneets_preserves_planck() {
        // If Δn = 0 and T_e = T_z, Kompaneets should not generate distortion
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 500);
        let delta_n = vec![0.0; grid.n];
        let theta_e = 1e-6; // some temperature
        let theta_z = theta_e; // equilibrium

        let result = kompaneets_step(&grid, &delta_n, theta_e, theta_z, 0.01);

        let max_dn: f64 = result
            .iter()
            .map(|x| x.abs())
            .fold(0.0, |a, b| if b.is_nan() { f64::NAN } else { a.max(b) });
        assert!(
            max_dn < 1e-12,
            "Kompaneets should preserve Planck: max|Δn| = {max_dn}"
        );
    }

    #[test]
    fn test_kompaneets_photon_number_conservation() {
        // Kompaneets scattering conserves photon number: ∫x² dn/dτ dx = 0
        // Start with a small perturbation and evolve
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 1000);
        let theta_e = 1e-6;
        let theta_z = theta_e;

        // Small Gaussian perturbation centered at x=3
        let mut delta_n: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| 1e-4 * (-(x - 3.0).powi(2) / 0.5).exp())
            .collect();

        let n_before: f64 = (1..grid.n)
            .map(|i| {
                let dx = grid.dx[i - 1];
                let x_mid = grid.x_half[i - 1];
                let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
                x_mid * x_mid * dn_mid * dx
            })
            .sum();

        // Evolve for several steps
        for _ in 0..10 {
            delta_n = kompaneets_step(&grid, &delta_n, theta_e, theta_z, 0.001);
        }

        let n_after: f64 = (1..grid.n)
            .map(|i| {
                let dx = grid.dx[i - 1];
                let x_mid = grid.x_half[i - 1];
                let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
                x_mid * x_mid * dn_mid * dx
            })
            .sum();

        let rel_change = (n_after - n_before).abs() / n_before.abs().max(1e-20);
        // Kompaneets is a number-conserving operator — boundary leakage
        // should be tiny on a grid extending to x_max=30
        assert!(
            rel_change < 1e-4,
            "Photon number not conserved: ΔN/N = {rel_change:.2e} (threshold 1e-4), \
             before={n_before:.4e}, after={n_after:.4e}"
        );
    }

    #[test]
    fn test_kompaneets_energy_conservation() {
        // Kompaneets scattering with T_e = T_z should conserve energy: ∫x³ Δn dx = 0.
        // Energy is only redistributed (not created) when there's no T_e−T_z offset.
        // Start with a Gaussian perturbation and evolve with T_e = T_z.
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 1000);
        let theta = 1e-6;

        // Gaussian perturbation at x=5
        let mut delta_n: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| 1e-4 * (-(x - 5.0).powi(2) / 2.0).exp())
            .collect();

        let energy_before: f64 = (1..grid.n)
            .map(|i| {
                let dx = grid.dx[i - 1];
                let x_mid = grid.x_half[i - 1];
                let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
                x_mid.powi(3) * dn_mid * dx
            })
            .sum();

        for _ in 0..10 {
            delta_n = kompaneets_step(&grid, &delta_n, theta, theta, 0.001);
        }

        let energy_after: f64 = (1..grid.n)
            .map(|i| {
                let dx = grid.dx[i - 1];
                let x_mid = grid.x_half[i - 1];
                let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
                x_mid.powi(3) * dn_mid * dx
            })
            .sum();

        // With T_e = T_z, pure Kompaneets conserves energy to first order.
        // The nonlinear Δn² term introduces O(Δn²) energy change, but for
        // Δn ~ 1e-4 the correction is O(1e-8), far below 1e-4.
        let rel_change = (energy_after - energy_before).abs() / energy_before.abs().max(1e-20);
        assert!(
            rel_change < 1e-4,
            "Kompaneets energy not conserved with T_e=T_z: ΔE/E = {rel_change:.2e}, \
             before={energy_before:.4e}, after={energy_after:.4e}"
        );
    }

    /// Quantitative Kompaneets check in the y-regime: injecting energy via
    /// T_e > T_z for one Thomson time should produce Δρ/ρ ≈ 4·y·G₃ where
    /// y = (θ_e − θ_z)·dτ is the standard y-parameter. This catches
    /// order-of-magnitude errors in the flux split, grid geometry, or time
    /// centring that the sign-only `test_kompaneets_te_gt_tz_positive_drho_all_solvers`
    /// would miss (audit M2).
    #[test]
    fn test_kompaneets_y_distortion_magnitude() {
        // Deep y-era, linear regime: small δρ_e, small dτ, no DC/BR, no T_e
        // evolution. δρ_e = 1e-3 keeps us well inside the Taylor branch where
        // the analytical Δρ/ρ = 4y expression is accurate.
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 2000);
        let delta_n = vec![0.0; grid.n];
        let theta_z = 1e-6;
        let delta_rho_e = 1e-3_f64;
        let theta_e = theta_z * (1.0 + delta_rho_e);
        let dtau = 1.0_f64;

        // Expected: pure y-distortion with y = (θ_e − θ_z)·dτ and Δρ/ρ = 4y.
        let y_expected = (theta_e - theta_z) * dtau;
        let drho_expected = 4.0 * y_expected;

        // Solve.
        let result = kompaneets_step_nonlinear(&grid, &delta_n, theta_e, theta_z, dtau);
        let drho = crate::spectrum::delta_rho_over_rho(&grid.x, &result);

        let rel_err = (drho - drho_expected).abs() / drho_expected.abs();
        eprintln!(
            "Kompaneets y-magnitude: drho={drho:.4e}, expected={drho_expected:.4e}, \
             rel_err={rel_err:.2e}"
        );
        // 5% tolerance allows for finite grid resolution, the small Δn² term,
        // and truncation in the integration stencil. A factor-of-10⁶ bug in
        // the flux split (the class this test is meant to catch) would show
        // up as rel_err ≫ 1.
        assert!(
            rel_err < 0.05,
            "Kompaneets y-magnitude off: drho={drho:.4e}, expected={drho_expected:.4e}, \
             rel_err={rel_err:.2e}"
        );
    }

    /// When T_e > T_z, all Kompaneets solver variants must produce Δρ/ρ > 0
    /// (energy flows from electrons to photons via upscattering).
    /// Tests CN (kompaneets_step), backward Euler (kompaneets_tridiagonal + Thomas),
    /// and nonlinear (kompaneets_step_nonlinear) in a single test.
    #[test]
    fn test_kompaneets_te_gt_tz_positive_drho_all_solvers() {
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 2000);
        let delta_n = vec![0.0; grid.n];
        let theta_z = 1e-4;
        let theta_e = theta_z * 1.001;

        // 1. Crank-Nicolson (kompaneets_step)
        let result_cn = kompaneets_step(&grid, &delta_n, theta_e, theta_z, 0.1);
        let drho_cn = crate::spectrum::delta_rho_over_rho(&grid.x, &result_cn);
        assert!(drho_cn > 0.0, "CN: Δρ/ρ > 0 expected, got {drho_cn:.4e}");

        // 2. Backward Euler (kompaneets_tridiagonal + Thomas solve)
        let (lower, diag, upper, source) = kompaneets_tridiagonal(&grid, theta_e, theta_z);
        let dtau = 1.0;
        let n = grid.n;
        let mut rhs: Vec<f64> = (0..n).map(|i| dtau * source[i]).collect();
        rhs[0] = 0.0;
        rhs[n - 1] = 0.0;
        let lhs_lower: Vec<f64> = lower.iter().map(|&a| -dtau * a).collect();
        let lhs_diag: Vec<f64> = diag
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if i == 0 || i == n - 1 {
                    1.0
                } else {
                    1.0 - dtau * b
                }
            })
            .collect();
        let lhs_upper: Vec<f64> = upper.iter().map(|&c| -dtau * c).collect();
        let result_be = thomas_solve(&lhs_lower, &lhs_diag, &lhs_upper, &mut rhs);
        let drho_be = crate::spectrum::delta_rho_over_rho(&grid.x, &result_be);
        assert!(drho_be > 0.0, "BE: Δρ/ρ > 0 expected, got {drho_be:.4e}");

        // 3. Nonlinear solver (kompaneets_step_nonlinear)
        let result_nl = kompaneets_step_nonlinear(&grid, &delta_n, theta_e, theta_z, 1.0);
        let drho_nl = crate::spectrum::delta_rho_over_rho(&grid.x, &result_nl);
        assert!(drho_nl > 0.0, "NL: Δρ/ρ > 0 expected, got {drho_nl:.4e}");

        eprintln!(
            "Source sign (all solvers): CN={drho_cn:.4e}, BE={drho_be:.4e}, NL={drho_nl:.4e}"
        );
    }

    /// Guards the analytic Planck cancellation in `kompaneets_rhs` (CLAUDE.md
    /// pitfall #1). The flux is written as
    ///   F = x⁴[(φ−1) n_pl(1+n_pl) + dΔn/dx + φ(2n_pl+1)Δn + φΔn²]
    /// so that with Δn = 0 and T_e = T_z (⇒ φ = 1), every term is identically
    /// zero BEFORE any finite differences touch n_pl. A naive flux that kept
    /// dn_pl/dx and +φ n_pl(1+n_pl) as separate pieces would leak O(dx²) ~ 10⁻⁴
    /// per point after the divergence, amplified by θ_e/x² at small x —
    /// ~1000× the physical y-signal ~10⁻⁵.
    ///
    /// Two probes:
    ///   (a) Δn = 0, T_e = T_z  →  rhs = 0 to machine precision.
    ///   (b) Δn = 0, T_e = T_z·(1+ε), ε = 1e-8  →  rhs scales linearly with ε
    ///       (the (φ−1) source is the ONLY nonzero piece).
    #[test]
    fn test_kompaneets_rhs_planck_cancellation() {
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 500);
        let delta_n = vec![0.0; grid.n];
        let theta = 1e-6;

        // (a) Exact cancellation at T_e = T_z.
        let rhs_eq = kompaneets_rhs(&grid, &delta_n, theta, theta);
        let max_abs_eq: f64 = rhs_eq
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, |a, b| a.max(b.abs()));
        assert!(
            max_abs_eq < 1e-20,
            "Planck + T_e=T_z must give rhs=0 to machine precision: max|rhs| = {max_abs_eq:.2e}. \
             Any value > ~1e-15 means the flux is using a finite-difference form of dn_pl/dx \
             instead of the analytic identity (CLAUDE.md pitfall #1)."
        );

        // (b) Linearity in (φ-1): halving ε halves max|rhs| to relative 1e-6.
        // If an O(dx²) Planck leak were present, rhs wouldn't scale with ε and
        // this ratio would stay near 1.
        let eps_big: f64 = 1e-6;
        let eps_small: f64 = 1e-8;
        let rhs_big = kompaneets_rhs(&grid, &delta_n, theta, theta * (1.0 + eps_big));
        let rhs_small = kompaneets_rhs(&grid, &delta_n, theta, theta * (1.0 + eps_small));
        let max_big: f64 = rhs_big
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, |a, b| a.max(b.abs()));
        let max_small: f64 = rhs_small
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, |a, b| a.max(b.abs()));
        let expected_ratio = eps_big / eps_small; // 100
        let actual_ratio = max_big / max_small;
        assert!(
            (actual_ratio / expected_ratio - 1.0).abs() < 1e-3,
            "rhs must scale ∝ (φ-1): max_big/max_small = {actual_ratio:.4} vs expected {expected_ratio:.4}. \
             Deviation > 0.1% means the RHS has a (φ-1)-independent residual (Planck leakage)."
        );
    }

    #[test]
    fn test_nonlinear_preserves_planck() {
        // Nonlinear solver on Δn should preserve zero when T_e = T_z
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 500);
        let delta_n = vec![0.0; grid.n];
        let theta = 1e-6;

        let result = kompaneets_step_nonlinear(&grid, &delta_n, theta, theta, 0.01);

        let max_err: f64 = result
            .iter()
            .map(|x| x.abs())
            .fold(0.0, |a, b| if b.is_nan() { f64::NAN } else { a.max(b) });
        eprintln!("Nonlinear preserves Planck: max|Δn| = {max_err:.4e}");
        assert!(max_err < 1e-12, "max|Δn| = {max_err}");
    }

    #[test]
    fn test_thomas_solve_inplace_2x2() {
        // [[2, -1], [-1, 2]] x = [1, 1] → x = [1, 1]
        let lower = vec![0.0, -1.0];
        let diag = vec![2.0, 2.0];
        let upper = vec![-1.0, 0.0];
        let mut rhs = vec![1.0, 1.0];
        let mut work = vec![0.0; 2];
        thomas_solve_inplace(&lower, &diag, &upper, &mut rhs, &mut work);
        assert!(
            (rhs[0] - 1.0).abs() < 1e-14,
            "x[0] = {}, expected 1.0",
            rhs[0]
        );
        assert!(
            (rhs[1] - 1.0).abs() < 1e-14,
            "x[1] = {}, expected 1.0",
            rhs[1]
        );
    }

    #[test]
    fn test_thomas_solve_inplace_3x3() {
        // Classic tridiagonal: [2,-1,0; -1,2,-1; 0,-1,2] x = [1, 0, 1]
        // Known solution: x = [1, 1, 1]
        let lower = vec![0.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0, 0.0];
        let mut rhs = vec![1.0, 0.0, 1.0];
        let mut work = vec![0.0; 3];
        thomas_solve_inplace(&lower, &diag, &upper, &mut rhs, &mut work);
        assert!(
            (rhs[0] - 1.0).abs() < 1e-14,
            "x[0] = {}, expected 1.0",
            rhs[0]
        );
        assert!(
            (rhs[1] - 1.0).abs() < 1e-14,
            "x[1] = {}, expected 1.0",
            rhs[1]
        );
        assert!(
            (rhs[2] - 1.0).abs() < 1e-14,
            "x[2] = {}, expected 1.0",
            rhs[2]
        );
    }

    #[test]
    fn test_thomas_solve_inplace_identity() {
        // Diagonal matrix (no off-diags): solution = rhs / diag
        let lower = vec![0.0, 0.0, 0.0, 0.0];
        let diag = vec![3.0, 5.0, 2.0, 7.0];
        let upper = vec![0.0, 0.0, 0.0, 0.0];
        let mut rhs = vec![9.0, 10.0, 6.0, 21.0];
        let mut work = vec![0.0; 4];
        thomas_solve_inplace(&lower, &diag, &upper, &mut rhs, &mut work);
        let expected = [3.0, 2.0, 3.0, 3.0];
        for i in 0..4 {
            assert!(
                (rhs[i] - expected[i]).abs() < 1e-14,
                "x[{i}] = {}, expected {}",
                rhs[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_coupled_inplace_preserves_planck() {
        // The production solver kompaneets_step_coupled_inplace should
        // preserve a Planck spectrum when T_e = T_z and no DC/BR coupling.
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 500);
        let mut delta_n = vec![0.0; grid.n];
        let theta = 1e-6;
        let dtau = 0.01;
        let mut ws = KompaneetsWorkspace::new(&grid);

        // Returns (converged, rho_e, last_max_delta)
        let (converged, rho_e, last_delta) = kompaneets_step_coupled_inplace(
            &grid,
            &mut delta_n,
            theta,
            theta,
            dtau,
            None,
            None,
            &mut ws,
            0.0,
            20,
        );

        assert!(converged, "Newton should converge for zero distortion");
        // rho_e should be 1.0 (T_e = T_z)
        assert!(
            (rho_e - 1.0).abs() < 1e-10,
            "rho_e should be 1.0 at equilibrium: {rho_e}"
        );
        // last Newton correction should be tiny
        assert!(
            last_delta < 1e-12,
            "Newton correction should be tiny: {last_delta:.4e}"
        );

        let max_dn: f64 = delta_n
            .iter()
            .map(|x| x.abs())
            .fold(0.0, |a, b| if b.is_nan() { f64::NAN } else { a.max(b) });
        assert!(
            max_dn < 1e-14,
            "Should preserve Planck: max|Δn| = {max_dn:.4e}"
        );
    }

    #[test]
    fn test_coupled_inplace_with_dcbr() {
        // Test the coupled solver with DC/BR coupling active.
        // With T_e = T_z and Δn = 0, DC/BR are at detailed balance.
        let grid = FrequencyGrid::log_uniform(1e-3, 30.0, 500);
        let mut delta_n = vec![0.0; grid.n];
        let theta = 1e-4;
        let dtau = 0.01;
        let mut ws = KompaneetsWorkspace::new(&grid);

        // At equilibrium (ρ_e=1), n_eq = 0 and emission rates are finite but irrelevant
        let emission_rates: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| {
                let k_dc = crate::double_compton::dc_emission_coefficient(x, theta);
                let x_e: f64 = 1.0;
                (k_dc / x.powi(3)) * (x_e.exp() - 1.0)
            })
            .collect();
        let n_eq: Vec<f64> = vec![0.0; grid.n];

        let dem = vec![0.0; grid.n];
        let dneq = vec![0.0; grid.n];
        let dcbr = DcbrCoupling {
            emission_rates: &emission_rates,
            n_eq_minus_n_pl: &n_eq,
            dem_drho_eq: &dem,
            dneq_drho_eq: &dneq,
            photon_source: None,
            cn_dcbr: false,
        };

        let (converged, rho_e, last_delta) = kompaneets_step_coupled_inplace(
            &grid,
            &mut delta_n,
            theta,
            theta,
            dtau,
            Some(&dcbr),
            None,
            &mut ws,
            0.0,
            20,
        );

        assert!(
            converged,
            "Newton should converge with DC/BR at equilibrium"
        );
        assert!((rho_e - 1.0).abs() < 1e-10, "rho_e should be 1.0: {rho_e}");
        assert!(
            last_delta < 1e-10,
            "Newton correction should be small: {last_delta:.4e}"
        );
    }

    /// Verify that a small temperature perturbation produces a Y_SZ spectral shape.
    ///
    /// For T_e slightly > T_z, the Kompaneets equation produces Δn ∝ Y_SZ(x).
    /// This is the defining property of the y-distortion. The Pearson correlation
    /// between the resulting Δn and Y_SZ(x) should be > 0.95.
    #[test]
    fn test_kompaneets_yields_ysz_shape() {
        let grid = FrequencyGrid::log_uniform(0.1, 25.0, 500);
        let delta_n = vec![0.0; grid.n];
        let theta_z = 1e-6;
        let theta_e = 1.001 * theta_z; // small perturbation: T_e = 1.001 T_z

        // Run one small step
        let result = kompaneets_step(&grid, &delta_n, theta_e, theta_z, 0.1);

        // Compute Y_SZ(x) at grid points
        let y_shape: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| crate::spectrum::y_shape(x))
            .collect();

        // Pearson correlation between result and Y_SZ
        let n = result.len();
        let mean_r: f64 = result.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y_shape.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_r = 0.0;
        let mut var_y = 0.0;
        for i in 0..n {
            let dr = result[i] - mean_r;
            let dy = y_shape[i] - mean_y;
            cov += dr * dy;
            var_r += dr * dr;
            var_y += dy * dy;
        }
        let corr = cov / (var_r.sqrt() * var_y.sqrt());
        eprintln!("Kompaneets Y_SZ correlation: {corr:.6}");

        assert!(
            corr > 0.95,
            "Kompaneets step with T_e > T_z should produce Y_SZ shape: correlation = {corr:.4}"
        );
    }
}
