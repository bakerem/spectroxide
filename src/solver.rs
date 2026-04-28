//! Main PDE solver for the cosmological thermalization problem.
//!
//! Key physics: energy injection → heats electrons (T_e > T_z) →
//! Kompaneets source drives y-type distortion → DC/BR create photons
//! to convert y → μ and approach Bose-Einstein equilibrium.
//!
//! The Kompaneets equation is split into:
//! 1. Conservative redistribution at T_e^eq (no net energy change)
//! 2. Injection source ∝ (ρ_e - ρ_eq) that adds energy from injection
//!
//! This ensures energy conservation: only injection adds energy.
//!
//! References:
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294

use crate::bremsstrahlung::{br_emission_coefficient_fast_preln, br_precompute};
use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::double_compton::{dc_high_freq_suppression, dc_prefactor};
use crate::electron_temp::ElectronTemperature;
use crate::energy_injection::InjectionScenario;
use crate::grid::{FrequencyGrid, GridConfig};
use crate::kompaneets::{KompaneetsWorkspace, kompaneets_step_coupled_inplace};
use crate::recombination::RecombinationHistory;
use crate::spectrum::planck;

/// Tunable solver parameters.
///
/// Defaults are production-quality and match the values used for all paper
/// runs. The fields that most users will want to change are [`Self::z_start`],
/// [`Self::z_end`], and [`Self::dtau_max`].
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Upper redshift at which integration begins (solver evolves from
    /// `z_start` down to `z_end`). Must satisfy `z_start > z_end`.
    pub z_start: f64,
    /// Lower redshift at which integration ends.
    pub z_end: f64,
    /// Maximum fractional change in ln(1+z) per step (limits step size for
    /// slow-varying sources). Default 0.02. Historical value 0.005 was
    /// calibrated for photon-injection bursts; for smooth (continuous) heat
    /// injection scenarios 0.02 gives the same μ/y to 4 significant figures
    /// with 5–6× fewer steps, since `dtau_max` takes over as the binding
    /// constraint at z ≲ 1.5×10⁶.
    pub dy_max: f64,
    /// Minimum allowed step size in z; smaller values trigger a warning.
    /// Default 1e-6.
    pub dz_min: f64,
    /// Maximum Compton optical depth per step. With exact exponential DC/BR
    /// (unconditionally stable), this primarily limits Kompaneets CN accuracy.
    /// Default 10.0 matches the CLI default (the value used for all paper
    /// runs). Raise to ~50 for exploratory runs where modest accuracy loss
    /// is acceptable.
    pub dtau_max: f64,
    /// Minimum redshift for number-conserving T-shift subtraction.
    /// Default 5e4: only subtract at z > nc_z_min where DC/BR is significant.
    /// Set to 0.0 to subtract at all redshifts (CosmoTherm-like behavior).
    pub nc_z_min: f64,
    /// Maximum dtau per step during active photon source injection.
    /// Limits the Kompaneets Δn² nonlinearity at the injection spike,
    /// which causes timestep-dependent wake formation at intermediate x.
    /// Default 1.0 (production quality, ~1% from converged).
    /// Set to 10.0 for exploratory/fast runs (~3% from converged).
    pub dtau_max_photon_source: f64,
    /// Maximum number of Newton iterations per Kompaneets step.
    /// Default 10. Increase for extreme injection parameters.
    pub max_newton_iter: usize,
    /// Use Crank-Nicolson (instead of backward Euler) for DC/BR.
    /// Second-order in time, matching CosmoTherm's scheme. May be less
    /// stable at very low x where DC/BR rates diverge.
    pub cn_dcbr: bool,
}

impl SolverConfig {
    /// Validate solver configuration parameters.
    ///
    /// Returns `Err` with a descriptive message if any parameter would cause
    /// the solver to fail or produce meaningless results.
    pub fn validate(&self) -> Result<(), String> {
        if !self.z_start.is_finite() || self.z_start <= 0.0 {
            return Err(format!("z_start must be positive, got {}", self.z_start));
        }
        if !self.z_end.is_finite() || self.z_end < 0.0 {
            return Err(format!("z_end must be non-negative, got {}", self.z_end));
        }
        if self.z_start <= self.z_end {
            return Err(format!(
                "z_start must be > z_end (solver evolves backwards), got z_start={}, z_end={}",
                self.z_start, self.z_end
            ));
        }
        if !self.dy_max.is_finite() || self.dy_max <= 0.0 {
            return Err(format!("dy_max must be positive, got {}", self.dy_max));
        }
        // Guardrail on the upper end: dy_max controls the fractional change
        // in ln(1+z) per step, and the adaptive-step derivation assumes this
        // is small. Default is 0.02; at 0.1 the linearization begins to fail
        // and step-count savings are illusory since dtau_max takes over.
        if self.dy_max > 0.1 {
            return Err(format!(
                "dy_max={} exceeds safe limit 0.1. Large dy_max invalidates the \
                 adaptive-stepping linearization; use a smaller value (default 0.02).",
                self.dy_max
            ));
        }
        if !self.dz_min.is_finite() || self.dz_min <= 0.0 {
            return Err(format!("dz_min must be positive, got {}", self.dz_min));
        }
        if !self.dtau_max.is_finite() || self.dtau_max <= 0.0 {
            return Err(format!("dtau_max must be positive, got {}", self.dtau_max));
        }
        if self.max_newton_iter < 2 {
            return Err(format!(
                "max_newton_iter must be >= 2, got {}",
                self.max_newton_iter
            ));
        }
        // Fokker-Planck validity: theta_e ~ 4.6e-10 * (1+z), so z > 5e6 gives
        // theta_e > 2.3e-3 where O(theta_e^2) corrections become significant.
        // At z > ~1e7 the Kompaneets equation is qualitatively wrong.
        if self.z_start > 1.0e7 {
            return Err(format!(
                "z_start={:.1e} implies theta_e ~ {:.1e}; Kompaneets Fokker-Planck \
                 approximation is invalid for theta_e > ~0.005. Max z_start ~ 1e7.",
                self.z_start,
                4.6e-10 * (1.0 + self.z_start)
            ));
        }
        Ok(())
    }

    /// Collect non-fatal validation warnings (e.g. regimes where the
    /// Kompaneets Fokker-Planck approximation begins to show O(θ_e²)
    /// corrections but is not outright invalid).
    pub(crate) fn soft_warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.z_start > 5.0e6 {
            out.push(format!(
                "z_start={:.1e} implies theta_e ~ {:.1e}. Kompaneets Fokker-Planck has \
                 O(theta_e^2) corrections that grow above z ~ 5e6. Results may have ~1% \
                 systematic errors.",
                self.z_start,
                4.6e-10 * (1.0 + self.z_start)
            ));
        }
        if self.dy_max > 0.05 {
            out.push(format!(
                "dy_max={:.3} is well above the default 0.02; step-size linearization \
                 errors grow rapidly past ~0.05. Validate against a smaller dy_max \
                 before trusting quantitative results.",
                self.dy_max
            ));
        }
        out
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            z_start: 3.0e6,
            z_end: 1.0,
            dy_max: 0.02,
            dz_min: 1e-6,
            dtau_max: 10.0,
            nc_z_min: 5.0e4,
            dtau_max_photon_source: 1.0,
            max_newton_iter: 10,
            cn_dcbr: false,
        }
    }
}

/// State of the photon distribution at a single redshift.
///
/// Returned by [`ThermalizationSolver::run_with_snapshots`]. The distortion
/// amplitudes (`mu`, `y`, accumulated `delta_t`) are extracted from `delta_n`
/// by a joint least-squares decomposition; see [`crate::distortion`].
#[derive(Debug, Clone)]
pub struct SolverSnapshot {
    /// Redshift of this snapshot.
    pub z: f64,
    /// Photon occupation-number perturbation `Δn(x) = n(x) − n_Planck(x)`
    /// on the solver's frequency grid.
    pub delta_n: Vec<f64>,
    /// Electron-to-photon temperature ratio `ρ_e = T_e / T_z` at this redshift.
    pub rho_e: f64,
    /// Bose-Einstein chemical-potential-like distortion amplitude μ.
    pub mu: f64,
    /// Compton y-parameter distortion amplitude.
    pub y: f64,
    /// Fractional energy injected by the active scenario up to this
    /// redshift, `Δρ / ρ`.
    pub delta_rho_over_rho: f64,
    /// Cumulative temperature shift `ΔT/T` subtracted by the number-conserving
    /// machinery (non-zero only when `number_conserving` is enabled).
    pub accumulated_delta_t: f64,
}

impl SolverSnapshot {
    /// Brightness-temperature deviation `T(x)/T_CMB − 1` on the given grid.
    pub fn brightness_temp(&self, x_grid: &[f64]) -> Vec<f64> {
        x_grid
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let n_total = planck(x) + self.delta_n[i];
                if n_total <= 0.0 || !n_total.is_finite() {
                    return 0.0;
                }
                x / (1.0 + 1.0 / n_total).ln() - 1.0
            })
            .collect()
    }
}

/// Cached backward-Euler ODE coefficients for the ρ_e equation.
///
/// Computed once per step in `update_temperatures()` and consumed by:
/// 1. The DC/BR emission rate computation (needs ρ_dcbr corrected for injection).
/// 2. The bordered Newton solver (couples ρ_e into the Kompaneets iteration).
#[derive(Debug, Clone, Copy)]
struct RhoECache {
    /// ρ_e at the start of the step (before Newton iteration).
    rho_e_old: f64,
    /// Compton coupling coefficient R = t_C / dtau × (some prefactor).
    r_compton: f64,
    /// RHS source: ρ_eq + δρ_inj (unscaled; R applied at point of use).
    rho_source: f64,
    /// H × t_C (Hubble expansion coupling).
    lambda_htc: f64,
    /// dH/dρ_e: derivative of DC+BR heating integral w.r.t. ρ_e.
    dh_drho: f64,
    /// Compton optical depth for this step (used for the DC/BR injection correction).
    dtau: f64,
}

/// Diagnostic counters and extrema tracked during solver evolution.
///
/// Grouped into a sub-struct to keep `ThermalizationSolver` focused on
/// physics state. Reset by `ThermalizationSolver::reset()`.
#[derive(Debug, Clone, Default)]
pub struct SolverDiagnostics {
    /// Number of times rho_e was clamped.
    /// Non-zero values indicate injection energy may be silently discarded.
    pub rho_e_clamped: usize,
    /// Number of times Newton iteration exhausted max_newton_iter
    /// without converging. Non-zero values indicate the solver may need more
    /// iterations or smaller step sizes.
    pub newton_exhausted: usize,
    /// Maximum uncapped emission rate encountered (NaN excluded).
    pub max_emission_rate: f64,
    /// Whether any NaN emission rate was encountered.
    pub nan_emission_detected: bool,
    /// Warning messages collected during solver evolution.
    /// Replaces eprintln! in library code for structured diagnostics.
    pub warnings: Vec<String>,
}

/// Full PDE solver for CMB spectral distortions.
///
/// Evolves the photon occupation number perturbation Δn(x, z) from `z_start`
/// down to `z_end` using a coupled implicit scheme:
/// - Crank-Nicolson for Kompaneets (frequency redistribution)
/// - Backward Euler for DC/BR emission (photon-number changing)
/// - Newton iteration coupling all three simultaneously
///
/// # Lifecycle
///
/// ```text
/// ThermalizationSolver::new(cosmo, grid_config)   // or ::builder(...)
///   → set_injection(scenario)                      // optional
///   → run_with_snapshots(&[z1, z2, ...])           // evolve the PDE
///   → snapshots[i].{mu, y, delta_t, ...}           // read results
/// ```
///
/// The solver can be re-run after calling `reset()`, which clears Δn and
/// diagnostic counters but keeps the cosmology and grid.
///
/// # Energy injection
///
/// Call `set_injection()` before running. Without an injection, Δn stays
/// near zero (only adiabatic cooling acts). Multiple runs with different
/// injections require `reset()` between runs.
///
/// # Configuration
///
/// See [`SolverConfig`] for timestep and physics options, and [`GridConfig`]
/// for frequency grid options. The builder API (`ThermalizationSolver::builder`)
/// provides a fluent interface for all configuration.
///
/// # Field visibility (unstable API)
///
/// Many fields are currently `pub` for use by examples, tests, and diagnostic
/// tooling. These are **not part of the stable public API** and may become
/// `pub(crate)` in a future release. Mutating state fields (`delta_n`, `z`,
/// `electron_temp`, `snapshots`, `step_count`, `accumulated_delta_t`) between
/// `run_with_snapshots` calls can break Newton-solver invariants; prefer the
/// builder and `set_injection` / `reset` methods for all changes of consequence.
pub struct ThermalizationSolver {
    /// Timestepping and physics-toggle configuration.
    pub config: SolverConfig,
    /// Background cosmology.
    pub cosmo: Cosmology,
    /// Frequency grid (non-uniform in x).
    pub grid: FrequencyGrid,
    /// Current photon occupation-number perturbation `Δn(x)` on the grid.
    pub delta_n: Vec<f64>,
    /// Electron temperature state `ρ_e = T_e/T_z` (perturbative
    /// quasi-stationary solve).
    pub electron_temp: ElectronTemperature,
    /// Current redshift of the integration.
    pub z: f64,
    /// Active injection scenario, if any. Set via [`Self::set_injection`].
    pub injection: Option<InjectionScenario>,
    /// Snapshots collected during the last run. Populated by
    /// [`Self::run_with_snapshots`] / [`Self::run`].
    pub snapshots: Vec<SolverSnapshot>,
    /// Number of timesteps taken in the current run.
    pub step_count: usize,
    /// Compton equilibrium ρ_eq = I₄/(4G₃) from the photon spectrum
    rho_eq: f64,
    /// Cached recombination history for fast X_e(z) lookups
    recomb: RecombinationHistory,
    /// Initial photon perturbation Δn(x) to use instead of zeros.
    /// Consumed (taken) by run_with_snapshots on first call.
    initial_delta_n: Option<Vec<f64>>,
    /// If true, disable DC/BR processes (Kompaneets only). For diagnostics.
    pub disable_dcbr: bool,
    /// If true (default), couple DC/BR into the Kompaneets Newton iteration
    /// instead of operator splitting. Uses IMEX: Crank-Nicolson for Kompaneets
    /// + backward Euler for DC/BR, solved simultaneously. More physically
    /// consistent than operator splitting, especially at z > 2×10⁶.
    pub coupled_dcbr: bool,
    /// Pre-allocated work buffer for DC/BR emission rates (per step)
    emission_rates: Vec<f64>,
    /// Pre-allocated work buffer for DC/BR equilibrium minus Planck
    n_eq_minus_n_pl: Vec<f64>,
    /// d(emission_rates)/d(ρ_eq), analytical. Used by the bordered Newton
    /// c-vector to close the Δn-row Jacobian on ρ_e (otherwise the solve
    /// is only linearly convergent in the ρ_e direction when DC/BR is
    /// strong, e.g. at z ≳ 10⁶ or during a photon-injection burst).
    dem_drho_eq: Vec<f64>,
    /// d(n_eq_minus_n_pl)/d(ρ_eq), analytical. See `dem_drho_eq`.
    dneq_drho_eq: Vec<f64>,
    /// Pre-allocated Kompaneets workspace (grid-constant arrays + per-step buffers)
    komp_ws: KompaneetsWorkspace,
    /// Precomputed Planck spectrum on the grid: planck(x[i])
    planck_grid: Vec<f64>,
    /// Diagnostic: scale factor for DC/BR emission rates. Default 1.0.
    /// Set to < 1.0 to test whether rates are too strong.
    pub dcbr_scale: f64,
    /// Subtract the temperature shift component from Δn after each
    /// DC/BR step at z > 5×10⁴, enforcing photon number conservation
    /// (∫x² Δn dx = 0). This prevents DC/BR-created photons from
    /// accumulating as a spurious temperature shift that feeds back into
    /// over-thermalization. See Chluba (2013), arXiv:1304.6120.
    /// Enabled by default.
    pub number_conserving: bool,
    /// Apply NC stripping every nc_stride steps (default 1 = every step).
    /// Higher values reduce NC-DC/BR feedback at high z.
    pub nc_stride: usize,
    /// Cumulative δT/T subtracted from Δn by number conservation.
    /// Added back to ρ_eq so T_e tracks the true (shifted) reference.
    pub accumulated_delta_t: f64,
    /// Precomputed g_bb(x[i]) = x e^x / (e^x - 1)² on the grid.
    /// Used by subtract_temperature_shift() to avoid recomputation.
    g_bb_grid: Vec<f64>,
    /// Discrete quadrature ∫x²·g_bb·dx using the same trapezoidal rule as
    /// the number integral. Used by subtract_temperature_shift() to ensure
    /// exact photon number conservation on the discrete grid, avoiding the
    /// O(N⁻²) drift from using the analytical 3·G₂.
    g2_gbb_discrete: f64,
    /// Diagnostic counters and extrema tracked during evolution.
    pub diag: SolverDiagnostics,
    /// Precomputed DC high-frequency suppression: exp(-2x) × polynomial(x) for each grid point.
    /// Depends only on x, computed once at construction.
    dc_suppression_grid: Vec<f64>,
    /// Precomputed DC suppression at cell midpoints x_half[i] = (x[i]+x[i+1])/2.
    /// Used in the DC/BR heating integral (dcbr_heating_with_derivative), which
    /// evaluates K_DC at midpoints rather than grid nodes.
    dc_suppression_half: Vec<f64>,
    /// Precomputed exp(x) - 1 for each grid point (for Bose factor Taylor expansion).
    exp_m1_grid: Vec<f64>,
    /// Precomputed exp(x) for each grid point (for Bose factor Taylor expansion).
    exp_grid: Vec<f64>,
    /// Precomputed ln(x) for each grid point (for BR Gaunt factor).
    ln_x_grid: Vec<f64>,
    /// Precomputed ln(x_half) for each cell midpoint (for BR Gaunt factor in heating integral).
    ln_x_half: Vec<f64>,
    /// Precomputed trapezoidal quadrature weights for ∫x² f(x) dx (per grid node).
    /// Used by subtract_temperature_shift() instead of recomputing x_half² × dx.
    quad_weights_x2: Vec<f64>,
    /// Pre-allocated buffer for photon source injection (source_rate × dt per grid point).
    /// Photons are injected directly into Δn; DC/BR handles absorption during
    /// the coupled step, with h_dc_br self-consistently feeding back to ρ_e.
    photon_source_buf: Vec<f64>,
    /// Cached backward Euler ODE coefficients from `update_temperatures()`.
    /// Used by the bordered Newton solver to couple ρ_e into the Kompaneets step.
    rho_e_ode_cache: Option<RhoECache>,
}

/// Compute DC+BR heating integral and optionally its finite-difference
/// derivative dH/dρ_e in a single pass over the frequency grid.
///
/// **Performance optimization**: DC emission coefficients K_DC(x, θ_z) are
/// independent of θ_e, so they're computed once and reused for all 3 FD
/// evaluations (current, +δ, -δ). Grid midpoints, dx, and n_mid are also
/// computed once. This replaces 6 separate O(N) grid sweeps with 1 combined
/// sweep (or 1 sweep + 2 lightweight inner loops for the FD passes).
///
/// Returns (h_dc_br, dh_drho).
fn dcbr_heating_with_derivative(
    x_grid: &[f64],
    delta_n: &[f64],
    theta_z: f64,
    theta_e: f64,
    n_h: f64,
    n_he: f64,
    n_e: f64,
    x_e_frac: f64,
    y_he_ii: f64,
    y_he_i: f64,
    compute_derivative: bool,
    ln_x_half: &[f64],
    dc_supp_half: &[f64],
) -> (f64, f64) {
    use crate::bremsstrahlung::{br_emission_coefficient_fast_preln, br_precompute};
    use crate::spectrum::planck;

    if theta_e < 1e-30 || n_e < 1e-30 {
        return (0.0, 0.0);
    }

    let phi = theta_z / theta_e;
    let norm = 1.0 / (4.0 * G3_PLANCK * theta_z);
    let n = x_grid.len();

    // Single pass: compute h_dc_br at current θ_e
    let mut integral = 0.0;

    // If computing derivative, also accumulate +/- integrals
    let delta_rho_fd = 1e-4;
    let rho_e = theta_e / theta_z;
    let phi_plus = theta_z / (theta_z * (rho_e + delta_rho_fd));
    let phi_minus = theta_z / (theta_z * (rho_e - delta_rho_fd));
    let theta_e_plus = theta_z * (rho_e + delta_rho_fd);
    let theta_e_minus = theta_z * (rho_e - delta_rho_fd);

    let mut integral_plus = 0.0;
    let mut integral_minus = 0.0;

    // Hoist x-independent DC prefactor out of the grid loop
    let dc_pre = dc_prefactor(theta_z);

    // Precompute BR for the 3 θ_e values. At function entry we've already
    // rejected theta_e < 1e-30 and n_e < 1e-30, so `br_precompute` here
    // cannot return `None`. Unwrapping outside the grid loop eliminates
    // per-cell Option-match overhead and lets LLVM auto-vectorize the body
    // (the inner call gets inlined + the straight-line flow is SIMD-able
    // because `planck`, `exp_m1`, `exp`, `ln` all auto-vec at -C
    // target-cpu=native on AVX-512 hardware).
    let br_pre = br_precompute(theta_e, theta_z, n_h, n_he, n_e, x_e_frac, y_he_ii, y_he_i)
        .expect("br_precompute: entry checks guarantee Some");

    // Derivative only runs when both FD shifts stay positive. If either
    // `theta_e_plus` / `theta_e_minus` would underflow `br_precompute`'s
    // positivity guard (ρ_e ≲ δ_FD = 1e-4, extremely uncommon), fall back
    // to the no-derivative path and return dh = 0.
    let fd_pre = if compute_derivative {
        match (
            br_precompute(
                theta_e_plus,
                theta_z,
                n_h,
                n_he,
                n_e,
                x_e_frac,
                y_he_ii,
                y_he_i,
            ),
            br_precompute(
                theta_e_minus,
                theta_z,
                n_h,
                n_he,
                n_e,
                x_e_frac,
                y_he_ii,
                y_he_i,
            ),
        ) {
            (Some(p), Some(m)) => Some((p, m)),
            _ => None,
        }
    } else {
        None
    };

    if let Some((br_pre_plus, br_pre_minus)) = fd_pre {
        for i in 1..n {
            let dx = x_grid[i] - x_grid[i - 1];
            let x_mid = 0.5 * (x_grid[i] + x_grid[i - 1]);
            let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
            let n_mid = planck(x_mid) + dn_mid;
            let ln_xm = ln_x_half[i - 1];

            let k_dc = dc_pre * dc_supp_half[i - 1];
            let k_br = br_emission_coefficient_fast_preln(x_mid, ln_xm, &br_pre);
            let k_br_p = br_emission_coefficient_fast_preln(x_mid, ln_xm, &br_pre_plus);
            let k_br_m = br_emission_coefficient_fast_preln(x_mid, ln_xm, &br_pre_minus);

            let x_e = x_mid * phi;
            let x_e_p = x_mid * phi_plus;
            let x_e_m = x_mid * phi_minus;
            let em = x_e.exp_m1();
            let em_p = x_e_p.exp_m1();
            let em_m = x_e_m.exp_m1();

            integral += (1.0 - n_mid * em) * (k_dc + k_br) * dx;
            integral_plus += (1.0 - n_mid * em_p) * (k_dc + k_br_p) * dx;
            integral_minus += (1.0 - n_mid * em_m) * (k_dc + k_br_m) * dx;
        }
    } else {
        for i in 1..n {
            let dx = x_grid[i] - x_grid[i - 1];
            let x_mid = 0.5 * (x_grid[i] + x_grid[i - 1]);
            let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
            let n_mid = planck(x_mid) + dn_mid;
            let ln_xm = ln_x_half[i - 1];

            let k_dc = dc_pre * dc_supp_half[i - 1];
            let k_br = br_emission_coefficient_fast_preln(x_mid, ln_xm, &br_pre);

            let x_e = x_mid * phi;
            let factor = 1.0 - n_mid * x_e.exp_m1();
            integral += factor * (k_dc + k_br) * dx;
        }
    }

    let h = integral * norm;
    let dh = if compute_derivative {
        (integral_plus * norm - integral_minus * norm) / (2.0 * delta_rho_fd)
    } else {
        0.0
    };

    (h, dh)
}

impl ThermalizationSolver {
    /// Construct a solver with the given cosmology and frequency grid.
    ///
    /// All other state (solver config, injection, flags) is set to defaults;
    /// mutate the public fields or call [`Self::set_injection`] /
    /// [`Self::set_config`] before running. For a fluent API that performs
    /// validation at build time, use [`Self::builder`] instead.
    pub fn new(cosmo: Cosmology, grid_config: GridConfig) -> Self {
        let grid = FrequencyGrid::new(&grid_config);
        let n = grid.n;
        let recomb = RecombinationHistory::new(&cosmo);
        let komp_ws = KompaneetsWorkspace::new(&grid);
        let planck_grid: Vec<f64> = grid.x.iter().map(|&x| planck(x)).collect();
        let g_bb_grid: Vec<f64> = grid.x.iter().map(|&x| crate::spectrum::g_bb(x)).collect();
        let dc_suppression_grid: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| dc_high_freq_suppression(x))
            .collect();
        let dc_suppression_half: Vec<f64> = grid
            .x_half
            .iter()
            .map(|&x| dc_high_freq_suppression(x))
            .collect();
        // Overflow at x > 500 produces +∞; downstream `is_finite` checks then
        // reject the emission rate. Using f64::MAX would pass through those
        // guards as a huge finite number (see audit H2).
        let exp_m1_grid: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| if x > 500.0 { f64::INFINITY } else { x.exp_m1() })
            .collect();
        let exp_grid: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| if x > 500.0 { f64::INFINITY } else { x.exp() })
            .collect();
        let ln_x_grid: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| if x < 1e-30 { -69.0 } else { x.ln() })
            .collect();
        let ln_x_half: Vec<f64> = grid
            .x_half
            .iter()
            .map(|&x| if x < 1e-30 { -69.0 } else { x.ln() })
            .collect();
        let quad_weights_x2: Vec<f64> = {
            let mut w = vec![0.0; n];
            for j in 1..n {
                let xh = grid.x_half[j - 1];
                let half_w = 0.5 * xh * xh * grid.dx[j - 1];
                w[j - 1] += half_w;
                w[j] += half_w;
            }
            w
        };
        let g2_gbb_discrete: f64 = {
            let mut sum = 0.0;
            for i in 1..grid.n {
                let gbb_mid = 0.5 * (g_bb_grid[i] + g_bb_grid[i - 1]);
                let x_half = grid.x_half[i - 1];
                sum += x_half * x_half * gbb_mid * grid.dx[i - 1];
            }
            sum
        };
        ThermalizationSolver {
            config: SolverConfig::default(),
            cosmo,
            grid,
            delta_n: vec![0.0; n],
            electron_temp: ElectronTemperature::default(),
            z: SolverConfig::default().z_start,
            injection: None,
            snapshots: Vec::new(),
            step_count: 0,
            rho_eq: 1.0,
            recomb,
            initial_delta_n: None,
            disable_dcbr: false,
            coupled_dcbr: true,
            emission_rates: vec![0.0; n],
            n_eq_minus_n_pl: vec![0.0; n],
            dem_drho_eq: vec![0.0; n],
            dneq_drho_eq: vec![0.0; n],
            komp_ws,
            planck_grid,
            dcbr_scale: 1.0,
            number_conserving: true,
            nc_stride: 1,
            accumulated_delta_t: 0.0,
            g_bb_grid,
            g2_gbb_discrete,
            diag: SolverDiagnostics::default(),
            rho_e_ode_cache: None,
            dc_suppression_grid,
            dc_suppression_half,
            exp_m1_grid,
            exp_grid,
            ln_x_grid,
            ln_x_half,
            quad_weights_x2,
            photon_source_buf: vec![0.0; n],
        }
    }

    /// Attach an energy-injection scenario, validating it first.
    ///
    /// Returns `Err` if the scenario parameters are unphysical (e.g. negative
    /// widths, impossible masses). Collects stimulated-emission warnings into
    /// [`SolverDiagnostics::warnings`].
    pub fn set_injection(&mut self, scenario: InjectionScenario) -> Result<(), String> {
        scenario.validate()?;

        for warning in scenario.warn_stimulated_emission() {
            self.diag.warnings.push(warning);
        }

        // Surface the case where the caller built a grid that doesn't span
        // the injection frequency. The refinement zone is silently clipped
        // to [x_min, x_max] during grid construction (see
        // energy_injection.rs::refinement_zones), so the injection spike
        // goes unresolved. The builder auto-refines via `suggested_x_min`,
        // but a bare `ThermalizationSolver::new` + `set_injection` bypasses
        // that path — warn the caller rather than silently mis-resolving.
        let x_min = self.grid.x.first().copied().unwrap_or(0.0);
        let x_max = self.grid.x.last().copied().unwrap_or(f64::INFINITY);
        let x_inj_opt = match &scenario {
            InjectionScenario::MonochromaticPhotonInjection { x_inj, .. } => Some(*x_inj),
            InjectionScenario::DecayingParticlePhoton { x_inj_0, .. } => Some(*x_inj_0),
            _ => None,
        };
        if let Some(x_inj) = x_inj_opt {
            if x_inj < x_min {
                self.diag.warnings.push(format!(
                    "Injection frequency x_inj={x_inj:.3e} is below the grid's x_min={x_min:.3e}. \
                     The refinement zone will be clipped away and the injection spike \
                     will not be resolved. Rebuild the grid with a lower x_min (see \
                     InjectionScenario::suggested_x_min) or use SolverBuilder, which \
                     auto-refines."
                ));
            } else if x_inj > x_max {
                self.diag.warnings.push(format!(
                    "Injection frequency x_inj={x_inj:.3e} is above the grid's x_max={x_max:.3e}. \
                     The injection source will be silently zero: the grid cannot represent \
                     photons above x_max. Extend x_max to cover x_inj + a few σ_x."
                ));
            }
        }

        self.injection = Some(scenario);
        Ok(())
    }

    /// Replace the solver configuration and reset the current redshift to
    /// the new `z_start`.
    pub fn set_config(&mut self, config: SolverConfig) {
        self.z = config.z_start;
        self.config = config;
    }

    /// Set an initial photon perturbation Δn(x) for the next PDE run.
    ///
    /// This replaces the default Δn = 0 initialization in `run_with_snapshots`.
    /// The perturbation is consumed (taken) on the next call to `run_with_snapshots`.
    ///
    /// Panics if any entry is non-finite — silently passing NaN/Inf into the
    /// solver causes a deep panic in the Newton step where the source isn't
    /// obvious. Caught early here so the user sees the real culprit.
    pub fn set_initial_delta_n(&mut self, delta_n: Vec<f64>) {
        assert_eq!(
            delta_n.len(),
            self.grid.n,
            "initial_delta_n length {} != grid size {}",
            delta_n.len(),
            self.grid.n
        );
        if let Some((idx, val)) = delta_n.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            panic!(
                "initial_delta_n contains non-finite value at index {idx} (={val}); \
                 this would silently propagate into Δn and abort the solver in a less \
                 informative location. Reject NaN/Inf at the user input boundary."
            );
        }
        self.initial_delta_n = Some(delta_n);
    }

    /// Reset solver state for reuse, keeping grid and recombination cache.
    ///
    /// Restores all configuration flags to their defaults: a reset solver
    /// behaves identically to a freshly constructed one (aside from the
    /// cached grid/recombination tables, which are preserved for speed).
    pub fn reset(&mut self) {
        for v in self.delta_n.iter_mut() {
            *v = 0.0;
        }
        self.electron_temp = ElectronTemperature::default();
        self.rho_eq = 1.0;
        self.injection = None;
        self.snapshots.clear();
        self.step_count = 0;
        self.initial_delta_n = None;
        self.rho_e_ode_cache = None;
        self.accumulated_delta_t = 0.0;
        self.diag = SolverDiagnostics::default();
        self.config = SolverConfig::default();
        self.z = self.config.z_start;
        self.disable_dcbr = false;
        self.coupled_dcbr = true;
        self.number_conserving = true;
        self.nc_stride = 1;
        self.dcbr_scale = 1.0;
        for v in self.emission_rates.iter_mut() {
            *v = 0.0;
        }
        for v in self.n_eq_minus_n_pl.iter_mut() {
            *v = 0.0;
        }
        for v in self.dem_drho_eq.iter_mut() {
            *v = 0.0;
        }
        for v in self.dneq_drho_eq.iter_mut() {
            *v = 0.0;
        }
        for v in self.photon_source_buf.iter_mut() {
            *v = 0.0;
        }
    }

    fn x_e_at(&self, z: f64) -> f64 {
        self.recomb.x_e(z)
    }

    fn adaptive_dz(&self) -> f64 {
        let x_e = self.x_e_at(self.z);
        let t_c = self.cosmo.t_compton(self.z, x_e);
        let h = self.cosmo.hubble(self.z);
        let theta_e_val = self.electron_temp.theta_e_with(self.cosmo.theta_z(self.z));
        if theta_e_val < 1e-30 {
            return self.config.dz_min;
        }
        let mut dz = self.config.dy_max * t_c * h * (1.0 + self.z) / theta_e_val;

        // Cap the Compton optical depth (dtau = dz / (H*(1+z)*t_C)) to limit
        // DC/BR backward Euler over-thermalization error.
        if self.config.dtau_max > 0.0 {
            let dz_from_dtau = self.config.dtau_max * h * (1.0 + self.z) * t_c;
            dz = dz.min(dz_from_dtau);
        }

        // Limit step size during active injection to resolve the heating profile.
        // At least 10 steps per sigma within the ±5σ window.
        // Before the injection (z > z_h + 5σ), allow larger steps but ensure
        // we don't overshoot the injection window entirely.
        // Applies to both SingleBurst and MonochromaticPhotonInjection.
        let burst_params: Option<(f64, f64)> = match &self.injection {
            Some(InjectionScenario::SingleBurst { z_h, sigma_z, .. }) => Some((*z_h, *sigma_z)),
            Some(InjectionScenario::MonochromaticPhotonInjection { z_h, sigma_z, .. }) => {
                Some((*z_h, *sigma_z))
            }
            _ => None,
        };
        let is_photon_source = matches!(
            &self.injection,
            Some(InjectionScenario::MonochromaticPhotonInjection { .. })
                | Some(InjectionScenario::DecayingParticlePhoton { .. })
                | Some(InjectionScenario::TabulatedPhotonSource { .. })
        );
        if let Some((z_h, sigma_z)) = burst_params {
            let z_upper = z_h + 5.0 * sigma_z;
            let z_lower = z_h - 5.0 * sigma_z;
            if self.z >= z_lower && self.z <= z_upper {
                // Inside injection window: fine stepping
                dz = dz.min(sigma_z / 10.0);
                // For photon injection, also limit dtau to resolve the
                // Kompaneets Δn² nonlinearity at the injection spike.
                // Without this, the spike amplitude Δn >> n_pl causes
                // timestep-dependent Kompaneets scattering (wake).
                if is_photon_source && self.config.dtau_max_photon_source > 0.0 {
                    let dz_from_dtau_photon =
                        self.config.dtau_max_photon_source * h * (1.0 + self.z) * t_c;
                    dz = dz.min(dz_from_dtau_photon);
                }
            } else if self.z > z_upper {
                // Pre-injection: coast with larger steps (10× normal dy_max step)
                // to avoid wasting thousands of fine steps before the burst begins.
                // NEVER overshoot the injection window boundary.
                let dz_coast = 10.0 * self.config.dy_max * t_c * h * (1.0 + self.z) / theta_e_val;
                dz = dz.max(dz_coast);
                // Respect dtau_max
                if self.config.dtau_max > 0.0 {
                    let dz_from_dtau = self.config.dtau_max * h * (1.0 + self.z) * t_c;
                    dz = dz.min(dz_from_dtau);
                }
                // Clamp so we land at the injection window boundary, not beyond it
                let dz_to_window = self.z - z_upper;
                if dz > dz_to_window && dz_to_window > 0.0 {
                    // Step to just outside the injection window (stop above z_upper)
                    dz = dz_to_window - sigma_z / 10.0;
                    if dz < 0.0 {
                        dz = dz_to_window;
                    }
                }
            }
            // Post-injection (z < z_lower): normal adaptive stepping
        }

        dz.max(self.config.dz_min).min(self.z * 0.05)
    }

    /// Update ρ_e from distortion feedback + injection.
    ///
    /// Two modes selected automatically by distortion amplitude:
    /// - **Small Δn** (|ΔG₃/G₃| ≤ 0.1): Perturbative Δρ_eq = ΔI₄/(4G₃) - ΔG₃/G₃.
    ///   Avoids the 0.1% cancellation error in the full I₄/(4G₃) computation.
    /// - **Large Δn** (|ΔG₃/G₃| > 0.1): Exact ρ_eq = I₄/(4G₃) using the full
    ///   occupation number n = n_pl + Δn. Retains the Δn² term in I₄ and the
    ///   nonlinear denominator. Necessary for strong depletions (e.g., dark photon
    ///   conversions with γ_con ~ O(1)) where the perturbative expansion breaks down.
    ///
    /// Returns (x_e, t_c, theta_z, max_dn_abs, dtau, hubble) at z_eval for reuse
    /// by the caller, avoiding redundant recomputation.
    fn update_temperatures(
        &mut self,
        z_eval: f64,
        actual_dz: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        // Fused max|Δn| scan + spectral integrals: single pass over delta_n
        // to avoid a redundant O(n) traversal. NaN detection is folded in.
        let mut max_dn: f64 = 0.0;
        let mut delta_i4 = 0.0;
        let mut delta_g3 = 0.0;
        let mut exact_i4 = 0.0;
        let mut exact_g3 = 0.0;
        // First element contributes to max_dn but not to the midpoint integrals
        let abs0 = self.delta_n[0].abs();
        if abs0.is_nan() {
            max_dn = f64::NAN;
        } else if abs0 > max_dn {
            max_dn = abs0;
        }
        for i in 1..self.grid.n {
            // Track max|Δn| with NaN propagation
            let abs_dn = self.delta_n[i].abs();
            if abs_dn.is_nan() || max_dn.is_nan() {
                max_dn = f64::NAN;
            } else if abs_dn > max_dn {
                max_dn = abs_dn;
            }
            // Spectral integrals (always computed; ~free since we're already touching the data)
            let dx = self.grid.dx[i - 1];
            let x_half = self.grid.x_half[i - 1];
            let x3 = self.grid.x_half_cubed[i - 1];
            let dn_mid = 0.5 * (self.delta_n[i] + self.delta_n[i - 1]);
            let n_pl = 0.5 * (self.planck_grid[i] + self.planck_grid[i - 1]);
            delta_g3 += x3 * dn_mid * dx;
            delta_i4 += x3 * x_half * (2.0 * n_pl + 1.0) * dn_mid * dx;
            let n_full = (n_pl + dn_mid).max(0.0);
            exact_g3 += x3 * n_full * dx;
            exact_i4 += x3 * x_half * n_full * (1.0 + n_full) * dx;
        }
        assert!(
            max_dn.is_finite(),
            "NaN/Inf detected in delta_n at z={}",
            z_eval
        );

        let delta_rho_eq = if max_dn > 1e-15 {
            // Use exact computation when the distortion is large enough that
            // the perturbative expansion (which drops Δn² in I₄ and linearizes
            // 1/(G₃+ΔG₃)) becomes inaccurate. The exact computation has ~0.1%
            // discretization error from computing I₄/(4G₃), which is negligible
            // for |ΔG₃/G₃| > 0.1 but catastrophic for tiny distortions (~10⁻⁵).
            if delta_g3.abs() / G3_PLANCK > 0.1 && exact_g3 > 1e-30 {
                exact_i4 / (4.0 * exact_g3) - 1.0
            } else {
                delta_i4 / (4.0 * G3_PLANCK) - delta_g3 / G3_PLANCK
            }
        } else {
            0.0
        };

        let x_e = self.x_e_at(z_eval);
        let t_c = self.cosmo.t_compton(z_eval, x_e);
        let theta_z_val = self.cosmo.theta_z(z_eval);

        let delta_rho_inj = if let Some(ref inj) = self.injection {
            let q_rel = inj.heating_rate(z_eval, &self.cosmo);
            q_rel * t_c / (4.0 * theta_z_val)
        } else {
            0.0
        };

        self.rho_eq = 1.0 + delta_rho_eq;

        // Compute Compton optical depth for this step
        let hubble = self.cosmo.hubble(z_eval);
        let dtau = if t_c > 0.0 && hubble > 0.0 {
            actual_dz / (hubble * (1.0 + z_eval) * t_c)
        } else {
            0.0
        };

        if theta_z_val > 1e-8 {
            let n_h = self.cosmo.n_h(z_eval);
            let n_he = self.cosmo.f_he() * n_h;
            let n_e = x_e * n_h;

            // Compton coupling coefficient R = (8/3)(ρ̃_γ/α_h) and adiabatic rate H·t_C.
            //
            // The T_e ODE in Thomson optical depth τ (Seager+ 1999, Chluba+ 2012):
            //   dρ_e/dτ = R·[(ρ_eq + δρ_inj − H_dcbr) − ρ_e] − H·t_C·ρ_e
            //
            // R = 8ρ_γ/(3 m_e c² n_total) = (8/3)(ρ̃_γ/α_h) is the Compton
            // coupling rate per unit Thomson τ. All terms that act through Compton
            // (equilibrium, injection, DC/BR) carry this factor. The adiabatic
            // cooling H·t_C·ρ_e is independent of the photon field.
            let alpha_h_ratio = (n_e + n_h + n_he) / n_e;
            let rho_gamma_per_e = KAPPA_GAMMA * theta_z_val.powi(4) * G3_PLANCK / n_e;
            let r_compton = (8.0 / 3.0) * rho_gamma_per_e / alpha_h_ratio;
            let lambda_htc = hubble * t_c;

            // DC/BR back-reaction on T_e. The heating integral of a pure
            // Planck spectrum vanishes identically (the (e^x − 1) Bose factor
            // cancels 1/n_pl), so for tiny Δn we skip it as a performance
            // optimisation. Fire on any non-trivial Δn regardless of whether
            // a scenario is set — custom initial conditions should still feel
            // DC/BR heating.
            let needs_dcbr_heating = max_dn > 1e-20;
            let (h_dc_br, dh_drho) = if needs_dcbr_heating && !self.disable_dcbr {
                let theta_e_val = self.electron_temp.rho_e * theta_z_val;
                let t_rad = theta_z_val * crate::constants::M_E_C2 / crate::constants::K_BOLTZMANN;
                let z_approx = (t_rad / self.cosmo.t_cmb - 1.0).max(0.0);
                let y_he_ii = crate::recombination::saha_he_ii(z_approx, &self.cosmo);
                let y_he_i = crate::recombination::saha_he_i(z_approx, &self.cosmo);
                let compute_fd = theta_z_val > 2e-5;
                dcbr_heating_with_derivative(
                    &self.grid.x,
                    &self.delta_n,
                    theta_z_val,
                    theta_e_val,
                    n_h,
                    n_he,
                    n_e,
                    x_e,
                    y_he_ii,
                    y_he_i,
                    compute_fd,
                    &self.ln_x_half,
                    &self.dc_suppression_half,
                )
            } else {
                (0.0, 0.0)
            };

            // Backward Euler integration of the full T_e ODE:
            //
            //   dρ_e/dτ = R·[(ρ_eq + δρ_inj − H_dcbr) − ρ_e] − H·t_C·ρ_e
            //
            // Linearizing H_dcbr around ρ_e^n and solving implicitly:
            //   ρ_e^{n+1} = [ρ_e^n + Δτ·R·(ρ_eq + δρ_inj − H₀ + H'·ρ_e^n)]
            //                / [1 + Δτ·(R·(1 + H') + H·t_C)]
            //
            // When R·Δτ ≫ 1 (pre-recombination): quasi-stationary
            //   ρ_e ≈ ρ_eq + δρ_inj (same as before — R cancels).
            // When R·Δτ ≪ 1, H·t_C·Δτ = dz/(1+z) ~ O(1) (post-recombination):
            //   ρ_e cools adiabatically as T_m ∝ (1+z)², with residual Compton
            //   heating from X_e ~ 10⁻⁴.
            let rho_e_old = self.electron_temp.rho_e;

            // Store unscaled source = ρ_eq + δρ_inj; R applied at point of use.
            let rho_source = self.rho_eq + delta_rho_inj;

            // Cache ODE coefficients for the bordered Newton solver.
            self.rho_e_ode_cache = Some(RhoECache {
                rho_e_old,
                r_compton,
                rho_source,
                lambda_htc,
                dh_drho,
                dtau,
            });

            let numerator =
                rho_e_old + dtau * r_compton * (rho_source - h_dc_br + dh_drho * rho_e_old);
            let denominator = 1.0 + dtau * (r_compton * (1.0 + dh_drho) + lambda_htc);
            let rho_e_raw = numerator / denominator;
            // BE (non-coupled) path: tight clamp at 1.5. Post-recombination
            // ρ_e can drop well below 1 (T_m ∝ (1+z)²); the upper bound
            // prevents unphysical overshoot of the perturbative step in weak-
            // Compton regimes. The bordered-Newton path (coupled mode) uses
            // a looser [0, 3] bound because bursts at high z can legitimately
            // push ρ_e above 1. Attempted M1 unification to a single range
            // degrades post-recombination accuracy and is rejected.
            //
            // NaN/∞ are rejected explicitly (audit H1): f64::NaN.clamp(a, b) ==
            // NaN and NaN comparisons return false, so a naïve clamp would
            // silently propagate a degenerate result into later solves.
            if !rho_e_raw.is_finite() {
                self.diag.rho_e_clamped += 1;
                // Keep the prior ρ_e rather than poisoning the solver state.
            } else {
                let rho_e_new = rho_e_raw.clamp(0.0, 1.5);
                if rho_e_new != rho_e_raw {
                    self.diag.rho_e_clamped += 1;
                }
                self.electron_temp.rho_e = rho_e_new;
            }
        } else {
            // θ_z too small for meaningful Compton physics.
            self.rho_e_ode_cache = None;
        }

        (x_e, t_c, theta_z_val, max_dn, dtau, hubble)
    }

    /// Subtract the temperature shift component from Δn to enforce
    /// photon number conservation: ∫x² Δn dx = 0.
    ///
    /// DC/BR creates photons at low x that accumulate as a temperature
    /// shift in Δn. This growing T-shift feeds back: ρ_e rises → DC/BR
    /// equilibrium target rises → more photon creation → larger T-shift.
    /// Subtracting the T-shift breaks this positive feedback loop.
    ///
    /// Algorithm:
    ///   1. ΔG₂ = ∫x² Δn dx  (photon number perturbation)
    ///   2. δT = ΔG₂ / (3 × G₂)  (temperature shift)
    ///   3. Δn[i] -= δT × g_bb(x[i])  (subtract T-shift component)
    ///   4. accumulated_δT += δT
    /// Subtract the number-conserving temperature shift from Δn.
    /// Returns the δT/T that was subtracted.
    fn subtract_temperature_shift(&mut self) -> f64 {
        // Compute ΔG₂ = ∫x² Δn dx using precomputed per-node quadrature weights.
        // Equivalent to midpoint trapezoidal rule but avoids recomputing x_half² × dx.
        let mut delta_g2 = 0.0;
        for i in 0..self.grid.n {
            delta_g2 += self.quad_weights_x2[i] * self.delta_n[i];
        }

        // δT/T = ΔG₂ / ∫x²·g_bb·dx
        // A temperature shift Δn = δT × g_bb(x) has ΔG₂ = δT × ∫x²·g_bb·dx.
        // Using the discrete quadrature g2_gbb_discrete (same trapezoidal rule as
        // the number integral above) ensures exact photon number conservation on
        // the discrete grid, avoiding O(N⁻²) drift from analytical 3·G₂.
        let delta_t = delta_g2 / self.g2_gbb_discrete;

        // Subtract the temperature shift from Δn
        for i in 0..self.grid.n {
            self.delta_n[i] -= delta_t * self.g_bb_grid[i];
        }

        self.accumulated_delta_t += delta_t;
        delta_t
    }

    /// Advance the solver by a single adaptively-chosen timestep.
    ///
    /// Returns the `dz` taken. Most users should call [`Self::run`] or
    /// [`Self::run_with_snapshots`] instead of stepping manually.
    pub fn step(&mut self) -> f64 {
        let dz = self.adaptive_dz();
        self.step_with_dz(dz)
    }

    /// Take a single timestep with a specified dz (instead of the adaptive choice).
    /// Used by `run_with_snapshots` to land exactly on requested snapshot redshifts.
    fn step_with_dz(&mut self, dz: f64) -> f64 {
        let z_new = (self.z - dz).max(self.config.z_end);
        let actual_dz = self.z - z_new;
        let z_mid = self.z - 0.5 * actual_dz;

        // update_temperatures computes ρ_e via backward Euler and returns dtau + hubble
        let (x_e, _t_c, theta_z_val, max_dn_abs, dtau, h) =
            self.update_temperatures(z_mid, actual_dz);

        let n_h = self.cosmo.n_h(z_mid);
        let n_he = self.cosmo.f_he() * n_h;
        let n_e = x_e * n_h;

        let has_phot_src = self
            .injection
            .as_ref()
            .map_or(false, |inj| inj.has_photon_source());

        let rho_e = self.electron_temp.rho_e;

        let theta_e_full = theta_z_val * rho_e;
        let n = self.grid.n;

        // Skip DC/BR computation at very low redshift where rates are negligible.
        // θ_z < 1e-8 corresponds to z ~ 22. For soft photon distortions (x ~ 1e-3),
        // BR absorption has τ > 1 down to z ~ 1700, so the threshold must be low
        // enough to capture post-recombination absorption.
        let skip_dcbr = self.disable_dcbr || theta_z_val < 1e-8;

        if !skip_dcbr {
            // Cache He ionization fractions once per step (same z for all grid points)
            let t_rad = theta_z_val * crate::constants::M_E_C2 / crate::constants::K_BOLTZMANN;
            let z_approx = (t_rad / self.cosmo.t_cmb - 1.0).max(0.0);
            let y_he_ii = crate::recombination::saha_he_ii(z_approx, &self.cosmo);
            let y_he_i = crate::recombination::saha_he_i(z_approx, &self.cosmo);

            // Save old DC/BR buffers for CN DC/BR option
            if self.config.cn_dcbr {
                self.komp_ws.dcbr_em_old[..n].copy_from_slice(&self.emission_rates[..n]);
                self.komp_ws.dcbr_neq_old[..n].copy_from_slice(&self.n_eq_minus_n_pl[..n]);
            }

            // Precompute DC/BR rates for the coupled solve (reuse pre-allocated buffers)
            // Hoist x-independent prefactors out of the grid loop
            let dc_pre = dc_prefactor(theta_z_val);
            let br_pre = br_precompute(
                theta_e_full,
                theta_z_val,
                n_h,
                n_he,
                n_e,
                x_e,
                y_he_ii,
                y_he_i,
            );
            // DC/BR equilibrium: Planck at T_e (Chluba & Sunyaev 2012, Eq. 8).
            //
            // The backward Euler for ρ_e gives ρ_e ≈ ρ_eq + δρ_inj (quasi-stationary).
            // Using the full ρ_e for DC/BR would double-count injection energy
            // (Kompaneets already injects via φ = 1/ρ_e). We subtract the
            // injection contribution:
            //   ρ_dcbr = ρ_e − R·δρ_inj × dτ/(1 + dτ(R(1+h') + H·t_C))
            //
            // For zero injection: ρ_dcbr = ρ_e (correct T_e target).
            // For injection: removes the δρ_inj bump, avoiding double-counting.
            let rho_eq_dcbr = if let Some(c) = self.rho_e_ode_cache {
                let delta_rho_inj = c.rho_source - self.rho_eq;
                let denom = 1.0 + c.dtau * (c.r_compton * (1.0 + c.dh_drho) + c.lambda_htc);
                let inj_contribution = if denom.abs() > 1e-30 {
                    c.r_compton * delta_rho_inj * c.dtau / denom
                } else {
                    0.0
                };
                (rho_e - inj_contribution).clamp(0.5, 2.0)
            } else {
                rho_e
            };
            let delta_rho = rho_eq_dcbr - 1.0;
            let inv_rho_eq = 1.0 / rho_eq_dcbr;

            // Hoist struct-field slices and scalars so LLVM can register-allocate
            // them and elide bounds checks inside the grid loop. Splitting on
            // `in_taylor` specialises the hot |δρ|<0.01 path into a straight-line
            // loop with no per-point branch on `delta_rho`.
            let dcbr_scale = self.dcbr_scale;
            let in_taylor = delta_rho.abs() < 0.01;
            let mut any_nan = false;
            let mut max_rate = self.diag.max_emission_rate;

            // Borrow the read-only and mut slices once, up-front. Disjoint
            // fields → the borrow checker accepts this.
            let xs: &[f64] = &self.grid.x[..n];
            let dc_supp: &[f64] = &self.dc_suppression_grid[..n];
            let ln_x: &[f64] = &self.ln_x_grid[..n];
            let exp_m1: &[f64] = &self.exp_m1_grid[..n];
            let exp_x: &[f64] = &self.exp_grid[..n];
            let planck_g: &[f64] = &self.planck_grid[..n];
            let em_out: &mut [f64] = &mut self.emission_rates[..n];
            let neq_out: &mut [f64] = &mut self.n_eq_minus_n_pl[..n];
            let dem_out: &mut [f64] = &mut self.dem_drho_eq[..n];
            let dneq_out: &mut [f64] = &mut self.dneq_drho_eq[..n];

            if in_taylor {
                // Taylor expansion of the Bose factor exp(x/ρ) − 1 around ρ=1.
                //   exp(x/ρ) − 1 = exp_m1(x) + [exp(x/ρ) − exp(x)]
                //               = exp_m1(x) − x · exp(x) · (ρ−1)/ρ + O((ρ−1)²)
                // so  bose_factor ≈ exp_m1(x) − x · δρ_inv · exp(x),
                //       δρ_inv ≡ (ρ−1)/ρ.
                // Valid for |ρ−1| < 0.01 — this branch replaces the direct
                // evaluation exp(x/ρ)−1 that near-cancels for small (ρ−1).
                // Analytic ρ-derivatives of em/neq below use the same expansion.
                let delta_rho_inv = delta_rho * inv_rho_eq;
                let inv_rho_eq2 = inv_rho_eq * inv_rho_eq;
                for i in 0..n {
                    let xi = xs[i];
                    let inv_xi3 = 1.0 / (xi * xi * xi);

                    let k_dc = dc_pre * dc_supp[i];
                    let k_br = match br_pre {
                        Some(ref pre) => br_emission_coefficient_fast_preln(xi, ln_x[i], pre),
                        None => 0.0,
                    };
                    let k_sum = k_dc + k_br;

                    let exp_xi = exp_x[i];
                    let bose_factor = exp_m1[i] - xi * delta_rho_inv * exp_xi;

                    let uncapped = dcbr_scale * k_sum * bose_factor * inv_xi3;
                    let finite = uncapped.is_finite();
                    any_nan |= uncapped.is_nan();
                    if finite && uncapped > max_rate {
                        max_rate = uncapped;
                    }
                    em_out[i] = if finite { uncapped } else { 0.0 };

                    let npl = planck_g[i];
                    let npl_1p = npl * (npl + 1.0);
                    neq_out[i] = xi * delta_rho_inv * npl_1p;

                    // Analytic ρ-derivatives (Taylor regime only). Outside the
                    // Taylor window we zero them, reproducing the legacy
                    // Picard-in-ρ_e behaviour the tests were validated against —
                    // the bordered Newton c-vector would otherwise be poisoned
                    // by exp(x/ρ_eq) growing across many decades.
                    let dbose_drho = -xi * exp_xi * inv_rho_eq2;
                    let dem_raw = dcbr_scale * k_sum * dbose_drho * inv_xi3;
                    dem_out[i] = if dem_raw.is_finite() { dem_raw } else { 0.0 };
                    let dneq_raw = npl_1p * xi * inv_rho_eq2;
                    dneq_out[i] = if dneq_raw.is_finite() { dneq_raw } else { 0.0 };
                }
            } else {
                // Full (non-Taylor) path — post-recombination regime where
                // ρ drifts to O(0.3). ρ-derivatives zeroed (see Taylor branch).
                for i in 0..n {
                    let xi = xs[i];
                    let inv_xi3 = 1.0 / (xi * xi * xi);

                    let k_dc = dc_pre * dc_supp[i];
                    let k_br = match br_pre {
                        Some(ref pre) => br_emission_coefficient_fast_preln(xi, ln_x[i], pre),
                        None => 0.0,
                    };

                    let xe = xi * inv_rho_eq;
                    let bose_factor = if xe > 500.0 {
                        f64::INFINITY
                    } else {
                        xe.exp_m1()
                    };
                    let uncapped = dcbr_scale * (k_dc + k_br) * bose_factor * inv_xi3;
                    let finite = uncapped.is_finite();
                    any_nan |= uncapped.is_nan();
                    if finite && uncapped > max_rate {
                        max_rate = uncapped;
                    }
                    em_out[i] = if finite { uncapped } else { 0.0 };

                    let npl = planck_g[i];
                    neq_out[i] = planck(xe) - npl;

                    dem_out[i] = 0.0;
                    dneq_out[i] = 0.0;
                }
            }

            if any_nan {
                self.diag.nan_emission_detected = true;
            }
            self.diag.max_emission_rate = max_rate;
        }

        // Fill photon source buffer (simple — no split-source logic needed)
        let source_active;
        if has_phot_src {
            let dt = actual_dz / (h * (1.0 + z_mid));
            if let Some(ref inj) = self.injection {
                for i in 0..n {
                    self.photon_source_buf[i] =
                        inj.photon_source_rate(self.grid.x[i], z_mid, &self.cosmo) * dt;
                }
            }
            // Use a threshold that excludes Gaussian tails > ~8σ from peak.
            // Peak source ~ O(1e-2), so 1e-20 catches everything within ~6σ
            // but correctly disables the bordered system in the far tails.
            source_active = self.photon_source_buf.iter().any(|&v| v.abs() > 1e-20);
        } else {
            for v in self.photon_source_buf.iter_mut() {
                *v = 0.0;
            }
            source_active = false;
        }

        // Photon source routing:
        //   - When coupled DC/BR Newton runs (below), the integrated source
        //     is passed through `DcbrCoupling::photon_source` so it enters
        //     the Newton residual directly. This keeps the CN "old" Kompaneets
        //     flux computed from the un-injected Δn_old, and lets DC/BR
        //     absorption act on the source within the same step.
        //   - When `skip_dcbr` is true, no Newton runs and the source is
        //     added via the explicit fallback further below. Nothing to do
        //     here in that case.
        //
        // We deliberately do NOT pre-add `photon_source_buf` into `delta_n`
        // here (legacy pre-add path). That approach was mathematically
        // equivalent except for the Kompaneets-CN "old" flux, which would
        // then reflect a state where the source had already been injected
        // at t_old — an O(dt · ∂K/∂t) distortion of the integrated injection.

        // Build ρ_e coupling for the bordered Newton system.
        // When active, ρ_e is solved simultaneously with Δn inside the Newton
        // iteration — no Picard outer loop needed. DC/BR spike absorption
        // immediately feeds back into ρ_e within the same Newton step.
        let rho_coupling = if !skip_dcbr && dtau > 0.0 {
            self.rho_e_ode_cache
                .map(|c| crate::kompaneets::RhoECoupling {
                    rho_e_old: c.rho_e_old,
                    r_compton: c.r_compton,
                    rho_source: c.rho_source,
                    lambda_exp: c.lambda_htc,
                    dh_drho: c.dh_drho,
                    h_norm: 1.0 / (4.0 * G3_PLANCK * theta_z_val),
                })
        } else {
            None
        };

        // Coupled Kompaneets + DC/BR + ρ_e Newton step.
        {
            let dcbr_coupling = if !skip_dcbr && self.coupled_dcbr {
                Some(crate::kompaneets::DcbrCoupling {
                    emission_rates: &self.emission_rates,
                    n_eq_minus_n_pl: &self.n_eq_minus_n_pl,
                    dem_drho_eq: &self.dem_drho_eq,
                    dneq_drho_eq: &self.dneq_drho_eq,
                    photon_source: if source_active && dtau > 0.0 {
                        Some(&self.photon_source_buf)
                    } else {
                        None
                    },
                    cn_dcbr: self.config.cn_dcbr,
                })
            } else {
                None
            };

            let (newton_converged, rho_e_out, newton_last_correction) =
                kompaneets_step_coupled_inplace(
                    &self.grid,
                    &mut self.delta_n,
                    theta_e_full,
                    theta_z_val,
                    dtau,
                    dcbr_coupling.as_ref(),
                    rho_coupling.as_ref(),
                    &mut self.komp_ws,
                    max_dn_abs,
                    self.config.max_newton_iter,
                );
            if !newton_converged {
                self.diag.newton_exhausted += 1;
                // Report the first exhaustion immediately, then at 10, 100,
                // 1000, ... so a persistently-failing solver surfaces the
                // problem instead of hiding behind a single stale warning.
                let n = self.diag.newton_exhausted;
                let report = n == 1 || (n >= 10 && n.is_power_of_two()) || n % 1000 == 0;
                if report {
                    let tol = 1e-8 * max_dn_abs + 1e-14;
                    self.diag.warnings.push(format!(
                        "Newton iteration did not converge at z={:.4e} \
                         after {} iterations (step {}, exhaustion #{}). \
                         Last correction |δx|={:.4e}, tol={:.4e}. \
                         Consider increasing max_newton_iter or reducing dtau_max.",
                        self.z,
                        self.config.max_newton_iter,
                        self.step_count,
                        n,
                        newton_last_correction,
                        tol
                    ));
                }
            }

            // Update ρ_e from the coupled solve.
            // Reject NaN/∞ explicitly (see audit H1): the bordered-system
            // denominator can go through zero and emit NaN, which the naïve
            // clamp would silently propagate into subsequent steps.
            if rho_coupling.is_some() {
                if !rho_e_out.is_finite() {
                    self.diag.rho_e_clamped += 1;
                    // Keep the prior ρ_e.
                } else {
                    let rho_clamped = rho_e_out.clamp(0.0, 3.0);
                    if rho_clamped != rho_e_out {
                        self.diag.rho_e_clamped += 1;
                    }
                    self.electron_temp.rho_e = rho_clamped;
                }
            }

            // DC/BR operator-split step (when not using coupled mode)
            if !skip_dcbr && !self.coupled_dcbr {
                for i in 1..n - 1 {
                    let em = self.emission_rates[i];
                    let neq = self.n_eq_minus_n_pl[i];
                    let decay = (-dtau * em).exp();
                    self.delta_n[i] = neq + (self.delta_n[i] - neq) * decay;
                }
            }
        }

        // Fallback pre-add for the paths where the photon source was NOT
        // plumbed through the Newton residual. Two cases:
        //   - `skip_dcbr`: no Newton runs (low-z or user-disabled DC/BR).
        //   - `!coupled_dcbr`: operator-split DC/BR runs after a Newton that
        //     did not receive the source through `DcbrCoupling::photon_source`.
        // In both cases we fall back to adding S·dt directly to Δn, which is
        // a forward-Euler-in-source treatment — acceptable because these
        // paths are diagnostic / post-recombination anyway.
        let source_via_newton = !skip_dcbr && self.coupled_dcbr;
        if let Some(ref inj) = self.injection {
            if inj.has_photon_source() && !source_via_newton {
                let dt = actual_dz / (h * (1.0 + z_mid));
                for i in 0..n {
                    let source = inj.photon_source_rate(self.grid.x[i], z_mid, &self.cosmo);
                    if source.abs() > 1e-50 {
                        self.delta_n[i] += source * dt;
                    }
                }
            }
        }

        // Number-conserving mode: subtract temperature shift G_bb from Δn.
        // This prevents DC/BR-created low-x photons from accumulating as a
        // secular T-shift that masks the physical distortion shape.
        if self.number_conserving
            && self.z > self.config.nc_z_min
            && (self.nc_stride <= 1 || self.step_count % self.nc_stride == 0)
        {
            self.subtract_temperature_shift();
        }

        self.z = z_new;
        self.step_count += 1;
        actual_dz
    }

    /// Integrate from `z_start` to `z_end`, recording a snapshot at each
    /// requested redshift.
    ///
    /// `snapshot_redshifts` may be given in any order; they are sorted
    /// internally. Redshifts above `z_start` or below `z_end` are clamped
    /// to the initial and final state respectively. The returned slice
    /// borrows from `self.snapshots`; for an owned result, use
    /// [`Self::run_to_result`].
    pub fn run_with_snapshots(&mut self, snapshot_redshifts: &[f64]) -> &[SolverSnapshot] {
        let initial_dn = self.initial_delta_n.take();
        let injection = self.injection.take();
        // Preserve user-set configuration across the internal reset: reset()
        // now clears all flags to their defaults to prevent stale toggles
        // leaking between back-to-back runs, but a single run_with_snapshots
        // call must honour whatever the caller configured via the builder or
        // direct field assignment.
        let saved_config = self.config.clone();
        let saved_warnings = std::mem::take(&mut self.diag.warnings);
        let saved_disable_dcbr = self.disable_dcbr;
        let saved_coupled_dcbr = self.coupled_dcbr;
        let saved_number_conserving = self.number_conserving;
        let saved_nc_stride = self.nc_stride;
        let saved_dcbr_scale = self.dcbr_scale;
        self.reset();
        self.config = saved_config;
        self.z = self.config.z_start;
        self.diag.warnings = saved_warnings;
        self.disable_dcbr = saved_disable_dcbr;
        self.coupled_dcbr = saved_coupled_dcbr;
        self.number_conserving = saved_number_conserving;
        self.nc_stride = saved_nc_stride;
        self.dcbr_scale = saved_dcbr_scale;
        self.injection = injection;

        // Frequency grid sanity: the μ/y decomposition silently returns
        // zeros when fewer than three points fall in [0.5, 18]. Surface
        // that case as a warning so users with a custom narrow grid see
        // a real signal instead of a flat-zero result.
        let band_count = crate::distortion::decomposition_band_count(&self.grid.x);
        if band_count < 3 {
            self.diag.warnings.push(format!(
                "Frequency grid has only {band_count} point(s) in the μ/y decomposition band \
                 [0.5, 18]. The decomposition will silently return μ=y=0; widen x_min/x_max \
                 or increase n_points to avoid this."
            ));
        }

        // DarkPhotonResonance installs its IC at z_start; warn users whose
        // z_start doesn't match the NWA resonance redshift (the builder
        // auto-corrects, but `set_config` on a bare solver bypasses that).
        let dp_z_res = self
            .injection
            .as_ref()
            .and_then(|inj| inj.dark_photon_params(&self.cosmo))
            .map(|(_g, z_res)| z_res);
        if let Some(z_res) = dp_z_res {
            if (self.config.z_start - z_res).abs() > 1e-6 * z_res {
                self.diag.warnings.push(format!(
                    "DarkPhotonResonance: z_start={:.3e} ≠ NWA resonance z_res={:.3e}; the \
                     depletion IC is installed at z_start, not z_res. Prefer SolverBuilder, \
                     which sets z_start = z_res automatically.",
                    self.config.z_start, z_res
                ));
            }
        }

        // Precedence: an explicit user-set Δn(x) via `set_initial_delta_n`
        // wins. Otherwise, if the injection scenario supplies its own IC
        // (e.g. DarkPhotonResonance), install that.
        if let Some(dn) = initial_dn {
            self.delta_n = dn;
        } else if let Some(ref inj) = self.injection {
            if let Some(dn) = inj.initial_delta_n(&self.grid.x, &self.cosmo) {
                assert_eq!(
                    dn.len(),
                    self.grid.n,
                    "injection initial_delta_n length {} != grid size {}",
                    dn.len(),
                    self.grid.n
                );
                self.delta_n = dn;
            }
        }

        // Sort snapshot redshifts descending (we integrate from high z to low z)
        let mut sorted_z: Vec<f64> = snapshot_redshifts.to_vec();
        sorted_z.sort_by(|a, b| b.total_cmp(a));
        let mut next_snap = 0;

        // Skip any snapshot redshifts at or above z_start (save initial state for them)
        while next_snap < sorted_z.len() && sorted_z[next_snap] >= self.z {
            self.save_snapshot_at(sorted_z[next_snap]);
            next_snap += 1;
        }

        let step_limit = 5_000_000;
        while self.z > self.config.z_end && self.step_count < step_limit {
            // Check if the next snapshot is between current z and what the step
            // would produce — if so, take a partial step to land exactly on it
            if next_snap < sorted_z.len() {
                let snap_z = sorted_z[next_snap];
                if snap_z >= self.config.z_end && snap_z < self.z {
                    let dz_to_snap = self.z - snap_z;
                    let dz_natural = self.adaptive_dz();
                    if dz_natural >= dz_to_snap {
                        // Would overshoot the snapshot — take exact step to land on it
                        self.step_with_dz(dz_to_snap);
                        // Save snapshots for all requested redshifts at this z
                        while next_snap < sorted_z.len() && self.z <= sorted_z[next_snap] {
                            self.save_snapshot_at(sorted_z[next_snap]);
                            next_snap += 1;
                        }
                        continue;
                    }
                }
            }

            self.step();

            // Save snapshots for any requested redshifts we've reached or passed
            while next_snap < sorted_z.len() && self.z <= sorted_z[next_snap] {
                self.save_snapshot_at(sorted_z[next_snap]);
                next_snap += 1;
            }

            if self.step_count % 100000 == 0 {
                self.diag.warnings.push(format!(
                    "Progress: z={:.1e} step={} ρ_e={:.6} ρ_eq={:.6}",
                    self.z, self.step_count, self.electron_temp.rho_e, self.rho_eq
                ));
            }
        }

        if self.step_count >= step_limit {
            self.diag.warnings.push(format!(
                "Step limit ({}) reached at z={:.4e}. \
                 Run may be incomplete. Consider increasing dtau_max or narrowing the z range.",
                step_limit, self.z
            ));
        }

        // Fill remaining snapshots below z_end with final state
        while next_snap < sorted_z.len() {
            self.save_snapshot();
            next_snap += 1;
        }

        // End-of-run diagnostics: ρ_e clamp counter. Individual clamps are
        // silent (only logged in diag), but a large count indicates that the
        // implicit T_e solve is regularly hitting the [0, 1.5] / [0, 3] bounds
        // — the clamped steps silently discard part of the injection energy
        // and shouldn't be ignored.
        if self.diag.rho_e_clamped > 10 {
            self.diag.warnings.push(format!(
                "ρ_e was clamped {} times during the run. Clamping discards injection \
                 energy from the T_e ODE silently; repeated hits suggest the injection \
                 amplitude is outside the linearized regime or dtau_max is too large \
                 for the source sharpness.",
                self.diag.rho_e_clamped
            ));
        }
        if self.diag.nan_emission_detected {
            self.diag.warnings.push(
                "NaN emission rate detected during the run. DC/BR rates were capped \
                 to keep the solver alive but the final Δn may be polluted; narrow \
                 x_min or reduce dtau_max and rerun."
                    .to_string(),
            );
        }

        &self.snapshots
    }

    /// Run the solver with `n_snapshots` log-spaced snapshot redshifts between
    /// z_start and z_end. Note: with `n_snapshots=1` the single snapshot is at
    /// z_start (the first log-spaced point), not z_end.
    pub fn run(&mut self, n_snapshots: usize) -> &[SolverSnapshot] {
        let log_s = self.config.z_start.ln();
        let log_e = self.config.z_end.ln();
        let zs: Vec<f64> = (0..n_snapshots)
            .map(|i| (log_s + (log_e - log_s) * i as f64 / (n_snapshots - 1).max(1) as f64).exp())
            .collect();
        self.run_with_snapshots(&zs)
    }

    fn save_snapshot(&mut self) {
        self.save_snapshot_at(self.z);
    }

    /// Save a snapshot with a specific redshift label.
    /// The solver state (delta_n, rho_e, etc.) is taken from the current state,
    /// but the snapshot's z field is set to the requested value.
    fn save_snapshot_at(&mut self, z: f64) {
        // Extract μ, y via the Bianchini & Fabbian (2022) nonlinear BE fit
        // on the T-shift-subtracted Δn (the fit recovers the accumulated
        // temperature shift separately through its δ parameter, so μ and y
        // are unaffected by whether the subtracted G_bb is added back).
        let (mu, y) = self.extract_mu_y_joint();
        let drho_spectrum = crate::spectrum::delta_rho_over_rho(&self.grid.x, &self.delta_n);
        // Total energy = energy in the spectral Δn + energy in the subtracted
        // T-shift. Δρ/ρ = 4δT/T for a temperature shift (Stefan-Boltzmann).
        let drho = drho_spectrum + 4.0 * self.accumulated_delta_t;

        // Reconstruct the full Δn by adding the accumulated T-shift back.
        // The NC mode subtracts the T-shift during evolution to prevent
        // DC/BR over-thermalization feedback, but the output should contain
        // the full spectral distortion (distortion + T-shift) for comparison
        // with CosmoTherm and other codes.
        let full_delta_n = if self.number_conserving && self.accumulated_delta_t.abs() > 1e-30 {
            let mut dn = self.delta_n.clone();
            for i in 0..self.grid.n {
                dn[i] += self.accumulated_delta_t * self.g_bb_grid[i];
            }
            dn
        } else {
            self.delta_n.clone()
        };

        self.snapshots.push(SolverSnapshot {
            z,
            delta_n: full_delta_n,
            rho_e: self.electron_temp.rho_e,
            mu,
            y,
            delta_rho_over_rho: drho,
            accumulated_delta_t: self.accumulated_delta_t,
        });
    }

    /// Extract `(μ, y)` from the current `Δn(x)` via the default joint
    /// least-squares decomposition (B&F 2022 nonlinear BE; see
    /// [`crate::distortion::decompose_distortion`]).
    pub fn extract_mu_y_joint(&self) -> (f64, f64) {
        let params = crate::distortion::decompose_distortion(&self.grid.x, &self.delta_n);
        (params.mu, params.y)
    }

    /// Run the solver and return an owned [`crate::output::SolverResult`]
    /// with a single snapshot at `z_obs`.
    ///
    /// This is the preferred entry point: the result does not borrow the
    /// solver and has built-in JSON/CSV/table serialization. For runs that
    /// need multiple intermediate snapshots, call
    /// [`Self::run_with_snapshots`] directly and inspect
    /// [`Self::snapshots`].
    pub fn run_to_result(&mut self, z_obs: f64) -> crate::output::SolverResult {
        self.run_with_snapshots(&[z_obs]);
        let snapshot = self
            .snapshots
            .last()
            .expect("run_with_snapshots produced no snapshot")
            .clone();
        crate::output::SolverResult {
            snapshot,
            x_grid: self.grid.x.clone(),
            step_count: self.step_count,
            diag_newton_exhausted: self.diag.newton_exhausted,
            warnings: self.diag.warnings.clone(),
        }
    }

    /// Create a builder for configuring a solver with a fluent API.
    ///
    /// # Example
    /// ```rust,no_run
    /// use spectroxide::prelude::*;
    ///
    /// let mut solver = ThermalizationSolver::builder(Cosmology::planck2018())
    ///     .grid(GridConfig::production())
    ///     .injection(InjectionScenario::SingleBurst {
    ///         z_h: 2e5, delta_rho_over_rho: 1e-5, sigma_z: 100.0,
    ///     })
    ///     .z_range(5e5, 1e3)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder(cosmo: Cosmology) -> SolverBuilder {
        SolverBuilder::new(cosmo)
    }
}

/// Fluent builder for [`ThermalizationSolver`].
///
/// Groups all configuration into a chainable API. The existing
/// `ThermalizationSolver::new()` + manual field mutation still works;
/// this is a convenience layer on top.
pub struct SolverBuilder {
    cosmo: Cosmology,
    grid_config: GridConfig,
    injection: Option<InjectionScenario>,
    initial_delta_n: Option<Vec<f64>>,
    z_start: Option<f64>,
    z_end: Option<f64>,
    dy_max: Option<f64>,
    dz_min: Option<f64>,
    dtau_max: Option<f64>,
    nc_z_min: Option<f64>,
    disable_dcbr: bool,
    coupled_dcbr: bool,
    number_conserving: bool,
    dcbr_scale: f64,
    no_auto_refine: bool,
    max_newton_iter: Option<usize>,
    nc_stride: Option<usize>,
}

impl SolverBuilder {
    fn new(cosmo: Cosmology) -> Self {
        SolverBuilder {
            cosmo,
            grid_config: GridConfig::default(),
            injection: None,
            initial_delta_n: None,
            z_start: None,
            z_end: None,
            dy_max: None,
            dz_min: None,
            dtau_max: None,
            nc_z_min: None,
            disable_dcbr: false,
            coupled_dcbr: true,
            number_conserving: true,
            dcbr_scale: 1.0,
            no_auto_refine: false,
            max_newton_iter: None,
            nc_stride: None,
        }
    }

    /// Set the frequency grid configuration.
    pub fn grid(mut self, config: GridConfig) -> Self {
        self.grid_config = config;
        self
    }

    /// Use the fast (500-point) grid for quick tests.
    pub fn grid_fast(mut self) -> Self {
        self.grid_config = GridConfig::fast();
        self
    }

    /// Use the production (4000-point) grid for high-accuracy runs.
    pub fn grid_production(mut self) -> Self {
        self.grid_config = GridConfig::production();
        self
    }

    /// Set the energy injection scenario.
    pub fn injection(mut self, scenario: InjectionScenario) -> Self {
        self.injection = Some(scenario);
        self
    }

    /// Set an initial photon perturbation Δn(x).
    pub fn initial_delta_n(mut self, delta_n: Vec<f64>) -> Self {
        self.initial_delta_n = Some(delta_n);
        self
    }

    /// Set the redshift range (z_start, z_end).
    pub fn z_range(mut self, z_start: f64, z_end: f64) -> Self {
        self.z_start = Some(z_start);
        self.z_end = Some(z_end);
        self
    }

    /// Set a complete solver config, overriding individual z/dy/dtau settings.
    pub fn solver_config(mut self, config: SolverConfig) -> Self {
        self.z_start = Some(config.z_start);
        self.z_end = Some(config.z_end);
        self.dy_max = Some(config.dy_max);
        self.dz_min = Some(config.dz_min);
        self.dtau_max = Some(config.dtau_max);
        self.nc_z_min = Some(config.nc_z_min);
        self.max_newton_iter = Some(config.max_newton_iter);
        self
    }

    /// Set the maximum Kompaneets step size.
    pub fn dy_max(mut self, val: f64) -> Self {
        self.dy_max = Some(val);
        self
    }

    /// Set the maximum Compton optical depth per step.
    pub fn dtau_max(mut self, val: f64) -> Self {
        self.dtau_max = Some(val);
        self
    }

    /// Disable DC/BR processes (Kompaneets only).
    pub fn disable_dcbr(mut self) -> Self {
        self.disable_dcbr = true;
        self
    }

    /// Use operator-split DC/BR instead of coupled IMEX.
    pub fn split_dcbr(mut self) -> Self {
        self.coupled_dcbr = false;
        self
    }

    /// Enable number-conserving T-shift subtraction (on by default).
    pub fn number_conserving(mut self) -> Self {
        self.number_conserving = true;
        self
    }

    /// Disable number-conserving T-shift subtraction.
    pub fn no_number_conserving(mut self) -> Self {
        self.number_conserving = false;
        self
    }

    /// Set the minimum redshift for number-conserving subtraction.
    pub fn nc_z_min(mut self, val: f64) -> Self {
        self.nc_z_min = Some(val);
        self
    }

    /// Set the NC stripping stride (strip every N steps).
    pub fn nc_stride(mut self, val: usize) -> Self {
        self.nc_stride = Some(val);
        self
    }

    /// Scale DC/BR emission rates by this factor (diagnostic).
    pub fn dcbr_scale(mut self, val: f64) -> Self {
        self.dcbr_scale = val;
        self
    }

    /// Set the maximum number of Newton iterations per Kompaneets step.
    pub fn max_newton_iter(mut self, val: usize) -> Self {
        self.max_newton_iter = Some(val);
        self
    }

    /// Disable automatic refinement zone insertion for photon injection scenarios.
    pub fn no_auto_refine(mut self) -> Self {
        self.no_auto_refine = true;
        self
    }

    /// Build the configured solver.
    ///
    /// Validates all configuration (cosmology, grid, solver config, injection)
    /// before constructing the solver. Returns `Err` with a descriptive message
    /// if any parameter is invalid.
    pub fn build(self) -> Result<ThermalizationSolver, String> {
        // Validate all configuration
        self.cosmo.validate()?;

        let mut grid_config = self.grid_config;

        // Auto-apply refinement zones and x_min adjustment from injection scenario
        if !self.no_auto_refine {
            if let Some(ref inj) = self.injection {
                for zone in inj.refinement_zones() {
                    grid_config.refinement_zones.push(zone);
                }
                // Lower x_min for low-frequency photon injection to prevent
                // boundary absorption (Dirichlet BC eats photons at x_min).
                if let Some(x_min) = inj.suggested_x_min() {
                    if x_min < grid_config.x_min {
                        grid_config.x_min = x_min;
                    }
                }
            }
        }

        grid_config.validate()?;

        let defaults = SolverConfig::default();

        // For DarkPhotonResonance the impulsive Δn depletion happens at the
        // NWA resonance z_res; evolving from a higher z_start is unphysical
        // (the conversion hasn't occurred yet) and evolving from lower misses
        // it entirely. Default z_start to z_res when the user didn't supply
        // an explicit value.
        let dp_z_res = self
            .injection
            .as_ref()
            .and_then(|inj| inj.dark_photon_params(&self.cosmo))
            .map(|(_gamma, z_res)| z_res);
        let resolved_z_start = match (self.z_start, dp_z_res) {
            (Some(z), _) => z,
            (None, Some(z_res)) => z_res,
            (None, None) => defaults.z_start,
        };

        let config = SolverConfig {
            z_start: resolved_z_start,
            z_end: self.z_end.unwrap_or(defaults.z_end),
            dy_max: self.dy_max.unwrap_or(defaults.dy_max),
            dz_min: self.dz_min.unwrap_or(defaults.dz_min),
            dtau_max: self.dtau_max.unwrap_or(defaults.dtau_max),
            nc_z_min: self.nc_z_min.unwrap_or(defaults.nc_z_min),
            dtau_max_photon_source: defaults.dtau_max_photon_source,
            max_newton_iter: self.max_newton_iter.unwrap_or(defaults.max_newton_iter),
            cn_dcbr: defaults.cn_dcbr,
        };
        config.validate()?;

        let mut deferred_warnings: Vec<String> = config.soft_warnings();

        if let Some(ref inj) = self.injection {
            inj.validate()?;

            // Check z_start is high enough to capture the injection
            if let Some((_z_center, z_upper)) = inj.characteristic_redshift() {
                if config.z_start < z_upper {
                    return Err(format!(
                        "z_start={:.1e} is below the injection window upper bound z={:.1e}. \
                         The solver would miss part or all of the injection. \
                         Set z_start >= {:.1e}.",
                        config.z_start, z_upper, z_upper
                    ));
                }
            }

            // Warn if the caller explicitly asked for a z_start that doesn't
            // coincide with the dark-photon resonance: the NWA IC is installed
            // there, so mismatched z_start accumulates spurious pre-resonance
            // thermalisation (or skips the resonance entirely).
            if let (Some(z_user), Some(z_res)) = (self.z_start, dp_z_res) {
                if (z_user - z_res).abs() > 1e-6 * z_res {
                    deferred_warnings.push(format!(
                        "DarkPhotonResonance: z_start={z_user:.3e} differs from NWA resonance \
                         redshift z_res={z_res:.3e}; μ/y will include spurious evolution between \
                         these redshifts. Omit z_start to auto-set it to z_res."
                    ));
                }
            }

            for warning in inj.warn_strong_distortion() {
                deferred_warnings.push(warning);
            }
            for warning in inj.warn_tabulated_coverage(config.z_start, config.z_end) {
                deferred_warnings.push(warning);
            }
            for warning in inj.warn_dark_photon_range(&self.cosmo) {
                deferred_warnings.push(warning);
            }
        }

        let mut solver = ThermalizationSolver::new(self.cosmo, grid_config);
        solver.set_config(config);
        solver.disable_dcbr = self.disable_dcbr;
        solver.coupled_dcbr = self.coupled_dcbr;
        solver.number_conserving = self.number_conserving;
        solver.dcbr_scale = self.dcbr_scale;
        if let Some(stride) = self.nc_stride {
            solver.nc_stride = stride;
        }

        if let Some(scenario) = self.injection {
            solver.set_injection(scenario)?;
        }

        if let Some(dn) = self.initial_delta_n {
            solver.set_initial_delta_n(dn);
        }

        solver.diag.warnings.extend(deferred_warnings);

        Ok(solver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_injection_stays_planck() {
        let cosmo = Cosmology::default();
        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
        solver.set_config(SolverConfig {
            z_start: 1.0e6,
            z_end: 1.0e5,
            ..SolverConfig::default()
        });
        for _ in 0..100 {
            if solver.z <= solver.config.z_end {
                break;
            }
            solver.step();
        }
        let max_dn: f64 = solver.delta_n.iter().map(|x| x.abs()).fold(0.0, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        // With Λ·ρ_e adiabatic cooling (correct formula), even a Planck
        // spectrum develops a small distortion from expansion cooling.
        // The signal is O(Λ) ~ 10⁻⁸, which is physical (adiabatic cooling).
        assert!(max_dn < 1e-6, "max|Δn| = {max_dn}");
    }

    #[test]
    fn test_pde_vs_greens_mu_era() {
        // Inject energy at z=2×10⁵ (deep μ-era).
        // The PDE solver should produce μ ≈ 1.4e-5, matching the Green's
        // function prediction to within ~10%.
        let cosmo = Cosmology::default();
        let z_h = 2.0e5;
        let drho = 1e-5;
        let sigma = 3000.0;

        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: sigma,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 5.0e5,
            z_end: 1.0e4,
            ..SolverConfig::default()
        });

        let snaps = solver.run_with_snapshots(&[1.0e4]);
        let last = snaps.last().unwrap();

        let dq_dz = |z: f64| -> f64 {
            drho * (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt()
        };
        let mu_gf = crate::greens::mu_from_heating(&dq_dz, 1e3, 5e6, 10000);

        eprintln!(
            "μ-era: PDE μ={:.4e} y={:.4e} Δρ/ρ={:.4e}, GF μ={:.4e}",
            last.mu, last.y, last.delta_rho_over_rho, mu_gf
        );

        // Energy conservation
        let energy_err = (last.delta_rho_over_rho - drho).abs() / drho;
        assert!(
            energy_err < 0.1,
            "μ-era energy: Δρ/ρ={:.4e} vs injected {drho:.4e}",
            last.delta_rho_over_rho
        );

        // μ should match GF to within 12%
        let mu_err = (last.mu - mu_gf).abs() / mu_gf.abs().max(1e-20);
        eprintln!("  μ PDE/GF agreement: {:.1}%", (1.0 - mu_err) * 100.0);
        assert!(
            mu_err < 0.12,
            "μ PDE={:.4e} vs GF={:.4e}, err={:.1}%",
            last.mu,
            mu_gf,
            mu_err * 100.0
        );
    }

    #[test]
    fn test_pde_vs_greens_y_era() {
        // Inject energy at z=5000 (y-era).
        // PDE solver y should match Green's function prediction within ~50%.
        let cosmo = Cosmology::default();
        let z_h = 5000.0;
        let drho = 1e-5;
        let sigma = 200.0;

        let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: sigma,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 1.0e4,
            z_end: 1.0e3,
            ..SolverConfig::default()
        });

        let snaps = solver.run_with_snapshots(&[1.0e3]);
        let last = snaps.last().unwrap();

        // Green's function: use positive dq_dz (heating convention)
        let dq_dz = |z: f64| -> f64 {
            drho * (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt()
        };
        let y_gf = crate::greens::y_from_heating(&dq_dz, 1e2, 1e5, 10000);

        let rel_err = (last.y - y_gf).abs() / y_gf.abs().max(1e-20);
        eprintln!(
            "y-era: PDE y={:.4e}, GF y={:.4e}, rel_err={:.1}%",
            last.y,
            y_gf,
            rel_err * 100.0
        );
        assert!(
            rel_err < 0.10,
            "PDE y={:.4e} vs GF y={:.4e}, rel_err={rel_err:.2}",
            last.y,
            y_gf
        );
    }

    #[test]
    fn test_energy_conservation() {
        // Energy injected should approximately match Δρ/ρ in the final spectrum.
        let cosmo = Cosmology::default();
        let drho = 1e-5;

        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 5e4,
                delta_rho_over_rho: drho,
                sigma_z: 2000.0,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 2.0e5,
            z_end: 1.0e3,
            ..SolverConfig::default()
        });

        let snaps = solver.run_with_snapshots(&[1.0e3]);
        let last = snaps.last().unwrap();

        let rel_err = (last.delta_rho_over_rho - drho).abs() / drho;
        eprintln!(
            "Energy conservation: Δρ/ρ_out={:.4e}, Δρ/ρ_in={:.4e}, rel_err={:.1}%",
            last.delta_rho_over_rho,
            drho,
            rel_err * 100.0
        );
        assert!(
            rel_err < 0.05,
            "Energy not conserved: Δρ/ρ_out={:.4e} vs {:.4e}",
            last.delta_rho_over_rho,
            drho
        );
    }

    #[test]
    fn test_pde_y_at_multiple_redshifts() {
        // Cross-validate PDE vs Green's function at z=10⁴, 3×10⁴, 5×10⁴
        let cosmo = Cosmology::default();
        let drho = 1e-5;

        for &(z_h, sigma) in &[(1e4, 500.0), (3e4, 1000.0), (5e4, 2000.0)] {
            let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::default());
            solver
                .set_injection(InjectionScenario::SingleBurst {
                    z_h,
                    delta_rho_over_rho: drho,
                    sigma_z: sigma,
                })
                .unwrap();
            solver.set_config(SolverConfig {
                z_start: z_h * 3.0,
                z_end: 500.0,
                ..SolverConfig::default()
            });
            let snaps = solver.run_with_snapshots(&[500.0]);
            let last = snaps.last().unwrap();

            let dq_dz = |z: f64| -> f64 {
                drho * (-(z - z_h).powi(2) / (2.0 * sigma * sigma)).exp()
                    / (2.0 * std::f64::consts::PI * sigma * sigma).sqrt()
            };
            let y_gf = crate::greens::y_from_heating(&dq_dz, 1e2, z_h * 5.0, 10000);

            let energy_err = (last.delta_rho_over_rho - drho).abs() / drho;
            eprintln!(
                "z_h={z_h:.0e}: PDE y={:.4e} GF y={:.4e}, Δρ/ρ={:.4e}, E_err={:.1}%",
                last.y,
                y_gf,
                last.delta_rho_over_rho,
                energy_err * 100.0
            );

            assert!(
                energy_err < 0.1,
                "Energy not conserved at z_h={z_h}: {:.4e} vs {drho:.4e}",
                last.delta_rho_over_rho
            );
        }
    }

    #[test]
    fn test_greens_function_consistency() {
        // Verify Green's function μ and y have correct scaling with Δρ/ρ
        let dq1 = |z: f64| -> f64 {
            1e-5 * (-(z - 3e5_f64).powi(2) / (2.0 * 5000.0_f64.powi(2))).exp()
                / (2.0 * std::f64::consts::PI * 5000.0_f64.powi(2)).sqrt()
        };
        let dq2 = |z: f64| -> f64 {
            2e-5 * (-(z - 3e5_f64).powi(2) / (2.0 * 5000.0_f64.powi(2))).exp()
                / (2.0 * std::f64::consts::PI * 5000.0_f64.powi(2)).sqrt()
        };

        let mu1 = crate::greens::mu_from_heating(&dq1, 1e3, 5e6, 5000);
        let mu2 = crate::greens::mu_from_heating(&dq2, 1e3, 5e6, 5000);

        // μ should scale linearly with Δρ/ρ
        let ratio = mu2 / mu1;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "μ should scale linearly: μ₂/μ₁ = {ratio:.4} (expected 2.0)"
        );
    }

    #[test]
    fn test_decomposition_pure_tshift() {
        // A pure temperature shift Δn = ε·G(x) should decompose to μ=0, y=0.
        let cosmo = Cosmology::default();
        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());

        let eps = 1e-5;
        for i in 0..solver.grid.n {
            let x = solver.grid.x[i];
            solver.delta_n[i] = eps * crate::spectrum::g_bb(x);
        }

        let (mu, y) = solver.extract_mu_y_joint();
        let drho = crate::spectrum::delta_rho_over_rho(&solver.grid.x, &solver.delta_n);
        eprintln!("Pure T-shift (ε={eps:.0e}): μ={mu:.4e}, y={y:.4e}, Δρ/ρ={drho:.4e}");
        eprintln!("  Expected: μ≈0, y≈0, Δρ/ρ≈{:.4e}", 4.0 * eps);

        assert!(
            mu.abs() < 5e-8,
            "μ should be ~0 for pure T-shift: μ={mu:.4e}"
        );
        assert!(y.abs() < 5e-8, "y should be ~0 for pure T-shift: y={y:.4e}");
    }

    #[test]
    fn test_decomposition_pure_mu() {
        // A pure μ distortion Δn = μ·M(x) should decompose to correct μ.
        let cosmo = Cosmology::default();
        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());

        let mu_val = 1e-5;
        for i in 0..solver.grid.n {
            let x = solver.grid.x[i];
            solver.delta_n[i] = mu_val * crate::spectrum::mu_shape(x);
        }

        let (mu, y) = solver.extract_mu_y_joint();
        let drho = crate::spectrum::delta_rho_over_rho(&solver.grid.x, &solver.delta_n);
        eprintln!("Pure μ (μ_in={mu_val:.0e}): μ_out={mu:.4e}, y={y:.4e}, Δρ/ρ={drho:.4e}");

        let rel_err = (mu - mu_val).abs() / mu_val;
        assert!(
            rel_err < 0.05,
            "μ extraction error: {rel_err:.2}% (μ_out={mu:.4e} vs {mu_val:.4e})"
        );
        assert!(y.abs() < 1e-7, "y should be ~0 for pure μ: y={y:.4e}");
    }

    #[test]
    fn test_snapshots_at_requested_redshifts() {
        let cosmo = Cosmology::default();
        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 5e4,
                delta_rho_over_rho: 1e-5,
                sigma_z: 2000.0,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 2.0e5,
            z_end: 500.0,
            ..SolverConfig::default()
        });

        let requested = [1.5e5, 1e5, 5e4, 1e4, 5e3, 1e3];
        let snaps = solver.run_with_snapshots(&requested);

        assert_eq!(snaps.len(), requested.len());
        for (snap, &z_req) in snaps.iter().zip(requested.iter()) {
            let rel_err = (snap.z - z_req).abs() / z_req;
            assert!(
                rel_err < 0.001,
                "Snapshot z={:.1} should be at requested z={:.1}, rel_err={:.4}",
                snap.z,
                z_req,
                rel_err
            );
        }
    }

    /// `.disable_dcbr()` produces the un-thermalized μ = 1.401 × Δρ/ρ exactly
    /// — no J_bb* suppression because DC/BR photon creation is off.
    ///
    /// Oracle:             pure Kompaneets (no photon sources): energy conserved,
    ///                     μ = (3/κ_c)·Δρ/ρ = 1.401·Δρ/ρ (SZ 1970 / Chluba 2013
    ///                     with J_bb* → 1 by construction).
    /// Expected:           μ = 1.401 × 10⁻⁵ for Δρ/ρ = 10⁻⁵
    /// Oracle uncertainty: 2% (Kompaneets convergence over Δτ ~ few)
    /// Tolerance:          5%
    ///
    /// Previous version asserted only μ > 0 and drho > 0 — sign-only; any
    /// 100× wrong normalization would have passed.
    #[test]
    fn test_solver_builder_disable_dcbr() {
        let drho = 1e-5_f64;
        let mut solver = SolverBuilder::new(Cosmology::default())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: drho,
                sigma_z: 100.0,
            })
            .z_range(2.1e5, 1e5)
            .disable_dcbr()
            .build()
            .unwrap();

        solver.run_with_snapshots(&[1e5]);
        let last = solver.snapshots.last().unwrap();

        let mu_expected = (3.0 / KAPPA_C) * drho; // = 1.401 * drho, no J_bb* with DC/BR off
        let mu_err = (last.mu - mu_expected).abs() / mu_expected;
        eprintln!(
            "disable_dcbr: μ = {:.4e}, expected (3/κ_c)·Δρ/ρ = {:.4e}, err = {:.2}%",
            last.mu,
            mu_expected,
            mu_err * 100.0
        );
        assert!(
            mu_err < 0.05,
            "disable_dcbr: μ = {:.4e} vs pure-Kompaneets target {:.4e} \
             (err {:.2}%, tol 5%)",
            last.mu,
            mu_expected,
            mu_err * 100.0
        );

        // Energy conservation: pure Kompaneets preserves ∫x³ Δn dx exactly
        // modulo scheme residual.
        let drho_err = (last.delta_rho_over_rho - drho).abs() / drho;
        assert!(
            drho_err < 0.02,
            "disable_dcbr energy conservation: Δρ_out = {:.4e} vs {drho:.4e} \
             (err {:.2}%, tol 2%)",
            last.delta_rho_over_rho,
            drho_err * 100.0,
        );
    }

    #[test]
    fn test_solver_builder_split_dcbr() {
        // Operator-split DC/BR mode
        let mut solver = SolverBuilder::new(Cosmology::default())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: 100.0,
            })
            .z_range(2.1e5, 1e5)
            .split_dcbr()
            .build()
            .unwrap();

        solver.run_with_snapshots(&[1e5]);
        let last = solver.snapshots.last().unwrap();
        assert!(last.delta_rho_over_rho.is_finite());
        // Split DC/BR should give positive μ for positive injection
        assert!(
            last.mu > 0.0,
            "split_dcbr: μ should be positive for heating: {:.4e}",
            last.mu
        );
    }

    #[test]
    fn test_solver_reset() {
        let cosmo = Cosmology::default();
        let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 5e4,
                delta_rho_over_rho: 1e-5,
                sigma_z: 2000.0,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 1e5,
            z_end: 1e4,
            ..SolverConfig::default()
        });

        // First run
        solver.run_with_snapshots(&[1e4]);
        let mu1 = solver.snapshots.last().unwrap().mu;
        assert!(mu1.abs() > 0.0);

        // Reset clears state
        solver.reset();
        assert!(solver.snapshots.is_empty());
        assert!(solver.delta_n.iter().all(|&v| v == 0.0));
        assert!(solver.injection.is_none());

        // Re-configure injection and z range, then run again
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 5e4,
                delta_rho_over_rho: 1e-5,
                sigma_z: 2000.0,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 1e5,
            z_end: 1e4,
            ..SolverConfig::default()
        });
        solver.run_with_snapshots(&[1e4]);
        let mu2 = solver.snapshots.last().unwrap().mu;

        // Should be reproducible
        let diff = (mu1 - mu2).abs() / mu1.abs().max(1e-20);
        assert!(
            diff < 1e-10,
            "Reset should give identical results: {mu1:.4e} vs {mu2:.4e}"
        );
    }

    #[test]
    fn test_solver_run_to_result() {
        let mut solver = SolverBuilder::new(Cosmology::default())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 5e4,
                delta_rho_over_rho: 1e-5,
                sigma_z: 2000.0,
            })
            .z_range(1e5, 1e4)
            .build()
            .unwrap();

        let result = solver.run_to_result(1e4);
        assert!(result.step_count > 0);
        assert!(!result.x_grid.is_empty());
        // run_to_result returns the snapshot at z_obs (≈1e4 here).
        assert!(result.snapshot.z < 1.1e4);

        // Can serialize to JSON
        let json = result.to_json();
        assert!(json.contains("\"pde_mu\":"));
        assert!(json.contains("\"diag_newton_exhausted\":"));
    }
}
