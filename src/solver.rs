//! Main partial differential equation (PDE) solver for the cosmological
//! thermalization problem.
//!
//! Key physics: energy injection heats electrons (T_e > T_z), which drives a
//! Kompaneets-sourced y-type distortion; double Compton (DC) and
//! bremsstrahlung (BR) then create photons that convert y → μ and approach
//! Bose-Einstein equilibrium.
//!
//! The solver evolves the single Kompaneets operator at the full electron
//! temperature θ_e = θ_z ρ_e, with ρ_e as an extra Newton unknown coupled to
//! the photon field; the (φ−1)·n_pl(1+n_pl) term (φ = 1/ρ_e) carries the
//! heating. Conceptually this decomposes into conservative redistribution at
//! equilibrium plus an injection source ∝ (ρ_e − ρ_eq), but the
//! implementation does not operator-split along that line.
//!
//! References:
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294

use crate::bremsstrahlung::{br_emission_coefficient_expc, br_precompute, gaunt_expc_factor};
use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::double_compton::{dc_high_freq_suppression, dc_prefactor};
use crate::electron_temp::ElectronTemperature;
use crate::energy_injection::InjectionScenario;
use crate::grid::{FrequencyGrid, GridConfig};
use crate::kompaneets::{KompaneetsWorkspace, kompaneets_step_coupled_inplace};
use crate::recombination::RecombinationHistory;
use crate::spectrum::planck;

/// Smallest `GridConfig::n_points` the solver treats as tested. Smaller grids
/// run, but the solver warns (R-1). `GridConfig::fast()` sits exactly here.
pub const MIN_TESTED_GRID_POINTS: usize = 1000;

/// Relative tolerance of the post-run energy-closure check (R-1): the solver
/// warns when the final Δρ/ρ misses the injected heat by more than this
/// fraction of it. Since ADR 0004, bursts at z_h = 1e3 to 2e6 deliver their
/// heat to within 0.7% at Δτ_max = 10 (`dev/audit/fix_a_cn_old_half_ab.md`).
pub const ENERGY_CLOSURE_REL_TOL: f64 = 0.05;

/// Absolute slack added to the energy-closure tolerance. It covers the
/// adiabatic-cooling baseline that every run carries whatever the injection:
/// Δρ/ρ = −4.9e-9 from z = 5e6 to 500 and −5.2e-9 to z = 1 (measured with a
/// null burst, N = 2000), so injections below ~1e-7 are not flagged for it.
pub const ENERGY_CLOSURE_ABS_FLOOR: f64 = 1e-8;

/// Redshift below which adiabatic cooling of the gas keeps a significant part
/// of injected heat from the photons. See [`ENERGY_CHECK_MAX_LATE_FRACTION`].
///
/// After recombination, Compton scattering hands a gas temperature excess to
/// the photons on a time that grows from 3e-5 Hubble times at z = 1000 to
/// 1e-2 at z = 500 and 0.25 at z = 200, and the excess cools adiabatically
/// in the meantime. An independent integration of that balance
/// (`dev/scripts/heatloss/heat_delivery_expectation.py`, narrow bursts with
/// σ_z = 0.04 z_h, run to z = 200) gives the fraction of heat that never
/// reaches the photons: 6.5e-5 at z_h = 1000, 6.4e-3 at 600, 1.4e-2 at 500,
/// and 0.12 at 300. Heat injected above z = 600 therefore loses at most 0.7%,
/// far inside [`ENERGY_CLOSURE_REL_TOL`]. Full table, including decays and
/// z_end = 100: `dev/audit/heat_delivery_near_recombination.md`.
pub const ENERGY_CHECK_Z_LATE: f64 = 600.0;

/// When more than this fraction of the gross injected heat falls below
/// [`ENERGY_CHECK_Z_LATE`], the energy-closure check allows all of that late
/// heat to go missing: its heating part widens the tolerance on the
/// shortfall side, and its cooling part on the excess side.
///
/// Heat below z = 600 can be lost almost entirely. Of a burst centered on
/// z_end = 200, 88% of the heat injected before z_end has not reached the
/// photons (most of it is still in the gas); a burst at z_h = 200 run to
/// z = 100 loses 41%. The rule therefore assumes the worst case. Below this
/// fraction the check keeps the plain tolerance: 5%, minus the 0.7% physical
/// loss above z = 600, minus the numerical delivery error after ADR 0004 (up
/// to 0.6% at z_h = 1e6 to 2e6, N = 4000, Δτ_max = 10), leaves 3.7% for
/// late heat lost. 3% keeps a margin.
pub const ENERGY_CHECK_MAX_LATE_FRACTION: f64 = 0.03;

/// Upper guard on ρ_e = T_e/T_z in the predictor, the coupled Newton solve,
/// and the DC/BR target temperature (ADR 0007).
///
/// This is a sanity guard, not physics. Heating q holds the electrons at
/// ρ_e − 1 ≈ q t_C / (4 θ_z) above the photons, so a narrow burst can drive
/// ρ_e far above 1: 5 at z_h = 5000 (Δρ/ρ = 3e-3) and 1600 at z_h = 1000
/// (1e-3). The earlier caps (1.5, 3, and 2) deleted the heat the electrons
/// could not hold, up to 99% of it; at this guard those runs close their
/// energy budget. Nothing in the solver overflows at large ρ_e, so reaching
/// the guard points to a failed solve. At the guard θ_e = 4.6e-3 at
/// z = 1000, near the limit of the nonrelativistic Kompaneets equation.
pub const RHO_E_GUARD_MAX: f64 = 1e4;

/// ρ_e above which a run below [`HOT_GAS_Z_MAX`] warns that the gas is too
/// hot for the fixed ionization history (ADR 0007).
///
/// The solver holds X_e on the standard recombination history. At z = 850,
/// ρ_e = 10 means T_e ≈ 2.3e4 K: the case-B recombination coefficient has
/// fallen by about 85%, and collisional ionization of hydrogen has set in.
/// The threshold is a judgment call, not a derived bound.
pub const HOT_GAS_RHO_E: f64 = 10.0;

/// Redshift below which [`HOT_GAS_RHO_E`] applies. Above about z = 1500
/// hydrogen stays ionized however hot the electrons are. Hot electrons would
/// still slow He I recombination (z ≈ 1600 to 2500), which the solver takes
/// from Saha at T_z; that changes n_e by at most about 8% and does not warn.
pub const HOT_GAS_Z_MAX: f64 = 1500.0;

/// Returns how far the final Δρ/ρ may fall below (first) and rise above
/// (second) the expected energy before the energy-closure check warns (R-1).
///
/// `gross` is the gross injected heat ∫|dq/dz| dz of the run; `late_abs` and
/// `late_net` are the gross and net heat injected below
/// [`ENERGY_CHECK_Z_LATE`]. Both bounds start at [`ENERGY_CLOSURE_REL_TOL`]
/// of `gross` plus [`ENERGY_CLOSURE_ABS_FLOOR`]. If `late_abs` exceeds
/// [`ENERGY_CHECK_MAX_LATE_FRACTION`] of `gross`, the late heating part,
/// (late_abs + late_net)/2, widens the shortfall bound, since the gas may keep
/// all of it from the photons, and the late cooling part,
/// (late_abs − late_net)/2, widens the excess bound for the same reason.
fn energy_closure_allowance(gross: f64, late_abs: f64, late_net: f64) -> (f64, f64) {
    let tol = ENERGY_CLOSURE_REL_TOL * gross + ENERGY_CLOSURE_ABS_FLOOR;
    let (late_heat, late_cool) = if late_abs > ENERGY_CHECK_MAX_LATE_FRACTION * gross {
        (
            (0.5 * (late_abs + late_net)).max(0.0),
            (0.5 * (late_abs - late_net)).max(0.0),
        )
    } else {
        (0.0, 0.0)
    };
    (tol + late_heat, tol + late_cool)
}

/// Tunable solver parameters.
///
/// Defaults are production-quality and match the values used for all paper
/// runs. The fields most callers want to change are [`Self::z_start`],
/// [`Self::z_end`], and [`Self::dtau_max`].
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Upper redshift at which integration begins (solver evolves from
    /// `z_start` down to `z_end`). Must satisfy `z_start > z_end`.
    pub z_start: f64,
    /// Lower redshift at which integration ends.
    pub z_end: f64,
    /// Maximum Compton y-parameter increment per step, Δy_C = θ_e Δτ (limits step size for
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
    /// Default 10.0 matches the command-line interface (CLI) default (the
    /// value used for all paper runs). Raise to ~50 for exploratory runs
    /// where modest accuracy loss is acceptable.
    pub dtau_max: f64,
    /// Minimum redshift for number-conserving T-shift subtraction.
    /// Default 5e4: only subtract at z > nc_z_min where DC/BR is significant.
    /// Set to 0.0 to subtract at all redshifts (CosmoTherm-like behavior).
    pub nc_z_min: f64,
    /// Maximum dtau per step during active photon source injection.
    /// Limits the Kompaneets Δn² nonlinearity at the injection spike,
    /// which causes timestep-dependent wake formation at intermediate x.
    /// Default 1.0 (production quality, ~1% from converged).
    /// Set to 10.0 for exploratory and fast runs (~3% from converged).
    pub dtau_max_photon_source: f64,
    /// Maximum number of Newton iterations per Kompaneets step.
    /// Default 10. Increase for extreme injection parameters.
    pub max_newton_iter: usize,
}

impl SolverConfig {
    /// Validates solver configuration parameters.
    ///
    /// Returns `Err` with a descriptive message if any parameter would cause
    /// the solver to fail or produce meaningless results.
    pub fn validate(&self) -> Result<(), String> {
        if !self.z_start.is_finite() || self.z_start <= 0.0 {
            return Err(format!("z_start must be positive, got {}", self.z_start));
        }
        // z_end = 0 is rejected: the adaptive step shrinks with z, so the run
        // never reaches z = 0 and stops at the step limit instead (R-2).
        if !self.z_end.is_finite() || self.z_end <= 0.0 {
            return Err(format!("z_end must be positive, got {}", self.z_end));
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
        // Guardrail on the upper end: dy_max controls the Compton-y increment
        // θ_e Δτ per step, and the adaptive-step derivation assumes this
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
        // The step limit applies only when the cap is positive, so a zero,
        // negative, or NaN cap would switch it off silently (R-2).
        if !self.dtau_max_photon_source.is_finite() || self.dtau_max_photon_source <= 0.0 {
            return Err(format!(
                "dtau_max_photon_source must be positive, got {}",
                self.dtau_max_photon_source
            ));
        }
        // 0 is valid and documented: subtract the T-shift at all redshifts.
        if !self.nc_z_min.is_finite() || self.nc_z_min < 0.0 {
            return Err(format!(
                "nc_z_min must be non-negative, got {}",
                self.nc_z_min
            ));
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

    /// Collects non-fatal validation warnings (for example, regimes where the
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
    /// Fractional energy `Δρ / ρ` measured in the spectrum at this redshift:
    /// `∫ x³ Δn dx / G₃` plus `4 ΔT/T` for the accumulated temperature shift.
    /// This is the energy the solver holds, not the scenario's nominal
    /// injected energy. The two differ by the solver's energy-conservation
    /// error and by adiabatic cooling, which removes energy even when
    /// nothing is injected.
    pub delta_rho_over_rho: f64,
    /// Cumulative temperature shift `ΔT/T` subtracted by the number-conserving
    /// machinery (non-zero only when `number_conserving` is enabled).
    pub accumulated_delta_t: f64,
}

impl SolverSnapshot {
    /// Computes the brightness-temperature deviation `T(x)/T_CMB − 1` on the given grid.
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

/// Cached backward-Euler ordinary differential equation (ODE) coefficients
/// for the ρ_e equation.
///
/// Computed once per step in `update_temperatures()` and consumed by
/// the bordered Newton solver, which couples ρ_e into the Kompaneets iteration.
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
}

/// Diagnostic counters and extrema tracked during solver evolution.
///
/// Grouped into a sub-struct to keep `ThermalizationSolver` focused on
/// physics state. Reset by `ThermalizationSolver::reset()`.
#[derive(Debug, Clone, Default)]
pub struct SolverDiagnostics {
    /// Number of steps on which ρ_e left the guard range
    /// [0, [`RHO_E_GUARD_MAX`]] or came out non-finite. The first such step
    /// in a run pushes a warning (see `ThermalizationSolver::guard_rho_e`);
    /// later ones only increment this counter.
    pub rho_e_clamped: usize,
    /// Number of times Newton iteration exhausted max_newton_iter
    /// without converging. Non-zero values indicate the solver may need more
    /// iterations or smaller step sizes.
    pub newton_exhausted: usize,
    /// Whether any NaN emission rate was encountered.
    pub nan_emission_detected: bool,
    /// Number of steps on which the DC/BR target temperature ρ_e fell outside
    /// the guard range [0.05, [`RHO_E_GUARD_MAX`]] and was clamped. The
    /// first such step pushes a warning; later ones only increment this
    /// counter.
    pub dcbr_target_clamped: usize,
    /// Whether the once-per-run warning for ρ_e reaching
    /// [`RHO_E_GUARD_MAX`] has been pushed.
    pub(crate) rho_e_guard_warned: bool,
    /// Whether the once-per-run warning for gas hotter than
    /// [`HOT_GAS_RHO_E`] below [`HOT_GAS_Z_MAX`] has been pushed.
    pub(crate) hot_gas_warned: bool,
    /// Whether the once-per-run warning for a negative or non-finite ρ_e has
    /// been pushed.
    pub(crate) rho_e_invalid_warned: bool,
    /// Whether the once-per-run DC/BR target-guard warning has been pushed.
    pub(crate) dcbr_target_warned: bool,
    /// Step on which `guard_rho_e` last saw a negative or non-finite ρ_e, so
    /// the DC/BR target guard does not warn a second time for the same event.
    pub(crate) rho_e_invalid_step: Option<usize>,
    /// Warning messages collected during solver evolution.
    /// Replaces eprintln! in library code for structured diagnostics.
    pub warnings: Vec<String>,
    /// Set when the run stopped early because Δn became NaN or infinite.
    /// [`ThermalizationSolver::try_run_with_snapshots`] and
    /// [`ThermalizationSolver::try_run_to_result`] return it as `Err`; the
    /// other run methods and [`ThermalizationSolver::step`] panic with it.
    pub failure: Option<String>,
}

/// Full PDE solver for cosmic microwave background (CMB) spectral distortions.
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
///   → snapshots[i].{mu, y, accumulated_delta_t, ...} // read results
/// ```
///
/// The solver can be re-run after calling `reset()`, which clears Δn and
/// diagnostic counters and restores `config` and all mode flags to their
/// defaults, keeping only the cosmology and grid (see [`Self::reset`]).
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
/// tooling. These are not part of the stable public API and may become
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
    /// Base point count `GridConfig::n_points` the grid was built from,
    /// before refinement zones add points. Drives the small-grid warning.
    grid_n_points: usize,
    /// Current photon occupation-number perturbation `Δn(x)` on the grid.
    pub delta_n: Vec<f64>,
    /// Electron temperature state `ρ_e = T_e/T_z` (perturbative
    /// quasi-stationary solve).
    pub electron_temp: ElectronTemperature,
    /// Current redshift of the integration.
    pub z: f64,
    /// Active injection scenario, if any. Set with [`Self::set_injection`].
    pub injection: Option<InjectionScenario>,
    /// Snapshots collected during the last run. Populated by
    /// [`Self::run_with_snapshots`] / [`Self::run`].
    pub snapshots: Vec<SolverSnapshot>,
    /// Number of timesteps taken in the current run.
    pub step_count: usize,
    /// Compton equilibrium ρ_eq = I₄/(4G₃) from the photon spectrum.
    rho_eq: f64,
    /// Cached recombination history for fast X_e(z) lookups.
    recomb: RecombinationHistory,
    /// Initial photon perturbation Δn(x) to use instead of zeros.
    /// Consumed (taken) by run_with_snapshots on first call.
    initial_delta_n: Option<Vec<f64>>,
    /// If true, disable DC/BR processes (Kompaneets only). For diagnostics.
    pub disable_dcbr: bool,
    /// If true (default), couple DC/BR into the Kompaneets Newton iteration
    /// instead of operator splitting. Uses an implicit-explicit (IMEX) scheme:
    /// Crank-Nicolson for Kompaneets and backward Euler for DC/BR, solved
    /// simultaneously. More physically
    /// consistent than operator splitting, especially at z > 2×10⁶.
    pub coupled_dcbr: bool,
    /// Pre-allocated work buffer for DC/BR emission rates (per step).
    emission_rates: Vec<f64>,
    /// Pre-allocated work buffer for DC/BR equilibrium minus Planck.
    n_eq_minus_n_pl: Vec<f64>,
    /// d(emission_rates)/d(ρ_eq), analytical. Used by the bordered Newton
    /// c-vector to close the Δn-row Jacobian on ρ_e (otherwise the solve
    /// is only linearly convergent in the ρ_e direction when DC/BR is
    /// strong, for example at z ≳ 10⁶ or during a photon-injection burst).
    dem_drho_eq: Vec<f64>,
    /// d(n_eq_minus_n_pl)/d(ρ_eq), analytical. See `dem_drho_eq`.
    dneq_drho_eq: Vec<f64>,
    /// Pre-allocated Kompaneets workspace (grid-constant arrays and per-step buffers).
    komp_ws: KompaneetsWorkspace,
    /// Precomputed Planck spectrum on the grid: planck(x[i]).
    planck_grid: Vec<f64>,
    /// Subtract the temperature shift component from Δn after each
    /// DC/BR step at z > 5×10⁴, enforcing photon number conservation
    /// (∫x² Δn dx = 0). This prevents DC/BR-created photons from
    /// accumulating as a spurious temperature shift that feeds back into
    /// over-thermalization. See Chluba (2013), arXiv:1304.6120.
    /// Enabled by default.
    pub number_conserving: bool,
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
    /// Precomputed Planck occupation at cell midpoints x_half[i] = (x[i]+x[i+1])/2.
    /// Used by the DC/BR heating integral, which evaluates n_pl at midpoints.
    planck_half: Vec<f64>,
    /// Precomputed 1/x³ for each grid point (DC/BR rate normalization).
    inv_x3_grid: Vec<f64>,
    /// Precomputed grid-constant half of the BR Gaunt exponential, x^(-√3/π),
    /// for each grid point. The Gaunt factor needs x_e = x/ρ_e; the remaining
    /// factor ρ_e^(√3/π) lives in the per-step `BrPrecomputed::ea_z*`.
    /// See [`crate::bremsstrahlung::gaunt_expc_factor`].
    expc_grid: Vec<f64>,
    /// Same at cell midpoints (for the BR Gaunt factor in the heating integral).
    expc_half: Vec<f64>,
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

/// Computes DC+BR heating integral and optionally its analytic derivative
/// dH/dρ_e in a single pass over the frequency grid.
///
/// The derivative differentiates the integrand B·(K_DC + K_BR) at fixed Δn,
/// densities, and He ionization fractions (exactly what the former central
/// finite difference held fixed):
///   dB/dρ_e     = n_mid · x φ² · e^{xφ}   with B = 1 − n_mid(e^{xφ} − 1)
///   dK_DC/dρ_e  = 0                        (K_DC depends only on θ_z)
///   dK_BR/dρ_e  — see `br_emission_coefficient_and_drho_expc`.
/// e^{xφ} reuses the exp_m1 already computed for B, and the Gaunt logistic
/// shares its exp with the softplus, so the derivative costs no transcendental
/// calls beyond the plain heating integral — it replaces the former three
/// full-integral evaluations (θ_e, θ_e(1±10⁻⁴)) with one.
///
/// Returns (h_dc_br, dh_drho).
fn dcbr_heating_with_derivative(
    x_grid: &[f64],
    x_half: &[f64],
    dx_arr: &[f64],
    planck_half: &[f64],
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
    expc_half: &[f64],
    dc_supp_half: &[f64],
) -> (f64, f64) {
    use crate::bremsstrahlung::{
        br_emission_coefficient_and_drho_expc, br_emission_coefficient_expc, br_precompute,
    };

    if theta_e < 1e-30 || n_e < 1e-30 {
        return (0.0, 0.0);
    }

    let phi = theta_z / theta_e;
    let norm = 1.0 / (4.0 * G3_PLANCK * theta_z);
    let n = x_grid.len();

    // Single pass: compute h_dc_br at current θ_e, and optionally dH/dρ_e
    let mut integral = 0.0;
    let mut d_integral = 0.0;

    // Historical guard from the finite-difference era: the FD stencil needed
    // θ_e(ρ_e − 10⁻⁴) ≥ 10⁻³⁰ for its minus-side `br_precompute` to succeed,
    // and returned dh = 0 otherwise (ρ_e ≲ 10⁻⁴, extremely uncommon). Kept
    // so the analytic derivative fires on exactly the same steps.
    let delta_rho_fd = 1e-4;
    let rho_e = theta_e / theta_z;

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

    let want_derivative = compute_derivative && theta_z * (rho_e - delta_rho_fd) >= 1e-30;

    if want_derivative {
        let phi2 = phi * phi;
        for i in 1..n {
            // dx, x_mid and planck(x_mid) are grid-constant: `dx_arr[i-1]` and
            // `x_half[i-1]` are defined as exactly these expressions in
            // `FrequencyGrid::new`, and `planck_half` caches the Planck
            // occupation there. Hoisting them out of the step loop removes one
            // exp() per cell per step.
            let dx = dx_arr[i - 1];
            let x_mid = x_half[i - 1];
            let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
            let n_mid = planck_half[i - 1] + dn_mid;
            let ec = expc_half[i - 1];

            let k_dc = dc_pre * dc_supp_half[i - 1];
            let (k_br, dk_br) = br_emission_coefficient_and_drho_expc(x_mid, ec, &br_pre);

            let x_e = x_mid * phi;
            let em = x_e.exp_m1();
            let bose = 1.0 - n_mid * em;

            integral += bose * (k_dc + k_br) * dx;
            // dB/dρ_e = n_mid · x_mid φ² · e^{x_e}, with e^{x_e} = em + 1.
            d_integral += (n_mid * x_mid * phi2 * (em + 1.0) * (k_dc + k_br) + bose * dk_br) * dx;
        }
    } else {
        for i in 1..n {
            let dx = dx_arr[i - 1];
            let x_mid = x_half[i - 1];
            let dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1]);
            let n_mid = planck_half[i - 1] + dn_mid;
            let ec = expc_half[i - 1];

            let k_dc = dc_pre * dc_supp_half[i - 1];
            let k_br = br_emission_coefficient_expc(x_mid, ec, &br_pre);

            let x_e = x_mid * phi;
            let factor = 1.0 - n_mid * x_e.exp_m1();
            integral += factor * (k_dc + k_br) * dx;
        }
    }

    let h = integral * norm;
    let dh = d_integral * norm;

    (h, dh)
}

impl ThermalizationSolver {
    /// Constructs a solver with the given cosmology and frequency grid.
    ///
    /// All other state (solver config, injection, flags) is set to defaults;
    /// mutate the public fields or call [`Self::set_injection`] /
    /// [`Self::set_config`] before running. For a fluent API that performs
    /// validation at build time, use [`Self::builder`] instead.
    pub fn new(cosmo: Cosmology, grid_config: GridConfig) -> Self {
        let grid = FrequencyGrid::new(&grid_config);
        let grid_n_points = grid_config.n_points;
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
        // ln(x) feeds only the BR Gaunt factor, and it enters there solely
        // through x^(-√3/π). Storing that factored form instead of ln(x)
        // removes both exp() calls from every Gaunt evaluation; the -69.0
        // clamp is preserved so the sentinel reaching `gaunt_expc_factor` is
        // unchanged (note `< -69.0` is strict, so the clamped value does not
        // itself trip the guard — same as before). The Gaunt fit needs
        // x_e = x/ρ_e; `br_precompute` folds the step's ρ_e^(√3/π) into ea_Z,
        // so this grid factor stays in the grid x (P-1).
        let expc_grid: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| gaunt_expc_factor(if x < 1e-30 { -69.0 } else { x.ln() }))
            .collect();
        let expc_half: Vec<f64> = grid
            .x_half
            .iter()
            .map(|&x| gaunt_expc_factor(if x < 1e-30 { -69.0 } else { x.ln() }))
            .collect();
        let planck_half: Vec<f64> = grid.x_half.iter().map(|&x| planck(x)).collect();
        let inv_x3_grid: Vec<f64> = grid.x.iter().map(|&x| 1.0 / (x * x * x)).collect();
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
            grid_n_points,
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
            number_conserving: true,
            accumulated_delta_t: 0.0,
            g_bb_grid,
            g2_gbb_discrete,
            diag: SolverDiagnostics::default(),
            rho_e_ode_cache: None,
            dc_suppression_grid,
            dc_suppression_half,
            exp_m1_grid,
            exp_grid,
            planck_half,
            inv_x3_grid,
            expc_grid,
            expc_half,
            quad_weights_x2,
            photon_source_buf: vec![0.0; n],
        }
    }

    /// Attaches an energy-injection scenario, validating it first.
    ///
    /// Returns `Err` if the scenario parameters are unphysical (such as negative
    /// widths, impossible masses). Collects stimulated-emission warnings into
    /// [`SolverDiagnostics::warnings`].
    pub fn set_injection(&mut self, scenario: InjectionScenario) -> Result<(), String> {
        scenario.validate()?;

        for warning in scenario.warn_stimulated_emission() {
            self.diag.warnings.push(warning);
        }
        #[cfg(feature = "axion")]
        if matches!(scenario, InjectionScenario::AxionResonance { .. }) {
            self.diag
                .warnings
                .push(crate::axion::EXPERIMENTAL_WARNING.to_string());
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

    /// Replaces the solver configuration and reset the current redshift to
    /// the new `z_start`.
    pub fn set_config(&mut self, config: SolverConfig) {
        self.z = config.z_start;
        self.config = config;
    }

    /// Sets an initial photon perturbation Δn(x) for the next PDE run.
    ///
    /// This replaces the default Δn = 0 initialization in `run_with_snapshots`.
    /// The perturbation is consumed (taken) on the next call to `run_with_snapshots`.
    ///
    /// Panics if any entry is non-finite — silently passing NaN/Inf into the
    /// solver causes a deep panic in the Newton step where the source isn't
    /// obvious. Caught early here so the caller sees the real culprit.
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

    /// Resets solver state for reuse, keeping grid and recombination cache.
    ///
    /// Restores all configuration flags to their defaults: a reset solver
    /// behaves identically to a freshly constructed one (aside from the
    /// cached grid and recombination tables, which are preserved for speed).
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
        let burst = self.injection.as_ref().and_then(|inj| inj.gaussian_burst());
        let is_photon_source = self
            .injection
            .as_ref()
            .is_some_and(|inj| inj.has_photon_source());
        if let Some((z_h, sigma_z)) = burst {
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

    /// Applies the guard range to a new electron temperature ρ_e and records
    /// the event.
    ///
    /// Returns the value to store, clamped to [0, [`RHO_E_GUARD_MAX`]], or
    /// `None` when `raw` is non-finite (the caller keeps the prior ρ_e). Every
    /// clamp or non-finite value increments `diag.rho_e_clamped`.
    ///
    /// The upper guard only catches a failed solve (ADR 0007). Heating that drives
    /// ρ_e past [`HOT_GAS_RHO_E`] below [`HOT_GAS_Z_MAX`] pushes a separate
    /// once-per-run warning, because gas that hot would change the ionization
    /// history, which the solver holds fixed. In coupled mode the predictor
    /// passes through here before the Newton solve replaces it, so a
    /// predictor above the threshold can warn when the final ρ_e stays just
    /// below it. A negative or non-finite ρ_e pushes its own once-per-run
    /// warning.
    fn guard_rho_e(&mut self, raw: f64, z: f64) -> Option<f64> {
        let out = if raw.is_finite() {
            Some(raw.clamp(0.0, RHO_E_GUARD_MAX))
        } else {
            None
        };
        if let Some(rho) = out
            && rho > HOT_GAS_RHO_E
            && z < HOT_GAS_Z_MAX
            && !self.diag.hot_gas_warned
        {
            self.diag.hot_gas_warned = true;
            self.diag.warnings.push(format!(
                "Hot gas: at z = {z:.4e} the electron temperature reached T_e/T_z = \
                 {rho:.3e}, above {HOT_GAS_RHO_E}. Gas this hot would recombine more slowly \
                 and start to ionize by collisions, which changes the ionization history; \
                 the solver holds X_e on the standard history, so results may be \
                 inaccurate."
            ));
        }
        if out == Some(raw) {
            return out;
        }
        self.diag.rho_e_clamped += 1;
        if raw.is_finite() && raw > RHO_E_GUARD_MAX {
            if !self.diag.rho_e_guard_warned {
                self.diag.rho_e_guard_warned = true;
                self.diag.warnings.push(format!(
                    "The electron temperature solve gave T_e/T_z = {raw:.4e} at z = {z:.4e}, \
                     above the sanity guard {RHO_E_GUARD_MAX:e}. T_e is clamped at the \
                     guard on this and any later such step, which loses heat. This points \
                     to a numerical failure; do not trust μ and y from this run."
                ));
            }
        } else {
            // Negative or non-finite (+∞ included): this step's event, for the
            // DC/BR target guard below.
            self.diag.rho_e_invalid_step = Some(self.step_count);
            if self.diag.rho_e_invalid_warned {
                return out;
            }
            self.diag.rho_e_invalid_warned = true;
            self.diag.warnings.push(format!(
                "The electron temperature solve gave T_e/T_z = {raw:.4e} at z = {z:.4e}; a \
                 negative value is clamped to 0 and a non-finite one is replaced by the \
                 previous step's value. Results may be inaccurate."
            ));
        }
        out
    }

    /// Updates ρ_e from distortion feedback and injection.
    ///
    /// Two modes selected automatically by distortion amplitude:
    /// - **Small Δn** (|ΔG₃/G₃| ≤ 0.1): Perturbative Δρ_eq = ΔI₄/(4G₃) - ΔG₃/G₃.
    ///   Avoids the 0.1% cancellation error in the full I₄/(4G₃) computation.
    /// - **Large Δn** (|ΔG₃/G₃| > 0.1): Exact ρ_eq = I₄/(4G₃) using the full
    ///   occupation number n = n_pl + Δn. Retains the Δn² term in I₄ and the
    ///   nonlinear denominator. Necessary for strong depletions (for example, dark photon
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
        }
        if !max_dn.is_finite() {
            // Leave the state untouched; step_with_dz sees the non-finite
            // max|Δn| and returns None without advancing z.
            self.record_nonfinite_delta_n();
            return (0.0, 0.0, 0.0, max_dn, 0.0, 0.0);
        }

        let delta_rho_eq = if max_dn > 1e-15 {
            // Use exact computation when the distortion is large enough that
            // the perturbative expansion (which drops Δn² in I₄ and linearizes
            // 1/(G₃+ΔG₃)) becomes inaccurate. The exact computation has ~0.1%
            // discretization error from computing I₄/(4G₃), which is negligible
            // for |ΔG₃/G₃| > 0.1 but catastrophic for tiny distortions (~10⁻⁵).
            // The exact moments are needed only on this rare branch, so they
            // get their own pass rather than being accumulated every step.
            // Same summation order as before, hence the same value.
            if delta_g3.abs() / G3_PLANCK > 0.1 {
                let mut exact_i4 = 0.0;
                let mut exact_g3 = 0.0;
                for i in 1..self.grid.n {
                    let dx = self.grid.dx[i - 1];
                    let x_half = self.grid.x_half[i - 1];
                    let x3 = self.grid.x_half_cubed[i - 1];
                    let dn_mid = 0.5 * (self.delta_n[i] + self.delta_n[i - 1]);
                    let n_pl = 0.5 * (self.planck_grid[i] + self.planck_grid[i - 1]);
                    let n_full = (n_pl + dn_mid).max(0.0);
                    exact_g3 += x3 * n_full * dx;
                    exact_i4 += x3 * x_half * n_full * (1.0 + n_full) * dx;
                }
                if exact_g3 > 1e-30 {
                    exact_i4 / (4.0 * exact_g3) - 1.0
                } else {
                    delta_i4 / (4.0 * G3_PLANCK) - delta_g3 / G3_PLANCK
                }
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
                    &self.grid.x_half,
                    &self.grid.dx,
                    &self.planck_half,
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
                    &self.expc_half,
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
            });

            let numerator =
                rho_e_old + dtau * r_compton * (rho_source - h_dc_br + dh_drho * rho_e_old);
            let denominator = 1.0 + dtau * (r_compton * (1.0 + dh_drho) + lambda_htc);
            let rho_e_raw = numerator / denominator;
            // Post-recombination ρ_e can drop well below 1 (T_m ∝ (1+z)²),
            // and narrow bursts can push it far above 1. The guard range is an
            // sanity guard shared with the coupled solve (ADR 0007).
            //
            // NaN/∞ are rejected explicitly (audit H1): f64::NaN.clamp(a, b) ==
            // NaN and NaN comparisons return false, so a naïve clamp would
            // silently propagate a degenerate result into later solves.
            //
            // A clamp or a non-finite value is counted and warned about once
            // per run (`guard_rho_e`).
            if let Some(rho_e_new) = self.guard_rho_e(rho_e_raw, z_eval) {
                self.electron_temp.rho_e = rho_e_new;
            }
            // `None`: keep the prior ρ_e rather than poisoning the solver state.
        } else {
            // θ_z too small for meaningful Compton physics.
            self.rho_e_ode_cache = None;
        }

        (x_e, t_c, theta_z_val, max_dn, dtau, hubble)
    }

    /// Subtracts the temperature shift component from Δn to enforce
    /// photon number conservation: ∫x² Δn dx = 0.
    ///
    /// DC/BR creates photons at low x that accumulate as a temperature
    /// shift in Δn. This growing T-shift feeds back: as ρ_e rises, the DC/BR
    /// equilibrium target rises, producing more photon creation and a larger T-shift.
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

    /// Advances the solver by a single adaptively-chosen timestep.
    ///
    /// Returns the `dz` taken. Most callers should call [`Self::run`] or
    /// [`Self::run_with_snapshots`] instead of stepping manually.
    ///
    /// # Panics
    ///
    /// Panics if Δn becomes NaN or infinite (see [`SolverDiagnostics::failure`]).
    pub fn step(&mut self) -> f64 {
        let dz = self.adaptive_dz();
        match self.step_with_dz(dz) {
            Some(dz) => dz,
            None => panic!("{}", self.diag.failure.as_deref().unwrap_or_default()),
        }
    }

    /// Records the NaN/Inf failure in [`SolverDiagnostics::failure`] for a
    /// Δn that is no longer finite at the current z.
    fn record_nonfinite_delta_n(&mut self) {
        let hint = if self.grid_n_points < MIN_TESTED_GRID_POINTS {
            format!(
                "the frequency grid is too coarse (n_points={}; use at least \
                 {MIN_TESTED_GRID_POINTS})",
                self.grid_n_points
            )
        } else {
            format!(
                "time steps too large (reduce dtau_max) or a strong injection \
                 (n_points={})",
                self.grid_n_points
            )
        };
        self.diag.failure = Some(format!(
            "NaN/Inf detected in delta_n at z={:.4e} (after step {}); the run stopped. \
             Likely cause: {hint}.",
            self.z, self.step_count
        ));
    }

    /// Takes a single timestep with a specified dz (instead of the adaptive choice).
    /// Used by `run_with_snapshots` to land exactly on requested snapshot redshifts.
    /// Returns the dz taken, or `None` without advancing if Δn is not finite
    /// (the message is then in [`SolverDiagnostics::failure`]).
    fn step_with_dz(&mut self, dz: f64) -> Option<f64> {
        let z_new = (self.z - dz).max(self.config.z_end);
        let actual_dz = self.z - z_new;
        let z_mid = self.z - 0.5 * actual_dz;

        // update_temperatures computes ρ_e via backward Euler and returns dtau + hubble
        let (x_e, _t_c, theta_z_val, max_dn_abs, dtau, h) =
            self.update_temperatures(z_mid, actual_dz);
        if !max_dn_abs.is_finite() {
            return None;
        }

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
            // DC/BR equilibrium: Planck at the actual electron temperature,
            // n → n_pl(x/ρ_e) (Chluba & Sunyaev 2012, Eq. 8). See
            // decisions/0002-relax-dc-br-toward-actual-electron-temperature.md.
            //
            // Energy bookkeeping: the photon rows gain Δτ·em·(neq − Δn) from
            // DC/BR, and the electron row loses H_dcbr, which is built from the
            // same `em` and `neq` arrays (`kompaneets.rs`, the ρ_e residual).
            // Whatever the target, the energy DC/BR give the photons is the
            // energy the electrons lose, so using the full ρ_e, including the
            // heating excess δρ_inj, does not count injected energy twice.
            //
            // The clamp to [0.05, RHO_E_GUARD_MAX] guards the Bose factor
            // exp(x/ρ) − 1. The upper bound does not engage, since
            // `guard_rho_e` already keeps ρ_e at or below it. Adiabatic cooling takes ρ_e below 0.5
            // near z ≈ 72 but not below 0.05 before DC/BR switch off at
            // θ_z = 1e-8 (z ≈ 21), so the lower bound only catches a broken
            // ρ_e. When the guard engages, the first occurrence in a run
            // pushes a warning, except on a step where `guard_rho_e` already
            // reported a negative or non-finite ρ_e; later ones only
            // increment the counter.
            let rho_eq_dcbr = if (0.05..=RHO_E_GUARD_MAX).contains(&rho_e) {
                rho_e
            } else {
                self.diag.dcbr_target_clamped += 1;
                let same_event = self.diag.rho_e_invalid_step == Some(self.step_count);
                if !self.diag.dcbr_target_warned && !same_event {
                    self.diag.dcbr_target_warned = true;
                    self.diag.warnings.push(format!(
                        "DC/BR target temperature ρ_e = {rho_e:.4} at z = {:.4e} is outside \
                         the guard range [0.05, {RHO_E_GUARD_MAX:e}] and was clamped; DC/BR \
                         emission and absorption use the clamped value on this and any later \
                         such step (count in SolverDiagnostics::dcbr_target_clamped).",
                        z_mid
                    ));
                }
                rho_e.clamp(0.05, RHO_E_GUARD_MAX)
            };
            let delta_rho = rho_eq_dcbr - 1.0;
            let inv_rho_eq = 1.0 / rho_eq_dcbr;

            // Hoist struct-field slices and scalars so LLVM can register-allocate
            // them and elide bounds checks inside the grid loop. Splitting on
            // `in_taylor` specialises the hot |δρ|<0.01 path into a straight-line
            // loop with no per-point branch on `delta_rho`.
            let in_taylor = delta_rho.abs() < 0.01;
            let mut any_nan = false;

            // Borrow the read-only and mut slices once, up-front. Disjoint
            // fields → the borrow checker accepts this.
            let xs: &[f64] = &self.grid.x[..n];
            let dc_supp: &[f64] = &self.dc_suppression_grid[..n];
            let inv_x3: &[f64] = &self.inv_x3_grid[..n];
            let expc: &[f64] = &self.expc_grid[..n];
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
                // Hoist the Option out of the loop so the body has no per-point
                // branch (same pattern as dcbr_heating_with_derivative).
                let br_ref = br_pre.as_ref();
                for i in 0..n {
                    let xi = xs[i];
                    let inv_xi3 = inv_x3[i];

                    let k_dc = dc_pre * dc_supp[i];
                    let k_br = match br_ref {
                        Some(pre) => br_emission_coefficient_expc(xi, expc[i], pre),
                        None => 0.0,
                    };
                    let k_sum = k_dc + k_br;

                    let exp_xi = exp_x[i];
                    let bose_factor = exp_m1[i] - xi * delta_rho_inv * exp_xi;

                    let uncapped = k_sum * bose_factor * inv_xi3;
                    let finite = uncapped.is_finite();
                    any_nan |= uncapped.is_nan();
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
                    let dem_raw = k_sum * dbose_drho * inv_xi3;
                    dem_out[i] = if dem_raw.is_finite() { dem_raw } else { 0.0 };
                    let dneq_raw = npl_1p * xi * inv_rho_eq2;
                    dneq_out[i] = if dneq_raw.is_finite() { dneq_raw } else { 0.0 };
                }
            } else {
                // Full (non-Taylor) path, for |ρ − 1| ≥ 0.01: post-recombination
                // cooling (ρ ≈ 0.3 to 1) and strong heating (ρ up to
                // RHO_E_GUARD_MAX, ADR 0007). ρ-derivatives zeroed (see Taylor
                // branch), so the Newton Jacobian lags the ρ dependence of
                // DC/BR here; energy still closes to 0.6% (ADR 0007 table).
                let br_ref = br_pre.as_ref();
                for i in 0..n {
                    let xi = xs[i];
                    let inv_xi3 = inv_x3[i];

                    let k_dc = dc_pre * dc_supp[i];
                    let k_br = match br_ref {
                        Some(pre) => br_emission_coefficient_expc(xi, expc[i], pre),
                        None => 0.0,
                    };

                    let xe = xi * inv_rho_eq;
                    let bose_factor = if xe > 500.0 {
                        f64::INFINITY
                    } else {
                        xe.exp_m1()
                    };
                    let uncapped = (k_dc + k_br) * bose_factor * inv_xi3;
                    let finite = uncapped.is_finite();
                    any_nan |= uncapped.is_nan();
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
        }

        // Fill photon source buffer (simple — no split-source logic needed)
        let source_active = if has_phot_src {
            let dt = actual_dz / (h * (1.0 + z_mid));
            if let Some(ref inj) = self.injection {
                let step_factor = inj.photon_source_step_factor(z_mid, &self.cosmo);
                for i in 0..n {
                    self.photon_source_buf[i] = inj.photon_source_rate_with_step_factor(
                        self.grid.x[i],
                        z_mid,
                        &self.cosmo,
                        step_factor,
                    ) * dt;
                }
            }
            // Use a threshold that excludes Gaussian tails > ~8σ from peak.
            // Peak source ~ O(1e-2), so 1e-20 catches everything within ~6σ
            // but correctly disables the bordered system in the far tails.
            self.photon_source_buf.iter().any(|&v| v.abs() > 1e-20)
        } else {
            // No zeroing needed: `photon_source_buf` is only ever *read* when
            // `source_active` is true, and `source_active` can only be set on
            // the branch above, which overwrites all n entries first. Its
            // contents are therefore unobservable here, and clearing them
            // every step cost an O(n) memset per step for nothing.
            false
        };

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
                // `None` (non-finite): keep the prior ρ_e.
                if let Some(rho_clamped) = self.guard_rho_e(rho_e_out, z_mid) {
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
                let step_factor = inj.photon_source_step_factor(z_mid, &self.cosmo);
                for i in 0..n {
                    let source = inj.photon_source_rate_with_step_factor(
                        self.grid.x[i],
                        z_mid,
                        &self.cosmo,
                        step_factor,
                    );
                    if source.abs() > 1e-50 {
                        self.delta_n[i] += source * dt;
                    }
                }
            }
        }

        // Number-conserving mode: subtract temperature shift G_bb from Δn.
        // This prevents DC/BR-created low-x photons from accumulating as a
        // secular T-shift that masks the physical distortion shape.
        if self.number_conserving && self.z > self.config.nc_z_min {
            self.subtract_temperature_shift();
        }

        self.z = z_new;
        self.step_count += 1;
        Some(actual_dz)
    }

    /// Integrates from `z_start` to `z_end`, recording a snapshot at each
    /// requested redshift.
    ///
    /// `snapshot_redshifts` may be given in any order; they are sorted
    /// internally. Redshifts above `z_start` or below `z_end` are clamped
    /// to the initial and final state respectively. The returned slice
    /// borrows from `self.snapshots`; for an owned result, use
    /// [`Self::run_to_result`].
    ///
    /// # Panics
    ///
    /// Panics if Δn becomes NaN or infinite. Use
    /// [`Self::try_run_with_snapshots`] to get that failure as an `Err`.
    pub fn run_with_snapshots(&mut self, snapshot_redshifts: &[f64]) -> &[SolverSnapshot] {
        if let Err(msg) = self.try_run_with_snapshots(snapshot_redshifts) {
            panic!("{msg}");
        }
        &self.snapshots
    }

    /// Like [`Self::run_with_snapshots`], but returns `Err` instead of
    /// panicking when Δn becomes NaN or infinite. The run stops at the
    /// failing step; the message is also kept in [`SolverDiagnostics::failure`].
    /// On `Err`, [`Self::snapshots`] keeps any snapshots saved before the
    /// failure was detected, and the last of them may already hold NaN.
    pub fn try_run_with_snapshots(
        &mut self,
        snapshot_redshifts: &[f64],
    ) -> Result<&[SolverSnapshot], String> {
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
        self.reset();
        self.config = saved_config;
        self.z = self.config.z_start;
        self.diag.warnings = saved_warnings;
        self.disable_dcbr = saved_disable_dcbr;
        self.coupled_dcbr = saved_coupled_dcbr;
        self.number_conserving = saved_number_conserving;
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

        // Below the tested grid range, results can be wrong by orders of
        // magnitude with no other symptom (R-1: 100 points gave Δρ/ρ = −1.5e-2
        // for an injected +1e-5).
        if self.grid_n_points < MIN_TESTED_GRID_POINTS {
            self.diag.warnings.push(format!(
                "Frequency grid has n_points={} (fewer than {MIN_TESTED_GRID_POINTS}, the \
                 smallest tested size). Coarse grids can return wrong μ, y, and Δρ/ρ \
                 without any other warning; use n_points >= {MIN_TESTED_GRID_POINTS} \
                 (2000 or more for production).",
                self.grid_n_points
            ));
        }

        // Resonant conversion scenarios (dark photon, axion) install their IC
        // at z_start; warn users whose z_start doesn't match the NWA resonance
        // redshift (the builder auto-corrects, but `set_config` on a bare
        // solver bypasses that).
        let res_z_res = self
            .injection
            .as_ref()
            .and_then(|inj| inj.resonance_params(&self.cosmo))
            .map(|(_g, z_res)| z_res);
        if let Some(z_res) = res_z_res {
            if (self.config.z_start - z_res).abs() > 1e-6 * z_res {
                self.diag.warnings.push(format!(
                    "Resonant conversion: z_start={:.3e} ≠ NWA resonance z_res={:.3e}; the \
                     depletion IC is installed at z_start, not z_res. Leave z_start unset \
                     (no --z-start, or SolverBuilder without z_range) to start at z_res.",
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

        // Energy already in the spectrum at z_start (a user-set initial Δn);
        // the post-run closure check adds it to the injected energy.
        let z_run_start = self.z;
        let drho_initial = self.total_delta_rho_over_rho();

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
                        if self.step_with_dz(dz_to_snap).is_none() {
                            return Err(self.diag.failure.clone().unwrap_or_default());
                        }
                        // Save snapshots for all requested redshifts at this z
                        while next_snap < sorted_z.len() && self.z <= sorted_z[next_snap] {
                            self.save_snapshot_at(sorted_z[next_snap]);
                            next_snap += 1;
                        }
                        continue;
                    }
                }
            }

            let dz = self.adaptive_dz();
            if self.step_with_dz(dz).is_none() {
                return Err(self.diag.failure.clone().unwrap_or_default());
            }

            // Save snapshots for any requested redshifts we've reached or passed
            while next_snap < sorted_z.len() && self.z <= sorted_z[next_snap] {
                self.save_snapshot_at(sorted_z[next_snap]);
                next_snap += 1;
            }

            // Progress goes to stderr only: it is not a warning, and the
            // Python wrapper re-raises every `warnings` entry (R-5).
            if self.step_count % 100000 == 0 {
                eprintln!(
                    "Progress: z={:.1e} step={} ρ_e={:.6} ρ_eq={:.6}",
                    self.z, self.step_count, self.electron_temp.rho_e, self.rho_eq
                );
            }
        }

        if self.step_count >= step_limit {
            self.diag.warnings.push(format!(
                "Step limit ({}) reached at z={:.4e}. \
                 Run may be incomplete. Consider increasing dtau_max or narrowing the z range.",
                step_limit, self.z
            ));
        }

        // The in-loop check sees a NaN only at the start of the next step, so
        // one produced by the final step is caught here.
        if self.delta_n.iter().any(|v| !v.is_finite()) {
            self.record_nonfinite_delta_n();
        }
        if let Some(msg) = &self.diag.failure {
            return Err(msg.clone());
        }

        // Fill remaining snapshots below z_end with final state
        while next_snap < sorted_z.len() {
            self.save_snapshot();
            next_snap += 1;
        }

        if self.diag.nan_emission_detected {
            self.diag.warnings.push(
                "NaN emission rate detected during the run. DC/BR rates were capped \
                 to keep the solver alive but the final Δn may be polluted; narrow \
                 x_min or reduce dtau_max and rerun."
                    .to_string(),
            );
        }

        if let Some(w) = self.energy_closure_warning(z_run_start, drho_initial) {
            self.diag.warnings.push(w);
        }

        Ok(&self.snapshots)
    }

    /// Returns the total Δρ/ρ in the current state: the spectral Δn plus the
    /// temperature shift that number-conserving mode subtracted.
    fn total_delta_rho_over_rho(&self) -> f64 {
        crate::spectrum::delta_rho_over_rho(&self.grid.x, &self.delta_n)
            + 4.0 * self.accumulated_delta_t
    }

    /// Compares the energy in the final spectrum with the heat the injection
    /// delivered between `z_start` and the final redshift (R-1). Returns a
    /// warning if they differ by more than [`ENERGY_CLOSURE_REL_TOL`] of the
    /// gross injected heat ∫|dq/dz| dz plus [`ENERGY_CLOSURE_ABS_FLOOR`].
    ///
    /// Applies only to scenarios with a known injected energy (single burst,
    /// heating table); photon injection is excluded because its closure is
    /// known to be off by 7–45% for x_inj ≲ 0.03 (investigation I-1).
    ///
    /// When more than [`ENERGY_CHECK_MAX_LATE_FRACTION`] of the heat
    /// falls below z = [`ENERGY_CHECK_Z_LATE`], the tolerance also allows all
    /// of that late heat to be lost ([`energy_closure_allowance`]).
    fn energy_closure_warning(&self, z_run_start: f64, drho_initial: f64) -> Option<String> {
        if self.step_count == 0 {
            return None;
        }
        let inj = self.injection.as_ref()?;
        let z_final = self.z;
        let injected = inj.injected_delta_rho_between(z_final, z_run_start)?;
        // Tolerances scale with the gross heat moved, so a table that
        // alternates heating and cooling (net near zero) is not flagged for
        // the ordinary step error on each half-cycle.
        let gross = inj.injected_abs_delta_rho_between(z_final, z_run_start)?;
        if gross == 0.0 || !gross.is_finite() || !injected.is_finite() {
            return None;
        }
        let (late_abs, late_net) = if z_final < ENERGY_CHECK_Z_LATE {
            let z_hi = ENERGY_CHECK_Z_LATE.min(z_run_start);
            (
                inj.injected_abs_delta_rho_between(z_final, z_hi)
                    .unwrap_or(0.0),
                inj.injected_delta_rho_between(z_final, z_hi).unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0)
        };
        let (allow_short, allow_excess) = energy_closure_allowance(gross, late_abs, late_net);
        let expected = drho_initial + injected;
        let measured = self.total_delta_rho_over_rho();
        let diff = measured - expected;
        if -allow_short <= diff && diff <= allow_excess {
            return None;
        }
        let late_note = if late_abs > ENERGY_CHECK_MAX_LATE_FRACTION * gross {
            format!(
                " {:.0}% of the heat was injected below z = {ENERGY_CHECK_Z_LATE:.0}, \
                 where the gas can keep it from the photons; the tolerance already \
                 allows for losing all of it.",
                100.0 * late_abs / gross
            )
        } else {
            String::new()
        };
        Some(format!(
            "Energy closure: the final spectrum holds Δρ/ρ = {measured:.4e}, but the \
             injection delivered {expected:.4e} between z = {z_run_start:.3e} and \
             z = {z_final:.3e} ({:+.1}% of the injected heat; tolerance ±{:.0}%).{late_note} \
             Possible causes: a coarse frequency grid (this one has n_points={}; \
             try more), large time steps (reduce dtau_max), a large injection \
             (|Δρ/ρ| near 1e-2 or more), or a diagnostic flag. Check that μ and y \
             are stable before trusting them.",
            100.0 * diff / gross,
            100.0 * ENERGY_CLOSURE_REL_TOL,
            self.grid_n_points,
        ))
    }

    /// Runs the solver with `n_snapshots` log-spaced snapshot redshifts between
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

    /// Saves a snapshot with a specific redshift label.
    /// The solver state (such as delta_n and rho_e) is taken from the current state,
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

    /// Extracts `(μ, y)` from the current `Δn(x)` using the default joint
    /// least-squares decomposition (B&F 2022 nonlinear BE; see
    /// [`crate::distortion::decompose_distortion`]).
    pub fn extract_mu_y_joint(&self) -> (f64, f64) {
        let params = crate::distortion::decompose_distortion(&self.grid.x, &self.delta_n);
        (params.mu, params.y)
    }

    /// Runs the solver and return an owned [`crate::output::SolverResult`]
    /// with a single snapshot at `z_obs`.
    ///
    /// This is the preferred entry point: the result does not borrow the
    /// solver and has built-in JSON, CSV, or table serialization. For runs that
    /// need multiple intermediate snapshots, call
    /// [`Self::run_with_snapshots`] directly and inspect
    /// [`Self::snapshots`].
    ///
    /// # Panics
    ///
    /// Panics if Δn becomes NaN or infinite. Use [`Self::try_run_to_result`]
    /// to get that failure as an `Err`.
    pub fn run_to_result(&mut self, z_obs: f64) -> crate::output::SolverResult {
        self.try_run_to_result(z_obs)
            .unwrap_or_else(|msg| panic!("{msg}"))
    }

    /// Like [`Self::run_to_result`], but returns `Err` instead of panicking
    /// when Δn becomes NaN or infinite.
    pub fn try_run_to_result(&mut self, z_obs: f64) -> Result<crate::output::SolverResult, String> {
        self.try_run_with_snapshots(&[z_obs])?;
        let snapshot = self
            .snapshots
            .last()
            .expect("run_with_snapshots produced no snapshot")
            .clone();
        Ok(crate::output::SolverResult {
            snapshot,
            x_grid: self.grid.x.clone(),
            step_count: self.step_count,
            diag_newton_exhausted: self.diag.newton_exhausted,
            warnings: self.diag.warnings.clone(),
            t_cmb: self.cosmo.t_cmb,
        })
    }

    /// Creates a builder for configuring a solver with a fluent API.
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

/// Validates a solver setup and returns the soft warnings to attach to it.
///
/// Shared by [`SolverBuilder::build`] and the CLI so that both apply the same
/// checks. Hard errors: an invalid cosmology, grid, config, or injection, and
/// an injection window that `z_start` cuts off or that `z_end` never reaches.
/// Soft warnings: [`SolverConfig::soft_warnings`] and the injection's range
/// warnings. Two warnings are left to the solver, which pushes them itself:
/// stimulated emission in [`ThermalizationSolver::set_injection`], and a
/// `z_start` away from a resonance redshift at the start of each run.
pub(crate) fn preflight_checks(
    cosmo: &Cosmology,
    grid_config: &GridConfig,
    config: &SolverConfig,
    injection: Option<&InjectionScenario>,
) -> Result<Vec<String>, String> {
    cosmo.validate()?;
    grid_config.validate()?;
    config.validate()?;

    let mut warnings = config.soft_warnings();
    let Some(inj) = injection else {
        return Ok(warnings);
    };
    inj.validate()?;

    if let Some((_z_center, z_upper)) = inj.characteristic_redshift() {
        if config.z_start < z_upper {
            return Err(format!(
                "z_start={:.3e} is below the injection window upper bound z={:.3e}. \
                 The solver would miss part or all of the injection. Set z_start >= {:.3e}.",
                config.z_start, z_upper, z_upper
            ));
        }
        // With z_end above the window the injection never fires, and the run
        // returns only the adiabatic-cooling signal.
        if config.z_end > z_upper {
            return Err(format!(
                "z_end={:.3e} is above the injection window upper bound z={:.3e}. \
                 The solver would terminate before the injection epoch is reached. \
                 Set z_end < {:.3e}.",
                config.z_end, z_upper, z_upper
            ));
        }
    }

    warnings.extend(inj.warn_strong_distortion());
    warnings.extend(inj.warn_tabulated_coverage(config.z_start, config.z_end));
    warnings.extend(inj.warn_dark_photon_range(cosmo));
    warnings.extend(inj.warn_axion_range(cosmo));
    Ok(warnings)
}

/// Fluent builder for [`ThermalizationSolver`].
///
/// Groups all configuration into a chainable API. The existing
/// `ThermalizationSolver::new()` and manual field mutation still works;
/// this is a convenience layer on top.
pub struct SolverBuilder {
    cosmo: Cosmology,
    grid_config: GridConfig,
    injection: Option<InjectionScenario>,
    config: SolverConfig,
    /// Whether the caller set `z_start`. If not, `build` uses the resonance
    /// redshift for resonant scenarios and `config.z_start` otherwise.
    z_start_explicit: bool,
    disable_dcbr: bool,
    coupled_dcbr: bool,
    number_conserving: bool,
}

impl SolverBuilder {
    fn new(cosmo: Cosmology) -> Self {
        SolverBuilder {
            cosmo,
            grid_config: GridConfig::default(),
            injection: None,
            config: SolverConfig::default(),
            z_start_explicit: false,
            disable_dcbr: false,
            coupled_dcbr: true,
            number_conserving: true,
        }
    }

    /// Sets the frequency grid configuration.
    pub fn grid(mut self, config: GridConfig) -> Self {
        self.grid_config = config;
        self
    }

    /// Uses the fast (1000-point) grid for quick tests.
    pub fn grid_fast(mut self) -> Self {
        self.grid_config = GridConfig::fast();
        self
    }

    /// Sets the energy injection scenario.
    pub fn injection(mut self, scenario: InjectionScenario) -> Self {
        self.injection = Some(scenario);
        self
    }

    /// Sets the redshift range (z_start, z_end).
    pub fn z_range(mut self, z_start: f64, z_end: f64) -> Self {
        self.config.z_start = z_start;
        self.config.z_end = z_end;
        self.z_start_explicit = true;
        self
    }

    /// Sets a complete solver config, overriding individual z/dy/dtau settings.
    pub fn solver_config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self.z_start_explicit = true;
        self
    }

    /// Sets the maximum Compton y-parameter increment per step, Δy_C = θ_e Δτ (the
    /// Kompaneets-accuracy limiter is `dtau_max`, not this).
    pub fn dy_max(mut self, val: f64) -> Self {
        self.config.dy_max = val;
        self
    }

    /// Sets the maximum Compton optical depth per step.
    pub fn dtau_max(mut self, val: f64) -> Self {
        self.config.dtau_max = val;
        self
    }

    /// Disables DC/BR processes (Kompaneets only).
    pub fn disable_dcbr(mut self) -> Self {
        self.disable_dcbr = true;
        self
    }

    /// Uses operator-split DC/BR instead of coupled IMEX.
    pub fn split_dcbr(mut self) -> Self {
        self.coupled_dcbr = false;
        self
    }

    /// Disables number-conserving T-shift subtraction.
    pub fn no_number_conserving(mut self) -> Self {
        self.number_conserving = false;
        self
    }

    /// Sets the maximum number of Newton iterations per Kompaneets step.
    pub fn max_newton_iter(mut self, val: usize) -> Self {
        self.config.max_newton_iter = val;
        self
    }

    /// Sets the maximum Compton optical depth per step while a photon source is
    /// active ([`SolverConfig::dtau_max_photon_source`], default 1.0).
    pub fn dtau_max_photon_source(mut self, val: f64) -> Self {
        self.config.dtau_max_photon_source = val;
        self
    }

    /// Sets the redshift below which the number-conserving T-shift subtraction
    /// is off ([`SolverConfig::nc_z_min`], default 5e4; 0 applies it at all z).
    pub fn nc_z_min(mut self, val: f64) -> Self {
        self.config.nc_z_min = val;
        self
    }

    /// Builds the configured solver.
    ///
    /// Validates all configuration (cosmology, grid, solver config, injection)
    /// before constructing the solver. Returns `Err` with a descriptive message
    /// if any parameter is invalid.
    pub fn build(self) -> Result<ThermalizationSolver, String> {
        // Before `resonance_params` below, which reads the cosmology.
        self.cosmo.validate()?;
        let mut grid_config = self.grid_config;

        // Auto-apply refinement zones and x_min adjustment from injection scenario
        if let Some(ref inj) = self.injection {
            inj.refine_grid(&mut grid_config);
        }

        // For resonant conversion scenarios (dark photon, axion) the impulsive
        // Δn depletion happens at the NWA resonance z_res; evolving from a
        // higher z_start is unphysical (the conversion hasn't occurred yet) and
        // evolving from lower misses it entirely. Default z_start to z_res when
        // the user didn't supply an explicit value.
        let mut config = self.config;
        if !self.z_start_explicit {
            if let Some((_gamma, z_res)) = self
                .injection
                .as_ref()
                .and_then(|inj| inj.resonance_params(&self.cosmo))
            {
                config.z_start = z_res;
            }
        }
        let deferred_warnings =
            preflight_checks(&self.cosmo, &grid_config, &config, self.injection.as_ref())?;

        let mut solver = ThermalizationSolver::new(self.cosmo, grid_config);
        solver.set_config(config);
        solver.disable_dcbr = self.disable_dcbr;
        solver.coupled_dcbr = self.coupled_dcbr;
        solver.number_conserving = self.number_conserving;

        if let Some(scenario) = self.injection {
            solver.set_injection(scenario)?;
        }

        solver.diag.warnings.extend(deferred_warnings);

        Ok(solver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R-1 exceptions, tested on the check itself so that each is exercised
    /// alone. The base run fails closure with a shortfall (a
    /// 100-point grid from z = 5e6 ends near Δρ/ρ = −1.5e-2 for an injected
    /// 1e-5, with all heat above z = 600), so the check fires. A second run
    /// adds a known initial Δn: the check must count that energy as expected,
    /// or it would warn on a well-resolved run. The same run, checked against
    /// a wrong expectation, gives a clean excess and a clean shortfall of
    /// equal size, and both must warn.
    #[test]
    fn test_energy_closure_skip_rules() {
        let burst = InjectionScenario::SingleBurst {
            z_h: 3000.0,
            delta_rho_over_rho: 1e-5,
            sigma_z: 120.0,
        };
        let mut solver = ThermalizationSolver::new(
            Cosmology::default(),
            GridConfig {
                n_points: 100,
                ..GridConfig::default()
            },
        );
        solver.set_injection(burst).unwrap();
        solver.set_config(SolverConfig {
            z_start: 5e6,
            z_end: 500.0,
            ..SolverConfig::default()
        });
        solver.run_with_snapshots(&[500.0]);
        assert!(!solver.diag.rho_e_guard_warned);
        assert!(solver.energy_closure_warning(5e6, 0.0).is_some());

        // Initial Δn = a·n_pl holds Δρ/ρ = a exactly (∫x³ n_pl dx = G₃), here
        // twice the injected heat. Resolved burst, 1000 points.
        let a = 2e-5;
        let mut solver = ThermalizationSolver::new(Cosmology::default(), GridConfig::fast());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: 8e3,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: 2.6e5,
            z_end: 1e5,
            ..SolverConfig::default()
        });
        let dn0: Vec<f64> = solver.grid.x.iter().map(|&x| a * planck(x)).collect();
        solver.set_initial_delta_n(dn0);
        solver.run_with_snapshots(&[1e5]);
        let closure: Vec<&String> = solver
            .diag
            .warnings
            .iter()
            .filter(|w| w.starts_with("Energy closure"))
            .collect();
        assert!(closure.is_empty(), "{closure:?}");
        // Dropping the initial energy from the expectation must fire.
        assert!(solver.energy_closure_warning(2.6e5, 0.0).is_some());

        // Counting the initial energy a = 2e-5 twice leaves a shortfall of a
        // (200% of the injected heat), which must warn too.
        assert!(solver.energy_closure_warning(2.6e5, 2.0 * a).is_some());
    }

    /// Late-heat rule after ADR 0004. A burst at z_h = 1000 (σ_z = 100) puts
    /// 3e-5 of its heat below z = [`ENERGY_CHECK_Z_LATE`] and physically
    /// loses 1.4e-4 of it by z = 200, so the check keeps its plain tolerance:
    /// the correct run is silent and a 30% error of either sign warns. The
    /// old rule (cutoff z = 2000) skipped this run. Δρ/ρ = 2e-7 makes the
    /// tolerance 10% of the heat, half of it the 1e-8 absolute floor.
    ///
    /// A burst at z_h = 600 puts half its heat below z = 600. This run uses
    /// Δρ/ρ = 1e-10. It is silent, and a shortfall of 5e-8, beyond the
    /// tolerance plus all the late heat, still warns; the old rule skipped it.
    ///
    /// The errors are made by shifting the expected energy, as in
    /// `test_energy_closure_skip_rules`. The widening itself is checked on
    /// [`energy_closure_allowance`] for heating, cooling, and mixed cases,
    /// with bounds worked out by hand.
    #[test]
    fn test_energy_closure_late_rule() {
        let run = |z_h: f64, amp: f64| {
            let burst = InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: amp,
                sigma_z: 100.0,
            };
            let late = burst
                .injected_abs_delta_rho_between(200.0, ENERGY_CHECK_Z_LATE)
                .unwrap()
                / burst
                    .injected_abs_delta_rho_between(200.0, z_h + 700.0)
                    .unwrap();
            let mut solver = ThermalizationSolver::new(Cosmology::default(), GridConfig::fast());
            solver.set_injection(burst).unwrap();
            solver.set_config(SolverConfig {
                z_start: z_h + 700.0,
                z_end: 200.0,
                ..SolverConfig::default()
            });
            solver.run_with_snapshots(&[200.0]);
            assert!(!solver.diag.rho_e_guard_warned, "z_h = {z_h}: ρ_e guard");
            let closure: Vec<&String> = solver
                .diag
                .warnings
                .iter()
                .filter(|w| w.starts_with("Energy closure"))
                .collect();
            assert!(closure.is_empty(), "z_h = {z_h}: {closure:?}");
            (solver, late)
        };

        let amp = 2e-7;
        let (solver, late) = run(1000.0, amp);
        assert!(late < 1e-4, "late fraction {late}");
        let z0 = 1700.0;
        assert!(solver.energy_closure_warning(z0, 0.0).is_none());
        let short = solver.energy_closure_warning(z0, 0.3 * amp);
        assert!(short.is_some(), "30% shortfall at z_h = 1000 must warn");
        let excess = solver.energy_closure_warning(z0, -0.3 * amp);
        assert!(
            excess.as_deref().is_some_and(|w| !w.contains("below z =")),
            "{excess:?}"
        );

        let (solver, late) = run(600.0, 1e-10);
        assert!(
            late > ENERGY_CHECK_MAX_LATE_FRACTION,
            "late fraction {late}"
        );
        let short = solver.energy_closure_warning(1300.0, 5e-8);
        assert!(
            short
                .as_deref()
                .is_some_and(|w| w.contains("below z = 600")),
            "{short:?}"
        );

        let g = 1e-5;
        let tol = ENERGY_CLOSURE_REL_TOL * g + ENERGY_CLOSURE_ABS_FLOOR;
        let close = |(s, e): (f64, f64), (s0, e0): (f64, f64)| {
            assert!(
                (s - s0).abs() < 1e-12 * g && (e - e0).abs() < 1e-12 * g,
                "got ({s:e}, {e:e}), expected ({s0:e}, {e0:e})"
            );
        };
        // At most ENERGY_CHECK_MAX_LATE_FRACTION late: plain tolerance.
        close(energy_closure_allowance(g, 0.03 * g, 0.03 * g), (tol, tol));
        // Late heating widens the shortfall side only.
        close(
            energy_closure_allowance(g, 0.5 * g, 0.5 * g),
            (tol + 0.5 * g, tol),
        );
        // Late cooling widens the excess side only.
        close(
            energy_closure_allowance(g, 0.5 * g, -0.5 * g),
            (tol, tol + 0.5 * g),
        );
        // 0.3 g of late heating and 0.1 g of late cooling.
        close(
            energy_closure_allowance(g, 0.4 * g, 0.2 * g),
            (tol + 0.3 * g, tol + 0.1 * g),
        );
    }

    /// `guard_rho_e` warns on hot gas only below [`HOT_GAS_Z_MAX`] and above
    /// [`HOT_GAS_RHO_E`], clamps only above [`RHO_E_GUARD_MAX`], and pushes
    /// each warning once per run (ADR 0007).
    #[test]
    fn test_guard_rho_e_thresholds() {
        let mut solver = ThermalizationSolver::new(Cosmology::default(), GridConfig::fast());
        let count = |s: &ThermalizationSolver, prefix: &str| {
            s.diag
                .warnings
                .iter()
                .filter(|w| w.starts_with(prefix))
                .count()
        };
        // Hot but below the threshold, and above the threshold before
        // recombination: stored unchanged, no warning.
        assert_eq!(solver.guard_rho_e(5.0, 1000.0), Some(5.0));
        assert_eq!(solver.guard_rho_e(20.0, 2000.0), Some(20.0));
        assert!(
            solver.diag.warnings.is_empty(),
            "{:?}",
            solver.diag.warnings
        );
        assert_eq!(solver.diag.rho_e_clamped, 0);
        // Above the threshold after recombination: warns, not clamped.
        assert_eq!(solver.guard_rho_e(20.0, 1000.0), Some(20.0));
        assert_eq!(count(&solver, "Hot gas"), 1);
        assert_eq!(solver.diag.rho_e_clamped, 0);
        // Past the sanity guard: clamped, counted, and warned once.
        for _ in 0..2 {
            assert_eq!(solver.guard_rho_e(2e4, 1000.0), Some(RHO_E_GUARD_MAX));
        }
        assert_eq!(solver.diag.rho_e_clamped, 2);
        assert_eq!(count(&solver, "The electron temperature solve gave"), 1);
        assert_eq!(count(&solver, "Hot gas"), 1);
    }

    /// Checks that the analytic dH/dρ_e in `dcbr_heating_with_derivative` matches a
    /// central finite difference of the heating integral itself, built here
    /// from two derivative-free calls at θ_z(ρ_e ± δ). This is the guard for
    /// the analytic derivative that replaced the original FD implementation:
    /// the end-to-end observables are insensitive to dh at the 1e-9 level, so
    /// only a direct comparison like this can catch a wrong derivative.
    #[test]
    fn test_dcbr_heating_analytic_derivative_matches_fd() {
        use crate::grid::{FrequencyGrid, GridConfig};
        use crate::spectrum::planck;

        let grid = FrequencyGrid::new(&GridConfig {
            n_points: 500,
            ..GridConfig::default()
        });
        let n = grid.x.len();
        let expc_half: Vec<f64> = (0..n - 1)
            .map(|i| gaunt_expc_factor((0.5 * (grid.x[i] + grid.x[i + 1])).ln()))
            .collect();
        let planck_half: Vec<f64> = grid.x_half.iter().map(|&x| planck(x)).collect();
        // K_DC shape factor: unity suffices — the FD identity must hold for
        // any fixed x-dependence.
        let dc_supp_half = vec![1.0; n - 1];

        // Representative states: (z, ρ_e), μ-era through recombination era.
        let states: [(f64, f64); 3] = [(3.0e6, 1.0 + 3e-5), (2.0e5, 1.0 - 1e-5), (2.0e4, 0.999)];
        // A smooth non-Planck distortion so the Bose factor term is exercised.
        let delta_n: Vec<f64> = grid
            .x
            .iter()
            .map(|&x| 1e-5 * planck(x) * (1.0 + planck(x)) * x)
            .collect();

        for &(z, rho_e) in &states {
            let theta_z = 4.6e-10 * (1.0 + z);
            let n_h = 0.19 * (1.0 + z).powi(3);
            let n_he = 0.012 * (1.0 + z).powi(3);
            let n_e = n_h; // fully ionized H is representative enough
            let (x_e_frac, y_he_ii, y_he_i) = (1.0, 0.5, 0.9);

            let h_at = |rho: f64, deriv: bool| {
                dcbr_heating_with_derivative(
                    &grid.x,
                    &grid.x_half,
                    &grid.dx,
                    &planck_half,
                    &delta_n,
                    theta_z,
                    theta_z * rho,
                    n_h,
                    n_he,
                    n_e,
                    x_e_frac,
                    y_he_ii,
                    y_he_i,
                    deriv,
                    &expc_half,
                    &dc_supp_half,
                )
            };

            let (_, dh_analytic) = h_at(rho_e, true);
            // δ = 1e-6: FD truncation O(δ²) ~ 1e-12 relative, rounding on the
            // difference ~ eps/δ ~ 1e-10 relative — both well under the 1e-6
            // assertion band.
            let delta = 1e-6;
            let (h_plus, _) = h_at(rho_e + delta, false);
            let (h_minus, _) = h_at(rho_e - delta, false);
            let dh_fd = (h_plus - h_minus) / (2.0 * delta);

            let rel = ((dh_analytic - dh_fd) / dh_fd).abs();
            assert!(
                rel < 1e-6,
                "z={z:.1e}, rho_e={rho_e}: dh_analytic={dh_analytic:.12e} vs dh_fd={dh_fd:.12e} (rel {rel:.2e})"
            );
        }
    }

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

    /// Checks that `.disable_dcbr()` produces the un-thermalized μ = 1.401 × Δρ/ρ exactly
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
        // μ with the temperature shift removed by photon-number conservation (ADR 0006).
        let mu = crate::distortion::decompose_number_conserving(&solver.grid.x, &last.delta_n).mu;

        let mu_expected = (3.0 / KAPPA_C) * drho; // = 1.401 * drho, no J_bb* with DC/BR off
        let mu_err = (mu - mu_expected).abs() / mu_expected;
        eprintln!(
            "disable_dcbr: μ = {:.4e}, expected (3/κ_c)·Δρ/ρ = {:.4e}, err = {:.2}%",
            mu,
            mu_expected,
            mu_err * 100.0
        );
        assert!(
            mu_err < 0.05,
            "disable_dcbr: μ = {:.4e} vs pure-Kompaneets target {:.4e} \
             (err {:.2}%, tol 5%)",
            mu,
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
    fn test_solver_builder_config_roundtrip() {
        // Regression: solver_config() used to drop dtau_max_photon_source,
        // silently substituting defaults in build(). Every field is
        // set to a non-default value and asserted individually, so wiring a
        // future field through the builder incompletely fails here.
        let supplied = SolverConfig {
            z_start: 2.1e5,
            z_end: 1e5,
            dy_max: 0.013,
            dz_min: 3e-6,
            dtau_max: 7.0,
            nc_z_min: 3.3e4,
            dtau_max_photon_source: 0.37,
            max_newton_iter: 14,
        };
        let solver = SolverBuilder::new(Cosmology::default())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: 100.0,
            })
            .solver_config(supplied.clone())
            .build()
            .unwrap();
        assert_eq!(solver.config.z_start, supplied.z_start);
        assert_eq!(solver.config.z_end, supplied.z_end);
        assert_eq!(solver.config.dy_max, supplied.dy_max);
        assert_eq!(solver.config.dz_min, supplied.dz_min);
        assert_eq!(solver.config.dtau_max, supplied.dtau_max);
        assert_eq!(solver.config.nc_z_min, supplied.nc_z_min);
        assert_eq!(
            solver.config.dtau_max_photon_source,
            supplied.dtau_max_photon_source
        );
        assert_eq!(solver.config.max_newton_iter, supplied.max_newton_iter);
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

    /// The builder and the CLI share `preflight_checks`, so the builder, like
    /// the CLI, rejects a run that ends before the injection window opens.
    #[test]
    fn test_builder_rejects_z_end_above_injection_window() {
        let err = ThermalizationSolver::builder(Cosmology::default())
            .grid_fast()
            .injection(InjectionScenario::SingleBurst {
                z_h: 2e5,
                delta_rho_over_rho: 1e-5,
                sigma_z: 100.0,
            })
            .z_range(3e5, 2.5e5)
            .build()
            .err()
            .expect("z_end above the burst window must be rejected");
        assert!(err.contains("z_end=2.500e5"), "{err}");
    }

    /// A-7: the builder sets `dtau_max_photon_source` and `nc_z_min`,
    /// and leaves each at its `SolverConfig` default when not called.
    #[test]
    fn test_builder_photon_nc_setters() {
        let defaults = SolverConfig::default();
        let plain = ThermalizationSolver::builder(Cosmology::default())
            .build()
            .unwrap();
        assert_eq!(
            plain.config.dtau_max_photon_source,
            defaults.dtau_max_photon_source
        );
        assert_eq!(plain.config.nc_z_min, defaults.nc_z_min);

        let set = ThermalizationSolver::builder(Cosmology::default())
            .dtau_max_photon_source(0.25)
            .nc_z_min(0.0)
            .build()
            .unwrap();
        assert_eq!(set.config.dtau_max_photon_source, 0.25);
        assert_eq!(set.config.nc_z_min, 0.0);
    }
}
