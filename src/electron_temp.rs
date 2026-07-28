//! Electron temperature state (ρ_e = T_e/T_z).
//!
//! This module provides the [`ElectronTemperature`] struct used by the solver
//! to hold the current T_e/T_z ratio. The production T_e update is performed
//! in `solver::compute_hubble_coefficients` using a perturbative form of the
//! Compton equilibrium:
//!
//!   Δρ_eq = ΔI₄/(4 G₃) − ΔG₃/G₃ × (I₄/(4G₃))
//!
//! computed from Δn only. The full form ρ_eq = I₄/(4G₃) (below) has
//! ~0.1% numerical error from near-cancellation that swamps the O(10⁻⁵)
//! physical signal — do not use it in the solver. It is retained here only
//! as a verification tool for off-path consistency checks and tests.
//!
//! References:
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 15-18]

use crate::spectrum::compton_equilibrium_ratio;

/// State of the electron temperature solver.
#[derive(Debug, Clone)]
pub struct ElectronTemperature {
    /// Current T_e/T_z ratio
    pub rho_e: f64,
}

impl Default for ElectronTemperature {
    fn default() -> Self {
        ElectronTemperature { rho_e: 1.0 }
    }
}

impl ElectronTemperature {
    /// θ_e from a precomputed θ_z value (cosmology-aware).
    ///
    /// Pass `cosmo.theta_z(z)` so a non-default T_CMB is honoured.
    #[inline]
    pub fn theta_e_with(&self, theta_z_val: f64) -> f64 {
        self.rho_e * theta_z_val
    }

    /// Set ρ_e from the full Compton-equilibrium form I₄/(4G₃).
    ///
    /// **Not used by the production solver.** The full form has ~0.1%
    /// numerical error from near-cancellation of the two integrals, which
    /// swamps the O(10⁻⁵) physical distortion signal. Retained only as a
    /// reference for off-path tests and verification; the solver uses the
    /// perturbative update in `solver::compute_hubble_coefficients` instead.
    pub fn update_equilibrium(&mut self, x_grid: &[f64], n_full: &[f64]) {
        self.rho_e = compton_equilibrium_ratio(x_grid, n_full);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::FrequencyGrid;
    use crate::spectrum::planck;

    #[test]
    fn test_equilibrium_for_planck() {
        let grid = FrequencyGrid::log_uniform(1e-4, 50.0, 5000);
        let n_pl: Vec<f64> = grid.x.iter().map(|&x| planck(x)).collect();

        let mut te = ElectronTemperature::default();
        te.update_equilibrium(&grid.x, &n_pl);

        assert!(
            (te.rho_e - 1.0).abs() < 1e-3,
            "ρ_e = {}, expected 1.0 for Planck",
            te.rho_e
        );
    }

    // test_theta_e_with_scaling removed: theta_e_with(θ_z) is defined as
    // rho_e * θ_z, so asserting (1.05 * θ_z).abs() < 1e-30 was tautological.

    /// Verify ρ_eq = 1 exactly for any Bose-Einstein distribution.
    ///
    /// Analytic anchor: for n_BE(x, μ) = 1/(e^{x+μ}-1), n(1+n) = −dn/dx, so
    /// integrating by parts gives I₄ = ∫x⁴ n(1+n)dx = 4∫x³ n dx = 4G₃
    /// identically — a BE spectrum is the Kompaneets stationary state, so its
    /// Compton-equilibrium temperature is T_z for ALL μ. Any deviation of
    /// ρ_e = I₄/(4G₃) from 1 is pure O(dx²) quadrature error, which must be
    /// μ-independent at this order and shrink under grid refinement.
    /// (Validation-audit finding P1-3: an earlier version asserted
    /// "μ>0 ⇒ ρ_e>1", which is wrong physics that passed on grid noise.)
    #[test]
    fn test_equilibrium_for_bose_einstein() {
        // Only positive μ: n_BE(x, μ) has a pole at x = |μ| for μ < 0.
        let grid = FrequencyGrid::log_uniform(1e-4, 50.0, 10000);
        for &mu in &[1e-4, 1e-3, 5e-3] {
            let n_be: Vec<f64> = grid
                .x
                .iter()
                .map(|&x| 1.0 / ((x + mu).exp() - 1.0))
                .collect();

            let mut te = ElectronTemperature::default();
            te.update_equilibrium(&grid.x, &n_be);

            // Analytic target ρ_eq = 1; ~2e-6 discretization floor at N=10000.
            assert!(
                (te.rho_e - 1.0).abs() < 1e-5,
                "μ={mu}: ρ_eq must be 1 for any BE spectrum, got {:.10}",
                te.rho_e
            );
        }

        // Grid-refinement check: the residual is discretization error, so it
        // must shrink (O(dx²) ⇒ ~4× per doubling; require at least 2×).
        let mu = 1e-3;
        let err_at = |n_points: usize| {
            let g = FrequencyGrid::log_uniform(1e-4, 50.0, n_points);
            let n_be: Vec<f64> = g.x.iter().map(|&x| 1.0 / ((x + mu).exp() - 1.0)).collect();
            (crate::spectrum::compton_equilibrium_ratio(&g.x, &n_be) - 1.0).abs()
        };
        let (e_coarse, e_fine) = (err_at(5000), err_at(20000));
        assert!(
            e_fine < 0.5 * e_coarse,
            "|ρ_eq−1| must be discretization error: N=5000 → {e_coarse:.2e}, N=20000 → {e_fine:.2e}"
        );
    }

    /// `update_equilibrium` must recover a *non-unity* temperature ratio.
    ///
    /// Analytic anchor: a Planck spectrum sampled at a shifted temperature,
    /// n(x) = n_pl(x/a) = 1/(e^{x/a}−1), is the Bose-Einstein stationary state
    /// at temperature a·T_z, so its Compton-equilibrium ratio is exactly `a`.
    /// Proof: dn/dx = −(1/a) n(1+n), hence n(1+n) = −a dn/dx and
    ///   I₄ = ∫x⁴ n(1+n)dx = a·4∫x³ n dx = 4a G₃  (integrate by parts, x⁴n→0),
    /// so ρ_eq = I₄/(4G₃) = a.
    ///
    /// This is the discriminating check the ρ_eq=1 tests above lack: they feed
    /// spectra whose equilibrium ratio equals the `Default` value 1.0, so a
    /// no-op `update_equilibrium` passes them. Here a = 1.05 ≠ 1, so a no-op
    /// (leaving rho_e = 1.0) fails. (Mutation-audit R2 survivor closure.)
    #[test]
    fn test_equilibrium_recovers_shifted_temperature() {
        let grid = FrequencyGrid::log_uniform(1e-4, 50.0, 10000);
        let a = 1.05;
        let n_shifted: Vec<f64> = grid.x.iter().map(|&x| planck(x / a)).collect();

        let mut te = ElectronTemperature::default();
        te.update_equilibrium(&grid.x, &n_shifted);

        // Analytic target ρ_eq = a; tolerance covers the O(dx²) quadrature
        // floor (~1e-4 here) with margin, while |a − 1| = 0.05 ≫ tol so a
        // no-op update is unambiguously rejected.
        assert!(
            (te.rho_e - a).abs() < 2e-3,
            "ρ_eq must recover the shifted-Planck temperature a={a}, got {:.8}",
            te.rho_e
        );
    }
}
