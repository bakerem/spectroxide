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

    /// Verify ρ_e for a Bose-Einstein distribution with known μ.
    ///
    /// For n_BE(x, μ) = 1/(e^{x+μ}-1), the Compton equilibrium ratio is:
    ///   ρ_e = I₄/(4G₃) where I₄ = ∫x⁴ n(1+n)dx, G₃ = ∫x³ n dx.
    /// For μ > 0: spectrum is harder than Planck → ρ_e > 1.
    /// For μ < 0: spectrum is softer → ρ_e < 1.
    #[test]
    fn test_equilibrium_for_bose_einstein() {
        let grid = FrequencyGrid::log_uniform(1e-4, 50.0, 10000);

        // Only test positive μ: n_BE(x, μ) = 1/(e^{x+μ}-1) is well-defined for μ > 0.
        // Negative μ causes a pole at x = |μ| which makes the integral diverge.
        for &mu in &[1e-4, 1e-3, 5e-3] {
            let n_be: Vec<f64> = grid
                .x
                .iter()
                .map(|&x| 1.0 / ((x + mu).exp() - 1.0))
                .collect();

            let mut te = ElectronTemperature::default();
            te.update_equilibrium(&grid.x, &n_be);

            // Independent numerical integration (proper trapezoidal for n(1+n))
            let mut g3 = 0.0;
            let mut i4 = 0.0;
            for i in 1..grid.n {
                let dx = grid.x[i] - grid.x[i - 1];
                let x_mid = 0.5 * (grid.x[i] + grid.x[i - 1]);
                let n_mid = 0.5 * (n_be[i] + n_be[i - 1]);
                let n_l = n_be[i - 1];
                let n_r = n_be[i];
                let nn1_mid = 0.5 * (n_l * (1.0 + n_l) + n_r * (1.0 + n_r));
                g3 += x_mid.powi(3) * n_mid * dx;
                i4 += x_mid.powi(4) * nn1_mid * dx;
            }
            let rho_expected = i4 / (4.0 * g3);

            let rel = (te.rho_e - rho_expected).abs() / rho_expected;
            assert!(
                rel < 1e-6,
                "μ={mu}: ρ_e={:.10} vs expected {rho_expected:.10}, err={rel:.2e}",
                te.rho_e
            );

            // Physical direction check: μ > 0 → fewer low-x photons → harder spectrum → ρ_e > 1
            assert!(
                te.rho_e > 1.0,
                "μ={mu}>0 should give ρ_e>1, got {:.10}",
                te.rho_e
            );

            // Larger μ → more deviation from 1 (use tolerance; at small μ the signal is ~μ²)
            if mu > 1e-4 {
                let rho_small = {
                    let n_small: Vec<f64> = grid
                        .x
                        .iter()
                        .map(|&x| 1.0 / ((x + 1e-4).exp() - 1.0))
                        .collect();
                    crate::spectrum::compton_equilibrium_ratio(&grid.x, &n_small)
                };
                assert!(
                    te.rho_e >= rho_small - 1e-9,
                    "Larger μ should give larger ρ_e: μ={mu} → {:.10} < μ=1e-4 → {rho_small:.10}",
                    te.rho_e
                );
            }
        }
    }
}
