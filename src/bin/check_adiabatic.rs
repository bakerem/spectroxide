//! Adiabatic cooling sanity check.
//!
//! Prints a few cosmology-derived quantities at a representative redshift
//! and the accumulated analytical Δρ/ρ from adiabatic cooling over the
//! y-to-μ transition range, for comparison against literature benchmarks.

use spectroxide::constants::*;
use spectroxide::prelude::*;

fn main() {
    let cosmo = Cosmology::default();
    let z = 5e4;

    let n_h = cosmo.n_h(z);
    let n_gamma = cosmo.n_gamma(z);
    let x_e = spectroxide::recombination::ionization_fraction(z, &cosmo);
    let h = cosmo.hubble(z);
    let t_c = cosmo.t_compton(z, x_e);
    let theta = theta_z(z);

    eprintln!("At z = {z:.0e}:");
    eprintln!("  n_h = {n_h:.4e} m⁻³");
    eprintln!("  n_gamma = {n_gamma:.4e} m⁻³");
    eprintln!("  n_h/n_gamma = {:.4e}", n_h / n_gamma);
    eprintln!("  x_e = {x_e:.4}");
    eprintln!("  H(z) = {h:.4e} s⁻¹");
    eprintln!("  t_C = {t_c:.4e} s");
    eprintln!("  θ_z = {theta:.4e}");
    eprintln!("  H × t_C = {:.4e}", h * t_c);

    // Solver-internal adiabatic cooling source term: q_rel = -(3/2) (x_e n_H/n_γ) H.
    // This enters the Kompaneets source via δρ_e = q_rel · t_C / (4 θ_z).
    let q_rel = -1.5 * x_e * n_h / n_gamma * h;
    let delta_rho_inj = q_rel * t_c / (4.0 * theta);
    eprintln!();
    eprintln!("q_rel         = {q_rel:.4e} s⁻¹");
    eprintln!("δρ_e per step = {delta_rho_inj:.4e}");

    // Analytical Δρ/ρ accumulated from adiabatic cooling over z ∈ [500, 5e4].
    // d(Δρ/ρ)/dt = q_rel, so Δρ/ρ = ∫ q_rel dz / (H(1+z)) = -(3/2) η ln[(1+z_max)/(1+z_min)]
    // because x_e n_H/n_γ ≡ η is z-independent when x_e ≈ 1.
    let z_min: f64 = 500.0;
    let z_max: f64 = 5.0e4;
    let eta = n_h / n_gamma;
    let delta_rho_analytic = -1.5 * eta * ((1.0 + z_max).ln() - (1.0 + z_min).ln());
    eprintln!();
    eprintln!("Analytical ∫ q_rel dt over z ∈ [{z_min:.0}, {z_max:.0e}]:");
    eprintln!("  η = n_H/n_γ     = {eta:.4e}");
    eprintln!("  -(3/2) η ln(…)  = {delta_rho_analytic:.4e}");
    eprintln!("  literature (Chluba 2005): ≈ -2.2e-9");
}
