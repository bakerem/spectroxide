use spectroxide::constants::*;
use spectroxide::distortion::{decompose_distortion, delta_n_to_intensity_mjy};
use spectroxide::energy_injection::InjectionScenario;
use spectroxide::grid::RefinementZone;
use spectroxide::prelude::*;
use spectroxide::spectrum::g_bb;

fn main() {
    let cosmo = Cosmology::default();
    let x_inj = 1e-3;
    let z_h = 3e5;
    let dn_over_n = 1e-5;
    let sigma_z = z_h * 0.025;
    let sigma_x = 0.05 * x_inj;

    let grid_config = GridConfig {
        x_min: 1e-5,
        x_max: 60.0,
        n_points: 4000,
        x_transition: 0.1,
        log_fraction: 0.3,
        refinement_zones: vec![RefinementZone {
            x_center: x_inj,
            x_width: 10.0 * sigma_x,
            n_points: 300,
        }],
    };

    let z_start = z_h + 7.0 * sigma_z;
    let config = SolverConfig {
        z_start,
        z_end: 200.0,
        dtau_max_photon_source: 1.0,
        ..Default::default()
    };

    let mut solver = ThermalizationSolver::new(cosmo.clone(), grid_config);
    solver.config = config;
    solver
        .set_injection(InjectionScenario::MonochromaticPhotonInjection {
            x_inj,
            delta_n_over_n: dn_over_n,
            z_h,
            sigma_z,
            sigma_x,
        })
        .unwrap();

    solver.run_with_snapshots(&[200.0]);

    let x = &solver.grid.x;
    let snap = &solver.snapshots[0];
    let params = decompose_distortion(x, &snap.delta_n);

    println!("=== Unstripped spectrum ===");
    println!(
        "mu={:.4e}, y={:.4e}, dT={:.4e}, drho={:.4e}",
        params.mu, params.y, params.delta_t_over_t, params.delta_rho_over_rho
    );

    // Post-hoc NC strip: subtract dN/N*G_bb/3 from delta_n
    let mut delta_g2 = 0.0;
    let mut g2_gbb = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        let xm = 0.5 * (x[i] + x[i - 1]);
        let dnm = 0.5 * (snap.delta_n[i] + snap.delta_n[i - 1]);
        let gm = 0.5 * (g_bb(x[i]) + g_bb(x[i - 1]));
        delta_g2 += xm * xm * dnm * dx;
        g2_gbb += xm * xm * gm * dx;
    }
    let delta_t_strip = delta_g2 / g2_gbb;

    let stripped: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| snap.delta_n[i] - delta_t_strip * g_bb(xi))
        .collect();

    let params_strip = decompose_distortion(x, &stripped);
    println!(
        "\n=== NC-stripped spectrum (delta_t stripped = {:.4e}) ===",
        delta_t_strip
    );
    println!(
        "mu={:.4e}, y={:.4e}, dT={:.4e}, drho={:.4e}",
        params_strip.mu,
        params_strip.y,
        params_strip.delta_t_over_t,
        params_strip.delta_rho_over_rho
    );

    // Peak intensities
    println!("\n=== Peak intensities ===");
    let mut max_unstrip = 0.0_f64;
    let mut max_strip = 0.0_f64;
    let mut max_nu_unstrip = 0.0;
    let mut max_nu_strip = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        let i_unstrip = delta_n_to_intensity_mjy(xi, snap.delta_n[i], cosmo.t_cmb);
        let i_strip = delta_n_to_intensity_mjy(xi, stripped[i], cosmo.t_cmb);
        let nu_ghz = xi * K_BOLTZMANN * cosmo.t_cmb / HPLANCK * 1e-9;
        if i_unstrip.abs() > max_unstrip.abs() {
            max_unstrip = i_unstrip;
            max_nu_unstrip = nu_ghz;
        }
        if i_strip.abs() > max_strip.abs() {
            max_strip = i_strip;
            max_nu_strip = nu_ghz;
        }
    }
    println!(
        "Unstripped: peak = {:.4e} MJy/sr at {:.1} GHz",
        max_unstrip, max_nu_unstrip
    );
    println!(
        "NC-stripped: peak = {:.4e} MJy/sr at {:.1} GHz",
        max_strip, max_nu_strip
    );
    println!(
        "Per dN/N: unstrip peak = {:.4e}, strip peak = {:.4e}",
        max_unstrip / dn_over_n,
        max_strip / dn_over_n
    );

    // Spectral profile at key frequencies
    println!("\nSpectral profile (per dN/N):");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "nu(GHz)", "I_unstrip", "I_strip", "x3dn_us", "x3dn_s"
    );
    for &xi in &[
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0,
    ] {
        if let Some(idx) = x.iter().position(|&v| v >= xi) {
            let nu_ghz = xi * K_BOLTZMANN * cosmo.t_cmb / HPLANCK * 1e-9;
            let i_u = delta_n_to_intensity_mjy(xi, snap.delta_n[idx], cosmo.t_cmb) / dn_over_n;
            let i_s = delta_n_to_intensity_mjy(xi, stripped[idx], cosmo.t_cmb) / dn_over_n;
            let x3_u = xi.powi(3) * snap.delta_n[idx] / dn_over_n;
            let x3_s = xi.powi(3) * stripped[idx] / dn_over_n;
            println!(
                "{:>8.1} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
                nu_ghz, i_u, i_s, x3_u, x3_s
            );
        }
    }

    // Compare heat injection with same energy
    println!("\n=== Heat injection comparison (same drho/rho) ===");
    let grid_config2 = GridConfig {
        x_min: 1e-5,
        x_max: 60.0,
        n_points: 4000,
        x_transition: 0.1,
        log_fraction: 0.3,
        refinement_zones: vec![],
    };
    let config2 = SolverConfig {
        z_start,
        z_end: 200.0,
        ..Default::default()
    };
    let mut solver2 = ThermalizationSolver::new(cosmo.clone(), grid_config2);
    solver2.config = config2;
    solver2
        .set_injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: 3.7022e-9,
            sigma_z,
        })
        .unwrap();
    solver2.run_with_snapshots(&[200.0]);
    let x2 = &solver2.grid.x;
    let heat = &solver2.snapshots[0];
    let p_heat = decompose_distortion(x2, &heat.delta_n);
    println!(
        "Heat: mu={:.4e}, y={:.4e}, dT={:.4e}",
        p_heat.mu, p_heat.y, p_heat.delta_t_over_t
    );

    // For reference: what CosmoTherm should see after NC stripping
    // is a spectrum like heat injection (positive mu) minus the number excess
    println!(
        "\nExpected: if dN/N excess were zero, mu should be ~ {:.4e}",
        1.401 * 3.7022e-9
    );
    println!("Actual mu (unstripped): {:.4e}", params.mu);
    println!("Actual mu (stripped): {:.4e}", params_strip.mu);

    // Check: what is the stripped dN/N?
    let mut delta_g2_strip = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        let xm = 0.5 * (x[i] + x[i - 1]);
        let dnm = 0.5 * (stripped[i] + stripped[i - 1]);
        delta_g2_strip += xm * xm * dnm * dx;
    }
    println!(
        "Stripped dN/N = {:.4e} (should be ~0)",
        delta_g2_strip / G2_PLANCK
    );
}
