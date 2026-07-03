//! Generate Rust↔Python parity fixtures (validation-audit Part B2).
//!
//! Evaluates every Rust function that has a pure-Python mirror in
//! `python/spectroxide/` on a deterministic input grid and writes the results
//! to JSON. `python/tests/test_parity.py` evaluates the Python mirrors on the
//! same inputs and asserts agreement to the per-group tolerance declared here.
//!
//! Usage:
//!   cargo run --release --example generate_parity_fixtures -- [output.json]
//!
//! Default output: python/tests/data/parity_fixtures.json (committed). The CI
//! parity job regenerates the fixture from the current Rust and re-runs the
//! Python comparison, so drift on either side fails CI.

use spectroxide::cosmology::Cosmology;
use spectroxide::dark_photon;
use spectroxide::greens;
use spectroxide::recombination;
use spectroxide::spectrum;

/// One fixture group: a named function, a cosmology label, a list of input
/// tuples, the Rust outputs, and the relative tolerance the Python mirror
/// must meet.
struct Group {
    name: &'static str,
    cosmo: &'static str, // "default" | "planck2018" | "none"
    inputs: Vec<Vec<f64>>,
    values: Vec<Option<Vec<f64>>>,
    rtol: f64,
    note: &'static str,
}

fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let (l0, l1) = (lo.ln(), hi.ln());
    (0..n)
        .map(|i| (l0 + (l1 - l0) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

fn scalar_group<F>(
    name: &'static str,
    cosmo: &'static str,
    inputs: Vec<Vec<f64>>,
    rtol: f64,
    note: &'static str,
    f: F,
) -> Group
where
    F: Fn(&[f64]) -> f64,
{
    let values = inputs
        .iter()
        .map(|inp| {
            let v = f(inp);
            if v.is_finite() {
                Some(vec![v])
            } else {
                None
            }
        })
        .collect();
    Group {
        name,
        cosmo,
        inputs,
        values,
        rtol,
        note,
    }
}

fn json_f64(v: f64) -> String {
    // Full round-trip precision; JSON has no Inf/NaN so callers must filter.
    format!("{v:.17e}")
}

fn write_json(groups: &[Group], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"generator\": \"examples/generate_parity_fixtures.rs\",\n");
    out.push_str("  \"schema\": 1,\n");
    out.push_str("  \"groups\": [\n");
    for (gi, g) in groups.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", g.name));
        out.push_str(&format!("      \"cosmo\": \"{}\",\n", g.cosmo));
        out.push_str(&format!("      \"rtol\": {},\n", json_f64(g.rtol)));
        out.push_str(&format!("      \"note\": \"{}\",\n", g.note));
        out.push_str("      \"inputs\": [");
        for (i, inp) in g.inputs.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push('[');
            out.push_str(
                &inp.iter()
                    .map(|v| json_f64(*v))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push(']');
        }
        out.push_str("],\n");
        out.push_str("      \"values\": [");
        for (i, val) in g.values.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match val {
                None => out.push_str("null"),
                Some(vs) => {
                    out.push('[');
                    out.push_str(
                        &vs.iter()
                            .map(|v| json_f64(*v))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    out.push(']');
                }
            }
        }
        out.push_str("]\n");
        out.push_str(if gi + 1 < groups.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    out.push_str("  ]\n}\n");
    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "python/tests/data/parity_fixtures.json".to_string());

    let cosmo_default = Cosmology::default();
    let cosmo_p18 = Cosmology::planck2018();
    let cosmos: [(&str, &Cosmology); 2] =
        [("default", &cosmo_default), ("planck2018", &cosmo_p18)];

    let mut groups: Vec<Group> = Vec::new();

    // --- Visibility functions (closed-form fits; exact mirrors) ------------
    let z_vis: Vec<Vec<f64>> = logspace(200.0, 5.0e6, 40).into_iter().map(|z| vec![z]).collect();
    groups.push(scalar_group(
        "visibility_j_bb",
        "none",
        z_vis.clone(),
        1e-11,
        "Chluba 2013 J_bb fit; closed form",
        |i| greens::visibility_j_bb(i[0]),
    ));
    groups.push(scalar_group(
        "visibility_j_bb_star",
        "none",
        z_vis.clone(),
        1e-11,
        "Chluba 2013 J_bb* fit; closed form",
        |i| greens::visibility_j_bb_star(i[0]),
    ));
    groups.push(scalar_group(
        "visibility_j_y",
        "none",
        z_vis.clone(),
        1e-11,
        "Chluba 2013 J_y fit; closed form",
        |i| greens::visibility_j_y(i[0]),
    ));
    groups.push(scalar_group(
        "visibility_j_mu",
        "none",
        z_vis.clone(),
        1e-11,
        "Chluba 2013 J_mu fit; closed form",
        |i| greens::visibility_j_mu(i[0]),
    ));

    // --- Heat-injection Green's function -----------------------------------
    let mut gf_inputs = Vec::new();
    for &z_h in &[1.0e4, 5.0e4, 3.0e5, 2.0e6] {
        for x in logspace(0.05, 25.0, 15) {
            gf_inputs.push(vec![x, z_h]);
        }
    }
    groups.push(scalar_group(
        "greens_function",
        "none",
        gf_inputs,
        1e-11,
        "G_th(x, z_h); closed-form visibility blend",
        |i| greens::greens_function(i[0], i[1]),
    ));

    // --- Critical frequencies & analytic P_s --------------------------------
    let z_xc: Vec<Vec<f64>> = logspace(1.0e3, 5.0e6, 25).into_iter().map(|z| vec![z]).collect();
    groups.push(scalar_group("x_c_dc", "none", z_xc.clone(), 1e-12, "DC critical frequency fit", |i| {
        greens::x_c_dc(i[0])
    }));
    groups.push(scalar_group("x_c_br", "none", z_xc.clone(), 1e-12, "BR critical frequency fit", |i| {
        greens::x_c_br(i[0])
    }));
    groups.push(scalar_group("x_c", "none", z_xc.clone(), 1e-12, "combined critical frequency", |i| {
        greens::x_c(i[0])
    }));

    let mut ps_inputs = Vec::new();
    for &z in &[1.0e4, 1.0e5, 1.0e6] {
        for &x in &[1.0e-3, 1.0e-2, 0.1, 1.0, 10.0] {
            ps_inputs.push(vec![x, z]);
        }
    }
    groups.push(scalar_group(
        "photon_survival_probability",
        "none",
        ps_inputs,
        1e-11,
        "analytic P_s = exp(-x_c/x)",
        |i| greens::photon_survival_probability(i[0], i[1]),
    ));

    // --- Numerical P_s (tau_ff integral; the referee-flagged call site) ----
    // Same trapezoid rule and step count on both sides; the tolerance is set
    // by the ionization-history table agreement, not the quadrature.
    let mut psn_inputs = Vec::new();
    for &z_h in &[300.0, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 5.0e4, 2.0e5] {
        for &x in &[1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 0.1, 1.0] {
            psn_inputs.push(vec![x, z_h]);
        }
    }
    for (label, cosmo) in cosmos {
        let c = cosmo;
        groups.push(scalar_group(
            "photon_survival_probability_numerical",
            label,
            psn_inputs.clone(),
            1e-5,
            "P_s = exp(-tau_ff), DC+BR optical depth with Peebles+He X_e",
            move |i| greens::photon_survival_probability_numerical(i[0], i[1], c),
        ));
    }

    // --- Spectral shapes -----------------------------------------------------
    let x_shapes: Vec<Vec<f64>> = logspace(1.0e-4, 30.0, 25).into_iter().map(|x| vec![x]).collect();
    groups.push(scalar_group("planck", "none", x_shapes.clone(), 1e-12, "Planck occupation", |i| {
        spectrum::planck(i[0])
    }));
    groups.push(scalar_group("g_bb", "none", x_shapes.clone(), 1e-12, "blackbody derivative shape", |i| {
        spectrum::g_bb(i[0])
    }));
    groups.push(scalar_group("mu_shape", "none", x_shapes.clone(), 1e-12, "mu distortion shape M(x)", |i| {
        spectrum::mu_shape(i[0])
    }));
    groups.push(scalar_group("y_shape", "none", x_shapes.clone(), 1e-12, "y distortion shape Y_SZ(x)", |i| {
        spectrum::y_shape(i[0])
    }));

    // --- Cosmology background ------------------------------------------------
    let z_bg: Vec<Vec<f64>> = vec![
        0.0, 10.0, 100.0, 500.0, 1100.0, 3000.0, 1.0e4, 1.0e5, 1.0e6, 5.0e6,
    ]
    .into_iter()
    .map(|z| vec![z])
    .collect();
    for (label, cosmo) in cosmos {
        let c = cosmo;
        groups.push(scalar_group("hubble", label, z_bg.clone(), 1e-12, "H(z) [1/s]", move |i| {
            c.hubble(i[0])
        }));
        groups.push(scalar_group("n_hydrogen", label, z_bg.clone(), 1e-12, "n_H(z) [1/m^3]", move |i| {
            c.n_h(i[0])
        }));
        groups.push(scalar_group(
            "baryon_photon_ratio",
            label,
            z_bg.clone(),
            1e-11,
            "R(z) = 3 rho_b / (4 rho_gamma)",
            move |i| c.baryon_photon_ratio(i[0]),
        ));
        groups.push(scalar_group("rho_gamma", label, z_bg.clone(), 1e-12, "photon energy density [J/m^3]", move |i| {
            c.rho_gamma(i[0])
        }));
        groups.push(scalar_group("omega_gamma", label, vec![vec![0.0]], 1e-12, "Omega_gamma today", move |_| {
            c.omega_gamma()
        }));
        groups.push(scalar_group(
            "cosmic_time",
            label,
            z_bg.clone(),
            1e-4,
            "t(z) by quadrature; tolerance set by quadrature scheme difference",
            move |i| c.cosmic_time(i[0]),
        ));
        // Recombination history: same Peebles TLA + He Saha algorithm on both
        // sides (forward Euler, dz = 0.5, Saha switch at X_H = 0.99).
        let z_xe: Vec<Vec<f64>> = logspace(1.0, 1.0e5, 60).into_iter().map(|z| vec![z]).collect();
        groups.push(scalar_group(
            "ionization_fraction",
            label,
            z_xe,
            1e-5,
            "Peebles TLA + He Saha; Rust integrates the ODE directly to z, Python interpolates a dz=0.5 table",
            move |i| recombination::ionization_fraction(i[0], c),
        ));
        // Compton y(z): Rust uses 128-pt midpoint with T_m decoupling below
        // z=200; Python mirrors the physics with 32-pt Gauss-Legendre, so the
        // tolerance is set by quadrature, not physics.
        let z_y: Vec<Vec<f64>> = vec![500.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
            .into_iter()
            .map(|z| vec![z])
            .collect();
        groups.push(scalar_group(
            "compton_y_parameter",
            label,
            z_y,
            2e-3,
            "y_gamma(z); quadrature schemes differ (midpoint-128 vs GL-32)",
            move |i| c.compton_y_parameter(i[0]),
        ));
    }

    // --- Dark photon (NWA helpers) -------------------------------------------
    let z_pl: Vec<Vec<f64>> = logspace(10.0, 1.0e7, 30).into_iter().map(|z| vec![z]).collect();
    let m_grid: Vec<Vec<f64>> = logspace(1.0e-14, 1.0e-4, 21).into_iter().map(|m| vec![m]).collect();
    for (label, cosmo) in cosmos {
        let c = cosmo;
        groups.push(scalar_group(
            "plasma_frequency_ev",
            label,
            z_pl.clone(),
            1e-6,
            "omega_pl(z) [eV]; tolerance from X_e table",
            move |i| dark_photon::plasma_frequency_ev(i[0], c),
        ));
        groups.push(Group {
            name: "resonance_redshift",
            cosmo: label,
            inputs: m_grid.clone(),
            values: m_grid
                .iter()
                .map(|i| dark_photon::resonance_redshift(i[0], c).map(|z| vec![z]))
                .collect(),
            rtol: 1e-5,
            note: "bisection to 1e-8; null = no resonance in bracket",
        });
        groups.push(Group {
            name: "gamma_con",
            cosmo: label,
            inputs: m_grid.clone(),
            values: m_grid
                .iter()
                .map(|i| {
                    dark_photon::gamma_con(1.0e-7, i[0], c).map(|(gc, zr)| vec![gc, zr])
                })
                .collect(),
            rtol: 5e-3,
            note: "CCJ24 Eq. 6 at epsilon = 1e-7; returns (gamma_con, z_res). Tolerance set by the finite-difference d ln omega_pl^2/d ln a acting on the interpolated X_e table",
        });
    }

    // --- Photon-injection Green's function ----------------------------------
    let mut gfp_inputs = Vec::new();
    for &(x_inj, z_h, sigma_x) in &[(1.0_f64, 1.0e4_f64, 0.05_f64), (0.01, 3.0e4, 0.0), (5.0, 3.0e5, 0.1)] {
        for x_obs in logspace(0.1, 20.0, 12) {
            gfp_inputs.push(vec![x_obs, x_inj, z_h, sigma_x]);
        }
    }
    for (label, cosmo) in cosmos {
        let c = cosmo;
        groups.push(scalar_group(
            "greens_function_photon",
            label,
            gfp_inputs.clone(),
            1e-3,
            "G_ph(x_obs; x_inj, z_h, sigma_x); tolerance from y_gamma quadrature",
            move |i| greens::greens_function_photon(i[0], i[1], i[2], i[3], c),
        ));
    }

    let mut mfp_inputs = Vec::new();
    for &z_h in &[3.0e5, 1.0e6, 3.0e6] {
        for &x_inj in &[0.01, 0.1, 1.0, 3.6, 10.0] {
            mfp_inputs.push(vec![x_inj, z_h, 1.0e-5]);
        }
    }
    groups.push(scalar_group(
        "mu_from_photon_injection",
        "none",
        mfp_inputs,
        1e-11,
        "mu-era photon injection; closed form",
        |i| greens::mu_from_photon_injection(i[0], i[1], i[2]),
    ));

    write_json(&groups, &path).expect("failed to write fixture file");
    let n_points: usize = groups.iter().map(|g| g.inputs.len()).sum();
    println!(
        "wrote {} groups ({} evaluation points) to {}",
        groups.len(),
        n_points,
        path
    );
}
