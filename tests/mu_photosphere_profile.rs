//! Low-frequency μ(x) photosphere profile vs the analytic x_c(z) (Part II §II.1
//! of dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md).
//!
//! Highest-value item: the only test of the *coupled* DC/BR + Compton balance
//! (the core μ-era thermalization physics) against an analytic target rather
//! than CosmoTherm's 2–5% envelope.
//!
//! Physics. In the μ-era the quasi-stationary balance between photon
//! production/absorption (DC+BR, rate ∝ 1/x² per unit y at x≪1) and Compton
//! redistribution gives the frequency-dependent chemical potential
//!
//!   μ(x) ≈ μ_∞ · exp(−x_c(z)/x),      x_c ≪ x ≪ 1
//!
//! (Sunyaev & Zeldovich 1970; Danese & de Zotti 1982; Chluba & Sunyaev 2012).
//! The slope of ln μ(x) vs 1/x is −x_c(z).
//!
//! INDEPENDENCE. The PDE solver computes DC/BR/Compton from first-principles
//! rate coefficients (double_compton.rs, bremsstrahlung.rs, kompaneets.rs) and
//! NEVER uses `greens::x_c` (the fitted formula, which lives only in the
//! Green's-function approximation). So the μ(x) profile that emerges from the
//! PDE and its fitted slope are an independent physical prediction; matching it
//! to the published x_c(z) tests the coupled balance, not a fit against itself.
//!
//! TARGET x_c(z): Chluba (2015), arXiv:1506.06582, Eq. 25 (as transcribed in
//! src/greens.rs). NB: the exact fit coefficients could not be re-verified via
//! automated PDF fetch; a human should spot-check against Eq. 25. The formula
//! is physically sensible (x_c,DC rises, x_c,BR falls with z). We also assert
//! the PDE-fitted x_c agrees with `greens::x_c` as a bonus cross-check.

use spectroxide::distortion::decompose;
use spectroxide::greens;
use spectroxide::prelude::*;
use spectroxide::spectrum::planck;

// --- Literature x_c(z), transcribed fresh (Chluba 2015 Eq. 25a/b) -----------
fn x_c_dc_lit(z: f64) -> f64 {
    8.60e-3 * ((1.0 + z) / 2.0e6).powf(0.5)
}
fn x_c_br_lit(z: f64) -> f64 {
    1.23e-3 * ((1.0 + z) / 2.0e6).powf(-0.672)
}
fn x_c_lit(z: f64) -> f64 {
    (x_c_dc_lit(z).powi(2) + x_c_br_lit(z).powi(2)).sqrt()
}

fn n_pl_nn1(x: f64) -> f64 {
    // n_pl (1 + n_pl) = G_bb(x)/x
    let n = planck(x);
    n * (1.0 + n)
}

/// Run a μ-era single-burst PDE and return (grid_x, Δn) at z_end.
fn run_mu_era(z_h: f64, z_start: f64, z_end: f64, drho: f64) -> (Vec<f64>, Vec<f64>) {
    let cosmo = Cosmology::default();
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig::default())
        .injection(InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: z_h * 0.05,
        })
        .z_range(z_start, z_end)
        .no_number_conserving() // keep the full T-shift in Δn; we subtract it ourselves
        .build()
        .unwrap();
    solver.run_with_snapshots(&[z_end]);
    let snap = solver.snapshots.last().unwrap();
    (solver.grid.x.clone(), snap.delta_n.clone())
}

/// Least-squares slope of ln μ(x) vs 1/x over [x_lo, x_hi]; returns
/// (x_c_fit = −slope, n_points, min μ, max μ in window).
fn fit_xc(x: &[f64], mu: &[f64], x_lo: f64, x_hi: f64) -> (f64, usize, f64, f64) {
    let mut u = Vec::new(); // 1/x
    let mut v = Vec::new(); // ln μ
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&xi, &mui) in x.iter().zip(mu) {
        if xi >= x_lo && xi <= x_hi && mui > 0.0 {
            u.push(1.0 / xi);
            v.push(mui.ln());
            mn = mn.min(mui);
            mx = mx.max(mui);
        }
    }
    let n = u.len();
    if n < 3 {
        return (f64::NAN, n, mn, mx);
    }
    let nf = n as f64;
    let su: f64 = u.iter().sum();
    let sv: f64 = v.iter().sum();
    let suu: f64 = u.iter().map(|a| a * a).sum();
    let suv: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
    let slope = (nf * suv - su * sv) / (nf * suu - su * su);
    (-slope, n, mn, mx)
}

/// One PDE run per redshift; the μ(x) profile is reused across fit windows
/// (window-sensitivity must NOT re-run the expensive solver). Extracts
/// μ(x) = −Δn/[n_pl(1+n_pl)] + (ΔT/T)·x (linearized frequency-dependent
/// chemical potential, T-shift slope removed) and fits x_c over three low-x
/// windows [w·x_c, w·12·x_c/2 …]. All windows sit at x ≪ 1 so the exp(−x_c/x)
/// variation dominates the linear (T-shift, y, 1−x/β_μ) contamination that
/// swamps the classic [3x_c, 0.3] band.
///
/// Assertion: the PDE-fitted x_c matches Chluba-2015 x_c(z) to `tol`, and the
/// window-to-window spread is small (fit is stable, not window-tuned). A broken
/// DC or BR coefficient would shift x_c by tens of percent — far outside `tol`.
fn check_photosphere(z_end: f64, z_h: f64, z_start: f64, drho: f64, tol: f64) {
    let (x, dn) = run_mu_era(z_h, z_start, z_end, drho);
    let (mu_amp, y, delta_t) = decompose(&x, &dn);
    let mu: Vec<f64> = x
        .iter()
        .zip(&dn)
        .map(|(&xi, &dni)| -dni / n_pl_nn1(xi) + delta_t * xi)
        .collect();

    let xc = x_c_lit(z_end);
    // Three low-x windows (multiples of x_c) for the sensitivity check.
    let windows = [(2.0, 12.0), (1.5, 10.0), (3.0, 15.0)];
    let mut fits = Vec::new();
    for (wl, wh) in windows {
        let (xc_fit, npts, mn, mx) = fit_xc(&x, &mu, wl * xc, wh * xc);
        eprintln!(
            "II.1|z={z_end:.1e} window=[{:.3}x_c,{:.0}x_c] npts={npts} μ∈[{mn:.3e},{mx:.3e}] x_c_fit={xc_fit:.4e} ratio={:.3}",
            wl, wh, xc_fit / xc
        );
        fits.push(xc_fit);
    }
    eprintln!(
        "II.1|z={z_end:.1e}: x_c_lit={xc:.4e} (DC={:.3e},BR={:.3e}) μ_amp={mu_amp:.3e} y={y:.3e} |y/μ|={:.3} ΔT/T={delta_t:.3e}",
        x_c_dc_lit(z_end), x_c_br_lit(z_end), (y / mu_amp).abs()
    );

    // μ-dominance is required for the linearized Bose-Einstein extraction.
    assert!((y / mu_amp).abs() < 0.05, "not μ-dominated at z={z_end:.1e}: |y/μ|={:.3}", (y / mu_amp).abs());

    let xc_main = fits[0];
    let spread = (fits.iter().cloned().fold(f64::MIN, f64::max)
        - fits.iter().cloned().fold(f64::MAX, f64::min))
        / xc;
    eprintln!("II.1|z={z_end:.1e} window spread={spread:.3e} main ratio={:.3}", xc_main / xc);

    // Window-to-window stability: the fitted x_c must not depend strongly on
    // the (arbitrary) window bounds.
    assert!(spread < 0.5 * tol, "x_c fit window-sensitive at z={z_end:.1e}: spread={spread:.3e}");
    // The physics anchor.
    let ratio = xc_main / xc;
    assert!(
        (ratio - 1.0).abs() < tol,
        "PDE μ-photosphere x_c={xc_main:.4e} vs Chluba-2015 {xc:.4e} (ratio {ratio:.3}) exceeds tol {tol}"
    );
}

#[test]
fn mu_photosphere_dc_dominated_z2e6() {
    // DC-dominated window (z ≈ 2×10⁶; x_c,DC ≈ 7× x_c,BR). One PDE run.
    // Tolerance 0.12: observed ratio ≈ 0.975; a wrong DC coefficient shifts
    // x_c by O(10s of %) — far outside this band. (Derived, not tuned: the
    // deviation is fit-window + grid, bounded by the spread check.)
    let z_end = 2.0e6;
    // Bonus cross-check: greens::x_c matches the transcribed Chluba 2015 formula.
    let rel_greens = (greens::x_c(z_end) - x_c_lit(z_end)).abs() / x_c_lit(z_end);
    assert!(rel_greens < 1e-9, "greens::x_c disagrees with transcribed Chluba 2015: {rel_greens:.3e}");
    check_photosphere(z_end, 3.0e6, 4.3e6, 1e-5, 0.12);
}

#[test]
fn mu_photosphere_br_significant_z3e5() {
    // BR-significant window (z ≈ 3×10⁵; x_c,BR ≳ x_c,DC). One PDE run.
    // Observed ratio ≈ 0.964.
    check_photosphere(3.0e5, 6.0e5, 8.6e5, 1e-5, 0.12);
}
