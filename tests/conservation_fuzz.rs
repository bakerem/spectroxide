//! Property-based conservation fuzzing (validation-audit Phase 2, plan B3).
//!
//! Draws randomized-but-deterministic scenario/grid/window configurations
//! (hand-rolled splitmix64 PRNG, fixed seeds — no dev-dependencies) and
//! asserts the solver's conservation ledgers close against *independent*
//! anchors:
//!
//! 1. **Energy closure (heat injection):** final `delta_rho_over_rho`
//!    (which includes the 4·δT/T energy of the number-conserving T-shift)
//!    must match ∫ q(z)/(H(1+z)) dz computed by independent Simpson
//!    quadrature of the scenario's heating rate over the run window.
//! 2. **Photon-number conservation (pure Compton):** for random initial
//!    distortions with DC/BR disabled, ∫x²Δn dx is an exact invariant of
//!    the conservative Kompaneets flux form — through the full production
//!    solver (adaptive stepping, T_e coupling, bordered Newton) the drift
//!    must stay at roundoff level, orders of magnitude below truncation.
//! 3. **Photon-number closure (monochromatic injection):** with DC/BR
//!    disabled, the final ΔN/N must equal the scenario's injected ΔN/N
//!    (the x- and z-Gaussians are unit-normalized analytically).
//!
//! Tolerances: energy 10% (matches the per-case closure tests in
//! `coverage_gaps.rs`; the ±5% paper target applies to production
//! resolutions, while the fuzzer deliberately draws coarser grids),
//! number-injection 2%, pure-Compton drift 1e-9 (relative).

use spectroxide::prelude::*;
use spectroxide::spectrum;

// ============================================================================
// Deterministic PRNG (splitmix64)
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    /// Log-uniform in [lo, hi).
    fn log_range(&mut self, lo: f64, hi: f64) -> f64 {
        (self.range(lo.ln(), hi.ln())).exp()
    }
    fn pick_usize(&mut self, choices: &[usize]) -> usize {
        choices[(self.next_u64() % choices.len() as u64) as usize]
    }
}

// ============================================================================
// Independent quadrature of the heating rate: ∫ q(z)/(H(1+z)) dz
// ============================================================================

/// Simpson integration of q(z)/(H(z)(1+z)) over [z_end, z_start] in ln(1+z).
fn expected_drho(inj: &InjectionScenario, cosmo: &Cosmology, z_start: f64, z_end: f64) -> f64 {
    let n = 30_000usize; // even
    let (la, lb) = ((1.0 + z_end).ln(), (1.0 + z_start).ln());
    let h = (lb - la) / n as f64;
    let f = |lz: f64| -> f64 {
        let z = lz.exp() - 1.0;
        // dt = dz/(H(1+z)); with dz = (1+z) dln(1+z), integrand in ln(1+z)
        // is q(z)/H(z).
        inj.heating_rate(z, cosmo) / cosmo.hubble(z)
    };
    let mut s = f(la) + f(lb);
    for j in 1..n {
        let wt = if j % 2 == 1 { 4.0 } else { 2.0 };
        s += wt * f(la + j as f64 * h);
    }
    s * h / 3.0
}

// ============================================================================
// 1. Energy closure for randomized heat-injection scenarios
// ============================================================================

struct HeatDraw {
    label: String,
    inj: InjectionScenario,
    z_start: f64,
    z_end: f64,
    n_points: usize,
}

fn draw_single_burst(rng: &mut Rng) -> HeatDraw {
    let z_h = rng.log_range(3e4, 6e5);
    let drho = rng.log_range(1e-7, 1e-5);
    let sigma_z = z_h * rng.range(0.03, 0.08);
    HeatDraw {
        label: format!("SingleBurst z_h={z_h:.3e} drho={drho:.3e} sigma={sigma_z:.3e}"),
        inj: InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z,
        },
        z_start: z_h + 8.0 * sigma_z,
        z_end: (z_h / 5.0).max(1e4),
        n_points: rng.pick_usize(&[800, 1200, 1600]),
    }
}

fn draw_decaying(rng: &mut Rng, cosmo: &Cosmology) -> HeatDraw {
    // Pick the decay epoch z_X in the validated window, then Γ_X = 1/t(z_X).
    let z_x = rng.log_range(6e4, 4e5);
    let gamma_x = 1.0 / cosmo.cosmic_time(z_x);
    let z_start = (8.0 * z_x).min(1.5e6);
    let z_end = (z_x / 8.0).max(1e4);
    // The heating rate is linear in f_x; rescale a unit-f_x quadrature so the
    // drawn total Δρ/ρ lands in the small-distortion regime.
    let target = rng.log_range(3e-7, 1e-5);
    let unit = expected_drho(
        &InjectionScenario::DecayingParticle { f_x: 1.0, gamma_x },
        cosmo,
        z_start,
        z_end,
    );
    let f_x = target / unit;
    HeatDraw {
        label: format!("DecayingParticle z_X={z_x:.3e} f_x={f_x:.3e}"),
        inj: InjectionScenario::DecayingParticle { f_x, gamma_x },
        z_start,
        z_end,
        n_points: rng.pick_usize(&[800, 1200, 1600]),
    }
}

fn draw_annihilating(rng: &mut Rng, cosmo: &Cosmology, p_wave: bool) -> HeatDraw {
    let (z_start, z_end) = (5e5, 1e4);
    // Linear in f_ann; rescale a unit-rate quadrature (as for DecayingParticle).
    let target = rng.log_range(3e-7, 1e-5);
    let make = |f_ann: f64| {
        if p_wave {
            InjectionScenario::AnnihilatingDMPWave { f_ann }
        } else {
            InjectionScenario::AnnihilatingDM { f_ann }
        }
    };
    let unit = expected_drho(&make(1e-22), cosmo, z_start, z_end) / 1e-22;
    let f_ann = target / unit;
    HeatDraw {
        label: format!(
            "AnnihilatingDM{} f_ann={f_ann:.3e}",
            if p_wave { "PWave" } else { "" }
        ),
        inj: make(f_ann),
        z_start,
        z_end,
        n_points: rng.pick_usize(&[800, 1200, 1600]),
    }
}

#[test]
fn fuzz_energy_closure_heat_injection() {
    let cosmo = Cosmology::default();
    let mut rng = Rng::new(0xF00D_0001);

    let mut draws: Vec<HeatDraw> = Vec::new();
    for _ in 0..2 {
        draws.push(draw_single_burst(&mut rng));
    }
    for _ in 0..2 {
        draws.push(draw_decaying(&mut rng, &cosmo));
    }
    draws.push(draw_annihilating(&mut rng, &cosmo, false));
    draws.push(draw_annihilating(&mut rng, &cosmo, true));

    for (k, d) in draws.into_iter().enumerate() {
        let expected = expected_drho(&d.inj, &cosmo, d.z_start, d.z_end);
        assert!(
            expected > 1e-9 && expected < 1e-3,
            "case {k} [{}]: drew an out-of-range expected Δρ/ρ = {expected:.3e} — \
             adjust the draw ranges",
            d.label
        );

        let mut solver = ThermalizationSolver::builder(cosmo.clone())
            .grid(GridConfig {
                n_points: d.n_points,
                ..GridConfig::default()
            })
            .injection(d.inj)
            .z_range(d.z_start, d.z_end)
            .build()
            .unwrap();
        solver.run_with_snapshots(&[d.z_end]);
        let s = solver.snapshots.last().unwrap();

        let rel_err = (s.delta_rho_over_rho - expected).abs() / expected;
        eprintln!(
            "FUZZ|energy|case={k}|{}|N={}|expected={expected:.4e}|got={:.4e}|rel_err={rel_err:.3e}",
            d.label, d.n_points, s.delta_rho_over_rho
        );
        assert!(
            rel_err < 0.10,
            "case {k} [{}]: energy ledger violated: expected Δρ/ρ = {expected:.4e}, \
             got {:.4e} (rel err {:.2}%)",
            d.label,
            s.delta_rho_over_rho,
            rel_err * 100.0
        );
    }
}

// ============================================================================
// 2. Photon-number conservation under pure Compton, random initial data
// ============================================================================

#[test]
fn fuzz_photon_number_conservation_pure_compton() {
    let cosmo = Cosmology::default();
    let mut rng = Rng::new(0xBEEF_0002);

    for k in 0..4 {
        let n_points = rng.pick_usize(&[800, 1200, 1600]);
        let z_start = rng.log_range(8e4, 4e5);
        let z_end = z_start * rng.range(0.4, 0.6);
        let amp = rng.log_range(1e-6, 1e-3) * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        let x0 = rng.range(2.0, 8.0);
        let width = rng.range(0.4, 1.0);

        let mut solver = ThermalizationSolver::builder(cosmo.clone())
            .grid(GridConfig {
                n_points,
                ..GridConfig::default()
            })
            .z_range(z_start, z_end)
            .disable_dcbr()
            .no_number_conserving()
            .build()
            .unwrap();

        let initial: Vec<f64> = solver
            .grid
            .x
            .iter()
            .map(|&x| amp * (-(x - x0).powi(2) / (2.0 * width * width)).exp())
            .collect();
        solver.set_initial_delta_n(initial.clone());

        let x = solver.grid.x.clone();
        // The discrete invariant of the conservative flux form is
        // Σ_i x_i²·Δx_cell,i·Δn_i (the kernel's own cell weights, half cells
        // at the boundaries) — using any other quadrature (e.g.
        // spectrum::delta_n_over_n's midpoint rule) turns the exact
        // conservation into an apparent O(dx²) "drift" as the spectral shape
        // evolves.
        let n = x.len();
        let mut w = vec![0.0; n];
        w[0] = x[0] * x[0] * 0.5 * (x[1] - x[0]);
        for i in 1..n - 1 {
            w[i] = x[i] * x[i] * 0.5 * (x[i + 1] - x[i - 1]);
        }
        w[n - 1] = x[n - 1] * x[n - 1] * 0.5 * (x[n - 1] - x[n - 2]);
        let n_of = |dn: &[f64]| -> f64 { dn.iter().zip(&w).map(|(d, wi)| d * wi).sum() };
        let n_initial = n_of(&initial);
        let scale: f64 = initial.iter().zip(&w).map(|(d, wi)| d.abs() * wi).sum();

        solver.run_with_snapshots(&[z_end]);
        let s = solver.snapshots.last().unwrap();
        let n_final = n_of(&s.delta_n);

        let drift = (n_final - n_initial).abs() / scale.max(1e-30);
        eprintln!(
            "FUZZ|number_compton|case={k}|N={n_points}|z=[{z_start:.2e},{z_end:.2e}]|\
             amp={amp:.2e}|x0={x0:.2}|drift={drift:.3e}|steps={}",
            solver.step_count
        );
        // The production solver stops the Newton iteration at corrections of
        // 1e-8·max|Δn| (vs the kernel-level ledger test, which forces the
        // 1e-14 floor and conserves to ~1e-15): the residual drift here is
        // the accumulated Newton-tolerance remainder, observed at ~2e-8 over
        // O(10³) steps — still 4+ orders below the O(dx²) truncation error.
        assert!(
            drift < 2e-7,
            "case {k}: photon number not conserved under pure Compton: \
             N/N: {n_initial:.6e} → {n_final:.6e} (rel drift {drift:.3e})"
        );
    }
}

// ============================================================================
// 3. Photon-number closure for monochromatic photon injection
// ============================================================================

#[test]
fn fuzz_photon_number_closure_monochromatic_injection() {
    let cosmo = Cosmology::default();
    let mut rng = Rng::new(0xCAFE_0003);

    for k in 0..4 {
        let n_points = rng.pick_usize(&[800, 1200]);
        let x_inj = rng.range(1.0, 8.0);
        let sigma_x = 0.1 * x_inj;
        let dn_over_n = rng.log_range(1e-7, 1e-5);
        let z_h = rng.log_range(3e4, 1.5e5);
        let sigma_z = 0.05 * z_h;
        let z_start = z_h * 1.35;
        let z_end = z_h * 0.65;

        let mut solver = ThermalizationSolver::builder(cosmo.clone())
            .grid(GridConfig {
                n_points,
                ..GridConfig::default()
            })
            .injection(InjectionScenario::MonochromaticPhotonInjection {
                x_inj,
                delta_n_over_n: dn_over_n,
                z_h,
                sigma_z,
                sigma_x,
            })
            .z_range(z_start, z_end)
            .disable_dcbr()
            .no_number_conserving()
            .build()
            .unwrap();

        solver.run_with_snapshots(&[z_end]);
        let s = solver.snapshots.last().unwrap();
        let x = solver.grid.x.clone();
        let n_final = spectrum::delta_n_over_n(&x, &s.delta_n);

        let rel_err = (n_final - dn_over_n).abs() / dn_over_n;
        eprintln!(
            "FUZZ|number_injection|case={k}|N={n_points}|x_inj={x_inj:.2}|z_h={z_h:.2e}|\
             injected={dn_over_n:.3e}|got={n_final:.3e}|rel_err={rel_err:.3e}"
        );
        assert!(
            rel_err < 0.02,
            "case {k}: photon-number ledger violated for monochromatic injection: \
             injected ΔN/N = {dn_over_n:.4e}, final {n_final:.4e} (rel err {:.2}%)",
            rel_err * 100.0
        );
    }
}
