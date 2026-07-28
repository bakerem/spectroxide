//! Exact physics identities the code must satisfy.
//!
//! Every test here anchors a *closed-form* or *published* number, never a value
//! read off from code output (CLAUDE.md pitfall #9). Provenance and the measured
//! numbers live in `dev/audit/PHYSICS_CHECKS_STATUS_2026-07-26.md`; the IDs
//! (T-PC-n) refer to that file's plan table.

use spectroxide::bremsstrahlung::br_emission_coefficient;
use spectroxide::constants::{
    theta_z, BETA_MU, C_LIGHT, G1_PLANCK, G2_PLANCK, G3_PLANCK, KAPPA_C, SIGMA_THOMSON, ZETA_3,
};
use spectroxide::cosmology::Cosmology;
use spectroxide::double_compton::dc_emission_coefficient;
use spectroxide::grid::FrequencyGrid;
use spectroxide::kompaneets::kompaneets_step_nonlinear;
use spectroxide::recombination::RecombinationHistory;
use spectroxide::spectrum::{g_bb, mu_shape, planck, y_shape};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Composite Simpson of ∫ f(x) dx over [x_lo, x_hi] performed in u = ln x, so
/// ∫ f dx = ∫ f(e^u) e^u du. All integrands here are smooth in ln x and decay
/// exponentially at both ends, so the O(h⁴) error reaches machine precision.
fn simpson_log<F: Fn(f64) -> f64>(f: F, x_lo: f64, x_hi: f64, n_intervals: usize) -> f64 {
    assert!(n_intervals % 2 == 0, "Simpson needs an even interval count");
    let (u_lo, u_hi) = (x_lo.ln(), x_hi.ln());
    let h = (u_hi - u_lo) / n_intervals as f64;
    let g = |u: f64| {
        let x = u.exp();
        f(x) * x
    };
    let mut acc = g(u_lo) + g(u_hi);
    for i in 1..n_intervals {
        let u = u_lo + i as f64 * h;
        acc += if i % 2 == 1 { 4.0 } else { 2.0 } * g(u);
    }
    acc * h / 3.0
}

/// Thomson optical depth from z = 0 out to each grid point, plus the visibility
/// function g = (dτ/dz) e^{−τ}. Returns (z, τ, g) on a uniform grid.
fn thomson_depth(cosmo: &Cosmology, z_max: f64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hist = RecombinationHistory::new(cosmo);
    let dz = z_max / n as f64;
    let mut z = Vec::with_capacity(n + 1);
    let mut dtau = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let zi = i as f64 * dz;
        // X_e is tabulated down to z = 1; below that the contribution to τ is
        // ~10⁻⁵ (no reionization in this model), so clamping is harmless.
        let x_e = hist.x_e(zi.max(1.0));
        let n_e = cosmo.n_e(zi, x_e);
        dtau.push(n_e * SIGMA_THOMSON * C_LIGHT / (cosmo.hubble(zi) * (1.0 + zi)));
        z.push(zi);
    }
    let mut tau = vec![0.0; n + 1];
    for i in 1..=n {
        tau[i] = tau[i - 1] + 0.5 * (dtau[i] + dtau[i - 1]) * dz;
    }
    let g: Vec<f64> = dtau
        .iter()
        .zip(&tau)
        .map(|(d, t)| d * (-t).exp())
        .collect();
    (z, tau, g)
}

/// Linear interpolation of the z at which a monotone y(z) first reaches `target`.
fn crossing(z: &[f64], y: &[f64], target: f64) -> f64 {
    for i in 1..y.len() {
        if y[i] >= target {
            let f = (target - y[i - 1]) / (y[i] - y[i - 1]);
            return z[i - 1] + f * (z[i] - z[i - 1]);
        }
    }
    f64::NAN
}

// ---------------------------------------------------------------------------
// T-PC-1 — Thomson optical depth and the last-scattering surface
// ---------------------------------------------------------------------------

/// **T-PC-1.** τ_T(z) = ∫₀^z n_e σ_T c / [H(1+z′)(1+z′)⁻¹ … ] dz′ must reach
/// unity at the last-scattering redshift.
///
/// Anchor: **Planck 2018** (arXiv:1807.06209, Table 2, TT,TE,EE+lowE+lensing)
/// z_* = 1089.92 ± 0.25, defined by τ(z_*) = 1.
///
/// Why this test exists: nothing in `src/` or `tests/` computed τ_T before, yet
/// the integrand X_e(z)·n_e·σ_T·c/[H(1+z)] is *exactly* the quantity behind the
/// Compton-y broadening y_γ and the photon survival probability P_s — the
/// photon-injection channel that `dev/audit/coverage_matrix.md` rows 6/7 flag as
/// the least independently anchored part of the code.
///
/// Tolerance (±12, i.e. 1.1%) is set by two effects, not by wishful thinking:
/// - **convention:** if the reference z_* is computed with reionization included
///   (τ_reio ≈ 0.054) the crossing moves down by 0.054/(dτ/dz) ≈ 6.8. This model
///   has no reionization, so ±7 of the band is definitional.
/// - **sensitivity:** a uniform +10% error in the integrand moves z_* by only
///   −8.05 (measured). So ±12 constrains n_e X_e σ_T c/H to ≈15%. That is a
///   weak-but-real bound; tightening it needs a like-for-like HyRec-2 run
///   (`dev/audit/xe_hyrec_comparison.md` has the infrastructure) — recorded as
///   the upgrade path, not silently assumed.
#[test]
fn test_thomson_optical_depth_and_last_scattering() {
    const Z_STAR_PLANCK18: f64 = 1089.92;

    for (label, cosmo) in [
        ("planck2018", Cosmology::planck2018()),
        ("default", Cosmology::default()),
    ] {
        let (z, tau, g) = thomson_depth(&cosmo, 1400.0, 140_000);
        let z_star = crossing(&z, &tau, 1.0);

        // Visibility function: peak position, the τ there, and the FWHM.
        let (mut i_pk, mut g_pk) = (0usize, 0.0f64);
        for (i, &gi) in g.iter().enumerate() {
            if gi > g_pk {
                g_pk = gi;
                i_pk = i;
            }
        }
        let above: Vec<f64> = z
            .iter()
            .zip(&g)
            .filter(|&(_, &gi)| gi > 0.5 * g_pk)
            .map(|(&zi, _)| zi)
            .collect();
        let fwhm = above.last().unwrap() - above.first().unwrap();

        eprintln!(
            "[{label}] z_*(τ=1) = {z_star:.2} (Planck18 {Z_STAR_PLANCK18} ± 0.25), \
             visibility peak z = {:.2} at τ = {:.3}, FWHM Δz = {fwhm:.0}",
            z[i_pk], tau[i_pk]
        );

        assert!(
            (z_star - Z_STAR_PLANCK18).abs() < 12.0,
            "[{label}] τ_T = 1 at z = {z_star:.2}, Planck 2018 gives \
             z_* = {Z_STAR_PLANCK18} ± 0.25 (band ±12 = reionization convention \
             ±7 ⊕ 15% integrand sensitivity)"
        );

        // The visibility peaks *before* τ = 1: g = τ′e^{−τ} is maximal where
        // τ″ = τ′², which for a rising τ′ happens at τ slightly under 1.
        assert!(
            z[i_pk] < z_star,
            "[{label}] visibility must peak below z_*: {:.2} vs {z_star:.2}",
            z[i_pk]
        );
        assert!(
            (0.7..1.0).contains(&tau[i_pk]),
            "[{label}] τ at the visibility peak = {:.4}, expected ∈ (0.7, 1)",
            tau[i_pk]
        );
        // Width of the last-scattering surface. Loose band: this is a shape
        // sanity check, not a literature anchor (quoted FWHM values vary with
        // the definition used).
        assert!(
            (150.0..260.0).contains(&fwhm),
            "[{label}] last-scattering FWHM Δz = {fwhm:.0}, expected ∈ (150, 260)"
        );
    }
}

// ---------------------------------------------------------------------------
// T-PC-2 — exact moment identities of the three distortion shapes
// ---------------------------------------------------------------------------

/// **T-PC-2.** The number (∫x²·dx) and energy (∫x³·dx) moments of the coded
/// spectral shapes G_bb, M, Y all have closed forms. Six identities:
///
/// | shape | ∫x²·dx | ∫x³·dx | meaning |
/// |---|---|---|---|
/// | G_bb | 3G₂ | 4G₃ | ΔN/N = 3ΔT/T, Δρ/ρ = 4ΔT/T |
/// | M | **0** | (κ_c/3)G₃ | μ carries no photons; μ = (3/κ_c)Δρ/ρ = 1.401 Δρ/ρ |
/// | Y | **0** | 4G₃ | Compton conserves N; Δρ/ρ = 4y |
///
/// The two zeros are the strongest statements: ∫x²M dx = 0 *is* the defining
/// condition for β_μ = 3ζ(3)/ζ(2), and ∫x²Y dx = 0 is photon-number
/// conservation under Compton scattering. Both follow from
/// ∫₀^∞ xⁿ e^x/(e^x−1)² dx = n! ζ(n) (integrate by parts; boundary term
/// vanishes for n ≥ 2).
///
/// Before this test only the *constants* β_μ and κ_c were anchored
/// (`test_beta_mu_from_zeta_functions`, `test_kappa_c_from_numerical_integration`);
/// the moments of the shape functions as actually coded were never checked, so
/// nothing tied the constants to the functions that use them.
#[test]
fn test_distortion_shape_moment_identities() {
    // Simpson in ln x over 12 decades: the integrands decay exponentially at
    // both ends, so truncation is ≲10⁻¹² and the O(h⁴) error is at roundoff.
    const N: usize = 200_000;
    const X_LO: f64 = 1e-12;
    const X_HI: f64 = 80.0;
    let m2 = |f: &dyn Fn(f64) -> f64| simpson_log(|x| x * x * f(x), X_LO, X_HI, N);
    let m3 = |f: &dyn Fn(f64) -> f64| simpson_log(|x| x * x * x * f(x), X_LO, X_HI, N);

    // ζ(2) = G1_PLANCK, ζ(3) = ZETA_3, ζ(4) = G3_PLANCK/6, G₂ = 2ζ(3).
    let zeta2 = G1_PLANCK;
    let zeta4 = G3_PLANCK / 6.0;
    assert!(
        (zeta4 - std::f64::consts::PI.powi(4) / 90.0).abs() < 1e-15,
        "ζ(4) bookkeeping"
    );

    // --- G_bb: the pure temperature shift --------------------------------
    let gbb2 = m2(&g_bb);
    let gbb3 = m3(&g_bb);
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    eprintln!("∫x²G_bb = {gbb2:.12e} (3G₂ = {:.12e})", 3.0 * G2_PLANCK);
    eprintln!("∫x³G_bb = {gbb3:.12e} (4G₃ = {:.12e})", 4.0 * G3_PLANCK);
    assert!(
        rel(gbb2, 3.0 * G2_PLANCK) < 1e-10,
        "∫x²G_bb dx must be 3G₂ = 6ζ(3) (⟹ ΔN/N = 3ΔT/T): got {gbb2:.15e}"
    );
    assert!(
        rel(gbb3, 4.0 * G3_PLANCK) < 1e-10,
        "∫x³G_bb dx must be 4G₃ = 24ζ(4) (⟹ Δρ/ρ = 4ΔT/T): got {gbb3:.15e}"
    );

    // --- M: the μ-distortion ---------------------------------------------
    // ∫x²M dx = (1/β_μ)·3!ζ(3) − 2!ζ(2) = 0 ⟺ β_μ = 3ζ(3)/ζ(2).
    let m_2 = m2(&mu_shape);
    let m_3 = m3(&mu_shape);
    let m_scale = simpson_log(|x| (x * x * mu_shape(x)).abs(), X_LO, X_HI, N);
    eprintln!("∫x²M = {m_2:.6e} (exactly 0; scale ∫|x²M| = {m_scale:.4})");
    assert!(
        m_2.abs() < 1e-9 * m_scale,
        "∫x²M dx must vanish identically (this is β_μ's defining condition): \
         got {m_2:.6e} against scale {m_scale:.4}"
    );
    // ∫x³M dx = 8ζ(2)ζ(4)/ζ(3) − 6ζ(3) = (κ_c/3)·G₃.
    let m3_closed = 8.0 * zeta2 * zeta4 / ZETA_3 - 6.0 * ZETA_3;
    eprintln!(
        "∫x³M = {m_3:.12e} (closed form {m3_closed:.12e}, κ_cG₃/3 = {:.12e})",
        KAPPA_C * G3_PLANCK / 3.0
    );
    assert!(
        rel(m_3, m3_closed) < 1e-10,
        "∫x³M dx must equal 8ζ(2)ζ(4)/ζ(3) − 6ζ(3): got {m_3:.15e}"
    );
    assert!(
        rel(m_3, KAPPA_C * G3_PLANCK / 3.0) < 1e-10,
        "∫x³M dx must equal (κ_c/3)G₃ — this is the identity behind \
         μ = (3/κ_c)Δρ/ρ = 1.401 Δρ/ρ: got {m_3:.15e}"
    );
    // β_μ is the zero crossing of M, and the zero of the number moment.
    assert!(
        mu_shape(BETA_MU).abs() < 1e-14,
        "M(β_μ) must vanish: got {:.3e}",
        mu_shape(BETA_MU)
    );

    // --- Y: the y-distortion ---------------------------------------------
    let y_2 = m2(&y_shape);
    let y_3 = m3(&y_shape);
    let y_scale = simpson_log(|x| (x * x * y_shape(x)).abs(), X_LO, X_HI, N);
    eprintln!("∫x²Y = {y_2:.6e} (exactly 0; scale ∫|x²Y| = {y_scale:.4})");
    eprintln!("∫x³Y = {y_3:.12e} (4G₃ = {:.12e})", 4.0 * G3_PLANCK);
    assert!(
        y_2.abs() < 1e-9 * y_scale,
        "∫x²Y dx must vanish identically (Compton scattering conserves photon \
         number): got {y_2:.6e} against scale {y_scale:.4}"
    );
    assert!(
        rel(y_3, 4.0 * G3_PLANCK) < 1e-10,
        "∫x³Y dx must equal 4G₃ (⟹ Δρ/ρ = 4y): got {y_3:.15e}"
    );
}

// ---------------------------------------------------------------------------
// T-PC-7 — DC/BR crossover redshift
// ---------------------------------------------------------------------------

/// **T-PC-7.** The redshift at which double-Compton and bremsstrahlung emission
/// balance, K_DC(x, z) = K_BR(x, z).
///
/// Anchor: the in-text claim "DC dominant z ≳ 10⁶, BR z ≲ 10⁵, crossover
/// z ≈ 3–4×10⁵" (independently derived in Round-1 finding P1-8, see
/// `dev/audit/double_compton_bremsstrahlung_audit.md`). Previously the suite
/// asserted only two one-sided inequalities at z = 2×10⁶ and z = 10⁴, plus the
/// x_c comparison in `test_photon_survival_regime_structure`; the crossover
/// *redshift* itself — the thing the paper states — was never checked.
///
/// The crossover is x-dependent: it is reported at the P1-8 reference point
/// x = 0.1 and, for the record, at x = 1.
#[test]
fn test_dc_br_crossover_redshift() {
    let cosmo = Cosmology::default();
    let ratio_at = |x: f64, z: f64| {
        let th = theta_z(z);
        let k_dc = dc_emission_coefficient(x, th);
        let k_br = br_emission_coefficient(
            x,
            th,
            th,
            cosmo.n_h(z),
            cosmo.n_he(z),
            cosmo.n_e(z, 1.0),
            1.0,
            &cosmo,
        );
        k_dc / k_br
    };

    // Bisect ln z for K_DC/K_BR = 1. The ratio rises monotonically with z
    // (DC ∝ θ_z², BR ∝ n_e θ^{-7/2}), which the bracket check enforces.
    let crossover = |x: f64| {
        let (mut lo, mut hi) = (1.0e4_f64, 1.0e7_f64);
        assert!(
            ratio_at(x, lo) < 1.0 && ratio_at(x, hi) > 1.0,
            "K_DC/K_BR must bracket unity between z = {lo:.0e} and {hi:.0e} at x = {x}: \
             got {:.3e} and {:.3e}",
            ratio_at(x, lo),
            ratio_at(x, hi)
        );
        for _ in 0..200 {
            let m = (lo * hi).sqrt(); // geometric midpoint
            if ratio_at(x, m) > 1.0 {
                hi = m;
            } else {
                lo = m;
            }
        }
        (lo * hi).sqrt()
    };

    let z_x01 = crossover(0.1);
    let z_x1 = crossover(1.0);
    eprintln!(
        "K_DC = K_BR at z = {z_x01:.3e} (x = 0.1, P1-8 reference point) and \
         z = {z_x1:.3e} (x = 1); in-text claim 3–4×10⁵"
    );

    assert!(
        (2.5e5..4.5e5).contains(&z_x01),
        "DC/BR crossover at x = 0.1 is z = {z_x01:.3e}; the paper states \
         3–4×10⁵ (band widened to 2.5–4.5×10⁵ to absorb the x-dependence)"
    );
    // One-sided sanity at the two redshifts the in-text claim names.
    assert!(
        ratio_at(0.1, 1.0e6) > 1.0,
        "DC must dominate at z = 10⁶: ratio {:.3}",
        ratio_at(0.1, 1.0e6)
    );
    assert!(
        ratio_at(0.1, 1.0e5) < 1.0,
        "BR must dominate at z = 10⁵: ratio {:.3}",
        ratio_at(0.1, 1.0e5)
    );
}

// ---------------------------------------------------------------------------
// T-PC-3 / T-PC-6 — exact moment and H-theorem identities of the Kompaneets
//                   kernel, tested on the production flux split
// ---------------------------------------------------------------------------

/// coth(x/2) = 1 + 2n_pl(x): the linearised Kompaneets drift coefficient.
fn coth_half(x: f64) -> f64 {
    (x / 2.0).cosh() / (x / 2.0).sinh()
}

/// Trapezoid on the (non-uniform) frequency grid.
fn trapz(x: &[f64], f: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 1..x.len() {
        acc += 0.5 * (f[i] + f[i - 1]) * (x[i] - x[i - 1]);
    }
    acc
}

/// Photon-number-weighted moment ⟨g(x)⟩ = ∫x²Δn g dx / ∫x²Δn dx.
fn moment<F: Fn(f64) -> f64>(grid: &FrequencyGrid, dn: &[f64], g: F) -> f64 {
    let w: Vec<f64> = grid.x.iter().zip(dn).map(|(&x, &d)| x * x * d).collect();
    let num: Vec<f64> = grid.x.iter().zip(&w).map(|(&x, &wi)| wi * g(x)).collect();
    trapz(&grid.x, &num) / trapz(&grid.x, &w)
}

/// A narrow log-normal bump in x, normalised to ΔN/N = `dn_over_n`.
fn log_normal_bump(grid: &FrequencyGrid, x_inj: f64, sigma_ln: f64, dn_over_n: f64) -> Vec<f64> {
    let raw: Vec<f64> = grid
        .x
        .iter()
        .map(|&x| {
            let z = (x / x_inj).ln() / sigma_ln;
            if z.abs() > 8.0 {
                0.0
            } else {
                (-0.5 * z * z).exp() / (x * x)
            }
        })
        .collect();
    let w: Vec<f64> = grid.x.iter().zip(&raw).map(|(&x, &r)| x * x * r).collect();
    let norm = dn_over_n * G2_PLANCK / trapz(&grid.x, &w);
    raw.iter().map(|r| r * norm).collect()
}

/// **T-PC-3.** Exact moment identities of the Kompaneets operator.
///
/// Linearising ∂n/∂y = x⁻²∂ₓ[x⁴(∂ₓn + n + n²)] about a Planck spectrum at the
/// *same* temperature (so the Planck identity kills the zeroth-order flux
/// exactly — CLAUDE.md pitfall #1) leaves
///
///   ∂Δn/∂y = x⁻²∂ₓ[x⁴(∂ₓΔn + coth(x/2)·Δn)],   coth(x/2) = 1 + 2n_pl,
///
/// and integrating by parts gives, for the number-weighted moments
/// ⟨xᵏ⟩ ≡ ∫x^{k+2}Δn dx / ∫x²Δn dx and **any** Δn that vanishes at the
/// boundaries:
///
///   k = 0:  dN/dy = 0                                  (number conservation)
///   k = 1:  d⟨x⟩/dy  = 4⟨x⟩ − ⟨x²coth(x/2)⟩            (drift)
///   k = 2:  d⟨x²⟩/dy = 10⟨x²⟩ − 2⟨x³coth(x/2)⟩         (drift + diffusion)
///
/// For a narrow bump at x′ the k = 1 identity collapses to
/// d⟨x⟩/dy = x′[4 − x′coth(x′/2)], which **vanishes at x′ = 3.8300** — the same
/// transcendental root as the Y_SZ zero crossing. The bump therefore drifts up
/// for x′ < 3.83 and down for x′ > 3.83, and the test spans both signs.
///
/// This is the only place the drift flux φ(2n_pl+1)Δn is pinned to an analytic
/// value: number and energy conservation are satisfied by *any* antisymmetric
/// flux split, so they cannot see an error in this coefficient. The electron
/// temperature is held at T_e = T_z, which removes the quasi-stationary T_e
/// feedback — with it, the energy the bump loses returns immediately as a
/// y-distortion and the identity applies to the bump component alone.
#[test]
fn test_kompaneets_moment_identities() {
    let grid = FrequencyGrid::log_uniform(1e-3, 60.0, 2400);
    let theta = 1.0e-3; // θ_e = θ_z: no T_e feedback, and y = θ·τ exactly
    let dtau = 0.02;
    let dy = theta * dtau;
    let n_steps = 1000;
    let y_tot = dy * n_steps as f64;

    for &x_inj in &[1.0_f64, 3.0, 3.83, 5.0] {
        let dn0 = log_normal_bump(&grid, x_inj, 0.08, 1e-6);

        let n_of = |dn: &[f64]| {
            let w: Vec<f64> = grid.x.iter().zip(dn).map(|(&x, &d)| x * x * d).collect();
            trapz(&grid.x, &w)
        };
        // RHS of the two identities, evaluated on a given state.
        let rate1 = |dn: &[f64]| {
            4.0 * moment(&grid, dn, |x| x) - moment(&grid, dn, |x| x * x * coth_half(x))
        };
        let rate2 = |dn: &[f64]| {
            10.0 * moment(&grid, dn, |x| x * x)
                - 2.0 * moment(&grid, dn, |x| x * x * x * coth_half(x))
        };

        let (n_i, x1_i, x2_i) = (
            n_of(&dn0),
            moment(&grid, &dn0, |x| x),
            moment(&grid, &dn0, |x| x * x),
        );

        // Composite trapezoid of the RHS along the trajectory: the two rates
        // are exact instantaneously, so sampling them every `chunk` steps and
        // integrating over y removes the O(y²) error of a two-point estimate
        // and leaves the identity itself as the only thing under test.
        let chunk = 50;
        let n_chunks = n_steps / chunk;
        let mut dn = dn0.clone();
        let (mut i1, mut i2) = (0.0_f64, 0.0_f64);
        let (mut r1_prev, mut r2_prev) = (rate1(&dn), rate2(&dn));
        for _ in 0..n_chunks {
            for _ in 0..chunk {
                dn = kompaneets_step_nonlinear(&grid, &dn, theta, theta, dtau);
            }
            let (r1, r2) = (rate1(&dn), rate2(&dn));
            let dy_chunk = dy * chunk as f64;
            i1 += 0.5 * (r1 + r1_prev) * dy_chunk;
            i2 += 0.5 * (r2 + r2_prev) * dy_chunk;
            r1_prev = r1;
            r2_prev = r2;
        }

        let (n_f, x1_f, x2_f) = (
            n_of(&dn),
            moment(&grid, &dn, |x| x),
            moment(&grid, &dn, |x| x * x),
        );
        let (meas1, pred1) = (x1_f - x1_i, i1);
        let (meas2, pred2) = (x2_f - x2_i, i2);
        let narrow = x_inj * (4.0 - x_inj * coth_half(x_inj));

        eprintln!(
            "x′={x_inj:4.2} y={y_tot:.3}: Δ⟨x⟩ meas={meas1:+11.7} pred={pred1:+11.7} \
             (narrow-bump rate {narrow:+8.4})   Δ⟨x²⟩ meas={meas2:+11.6} pred={pred2:+11.6}   \
             ΔN/N={:+.2e}",
            (n_f - n_i) / n_i
        );

        assert!(
            ((n_f - n_i) / n_i).abs() < 1e-9,
            "x′={x_inj}: pure Compton must conserve photon number: ΔN/N = {:.3e}",
            (n_f - n_i) / n_i
        );
        assert!(
            (meas1 - pred1).abs() < 5e-3 * pred1.abs(),
            "x′={x_inj}: Δ⟨x⟩ = {meas1:.8} but ∫[4⟨x⟩ − ⟨x²coth(x/2)⟩]dy = {pred1:.8} \
             (exact moment identity, 0.5% tolerance)"
        );
        assert!(
            (meas2 - pred2).abs() < 5e-3 * pred2.abs(),
            "x′={x_inj}: Δ⟨x²⟩ = {meas2:.8} but ∫[10⟨x²⟩ − 2⟨x³coth(x/2)⟩]dy = {pred2:.8} \
             (exact moment identity, 0.5% tolerance)"
        );
        // The measured and predicted changes must at least agree in sign.
        assert!(
            meas1.signum() == pred1.signum(),
            "x′={x_inj}: Δ⟨x⟩ sign {meas1:+.3e} disagrees with the identity {pred1:+.3e}"
        );
        // Sign structure about the narrow-bump fixed point x′ = 3.8300. Note a
        // *finite-width* bump has its fixed point slightly lower (⟨x²coth⟩ picks
        // up the curvature), which is why x′ = 3.83 already drifts down here —
        // the exact identity predicts that, the narrow-bump limit does not.
        if x_inj < 3.8 {
            assert!(
                meas1 > 0.0,
                "x′={x_inj} < 3.830: the bump must drift up, got Δ⟨x⟩ = {meas1:.4e}"
            );
        } else if x_inj > 3.9 {
            assert!(
                meas1 < 0.0,
                "x′={x_inj} > 3.830: the bump must drift down, got Δ⟨x⟩ = {meas1:.4e}"
            );
        }
    }
}

/// **T-PC-6.** H-theorem for the Kompaneets equation, value *and* sign.
///
/// With ψ ≡ ln[n/(1+n)] + x, the full nonlinear flux is exactly
/// F = x⁴n(1+n)ψ′, so the functional
///
///   H[n] = ∫x²[n ln n − (1+n)ln(1+n) + x n] dx   (= ρ − S, S the photon entropy)
///
/// obeys, for T_e = T_z and no DC/BR,
///
///   dH/dy = −∫x⁴ n(1+n) (ψ′)² dx ≤ 0,
///
/// vanishing only for ψ = const, i.e. the Bose–Einstein spectrum. This pins the
/// discrete flux in a way conservation laws cannot: number and energy
/// conservation hold for *any* antisymmetric flux, but the dissipation rate
/// fixes the relative weight of the diffusion and drift pieces. Both the sign
/// (monotonicity along the trajectory) and the *value* of dH/dy are checked.
///
/// The dissipation integrand is evaluated from the Planck-subtracted flux
/// F = x⁴[Δn′ + (2n_pl+1)Δn + Δn²] (the production split), so it is identically
/// zero wherever Δn = 0 instead of a 0/0 ratio at large x.
#[test]
fn test_kompaneets_h_theorem() {
    let grid = FrequencyGrid::log_uniform(1e-3, 60.0, 2400);
    let theta = 1.0e-3;
    let dtau = 0.02;
    let dy = theta * dtau;

    // Amplitude large enough that ΔH (second order in Δn) sits far above
    // roundoff, small enough that the Δn² flux term stays a correction.
    let dn = {
        let mut d = log_normal_bump(&grid, 2.0, 0.25, 3e-3);
        // Add a soft-side deficit so the state is not a pure bump: the theorem
        // must hold for an arbitrary perturbation.
        for (i, &x) in grid.x.iter().enumerate() {
            let z = (x / 0.4_f64).ln() / 0.5;
            d[i] -= 0.15 * (-0.5 * z * z).exp() / (x * x) * 3e-3 * G2_PLANCK;
        }
        d
    };

    // H[n] and the analytic dissipation −dH/dy.
    let h_of = |dn: &[f64]| {
        let f: Vec<f64> = grid
            .x
            .iter()
            .zip(dn)
            .map(|(&x, &d)| {
                let n = planck(x) + d;
                x * x * (n * n.ln() - (1.0 + n) * (1.0 + n).ln() + x * n)
            })
            .collect();
        trapz(&grid.x, &f)
    };
    let dissipation = |dn: &[f64]| {
        // F/x⁴ at cell interfaces from the production split, then
        // ∫ F²/(x⁴ n(1+n)) dx on the interface grid.
        let mut xs = Vec::with_capacity(grid.n - 1);
        let mut f = Vec::with_capacity(grid.n - 1);
        for i in 0..grid.n - 1 {
            let xh = grid.x_half[i];
            let npl = planck(xh);
            let dh = 0.5 * (dn[i] + dn[i + 1]);
            let ddx = (dn[i + 1] - dn[i]) / (grid.x[i + 1] - grid.x[i]);
            let flux_over_x4 = ddx + (2.0 * npl + 1.0) * dh + dh * dh;
            let n = npl + dh;
            xs.push(xh);
            f.push(xh.powi(4) * flux_over_x4 * flux_over_x4 / (n * (1.0 + n)));
        }
        trapz(&xs, &f)
    };

    let h0 = h_of(&dn);
    let mut state = dn.clone();
    let mut h_prev = h0;
    let mut d_prev = dissipation(&dn);
    let mut integral = 0.0_f64;
    let n_chunks = 20;
    let steps_per_chunk = 50;
    let dy_chunk = dy * steps_per_chunk as f64;
    for c in 0..n_chunks {
        for _ in 0..steps_per_chunk {
            state = kompaneets_step_nonlinear(&grid, &state, theta, theta, dtau);
        }
        let h = h_of(&state);
        assert!(
            h <= h_prev + 1e-15 * h_prev.abs(),
            "H must not increase (chunk {c}): {h:.15e} > {h_prev:.15e}"
        );
        h_prev = h;
        let d = dissipation(&state);
        integral += 0.5 * (d + d_prev) * dy_chunk;
        d_prev = d;
    }
    let y_tot = dy * (n_chunks * steps_per_chunk) as f64;
    let meas = h_prev - h0;
    let pred = -integral;
    eprintln!(
        "H-theorem: H₀={h0:.10e} → H={h_prev:.10e} over y={y_tot:.3}; \
         ΔH meas={meas:.6e} pred=−∫dy∫x⁴n(1+n)ψ′²dx={pred:.6e}  ratio={:.5}",
        meas / pred
    );
    assert!(
        meas < 0.0,
        "H must decrease under pure Compton scattering: ΔH = {meas:.6e}"
    );
    assert!(
        (meas / pred - 1.0).abs() < 0.01,
        "ΔH = {meas:.6e} must equal −∫dy∫x⁴n(1+n)(ψ′)²dx = {pred:.6e} \
         (ratio {:.5}, tolerance 1%)",
        meas / pred
    );
}

// ---------------------------------------------------------------------------
// T-PC-5 — the quasi-stationary electron temperature returns the energy
// ---------------------------------------------------------------------------

/// **T-PC-5.** Under pure Compton scattering with a self-consistent
/// quasi-stationary T_e, the *photon* energy is conserved even though the
/// Kompaneets drift alone would move it.
///
/// This is the reason `test_kompaneets_moment_identities` has to be a
/// kernel-level test. A bump at x′ < 3.83 gains energy from the electrons at the
/// rate d⟨x⟩/dy = 4⟨x⟩ − ⟨x²coth(x/2)⟩ > 0; the electrons cool, so ρ_e − 1 < 0;
/// and because the electron heat capacity is ~10⁻⁹ of the photons', the
/// quasi-stationary T_e immediately gives the energy back as a broad
/// y-distortion. The two effects cancel to the heat-capacity ratio, so
/// ⟨x⟩ of the *total* Δn barely moves and the drift identity is unobservable
/// through the full solver.
///
/// The test asserts the cancellation quantitatively: the photon energy must stay
/// put to <1% while the drift alone predicts a 6.6% change, and the induced
/// ρ_e − 1 must have the sign and magnitude the number-weighted moment
/// ρ_e − 1 = (ΔN/N)(G₂/4G₃)⟨x[x coth(x/2) − 4]⟩ predicts.
#[test]
fn test_quasistationary_te_returns_bump_energy() {
    use spectroxide::grid::GridConfig;
    use spectroxide::solver::{SolverConfig, ThermalizationSolver};

    let cosmo = Cosmology::default();
    let (z_start, z_end) = (3.0e4, 1.0e4);

    // Analytic y_c over the run (Simpson in z) — the same integrand T-PC-1
    // anchors against z_*, here weighted by θ_z.
    let hist = RecombinationHistory::new(&cosmo);
    let n_int = 200_000;
    let dz = (z_start - z_end) / n_int as f64;
    let integrand = |z: f64| {
        let x_e = hist.x_e(z);
        cosmo.theta_z(z) * cosmo.n_e(z, x_e) * SIGMA_THOMSON * C_LIGHT
            / (cosmo.hubble(z) * (1.0 + z))
    };
    let mut y_c = integrand(z_end) + integrand(z_start);
    for i in 1..n_int {
        let z = z_end + i as f64 * dz;
        y_c += if i % 2 == 1 { 4.0 } else { 2.0 } * integrand(z);
    }
    y_c *= dz / 3.0;

    let x_inj = 1.0_f64;
    let dn_over_n = 1.0e-4;
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::production());
    solver.disable_dcbr = true;
    let dn0 = log_normal_bump(&solver.grid, x_inj, 0.15, dn_over_n);

    // Pure-Kompaneets prediction for the *photon* energy change, from the exact
    // first-moment identity evaluated on the initial state.
    let rate1 = 4.0 * moment(&solver.grid, &dn0, |x| x)
        - moment(&solver.grid, &dn0, |x| x * x * coth_half(x));
    let e_bump = dn_over_n * G2_PLANCK * moment(&solver.grid, &dn0, |x| x) / G3_PLANCK;
    let de_kompaneets = dn_over_n * G2_PLANCK * rate1 * y_c / G3_PLANCK;
    // Number-weighted moment that sets the electron-temperature response.
    let rho_e_pred = dn_over_n * G2_PLANCK / (4.0 * G3_PLANCK)
        * moment(&solver.grid, &dn0, |x| x * (x * coth_half(x) - 4.0));

    solver.set_initial_delta_n(dn0);
    solver.set_config(SolverConfig {
        z_start,
        z_end,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&[z_end]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!(
        "y_c = {y_c:.5e}; bump Δρ/ρ = {e_bump:.5e} → solver {:.5e} \
         (change {:+.3}%, pure-Kompaneets drift would give {:+.3}%)",
        snap.delta_rho_over_rho,
        100.0 * (snap.delta_rho_over_rho / e_bump - 1.0),
        100.0 * de_kompaneets / e_bump
    );
    eprintln!(
        "  ρ_e − 1 = {:.4e} (moment prediction {rho_e_pred:.4e}, ratio {:.3})",
        snap.rho_e - 1.0,
        (snap.rho_e - 1.0) / rho_e_pred
    );

    // The drift is not small: this is what makes the null non-trivial.
    assert!(
        de_kompaneets / e_bump > 0.03,
        "test is vacuous unless the pure-Kompaneets drift is a several-percent \
         effect: predicted {:.4}%",
        100.0 * de_kompaneets / e_bump
    );
    assert!(
        (snap.delta_rho_over_rho / e_bump - 1.0).abs() < 0.01,
        "photon energy must be conserved to <1% once T_e responds (the drift \
         energy comes straight back as a y-distortion): {:.5e} vs {e_bump:.5e}",
        snap.delta_rho_over_rho
    );
    // Sign and magnitude of the electron response. The bump *gains* energy from
    // the electrons at x′ = 1 < 3.83, so the electrons must end up cooler.
    assert!(
        snap.rho_e - 1.0 < 0.0 && rho_e_pred < 0.0,
        "a bump below x′ = 3.83 must cool the electrons: ρ_e − 1 = {:.3e}",
        snap.rho_e - 1.0
    );
    assert!(
        ((snap.rho_e - 1.0) / rho_e_pred - 1.0).abs() < 0.3,
        "ρ_e − 1 = {:.4e} must match the number-weighted moment prediction \
         (ΔN/N)(G₂/4G₃)⟨x[x coth(x/2) − 4]⟩ = {rho_e_pred:.4e} within 30% \
         (the residual is the O(y_c) evolution of the bump during the run)",
        snap.rho_e - 1.0
    );
}

// ---------------------------------------------------------------------------
// T-PC-8 — independence of the artificial grid boundaries
// ---------------------------------------------------------------------------

/// **T-PC-8.** μ, y and Δρ/ρ must be converged with respect to the *artificial*
/// grid boundaries x_min and x_max, not just with respect to the point count.
///
/// `convergence_order.rs` and `mms_convergence.rs` refine N and dτ at fixed
/// [x_min, x_max]. But x_min is where a Dirichlet Δn = 0 condition sits on top of
/// the BR sink (photons leaving the grid are lost) and x_max truncates the G₃
/// integrals — CLAUDE.md pitfall #7 asserts x_max ≥ 30 in `GridConfig::validate`
/// but nothing checks that the *production* choice is actually converged.
///
/// The test refines both boundaries beyond production (x_min ↓, x_max ↑) and
/// requires the observables to move by <1%. The deliberately coarse variant is
/// reported, not asserted: it exists to show the test has teeth.
#[test]
fn test_grid_boundary_independence() {
    use spectroxide::energy_injection::InjectionScenario;
    use spectroxide::grid::GridConfig;
    use spectroxide::solver::{SolverConfig, ThermalizationSolver};

    let cosmo = Cosmology::default();
    let z_h = 1.0e5;
    let drho = 1.0e-5;
    let run = |x_min: f64, x_max: f64, n_points: usize| {
        let mut solver = ThermalizationSolver::new(
            cosmo.clone(),
            GridConfig {
                x_min,
                x_max,
                n_points,
                ..GridConfig::production()
            },
        );
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: 0.01 * z_h,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: z_h * 1.1,
            z_end: 1.0e3,
            ..SolverConfig::default()
        });
        solver.run_with_snapshots(&[1.0e3]);
        let s = solver.snapshots.last().unwrap();
        (s.mu, s.y, s.delta_rho_over_rho)
    };

    // Production grid, then the same boundaries at higher N: the difference
    // between these two is *resolution* sensitivity, which the convergence
    // tests already own. The boundary variants are then compared against the
    // matched-N control so the two effects are not confounded.
    let (mu_ref, y_ref, e_ref) = run(1e-5, 60.0, 4000);
    let (mu_c, y_c, e_c) = run(1e-5, 60.0, 4400);
    eprintln!(
        "production (x_min=1e-5, x_max=60, N=4000): μ={mu_ref:.6e}, y={y_ref:.6e}, Δρ/ρ={e_ref:.6e}\n\
         control    (x_min=1e-5, x_max=60, N=4400): μ={mu_c:.6e} ({:+.3}%), y={y_c:.6e} ({:+.3}%), \
         Δρ/ρ={e_c:.6e} ({:+.3}%)  ← resolution only",
        100.0 * (mu_c / mu_ref - 1.0),
        100.0 * (y_c / y_ref - 1.0),
        100.0 * (e_c / e_ref - 1.0)
    );

    for (label, x_min, x_max) in [
        ("x_min refined ×3.3", 3e-6_f64, 60.0_f64),
        ("x_max extended to 100", 1e-5, 100.0),
    ] {
        let (mu, y, e) = run(x_min, x_max, 4400);
        let d = |a: f64, b: f64| (a - b).abs() / b.abs();
        eprintln!(
            "  {label} (N=4400): μ={mu:.6e} ({:+.3}%), y={y:.6e} ({:+.3}%), \
             Δρ/ρ={e:.6e} ({:+.3}%)  [vs matched-N control]",
            100.0 * (mu / mu_c - 1.0),
            100.0 * (y / y_c - 1.0),
            100.0 * (e / e_c - 1.0)
        );
        assert!(
            d(mu, mu_c) < 0.01,
            "{label}: μ moved {:.3}% (>1%) — the production x_min/x_max are not \
             converged for μ",
            100.0 * (mu / mu_c - 1.0)
        );
        assert!(
            d(y, y_c) < 0.01,
            "{label}: y moved {:.3}% (>1%) — the production x_min/x_max are not \
             converged for y",
            100.0 * (y / y_c - 1.0)
        );
        assert!(
            d(e, e_c) < 1e-3,
            "{label}: Δρ/ρ moved {:.4}% (>0.1%) — energy is leaking through an \
             artificial boundary",
            100.0 * (e / e_c - 1.0)
        );
    }
}

// ---------------------------------------------------------------------------
// T-PC-4 — the thermalization exponent α_th = 5/2, from the PDE alone
// ---------------------------------------------------------------------------

/// **T-PC-4.** The PDE's own thermalization optical depth must scale as
/// τ_th ∝ z^{5/2}.
///
/// `science_deep_thermalization_pde_z3e6` (R2 finding P1) pins the DC emission
/// *normalisation* at one redshift via ∂lnμ/∂lnK_DC ≈ −τ/2. It says nothing
/// about the **exponent** α_th = 5/2, which is a different combination —
/// α_th follows from K_DC ∝ θ_z², the Thomson rate n_eσ_Tc and H ∝ z² in the
/// radiation era, and the paper states it as an analytic result. Before this
/// test the exponent appeared only inside the Chluba 2013 visibility *fit*
/// (`Z_MU` and `visibility_j_bb_star`), never as a property of the solver.
///
/// Method: with J_μ = 1 to 5 decimal places over 2–3×10⁶, the suppression the
/// PDE produces is S(z_h) = μ_PDE/(1.401 Δρ) and τ_eff = −ln S. The overall
/// 2–5% PDE↔GF offset in μ would bias τ_eff by a constant δ and hence α by
/// ~δ/τ, so S is normalised by a calibration run at z_h = 3×10⁵ where the
/// suppression is only 0.9%. What remains is a pure statement about the code's
/// *z-scaling*.
///
/// Note the reference is not exactly 5/2: the Chluba fit's own local slope over
/// this window is 2.47, because J_bb* is not a pure exponential of a power law.
/// The band ±0.15 admits both.
///
/// **Cost: ~7 minutes** (three deep-μ-era runs plus one calibration run), so it
/// is `#[ignore]`d like the other expensive convergence checks. Run with
/// `cargo test --release --test physics_identities -- --ignored --nocapture`.
#[test]
#[ignore = "expensive: ~7 min of deep-μ-era PDE runs"]
fn test_thermalization_exponent_five_halves() {
    use spectroxide::energy_injection::InjectionScenario;
    use spectroxide::greens::{visibility_j_bb_star, visibility_j_mu};
    use spectroxide::grid::GridConfig;
    use spectroxide::solver::{SolverConfig, ThermalizationSolver};

    let drho = 1.0e-5;
    let mu_of = |z_h: f64| {
        let mut solver = ThermalizationSolver::new(Cosmology::default(), GridConfig::production());
        solver
            .set_injection(InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho: drho,
                sigma_z: z_h * 0.01,
            })
            .unwrap();
        solver.set_config(SolverConfig {
            z_start: z_h * 1.5,
            z_end: 500.0,
            ..SolverConfig::default()
        });
        solver.run_with_snapshots(&[500.0]);
        solver.snapshots.last().unwrap().mu
    };

    // Calibration: z_h = 3×10⁵ is deep in the μ-era but barely thermalized, so
    // it measures the PDE's unsuppressed μ including any PDE↔GF offset.
    let z_cal = 3.0e5;
    let s_cal = mu_of(z_cal) / (3.0 / KAPPA_C * drho * visibility_j_mu(z_cal));
    let j_cal = visibility_j_bb_star(z_cal);
    eprintln!(
        "calibration z_h={z_cal:.1e}: PDE S = {s_cal:.5}, Chluba J_bb* = {j_cal:.5} \
         (PDE↔GF offset {:+.2}%)",
        100.0 * (s_cal / j_cal - 1.0)
    );

    let z_hs = [2.0e6_f64, 2.5e6, 3.0e6];
    let mut ln_z = Vec::new();
    let mut ln_tau = Vec::new();
    for &z_h in &z_hs {
        let s = mu_of(z_h) / (3.0 / KAPPA_C * drho * visibility_j_mu(z_h));
        // Divide out the calibration offset: S_true = (S/S_cal)·J_bb*(z_cal).
        let s_true = s / s_cal * j_cal;
        let tau = -s_true.ln();
        eprintln!(
            "  z_h={z_h:.2e}: S_PDE={s:.6}, S_norm={s_true:.6}, τ_eff={tau:.5} \
             (Chluba fit τ = {:.5})",
            -visibility_j_bb_star(z_h).ln()
        );
        assert!(
            tau > 0.5 && tau < 6.0,
            "z_h={z_h:.2e} must land in the exponential tail: τ_eff = {tau:.4}"
        );
        ln_z.push(z_h.ln());
        ln_tau.push(tau.ln());
    }

    // Least-squares slope of ln τ vs ln z.
    let n = ln_z.len() as f64;
    let (mz, mt) = (
        ln_z.iter().sum::<f64>() / n,
        ln_tau.iter().sum::<f64>() / n,
    );
    let num: f64 = ln_z
        .iter()
        .zip(&ln_tau)
        .map(|(z, t)| (z - mz) * (t - mt))
        .sum();
    let den: f64 = ln_z.iter().map(|z| (z - mz).powi(2)).sum();
    let alpha = num / den;

    // Same fit applied to the Chluba visibility itself, for reference.
    let fit_tau: Vec<f64> = z_hs
        .iter()
        .map(|&z| (-visibility_j_bb_star(z).ln()).ln())
        .collect();
    let mtf = fit_tau.iter().sum::<f64>() / n;
    let alpha_fit: f64 = ln_z
        .iter()
        .zip(&fit_tau)
        .map(|(z, t)| (z - mz) * (t - mtf))
        .sum::<f64>()
        / den;

    eprintln!("α_th: PDE = {alpha:.4}, Chluba J_bb* fit = {alpha_fit:.4}, analytic 5/2");
    assert!(
        (alpha - 2.5).abs() < 0.15,
        "the PDE's thermalization optical depth must scale as z^{{5/2}}: fitted \
         α_th = {alpha:.4} over z_h ∈ [2,3]×10⁶ (band 2.5 ± 0.15; the Chluba fit's \
         own local slope here is {alpha_fit:.4})"
    );
}

// ---------------------------------------------------------------------------
// T-PC-9 — the electron-temperature balance against the standard Compton rate
// ---------------------------------------------------------------------------

/// **T-PC-9.** With no injection, the quasi-stationary electron temperature is
/// fixed by adiabatic cooling against Compton heating:
///
///   dρ_e/dt = Γ_C(1 − ρ_e) − H ρ_e   ⟹   ρ_e − 1 = −H/(Γ_C + H),
///
/// where ρ_e = T_e/T_z (the −H rather than −2H is because T_z ∝ (1+z) already
/// carries one power of the expansion) and Γ_C is the standard Compton-cooling
/// rate of the baryon fluid
///
///   Γ_C = (8 σ_T u_γ)/(3 m_e c) · n_e/(n_e + n_H + n_He)
///
/// (Seager, Sasselov & Scott 1999; Chluba & Sunyaev 2012 Eq. 15–18). The
/// n_e/n_tot factor is the heat shared with neutral H and He. Before this test
/// the T_e normalisation was anchored only internally
/// (`test_full_te_perturbative_vs_brute_force` compares two of the code's own
/// paths) and indirectly through the CosmoTherm adiabatic-cooling μ.
///
/// The balance term falls as −H/Γ_C ∝ z⁻² while the adiabatic-cooling
/// *distortion* accumulates and feeds back through δρ_eq = ΔI₄/(4G₃) − ΔG₃/G₃,
/// so the two are comparable at high z and the balance only dominates at low z.
/// Measured ratio solver/analytic: 5.45 (z = 10⁵), 1.775 (3×10⁴), 1.121 (10⁴),
/// **1.0042 (3×10³)**. The assertion therefore lives at z ≤ 3×10³, with the
/// monotone approach checked as structure. DC/BR heating is *not* the residual:
/// disabling it moves every ratio by <0.1%.
#[test]
fn test_electron_temperature_compton_adiabatic_balance() {
    use spectroxide::constants::{M_ELECTRON, SIGMA_THOMSON};
    use spectroxide::grid::GridConfig;
    use spectroxide::solver::{SolverConfig, ThermalizationSolver};

    let cosmo = Cosmology::default();
    let hist = RecombinationHistory::new(&cosmo);
    let rho_e_analytic = |z: f64| {
        let x_e = hist.x_e(z);
        let n_h = cosmo.n_h(z);
        let n_he = cosmo.n_he(z);
        let n_e = cosmo.n_e(z, x_e);
        let gamma_c = 8.0 * SIGMA_THOMSON * cosmo.rho_gamma(z) / (3.0 * M_ELECTRON * C_LIGHT)
            * n_e
            / (n_e + n_h + n_he);
        let h = cosmo.hubble(z);
        1.0 - h / (gamma_c + h)
    };

    let checkpoints = [1.0e5_f64, 3.0e4, 1.0e4, 3.0e3, 1.5e3];
    let mut solver = ThermalizationSolver::new(cosmo.clone(), GridConfig::production());
    solver.set_config(SolverConfig {
        z_start: 5.0e5,
        z_end: 1.2e3,
        ..SolverConfig::default()
    });
    solver.run_with_snapshots(&checkpoints);

    let mut prev_ratio = f64::INFINITY;
    for (snap, &z) in solver.snapshots.iter().zip(&checkpoints) {
        let pred = rho_e_analytic(z);
        let got = snap.rho_e;
        let ratio = (got - 1.0) / (pred - 1.0);
        eprintln!(
            "z={z:9.2e}: ρ_e − 1 solver = {:+.5e}, analytic −H/(Γ_C+H) = {:+.5e} \
             (ratio {ratio:.4})",
            got - 1.0,
            pred - 1.0
        );
        assert!(
            got < 1.0,
            "z={z:.2e}: adiabatic cooling must leave the electrons colder than the \
             radiation: ρ_e − 1 = {:+.4e}",
            got - 1.0
        );
        // The distortion feedback dies away relative to the balance term as z
        // drops, so the ratio must approach 1 from above, monotonically.
        assert!(
            ratio > 0.9 && ratio < prev_ratio + 1e-3,
            "z={z:.2e}: solver/analytic ratio {ratio:.4} must approach 1 from above \
             monotonically (previous {prev_ratio:.4})"
        );
        prev_ratio = ratio;
        if z <= 3.0e3 {
            assert!(
                (ratio - 1.0).abs() < 0.03,
                "z={z:.2e}: ρ_e − 1 = {:.5e} must match the Compton/adiabatic balance \
                 −H/(Γ_C + H) = {:.5e} within 3% once the balance term dominates \
                 (ratio {ratio:.4}) — this is what pins Γ_C",
                got - 1.0,
                pred - 1.0
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T-PS-1..3 — photon-path critical frequency, placed where it is observable
//
// Sensitivity-directed (the R2/P1 method): before writing these, the log
// derivative ∂ln μ/∂ln x_c was mapped over (x_inj, z_h) with
// `scratchpad/r3/photon_sensitivity.py`. Measured ∂ln μ/∂ln(x_c coefficient):
//
//        z_h  | x_inj = x_c | x_inj = 0.1 | x_inj = 1 | x_inj = 5
//       1e4   |   −1.034 BR |   −0.452    |  −0.061   |  +0.022
//       3e4   |   −1.013 BR |   −0.214    |  −0.029   |  +0.011
//       2e5   |   −0.823 BR |   −0.054    |  −0.007   |  +0.003
//       1e6   |   −0.910 DC |   −0.060    |  −0.008   |  +0.003
//       2e6   |   −0.987 DC |   −0.088    |  −0.012   |  +0.004
//
// So x_c is O(1)-observable only at x_inj ≈ x_c, and the two coefficients
// separate cleanly: BR carries it below z ~ 1e5, DC above. At x_inj ≥ 1 —
// where the rest of the photon suite sits — a 50% coefficient error moves μ by
// under 3%, which is the same blind spot that let K_DC be wrong by 1.535×
// (R2 finding P1). Analytically, with u ≡ P_s x₀/x_inj,
//     ∂ln μ/∂ln P_s = −u/(1−u),   ∂ln P_s/∂ln x_c = −x_c/x_inj,
// so ∂ln μ/∂ln x_c → −x_c/x_inj for u ≫ 1, which is O(1) exactly at x_inj ≈ x_c.
//
// Before this, every x_c test asserted an *ordering* (x_c^DC > x_c^BR at high z)
// or a *bound* (P_s > 0.99); none asserted a value, so the coefficients
// themselves were unpinned.
// ---------------------------------------------------------------------------

/// T-PS-1 (class i). P_s(x, z) = exp(−x_c(z)/x) — Chluba 2015 Eq. 24 — so at
/// the critical frequency itself the survival probability is exactly 1/e,
/// independently of redshift. This ties `photon_survival_probability` to `x_c`:
/// any inconsistency between the two breaks it at machine precision.
#[test]
fn test_photon_survival_at_critical_frequency_is_one_over_e() {
    let inv_e = (-1.0f64).exp();
    for &z in &[1.0e4, 3.0e4, 2.0e5, 1.0e6, 2.0e6, 5.0e6] {
        let xc = spectroxide::greens::x_c(z);
        let ps = spectroxide::greens::photon_survival_probability(xc, z);
        assert!(
            (ps - inv_e).abs() < 1e-14,
            "z={z:.2e}: P_s(x_c) = {ps:.15e} must equal 1/e = {inv_e:.15e} \
             by the definition of x_c in Chluba 2015 Eq. 24"
        );
    }
}

/// T-PS-2 (class ii). Value anchors on the critical-frequency coefficients,
/// re-evaluated in the test straight from Chluba 2015 Eqs. 25a/25b rather than
/// read from the source file:
///
///   x_c^DC(z) = 8.60e-3 · ((1+z)/2e6)^(1/2)
///   x_c^BR(z) = 1.23e-3 · ((1+z)/2e6)^(-0.672)
///   x_c       = sqrt(x_c^DC² + x_c^BR²)
///
/// Checked at the two redshifts used by T-PS-3, i.e. where each term dominates
/// and where μ is O(1)-sensitive to it.
#[test]
fn test_critical_frequency_values_vs_chluba2015_eq25() {
    for &z in &[1.0e4, 3.0e4, 2.0e5, 1.0e6, 2.0e6] {
        let r: f64 = (1.0 + z) / 2.0e6;
        let want_dc = 8.60e-3 * r.sqrt();
        let want_br = 1.23e-3 * r.powf(-0.672);
        let want = (want_dc * want_dc + want_br * want_br).sqrt();

        let got_dc = spectroxide::greens::x_c_dc(z);
        let got_br = spectroxide::greens::x_c_br(z);
        let got = spectroxide::greens::x_c(z);

        assert!(
            (got_dc / want_dc - 1.0).abs() < 1e-12,
            "z={z:.2e}: x_c^DC = {got_dc:.9e}, Chluba 2015 Eq. 25a gives {want_dc:.9e}"
        );
        assert!(
            (got_br / want_br - 1.0).abs() < 1e-12,
            "z={z:.2e}: x_c^BR = {got_br:.9e}, Chluba 2015 Eq. 25b gives {want_br:.9e}"
        );
        assert!(
            (got / want - 1.0).abs() < 1e-12,
            "z={z:.2e}: x_c = {got:.9e}, quadrature of Eq. 25a/25b gives {want:.9e}"
        );
    }
}

/// T-PS-3 (class i + ii, sensitivity-directed). μ per ΔN/N for injection *at*
/// x_inj = x_c(z_h), where ∂ln μ/∂ln x_c ≈ −1 (table above). There P_s = 1/e
/// exactly (T-PS-1), so Chluba 2015's μ response collapses to a closed form
/// built only from analytic constants and published visibility fits:
///
///   μ/(ΔN/N) = α_ρ x_c (3/κ_c) J_bb*(z_h) J_μ(z_h) · (1 − x₀/(e·x_c))
///
/// with α_ρ = G₂/G₃ and x₀ = 4/(3α_ρ). Two redshifts, one per dominant term:
/// z_h = 1e4 (BR-dominated, sensitivity −1.034) and z_h = 2e6 (DC-dominated,
/// −0.987). A coefficient error in Eq. 25a/25b propagates ~1:1 into μ here,
/// versus ~0.01:1 at the x_inj ≥ 1 points the rest of the suite uses.
#[test]
fn test_photon_mu_at_critical_frequency_closed_form() {
    use spectroxide::constants::{ALPHA_RHO, KAPPA_C, X_BALANCED};

    let inv_e = (-1.0f64).exp();
    for &z_h in &[1.0e4, 2.0e6] {
        // Inject at the *literature* critical frequency (Chluba 2015 Eq. 25),
        // recomputed here, NOT at the code's `x_c`. That is what makes this an
        // anchor on the coefficients rather than a plumbing check: if the coded
        // x_c drifts, the code's P_s(x_c^lit) is no longer 1/e and the closed
        // form below misses by ~the same relative amount (∂ln μ/∂ln x_c ≈ −1).
        let r: f64 = (1.0 + z_h) / 2.0e6;
        let xc_dc = 8.60e-3 * r.sqrt();
        let xc_br = 1.23e-3 * r.powf(-0.672);
        let xc = (xc_dc * xc_dc + xc_br * xc_br).sqrt();
        let j_bb = spectroxide::greens::visibility_j_bb_star(z_h);
        let j_mu = spectroxide::greens::visibility_j_mu(z_h);

        // P_s = 1/e at x_inj = x_c, so the number-vs-energy balance factor is
        // exactly 1 − x₀/(e·x_c).
        let want = ALPHA_RHO * xc * (3.0 / KAPPA_C) * j_bb * j_mu
            * (1.0 - inv_e * X_BALANCED / xc);
        let got = spectroxide::greens::mu_from_photon_injection(xc, z_h, 1.0);

        assert!(
            (got / want - 1.0).abs() < 1e-12,
            "z_h={z_h:.2e}, x_inj=x_c={xc:.6e}: μ/(ΔN/N) = {got:.9e}, closed form \
             α_ρ x_c (3/κ_c) J_bb* J_μ (1 − x₀/(e x_c)) = {want:.9e}"
        );

        // x_inj = x_c ≪ x₀ ⇒ number-dominated ⇒ μ < 0. Guards the sign of the
        // balance term, which the ratio test above cannot see.
        assert!(
            got < 0.0,
            "z_h={z_h:.2e}: injection at x_c = {xc:.3e} ≪ x₀ = {X_BALANCED:.3} is \
             number-dominated and must give μ < 0, got {got:.4e}"
        );
    }
}
